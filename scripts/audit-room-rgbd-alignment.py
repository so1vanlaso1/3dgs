"""Reproject original unfilled depth and create exploratory alignment evidence."""
import argparse
import csv
import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/realsense_20261030_134523_clean'
OUT = ROOT/'results/room_improve_rgbd_alignment_20261103'
DB = ROOT/'20261030_134523.db3'

def module(name):
    spec=importlib.util.spec_from_file_location(name, ROOT/'scripts'/name)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def tile(image, title):
    a=Image.new('RGB',(834,495),'white'); a.paste(Image.fromarray(image), (0,32))
    ImageDraw.Draw(a).text((6,8),title,fill='black'); return a

def main():
    global OUT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=OUT)
    ap.add_argument('--names',nargs='+',help='Explicit prepared frame names; default is the six-view audit')
    args=ap.parse_args()
    OUT=args.output.resolve()
    OUT.mkdir(parents=True,exist_ok=False)
    adapter=module('prepare-sensor-depth.py')
    clean=module('clean-realsense-scene.py')
    cal=json.loads((DATA/'calibration_clean.json').read_text())
    manifest=json.loads((DATA/'capture_manifest.json').read_text())
    frames=manifest['frames']; byname={f['name']:f for f in frames}; first=frames[0]['timestamp_ns']
    selections=json.loads((ROOT/'results/room_improve_baseline_inspection_20261103/inspection_poses.json').read_text())['views']
    names=[v['image_name'] for v in selections if v['requested_seconds'] in (43,61,68,110)]
    with (DATA/'imu/gyro_frame_associations.csv').open(newline='') as f:
        gyro={r['name']:float(np.linalg.norm([float(r[k]) for k in ('x','y','z')])) for r in csv.DictReader(f)}
    # One high angular-rate frame near each difficult time interval, not a validated turn classification.
    if args.names:
        names=args.names
    else:
        for lo,hi in ((57,72),(105,115)):
            candidates=[f for f in frames if lo <= (f['timestamp_ns']-first)/1e9 <= hi and f['name'] not in names]
            names.append(max(candidates,key=lambda f:gyro[f['name']])['name'])
    R=np.array(cal['depth_to_color_R']); t=np.array(cal['depth_to_color_t_m']); K=np.array(cal['K'])
    d=cal['raw_depth']; w,h=cal['width'],cal['height']; units=cal['depth_units']
    assert all(float(x)==0 for x in d['coeffs'].split(','))
    before=DB.stat(); reports=[]
    with sqlite3.connect(DB.as_uri()+'?mode=ro',uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        for name in names:
            f=byname[name]
            row=db.execute('select timestamp,data from messages where id=?',(f['depth_message_id'],)).fetchone()
            assert row[0]==f['depth_timestamp_ns']
            dep,header=adapter.decode_depth(row[1]); assert abs(header-f['depth_header_seconds'])<1e-6
            z=dep*units
            valid=np.isfinite(z)&(z>.3)&(z<6.)
            yy,xx=np.nonzero(valid); v=z[valid]
            xyz=np.column_stack(((xx-float(d['ppx']))/float(d['fx'])*v,(yy-float(d['ppy']))/float(d['fy'])*v,v))@R.T+t
            uv=xyz[:,:2]/xyz[:,2:3]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
            x=np.rint(uv[:,0]).astype(int); y=np.rint(uv[:,1]).astype(int)
            keep=(xyz[:,2]>0)&(x>=0)&(x<w)&(y>=0)&(y<h)
            x,y,zp=x[keep],y[keep],xyz[keep,2]
            key=y*w+x; order=np.lexsort((zp,key)); _,idx=np.unique(key[order],return_index=True); chosen=order[idx]
            dense=np.zeros((h,w),np.float32); dense[y[chosen],x[chosen]]=zp[chosen]
            # Reconstruct the training path before its 16k sampling for consistency checks.
            px,cd,valid_fraction=clean.clean_depth(dep,units,d,R,t,K,w,h)
            stored=np.load(DATA/'aligned_depth_meters'/(name+'.npz'))
            pk=np.rint(px[:,1]).astype(int)*w+np.rint(px[:,0]).astype(int)
            sk=np.rint(stored['points'][:,1]).astype(int)*w+np.rint(stored['points'][:,0]).astype(int)
            order=np.argsort(pk); indices=np.searchsorted(pk[order],sk)
            assert (indices<len(pk)).all() and np.array_equal(pk[order][indices],sk)
            maxdiff=float(np.max(np.abs(cd[order][indices]-stored['depths'])))
            assert maxdiff < 1e-5
            np.savez_compressed(OUT/(name+'.dense_depth.npz'),depth_m=dense)
            rgb=np.asarray(Image.open(DATA/'images'/name).convert('RGB'))
            validrgb=dense>0
            color=cv2.applyColorMap((np.clip((dense-.3)/5.7,0,1)*255).astype('uint8'),cv2.COLORMAP_TURBO)[:,:,::-1]
            color[~validrgb]=0
            blend=rgb.copy(); blend[validrgb]=(rgb[validrgb]*.45+color[validrgb]*.55).astype('uint8')
            # Compare only adjacent valid projected pixels; holes are never edges and no filling occurs.
            edge=np.zeros((h,w),bool)
            for axis in (0,1):
                a,b=(dense[:-1,:],dense[1:,:]) if axis==0 else (dense[:,:-1],dense[:,1:])
                e=(a>0)&(b>0)&(np.abs(a-b)>np.maximum(.06,.025*np.minimum(a,b)))
                if axis==0: edge[:-1,:]|=e
                else: edge[:,:-1]|=e
            rgb_edge=cv2.Canny(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),80,160)>0
            # Distance is exploratory and unsigned; many unrelated RGB texture edges are present.
            dist=cv2.distanceTransform((~rgb_edge).astype('uint8'),cv2.DIST_L2,3)
            ds=dist[edge]; edge_overlay=rgb.copy(); edge_overlay[edge]=[255,0,255]
            cleanmap=np.zeros((h,w),np.float32); ix=np.rint(px[:,0]).astype(int); iy=np.rint(px[:,1]).astype(int); cleanmap[iy,ix]=cd
            cleaned=rgb.copy(); cleaned[cleanmap>0]=(.45*rgb[cleanmap>0]+.55*np.array([0,220,255])).astype('uint8')
            panel=Image.new('RGB',(1668,990),'white')
            for j,(im,title) in enumerate([(rgb,'Prepared RGB'),(blend,'Unfilled projected depth; untinted pixels have no projected sample'),(edge_overlay,'Magenta: depth jumps between valid adjacent pixels'),(cleaned,'Cyan: preprocessing retained depth, before sampling')]):
                panel.paste(tile(im,title),((j%2)*834,(j//2)*495))
            panel.save(OUT/(name+'.comparison.jpg'),quality=94)
            values=ds.tolist()
            report=dict(image_name=name,replay_seconds=(f['timestamp_ns']-first)/1e9,
                rgb_message_id=f['message_id'],depth_message_id=f['depth_message_id'],pair_gap_ms=f['pair_delta_seconds']*1000,
                angular_rate_rad_s=gyro[name],raw_projected_valid_pixels=int(validrgb.sum()),raw_projected_fraction=float(validrgb.mean()),
                clean_projected_valid_pixels=len(cd),clean_sensor_valid_fraction=valid_fraction,
                stored_sample_max_depth_difference_m=maxdiff,depth_edge_pixels=int(edge.sum()),
                exploratory_unsigned_distance_to_any_rgb_canny_edge_px_percentiles=np.percentile(ds,[50,90]).tolist() if len(ds) else None,
                comparison=name+'.comparison.jpg',dense_depth=name+'.dense_depth.npz')
            reports.append(report); print(name,round(report['replay_seconds'],3),'valid',round(validrgb.mean(),3),'edges',len(values),'gyro',round(gyro[name],3),flush=True)
    after=DB.stat(); assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    (OUT/'report.json').write_text(json.dumps(dict(views=reports,
      method='Original paired depth, metric units, no hole filling or smoothing; calibrated point projection and nearest-Z per rounded RGB pixel. Zero denotes missing.',
      caveats='Projected holes include sampling and occlusion as well as invalid source depth. Edge metric uses nearest arbitrary RGB Canny edge; not a calibrated alignment error or acceptance test. Motion measure is nearest gyro sample, not integrated pose.',
      depth_range_m=[.3,6], threshold='adjacent depth jump > max(0.06 m, 2.5% nearer depth)', rgb_canny_thresholds=[80,160]),indent=2),encoding='utf-8')

if __name__=='__main__': main()
