"""Read nearby original RGB-D frames around the selected black-box target."""
import csv
import importlib.util
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/realsense_20261030_134523_clean'
DB=ROOT/'20261030_134523.db3'
OUT=ROOT/'results/room_improve_target_raw_20261103'

def load(name):
    s=importlib.util.spec_from_file_location(name,ROOT/'scripts'/name);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    clean=load('clean-realsense-scene.py');capture=load('prepare-realsense-scene.py');adapter=load('prepare-sensor-depth.py')
    cal=json.loads((DATA/'calibration_clean.json').read_text()); manifest=json.loads((DATA/'capture_manifest.json').read_text()); frame=next(f for f in manifest['frames'] if f['name']=='frame_000024.png')
    c,d=cal['raw_color'],cal['raw_depth'];K0=clean.intrinsic(c); width=848;height=round(int(c['height'])*width/int(c['width']));Kt=K0.copy();Kt[0]*=width/int(c['width']);Kt[1]*=height/int(c['height'])
    yy,xx=np.mgrid[:height,:width];ideal=np.column_stack(((xx.ravel()-Kt[0,2])/Kt[0,0],(yy.ravel()-Kt[1,2])/Kt[1,1]));coeff=np.array([float(v) for v in c['coeffs'].split(',')]);raw=clean.invert_brown(ideal,coeff)
    maps=(raw*[K0[0,0],K0[1,1]]+[K0[0,2],K0[1,2]]).reshape(height,width,2).astype('f4');mx,my=cal['crop_xy'];maps=maps[my:height-my,mx:width-mx]
    K=np.array(cal['K']);R=np.array(cal['depth_to_color_R']);t=np.array(cal['depth_to_color_t_m']);w,h=cal['width'],cal['height']
    with (DATA/'imu/gyro.csv').open(newline='') as f:
        grow=list(csv.DictReader(f))
    gt=np.array([int(r['timestamp_ns']) for r in grow],dtype=np.int64)
    before=DB.stat(); records=[]
    with sqlite3.connect(DB.as_uri()+'?mode=ro',uri=True) as db:
        db.execute('PRAGMA query_only=ON');topics=db.execute('select id,name from topics').fetchall()
        rgbid=next(i for i,n in topics if n.endswith('/Color_0/image/data'));did=next(i for i,n in topics if n.endswith('/Depth_0/image/data'))
        rgbrows=db.execute('select id,timestamp from messages where topic_id=? order by timestamp',(rgbid,)).fetchall();drows=db.execute('select id,timestamp from messages where topic_id=? order by timestamp',(did,)).fetchall()
        rtime=np.array([r[1] for r in rgbrows],dtype=np.int64);dtime=np.array([r[1] for r in drows],dtype=np.int64)
        for offset in (-.30,-.15,-.07,0,.07,.15,.30):
            i=int(np.argmin(abs(rtime-(frame['timestamp_ns']+int(offset*1e9)))));mid,stamp=rgbrows[i];j=int(np.argmin(abs(dtime-stamp)));didval,dstamp=drows[j]
            rgb,_=capture.CDR(db.execute('select data from messages where id=?',(mid,)).fetchone()[0]).image();rgb=cv2.remap(rgb,maps[:,:,0],maps[:,:,1],cv2.INTER_LINEAR)
            if offset==0:
                reference=np.asarray(Image.open(DATA/'images'/frame['name']).convert('RGB'))
                assert mid==frame['message_id'] and didval==frame['depth_message_id'] and np.array_equal(rgb,reference)
            dep,_=adapter.decode_depth(db.execute('select data from messages where id=?',(didval,)).fetchone()[0]);z=dep*cal['depth_units'];valid=(z>.3)&(z<6)&np.isfinite(z);ys,xs=np.nonzero(valid);v=z[valid]
            xyz=np.column_stack(((xs-float(d['ppx']))/float(d['fx'])*v,(ys-float(d['ppy']))/float(d['fy'])*v,v))@R.T+t
            uv=xyz[:,:2]/xyz[:,2:3]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]];x=np.rint(uv[:,0]).astype(int);y=np.rint(uv[:,1]).astype(int);keep=(xyz[:,2]>0)&(x>=0)&(x<w)&(y>=0)&(y<h);x,y,zp=x[keep],y[keep],xyz[keep,2];keys=y*w+x;order=np.lexsort((zp,keys));_,first=np.unique(keys[order],return_index=True);ind=order[first];dense=np.zeros((h,w),np.float32);dense[y[ind],x[ind]]=zp[ind]
            edge=np.zeros((h,w),bool)
            for axis in (0,1):
                a,b=(dense[:-1],dense[1:]) if axis==0 else (dense[:,:-1],dense[:,1:]);e=(a>0)&(b>0)&(abs(a-b)>np.maximum(.06,.025*np.minimum(a,b)))
                if axis==0:edge[:-1]|=e
                else:edge[:,:-1]|=e
            overlay=rgb.copy();overlay[edge]=[255,0,255]
            prefix=f'rgb_{mid}';Image.fromarray(rgb).save(OUT/(prefix+'.png'));Image.fromarray(overlay).save(OUT/(prefix+'.edges.png'));np.savez_compressed(OUT/(prefix+'.depth.npz'),depth_m=dense)
            gi=int(np.argmin(abs(gt-stamp)));g=grow[gi];mag=float(np.linalg.norm([float(g[k]) for k in ('x','y','z')]))
            rec=dict(rgb_message_id=mid,depth_message_id=didval,rgb_timestamp_ns=stamp,depth_timestamp_ns=dstamp,offset_from_reference_s=(stamp-frame['timestamp_ns'])/1e9,pair_gap_ms=abs(stamp-dstamp)/1e6,gyro_rad_s=mag,gyro_gap_ms=abs(int(gt[gi])-stamp)/1e6,rgb=prefix+'.png',edges=prefix+'.edges.png',depth=prefix+'.depth.npz')
            records.append(rec);print(mid,round(rec['offset_from_reference_s'],3),'gyro',round(mag,3),flush=True)
    assert len({r['rgb_message_id'] for r in records})==len(records)
    after=DB.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    sheet=Image.new('RGB',(834,8*260),'white');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(records):
        a=Image.open(OUT/r['rgb']).resize((417,232));b=Image.open(OUT/r['edges']).resize((417,232));y=i*260;sheet.paste(a,(0,y+25));sheet.paste(b,(417,y+25));draw.text((3,y+5),f"RGB {r['rgb_message_id']} offset {r['offset_from_reference_s']:.3f}s gyro {r['gyro_rad_s']:.3f}; depth jumps magenta",fill='black')
    sheet.crop((0,0,834,len(records)*260)).save(OUT/'contact_sheet.jpg',quality=94)
    (OUT/'manifest.json').write_text(json.dumps(dict(reference=frame['name'],records=records,reference_rgb_exact_match=True,source_read_only=True,
        note='Selected original RGB frames, nearest timestamp depth; unfilled projection, 0 denotes no sample. Gyro is nearest single sample.'),indent=2),encoding='utf-8')

if __name__=='__main__':main()
