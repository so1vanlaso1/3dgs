"""Diagnostic multi-view point evidence on training frames; no point filtering."""
import json
from pathlib import Path
import numpy as np
import pycolmap
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/realsense_20261030_134523_clean'
OUT=ROOT/'results/room_improve_multiview_diagnostic_20261103'

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    model=pycolmap.Reconstruction(DATA/'rgbd/dataset/sparse')
    ids=np.array(sorted(model.points3D.keys()));pts=np.array([model.points3D[int(i)].xyz for i in ids])
    cam=next(iter(model.cameras.values()));fx,fy,cx,cy=cam.params;ims=sorted(model.images.values(),key=lambda im:im.name)
    results={};names={}
    for label,offset in [('development',2),('check',10)]:
        selected=ims[offset::16]; names[label]=[im.name for im in selected]
        counts=np.zeros((len(ids),4),np.uint16)
        for j,im in enumerate(selected):
            tf=im.cam_from_world().matrix();q=pts@tf[:,:3].T+tf[:,3];z=q[:,2]
            ix=np.flatnonzero((z>.3)&(z<6));q=q[ix];zz=z[ix]
            uv=np.column_stack((fx*q[:,0]/zz+cx,fy*q[:,1]/zz+cy));ok=(uv[:,0]>=0)&(uv[:,0]<cam.width)&(uv[:,1]>=0)&(uv[:,1]<cam.height)
            ix=ix[ok];uv=uv[ok];zz=zz[ok]
            with np.load(DATA/'aligned_depth_meters'/(im.name+'.npz')) as d: xy=d['points'];depth=d['depths']
            dist,idx=cKDTree(xy).query(uv,distance_upper_bound=2,workers=-1);ok=np.isfinite(dist);ix=ix[ok];delta=zz[ok]-depth[idx[ok]]
            counts[ix,0]+=1;counts[ix,1]+=np.abs(delta)<=.08;counts[ix,2]+=delta<-.15;counts[ix,3]+=delta>.15
            if j%10==0:print(label,j,im.name,flush=True)
        results[label]=counts
    d=results['development'];c=results['check'];flag=(d[:,2]>=3)&(d[:,1]==0)
    report=dict(points=len(ids),views=names, thresholds=dict(nearest_pixel=2,support_m=.08,free_space_m=.15),
      development_flagged=int(flag.sum()),check_flagged_with_any_support=int(np.sum(flag & (c[:,1]>0))),
      check_flagged_with_ge1_free=int(np.sum(flag & (c[:,2]>=1))),check_flagged_without_match=int(np.sum(flag & (c[:,0]==0))),
      limitations='Same recording, original unvalidated poses and sampled depths; temporally correlated frames and systematic errors remain. Free space in front of depth is diagnostic only, missing and behind are not contradictions. Thresholds exploratory; not independent physical truth. No points deleted.')
    np.savez_compressed(OUT/'evidence.npz',point_ids=ids,points_m=pts,development=d,check=c,development_flagged=flag)
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='views'},indent=2),flush=True)
if __name__=='__main__':main()
