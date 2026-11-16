"""Check experimental cloud against independent-source unfilled depth maps."""
import json
from pathlib import Path
import cv2
import numpy as np
import pycolmap
from plyfile import PlyData,PlyElement

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/realsense_20261030_134523_clean';OUT=ROOT/'results/room_improve_full_source_cloud_20261103';DENSE=ROOT/'results/room_improve_full_cloud_dense_checks_20261103'
def main():
    base=json.loads((OUT/'report.json').read_text());m=json.loads((DATA/'capture_manifest.json').read_text());mf={f['name']:f for f in m['frames']}
    with np.load(OUT/'evidence.npz') as f:p=f['points_m'];src=f['source_group_index']
    source_time=np.array([mf[n]['timestamp_ns'] for n in base['development_images']],np.int64)[src];model=pycolmap.Reconstruction(DATA/'rgbd/dataset/sparse');ims={i.name:i for i in model.images.values()};cam=next(iter(model.cameras.values()));fx,fy,cx,cy=cam.params
    counts=np.zeros((2,len(p),4),np.uint16)
    for j,name in enumerate(base['check_images']):
        with np.load(DENSE/(name+'.dense_depth.npz')) as f:z=f['depth_m']
        assert z.shape==(cam.height,cam.width) and np.isfinite(z).all()
        low=cv2.erode(z,np.ones((3,3),np.uint8));high=cv2.dilate(z,np.ones((3,3),np.uint8));stable=(low>.3)&(high<6)&((high-low)<np.maximum(.03,z*.02))
        tf=ims[name].cam_from_world().matrix();q=p@tf[:,:3].T+tf[:,3];qz=q[:,2];ok=(qz>.3)&(qz<6)&(np.abs(source_time-mf[name]['timestamp_ns'])>5e9)
        ix=np.flatnonzero(ok);q=q[ix];pred=qz[ix];u=np.rint(fx*q[:,0]/pred+cx).astype(int);v=np.rint(fy*q[:,1]/pred+cy).astype(int);inimage=(u>=0)&(u<cam.width)&(v>=0)&(v<cam.height)
        ix=ix[inimage];pred=pred[inimage];u=u[inimage];v=v[inimage];meas=z[v,u];valid=meas>0;isstable=stable[v,u]
        for k,validk in enumerate((valid,valid&isstable)):
            ii=ix[validk];delta=pred[validk]-meas[validk];counts[k,ii,0]+=1;counts[k,ii,1]+=np.abs(delta)<=.08;counts[k,ii,2]+=delta<-.15;counts[k,ii,3]+=delta>.15
        if j%10==0:print('dense check',j,name,flush=True)
    np.savez_compressed(OUT/'dense_check_evidence.npz',all_valid=counts[0],stable_3x3=counts[1]);report={}
    for k,label in enumerate(('all_valid','stable_3x3')):
        c=counts[k];report[label]=dict(no_observation=int((c[:,0]==0).sum()),any_support=int((c[:,1]>0).sum()),
           ge3_free_no_support=int(((c[:,2]>=3)&(c[:,1]==0)).sum()),ge3_free_with_support=int(((c[:,2]>=3)&(c[:,1]>0)).sum()),any_behind=int((c[:,3]>0).sum()))
    s=counts[1];no=s[:,0]==0;sup=s[:,1]>0;conf=(s[:,2]>=3)&~sup;mixed=(s[:,2]>=3)&sup;other=~(no|sup|conf)
    rgb=np.empty((len(p),3),np.uint8);rgb[no]=[255,220,0];rgb[other]=[165,165,165];rgb[sup]=[40,200,70];rgb[mixed]=[255,140,0];rgb[conf]=[240,40,40]
    v=np.empty(len(p),dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
    for i,k in enumerate(('x','y','z')):v[k]=p[:,i]
    for i,k in enumerate(('red','green','blue')):v[k]=rgb[:,i]
    path=OUT/'development_dense_stable_uncertainty.ply';PlyData([PlyElement.describe(v,'vertex')],text=False).write(path);reload=PlyData.read(path)['vertex'].data
    assert len(reload)==len(p) and all(np.array_equal(reload[k],v[k]) for k in v.dtype.names) and (counts[:,:,1:]<=counts[:,:,:1]).all()
    report.update(points=len(p),views=base['check_images'],source_check_time_gap_seconds=5,thresholds='support <= 0.08m; free-space more than 0.15m in front; behind >0.15m',
      stable_rule='all 3x3 projected depths valid in 0.3–6m, max-min < max(0.03m, 2% center Z); no hole filling',
      colors='yellow no stable observation; gray stable observation but no support and <3 conflicts; green any support and <3 conflicts; orange support and >=3 conflicts; red >=3 conflicts and no support',
      caveat='Disjoint depth-source/check frames separated >5s from each source; shared estimated poses and calibration; test views may contribute to pose graph. Occlusion, dynamics, depth inaccuracies, source uncertainty and systematic pose errors can alter labels. Direct projected depth, not physical ground truth; no filtering, no training.')
    (OUT/'dense_check_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:report[k] for k in ('points','all_valid','stable_3x3')},indent=2))
if __name__=='__main__':main()
