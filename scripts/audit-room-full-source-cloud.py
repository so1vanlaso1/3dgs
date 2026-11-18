"""Development-only depth cloud and temporally separated check-view evidence.

Diagnostic only: poses and RGB-depth calibration remain unvalidated.
"""
import json
from pathlib import Path
import numpy as np
import pycolmap
from PIL import Image
from scipy.spatial import cKDTree
from plyfile import PlyData,PlyElement

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/realsense_20261030_134523_clean'
OUT=ROOT/'results/room_improve_full_source_cloud_20261103'
def main():
    OUT.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(4913)
    model=pycolmap.Reconstruction(DATA/'rgbd/dataset/sparse');ims=sorted(model.images.values(),key=lambda i:i.name);cam=next(iter(model.cameras.values()));fx,fy,cx,cy=cam.params
    check=ims[10::16];check_names={i.name for i in check}
    dev=[im for index,im in enumerate(ims) if index%8!=0 and im.name not in check_names]
    m=json.loads((DATA/'capture_manifest.json').read_text());mf={r['name']:r for r in m['frames']}
    points=[];colors=[];sources=[]
    for j,im in enumerate(dev):
        with np.load(DATA/'aligned_depth_meters'/(im.name+'.npz')) as d:uv=d['points'];z=d['depths']
        keep=rng.choice(len(z),min(1500,len(z)),replace=False);uv=uv[keep];z=z[keep]
        q=np.c_[(uv[:,0]-cx)*z/fx,(uv[:,1]-cy)*z/fy,z];T=np.eye(4);T[:3]=im.cam_from_world().matrix();W=np.linalg.inv(T);world=q@W[:3,:3].T+W[:3,3]
        rgb=np.asarray(Image.open(DATA/'images'/im.name).convert('RGB'));x=np.clip(np.rint(uv[:,0]).astype(int),0,cam.width-1);y=np.clip(np.rint(uv[:,1]).astype(int),0,cam.height-1)
        points.append(world);colors.append(rgb[y,x]);sources.append(np.full(len(z),j,np.int16))
        if j%75==0:print('source',j,im.name,flush=True)
    points=np.concatenate(points);colors=np.concatenate(colors);sources=np.concatenate(sources);sampled=len(points)
    # One original sample per 2.5 cm voxel; no averaging, no claimed denoising.
    order=rng.permutation(len(points));vox=np.floor(points[order]/.025).astype(np.int32);_,ix=np.unique(vox,axis=0,return_index=True);keep=order[ix]
    if len(keep)>180000:keep=rng.choice(keep,180000,replace=False)
    points=points[keep];colors=colors[keep];sources=sources[keep];times=np.array([mf[i.name]['timestamp_ns'] for i in dev],dtype=np.int64)[sources]
    n=len(points);counts=np.zeros((n,4),np.uint16)
    for j,im in enumerate(check):
        tf=im.cam_from_world().matrix();q=points@tf[:,:3].T+tf[:,3];z=q[:,2];stamp=mf[im.name]['timestamp_ns'];ok=(z>.3)&(z<6)&(np.abs(times-stamp)>5e9)
        ix=np.flatnonzero(ok);q=q[ix];zz=z[ix];uv=np.c_[fx*q[:,0]/zz+cx,fy*q[:,1]/zz+cy];inside=(uv[:,0]>=0)&(uv[:,0]<cam.width)&(uv[:,1]>=0)&(uv[:,1]<cam.height)
        ix=ix[inside];uv=uv[inside];zz=zz[inside]
        with np.load(DATA/'aligned_depth_meters'/(im.name+'.npz')) as d: xy=d['points'];depth=d['depths']
        dist,near=cKDTree(xy).query(uv,distance_upper_bound=2,workers=-1);valid=np.isfinite(dist);ix=ix[valid];delta=zz[valid]-depth[near[valid]]
        counts[ix,0]+=1;counts[ix,1]+=np.abs(delta)<=.08;counts[ix,2]+=delta<-.15;counts[ix,3]+=delta>.15
        if j%10==0:print('check',j,im.name,flush=True)
    np.savez_compressed(OUT/'evidence.npz',points_m=points,colors=colors,source_group_index=sources,check_counts=counts)
    vertices=np.empty(n,dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
    for i,k in enumerate(('x','y','z')):vertices[k]=points[:,i]
    for i,k in enumerate(('red','green','blue')):vertices[k]=colors[:,i]
    PlyData([PlyElement.describe(vertices,'vertex')],text=False).write(OUT/'development_only_unfiltered.ply')
    read=PlyData.read(OUT/'development_only_unfiltered.ply')['vertex'].data;assert len(read)==n and all(np.array_equal(read[k],vertices[k]) for k in vertices.dtype.names)
    report=dict(seed=4913,source_raw_samples=sampled,samples_per_source_frame=1500,voxel_m=.025,cap=180000,retained=n,development_images=[i.name for i in dev],check_images=[i.name for i in check],
       min_source_check_time_gap_s=5,nearest_depth_radius_px=2,support_m=.08,free_space_m=.15,
       check_no_matching_depth=int((counts[:,0]==0).sum()),check_any_support=int((counts[:,1]>0).sum()),
       check_ge3_free_no_support=int(((counts[:,2]>=3)&(counts[:,1]==0)).sum()),check_ge3_free_with_support=int(((counts[:,2]>=3)&(counts[:,1]>0)).sum()),
       check_any_occluded_behind=int((counts[:,3]>0).sum()),
       caveat='Unfiltered experimental subsampled depth; source and check RGB/depth pairs disjoint and >5 s from each point source, but original poses/graph, alignment, scene and preprocessing are shared. Pose graph may use check images. No absolute ground truth; voxel/cap changes representation, not validated cleaning. No trained model.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if not k.endswith('_images')},indent=2))
if __name__=='__main__':main()
