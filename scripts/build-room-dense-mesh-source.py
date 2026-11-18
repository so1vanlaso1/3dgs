"""Create denser development-only RGB-D sample for an experimental mesh."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pycolmap
from PIL import Image
from plyfile import PlyData, PlyElement

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/room_improve_sdk_rectified_20261103'
BASE=ROOT/'results/room_improve_sdk_source_cloud_20261103'
OUT=ROOT/'results/room_improve_sdk_dense_mesh_source_20261103'

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    names=json.loads((BASE/'report.json').read_text(encoding='utf8'))['development_images']
    model=pycolmap.Reconstruction(DATA/'rgbd/dataset/sparse'); ims={x.name:x for x in model.images.values()}
    cam=next(iter(model.cameras.values())); fx,fy,cx,cy=cam.params
    rng=np.random.default_rng(4913); points=[]; colors=[]; sources=[]
    for j,name in enumerate(names):
        with np.load(DATA/'aligned_depth_meters'/(name+'.npz')) as f:
            uv=f['points']; z=f['depths']
        select=rng.choice(len(z),min(8000,len(z)),replace=False); uv=uv[select]; z=z[select]
        q=np.c_[(uv[:,0]-cx)*z/fx,(uv[:,1]-cy)*z/fy,z]
        T=ims[name].cam_from_world().matrix(); world=(q-T[:,3])@T[:,:3]
        rgb=np.asarray(Image.open(DATA/'images'/name).convert('RGB'))
        x=np.clip(np.rint(uv[:,0]).astype(int),0,cam.width-1); y=np.clip(np.rint(uv[:,1]).astype(int),0,cam.height-1)
        points.append(world.astype('f4'));colors.append(rgb[y,x]);sources.append(np.full(len(z),j,'i2'))
        if j%100==0:print('source',j,name,flush=True)
    p=np.concatenate(points); c=np.concatenate(colors); s=np.concatenate(sources); sampled=len(p)
    order=rng.permutation(sampled);vox=np.floor(p[order]/.025).astype('i4');_,indices=np.unique(vox,axis=0,return_index=True);keep=order[indices]
    if len(keep)>900000:keep=rng.choice(keep,900000,replace=False)
    p=p[keep];c=c[keep];s=s[keep]
    np.savez_compressed(OUT/'evidence.npz',points_m=p,colors=c,source_group_index=s)
    v=np.empty(len(p),dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
    for i,k in enumerate(('x','y','z')):v[k]=p[:,i]
    for i,k in enumerate(('red','green','blue')):v[k]=c[:,i]
    path=OUT/'development_only_unfiltered.ply';PlyData([PlyElement.describe(v,'vertex')],text=False).write(path)
    r=PlyData.read(path)['vertex'].data;assert len(r)==len(p) and all(np.array_equal(r[k],v[k]) for k in v.dtype.names)
    report=dict(development_images=names,samples_per_frame=8000,raw_samples=sampled,voxel_m=.025,cap=900000,retained=len(p),seed=4913,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),caveat='No cleaning, pose/calibration validation, or independent check-view uncertainty labels. Uses original 387 development frames and excludes original 60 heldout and 30 check frames as sources.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in report.items() if k!='development_images'},indent=2))
if __name__=='__main__':main()
