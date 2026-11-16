"""Export candidate poses as a separate COLMAP dataset with depth initialization."""
import json
import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

root=Path('data/realsense_20261030_111354_all30')
source=root/'rgbd/dataset'
ap=argparse.ArgumentParser()
ap.add_argument('--prefer-people',action='store_true',help='Exclude late revisit training frames 6300 through 6800; preserve evaluation split')
args=ap.parse_args()
target=Path('data/courtyard_revisit_people_20261102' if args.prefer_people else 'data/courtyard_revisit_repair_20261102')
result=Path('results/courtyard_revisit_repair_20261102')
report=json.loads((result/'candidate_pose_report.json').read_text())
if not report['passed']: raise RuntimeError('candidate failed pose validation')
poses=np.load(result/'candidate_poses.npz')['camtoworlds']
cal=json.loads((root/'rgbd/calibration.json').read_text()); frames=cal['frames']
split=json.loads((root/'experiment_split.json').read_text()); training=set(split['train'])
if args.prefer_people:
    removed={n for n in training if 6300<=int(Path(n).stem.split('_')[-1])<=6800}
    training-=removed
    split['train']=[n for n in split['train'] if n in training]
    split['excluded']=sorted(set(split.get('excluded',[])) | removed)
    split['people_preference_note']='Excluded late revisit training frames 6300-6800 to reduce conflicting observations where group is absent; validation frames unchanged.'
    print('EXCLUDED_LATE_TRAINING_FRAMES',len(removed),flush=True)
if len(poses)!=len(frames):raise ValueError('frame and pose counts differ')
for d in ('images','sensor_depth','sparse'): (target/d).mkdir(parents=True,exist_ok=True)
(target/'experiment_split.json').write_text(json.dumps(split,indent=2),encoding='utf-8')
if not (target/'capture_manifest.json').exists():
    os.link(root/'capture_manifest.json', target/'capture_manifest.json')
K=np.array(cal['K']); w,h=cal['width'],cal['height']
(target/'sparse/cameras.txt').write_text(f'1 PINHOLE {w} {h} {K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}\n')
rng=np.random.default_rng(42); lines=[]; points=[]; colors=[]
for i,(f,pose) in enumerate(zip(frames,poses)):
    name=f['name']
    for sub,filename in [('images',name),('sensor_depth',name+'.npz')]:
        dst=target/sub/filename
        if not dst.exists():os.link(source/sub/filename,dst)
    inv=np.linalg.inv(pose); q=Rotation.from_matrix(inv[:3,:3]).as_quat(); t=inv[:3,3]
    lines.append(f'{i+1} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]} 1 {name}\n\n')
    if i%10==0 and name in training:
        with np.load(source/'sensor_depth'/(name+'.npz')) as depth:
            uv,z=depth['points'],depth['depths']
        sample=rng.choice(len(z),min(1500,len(z)),replace=False);uv,z=uv[sample],z[sample]
        cam=np.column_stack(((uv[:,0]-K[0,2])*z/K[0,0],(uv[:,1]-K[1,2])*z/K[1,1],z))
        points.append(cam@pose[:3,:3].T+pose[:3,3]); im=cv2.imread(str(source/'images'/name))[:,:,::-1]
        pixels=np.rint(uv).astype(int);colors.append(im[pixels[:,1],pixels[:,0]])
    if i%1000==0:print('EXPORT',i,'/',len(poses),flush=True)
(target/'sparse/images.txt').write_text(''.join(lines))
xyz=np.concatenate(points);rgb=np.concatenate(colors)
_,keep=np.unique(np.floor(xyz/.025).astype(np.int32),axis=0,return_index=True)
if len(keep)>180000:keep=rng.choice(keep,180000,replace=False)
with (target/'sparse/points3D.txt').open('w') as f:
    for n,k in enumerate(keep,1):
        x,y,z=xyz[k];r,g,b=rgb[k];f.write(f'{n} {x} {y} {z} {r} {g} {b} 0\n')
model=pycolmap.Reconstruction();model.read_text(target/'sparse')
if model.num_reg_images()!=len(poses) or model.num_points3D()!=len(keep):raise RuntimeError('bad export')
model.write(target/'sparse')
print('EXPORT_COMPLETE',len(poses),'images',len(keep),'points',flush=True)
