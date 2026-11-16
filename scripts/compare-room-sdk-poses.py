"""Compare candidate and baseline camera trajectories after robust rigid alignment."""
import csv
import json
from pathlib import Path
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];A=ROOT/'data/realsense_20261030_134523_clean/rgbd/dataset/sparse';B=ROOT/'data/room_improve_sdk_rectified_20261103/rgbd/dataset/sparse';OUT=ROOT/'results/room_improve_sdk_pose_comparison_20261103'
def get(model):
    result={}
    for im in model.images.values():
        w=np.eye(4);w[:3]=im.cam_from_world().matrix();result[im.name]=np.linalg.inv(w)
    return result
def fit(a,b):
    ca=a.mean(0);cb=b.mean(0);u,_,vt=np.linalg.svd((a-ca).T@(b-cb));r=vt.T@u.T
    if np.linalg.det(r)<0:vt[-1]*=-1;r=vt.T@u.T
    return r,cb-r@ca
def main():
    OUT.mkdir(parents=True,exist_ok=False);a=get(pycolmap.Reconstruction(A));b=get(pycolmap.Reconstruction(B));names=sorted(set(a)&set(b));split=json.loads((ROOT/'data/room_improve_sdk_rectified_20261103/experiment_split.json').read_text());held=set(split['val']);train=np.array([n not in held for n in names]);x=np.array([a[n][:3,3] for n in names]);y=np.array([b[n][:3,3] for n in names]);indices=np.flatnonzero(train);rng=np.random.default_rng(20261103);best=np.zeros(len(indices),bool)
    for _ in range(2000):
        s=rng.choice(indices,3,replace=False)
        if np.linalg.norm(np.cross(x[s[1]]-x[s[0]],x[s[2]]-x[s[0]]))<1e-5:continue
        R,t=fit(x[s],y[s]);err=np.linalg.norm(x[indices]@R.T+t-y[indices],axis=1);mask=err<.25
        if mask.sum()>best.sum():best=mask
    if best.sum()<3:raise RuntimeError('insufficient rigid alignment points')
    R,t=fit(x[indices[best]],y[indices[best]]);d=np.linalg.norm(x@R.T+t-y,axis=1);rot=np.array([np.degrees(Rotation.from_matrix((R@a[n][:3,:3]).T@b[n][:3,:3]).magnitude()) for n in names])
    rows=[dict(name=n,role='val' if n in held else 'train',position_difference_m=float(d[i]),rotation_difference_deg=float(rot[i]),fit_inlier=bool(train[i] and d[i]<.25)) for i,n in enumerate(names)]
    with (OUT/'per_frame.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    def q(v):return np.percentile(v,[50,90,95,100]).tolist()
    report=dict(common_images=len(names),train_images=int(train.sum()),heldout_images=int((~train).sum()),alignment='rigid only, original metric coordinates to candidate; 2000 random three-center fits, 0.25m inlier radius; refit best training subset',
      rotation=R.tolist(),translation=t.tolist(),training_fit_inliers=int(best.sum()),train_translation_m_percentiles=q(d[train]),heldout_translation_m_percentiles=q(d[~train]),
      train_rotation_deg_percentiles=q(rot[train]),heldout_rotation_deg_percentiles=q(rot[~train]),
      caveat='Differences between two estimated trajectories, not errors against ground truth. Both graphs may use all frames, including validation frames, and share source data. RANSAC selection and fit affect alignment; low error does not prove correctness. No pose changed.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
