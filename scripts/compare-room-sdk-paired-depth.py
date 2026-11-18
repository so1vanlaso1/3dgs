"""Compare identical source depth samples under original and candidate graphs."""
import json
from pathlib import Path
import numpy as np
import pycolmap

ROOT=Path(__file__).resolve().parents[1];A=ROOT/'results/room_improve_full_source_cloud_20261103';B=ROOT/'results/room_improve_sdk_source_cloud_20261103';OUT=ROOT/'results/room_improve_sdk_paired_depth_20261103'
def keys(directory,modeldir):
    meta=json.loads((directory/'report.json').read_text());m=pycolmap.Reconstruction(modeldir);ims={x.name:x for x in m.images.values()};cam=next(iter(m.cameras.values()));fx,fy,cx,cy=cam.params
    with np.load(directory/'evidence.npz') as f:p=f['points_m'];src=f['source_group_index']
    key=np.empty(len(p),np.int64);zkey=np.empty(len(p),np.float64)
    for j,name in enumerate(meta['development_images']):
        ix=np.flatnonzero(src==j)
        if not len(ix):continue
        T=ims[name].cam_from_world().matrix();q=p[ix]@T[:,:3].T+T[:,3];z=q[:,2];u=np.rint(fx*q[:,0]/z+cx).astype(np.int64);v=np.rint(fy*q[:,1]/z+cy).astype(np.int64)
        assert (u>=0).all() and (u<cam.width).all() and (v>=0).all() and (v<cam.height).all()
        key[ix]=j*(cam.width*cam.height)+v*cam.width+u;zkey[ix]=z
    assert len(np.unique(key))==len(key)
    with np.load(directory/'dense_check_evidence.npz') as f:c=f['stable_3x3']
    return key,zkey,c,meta
def main():
    OUT.mkdir(parents=True,exist_ok=False);ka,za,ca,ma=keys(A,ROOT/'data/realsense_20261030_134523_clean/rgbd/dataset/sparse');kb,zb,cb,mb=keys(B,ROOT/'data/room_improve_sdk_rectified_20261103/rgbd/dataset/sparse')
    assert ma['development_images']==mb['development_images'] and ma['check_images']==mb['check_images']
    common,ia,ib=np.intersect1d(ka,kb,assume_unique=True,return_indices=True);assert len(common)>0
    assert np.max(abs(za[ia]-zb[ib]))<1e-5
    a=ca[ia];b=cb[ib]
    def categories(c):
        return np.where(c[:,0]==0,0,np.where(c[:,1]>0,1,np.where(c[:,2]>=3,2,3)))
    aa=categories(a);bb=categories(b);table=np.zeros((4,4),int)
    for i,j in zip(aa,bb):table[i,j]+=1
    np.savez_compressed(OUT/'matched_evidence.npz',source_pixel_keys=common,old_counts=a,new_counts=b,old_category=aa,new_category=bb)
    report=dict(matched_source_depth_samples=len(common),old_total=len(ka),candidate_total=len(kb),max_source_z_difference_m=float(np.max(abs(za[ia]-zb[ib]))),
       categories=['no stable observation','any support (may also conflict)','3+ free conflicts and no support','other observed'],
       rows_old_columns_candidate=table.tolist(),old_category_counts=np.bincount(aa,minlength=4).tolist(),candidate_category_counts=np.bincount(bb,minlength=4).tolist(),
       caveat='Matched source depth pixel and same check frame identities, but graph poses differ, so projections/visible observations differ. Check frames share data/calibration and may be used in both graphs; no independent physical ground truth. Voxel selection conditions matched subset; cannot extrapolate to all source samples. Support/conflict thresholds exploratory; no cleaning or training decision.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
