"""Screen initialization points against sampled RGB-D views; diagnostic, not deletion."""
import json
from pathlib import Path
import numpy as np
import pycolmap
from scipy.spatial import cKDTree
root=Path('data/realsense_20261030_134523_clean')
out=Path('results/realsense_20261030_134523_120k/point_cloud_audit'); out.mkdir(parents=True,exist_ok=True)
m=pycolmap.Reconstruction(root/'rgbd/dataset/sparse')
pts=np.array([p.xyz for p in m.points3D.values()]); ids=np.array(list(m.points3D.keys()))
cam=next(iter(m.cameras.values())); fx,fy,cx,cy=cam.params
images=sorted(m.images.values(),key=lambda i:i.name)[::8]
matched=np.zeros(len(pts),np.uint16); support=np.zeros(len(pts),np.uint16); free=np.zeros(len(pts),np.uint16); behind=np.zeros(len(pts),np.uint16)
for k,im in enumerate(images):
    tf=im.cam_from_world().matrix(); q=pts@tf[:,:3].T+tf[:,3]; z=q[:,2]
    ok=(z>.3)&(z<6); idx=np.flatnonzero(ok); q=q[idx]; zz=z[idx]
    uv=np.column_stack((fx*q[:,0]/zz+cx,fy*q[:,1]/zz+cy)); inside=(uv[:,0]>=0)&(uv[:,0]<cam.width)&(uv[:,1]>=0)&(uv[:,1]<cam.height)
    idx=idx[inside]; uv=uv[inside]; zz=zz[inside]
    with np.load(root/'aligned_depth_meters'/(im.name+'.npz')) as d: xy=d['points']; depth=d['depths']
    dist,j=cKDTree(xy).query(uv,distance_upper_bound=2,workers=-1); valid=np.isfinite(dist); idx=idx[valid]; delta=zz[valid]-depth[j[valid]]
    matched[idx]+=1; support[idx]+=np.abs(delta)<=.08; free[idx]+=delta<-.15; behind[idx]+=delta>.15
    if k%10==0: print(k,im.name,flush=True)
mask=(free>=3)&(support==0); mixed=(free>=3)&(support>0)
tree=cKDTree(pts); dd,_=tree.query(pts,k=6,workers=-1); nn=dd[:,5]
report={'points':len(pts),'sampled_views':len(images),'view_names':[im.name for im in images], 'pixel_max_distance':2,'support_tolerance_m':.08,'free_space_margin_m':.15,'no_valid_depth_views':int(np.sum(matched==0)),'any_support':int((support>0).sum()),'at_least_3_free_no_support':int(mask.sum()),'at_least_3_free_with_support':int(mixed.sum()),'fifth_neighbor_m_percentiles_all':np.percentile(nn,[50,90,99]).tolist(),'fifth_neighbor_m_percentiles_flagged':np.percentile(nn[mask],[50,90,99]).tolist() if mask.any() else [],'limitations':'Sparse depth samples; observations may be temporally correlated, dynamic, occluded or affected by pose errors. No ground truth or deletion. Intermediate residuals uncounted.'}
(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
np.savez_compressed(out/'point_evidence.npz',point_ids=ids,points_m=pts,matched_views=matched,support_views=support,free_views=free,behind_views=behind,fifth_neighbor_m=nn)
print(json.dumps(report,indent=2),flush=True)
