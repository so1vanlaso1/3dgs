"""Measure all original RGB-D pose-edge residuals before and after correction."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

root=Path('data/realsense_20261030_111354_all30/rgbd')
out=Path('results/courtyard_revisit_repair_20261102')
old=np.load(root/'poses.npz')['camtoworlds']
new=np.load(out/'candidate_poses.npz')['camtoworlds']
edges=json.loads((root/'edges.json').read_text())
i=np.array([e['i'] for e in edges]); j=np.array([e['j'] for e in edges]); T=np.array([e['T'] for e in edges])
def metrics(poses):
    predicted=np.linalg.inv(poses[j])@poses[i]
    error=np.linalg.inv(T)@predicted
    translation=np.linalg.norm(error[:,:3,3],axis=1)
    rotation=np.degrees(Rotation.from_matrix(error[:,:3,:3]).magnitude())
    return translation,rotation
a,b=metrics(old);c,d=metrics(new)
def summary(x):return np.percentile(x,[50,90,95,99,100]).tolist()
report=dict(total_edges=len(edges),percentiles=[50,90,95,99,100],
    old_translation_m=summary(a),new_translation_m=summary(c),
    old_rotation_deg=summary(b),new_rotation_deg=summary(d),
    additional_edges_above_10cm=int((c>.1).sum()-(a>.1).sum()),
    additional_edges_above_3deg=int((d>3).sum()-(b>3).sum()),
    note='These are original fitted graph constraints, not independent ground truth; all edges include original rejected edges.')
pair=np.load(out/'pair_3236_6470_depth.npz')
x,y=pair['a'],pair['b']
def matching_depth_error(poses):
    transform=np.linalg.inv(poses[6470])@poses[3236]
    return np.linalg.norm(x@transform[:3,:3].T+transform[:3,3]-y,axis=1)
old_depth,new_depth=matching_depth_error(old),matching_depth_error(new)
report['heldout_depth_correspondences']=dict(count=len(x),
    all_pairs_old_m=summary(old_depth),all_pairs_candidate_m=summary(new_depth),
    selected_ransac_inliers=int(pair['rigid_inliers'].sum()),
    selected_inliers_old_m=summary(old_depth[pair['rigid_inliers']]),
    selected_inliers_candidate_m=summary(new_depth[pair['rigid_inliers']]),
    limitation='Nearby revisit shares later image with fitted constraints; inliers selected using the heldout pair itself; not independent ground truth.')
(out/'original_edge_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
