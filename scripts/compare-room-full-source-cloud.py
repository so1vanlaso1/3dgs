"""Compare experimental sample coverage, retaining all samples and flags."""
import json
from pathlib import Path
import numpy as np
import pycolmap
from scipy.spatial import cKDTree
from plyfile import PlyData,PlyElement

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/room_improve_full_source_cloud_20261103'
def main():
    with np.load(OUT/'evidence.npz') as d:points=d['points_m'];counts=d['check_counts']
    model=pycolmap.Reconstruction(ROOT/'data/realsense_20261030_134523_clean/rgbd/dataset/sparse');original=np.array([p.xyz for p in model.points3D.values()])
    a=cKDTree(points).query(original,workers=-1)[0];b=cKDTree(original).query(points,workers=-1)[0]
    no=counts[:,0]==0;support=(counts[:,1]>0);conf=(counts[:,2]>=3)&(~support);mixed=(counts[:,2]>=3)&support;other=~(no|support|conf)
    color=np.empty((len(points),3),np.uint8);color[no]=[255,220,0];color[other]=[165,165,165];color[support]=[40,200,70];color[mixed]=[255,140,0];color[conf]=[240,40,40]
    v=np.empty(len(points),dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
    for i,k in enumerate(('x','y','z')):v[k]=points[:,i]
    for i,k in enumerate(('red','green','blue')):v[k]=color[:,i]
    file=OUT/'development_only_uncertainty_colors.ply';PlyData([PlyElement.describe(v,'vertex')],text=False).write(file)
    q=PlyData.read(file)['vertex'].data;assert len(q)==len(points) and all(np.array_equal(q[k],v[k]) for k in v.dtype.names)
    report=dict(original_points=len(original),experimental_points=len(points),original_to_experimental_m_percentiles=np.percentile(a,[50,90,95]).tolist(),
      experimental_to_original_m_percentiles=np.percentile(b,[50,90,95]).tolist(),original_within_025m_fraction=float((a<.025).mean()),
      colors=dict(yellow_unknown_no_match=int(no.sum()),red_ge3_conflicts_no_support=int(conf.sum()),orange_ge3_conflicts_with_support=int(mixed.sum()),
        green_some_support_without_ge3_conflicts=int((support&~mixed).sum()),gray_matched_without_support_or_ge3conflict=int(other.sum())),
      note='Nearest-neighbor distance is sampling coverage relative to original initialization, not geometry accuracy. Pose and depth calibration unvalidated. Uncertainty colors reflect selected sparse check views, not verified erroneous geometry. Full-source experimental cloud is not accepted for training or established as improved.')
    assert sum(report['colors'].values())==len(points)
    (OUT/'coverage_comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
