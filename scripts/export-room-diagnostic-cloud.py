"""Export original points with evidence colors for review, not training."""
import json
from pathlib import Path
import numpy as np
import pycolmap
from plyfile import PlyData,PlyElement

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/room_improve_multiview_diagnostic_20261103'
def main():
    with np.load(OUT/'evidence.npz') as data:
        ids=data['point_ids'];xyz=data['points_m'];flag=data['development_flagged'];check=data['check']
    model=pycolmap.Reconstruction(ROOT/'data/realsense_20261030_134523_clean/rgbd/dataset/sparse')
    color=np.array([model.points3D[int(i)].color for i in ids],np.uint8)
    support=flag & (check[:,1]>0);unmatched=flag & (check[:,0]==0);other=flag & ~support & ~unmatched
    color[other]=[255,0,0];color[support]=[255,150,0];color[unmatched]=[255,255,0]
    v=np.empty(len(ids),dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
    for i,name in enumerate(('x','y','z')):v[name]=xyz[:,i]
    for i,name in enumerate(('red','green','blue')):v[name]=color[:,i]
    path=OUT/'diagnostic_all_points_colored.ply';PlyData([PlyElement.describe(v,'vertex')],text=False).write(path)
    loaded=PlyData.read(path)['vertex'].data
    assert len(loaded)==len(ids) and all(np.array_equal(loaded[n],v[n]) for n in v.dtype.names)
    report=dict(points=len(ids),removed_points=0,red=int(other.sum()),orange=int(support.sum()),yellow=int(unmatched.sum()),
       description='Original coordinates, all points retained, float32 PLY; red = development flagged with check observation and no support, orange = development flagged with check support, yellow = development flagged without check match. Other points retain original colors. Diagnostic visualization only; NOT a cleaned training candidate.')
    (OUT/'diagnostic_cloud.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
