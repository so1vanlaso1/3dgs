"""Visualize diagnostic depth-consistency labels on original initialization points."""
import sys,time
from pathlib import Path
import numpy as np
import pycolmap,viser
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from datasets.colmap import Parser
root=Path('data/realsense_20261030_134523_clean/rgbd/dataset')
audit=Path('results/realsense_20261030_134523_120k/point_cloud_audit/point_evidence.npz')
p=Parser(str(root),factor=1,normalize=True,load_exposure=False)
m=pycolmap.Reconstruction(root/'sparse'); sorted_ids=np.array(sorted(m.points3D.keys()))
d=np.load(audit); order=np.argsort(d['point_ids']); assert np.array_equal(d['point_ids'][order],sorted_ids)
free=d['free_views'][order]; support=d['support_views'][order]; red=(free>=3)&(support==0); orange=(free>=3)&(support>0); other=~(red|orange)
s=viser.ViserServer(host='127.0.0.1',port=8090,label='Point cloud depth audit')
s.scene.add_point_cloud('/Other points',p.points[other],p.points_rgb[other],point_size=.003,point_shape='circle',precision='float32')
s.scene.add_point_cloud('/Red - conflict, no support',p.points[red],(255,25,25),point_size=.009,point_shape='circle',precision='float32')
s.scene.add_point_cloud('/Orange - conflict and support',p.points[orange],(255,155,0),point_size=.007,point_shape='circle',precision='float32')
center=np.median(p.points,axis=0); span=float(np.max(np.percentile(p.points,95,axis=0)-np.percentile(p.points,5,axis=0)))
s.initial_camera.look_at=center; s.initial_camera.position=center+np.array([span,-span,span]); s.initial_camera.up=(0,0,1)
print(f'OTHER {other.sum()} RED {red.sum()} ORANGE {orange.sum()}',flush=True)
while True: time.sleep(10)
