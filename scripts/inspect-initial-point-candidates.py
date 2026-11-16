import json
from pathlib import Path
import numpy as np, pycolmap
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
root=Path('data/realsense_20261030_134523_clean'); out=Path('results/realsense_20261030_134523_120k/point_cloud_audit')
d=np.load(out/'point_evidence.npz'); pts=d['points_m']; free=d['free_views']; support=d['support_views']; ids=d['point_ids']; candidates=np.flatnonzero((free>=3)&(support==0)); m=pycolmap.Reconstruction(root/'rgbd/dataset/sparse'); cam=next(iter(m.cameras.values())); fx,fy,cx,cy=cam.params
chosen=[]
for i in candidates[np.argsort(-free[candidates].astype(int))]:
 if not chosen or np.min(np.linalg.norm(pts[chosen]-pts[i],axis=1))>.35: chosen.append(int(i))
 if len(chosen)==8: break
images=sorted(m.images.values(),key=lambda x:x.name)[::8]; records=[]; sheet=Image.new('RGB',(900, len(chosen)*275),'white'); draw=ImageDraw.Draw(sheet)
for row,i in enumerate(chosen):
 matches=[]
 for im in images:
  t=im.cam_from_world().matrix(); q=t[:,:3]@pts[i]+t[:,3]; z=q[2]
  if z<=.3 or z>=6: continue
  u=fx*q[0]/z+cx; v=fy*q[1]/z+cy
  if not (0<=u<cam.width and 0<=v<cam.height): continue
  with np.load(root/'aligned_depth_meters'/(im.name+'.npz')) as dep: xy=dep['points']; ds=dep['depths']
  dist,j=cKDTree(xy).query([u,v],distance_upper_bound=2)
  if np.isfinite(dist) and ds[j]-z>.15: matches.append((im.name,float(u),float(v),float(z),float(ds[j])))
 for col,match in enumerate(matches[:2]):
  name,u,v,z,depth=match; image=Image.open(root/'images'/name).convert('RGB'); painter=ImageDraw.Draw(image); painter.ellipse((u-10,v-10,u+10,v+10),outline='red',width=4); image.thumbnail((440,245)); sheet.paste(image,(col*450,row*275+25)); draw.text((col*450+4,row*275+5),f'ID {ids[i]} {name} point {z:.2f}m, depth {depth:.2f}m',fill='black')
 records.append({'point_id':int(ids[i]),'position_m':pts[i].tolist(),'free_views':int(free[i]),'support_views':int(support[i]),'shown_conflicting_views':matches[:2]})
sheet.save(out/'candidate_source_sheet.jpg',quality=92); (out/'candidate_examples.json').write_text(json.dumps(records,indent=2),encoding='utf8'); print(json.dumps(records,indent=2))
