"""Render small, repeatable novel-view paths around recorded room anchors."""
import hashlib
import json
import math
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch

os.environ.setdefault('NUM_CHANNELS','3,4')
from gsplat.rendering import rasterization

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/room_improve_baseline_inspection_20261103'
OUT=ROOT/'results/room_improve_free_path_v02_20261103'
CKPT=ROOT/'results/realsense_20261030_134523_120k/train/ckpts/final_post_update.pt'
ANCHORS=[(43,'wide room'),(61,'difficult doorway'),(68,'exhibit'),(110,'reflective case')]
STEPS=[-1.,-.5,0.,.5,1.]

@torch.no_grad()
def main():
    OUT.mkdir(parents=True,exist_ok=False);doc=json.loads((BASE/'inspection_poses.json').read_text());views={v['requested_seconds']:v for v in doc['views']}
    s={k:v.cuda() for k,v in torch.load(CKPT,map_location='cpu',weights_only=True)['splats'].items()};colors=torch.cat((s['sh0'],s['shN']),1)
    degree=int(math.sqrt(colors.shape[1])-1);frames=[]
    for sec,label in ANCHORS:
        v=views[sec];anchor=np.asarray(v['camera_to_world_normalized']);K=np.asarray(v['intrinsics']);w,h=v['width'],v['height']
        for n,u in enumerate(STEPS):
            # Camera-local lateral motion and yaw; distinct non-recorded poses except u=0.
            angle=math.radians(2)*u;c=math.cos(angle);z=math.sin(angle)
            rot=np.array([[c,0,z],[0,1,0],[-z,0,c]])
            pose=anchor.copy();pose[:3,3]=anchor[:3,3]+anchor[:3,:3]@np.array([.02*u,0,0]);pose[:3,:3]=anchor[:3,:3]@rot
            rgb,_,_=rasterization(s['means'],s['quats'],torch.exp(s['scales']),torch.sigmoid(s['opacities']),colors,
                torch.from_numpy(np.linalg.inv(pose)[None]).float().cuda(),torch.from_numpy(K[None]).float().cuda(),w,h,sh_degree=degree,packed=True)
            im=rgb[0].clamp(0,1).cpu().numpy();assert np.isfinite(im).all()
            fn=f'{sec:03d}s_{n:02d}.png';Image.fromarray((im*255).astype('uint8')).save(OUT/fn)
            frames.append(dict(anchor_seconds=sec,label=label,index=n,parameter=u,novel_view=bool(u),camera_to_world_normalized=pose.tolist(),
              intrinsics=K.tolist(),width=w,height=h,vertical_fov_radians=v['vertical_fov_radians'],aspect=v['aspect'],file=fn))
            print(sec,n,flush=True)
    sheet=Image.new('RGB',(5*424,4*265),'white');d=ImageDraw.Draw(sheet)
    for i,v in enumerate(frames):
        x=(i%5)*424;y=(i//5)*265;d.text((x+4,y+7),f"{v['label']}  u={v['parameter']:+.1f}",fill='black')
        sheet.paste(Image.open(OUT/v['file']).resize((424,235)),(x,y+30))
    sheet.save(OUT/'contact_sheet.jpg',quality=94)
    vid=cv2.VideoWriter(str(OUT/'inspection_segments.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),2,(w,h));assert vid.isOpened()
    for v in frames:vid.write(cv2.imread(str(OUT/v['file'])))
    vid.release()
    manifest=dict(checkpoint=str(CKPT),checkpoint_sha256=hashlib.sha256(CKPT.read_bytes()).hexdigest(),
      render_settings=dict(sh_degree=degree,packed=True,background='black',num_channels=os.environ['NUM_CHANNELS'],fps=2),
      path_description='Four independent local lateral/yaw sweeps anchored at recorded poses; cuts between regions. Not a collision-checked continuous walk. 0.02 trainer-normalized units and 2 degrees at each sweep end.',
      coordinate_system='trainer normalized camera-to-world; align candidate world frames before comparison',frames=frames)
    (OUT/'path.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
if __name__=='__main__':main()
