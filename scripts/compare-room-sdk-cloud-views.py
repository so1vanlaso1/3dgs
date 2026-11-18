"""Compare baseline and SDK candidates in captured views and aligned world views."""
import json
from pathlib import Path
import cv2
import numpy as np
import pycolmap
from PIL import Image,ImageDraw
from plyfile import PlyData

ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'data/realsense_20261030_134523_clean';NEW=ROOT/'data/room_improve_sdk_rectified_20261103';OUT=ROOT/'results/room_improve_sdk_cloud_views_20261103'
def read_model(path):
    m=pycolmap.Reconstruction(path/'rgbd/dataset/sparse');return m,{i.name:i for i in m.images.values()}
def render(points,color,pose,K,w,h):
    tf=pose.cam_from_world().matrix();q=points@tf[:,:3].T+tf[:,3];z=q[:,2];good=(z>.3)&(z<6);q=q[good];z=z[good];rgb=color[good];u=np.rint(K[0,0]*q[:,0]/z+K[0,2]).astype(int);v=np.rint(K[1,1]*q[:,1]/z+K[1,2]).astype(int);good=(u>=0)&(u<w)&(v>=0)&(v<h);u=u[good];v=v[good];z=z[good];rgb=rgb[good];key=v*w+u;order=np.lexsort((z,key));_,idx=np.unique(key[order],return_index=True);sel=order[idx];out=np.zeros((h,w,3),np.uint8);out[v[sel],u[sel]]=rgb[sel];return cv2.dilate(out,np.ones((3,3),np.uint8)),len(sel)
def main():
    OUT.mkdir(parents=True,exist_ok=False);om,oi=read_model(OLD);nm,ni=read_model(NEW);cam=next(iter(om.cameras.values()));fx,fy,cx,cy=cam.params;K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    old=np.array([p.xyz for p in om.points3D.values()]);oc=np.array([p.color for p in om.points3D.values()],np.uint8);p=PlyData.read(ROOT/'results/room_improve_sdk_source_cloud_20261103/development_only_unfiltered.ply')['vertex'].data;new=np.c_[p['x'],p['y'],p['z']];nc=np.c_[p['red'],p['green'],p['blue']]
    trans=json.loads((ROOT/'results/room_improve_sdk_pose_comparison_20261103/report.json').read_text());R=np.array(trans['rotation']);t=np.array(trans['translation']);new_old=(new-t)@R
    views=json.loads((ROOT/'results/room_improve_baseline_inspection_20261103/inspection_poses.json').read_text())['views'];rows=[];captured=[];fixed=[]
    for v in (x for x in views if x['requested_seconds'] in (43,61,68,110)):
        name=v['image_name'];oldim,oldn=render(old,oc,oi[name],K,cam.width,cam.height);newim,newn=render(new,nc,ni[name],K,cam.width,cam.height);fixedim,fixn=render(new_old,nc,oi[name],K,cam.width,cam.height)
        for mode,imgs in [('captured',[Image.open(OLD/'images'/name).convert('RGB'),Image.fromarray(oldim),Image.open(NEW/'images'/name).convert('RGB'),Image.fromarray(newim)]),('fixed_world',[Image.fromarray(oldim),Image.fromarray(fixedim)])]:
            panel=Image.new('RGB',(len(imgs)*cam.width,cam.height+30),'white');d=ImageDraw.Draw(panel);titles=['baseline RGB','baseline cloud','SDK RGB','SDK cloud, candidate pose'] if mode=='captured' else ['baseline cloud','SDK cloud, rigidly aligned']
            for j,img in enumerate(imgs):panel.paste(img,(j*cam.width,30));d.text((j*cam.width+4,7),titles[j],fill='black')
            panel.save(OUT/f"{mode}_{v['requested_seconds']:03d}s.jpg",quality=94)
            (captured if mode=='captured' else fixed).append(panel.resize((4*300 if mode=='captured' else 2*600,177)))
        rows.append(dict(requested_seconds=v['requested_seconds'],name=name,baseline_occupied_pixels=oldn,candidate_captured_occupied_pixels=newn,candidate_fixed_world_occupied_pixels=fixn))
    for filename,tiles in [('captured_contact_sheet.jpg',captured),('fixed_world_contact_sheet.jpg',fixed)]:
        sheet=Image.new('RGB',(1200,177*len(tiles)),'white')
        for j,img in enumerate(tiles):sheet.paste(img,(0,j*177))
        sheet.save(OUT/filename,quality=94)
    (OUT/'report.json').write_text(json.dumps(dict(views=rows,method='Captured views use each candidate own camera poses; fixed-world view transforms candidate points to baseline via robust camera-center alignment then uses baseline pose. 3x3 channelwise dilation used for visualization only.',
      caveat='Pose/world alignment is approximate; graph trajectories differ non-rigidly and both use same capture. Fixed-world alignment is not ground truth. Point sampling, color remap, pose graph, and initialization differ simultaneously; occupancy is not accuracy or complete surface coverage. No training or acceptance.'),indent=2),encoding='utf-8');print('saved',len(rows),'view pairs')
if __name__=='__main__':main()
