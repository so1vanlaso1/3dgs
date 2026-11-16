"""Project two point clouds into fixed baseline camera views for visual review."""
import json
from pathlib import Path
import cv2
import numpy as np
import pycolmap
from PIL import Image,ImageDraw
from plyfile import PlyData

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/realsense_20261030_134523_clean';OUT=ROOT/'results/room_improve_cloud_fixed_views_v02_20261103'
P=ROOT/'results/room_improve_full_source_cloud_20261103/development_only_unfiltered.ply'

def render(points,color,T,K,w,h):
    q=points@T[:,:3].T+T[:,3];z=q[:,2];valid=(z>.3)&(z<6)&np.isfinite(q).all(axis=1);q=q[valid];z=z[valid];col=color[valid]
    u=np.rint(K[0,0]*q[:,0]/z+K[0,2]).astype(int);v=np.rint(K[1,1]*q[:,1]/z+K[1,2]).astype(int);valid=(u>=0)&(u<w)&(v>=0)&(v<h);u=u[valid];v=v[valid];z=z[valid];col=col[valid]
    key=v*w+u;order=np.lexsort((z,key));_,idx=np.unique(key[order],return_index=True);ii=order[idx]
    img=np.zeros((h,w,3),np.uint8);img[v[ii],u[ii]]=col[ii]
    return img,len(ii)

def main():
    OUT.mkdir(parents=True,exist_ok=False);m=pycolmap.Reconstruction(DATA/'rgbd/dataset/sparse');images={i.name:i for i in m.images.values()};cam=next(iter(m.cameras.values()));fx,fy,cx,cy=cam.params;K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    original=np.array([p.xyz for p in m.points3D.values()]);oldcol=np.array([p.color for p in m.points3D.values()],np.uint8);v=PlyData.read(P)['vertex'].data;new=np.column_stack([v[k] for k in ('x','y','z')]);newcol=np.column_stack([v[k] for k in ('red','green','blue')])
    views=json.loads((ROOT/'results/room_improve_baseline_inspection_20261103/inspection_poses.json').read_text())['views'];chosen=[v for v in views if v['requested_seconds'] in (43,61,68,110)]
    report=[];panels=[]
    for v in chosen:
        name=v['image_name'];T=images[name].cam_from_world().matrix();im=Image.open(DATA/'images'/name).convert('RGB');old,no=render(original,oldcol,T,K,cam.width,cam.height);newim,nn=render(new,newcol,T,K,cam.width,cam.height)
        panel=Image.new('RGB',(3*cam.width,cam.height+30),'white');draw=ImageDraw.Draw(panel)
        oldvis=cv2.dilate(old,np.ones((3,3),np.uint8));newvis=cv2.dilate(newim,np.ones((3,3),np.uint8))
        for i,(img,title) in enumerate([(im,'Prepared RGB'),(Image.fromarray(oldvis),'Original cloud (dilated points)'),(Image.fromarray(newvis),'Experimental cloud (dilated points)')]):
            panel.paste(img,(i*cam.width,30));draw.text((i*cam.width+4,7),title,fill='black')
        fn=f"{v['requested_seconds']:03d}s.png";panel.save(OUT/fn);panels.append(panel.resize((1251,247)))
        report.append(dict(requested_seconds=v['requested_seconds'],image_name=name,original_occupied_pixels=no,experimental_occupied_pixels=nn,image=fn))
    sheet=Image.new('RGB',(1251,4*247),'white')
    for i,p in enumerate(panels):sheet.paste(p,(0,i*247))
    sheet.save(OUT/'contact_sheet.jpg',quality=92)
    (OUT/'report.json').write_text(json.dumps(dict(views=report,method='Original and experimental points in original metric COLMAP coordinates; nearest Z per rounded pixel. Display applies 3x3 channelwise dilation for visibility; occupied-pixel counts are computed before dilation.',
      caveat='Display dilation may merge nearby points and alter colors, and does not represent measured surfaces. Occupied pixels are sampling/projection density, not accuracy or scene completeness. Views were selected before this comparison but have been inspected during research. Pose and calibration are shared and unvalidated; these are captured views, not novel-view correctness.'),indent=2),encoding='utf-8')
    print('saved',len(report),'fixed view comparisons')
if __name__=='__main__':main()
