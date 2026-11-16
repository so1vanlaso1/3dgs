"""Visual spot checks of high-conflict points, without deleting them."""
import json
from pathlib import Path
import cv2
import numpy as np
import pycolmap
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/realsense_20261030_134523_clean';SRC=ROOT/'results/room_improve_full_source_cloud_20261103';DEP=ROOT/'results/room_improve_full_cloud_dense_checks_20261103';OUT=ROOT/'results/room_improve_dense_conflict_examples_20261103'
def select(mask,pts,n):
    out=[]
    for i in np.flatnonzero(mask):
        if not out or np.min(np.linalg.norm(pts[out]-pts[i],axis=1))>.6:out.append(int(i))
        if len(out)==n:return out
    raise ValueError('not enough spatially separate examples')
def main():
    OUT.mkdir(parents=True,exist_ok=False);report=json.loads((SRC/'report.json').read_text());manifest=json.loads((DATA/'capture_manifest.json').read_text());mf={r['name']:r for r in manifest['frames']}
    with np.load(SRC/'evidence.npz') as f:points=f['points_m'];sources=f['source_group_index']
    with np.load(SRC/'dense_check_evidence.npz') as f:c=f['stable_3x3']
    selected=select((c[:,2]>=3)&(c[:,1]==0),points,4)+select((c[:,2]>=3)&(c[:,1]>0),points,4)
    model=pycolmap.Reconstruction(DATA/'rgbd/dataset/sparse');ims={im.name:im for im in model.images.values()};cam=next(iter(model.cameras.values()));fx,fy,cx,cy=cam.params
    def project(name,idx):
        q=ims[name].cam_from_world().matrix();p=points[idx]@q[:,:3].T+q[:,3];return np.array([fx*p[0]/p[2]+cx,fy*p[1]/p[2]+cy]),p[2]
    observations={i:[] for i in selected}
    for name in report['check_images']:
        with np.load(DEP/(name+'.dense_depth.npz')) as f:z=f['depth_m']
        low=cv2.erode(z,np.ones((3,3),np.uint8));high=cv2.dilate(z,np.ones((3,3),np.uint8));stable=(low>.3)&(high<6)&((high-low)<np.maximum(.03,z*.02))
        for i in selected:
            source=report['development_images'][int(sources[i])]
            if abs(mf[source]['timestamp_ns']-mf[name]['timestamp_ns'])<=5e9:continue
            uv,pred=project(name,i);x,y=np.rint(uv).astype(int)
            if pred>.3 and pred<6 and 0<=x<cam.width and 0<=y<cam.height and stable[y,x]:
                observations[i].append(dict(name=name,x=float(uv[0]),y=float(uv[1]),difference_m=float(pred-z[y,x])))
    rows=[];tiles=[]
    for i in selected:
        src=report['development_images'][int(sources[i])];uv,_=project(src,i);free=[a for a in observations[i] if a['difference_m']<-.15];sup=[a for a in observations[i] if abs(a['difference_m'])<=.08];assert len(free)>=3
        choices=[dict(name=src,x=float(uv[0]),y=float(uv[1]),caption='source'),dict(**min(free,key=lambda a:a['difference_m']),caption='check conflict')]
        if sup:choices.append(dict(**min(sup,key=lambda a:abs(a['difference_m'])),caption='check support'))
        tile=Image.new('RGB',(3*834,495),'white');draw=ImageDraw.Draw(tile)
        for k,v in enumerate(choices):
            im=Image.open(DATA/'images'/v['name']).convert('RGB');tile.paste(im,(k*834,30));x=k*834+v['x'];y=30+v['y'];draw.ellipse((x-10,y-10,x+10,y+10),outline='red',width=3)
            draw.text((k*834+4,7),f"{v['caption']} {v['name']} delta={v.get('difference_m',0):+.3f}m",fill='black')
        name=f'point_{i}.jpg';tile.save(OUT/name,quality=90);tiles.append(tile.resize((834,165)));rows.append(dict(point_index=i,source=src,views=choices,image=name))
    sheet=Image.new('RGB',(2*834,4*165),'white')
    for j,tile in enumerate(tiles):sheet.paste(tile,((j%2)*834,(j//2)*165))
    sheet.save(OUT/'contact_sheet.jpg',quality=93);(OUT/'report.json').write_text(json.dumps(dict(examples=rows,caveat='Selected spatially separated examples, most negative check residual and closest support. Not representative; points project to images, and aligned depth/pose/occlusion remain uncertain.'),indent=2),encoding='utf-8');print('saved',len(rows),'examples')
if __name__=='__main__':main()
