"""Save representative sampled-depth point conflicts for visual review."""
import json
from pathlib import Path
import numpy as np
import pycolmap
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/realsense_20261030_134523_clean';OUT=ROOT/'results/room_improve_multiview_examples_20261103'
def pick(mask,pts,n):
    choices=np.flatnonzero(mask);chosen=[]
    for idx in choices:
        if not chosen or np.min(np.linalg.norm(pts[chosen]-pts[idx],axis=1))>.5:chosen.append(int(idx))
        if len(chosen)==n:break
    return chosen
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    with np.load(ROOT/'results/room_improve_multiview_diagnostic_20261103/evidence.npz') as f:
        ids=f['point_ids'];pts=f['points_m'];flag=f['development_flagged'];checks=f['check']
    selected=pick(flag & (checks[:,1]>0),pts,4)+pick(flag & (checks[:,1]==0)&(checks[:,2]>0),pts,4)
    assert len(selected)==8
    report=json.loads((ROOT/'results/room_improve_multiview_diagnostic_20261103/report.json').read_text())
    model=pycolmap.Reconstruction(DATA/'rgbd/dataset/sparse');images={i.name:i for i in model.images.values()};cam=next(iter(model.cameras.values()));fx,fy,cx,cy=cam.params
    observed={int(i):{'development':[],'check':[]} for i in selected}
    for group in ('development','check'):
        for name in report['views'][group]:
            im=images[name];tf=im.cam_from_world().matrix();q=pts[selected]@tf[:,:3].T+tf[:,3];z=q[:,2]
            uv=np.zeros((len(q),2));valid=(z>.3)&(z<6);uv[valid,0]=fx*q[valid,0]/z[valid]+cx;uv[valid,1]=fy*q[valid,1]/z[valid]+cy
            valid &= (uv[:,0]>=0)&(uv[:,0]<cam.width)&(uv[:,1]>=0)&(uv[:,1]<cam.height)
            with np.load(DATA/'aligned_depth_meters'/(name+'.npz')) as d:xy=d['points'];dep=d['depths']
            distance,ind=cKDTree(xy).query(uv,distance_upper_bound=2,workers=-1)
            for k,idx in enumerate(selected):
                if valid[k] and np.isfinite(distance[k]):
                    observed[idx][group].append(dict(name=name,x=float(uv[k,0]),y=float(uv[k,1]),delta_m=float(z[k]-dep[ind[k]]),pixel_distance=float(distance[k])))
    panels=[];metadata=[]
    for idx in selected:
        dev=[r for r in observed[idx]['development'] if r['delta_m']<-.15]
        check=observed[idx]['check'];support=[r for r in check if abs(r['delta_m'])<=.08];conflict=[r for r in check if r['delta_m']<-.15]
        assert dev and (support or conflict)
        rows=[min(dev,key=lambda r:r['delta_m']),min(support,key=lambda r:abs(r['delta_m'])) if support else min(conflict,key=lambda r:r['delta_m'])]
        tile=Image.new('RGB',(834,2*495),'white');draw=ImageDraw.Draw(tile)
        for k,r in enumerate(rows):
            im=Image.open(DATA/'images'/r['name']).convert('RGB');tile.paste(im,(0,k*495+30));x,y=r['x'],r['y']+k*495+30
            draw.ellipse((x-9,y-9,x+9,y+9),outline='red',width=3)
            draw.text((5,k*495+7),f"point {ids[idx]} {'dev conflict' if k==0 else 'check'} {r['name']} pointZ-depth={r['delta_m']:+.3f}m",fill='black')
        path=f'point_{ids[idx]}.jpg';tile.save(OUT/path,quality=90);panels.append(tile.resize((417,495)))
        metadata.append(dict(point_id=int(ids[idx]),development=rows[0],check=rows[1],image=path))
    sheet=Image.new('RGB',(4*417,2*495),'white')
    for n,p in enumerate(panels):sheet.paste(p,((n%4)*417,(n//4)*495))
    sheet.save(OUT/'contact_sheet.jpg',quality=90)
    (OUT/'examples.json').write_text(json.dumps(dict(examples=metadata,selection='First spatially separated flagged points in each category; each image chosen for strongest development conflict and closest support or strongest conflict in check. Not representative or independent.',
       caveat='Projection is a point, nearest sparse depth within 2 pixels; offsets, occlusion, dynamics, depth edges and pose errors may explain disagreements.'),indent=2),encoding='utf-8')
    print('saved',len(metadata),'examples')
if __name__=='__main__':main()
