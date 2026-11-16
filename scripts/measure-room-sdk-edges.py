"""Compare same provisional depth transition against two RGB remaps."""
import csv
import importlib.util
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/room_improve_sdk_edge_measurements_20261103'
s=importlib.util.spec_from_file_location('transition',ROOT/'scripts/audit-room-transition-profile.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
FIRST=ROOT/'results/room_improve_sdk_remap_comparison_20261103';EXTRA=ROOT/'results/room_improve_sdk_remap_extra_20261103'
DEP1=ROOT/'results/room_improve_sdk_candidate_depth_20261103';DEP2=ROOT/'results/room_improve_sdk_candidate_extra_depth_20261103'
# Strips chosen after viewing both variants; check frames were picked retrospectively.
CASES=[('kiosk',1,197,EXTRA,DEP2,'nearby_check'),('kiosk',2,182,EXTRA,DEP2,'nearby_check'),('kiosk',3,140,FIRST,DEP1,'development'),
       ('box',25,420,EXTRA,DEP2,'nearby_check'),('box',26,294,EXTRA,DEP2,'nearby_check'),('box',27,145,FIRST,DEP1,'development')]
YS=[385,395,405,415,425,435]
def main():
    OUT.mkdir(parents=True,exist_ok=False);records=[];summaries=[]
    for objectname,num,x,images,depth,group in CASES:
        name=f'frame_{num:06}.png';lo=x-28;hi=x+28;left=(lo,lo+10);right=(hi-10,hi);search=(lo+10,hi-10)
        with np.load(depth/(name+'.dense_depth.npz')) as f:z=f['depth_m']
        inputs={k:np.asarray(Image.open(images/f'{group}_{name}.{k}.png').convert('RGB')) for k in ('baseline','sdk_consistent')}
        panel=Image.new('RGB',((hi-lo)*6,(463-370)*3+35),'white');draw=ImageDraw.Draw(panel)
        for j,key in enumerate(('baseline','sdk_consistent')):
            rgb=inputs[key];gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);panel.paste(Image.fromarray(rgb[370:463,lo:hi]).resize(((hi-lo)*3,(463-370)*3),Image.Resampling.NEAREST),(j*(hi-lo)*3,35));draw.text((j*(hi-lo)*3+4,8),key,fill='black')
            for y in YS:
                r=m.profile(z,gray,y,left,right,search);records.append(dict(object=objectname,frame=name,group=group,mapping=key,**r))
                if r['rgb_x'] is not None:
                    yy=35+(y-370)*3;rx=j*(hi-lo)*3+(r['rgb_x']-lo)*3;draw.ellipse((rx-3,yy-3,rx+3,yy+3),fill='cyan')
                    for bx in (r['band_start'],r['band_end']):
                        bx=j*(hi-lo)*3+(bx-lo)*3;draw.ellipse((bx-3,yy-3,bx+3,yy+3),fill='magenta')
        panel.save(OUT/f'{objectname}_{num:06}.png')
        for key in ('baseline','sdk_consistent'):
            rr=[r for r in records if r['frame']==name and r['mapping']==key];values=[r['rgb_distance_to_band_px'] for r in rr if r['rgb_distance_to_band_px'] is not None]
            summaries.append(dict(object=objectname,frame=name,group=group,mapping=key,usable_bands=len(values),median_rgb_distance_to_band_px=float(np.median(values)) if values else None,holes_in_band=sum(r['holes_in_band']>0 for r in rr if r['holes_in_band'] is not None)))
    with (OUT/'rows.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    (OUT/'report.json').write_text(json.dumps(dict(summary=summaries,cases=[dict(object=o,frame=n,center=x,group=g) for o,n,x,_,__,g in CASES],
      caveat='ROI chosen after inspecting both remaps; sample rows and nearby frames correlated. Strongest RGB edge and 10–90% depth transition may correspond to different physical surfaces. Holes not interpolated. Distance to band is a lower-bound proxy, not calibration ground truth. Nearby-check selection retrospective, not blind independent validation.'),indent=2),encoding='utf-8')
    for r in summaries:print(r['object'],r['frame'],r['mapping'],r['usable_bands'],r['median_rgb_distance_to_band_px'],'holes',r['holes_in_band'])
if __name__=='__main__':main()
