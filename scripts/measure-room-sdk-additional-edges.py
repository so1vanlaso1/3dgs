"""Exploratory alternate-remap checks on additional exhibit boundaries."""
import csv
import importlib.util
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'results/room_improve_sdk_remap_comparison_20261103';DEP=ROOT/'results/room_improve_sdk_candidate_depth_20261103';OUT=ROOT/'results/room_improve_sdk_extra_edges_20261103'
s=importlib.util.spec_from_file_location('transition',ROOT/'scripts/audit-room-transition-profile.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
CASES=[('post',191,440,[330,340,350,360,370],'development'),('post',195,420,[330,340,350,360,370],'check'),
       ('figure',239,180,[120,130,140,150,160],'development'),('figure',243,225,[120,140,160,180,200],'check')]
def main():
    OUT.mkdir(parents=True,exist_ok=False);rows=[];summary=[]
    for label,num,x,ys,group in CASES:
        name=f'frame_{num:06}.png';lo=x-30;hi=x+30;left=(lo,lo+10);right=(hi-10,hi);search=(lo+10,hi-10);y0=min(ys)-10;y1=max(ys)+10
        with np.load(DEP/(name+'.dense_depth.npz')) as f:z=f['depth_m']
        panel=Image.new('RGB',((hi-lo)*6,(y1-y0)*3+35),'white');draw=ImageDraw.Draw(panel)
        for j,key in enumerate(('baseline','sdk_consistent')):
            rgb=np.asarray(Image.open(SRC/f'{group}_{name}.{key}.png').convert('RGB'));gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);panel.paste(Image.fromarray(rgb[y0:y1,lo:hi]).resize(((hi-lo)*3,(y1-y0)*3),Image.Resampling.NEAREST),(j*(hi-lo)*3,35));draw.text((j*(hi-lo)*3+4,7),key,fill='black')
            for y in ys:
                q=m.profile(z,gray,y,left,right,search);rows.append(dict(object=label,frame=name,group=group,mapping=key,**q))
                if q['rgb_x'] is not None:
                    yy=35+(y-y0)*3;rx=j*(hi-lo)*3+(q['rgb_x']-lo)*3;draw.ellipse((rx-3,yy-3,rx+3,yy+3),fill='cyan')
                    for bx in (q['band_start'],q['band_end']):
                        xx=j*(hi-lo)*3+(bx-lo)*3;draw.ellipse((xx-3,yy-3,xx+3,yy+3),fill='magenta')
        panel.save(OUT/f'{label}_{num:06}.png')
        for key in ('baseline','sdk_consistent'):
            q=[r for r in rows if r['object']==label and r['frame']==name and r['mapping']==key];good=[r for r in q if r['rgb_distance_to_band_px'] is not None]
            summary.append(dict(object=label,frame=name,group=group,mapping=key,band_count=len(good),median_band_distance=float(np.median([r['rgb_distance_to_band_px'] for r in good])) if good else None,holes=sum(r['holes_in_band']>0 for r in good)))
    with (OUT/'rows.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (OUT/'report.json').write_text(json.dumps(dict(summary=summary,warning='ROIs selected after inspecting both mappings; independent strongest RGB gradient may not identify the same surface as depth. Nearby views correlated; missing or multiple surfaces and holes mean rows are exploratory only.'),indent=2),encoding='utf-8')
    for r in summary:print(r)
if __name__=='__main__':main()
