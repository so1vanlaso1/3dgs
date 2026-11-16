"""Record exploratory box-edge scanlines and their supporting pixels."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/room_improve_target_search_20261103'
name='frame_000024.png'

def main():
    rgb=np.asarray(Image.open(ROOT/'data/realsense_20261030_134523_clean/images'/name).convert('RGB'))
    with np.load(ROOT/'results/room_improve_target_projections_20261103'/(name+'.dense_depth.npz')) as p:z=p['depth_m']
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    grad=abs(cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3))
    # Narrow strip chosen visually around the left edge of the black box.
    lo,hi=488,515; scanlines=[360,370,380,390,400,410,420,430,440]
    rows=[]; crop=Image.fromarray(rgb[345:455,480:525]).resize((450,1100)); draw=ImageDraw.Draw(crop)
    for y in scanlines:
        rgbx=lo+int(np.argmax(grad[y,lo:hi])); a=z[y,lo:hi-1];b=z[y,lo+1:hi]; delta=abs(a-b)
        valid=(a>0)&(b>0)&(delta>np.maximum(.06,.025*np.minimum(a,b)))
        indices=np.flatnonzero(valid)
        if len(indices)!=1:
            rows.append(dict(y=y,rgb_x=rgbx,depth_x='',signed_depth_minus_rgb_px='',depth_a_m='',depth_b_m='',status=f'rejected_{len(indices)}_candidate_jumps'))
            continue
        i=int(indices[0]); dx=lo+i
        rows.append(dict(y=y,rgb_x=rgbx,depth_x=dx,signed_depth_minus_rgb_px=dx-rgbx,depth_a_m=float(a[i]),depth_b_m=float(b[i]),status='candidate'))
        # Points sit on same row; magenta drawn above cyan if coincident.
        rx=(rgbx-480)*10+5; xx=(dx-480)*10+5; yy=(y-345)*10+5
        draw.ellipse((rx-5,yy-5,rx+5,yy+5),fill='cyan')
        draw.ellipse((xx-3,yy-3,xx+3,yy+3),fill='magenta')
    crop.save(OUT/'black_box_markers.png')
    with (OUT/'black_box_scanlines.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    values=[r['signed_depth_minus_rgb_px'] for r in rows if r['status']=='candidate']
    result=dict(frame=name,region='black box left edge',strip_x=[lo,hi],rows=scanlines,
        candidate_count=len(values),missing_or_ambiguous=len(scanlines)-len(values),
        median_depth_minus_rgb_px=float(np.median(values)),minimum_px=min(values),maximum_px=max(values),
        convention='depth jump assigned to left pixel of adjacent pair; sign is depth x minus RGB x',
        limitations='Exploratory visually selected region; rows are correlated, not independent; simple jump matching does not establish calibration or timing accuracy.')
    (OUT/'black_box_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__': main()
