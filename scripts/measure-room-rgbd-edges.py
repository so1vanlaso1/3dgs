"""Exploratory signed edge offsets on selected silhouette strips; no fitting."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/realsense_20261030_134523_clean'
INPUT=ROOT/'results/room_improve_alignment_motion_20261103'
OUT=ROOT/'results/room_improve_edge_measurements_v02_20261103'
# Hand-selected bounding strips around the left edge of the standing exhibit's head.
# (frame, role, x_min, y_min, x_max, y_max). Upper bounds exclusive.
REGIONS=[('head','197','development',445,186,477,211),('head','200','development',320,158,355,183),
         ('head','204','development',307,154,340,189),('head','198','validation',385,156,420,183),
         ('head','203','validation',310,144,344,175),
         ('backdrop_right','197','development',535,85,565,140),
         ('backdrop_right','200','development',411,75,445,130),
         ('backdrop_right','204','development',385,75,415,130),
         ('backdrop_right','198','validation',480,75,515,130),
         ('backdrop_right','203','validation',405,75,435,130)]

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    meta={x['image_name']:x for x in json.loads((INPUT/'report.json').read_text())['views']}
    rows=[]; summaries=[]
    for feature,num,role,x0,y0,x1,y1 in REGIONS:
        name=f'frame_000{num}.png'; rgb=np.asarray(Image.open(DATA/'images'/name).convert('RGB'))
        with np.load(INPUT/(name+'.dense_depth.npz')) as p: z=p['depth_m']
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        # Within each manually defined strip, pick largest horizontal RGB gradient
        # and independently the largest adjacent valid depth jump on each row.
        gx=np.abs(cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3))
        crop=Image.fromarray(rgb[y0-10:y1+10,x0-10:x1+10]).resize(((x1-x0+20)*6,(y1-y0+20)*6))
        draw=ImageDraw.Draw(crop); chosen=[]
        for y in range(y0,y1):
            gs=gx[y,x0:x1]; rgbx=x0+int(np.argmax(gs)); strength=float(np.max(gs))
            a=z[y,x0:x1-1];b=z[y,x0+1:x1]; jumps=np.abs(a-b)
            valid=(a>0)&(b>0)&(jumps>np.maximum(.06,.025*np.minimum(a,b)))
            if strength<60 or not valid.any():
                rows.append(dict(feature=feature,frame=name,role=role,row_y=y,rgb_x=rgbx,rgb_gradient=strength,depth_x='',signed_depth_minus_rgb_px='',status='missing depth jump' if not valid.any() else 'weak RGB edge'))
                continue
            ds=np.where(valid,jumps,-1); dx=x0+int(np.argmax(ds)); displacement=dx-rgbx
            rows.append(dict(feature=feature,frame=name,role=role,row_y=y,rgb_x=rgbx,rgb_gradient=strength,depth_x=dx,signed_depth_minus_rgb_px=displacement,status='candidate'))
            chosen.append(displacement)
            yy=(y-(y0-10))*6+3; xr=(rgbx-(x0-10))*6+3; xd=(dx-(x0-10))*6+3
            draw.ellipse((xr-3,yy-3,xr+3,yy+3),fill='cyan')
            draw.ellipse((xd-3,yy-3,xd+3,yy+3),fill='magenta')
        crop.save(OUT/f'{feature}_{num}_{role}_markers.png')
        m=meta[name]
        summaries.append(dict(feature=feature,frame=name,role=role,box_xyxy=[x0,y0,x1,y1],replay_seconds=m['replay_seconds'],gyro_rad_s=m['angular_rate_rad_s'],
            row_count=y1-y0,candidate_rows=len(chosen),missing_or_weak_rows=y1-y0-len(chosen),
            signed_depth_minus_rgb_median_px=float(np.median(chosen)) if chosen else None,
            signed_depth_minus_rgb_p10_p90_px=np.percentile(chosen,[10,90]).tolist() if chosen else None,
            candidate_offsets_px=chosen,marker_file=f'{feature}_{num}_{role}_markers.png'))
    with (OUT/'scanline_measurements.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'report.json').write_text(json.dumps(dict(regions=summaries,
       method='Exploratory independent strongest RGB Sobel gradient and strongest valid adjacent depth jump per horizontal scanline within manually specified strips; sign is depth x minus RGB x.',
       limitations='Not manual point annotation or ground truth. RGB texture or multiple surfaces can create false matches; rows are correlated. Depth points are rounded before comparison. Validation frames share scene and trajectory.'),indent=2),encoding='utf-8')
    for s in summaries: print(s['feature'],s['frame'],s['role'],s['candidate_rows'],s['row_count'],s['signed_depth_minus_rgb_median_px'],s['signed_depth_minus_rgb_p10_p90_px'])

if __name__=='__main__': main()
