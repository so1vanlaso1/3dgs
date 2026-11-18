"""Unfilled-depth and remap review, with provenance and no automatic edge truth.

The frozen ROIs reuse previously recorded structure candidates; no calibration
parameters are fitted. Final reserved blocks are forbidden in the visual review.
"""
import csv
import importlib.util
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('r1',ROOT/'scripts/run-db3-r0-r1.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
OUT=m.RUN/'alignment_review'
CASES=[(3,'kiosk',(95,365,185,455)),(4,'kiosk check',(10,365,160,455)),
       (5,'kiosk check',(0,350,130,455)),(27,'opaque box',(105,365,190,455)),
       (29,'opaque box',(0,365,180,455)),(145,'box/wall prospective check',(0,300,400,463)),
       (197,'head top',(430,160,505,220)),(200,'head top',(325,135,385,195)),
       (203,'head top',(320,120,390,180)),(295,'exhibit prospective check',(300,150,780,450)),
       (131,'wide doorway',(240,80,420,423)),(183,'distant doorway',(400,80,700,400)),
       (204,'thin rails',(570,240,700,420)),(329,'reflective glass',(30,50,440,430)),
       (443,'case/column',(400,60,800,450)),(471,'end of route',(0,0,834,463))]


def main():
    OUT.mkdir(exist_ok=False)
    m.write(OUT/'protocol.json',dict(cases=[dict(frame=f'frame_{n:06d}.png',region=l,roi_xyxy=b) for n,l,b in CASES],
        changed_factor='RGB projection only; same K/crop/source pairs and raw unfiltered Z-buffer projection',
        numerical_tolerance_raw_px=.1,
        physical_review='Annotate RGB-only and depth-only separately; repeat RGB reads before combining. Freeze tolerance at max(2 px, repeat annotation spread plus 1 px); broad transitions, holes, unclear physical correspondences and rotation-gap estimate >1 px cannot pass a precise boundary gate.',
        scope='Same exposed recording; prospective holds apply to later fitting, not independent external ground truth. No fitted calibration or timing parameters.'))
    cal=m.read(m.DATA/'calibration_clean.json');K=np.array(cal['K']);w,h=cal['width'],cal['height']
    manifest=m.read(m.DATA/'capture_manifest.json');frames={f['name']:f for f in manifest['frames']}
    with (m.RUN/'sensor_timing_pairs.csv').open(newline='',encoding='utf-8') as f:
        timing={int(r['rgb_message_id']):r for r in csv.DictReader(f)}
    maps,valid=m.p.remap(cal['raw_color'],K,w,h);assert valid.all()
    oldmaps,_=m.p.remap(cal['raw_color'],K,w,h,m.p.LEGACY)
    mapdelta=np.linalg.norm(maps-oldmaps,axis=2);rows=[]
    with m.connect() as db:
        for num,label,box in CASES:
            name=f'frame_{num:06d}.png';f=frames[name];assert f['prospective_role']!='final_prospective'
            depth,_=m.clean.adapter.decode_depth(db.execute('select data from messages where id=?',(f['depth_message_id'],)).fetchone()[0])
            _,_,_,e=m.p.align_depth(depth,cal['depth_units'],cal['raw_depth'],cal['depth_to_color_R'],cal['depth_to_color_t_m'],K,w,h,filtered=False)
            z=e['depth_m'];np.savez_compressed(OUT/(name+'.raw_depth.npz'),depth_m=z,observed=e['observed'],multiple_projected_samples=e['multiple_projected_samples'])
            old=np.array(Image.open(m.BASE/'images'/name).convert('RGB'));new=np.array(Image.open(m.DATA/'images'/name).convert('RGB'))
            # Continuous valid neighboring samples only, never depth-hole outlines.
            edge=np.zeros((h,w),bool)
            for axis in (0,1):
                a,b=(z[:-1,:],z[1:,:]) if axis==0 else (z[:,:-1],z[:,1:])
                mask=(a>0)&(b>0)&(abs(a-b)>np.maximum(.06,.025*np.minimum(a,b)))
                if axis==0:edge[:-1,:]|=mask
                else:edge[:,:-1]|=mask
            color=cv2.applyColorMap((np.clip((z-.3)/5.7,0,1)*255).astype('u1'),cv2.COLORMAP_TURBO)[:,:,::-1];color[z<=0]=0
            panel=Image.new('RGB',(2*w,2*(h+30)),'white');draw=ImageDraw.Draw(panel)
            for j,(im,title) in enumerate([(old,'Historical RGB'),(new,'Pinned SDK RGB'),(color,'Unfilled camera-Z; black = unknown'),(new.copy(),'SDK RGB + magenta valid depth jumps')]):
                if j==3:im[edge]=[255,0,255]
                x,y=(j%2)*w,(j//2)*(h+30);panel.paste(Image.fromarray(im),(x,y+30));draw.text((x+5,y+7),title,fill='black')
            panel.save(OUT/f'{num:03d}_full.jpg',quality=94)
            x0,y0,x1,y1=box;cw,ch=x1-x0,y1-y0
            scale=min(5, max(1,700//cw));crop=Image.new('RGB',(cw*scale*3,ch*scale+55),'white');d=ImageDraw.Draw(crop)
            for j,(im,title) in enumerate([(old,'Historical RGB'),(new,'Pinned SDK RGB'),(color,'Unfilled depth')]):
                xx=j*cw*scale;crop.paste(Image.fromarray(im[y0:y1,x0:x1]).resize((cw*scale,ch*scale),Image.Resampling.NEAREST),(xx,55));d.text((xx+3,4),title,fill='black')
                d.text((xx+3,20),f'{label}; origin ({x0},{y0}); scale {scale}',fill='black')
                for gx in range((x0//10+1)*10,x1,10):
                    if cw>150 and gx%50:continue
                    x=xx+(gx-x0)*scale;d.line((x,55,x,ch*scale+55),fill='yellow');d.text((x+1,40),str(gx),fill='black')
                for gy in range((y0//10+1)*10,y1,10):
                    if ch>150 and gy%50:continue
                    y=(gy-y0)*scale+55;d.line((xx,y,xx+cw*scale,y),fill='yellow');d.text((xx+1,y+1),str(gy),fill='yellow')
            crop.save(OUT/f'{num:03d}_crop.png')
            t=timing[f['message_id']];sub=z[y0:y1,x0:x1];values=sub[sub>0]
            rows.append(dict(frame=name,region=label,role=f['prospective_role'],roi_xyxy=box,
                rgb_message_id=f['message_id'],depth_message_id=f['depth_message_id'],
                full_projected_valid_fraction=float((z>0).mean()),roi_projected_valid_fraction=float((sub>0).mean()),
                roi_depth_p10_median_p90_m=np.percentile(values,[10,50,90]).tolist() if len(values) else None,
                gyro_rad_s=float(t['nearest_gyro_rad_s']),signed_exposure_midpoint_gap_ms=float(t['signed_exposure_midpoint_depth_minus_rgb_ms']),
                estimated_rotation_during_sensor_gap_px=float(t['estimated_rotation_during_sensor_gap_px']),crop=f'{num:03d}_crop.png'))
    m.write(OUT/'report.json',dict(views=rows,source_map_change_raw_rgb_px_percentiles=np.percentile(mapdelta,[0,50,90,100]).tolist(),
        physical_gate='Requires reviewed annotations; numerical compatibility alone is insufficient',
        no_hole_filling=True,no_calibration_fit=True,final_blocks_visually_inspected=False))
    print('Review evidence generated',len(rows),'views',flush=True)


if __name__=='__main__':main()
