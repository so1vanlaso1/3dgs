"""Make reproducible visual depth probes; no automatic correspondence claims."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/room_improve_physical_edges_v02_20261103'
RAW = ROOT / 'results/room_improve_target_raw_20261103'
PROJ = ROOT / 'results/room_improve_target_projections_20261103'
DATA = ROOT / 'data/realsense_20261030_134523_clean'

# Boxes are chosen after viewing RGB: they sample interiors, not ground-truth boundaries.
ITEMS = [
    dict(id='box_reference_6487', rgb=RAW/'rgb_6487.png', depth=RAW/'rgb_6487.depth.npz',
         crop=(460,350,550,455), a=(475,390,488,420), b=(511,390,524,420),
         a_label='left: wall or column', b_label='right: box face'),
    dict(id='box_early_6253', rgb=RAW/'rgb_6253.png', depth=RAW/'rgb_6253.depth.npz',
         crop=(520,350,610,455), a=(530,390,543,420), b=(570,390,583,420),
         a_label='left: wall or column', b_label='right: box face'),
    dict(id='kiosk_000000', rgb=DATA/'images/frame_000000.png', depth=PROJ/'frame_000000.png.dense_depth.npz',
         crop=(180,350,330,463), a=(185,400,202,430), b=(232,400,252,430),
         a_label='left: wall', b_label='right: kiosk pedestal'),
    dict(id='statue_000276', rgb=DATA/'images/frame_000276.png', depth=PROJ/'frame_000276.png.dense_depth.npz',
         crop=(525,180,665,315), a=(585,225,605,245), b=(642,225,660,245),
         a_label='left: statue head', b_label='right: backdrop'),
]

def stats(z, rect):
    x0,y0,x1,y1=rect; sub=z[y0:y1,x0:x1]; vals=sub[sub>0]
    return dict(rect_xyxy=rect, valid=int(len(vals)), total=int(sub.size),
                median_m=round(float(np.median(vals)),4) if len(vals) else None,
                p10_p90_m=[round(float(v),4) for v in np.percentile(vals,[10,90])] if len(vals) else None)

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    results=[]
    for item in ITEMS:
        rgb=np.asarray(Image.open(item['rgb']).convert('RGB'))
        with np.load(item['depth']) as f: z=f['depth_m']
        assert z.shape==rgb.shape[:2] and np.isfinite(z).all()
        x0,y0,x1,y1=item['crop']; sub=z[y0:y1,x0:x1]
        color=cv2.applyColorMap((np.clip((sub-.3)/5.7,0,1)*255).astype('uint8'),cv2.COLORMAP_TURBO)[:,:,::-1]
        color[sub<=0]=0
        scale=5; width=(x1-x0)*scale; height=(y1-y0)*scale
        panel=Image.new('RGB',(width*2, height+70),'white'); draw=ImageDraw.Draw(panel)
        panel.paste(Image.fromarray(rgb[y0:y1,x0:x1]).resize((width,height),Image.Resampling.NEAREST),(0,35))
        panel.paste(Image.fromarray(color).resize((width,height),Image.Resampling.NEAREST),(width,35))
        draw.text((4,8),'RGB: colored boxes are interior probes',fill='black')
        draw.text((width+4,8),'Depth: black = missing; colors = distance',fill='black')
        for key,col in [('a','cyan'),('b','yellow')]:
            ax,ay,bx,by=item[key]; r=((ax-x0)*scale,35+(ay-y0)*scale,(bx-x0)*scale,35+(by-y0)*scale)
            draw.rectangle(r,outline=col,width=3); draw.rectangle((r[0]+width,r[1],r[2]+width,r[3]),outline=col,width=3)
        if item['id'].startswith('box_'):
            for sy in (400,420):
                py=35+(sy-y0)*scale
                draw.line((0,py,width*2,py),fill='magenta',width=2)
        draw.text((4,height+45),'Cyan: '+item['a_label']+'; yellow: '+item['b_label'],fill='black')
        panel.save(OUT/(item['id']+'.png'))
        results.append(dict(id=item['id'],rgb=str(item['rgb'].relative_to(ROOT)),depth=str(item['depth'].relative_to(ROOT)),
            crop_xyxy=item['crop'],probes=[dict(label=item[k+'_label'],**stats(z,item[k])) for k in ('a','b')]))
    manifest=json.loads((DATA/'capture_manifest.json').read_text()); frames={f['name']:f for f in manifest['frames']}
    first=manifest['frames'][0]['timestamp_ns']
    reserved=[]
    for name in ['frame_000004.png','frame_000145.png','frame_000295.png']:
        f=frames[name]; reserved.append(dict(name=name,rgb_message_id=f['message_id'],depth_message_id=f['depth_message_id'],
            rgb_timestamp_ns=f['timestamp_ns'],depth_timestamp_ns=f['depth_timestamp_ns'],replay_seconds=(f['timestamp_ns']-first)/1e9))
    (OUT/'probes_and_reservations.json').write_text(json.dumps(dict(items=results,reserved_frames=reserved,
      note='Interior regions selected after visual inspection; medians are projected camera-Z, not proof of surface identity. Reserved frame identities only, not an evaluated validation set.'),indent=2),encoding='utf-8')
    profiles=[]
    for mid,lo,hi in [(6487,480,530),(6253,540,590)]:
        with np.load(RAW/f'rgb_{mid}.depth.npz') as f: z=f['depth_m']
        for y in (400,420):
            for x in range(lo,hi):
                val=float(z[y,x]); profiles.append(dict(rgb_message_id=mid,y=y,x=x,depth_m=round(val,6) if val>0 else '',valid=bool(val>0)))
    with (OUT/'box_depth_profiles.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=profiles[0]); w.writeheader(); w.writerows(profiles)
    print(json.dumps(dict(items=results,reserved_frames=reserved),indent=2))

if __name__=='__main__': main()
