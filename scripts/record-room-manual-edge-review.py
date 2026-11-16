"""Record reviewed edge candidates and deliberately rejected post transitions."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/realsense_20261030_134523_clean/images'
DEPTH=ROOT/'results/room_improve_alignment_motion_20261103'
OUT=ROOT/'results/room_improve_manual_edges_20261103'

# frame, feature, coordinate axis of displacement, fixed coordinate, RGB boundary,
# depth jump left/top sample, assessment. Selected after visual review; examples
# and same-scene validation are not independent ground truth.
ITEMS=[('197','statue head top','y',470,193,193,'candidate'),
       ('197','statue head top','y',475,190,189,'candidate'),
       ('200','statue head top','y',350,165,162,'candidate'),
       ('200','statue head top','y',355,160,160,'candidate'),
       ('203','statue head top','y',350,147,148,'candidate_same_scene_validation'),
       ('204','right railing post left edge','x',305,637,631,'rejected_broad_ambiguous_transition'),
       ('204','right railing post left edge','x',310,635,628,'rejected_broad_ambiguous_transition'),
       ('204','right railing post left edge','x',315,630,627,'rejected_broad_ambiguous_transition'),
       ('197','seated statue head top','y',460,271,271,'rejected_multiple_depth_jumps'),
       ('200','seated statue head top','y',350,273,269,'rejected_multiple_depth_jumps'),
       ('204','seated statue head top','y',380,296,286,'rejected_unclear_correspondence')]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,(num,feature,axis,fixed,rgbcoord,depthcoord,status) in enumerate(ITEMS):
        name=f'frame_000{num}.png'
        rgb=np.asarray(Image.open(DATA/name).convert('RGB'))
        with np.load(DEPTH/(name+'.dense_depth.npz')) as z: depth=z['depth_m']
        if axis=='x': rx,ry=rgbcoord,fixed; dx,dy=depthcoord,fixed; a,b=depth[dy,dx],depth[dy,dx+1]
        else: rx,ry=fixed,rgbcoord; dx,dy=fixed,depthcoord; a,b=depth[dy,dx],depth[dy+1,dx]
        assert a>0 and b>0 and abs(a-b)>max(.06,.025*min(a,b))
        # Crops display the actual manual RGB and depth points, not an image-derived fit.
        cx,cy=rx,ry; x0=max(0,cx-25); x1=min(rgb.shape[1],cx+26); y0=max(0,cy-25); y1=min(rgb.shape[0],cy+26)
        tile=Image.fromarray(rgb[y0:y1,x0:x1]).resize(((x1-x0)*8,(y1-y0)*8))
        d=ImageDraw.Draw(tile)
        for x,y,color in [(rx,ry,'cyan'),(dx,dy,'magenta')]:
            x=(x-x0)*8+4;y=(y-y0)*8+4;d.ellipse((x-5,y-5,x+5,y+5),fill=color)
        fn=f'{i+1:02d}_{num}_{axis}.png';tile.save(OUT/fn)
        rows.append(dict(frame=name,feature=feature,axis=axis,fixed_coordinate=fixed,
            rgb_coordinate=rgbcoord,depth_jump_start_coordinate=depthcoord,
            signed_depth_start_minus_rgb_px=depthcoord-rgbcoord,
            depth_start_m=float(a),depth_next_m=float(b),assessment=status,marker_file=fn))
    with (OUT/'reviewed_candidates.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)
    (OUT/'reviewed_candidates.json').write_text(json.dumps({'entries':rows,
      'method':'Visual selection assisted by RGB Sobel gradient and adjacent-depth-jump inspection. Cyan marks selected RGB pixel; magenta marks the first depth pixel. Depth jump boundary lies between pixels.',
      'warning':'Same-scene examples, not independent ground truth; post and seated-statue transitions are rejected, not used for correction. Standing head-top readings are exploratory candidates.'},indent=2),encoding='utf-8')
    for r in rows: print(r['frame'],r['feature'],r['axis'],r['fixed_coordinate'],r['signed_depth_start_minus_rgb_px'],r['assessment'])

if __name__=='__main__': main()
