"""Exploratory RGB/depth transition bands; records missing samples explicitly."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results/room_improve_target_raw_20261103'
PROJ=ROOT/'results/room_improve_target_projections_20261103'
DATA=ROOT/'data/realsense_20261030_134523_clean/images'
OUT=ROOT/'results/room_improve_transition_profiles_20261103'
# This development protocol and regions were specified before opening reserved frames.
CASES=[
 dict(id='box_reference',rgb=RAW/'rgb_6487.png',depth=RAW/'rgb_6487.depth.npz',left=(475,488),right=(511,524),search=(488,511),ys=[380,390,400,410,420,430]),
 dict(id='box_early',rgb=RAW/'rgb_6253.png',depth=RAW/'rgb_6253.depth.npz',left=(530,543),right=(570,583),search=(543,570),ys=[380,390,400,410,420,430]),
 dict(id='kiosk',rgb=DATA/'frame_000000.png',depth=PROJ/'frame_000000.png.dense_depth.npz',left=(185,202),right=(232,252),search=(202,232),ys=[390,400,410,420,430,440]),
 dict(id='statue',rgb=DATA/'frame_000276.png',depth=PROJ/'frame_000276.png.dense_depth.npz',left=(585,605),right=(642,660),search=(605,642),ys=[225,230,235,240,245,250]),
]

def profile(z,gray,y,left,right,search):
    l=z[y,left[0]:left[1]]; r=z[y,right[0]:right[1]]
    l=l[l>0];r=r[r>0]
    out=dict(y=y,status='rejected',reason='',rgb_x=None,band_start=None,band_end=None,holes_in_band=None,rgb_distance_to_band_px=None,left_z=None,right_z=None)
    if len(l)<5 or len(r)<5: out['reason']='too few valid interior samples'; return out
    zl,zr=float(np.median(l)),float(np.median(r));out.update(left_z=zl,right_z=zr)
    if abs(zr-zl)<.15: out['reason']='interior depth difference below 0.15 m';return out
    lo,hi=search;x=np.arange(lo,hi);vals=z[y,lo:hi];valid=vals>0;p=(vals-zl)/(zr-zl)
    a=x[valid & (p>=.1)]; b=x[valid & (p>=.9)]
    if not len(a) or not len(b): out['reason']='no observed 10% or 90% transition';return out
    start=int(a[0]);end=int(b[b>=start][0]) if np.any(b>=start) else -1
    if end<start: out['reason']='transition ordering failed';return out
    # Do not bridge gaps or pretend these samples identify a precise physical silhouette.
    holes=int(np.sum(z[y,start:end+1]<=0)); grad=np.abs(cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)[y,lo:hi]);rx=lo+int(np.argmax(grad))
    dist=max(start-rx,rx-end,0)
    out.update(rgb_x=rx,band_start=start,band_end=end,holes_in_band=holes,rgb_distance_to_band_px=dist,
               status='candidate' if holes==0 else 'ambiguous',reason='no holes in observed band; correspondence still unverified' if holes==0 else 'missing projected samples inside band')
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=False); rows=[]
    for case in CASES:
        rgb=np.asarray(Image.open(case['rgb']).convert('RGB'));gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        with np.load(case['depth']) as f:z=f['depth_m']
        assert z.shape==gray.shape and np.isfinite(z).all()
        for y in case['ys']: rows.append(dict(case=case['id'],**profile(z,gray,y,case['left'],case['right'],case['search'])))
    with (OUT/'development_rows.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    report=dict(protocol='Per row: medians from interior strips, 10%-to-90% first observed transition positions, holes counted; strongest horizontal RGB gradient compared to band. No interpolation or error bound.',
        selection='Development regions and rows selected after visual inspection; no independent validation.',
        cases=[dict(id=c['id'],left=c['left'],right=c['right'],search=c['search'],ys=c['ys']) for c in CASES],rows=rows)
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for c in CASES:
        rr=[r for r in rows if r['case']==c['id']];print(c['id'],[(r['status'],r['rgb_distance_to_band_px'],r['holes_in_band']) for r in rr],flush=True)

if __name__=='__main__':main()
