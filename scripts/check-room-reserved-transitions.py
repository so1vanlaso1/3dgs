"""Apply the previously defined development profile to reserved frames."""
import csv
import importlib.util
import json
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('transitions',ROOT/'scripts/audit-room-transition-profile.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
OUT=ROOT/'results/room_improve_transition_reserved_20261103'
P=ROOT/'results/room_improve_reserved_alignment_20261103'
I=ROOT/'data/realsense_20261030_134523_clean/images'
# Coordinates chosen after inspecting reserved-frame overlays: not blind validation.
CASES=[
 dict(id='kiosk_004',name='frame_000004.png',left=(50,75),right=(125,140),search=(75,125),ys=[400,410,420,430,440]),
 dict(id='box_145',name='frame_000145.png',left=(25,65),right=(120,145),search=(65,120),ys=[390,400,410,420,430]),
 dict(id='statue_295',name='frame_000295.png',left=(460,480),right=(510,530),search=(480,510),ys=[270,280,290,300,310]),
]
def main():
    OUT.mkdir(parents=True,exist_ok=False);rows=[]
    for c in CASES:
        gray=cv2.imread(str(I/c['name']),cv2.IMREAD_GRAYSCALE)
        with np.load(P/(c['name']+'.dense_depth.npz')) as f:z=f['depth_m']
        for y in c['ys']:rows.append(dict(case=c['id'],**m.profile(z,gray,y,c['left'],c['right'],c['search'])))
    with (OUT/'reserved_rows.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (OUT/'report.json').write_text(json.dumps(dict(cases=CASES,rows=rows,note='Profile algorithm fixed before viewing reserved frames; case locations and rows selected after inspection. Adjacent views / related objects, no independent calibration ground truth; no correction was proposed.'),indent=2),encoding='utf-8')
    for c in CASES:
        print(c['id'],[(r['status'],r['rgb_distance_to_band_px'],r['holes_in_band'],r['reason']) for r in rows if r['case']==c['id']])
if __name__=='__main__':main()
