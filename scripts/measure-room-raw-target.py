"""Check the same provisional box boundary in seven original nearby frames."""
import csv,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'results/room_improve_target_raw_20261103'
# Hand-selected boxes around the same left edge; fixed in advance of measuring.
STRIPS={6253:(545,572),6383:(520,544),6433:(505,528),6487:(488,515),
        6539:(470,495),6617:(440,465),6721:(405,435)}
SCANLINES=[370,380,390,400,410,420,430,440]

def main():
    records=json.loads((P/'manifest.json').read_text())['records']; rows=[];summaries=[]
    for r in records:
        mid=r['rgb_message_id'];lo,hi=STRIPS[mid]
        rgb=np.asarray(Image.open(P/r['rgb']).convert('RGB'));gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);grad=abs(cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3))
        with np.load(P/r['depth']) as f:z=f['depth_m']
        crop=Image.fromarray(rgb[355:450,lo-10:hi+10]).resize(((hi-lo+20)*8,760));draw=ImageDraw.Draw(crop);vals=[]
        for y in SCANLINES:
            rgbx=lo+int(np.argmax(grad[y,lo:hi]));strength=float(grad[y,rgbx]);a=z[y,lo:hi-1];b=z[y,lo+1:hi];delta=abs(a-b)
            valid=(a>0)&(b>0)&(delta>np.maximum(.06,.025*np.minimum(a,b)));idx=np.flatnonzero(valid)
            if len(idx)!=1 or strength<60:
                rows.append(dict(rgb_message_id=mid,y=y,rgb_x=rgbx,depth_x='',signed_depth_minus_rgb_px='',depth_a_m='',depth_b_m='',status=f'rejected_{len(idx)}_jumps_or_weak_rgb'));continue
            k=int(idx[0]);dx=lo+k;off=dx-rgbx;vals.append(off)
            rows.append(dict(rgb_message_id=mid,y=y,rgb_x=rgbx,depth_x=dx,signed_depth_minus_rgb_px=off,depth_a_m=float(a[k]),depth_b_m=float(b[k]),status='candidate'))
            ry=(y-355)*8+4;rx=(rgbx-(lo-10))*8+4;xx=(dx-(lo-10))*8+4
            draw.ellipse((rx-4,ry-4,rx+4,ry+4),fill='cyan');draw.ellipse((xx-2,ry-2,xx+2,ry+2),fill='magenta')
        crop.save(P/f'rgb_{mid}.box_markers.png')
        summaries.append(dict(rgb_message_id=mid,role='development' if mid==6487 else 'nearby_view_check',
          offset_from_reference_s=r['offset_from_reference_s'],gyro_rad_s=r['gyro_rad_s'],strip_x=[lo,hi],candidate_rows=len(vals),
          median_px=float(np.median(vals)) if vals else None,range_px=[min(vals),max(vals)] if vals else None,marker_file=f'rgb_{mid}.box_markers.png'))
    with (P/'box_sequence_measurements.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (P/'box_sequence_report.json').write_text(json.dumps(dict(views=summaries,
       convention='depth x minus RGB x; depth jump uses left sample',
       note='Manually selected strips on same object; automated strongest RGB gradient and unique valid depth jump. Correlated frames and rows; selected after inspecting views; no independent calibration validation.'),indent=2),encoding='utf-8')
    for s in summaries:print(s['rgb_message_id'],s['candidate_rows'],s['median_px'],s['range_px'])

if __name__=='__main__':main()
