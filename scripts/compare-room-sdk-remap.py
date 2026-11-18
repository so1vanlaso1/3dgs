"""Compare baseline vs SDK-consistent inverse-Brown remap on stored frames."""
import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/realsense_20261030_134523_clean';DEP=ROOT/'results/room_improve_sdk_candidate_depth_20261103';OUT=ROOT/'results/room_improve_sdk_remap_comparison_20261103'
spec=importlib.util.spec_from_file_location('clean',ROOT/'scripts/clean-realsense-scene.py');clean=importlib.util.module_from_spec(spec);spec.loader.exec_module(clean)
# Frame sets chosen before seeing candidate remaps. Previously inspected nearby scenes, not blind independent scenes.
GROUPS={'development':[3,27,191,239,443],'check':[5,29,195,243,445]}
def main():
    OUT.mkdir(parents=True,exist_ok=False);cal=json.loads((DATA/'calibration_clean.json').read_text());manifest=json.loads((DATA/'capture_manifest.json').read_text());frames={r['name']:r for r in manifest['frames']};c=cal['raw_color'];K=np.array(cal['K']);w,h=cal['width'],cal['height'];coeff=np.array([float(v) for v in c['coeffs'].split(',')]);rawK=clean.intrinsic(c)
    yy,xx=np.mgrid[:h,:w];ideal=np.column_stack(((xx.ravel()-K[0,2])/K[0,0],(yy.ravel()-K[1,2])/K[1,1]))
    maps={}
    for name,rays in [('baseline',clean.invert_brown(ideal,coeff)),('sdk_consistent',clean.brown(ideal,coeff))]:
        maps[name]=(rays*[rawK[0,0],rawK[1,1]]+[rawK[0,2],rawK[1,2]]).reshape(h,w,2).astype('f4')
    displacement=np.linalg.norm(maps['sdk_consistent']-maps['baseline'],axis=2);rows=[]
    with sqlite3.connect((ROOT/'20261030_134523.db3').as_uri()+'?mode=ro',uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        for group,nums in GROUPS.items():
            for num in nums:
                name=f'frame_{num:06}.png';f=frames[name];stamp,blob=db.execute('select timestamp,data from messages where id=?',(f['message_id'],)).fetchone();assert stamp==f['timestamp_ns']
                rgb,header=clean.capture.CDR(blob).image();assert rgb.shape==(int(c['height']),int(c['width']),3)
                imgs={key:cv2.remap(rgb,mp[:,:,0],mp[:,:,1],cv2.INTER_LINEAR) for key,mp in maps.items()};saved=np.asarray(Image.open(DATA/'images'/name).convert('RGB'))
                diff=np.abs(imgs['baseline'].astype(int)-saved.astype(int));assert diff.max()==0
                with np.load(DEP/(name+'.dense_depth.npz')) as x:z=x['depth_m']
                edge=np.zeros((h,w),bool)
                for axis in (0,1):
                    a,b=(z[:-1,:],z[1:,:]) if axis==0 else (z[:,:-1],z[:,1:]);e=(a>0)&(b>0)&(abs(a-b)>np.maximum(.06,.025*np.minimum(a,b)))
                    if axis==0:edge[:-1,:]|=e
                    else:edge[:,:-1]|=e
                panel=Image.new('RGB',(2*w,2*(h+30)),'white');draw=ImageDraw.Draw(panel)
                for j,key in enumerate(('baseline','sdk_consistent')):
                    img=imgs[key];overlay=img.copy();overlay[edge]=[255,0,255]
                    Image.fromarray(img).save(OUT/f'{group}_{name}.{key}.png')
                    panel.paste(Image.fromarray(img),(j*w,30));panel.paste(Image.fromarray(overlay),(j*w,h+60));draw.text((j*w+5,7),key+' RGB',fill='black');draw.text((j*w+5,h+37),key+' with magenta depth jumps',fill='black')
                panel.save(OUT/f'{group}_{name}.comparison.jpg',quality=94)
                rows.append(dict(group=group,image_name=name,rgb_message_id=f['message_id'],depth_message_id=f['depth_message_id'],rgb_timestamp_ns=f['timestamp_ns'],depth_timestamp_ns=f['depth_timestamp_ns'],baseline_max_abs_rgb_difference=int(diff.max()),
                    raw_input_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),comparison=f'{group}_{name}.comparison.jpg'))
                print(group,name,'baseline exact',diff.max(),flush=True)
    (OUT/'report.json').write_text(json.dumps(dict(frames=rows,source_map_displacement_raw_rgb_px_percentiles=np.percentile(displacement,[50,90,99,100]).tolist(),
      proposed_mapping='For output ideal rays, forward Brown maps to original raw color pixels, inverse to SDK inverse-Brown deprojection.',
      caveat='SDK-consistent is a candidate math convention, not proven physical calibration or improvement. Rectified camera intrinsics and depth projections held fixed. Frame groups selected before candidate remaps; check views share locations and may have been previously inspected. Depth edges ambiguous, no metric fit, original DB3 read-only.'),indent=2),encoding='utf-8')
if __name__=='__main__':main()
