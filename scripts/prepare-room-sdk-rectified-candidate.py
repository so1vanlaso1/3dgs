"""Create isolated RGB-only candidate with SDK-consistent inverse-Brown remap.

No camera poses, point cloud, depth arrays or trained model are copied/changed.
"""
import csv
import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/realsense_20261030_134523_clean';OUT=ROOT/'data/room_improve_sdk_rectified_20261103';DB=ROOT/'20261030_134523.db3'
spec=importlib.util.spec_from_file_location('clean',ROOT/'scripts/clean-realsense-scene.py');clean=importlib.util.module_from_spec(spec);spec.loader.exec_module(clean)
def main():
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'images').mkdir();cal=json.loads((DATA/'calibration_clean.json').read_text());manifest=json.loads((DATA/'capture_manifest.json').read_text());c=cal['raw_color'];K=np.array(cal['K']);w,h=cal['width'],cal['height'];coeff=np.array(list(map(float,c['coeffs'].split(','))))
    mp,valid=clean.projection.remap(c,K,w,h,clean.projection.SDK)
    assert valid.all()
    with (ROOT/'docs/room_baseline_heldout_20261103.csv').open(newline='',encoding='utf-8') as f:held={r['image_name'] for r in csv.DictReader(f)}
    before=DB.stat();rows=[]
    with sqlite3.connect(DB.as_uri()+'?mode=ro',uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        for j,f in enumerate(manifest['frames']):
            name=f['name'];stamp,blob=db.execute('select timestamp,data from messages where id=?',(f['message_id'],)).fetchone();assert stamp==f['timestamp_ns']
            raw,header=clean.capture.CDR(blob).image();assert raw.shape==(int(c['height']),int(c['width']),3)
            img=cv2.remap(raw,mp[:,:,0],mp[:,:,1],cv2.INTER_LINEAR);assert img.shape==(h,w,3)
            path=OUT/'images'/name;Image.fromarray(img).save(path,compress_level=1)
            rows.append(dict(name=name,rgb_message_id=f['message_id'],rgb_timestamp_ns=f['timestamp_ns'],depth_message_id=f['depth_message_id'],depth_timestamp_ns=f['depth_timestamp_ns'],held_out=name in held,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            if j%50==0:print(j,name,flush=True)
    after=DB.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns) and len(rows)==477 and sum(r['held_out'] for r in rows)==60
    report=dict(status='RGB candidate only; pose reconstruction, RGB-depth verification and training not complete',source=str(DB),
       source_sha256=json.loads((ROOT/'results/realsense_20261030_134523_120k/provenance.json').read_text())['source_sha256'],source_hash_note='Baseline provenance hash, independently rechecked earlier; not recomputed in this command.',
       projection_version=clean.projection.SDK, remap='Pinned SDK projection of output ideal rays to raw RGB pixels',
       dimensions=[w,h],K=K.tolist(),valid_mapping_fraction=float(valid.mean()),num_images=len(rows),heldout_images=sum(r['held_out'] for r in rows),frames=rows,
       limitation='Same original image IDs and K, different RGB pixel mapping. Original poses, point cloud, depth screening and trained baseline do not automatically apply; rebuild/revalidate independently. Existing depth projections in that K may be reused only as provisional measurements, pending physical alignment checks.')
    cal['projection_version']=clean.projection.SDK
    (OUT/'calibration_clean.json').write_text(json.dumps(cal,indent=2),encoding='utf-8')
    (OUT/'candidate_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('saved',len(rows),'images, heldout',sum(r['held_out'] for r in rows),flush=True)
if __name__=='__main__':main()
