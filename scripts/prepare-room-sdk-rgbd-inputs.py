"""Create isolated depth-input copies and preserve original source-ID split."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'data/realsense_20261030_134523_clean';OUT=ROOT/'data/room_improve_sdk_rectified_20261103'
def main():
    manifest=json.loads((OUT/'candidate_manifest.json').read_text());names=[f['name'] for f in manifest['frames']]
    with (ROOT/'docs/room_baseline_heldout_20261103.csv').open(newline='',encoding='utf-8') as f:held={r['image_name'] for r in csv.DictReader(f)}
    assert len(names)==477 and len(held)==60 and held<=set(names)
    p=OUT/'aligned_depth_meters';p.mkdir(exist_ok=False);hashes=[]
    for name in names:
        src=SRC/'aligned_depth_meters'/(name+'.npz');dst=p/(name+'.npz');shutil.copyfile(src,dst)
        a=hashlib.sha256(src.read_bytes()).hexdigest();b=hashlib.sha256(dst.read_bytes()).hexdigest();assert a==b;hashes.append(dict(name=name,sha256=a))
    split=dict(train=[n for n in names if n not in held],val=[n for n in names if n in held],
       note='Original source-identity split; pose recovery and optimization still see all frames. Dataset training loader must support this explicit split before any matched training.')
    (OUT/'experiment_split.json').write_text(json.dumps(split,indent=2),encoding='utf-8')
    (OUT/'depth_copy_provenance.json').write_text(json.dumps(dict(files=hashes,source=str(SRC/'aligned_depth_meters'),same_K=True,
      caveat='Copied original aligned depth samples, not physically validated. RGB mapping changed; this is a provisional candidate for testing, not acceptance.'),indent=2),encoding='utf-8')
    print('copied',len(hashes),'depth files; train',len(split['train']),'heldout',len(split['val']))
if __name__=='__main__':main()
