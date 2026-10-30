"""15 FPS sampling and matching experiment; never modifies the 2 FPS dataset."""
import argparse
import importlib.util
import json
import os
import sqlite3
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '8')
import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / 'data/realsense_20261029_140739'
ROOT = REPO / 'data/realsense_20261029_140739_15fps'
RESULT = REPO / 'results/realsense_15fps_experiment'
spec = importlib.util.spec_from_file_location('capture15', REPO/'scripts/prepare-realsense-scene.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def extract():
    ROOT.mkdir(parents=True, exist_ok=True)
    RESULT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'capture_manifest.json').exists():
        print('EXTRACTION already complete', flush=True)
        return
    baseline = json.loads((BASE/'capture_manifest.json').read_text())
    original = {f['message_id']: f for f in baseline['frames']}
    recording = Path(baseline['source'])
    frames = []
    image_dir = ROOT/'raw_images'
    image_dir.mkdir(exist_ok=True)
    with sqlite3.connect(recording.as_uri()+'?mode=ro', uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        tid = db.execute('select id from topics where name=?', (baseline['rgb_topic'],)).fetchone()[0]
        rows = db.execute('select id,timestamp from messages where topic_id=? order by timestamp', (tid,)).fetchall()
        first, last = rows[0][1], rows[-1][1]
        bins = {}
        for mid, stamp in rows:
            b = (stamp-first)*15//1_000_000_000
            bins.setdefault(b, []).append((mid, stamp))
        for j, (b, choices) in enumerate(sorted(bins.items())):
            anchors = [(m,s) for m,s in choices if m in original]
            candidates = []
            for mid, stamp in anchors or choices:
                if mid in original:
                    f = original[mid]
                    candidates.append((f['sharpness_640'], mid, stamp, None, f))
                else:
                    blob = db.execute('select data from messages where id=?', (mid,)).fetchone()[0]
                    rgb, header = capture.CDR(blob).image()
                    gray = cv2.cvtColor(cv2.resize(rgb, (640,360)), cv2.COLOR_RGB2GRAY)
                    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    candidates.append((score, mid, stamp, rgb, header))
            score, mid, stamp, rgb, header = max(candidates, key=lambda x:x[0])
            name = f'frame_{len(frames):06d}.jpg'
            dst = image_dir/name
            if mid in original:
                src = BASE/'raw_images'/original[mid]['name']
                if not dst.exists():
                    os.link(src, dst)
                width,height = header['width'],header['height']
            else:
                Image.fromarray(rgb).save(dst, quality=96, subsampling=0)
                height,width = rgb.shape[:2]
            frames.append(dict(name=name,message_id=mid,timestamp_ns=stamp,time_seconds=stamp/1e9,
                               sharpness_640=score,width=width,height=height,
                               header_seconds=header['header_seconds'],frame_id=header['frame_id'],
                               baseline_name=original[mid]['name'] if mid in original else None))
            if j%100==0:
                print('EXTRACT',j+1,len(bins),flush=True)
        stat=recording.stat()
        assert stat.st_size==baseline['source_bytes'] and stat.st_mtime_ns==baseline['source_mtime_ns']
        assert set(original).issubset({f['message_id'] for f in frames})
        manifest=dict(source=str(recording),source_bytes=stat.st_size,source_mtime_ns=stat.st_mtime_ns,
                      source_open_mode='read-only',rgb_topic=baseline['rgb_topic'],target_fps=15,
                      duration_seconds=(last-first)/1e9,selection='sharpest RGB in disjoint 1/15-second bins; original 2 FPS frames retained for exact comparison',
                      actual_average_fps=len(frames)/((last-first)/1e9),frames=frames)
        save(ROOT/'capture_manifest.json',manifest)
    (ROOT/'calibration.json').write_bytes((BASE/'calibration.json').read_bytes())
    print('EXTRACTION_COMPLETE',len(frames),flush=True)


def features():
    import pycolmap as p
    manifest=json.loads((ROOT/'capture_manifest.json').read_text())
    remap={f['baseline_name']:f['name'] for f in manifest['frames'] if f['baseline_name']}
    dbpath=ROOT/'colmap.db'
    if not dbpath.exists():
        # Reuse observations/matches, never baseline camera poses. Image IDs stay
        # unchanged so existing verified long-range correspondences remain valid.
        with sqlite3.connect((BASE/'colmap.db').as_uri()+'?mode=ro',uri=True) as src, sqlite3.connect(dbpath) as dst:
            src.backup(dst)
            rows=dst.execute('select image_id,name from images').fetchall()
            for iid,name in rows:
                dst.execute('update images set name=? where image_id=?',('old_'+name,iid))
            for iid,name in rows:
                dst.execute('update images set name=? where image_id=?',(remap[name],iid))
            dst.commit()
    # Match baseline feature settings exactly for cached common observations.
    reader=p.ImageReaderOptions();reader.camera_model='OPENCV'
    cal=json.loads((ROOT/'calibration.json').read_text())
    values=dict(v.split('=',1) for v in next(v for k,v in cal.items() if k.endswith('/Color_0/camera_info')).split(';'))
    reader.camera_params=','.join([values[k] for k in ['fx','fy','ppx','ppy']]+['0']*4)
    opts=p.FeatureExtractionOptions();opts.num_threads=8;opts.max_image_size=1280
    opts.sift.max_num_features=4096;opts.sift.first_octave=0
    p.extract_features(dbpath,ROOT/'raw_images',camera_mode=p.CameraMode.SINGLE,
                       reader_options=reader,extraction_options=opts,device=p.Device.cpu)
    with sqlite3.connect(dbpath) as db:
        n=db.execute('select count(*) from descriptors').fetchone()[0]
    assert n==len(manifest['frames'])
    print('FEATURES_COMPLETE',n,flush=True)


def match():
    import pycolmap as p
    frames=json.loads((ROOT/'capture_manifest.json').read_text())['frames']
    pairs=[]
    for i,f in enumerate(frames):
        for delta in (1,2,4,8,16,32):
            if i+delta<len(frames):
                pairs.append(f['name']+' '+frames[i+delta]['name'])
    pairpath=ROOT/'15fps_pairs.txt';pairpath.write_text('\n'.join(pairs)+'\n')
    opts=p.FeatureMatchingOptions();opts.num_threads=8
    po=p.ImportedPairingOptions();po.match_list_path=str(pairpath)
    print('MATCH_PAIRS',len(pairs),flush=True)
    p.match_image_pairs(ROOT/'colmap.db',matching_options=opts,pairing_options=po,device=p.Device.cpu)
    print('MATCH_COMPLETE',flush=True)


def split():
    import pycolmap
    frames=json.loads((ROOT/'capture_manifest.json').read_text())['frames']
    model=pycolmap.Reconstruction(BASE/'rgbd/dataset/sparse')
    names=sorted(im.name for im in model.images.values())
    baseline_val=set(names[::8])
    anchors=[f for f in frames if f['baseline_name'] in baseline_val]
    times=np.array([f['time_seconds'] for f in anchors])
    val=[f['name'] for f in anchors]
    train=[f['name'] for f in frames if np.min(np.abs(times-f['time_seconds']))>.2]
    save(ROOT/'experiment_split.json',dict(train=train,val=val,
         policy='Original 2 FPS held-out images; exclude training within +/-0.2 seconds of them',
         baseline_val_names=sorted(baseline_val),excluded_views=len(frames)-len(train)-len(val)))
    assert len(val)==79 and not set(train)&set(val)
    print('SPLIT_COMPLETE',len(train),len(val),flush=True)


def sfm_anchors():
    import pycolmap as p
    frames=json.loads((ROOT/'capture_manifest.json').read_text())['frames']
    names=[f['name'] for f in frames if f['baseline_name']]
    options=p.IncrementalPipelineOptions()
    options.num_threads=8;options.random_seed=42
    options.image_names=set(names)
    options.max_num_models=12;options.min_model_size=15;options.init_num_trials=50
    options.mapper.init_min_tri_angle=4;options.mapper.filter_min_tri_angle=.5
    options.triangulation.min_angle=.5;options.max_runtime_seconds=600
    target=ROOT/'reconstruction_global';target.mkdir(exist_ok=True)
    print('Rebuilding SfM on',len(names),'original source observations for metric loop anchors',flush=True)
    models=p.incremental_mapping(ROOT/'colmap.db',ROOT/'raw_images',target,options=options)
    report=dict(purpose='Fresh SfM on original 694 source frames for depth-validated relative anchors; no baseline poses copied',
                selected_anchor_frames=len(names),models=[dict(id=i,registered_images=m.num_reg_images(),
                points3D=m.num_points3D(),mean_reprojection_error=m.compute_mean_reprojection_error()) for i,m in models.items()])
    save(ROOT/'global_reconstruction_report.json',report)
    print('SFM_ANCHORS_COMPLETE',report,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['extract','features','match','split','sfm_anchors'])
    args=ap.parse_args()
    start=time.time();globals()[args.stage]()
    RESULT.mkdir(exist_ok=True,parents=True)
    save(RESULT/(args.stage+'_complete.json'),dict(stage=args.stage,elapsed_seconds=time.time()-start))
