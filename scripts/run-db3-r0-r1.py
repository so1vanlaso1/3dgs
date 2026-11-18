"""Reproducible R0 baseline freeze and R1 input/calibration/timestamp work.

No pose recovery or training; those are R2/R3. Every stage writes a fresh artifact.
Run stages in order with the repository's .venv Python, from any working directory.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import sqlite3
import struct
import subprocess
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'data/realsense_20261030_134523_clean'
RUN = ROOT/'results/db3_r0_r1_20261110'
DATA = ROOT/'data/db3_r1_sdk_20261110'
DB = ROOT/'20261030_134523.db3'
CKPT = ROOT/'results/realsense_20261030_134523_120k/train/ckpts/final_post_update.pt'


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/name)
    obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj)
    return obj


clean = module('clean-realsense-scene.py')
p = clean.projection


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    if path.exists():
        raise FileExistsError(f'Preserving existing evidence: {path}')
    clean.write(path, value)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def connect():
    db = sqlite3.connect(DB.as_uri()+'?mode=ro', uri=True)
    db.execute('PRAGMA query_only=ON')
    return db


def freeze():
    RUN.mkdir(parents=True, exist_ok=False)
    start = time.time(); stat = DB.stat()
    print('Hashing full recording', stat.st_size, flush=True)
    source_hash = sha(DB)
    expected = read(ROOT/'results/realsense_20261030_134523_120k/provenance.json')['source_sha256']
    assert source_hash == expected, 'Source differs from baseline provenance'
    assert (DB.stat().st_size, DB.stat().st_mtime_ns) == (stat.st_size, stat.st_mtime_ns)
    manifest = read(BASE/'capture_manifest.json'); frames = manifest['frames']
    with (ROOT/'docs/room_baseline_heldout_20261103.csv').open(newline='', encoding='utf-8') as f:
        held = {r['image_name'] for r in csv.DictReader(f)}
    assert len(held) == 60 and len(frames) == 477 and held <= {f['name'] for f in frames}
    # Freeze prospective checks before producing their new remaps. Old footage and
    # old pose estimates are exposed; these reservations apply only to future work.
    origin = frames[0]['timestamp_ns']
    blocks = [('geometry_check', 1., 2.), ('geometry_check', 48., 49.),
              ('geometry_check', 98., 99.), ('final_prospective', 23., 25.),
              ('final_prospective', 88., 90.), ('final_prospective', 145., 147.)]
    def role(seconds):
        return next((kind for kind, lo, hi in blocks if lo-.5 <= seconds <= hi+.5), 'fit_development')
    with connect() as db:
        rows = db.execute("select id,timestamp from messages where topic_id=(select id from topics where name like '%/Color_0/image/data') order by timestamp").fetchall()
    source_roles = [dict(rgb_message_id=i, timestamp_ns=s, replay_seconds=(s-origin)/1e9,
                         role=role((s-origin)/1e9)) for i, s in rows]
    role_map = {f['rgb_message_id']: f['role'] for f in source_roles}
    split = dict(train=[f['name'] for f in frames if f['name'] not in held],
                 val=[f['name'] for f in frames if f['name'] in held],
                 scope='Historical appearance split; exposed development evaluation, jointly estimated poses')
    future = dict(blocks=[dict(role=k, start_seconds=a, end_seconds=b, buffer_seconds=.5) for k,a,b in blocks],
        source_frames=source_roles,
        future_fit_rgb_message_ids=[r['rgb_message_id'] for r in source_roles if r['role']=='fit_development'],
        future_geometry_check_rgb_message_ids=[r['rgb_message_id'] for r in source_roles if r['role']=='geometry_check'],
        future_final_rgb_message_ids=[r['rgb_message_id'] for r in source_roles if r['role']=='final_prospective'],
        rule='Exclude reserved blocks including buffers from future geometry fitting and splat training. Do not load final blocks during R1 visual review. Old graph already exposed; no historical independence claim.')
    write(RUN/'historical_split.json', split); write(RUN/'prospective_checks.json', future)
    files = [CKPT, BASE/'capture_manifest.json', BASE/'calibration_clean.json', BASE/'calibration_raw.json',
             BASE/'rgbd/report.json', BASE/'rgbd/poses.npy', ROOT/'results/realsense_20261030_134523_120k/train/cfg.yml',
             ROOT/'results/realsense_20261030_134523_120k/preview/verification.json']
    files += list((BASE/'rgbd/dataset/sparse').glob('*'))
    files += list((ROOT/'results/room_improve_baseline_inspection_20261103').glob('*'))
    files += list((ROOT/'results/room_improve_free_path_v02_20261103').glob('*'))
    records = []
    for src in files:
        if not src.is_file():
            continue
        dst = RUN/'baseline'/src.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
        h = sha(src); assert sha(dst) == h
        records.append(dict(source=str(src.relative_to(ROOT)), saved_copy=str(dst.relative_to(RUN)), bytes=src.stat().st_size, sha256=h))
    identities = [dict(name=f['name'], rgb_message_id=f['message_id'], depth_message_id=f['depth_message_id'],
                       rgb_timestamp_ns=f['timestamp_ns'], depth_timestamp_ns=f['depth_timestamp_ns'],
                       appearance_role='development_eval' if f['name'] in held else 'train',
                       prospective_role=role_map[f['message_id']],
                       rgb_sha256=sha(BASE/'images'/f['name']),
                       depth_sha256=sha(BASE/'aligned_depth_meters'/(f['name']+'.npz'))) for f in frames]
    try:
        gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'], text=True).strip()
    except OSError:
        gpu = 'unavailable'
    write(RUN/'baseline_manifest.json', dict(source=str(DB), source_bytes=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns, source_sha256=source_hash, baseline_source_hash_matches=True,
        files=records, frames=identities, platform=platform.platform(), python=platform.python_version(), gpu=gpu,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        worktree=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        elapsed_seconds=time.time()-start, phase='R0 frozen provenance; visual failure map follows',
        source_policy='Raw DB read-only; baseline checkpoint copied and hash-verified; original artifacts preserved'))
    print('R0 source and baseline frozen', source_hash, flush=True)


def sdk_audit():
    import sys
    sys.path.insert(0, str(ROOT/'.cache/room_sdk_audit/install'))
    import pyrealsense2 as rs
    assert rs.__version__ == '2.59.1'
    cal = read(BASE/'calibration_clean.json'); results = []
    for key in ('raw_color', 'raw_depth'):
        info = cal[key]; intr = rs.intrinsics()
        for name in ('width', 'height'): setattr(intr, name, int(info[name]))
        for name in ('fx', 'fy', 'ppx', 'ppy'): setattr(intr, name, float(info[name]))
        intr.model = {0: rs.distortion.none, 2: rs.distortion.inverse_brown_conrady, 4: rs.distortion.brown_conrady}[p.MODEL_IDS[info['model']]]
        intr.coeffs = p.coefficients(info).tolist()
        grid = np.array([(x, y) for y in np.linspace(0, intr.height-1, 25) for x in np.linspace(0, intr.width-1, 41)])
        rng = np.random.default_rng(11010)
        pixels = np.vstack((grid, rng.uniform([0,0],[intr.width-1,intr.height-1],(2000,2))))
        max_proj = max_dep = max_round = 0.
        for z in (.3,.7,2.,5.9):
            dep = p.deproject(pixels, np.full(len(pixels),z), info)
            sd = np.array([rs.rs2_deproject_pixel_to_point(intr, uv.tolist(), z) for uv in pixels])
            sp = np.array([rs.rs2_project_point_to_pixel(intr, xyz.tolist()) for xyz in dep])
            max_dep = max(max_dep, float(np.linalg.norm(dep-sd,axis=1).max()))
            max_proj = max(max_proj, float(np.linalg.norm(p.project(dep, info)-sp,axis=1).max()))
            max_round = max(max_round, float(np.linalg.norm(p.project(dep, info)-pixels,axis=1).max()))
        results.append(dict(stream=key,model=info['model'],model_id=p.MODEL_IDS[info['model']],samples=len(pixels)*4,
            depths_m=[.3,.7,2.,5.9],sdk_project_max_raw_pixel_error=max_proj,
            sdk_deproject_max_m_error=max_dep,roundtrip_max_raw_pixel_error=max_round))
    maps, valid = p.remap(cal['raw_color'], np.array(cal['K']),cal['width'],cal['height'])
    K = np.array(cal['K']); indices = np.array([(x,y) for y in np.linspace(0,cal['height']-1,23,dtype=int) for x in np.linspace(0,cal['width']-1,37,dtype=int)])
    uv = maps[indices[:,1],indices[:,0]]
    back = p.deproject(uv,np.ones(len(uv)),cal['raw_color'])
    reproj = back[:,:2]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
    crop_roundtrip = float(np.linalg.norm(reproj-indices,axis=1).max())
    raw = read(BASE/'calibration_raw.json')
    depth_tf = next(v for k,v in raw.items() if k.endswith('/Depth_0/tf/ref_0'))
    color_tf = next(v for k,v in raw.items() if k.endswith('/Color_0/tf/ref_0'))
    R,t=p.relative_extrinsics(depth_tf,color_tf)
    ext=rs.extrinsics();ext.rotation=R.ravel(order='F').tolist();ext.translation=t.tolist()
    xyz=np.array([[.1,.2,.7],[-1,.4,2.],[2,-1,5.9]])
    error=float(np.max(np.linalg.norm(np.array([rs.rs2_transform_point_to_point(ext,x.tolist()) for x in xyz])-(xyz@R.T+t),axis=1)))
    passed=all(r['sdk_project_max_raw_pixel_error']<.1 and r['roundtrip_max_raw_pixel_error']<.1 and r['sdk_deproject_max_m_error']<1e-5 for r in results)
    passed &= bool(valid.all()) and crop_roundtrip<.1 and error<1e-5
    wheel=next((ROOT/'.cache/room_sdk_audit').glob('*.whl'))
    write(RUN/'sdk_audit.json',dict(projection_version=p.SDK,sdk_version=rs.__version__,wheel_sha256=sha(wheel),
        source_reference='https://github.com/realsenseai/librealsense/blob/v2.59.1/src/rs.cpp#L4012-L4164',
        tests=results,crop_xy=cal['crop_xy'],dimensions=[cal['width'],cal['height']],
        target_map_all_valid=bool(valid.all()),scaled_cropped_roundtrip_max_px=crop_roundtrip,
        extrinsics_sdk_transform_max_m=error,rotation_layout='column-major',depth_units_m=cal['depth_units'],
        gate_raw_rgb_px=.1,numerical_gate_pass=bool(passed),
        physical_accuracy='Not established by SDK compatibility'))
    assert passed, 'SDK gate failed; do not prepare candidate'
    print('SDK numerical gate PASS',results,flush=True)


def prepare():
    assert read(RUN/'sdk_audit.json')['numerical_gate_pass']
    start=time.time(); DATA.mkdir(parents=True,exist_ok=False)
    for d in ('images','aligned_depth_meters','projection_evidence'): (DATA/d).mkdir()
    cal=read(BASE/'calibration_clean.json'); cal['projection_version']=p.SDK
    K=np.array(cal['K']); w,h=cal['width'],cal['height']
    maps,valid=p.remap(cal['raw_color'],K,w,h);assert valid.all()
    np.savez_compressed(DATA/'rgb_remap.npz',map_xy=maps,valid=valid)
    write(DATA/'calibration_clean.json',cal)
    shutil.copyfile(BASE/'calibration_raw.json',DATA/'calibration_raw.json')
    manifest=read(BASE/'capture_manifest.json'); frames=[];rng=np.random.default_rng(42)
    baseline=read(RUN/'baseline_manifest.json'); before=DB.stat()
    split=read(RUN/'historical_split.json'); prospective=read(RUN/'prospective_checks.json')
    roles={r['rgb_message_id']:r['role'] for r in prospective['source_frames']}
    with connect() as db:
        for j,f in enumerate(manifest['frames']):
            ts,blob=db.execute('select timestamp,data from messages where id=?',(f['message_id'],)).fetchone();assert ts==f['timestamp_ns']
            rgb,rh=clean.capture.CDR(blob).image()
            ts,blob=db.execute('select timestamp,data from messages where id=?',(f['depth_message_id'],)).fetchone();assert ts==f['depth_timestamp_ns']
            depth,dh=clean.adapter.decode_depth(blob)
            assert abs(dh-f['depth_header_seconds'])<1e-6 and abs(rh['header_seconds']-f['header_seconds'])<1e-6
            image=cv2.remap(rgb,maps[:,:,0],maps[:,:,1],cv2.INTER_LINEAR)
            Image.fromarray(image).save(DATA/'images'/f['name'],compress_level=1)
            px,z,fraction,e=p.align_depth(depth,cal['depth_units'],cal['raw_depth'],cal['depth_to_color_R'],cal['depth_to_color_t_m'],K,w,h)
            keep=rng.choice(len(z),min(16000,len(z)),replace=False)
            dest=DATA/'aligned_depth_meters'/(f['name']+'.npz')
            np.savez_compressed(dest,points=px[keep],depths=z[keep])
            with np.load(BASE/'aligned_depth_meters'/(f['name']+'.npz')) as a:
                assert np.array_equal(a['points'],px[keep]) and np.array_equal(a['depths'],z[keep]), 'Depth policy unexpectedly changed'
            # Evidence masks contain no RGB. Missing is not classified as free space;
            # collisions mark potential occlusions, not verified physical disocclusions.
            masks=DATA/'projection_evidence'/(f['name']+'.npz')
            np.savez_compressed(masks,observed=e['observed'],missing=e['missing'],multiple_projected_samples=e['multiple_projected_samples'])
            frames.append(dict(f,rgb_sha256=sha(DATA/'images'/f['name']),depth_sha256=sha(dest),
                               projection_version=p.SDK,prospective_role=roles[f['message_id']],
                               discarded_projection_collisions=e['discarded_collisions']))
            if j%50==0: print('R1 prepare',j,'/',len(manifest['frames']),flush=True)
    after=DB.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    manifest=dict(manifest,frames=frames,source_sha256=baseline['source_sha256'],
        parameters=dict(manifest['parameters'],projection_version=p.SDK),
        stage='rectified_inputs_only',poses_features_matches='Must be rebuilt in R2; none copied')
    write(DATA/'capture_manifest.json',manifest);write(DATA/'experiment_split.json',split)
    future_train=[f['name'] for f in frames if f['name'] in split['train'] and roles[f['message_id']]=='fit_development']
    write(DATA/'reconstruction_split.json',dict(train=future_train,
        val=[f['name'] for f in frames if roles[f['message_id']]=='geometry_check'],
        final=[f['name'] for f in frames if roles[f['message_id']]=='final_prospective'],
        historical_eval=split['val'],rule='Use this reduced train list for future training. Historical split is for reporting only; geometry fitting must also exclude all reserved native source IDs in prospective_checks.json.'))
    write(DATA/'input_validation.json',dict(status='numerically_consistent_inputs; physical_alignment_gate_pending',
        frames=len(frames),projection_version=p.SDK,all_rgb_maps_valid=True,
        depth_regenerated_and_exactly_matches_baseline_arrays=True,historical_split_counts=[len(split['train']),len(split['val'])],
        future_training_frames=len(future_train),depth_hole_filling=False,
        missing_policy='Unknown; not empty space',occlusion_policy='Nearest camera Z; collision masks expose multiple samples. Physical disocclusion classification requires later multiview poses.',
        poses_copied=False,features_copied=False,elapsed_seconds=time.time()-start))
    print('R1 inputs complete',len(frames),flush=True)


def stamp_header(blob):
    c=clean.capture.CDR(blob);sec=c.uint32();ns=c.uint32();frame=c.string()
    return int(sec)*1_000_000_000+int(ns),frame


def timing():
    start=time.time(); streams={};metadata={}; gyro=[]
    with connect() as db:
        topics=db.execute('select id,name,type from topics').fetchall()
        for tid,name,typ in topics:
            if name.endswith(('/Color_0/image/data','/Depth_0/image/data','/imu/data')):
                rows=[]
                for mid,ts,blob in db.execute('select id,timestamp,substr(data,1,512) from messages where topic_id=? order by timestamp',(tid,)):
                    hs,frame=stamp_header(blob);rows.append((mid,ts,hs))
                    if '/Gyro_' in name:
                        c,_,_=clean.header(blob);c.offset=4+((c.offset-4+7)//8)*8
                        vals=struct.unpack_from('<37d',blob,c.offset);gyro.append((ts,*vals[13:16]))
                a=np.array(rows,dtype=np.int64); streams[name]=dict(rows=a,frame=frame)
            elif name.endswith('/metadata'):
                # Read all small metadata strings, no image payload decoding.
                values=[]
                for mid,ts,blob in db.execute('select id,timestamp,data from messages where topic_id=? order by timestamp',(tid,)):
                    values.append(dict(message_id=mid,timestamp_ns=ts,text=clean.capture.CDR(blob).string()))
                metadata[name]=values
        options={name:clean.capture.CDR(blob).string() for name,blob in db.execute("select t.name,m.data from topics t join messages m on m.topic_id=t.id where t.type='std_msgs/msg/String' and (t.name like '%/Exposure/value' or t.name like '%/Global_Time_Enabled/value' or t.name like '%/Inter_Cam_Sync_Mode/value')")}
    summaries=[]
    for name,s in streams.items():
        a=s['rows'];summaries.append(dict(topic=name,count=len(a),frame_id=s['frame'],
            database_strictly_increasing=bool(np.all(np.diff(a[:,1])>0)),
            header_strictly_increasing=bool(np.all(np.diff(a[:,2])>0)),
            database_minus_header_ms_percentiles=np.percentile((a[:,1]-a[:,2])/1e6,[0,50,95,100]).tolist(),
            interval_ms_percentiles=np.percentile(np.diff(a[:,2])/1e6,[0,50,95,100]).tolist()))
    rgb=next(s['rows'] for n,s in streams.items() if '/Color_' in n)
    dep=next(s['rows'] for n,s in streams.items() if '/Depth_' in n)
    i=np.searchsorted(dep[:,1],rgb[:,1]);i=np.clip(i,1,len(dep)-1)
    i-= (np.abs(dep[i-1,1]-rgb[:,1])<=np.abs(dep[i,1]-rgb[:,1])).astype(int)
    delta=(dep[i,1]-rgb[:,1])/1e6; hd=(dep[i,2]-rgb[:,2])/1e6
    ga=np.array(gyro);gj=np.searchsorted(ga[:,0],rgb[:,1]);gj=np.clip(gj,1,len(ga)-1)
    gj-= (np.abs(ga[gj-1,0]-rgb[:,1])<=np.abs(ga[gj,0]-rgb[:,1])).astype(int)
    speed=np.linalg.norm(ga[gj,1:],axis=1)
    # This is an angular-motion scale estimate, never a correction vector.
    cal=read(BASE/'calibration_clean.json'); shift=float(cal['K'][0][0])*speed*np.abs(delta)/1000
    frozen=read(RUN/'prospective_checks.json');roles={r['rgb_message_id']:r['role'] for r in frozen['source_frames']}
    rows=[dict(rgb_message_id=int(r[0]),depth_message_id=int(dep[j,0]),rgb_timestamp_ns=int(r[1]),
               signed_depth_minus_rgb_ms=float(d),signed_header_depth_minus_rgb_ms=float(h),
               nearest_gyro_rad_s=float(g),rotational_pixel_scale_estimate=float(s),
               prospective_role=roles[int(r[0])]) for r,j,d,h,g,s in zip(rgb,i,delta,hd,speed,shift)]
    with (RUN/'timing_pairs.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    groups=[]
    for label,mask in [('low_motion',speed<=np.quantile(speed,.25)),('high_motion',speed>=np.quantile(speed,.75))]:
        groups.append(dict(group=label,count=int(mask.sum()),gyro_rad_s_percentiles=np.percentile(speed[mask],[0,50,100]).tolist(),
            rotational_pixel_scale_percentiles=np.percentile(shift[mask],[50,95,100]).tolist()))
    write(RUN/'timing_metadata.json',metadata)
    write(RUN/'timing_report.json',dict(streams=summaries,options=options,
        paired_frames=len(rows),signed_depth_minus_rgb_ms_percentiles=np.percentile(delta,[0,50,95,100]).tolist(),
        signed_header_depth_minus_rgb_ms_percentiles=np.percentile(hd,[0,50,95,100]).tolist(),
        database_header_pair_difference_max_ms=float(np.abs(delta-hd).max()),motion_groups=groups,
        metadata_topics=[dict(topic=n,count=len(v),first=v[0] if v else None) for n,v in metadata.items()],
        decision='Retain recorded nearest pair. No timing correction accepted. Use low-motion observations for boundary checks; defer motion compensation until validated local poses exist.',
        caveats='Header/database agreement does not prove exposure simultaneity or an absolute sensor-clock domain. Gyro nearest-sample f*omega*dt is a scale diagnostic, not integrated image displacement; translation, exposure, lever arm and rolling shutter are not modeled.',
        elapsed_seconds=time.time()-start))
    print('Timing complete',np.percentile(delta,[0,50,95,100]).tolist(),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['freeze','sdk','prepare','timing'])
    a=ap.parse_args();globals()[dict(sdk='sdk_audit').get(a.stage,a.stage)]()


if __name__=='__main__':main()
