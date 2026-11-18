"""Final integrity audit of R0/R1 artifacts; no pose/geometry accuracy inference."""
import importlib.util
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('r1',ROOT/'scripts/run-db3-r0-r1.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def main(output_dir=None):
    destination=Path(output_dir) if output_dir else m.RUN/'reaudits'/time.strftime('%Y%m%d_%H%M%S')
    destination.mkdir(parents=True,exist_ok=False)
    start=time.time();frozen=m.read(m.RUN/'baseline_manifest.json');manifest=m.clean.check_source(m.DATA)
    assert m.sha(m.DB)==frozen['source_sha256']
    for f in frozen['files']:
        assert m.sha(ROOT/f['source'])==f['sha256']
        assert m.sha(m.RUN/f['saved_copy'])==f['sha256']
    baseline={f['name']:f for f in frozen['frames']};frames=manifest['frames']
    cal=m.read(m.DATA/'calibration_clean.json');w,h=cal['width'],cal['height'];K=np.array(cal['K'])
    oldcal=m.read(m.BASE/'calibration_clean.json')
    assert cal['projection_version']==m.p.SDK and cal['crop_xy']==[7,7]
    assert np.array_equal(K,np.array(oldcal['K']))
    maps,valid=m.p.remap(cal['raw_color'],K,w,h);assert valid.all()
    stored=np.load(m.DATA/'rgb_remap.npz');assert np.array_equal(stored['map_xy'],maps)
    source_roles={r['rgb_message_id']:r['role'] for r in m.read(m.RUN/'prospective_checks.json')['source_frames']}
    assert len(source_roles)==4764
    for f in frames:
        b=baseline[f['name']]
        for key,newkey in [('rgb_message_id','message_id'),('depth_message_id','depth_message_id'),('rgb_timestamp_ns','timestamp_ns'),('depth_timestamp_ns','depth_timestamp_ns')]:assert b[key]==f[newkey]
        assert m.sha(m.BASE/'images'/f['name'])==b['rgb_sha256']
        assert m.sha(m.BASE/'aligned_depth_meters'/(f['name']+'.npz'))==b['depth_sha256']
        assert m.sha(m.DATA/'images'/f['name'])==f['rgb_sha256']
        assert m.sha(m.DATA/'aligned_depth_meters'/(f['name']+'.npz'))==f['depth_sha256']
        with np.load(m.DATA/'aligned_depth_meters'/(f['name']+'.npz')) as a, np.load(m.BASE/'aligned_depth_meters'/(f['name']+'.npz')) as b:
            assert np.array_equal(a['points'],b['points']) and np.array_equal(a['depths'],b['depths'])
            uv,z=a['points'],a['depths'];xy=np.rint(uv).astype(int)
            assert np.isfinite(uv).all() and np.isfinite(z).all() and (z>0).all()
            assert (xy>=0).all() and (xy[:,0]<w).all() and (xy[:,1]<h).all()
            assert len(np.unique(xy[:,1]*w+xy[:,0]))==len(z)
        with np.load(m.DATA/'projection_evidence'/(f['name']+'.npz')) as e:
            assert e['observed'].shape==(h,w) and np.array_equal(e['missing'],~e['observed'])
            assert e['observed'][xy[:,1],xy[:,0]].all()
            assert not (e['multiple_projected_samples'] & ~e['observed']).any()
    split=m.read(m.DATA/'reconstruction_split.json');train=set(split['train']);checks=set(split['val']);final=set(split['final']);oldval=set(split['historical_eval'])
    assert not (train&checks or train&final or train&oldval or checks&final)
    byname={f['name']:f for f in frames}
    assert all(source_roles[byname[n]['message_id']]=='fit_development' for n in train)
    review=m.read(m.RUN/'alignment_review/report.json');oldmaps,_=m.p.remap(cal['raw_color'],K,w,h,m.p.LEGACY)
    # Verify old and new remaps from the original RGB, plus saved unfilled depth
    # against the final implementation, including its serialized-border guard.
    with m.connect() as db:
        for r in review['views']:
            f=byname[r['frame']];assert source_roles[f['message_id']]!='final_prospective'
            raw,_=m.clean.capture.CDR(db.execute('select data from messages where id=?',(f['message_id'],)).fetchone()[0]).image()
            assert np.array_equal(cv2.remap(raw,oldmaps[:,:,0],oldmaps[:,:,1],cv2.INTER_LINEAR),np.array(Image.open(m.BASE/'images'/f['name'])))
            assert np.array_equal(cv2.remap(raw,maps[:,:,0],maps[:,:,1],cv2.INTER_LINEAR),np.array(Image.open(m.DATA/'images'/f['name'])))
            depth,_=m.clean.adapter.decode_depth(db.execute('select data from messages where id=?',(f['depth_message_id'],)).fetchone()[0])
            _,_,_,e=m.p.align_depth(depth,cal['depth_units'],cal['raw_depth'],cal['depth_to_color_R'],cal['depth_to_color_t_m'],K,w,h,filtered=False)
            with np.load(m.RUN/'alignment_review'/(f['name']+'.raw_depth.npz')) as a:
                assert np.array_equal(a['depth_m'],e['depth_m'])
    commands=[sys.executable,'-m','pytest','tests/test_realsense_projection.py','tests/test_realsense_cleaning.py',
        'tests/test_room_sdk_remap.py','tests/test_room_sdk_pose_comparison.py','tests/test_room_transition_profile.py',
        '--basetemp',str(destination/'pytest_verified_tmp'),'-q']
    proc=subprocess.run(commands,cwd=ROOT,text=True,capture_output=True)
    (destination/'tests.log').write_text(proc.stdout+proc.stderr,encoding='utf-8');assert proc.returncode==0
    files=['scripts/realsense_projection.py','scripts/clean-realsense-scene.py','scripts/bridge-clean-rgbd.py',
        'scripts/bridge-room-sdk-candidate.py','scripts/prepare-room-sdk-rectified-candidate.py',
        'scripts/run-db3-r0-r1.py','scripts/render-db3-r0-baseline.py','scripts/audit-db3-r1-metadata.py',
        'scripts/review-db3-r1-alignment.py','scripts/measure-db3-r1-reviewed-boundaries.py',
        'scripts/verify-db3-r0-r1.py','tests/test_realsense_projection.py']
    code=[]
    for name in files:
        dst=destination/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dst)
        code.append(dict(path=name,sha256=m.sha(dst)))
    artifacts=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=m.sha(p))
               for base in (m.RUN,m.DATA) for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts
               and 'pytest' not in str(p) and 'baseline' not in p.relative_to(base).parts]
    for row in m.read(m.RUN/'verification.json')['source_code']:
        assert m.sha(m.RUN/'source_snapshot'/row['path'])==row['sha256']
    m.write(destination/'verification.json',dict(source_sha256_unchanged=frozen['source_sha256'],
        baseline_artifacts_and_477_image_depth_hashes_unchanged=True,baseline_copy_hashes_match=True,
        candidate_frames=len(frames),candidate_rgb_depth_hashes_verified=True,unique_finite_positive_depth_samples=True,
        candidate_depth_arrays_identical_to_baseline=True,visibility_masks_consistent=True,
        crop_K_source_IDs_preserved=True,legacy_and_SDK_RGB_reproduced_from_raw_review_frames=len(review['views']),
        final_unfilled_depth_evidence_reproduced=True,split_counts={k:len(v) for k,v in [('train',train),('geometry_checks',checks),('final',final),('historical_eval',oldval)]},
        future_train_disjoint_from_every_reserved_set=True,final_blocks_visually_reviewed=False,
        test_command=commands,test_exit_code=proc.returncode,test_result=proc.stdout.splitlines()[-1],
        precise_physical_alignment_gate_pass=False,new_Gaussian_model_trained=False,
        source_code=code,artifact_hashes=artifacts,elapsed_seconds=time.time()-start))
    print('R0/R1 integrity PASS; physical gate remains PARTIAL',proc.stdout.splitlines()[-1],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output-dir',type=Path)
    main(parser.parse_args().output_dir)
