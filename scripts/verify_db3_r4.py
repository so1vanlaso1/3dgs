"""Independent R4 artifact, reservation, dense-depth and chain audit."""
import argparse
import importlib.util
from pathlib import Path
import sqlite3
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('route',ROOT/'scripts/db3_r4_route.py')
r=importlib.util.module_from_spec(s);s.loader.exec_module(r);m=r.m


def verify(out):
    target=out/'final_verification.json'
    if target.exists():raise FileExistsError(target)
    start=time.time();protocol=m.read(out/'protocol.json')
    report=m.read(out/'selected_edges.json');edges=report['edges'];retries=m.read(out/'native_retries.json')['retries']
    expected=protocol['candidate_boundaries']
    assert len(expected)==len(set(expected)) and {x['boundary'] for x in edges}==set(expected) and len(edges)==len(expected)
    failed={e['boundary'] for e in edges if not e['accepted']}
    assert {x['boundary'] for x in retries}==failed and len(retries)==len(failed)
    frames=m.read(m.INPUT/'capture_manifest.json')['frames'];native=m.read(m.R1/'prospective_checks.json')['source_frames']
    byid={f['rgb_message_id']:f for f in native};hist_names=set(m.read(m.INPUT/'reconstruction_split.json')['historical_eval'])
    historical={f['message_id'] for f in frames if f['name'] in hist_names}
    allow=lambda mid:r.eligible(dict(message_id=mid),byid,historical)
    for edge in edges:
        idx=edge['boundary'];assert edge['source_ids']==[f['message_id'] for f in frames[idx:idx+2]]
        assert all(allow(mid) for mid in edge['source_ids'])
    for row in retries:
        a,b=frames[row['boundary']:row['boundary']+2]
        ids=[f['rgb_message_id'] for f in native if a['timestamp_ns']<=f['timestamp_ns']<=b['timestamp_ns']]
        assert row['native_source_ids']==ids and all(allow(mid) for mid in ids)
        assert row['total_native_edges']==len(ids)-1
        T=np.eye(4)
        for i,edge in enumerate(row['edges']):
            assert edge['source_ids']==ids[i:i+2]
            if edge['accepted']:T=np.array(edge['transform'])@T
            else:assert i==len(row['edges'])-1
        complete=len(row['edges'])==len(ids)-1 and all(e['accepted'] for e in row['edges'])
        assert complete==(row['endpoint_check'] is not None)
        assert row['accepted']==bool(complete and row['endpoint_check']['passed'])
        if complete:assert np.allclose(T,row['composed_transform'],atol=1e-12,rtol=0)
    inputs=m.read(out/'inputs_manifest.json');assert inputs['all_loaded_frames_are_fit_only']
    assert len(inputs['frames'])==len({f['source_id'] for f in inputs['frames']})
    for f in inputs['frames']:
        mid=f['source_id'];assert allow(mid)
        path=out/'inputs'/f'{mid}.npz';assert m.sha(path)==f['sha256']
        with np.load(path) as a:
            assert np.array_equal(a['check'],m.check_tile(a['uv']))
            assert len(a['uv'])==len(a['desc'])==len(a['check'])
            assert np.isfinite(a['uv']).all() and np.isfinite(a['desc']).all() and np.isfinite(a['depth']).all()
    # Stratified raw-source reprojection of both selected and newly decoded inputs.
    cal=m.read(m.INPUT/'calibration_clean.json');K=np.array(cal['K'])
    ordered=sorted(inputs['frames'],key=lambda x:x['seconds']);chosen=[ordered[i] for i in np.unique(np.linspace(0,len(ordered)-1,16).astype(int))]
    raw=Path(m.read(m.INPUT/'capture_manifest.json')['source']).resolve()
    with sqlite3.connect(raw.as_uri()+'?mode=ro',uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        for f in chosen:
            z,_=m.clean.adapter.decode_depth(db.execute('select data from messages where id=?',(f['depth_source_id'],)).fetchone()[0])
            *_,e=m.projection.align_depth(z,cal['depth_units'],cal['raw_depth'],cal['depth_to_color_R'],cal['depth_to_color_t_m'],K,cal['width'],cal['height'])
            with np.load(out/'inputs'/f'{f["source_id"]}.npz') as a:assert np.array_equal(a['depth'],e['depth_m'])
    for name,sha in inputs['sources'].items():assert m.sha(ROOT/name)==sha
    for name,sha in m.read(out/'delivery/artifact_hashes.json').items():assert m.sha(out/'delivery'/name)==sha
    cameras=m.read(out/'delivery/baseline_camera_provenance.json')
    frozen_cameras=m.read(m.R1/'fixed_views/cameras.json')
    assert cameras['image_names']==frozen_cameras['all_image_names']
    assert np.allclose(cameras['normalization_transform'],frozen_cameras['normalization_transform'],atol=1e-10,rtol=0)
    assert np.allclose(cameras['camera_to_world_normalized'],frozen_cameras['all_camera_to_world_normalized'],atol=1e-10,rtol=0)
    for sample in cameras['replay_samples']:
        rotation=np.array(sample['rotation_matrix'])
        assert np.allclose(rotation.T@rotation,np.eye(3),atol=1e-9) and np.linalg.det(rotation)>0
        assert 0<sample['vertical_fov_radians']<np.pi
    decision=m.read(out/'phase_decision.json');summary=m.read(out/'route_summary.json')['counts']
    assert summary['candidate_boundaries']==len(edges) and summary['native_recovered']==sum(x['accepted'] for x in retries)
    assert summary['fitting_inputs_loaded']==len(inputs['frames'])
    assert len(edges)+len(report['excluded_boundaries'])==len(frames)-1
    assert not decision['whole_scene_scientifically_accepted'] and not decision['new_global_model']
    assert not decision['whole_scene_training_authorized_by_evidence']
    assert m.read(out/'integrity/integrity_audit.json')['raw_sha256_unchanged']==protocol['source_sha256']
    for name,sha in m.read(out/'source_manifest.json').items():assert m.sha(out/'source_snapshot'/name)==sha
    endings=[x['endpoint_check'] for x in retries if x['endpoint_check'] is not None]
    m.write(target,dict(passed=True,selected_edge_roster_complete=True,all_eligible_failures_retried=True,
        all_inputs_hash_valid_and_fit_only=True,all_fit_check_tile_masks_reproduced=True,
        native_chain_composition_verified=True,delivery_artifact_hashes_match=True,
        baseline_cameras_and_normalization_match_frozen_R0=True,
        stratified_raw_depth_reproductions=len(chosen),reproduced_source_ids=[f['source_id'] for f in chosen],
        complete_native_chains=len(endings),endpoint_insufficient_matches=sum(x['matches']<20 for x in endings),
        endpoint_failed_with_sufficient_matches=sum(x['matches']>=20 and not x['passed'] for x in endings),
        elapsed_seconds=time.time()-start,
        caveat='Integrity and consistency audit, not independent physical/global reconstruction acceptance.'))
    print('R4 INDEPENDENT VERIFICATION PASS',len(inputs['frames']),'inputs',len(retries),'native retries',flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(__doc__);a.add_argument('--output-dir',type=Path,required=True)
    verify(a.parse_args().output_dir)
