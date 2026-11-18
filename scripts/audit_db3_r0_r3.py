"""Verify source/baseline preservation, exclusions and actual R0--R3 exit gates."""
import importlib.util
import argparse
import hashlib
from pathlib import Path
import shutil
import time
import numpy as np
import pycolmap
import torch

s=importlib.util.spec_from_file_location('local',Path(__file__).with_name('db3_local_geometry.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def verify_pilot_inputs(out, bylocal):
    """Bind the delivered cameras/cloud/depth samples to the frozen fit evidence."""
    report=m.read(out/'initialization_report.json');geo=Path(report['geometry'])
    model=pycolmap.Reconstruction(out/'dataset/sparse')
    real=pycolmap.Reconstruction(geo/'metric_joint')
    expected={im.name:im.cam_from_world().matrix() for im in real.images.values()}
    for row in m.read(geo/'evaluation_localization.json')['frames']:
        if row['accepted']:expected[row['name']]=np.asarray(row['world_to_camera'])[:3]
    assert set(expected)=={im.name for im in model.images.values()}
    for im in model.images.values():
        np.testing.assert_allclose(im.cam_from_world().matrix(),expected[im.name],atol=1e-10,rtol=0)
        assert m.sha(out/'dataset/images'/im.name)==bylocal[im.name]['rgb_sha256']
    with np.load(out/'supported_initialization.npz') as cloud:
        ids=sorted(model.points3D)
        np.testing.assert_allclose([model.points3D[i].xyz for i in ids],cloud['xyz'],atol=1e-10,rtol=0)
        np.testing.assert_array_equal([model.points3D[i].color for i in ids],cloud['rgb'])
        assert all(model.points3D[i].track.length()==0 for i in ids)
    for row in report['stats']:
        name=row['name'];assert row['source_id']==bylocal[name]['rgb_message_id']
        with np.load(out/'dataset/evidence'/(name+'.npz')) as ev, np.load(out/'dataset/depth'/(name+'.npz')) as samples:
            chosen=np.random.default_rng(11010+row['source_id']).choice(len(ev['depths']),min(4096,len(ev['depths'])),replace=False)
            for key in ('points','depths','confidence'):np.testing.assert_array_equal(samples[key],ev[key][chosen])
            assert not m.check_tile(samples['points']).any()
    provenance=m.read(out/'camera_provenance.json');transform=np.asarray(provenance['normalization_transform'])
    scale=np.linalg.norm(transform[:3,0]);normalized=np.asarray(provenance['camera_to_world_normalized'])
    for name,C in zip(provenance['image_names'],normalized):
        W=np.eye(4);W[:3]=expected[name];want=transform@np.linalg.inv(W);want[:3,:3]/=scale
        np.testing.assert_allclose(C,want,atol=1e-6,rtol=1e-6)
        assert provenance['source_id_by_image'][name]==bylocal[name]['rgb_message_id']
    delivered=m.read(m.RUN/'local_delivery.json')['submaps'][out.parent.name]
    assert m.sha(out/'camera_provenance.json')==delivered['normalization_sha256']
    assert m.sha(m.ROOT/delivered['checkpoint'])==delivered['checkpoint_sha256']
    viewer=m.read(out/'viewer.json')
    assert viewer['checkpoint']==delivered['checkpoint'] and viewer['dataset']==delivered['dataset']
    return dict(matching_metric_cameras=len(expected),initializer_points=model.num_points3D(),
        exact_frozen_training_depth_files=len(report['stats']),normalized_camera_delivery_matches=True)


def main(output_dir=None):
    # A re-audit must not replace the code/evidence that produced an earlier run.
    destination=Path(output_dir) if output_dir else m.RUN/'reaudits'/time.strftime('%Y%m%d_%H%M%S')
    destination.mkdir(parents=True,exist_ok=False)
    start=time.time();frozen=m.read(m.R1/'baseline_manifest.json');source=Path(m.read(m.DATA/'manifest.json')['source'])
    print('VERIFY full raw SHA256',flush=True);assert m.sha(source)==frozen['source_sha256']
    for f in frozen['files']:
        assert m.sha(m.ROOT/f['source'])==f['sha256'];assert m.sha(m.R1/f['saved_copy'])==f['sha256']
    for f in frozen['frames']:
        original=m.ROOT/'data/realsense_20261030_134523_clean'
        assert m.sha(original/'images'/f['name'])==f['rgb_sha256']
        assert m.sha(original/'aligned_depth_meters'/(f['name']+'.npz'))==f['depth_sha256']
    inputs=m.read(m.INPUT/'capture_manifest.json')
    for f in inputs['frames']:
        assert m.sha(m.INPUT/'images'/f['name'])==f['rgb_sha256']
        assert m.sha(m.INPUT/'aligned_depth_meters'/(f['name']+'.npz'))==f['depth_sha256']
    reservations={r['rgb_message_id']:r['role'] for r in m.read(m.R1/'prospective_checks.json')['source_frames']}
    original_split=m.read(m.INPUT/'reconstruction_split.json');byname={f['name']:f for f in inputs['frames']}
    historical={byname[n]['message_id'] for n in original_split['historical_eval']}
    rows=m.read(m.DATA/'manifest.json')['frames'];bylocal={f['name']:f for f in rows}
    for f in rows:
        assert m.sha(m.DATA/'images'/f['name'])==f['rgb_sha256'];assert m.sha(m.DATA/'depth'/(f['name']+'.npz'))==f['depth_sha256']
        assert reservations[f['rgb_message_id']]=='fit_development'
        if f['role']=='fit':assert f['rgb_message_id'] not in historical
    local_checks={};paired={};pilot_inputs={}
    for region in ('matte','exhibit'):
        for rate in (3,10):
            root=m.RUN/region/f'rate{rate}';model=pycolmap.Reconstruction(root/'metric_joint')
            assert model.num_points3D()>0 and model.compute_mean_track_length()>2
            requested=m.read(root/'input.json')['frames'];fitnames={f['name'] for f in requested}
            assert all(im.name in fitnames and bylocal[im.name]['rgb_message_id'] not in historical for im in model.images.values())
            for im in model.images.values():
                xy=np.array([pt.xy for pt in im.points2D])
                assert not m.check_tile(xy).any(),'Spatial check entered map'
            check=m.read(root/'validation.json');local_checks[f'{region}_rate{rate}']=dict(pass_gate=check['local_consistency_pass'],registered_fraction=check['registered_fraction'])
        out=m.RUN/region/'pilot';split=m.read(out/'dataset/experiment_split.json')
        initialization=m.read(out/'initialization_report.json')
        assert m.sha(out/'supported_initialization.npz')==initialization['initialization_sha256']
        assert m.sha(Path(initialization['geometry'])/'validation.json')==initialization['geometry_validation_sha256']
        assert not set(split['train'])&set(split['val'])
        assert all(bylocal[n]['original_train'] and bylocal[n]['role']=='fit' for n in split['train'])
        assert all(bylocal[n]['role']=='eval' for n in split['val'])
        pilot_inputs[region]=verify_pilot_inputs(out,bylocal)
        record=m.read(out/'paired_evaluation.json');assert record['gates']['equal_initialization_and_replay']
        exps=[]
        for arm in ('off','current','confidence'):
            arm_directory='current_safe' if arm=='current' and (out/'current_safe').exists() else arm
            run=out/arm_directory;exp=m.read(run/'experiment.json');state=torch.load(run/'ckpts/final_post_update.pt',map_location='cpu',weights_only=True)['splats']
            assert exp['completed_updates']==7000 and len(exp['camera_sequence'])==7000
            assert hashlib.sha256(np.asarray(exp['camera_sequence'],dtype='<i8').tobytes()).hexdigest()==exp['camera_sequence_sha256']
            assert exp['camera_sequence']==m.read(out/'camera_sequence.json')
            assert exp['final_count']==len(state['means'])<=180000
            assert all(torch.isfinite(v).all() for v in state.values())
            assert m.sha(run/'ckpts/final_post_update.pt')==record['results'][arm]['checkpoint_sha256']
            exps.append(exp)
        assert len({e['initial_sha256'] for e in exps})==1 and len({e['camera_sequence_sha256'] for e in exps})==1
        paired[region]=dict(policy_ablation_complete=True,confidence_policy_gate=record['quantitative_promotion_pass'],
                            pipeline_gates={a:v['quantitative_pass'] for a,v in record['pipeline_against_baseline'].items()})
    followup=None
    extra=m.RUN/'exhibit/pilot/confidence_supported_15k'
    if (extra/'experiment.json').exists():
        exp=m.read(extra/'experiment.json');ref=m.read(extra.parent/'confidence/experiment.json')
        assert exp['completed_updates']==15000 and len(exp['camera_sequence'])==15000
        assert hashlib.sha256(np.asarray(exp['camera_sequence'],dtype='<i8').tobytes()).hexdigest()==exp['camera_sequence_sha256']
        assert exp['initial_sha256']==ref['initial_sha256'] and exp['camera_sequence'][:7000]==ref['camera_sequence']
        tensors=torch.load(extra/'ckpts/final_post_update.pt',map_location='cpu',weights_only=True)['splats']
        assert all(torch.isfinite(t).all() for t in tensors.values()) and len(tensors['means'])<=180000
        checks=m.read(extra.parent/'selected_policy_confirmation_result_supported_15k.json')
        evaluated=m.read(extra.parent/'paired_evaluation_supported_15k.json')
        assert m.sha(extra/'ckpts/final_post_update.pt')==evaluated['results']['confidence']['checkpoint_sha256']
        followup=dict(steps=15000,finite=True,initialization_identical=True,original_replay_prefix_identical=True,
                      coarse_gate=checks['coarse_quantitative_pass'],structure_gate=checks['manual_structure_quantitative_pass'],
                      confidence_policy_superiority_claim=False)
    transitions=m.read(m.RUN/'transitions_revisits_v2/report.json')
    assert all(reservations[i]=='fit_development' and i not in historical for i in transitions['source_ids_loaded'])
    names=['scripts/db3_local_geometry.py','scripts/db3_validate_local.py','scripts/db3_local_pilot.py',
        'scripts/db3_evaluate_pilot.py','scripts/db3_transition_revisit.py','scripts/run-db3-local-pilot.ps1','scripts/audit_db3_r0_r3.py',
        'scripts/db3_confirm_local.py','scripts/package_db3_local.py','scripts/plot_db3_local_trajectories.py',
        'scripts/write_db3_campaign_summary.py',
        'scripts/diagnose_db3_submap_support.py',
        'scripts/db3_acceptance.py','scripts/diagnose_db3_support_stages.py',
        'examples/simple_trainer.py','examples/datasets/colmap.py','gsplat/losses.py',
        'tests/test_db3_local_geometry.py','tests/test_sensor_depth_confidence.py','tests/test_db3_acceptance.py']
    snapshots=[]
    for name in names:
        dst=destination/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(m.ROOT/name,dst)
        historical=m.RUN/'source_snapshot'/name
        snapshots.append(dict(path=name,sha256=m.sha(dst),historical_snapshot_sha256=m.sha(historical) if historical.exists() else None))
    # Verify the historical snapshots against their historical audit, not today's code.
    for row in m.read(m.RUN/'integrity_audit.json')['source_code']:
        assert m.sha(m.RUN/'source_snapshot'/row['path'])==row['sha256']
    m.write(destination/'integrity_audit.json',dict(raw_sha256_unchanged=frozen['source_sha256'],baseline_and_R1_inputs_unchanged=True,
        reserved_native_and_historical_eval_excluded_from_all_fits=True,spatial_check_features_absent_from_mapping=True,
        actual_real_tracks_and_metric_depth_BA=True,six_completed_7000_update_pilots=True,selected_six_checkpoints_finite=True,
        equal_initializations_and_camera_replays_within_each_region=True,pilot_inputs=pilot_inputs,local_checks=local_checks,paired=paired,additional_supported_15k_followup=followup,
        R0='complete; preserved',R1='numerical/input/timing work complete; precise physical calibration partial',
        R2='local consistency gates and concrete D8 separate-submap experiment complete; global geometry unaccepted',
        R3='See paired, selected-policy confirmation and visual review; do not infer success from completed jobs',
        all_R0_R3_acceptance_gates_pass=False,source_code=snapshots,elapsed_seconds=time.time()-start))
    print('INTEGRITY PASS; scientific phase acceptance is reported separately',time.time()-start,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir',type=Path)
    main(parser.parse_args().output_dir)
