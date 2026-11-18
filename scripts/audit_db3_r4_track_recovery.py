"""Verify bounded recovery artifacts and expose original endpoint outliers.

Read-only on both experiments; output is a new audit namespace. Does not remove
or reclassify any failed original check.
"""
import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

spec=importlib.util.spec_from_file_location('tracks',Path(__file__).with_name('db3_r4_track_recovery.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
m=r.m
RUN=r.ROOT/'results/db3_r4_track_recovery_20261111'
FOLLOW=r.ROOT/'results/db3_r4_patch_followup_20261111'


def audit(out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=m.read(RUN/'protocol.json')
    cal=m.read(m.INPUT/'calibration_clean.json');K=np.array(cal['K'])
    for name,expected in protocol['source_hashes'].items():assert m.sha(r.ROOT/name)==expected
    reservations={f['rgb_message_id']:f for f in m.read(m.R1/'prospective_checks.json')['source_frames']}
    names=set(m.read(m.INPUT/'reconstruction_split.json')['historical_eval'])
    historical={f['message_id'] for f in m.read(m.INPUT/'capture_manifest.json')['frames'] if f['name'] in names}
    counters=dict(input_hashes=0,fit_observations=0,check_observations=0,patch_observations=0,source_snapshots=0)
    summaries=[]
    for root in (RUN,FOLLOW):
        for name,expected in m.read(root/'source_hashes.json').items():
            assert m.sha(root/'source_snapshot'/name)==expected
            counters['source_snapshots']+=1
        assert not m.read(root/'result.json')['training_authorized']
    for case in protocol['cases']:
        prior=RUN/case['label'];follow=FOLLOW/case['label'];features=[]
        inputs=m.read(prior/'inputs.json')
        assert [f['source_id'] for f in inputs]==case['source_ids']
        for f in inputs:
            assert r.route.eligible(dict(message_id=f['source_id']),reservations,historical)
            path=prior/'inputs'/f'{f["source_id"]}.npz';assert m.sha(path)==f['sha256']
            with np.load(path) as data:features.append({k:data[k].copy() for k in data.files})
            assert np.array_equal(features[-1]['check'],m.check_tile(features[-1]['uv']))
            counters['input_hashes']+=1
        tracks=m.read(prior/'tracks.json')
        reproduced,rejected=r.components(tracks['links'],features)
        assert {k:[[list(x) for x in tr] for tr in v] for k,v in reproduced.items()}==tracks['tracks']
        assert rejected==tracks['rejected']
        fit_nodes={tuple(obs) for tr in tracks['tracks']['fit'] for obs in tr}
        check_nodes={tuple(obs) for tr in tracks['tracks']['check'] for obs in tr}
        assert not fit_nodes & check_nodes
        assert all(not features[c]['check'][i] for c,i in fit_nodes)
        assert all(features[c]['check'][i] for c,i in check_nodes)
        counters['fit_observations']+=len(fit_nodes);counters['check_observations']+=len(check_nodes)
        r.observations(features,K)
        checks=r.fixed_checks(features,tracks['tracks']['check'],case['check_pairs'])
        assert checks==m.read(prior/'frozen_checks.json')
        initial=np.array(m.read(prior/'initial_poses.json'))
        for path in (prior,follow):
            result=m.read(path/'result.json');ba=m.read(path/'optimization.json')
            poses=np.array(m.read(path/'optimized_poses.json'))
            assert np.isfinite(poses).all() and np.allclose(poses[0],np.eye(4))
            for T in poses:
                assert np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-8)
                assert abs(np.linalg.det(T[:3,:3])-1)<1e-8
            assert all(tuple(obs) in fit_nodes for tr in ba['fitting_tracks'] for obs in tr)
            measured=r.evaluate(features,checks,poses,K)
            saved=result['ba_on_new_checks'] if path==prior else result['ba_sift_checks']
            assert measured==saved
            # Independently recompute metric separations from the saved matrices.
            for row,saved_row in zip(checks,measured):
                a,b=row['cameras'];T=poses[b]@np.linalg.inv(poses[a]);errors=[]
                for _,ia,ib in row['refs']:
                    pa=features[a]['xyz'][ia];pb=features[b]['xyz'][ib]
                    errors.append(float(np.linalg.norm(T[:3,:3]@pa+T[:3,3]-pb)))
                assert np.allclose(errors,saved_row['errors_m'],atol=1e-12)
            assert not result['training_authorized'] and not result['whole_scene_accepted']
        with np.load(follow/'patch_tracks.npz') as data:uv=data['uv']
        assert all(m.check_tile(xy).all() for xy in uv)
        counters['patch_observations']+=int(np.prod(uv.shape[:2]))
        patch_features=[dict(uv=xy,depth=f['depth']) for xy,f in zip(uv,features)]
        r.observations(patch_features,K)
        patch_tracks=[[(c,i) for c in range(len(features))] for i in range(uv.shape[1])]
        patch_checks=r.fixed_checks(patch_features,patch_tracks,case['check_pairs'])
        assert patch_checks==m.read(follow/'frozen_patch_checks.json')
        second=m.read(follow/'result.json');poses=np.array(m.read(follow/'optimized_poses.json'))
        assert r.evaluate(patch_features,patch_checks,poses,K)==second['ba_patch_checks']
        summaries.append(dict(label=case['label'],patch_counts=[len(x['refs']) for x in patch_checks],
                              source_ids=case['source_ids'],candidate_pass=second['bounded_candidate_pass']))
        if case['boundary']==449:inspect_outliers(prior,features,initial,K,out)
    manifest=m.read(r.R4/'delivery/manifest.json')
    for name in ('whole_route','matte'):
        assert m.sha(r.ROOT/manifest[name]['checkpoint'])==manifest[name]['checkpoint_sha256']
    hashes={str(p.relative_to(root.parent)):m.sha(p) for root in (RUN,FOLLOW) for p in root.rglob('*') if p.is_file()}
    m.write(out/'experiment_artifact_hashes.json',hashes)
    m.write(out/'audit.json',dict(passed=True,counters=counters,cases=summaries,
        selected_baseline_and_matte_hashes_unchanged=True,all_reserved_identities_excluded=True,
        fit_check_components_disjoint=True,check_counts_and_residuals_reproduced=True,
        experiment_artifacts_sealed=len(hashes),raw_full_hash_rerun=False,
        caveat='Integrity and local consistency verification, not physical accuracy or a new reconstruction.'))
    print('AUDIT PASS',counters,'sealed',len(hashes),'artifacts',flush=True)


def inspect_outliers(prior,features,poses,K,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a,b=features[0],features[-1]
    ia=np.nonzero(a['check'])[0];ib=np.nonzero(b['check'])[0]
    pairs=m.mutual(a['desc'][ia],b['desc'][ib]);ia,ib=ia[pairs[:,0]],ib[pairs[:,1]]
    valid=a['valid'][ia]&b['valid'][ib];ia,ib=ia[valid],ib[valid]
    T=poses[-1];errors=np.linalg.norm(r.transform_points(T,a['xyz'][ia])-b['xyz'][ib],axis=1)
    images=[np.array(Image.open(prior/f'camera_{i:02d}.png').convert('RGB')) for i in (0,len(features)-1)]
    rows=[]
    for j,(x,y,err) in enumerate(zip(ia,ib,errors)):
        rows.append(dict(match_index=j,uv_a=a['uv'][x].tolist(),uv_b=b['uv'][y].tolist(),
                         depth_a_m=float(a['z'][x]),depth_b_m=float(b['z'][y]),chain_separation_m=float(err)))
    m.write(out/'boundary_449_original_matches.json',rows)
    worst=np.argsort(-errors)[:6]
    fig,axes=plt.subplots(6,2,figsize=(8,14))
    for row,j in enumerate(worst):
        for col,(im,uv) in enumerate(zip(images,[a['uv'][ia[j]],b['uv'][ib[j]]])):
            axes[row,col].imshow(im)
            axes[row,col].scatter(*uv,s=90,facecolors='none',edgecolors='red',linewidths=1.5)
            axes[row,col].set_xlim(uv[0]-45,uv[0]+45);axes[row,col].set_ylim(uv[1]+45,uv[1]-45)
            axes[row,col].set_title(f'Match {j}: {"first" if col==0 else "last"} | error {errors[j]:.3f} m',fontsize=10)
            axes[row,col].axis('off')
    fig.suptitle('Boundary 449: original check correspondences, worst six\nDiagnostic crops; no match removed and no acceptance changed',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'boundary_449_original_match_crops.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output-dir',type=Path,required=True)
    audit(p.parse_args().output_dir)
