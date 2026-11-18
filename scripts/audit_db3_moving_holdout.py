"""Reproduce exclusion and residual evidence without changing either experiment."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

spec=importlib.util.spec_from_file_location('moving',Path(__file__).with_name('db3_moving_holdout.py'))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
r,m=p.r,p.m
RUN=r.ROOT/'results/db3_moving_holdout_20261111'
FOLLOW=r.ROOT/'results/db3_tracked_fit_20261111'


def independent_errors(features,checks,poses,K,saved):
    for row,target in zip(checks,saved):
        a,b=row['cameras'];T=poses[b]@np.linalg.inv(poses[a]);back=np.linalg.inv(T)
        metric=[];pixels=[]
        for _,ia,ib in row['refs']:
            ua,ub=features[a]['uv'][ia],features[b]['uv'][ib]
            xa,xb=features[a]['xyz'][ia],features[b]['xyz'][ib]
            qa=T[:3,:3]@xa+T[:3,3];qb=back[:3,:3]@xb+back[:3,3]
            project=lambda q: (K@q)[:2]/q[2]
            metric.append(np.linalg.norm(qa-xb));pixels.append(max(np.linalg.norm(project(qa)-ub),np.linalg.norm(project(qb)-ua)))
        assert np.allclose(metric,target['errors_m'],atol=1e-10)
        assert np.allclose(pixels,target['errors_px'],atol=1e-8)


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=m.read(RUN/'protocol.json');K=np.array(m.read(m.INPUT/'calibration_clean.json')['K'])
    for path,h in protocol['source_hashes'].items():assert m.sha(r.ROOT/path)==h
    reservations={f['rgb_message_id']:f for f in m.read(m.R1/'prospective_checks.json')['source_frames']}
    split=set(m.read(m.INPUT/'reconstruction_split.json')['historical_eval'])
    historical={f['message_id'] for f in m.read(m.INPUT/'capture_manifest.json')['frames'] if f['name'] in split}
    counts=dict(inputs=0,source_snapshots=0,checked_poses=0,tracked_fit_observations=0,moving_check_observations=0)
    summaries=[]
    for root in (RUN,FOLLOW):
        for name,h in m.read(root/'source_hashes.json').items():
            assert m.sha(root/'source_snapshot'/name)==h;counts['source_snapshots']+=1
        assert not m.read(root/'result.json')['training_authorized']
    for case in protocol['cases']:
        dest=RUN/case['label'];follow=FOLLOW/case['label'];fs=[]
        rows=m.read(dest/'inputs.json');assert [f['source_id'] for f in rows]==case['source_ids']
        for f in rows:
            assert r.route.eligible(dict(message_id=f['source_id']),reservations,historical)
            base=dest/'inputs'/str(f['source_id'])
            assert m.sha(base.with_suffix('.npz'))==f['feature_sha256']
            assert m.sha(base.with_suffix('.png'))==f['image_sha256']
            with np.load(base.with_suffix('.npz')) as data:fs.append({k:data[k].copy() for k in data.files})
            assert np.array_equal(fs[-1]['check'],m.check_tile(fs[-1]['uv']));counts['inputs']+=1
        with np.load(dest/'moving_checks.npz') as data:
            uv=data['uv'];reserved=[data[f'reserved_{c}'] for c in range(len(fs))]
        assert m.check_tile(uv[0]).all()
        for c,xy in enumerate(uv):assert (cKDTree(reserved[c]).query(xy)[0]<1e-5).all()
        counts['moving_check_observations']+=int(np.prod(uv.shape[:2]))
        trackdata=m.read(dest/'tracks.json')
        blocked=p.propagate_exclusions(fs,trackdata['links'],reserved)
        with np.load(dest/'exclusions.npz') as data:
            assert all(np.array_equal(b,data[f'camera_{c}']) for c,b in enumerate(blocked))
        for c,b in enumerate(blocked):
            assert b[fs[c]['check']].all()
            if len(reserved[c]):assert (cKDTree(reserved[c]).query(fs[c]['uv'][~b])[0]>12).all()
        for name,h in m.read(dest/'pre_fit_seal.json').items():assert m.sha(dest/name)==h
        checkfs=[dict(uv=xy,depth=f['depth']) for xy,f in zip(uv,fs)];r.observations(checkfs,K);r.observations(fs,K)
        checks=r.fixed_checks(checkfs,[[(c,i) for c in range(len(fs))] for i in range(uv.shape[1])],case['check_pairs'])
        assert checks==m.read(dest/'frozen_checks.json')
        original=m.read(dest/'original_checks.json')
        with np.load(follow/'fit_features.npz') as data:
            fituv=[data[f'uv_{c}'] for c in range(len(fs))]
        for c,xy in enumerate(fituv):
            assert not m.check_tile(xy).any()
            if len(reserved[c]):assert (cKDTree(reserved[c]).query(xy)[0]>12).all()
            counts['tracked_fit_observations']+=len(xy)
        for folder in (dest,follow):
            result=m.read(folder/'result.json');assert result['complete']
            allowed_tracks=m.read(folder/'tracks.json')['tracks']
            allowed_tracks=allowed_tracks['fit'] if isinstance(allowed_tracks,dict) else allowed_tracks
            allowed={tuple(tuple(node) for node in tr) for tr in allowed_tracks}
            for mode,summary in result['candidates'].items():
                optimization=m.read(folder/f'{mode}_optimization.json')
                assert all(tuple(tuple(node) for node in tr) in allowed for tr in optimization['fitting_tracks'])
                if folder==dest:assert all(not blocked[c][i] for tr in optimization['fitting_tracks'] for c,i in tr)
                poses=np.array(m.read(folder/f'{mode}_poses.json'))
                assert np.isfinite(poses).all() and np.allclose(poses[0],np.eye(4))
                for T in poses:
                    assert np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-8)
                    assert abs(np.linalg.det(T[:3,:3])-1)<1e-8
                counts['checked_poses']+=len(poses)
                independent_errors(checkfs,checks,poses,K,summary['checks'])
                independent_errors(fs,original,poses,K,summary['original_checks'])
                assert not summary['pose_improvement_pass']
            assert not result['training_authorized'] and not result['whole_scene_accepted']
        summary=m.read(follow/'result.json')
        summaries.append(dict(label=case['label'],validation_tracks=len(uv[0]),depth_check_counts=[len(x['refs']) for x in checks],
                              tracked_fit_count=len(m.read(follow/'tracks.json')['tracks']),
                              local_consistency={key:v['local_consistency_pass'] for key,v in summary['candidates'].items()}))
        if case['label']=='target':plot_target(dest,follow,case,uv,checkfs,checks,out)
    manifest=m.read(r.R4/'delivery/manifest.json')
    for label in ('whole_route','matte'):
        assert m.sha(r.ROOT/manifest[label]['checkpoint'])==manifest[label]['checkpoint_sha256']
    hashes={str(f.relative_to(root.parent)):m.sha(f) for root in (RUN,FOLLOW) for f in root.rglob('*') if f.is_file()}
    m.write(out/'artifact_hashes.json',hashes)
    m.write(out/'audit.json',dict(passed=True,counts=counts,cases=summaries,artifacts_sealed=len(hashes),
        original_models_unchanged=True,reservations_and_fit_exclusions_verified=True,residuals_independently_reproduced=True,
        raw_full_hash_recomputed=False,physical_accuracy_established=False))
    print('MOVING HOLDOUT AUDIT PASS',counts,'artifacts',len(hashes),flush=True)


def plot_target(dest,follow,case,uv,checkfs,checks,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=m.read(follow/'result.json');baseline=result['old_chain_checks'][0]
    selected=result['candidates']['pixel_depth']['checks'][0]
    labels=['Old chain','Tracked pixel + depth','Tracked metric']
    values=[baseline,selected,result['candidates']['metric']['checks'][0]]
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    axs[0].bar(range(3),[x['separation_m'][1]*100 for x in values]);axs[0].axhline(12,color='red',linestyle='--',label='12 cm limit')
    axs[0].set_ylabel('Endpoint p90 separation (cm)');axs[0].set_ylim(0,13);axs[0].legend()
    axs[1].bar(range(3),[x['symmetric_reprojection_px'][1] for x in values]);axs[1].axhline(4,color='red',linestyle='--',label='4 px limit')
    axs[1].set_ylabel('Endpoint p90 reprojection (px)');axs[1].legend()
    for ax in axs:ax.set_xticks(range(3),labels,rotation=20,ha='right')
    fig.suptitle('Difficult window: 58 fixed endpoint checks | depth consistency passes, image alignment fails')
    fig.tight_layout();fig.savefig(out/'target_geometry_comparison.png',dpi=160);plt.close(fig)
    images=[np.array(Image.open(dest/'inputs'/f'{case["source_ids"][c]}.png').convert('RGB')) for c in (0,len(case['source_ids'])-1)]
    worst=np.argsort(-np.array(selected['errors_px']))[:6];refs=checks[0]['refs']
    fig,axs=plt.subplots(6,2,figsize=(8,14))
    for row,j in enumerate(worst):
        _,ia,ib=refs[j]
        for col,(im,xy) in enumerate(zip(images,[uv[0,ia],uv[-1,ib]])):
            axs[row,col].imshow(im);axs[row,col].scatter(*xy,s=90,facecolors='none',edgecolors='red')
            axs[row,col].set_xlim(xy[0]-40,xy[0]+40);axs[row,col].set_ylim(xy[1]+40,xy[1]-40);axs[row,col].axis('off')
            axs[row,col].set_title(f'Check {j}: {"first" if col==0 else "last"} | {selected["errors_px"][j]:.2f} px')
    fig.suptitle('Target: six worst unchanged moving-check correspondences\nInspection only; none removed from acceptance')
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'target_check_crops.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    a=argparse.ArgumentParser(__doc__);a.add_argument('--output-dir',type=Path,required=True);main(a.parse_args().output_dir)
