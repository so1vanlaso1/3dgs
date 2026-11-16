"""Exposed follow-up: independent-of-pose patch tracks and fixed BA budget.

No original reservation/gate is relaxed. Patch tracks stay inside existing check
tiles in every frame, so their observations cannot enter the existing fitter.
"""
import argparse
import importlib.util
from pathlib import Path
import shutil
import sqlite3

import cv2
import numpy as np

spec=importlib.util.spec_from_file_location('tracks',Path(__file__).with_name('db3_r4_track_recovery.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
m=r.m
PARENT=r.ROOT/'results/db3_r4_track_recovery_20261111'


def patch_tracks(images):
    """Track corners forward/backward; all acceptance tests are pose-independent."""
    gray=[cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) for im in images]
    h,w=gray[0].shape
    yy,xx=np.mgrid[:h,:w]
    mask=m.check_tile(np.c_[xx.ravel(),yy.ravel()]).reshape(h,w)
    mask[:12]=False;mask[-12:]=False;mask[:,:12]=False;mask[:,-12:]=False
    corners=cv2.goodFeaturesToTrack(gray[0],maxCorners=3000,qualityLevel=.005,minDistance=7,
                                   mask=(mask*255).astype('u1'),blockSize=7)
    if corners is None:return np.empty((len(images),0,2)),dict(seed_corners=0,surviving_tracks=0)
    points=corners.reshape(-1,2)
    alive=np.ones(len(points),bool);tracks=[points.copy()]
    stages=[]
    settings=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
    for j in range(1,len(gray)):
        following,fs,_=cv2.calcOpticalFlowPyrLK(gray[j-1],gray[j],points[:,None].astype('f4'),None,**settings)
        back,bs,_=cv2.calcOpticalFlowPyrLK(gray[j],gray[j-1],following,None,**settings)
        following=following.reshape(-1,2);back=back.reshape(-1,2)
        good=(fs.ravel()>0)&(bs.ravel()>0)&np.isfinite(following).all(1)&(np.linalg.norm(back-points,axis=1)<.75)
        good &= (following[:,0]>=12)&(following[:,0]<w-12)&(following[:,1]>=12)&(following[:,1]<h-12)
        good &= m.check_tile(following)
        alive &= good
        # Normalize brightness/contrast in local patches; no fitted pose/residual involved.
        for k in np.nonzero(alive)[0]:
            a=cv2.getRectSubPix(gray[j-1],(11,11),tuple(points[k])).astype(float).ravel()
            b=cv2.getRectSubPix(gray[j],(11,11),tuple(following[k])).astype(float).ravel()
            a-=a.mean();b-=b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
            if den<=1e-6 or np.dot(a,b)/den<.85:alive[k]=False
        tracks.append(following.copy());points=following
        stages.append(dict(camera=j,survivors=int(alive.sum())))
    tracks=np.array(tracks)[:,alive]
    # Deduplicate collapsed tracks without any geometry-based decision.
    keep=[]
    for i in range(tracks.shape[1]):
        if all(np.linalg.norm(tracks[-1,i]-tracks[-1,k])>=3 for k in keep):keep.append(i)
    tracks=tracks[:,keep]
    return tracks,dict(seed_corners=len(corners),surviving_tracks=len(keep),stages=stages)


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    parent=m.read(PARENT/'protocol.json')
    protocol=dict(date='2026-11-11',timezone='Asia/Saigon',cases=parent['cases'],
        reason='First persistent-SIFT run has insufficient target checks, control depth regression and two solver-budget stops. This is a disclosed adaptive development follow-up, not untouched confirmation.',
        patches=dict(corner_quality=.005,max_corners=3000,min_distance_px=7,window=21,pyramid_levels=3,
                     bidirectional_px=.75,neighbor_patch_ncc=.85,patch_size=11,minimum_border_px=12,
                     all_observations_in_original_check_tiles=True,endpoint_dedup_px=3),
        fit='Exactly the first run fitting tracks, initialization, calibration and objective; fixed 240 function evaluation cap for every case, no check-based tuning.',
        check='Track first-frame corners through every native frame; freeze surviving identities before BA. No camera/pose/error threshold used for tracking. Stable-depth rule unchanged. Keep original SIFT checks and original direct endpoint checks.',
        gates=parent['gate'],training_authorized=False,whole_scene_accepted=False,
        limits='Repeated exposed data and shared sensor, no independent physical reference. More patch matches may be correlated; match counts are not surface accuracy.',
        parent_protocol_sha256=m.sha(PARENT/'protocol.json'),parent_result_sha256=m.sha(PARENT/'result.json'))
    m.write(out/'protocol.json',protocol)
    (out/'source_snapshot').mkdir()
    for p in (Path(__file__),Path(r.__file__)):
        shutil.copyfile(p,out/'source_snapshot'/p.name)
    m.write(out/'source_hashes.json',{p.name:m.sha(p) for p in (out/'source_snapshot').iterdir()})
    print('FOLLOWUP PROTOCOL FROZEN',flush=True)


def run(out):
    protocol=m.read(out/'protocol.json')
    assert m.sha(PARENT/'protocol.json')==protocol['parent_protocol_sha256']
    assert m.sha(PARENT/'result.json')==protocol['parent_result_sha256']
    for p in (Path(__file__),Path(r.__file__)):
        assert m.sha(p)==m.read(out/'source_hashes.json')[p.name]
    cal=m.read(m.INPUT/'calibration_clean.json');K=np.array(cal['K'])
    maps,_=m.projection.remap(cal['raw_color'],K,cal['width'],cal['height'])
    source=Path(m.read(m.INPUT/'capture_manifest.json')['source']).resolve()
    cv2.setNumThreads(6);cv2.setRNGSeed(11010)
    results=[]
    with sqlite3.connect(source.as_uri()+'?mode=ro',uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        for case in protocol['cases']:
            dest=out/case['label'];dest.mkdir(exist_ok=False)
            prior=PARENT/case['label'];features=[];images=[]
            cached={f['source_id']:f for f in m.read(prior/'inputs.json')}
            for mid in case['source_ids']:
                p=prior/'inputs'/f'{mid}.npz';assert m.sha(p)==cached[mid]['sha256']
                with np.load(p) as data:f={k:data[k].copy() for k in data.files}
                features.append(f)
                rgb,_=m.clean.capture.CDR(db.execute('select data from messages where id=?',(mid,)).fetchone()[0]).image()
                images.append(cv2.remap(rgb,maps[:,:,0],maps[:,:,1],cv2.INTER_LINEAR))
            uv,tracking=patch_tracks(images)
            m.write(dest/'patch_tracking.json',tracking)
            patch_features=[dict(uv=xy,depth=f['depth'],check=np.ones(len(xy),bool)) for xy,f in zip(uv,features)]
            r.observations(patch_features,K)
            tracks=[[(c,i) for c in range(len(features))] for i in range(uv.shape[1])]
            checks=r.fixed_checks(patch_features,tracks,case['check_pairs'])
            np.savez_compressed(dest/'patch_tracks.npz',uv=uv)
            m.write(dest/'frozen_patch_checks.json',checks)
            assert all(m.check_tile(xy).all() for xy in uv)
            r.observations(features,K)
            fitting=m.read(prior/'tracks.json')['tracks']['fit']
            initial=np.array(m.read(prior/'initial_poses.json'))
            poses,ba=r.optimize(features,fitting,initial,K,max_nfev=240)
            m.write(dest/'optimization.json',ba)
            m.write(dest/'optimized_poses.json',poses.tolist())
            patch_before=r.evaluate(patch_features,checks,initial,K)
            patch_after=r.evaluate(patch_features,checks,poses,K)
            sift_checks=m.read(prior/'frozen_checks.json')
            sift_before=r.evaluate(features,sift_checks,initial,K)
            sift_after=r.evaluate(features,sift_checks,poses,K)
            a,b=features[0],features[-1];ia=np.nonzero(a['check'])[0];ib=np.nonzero(b['check'])[0]
            matches=m.mutual(a['desc'][ia],b['desc'][ib]);ia=ia[matches[:,0]];ib=ib[matches[:,1]]
            valid=a['valid'][ia]&b['valid'][ib];ia,ib=ia[valid],ib[valid]
            original=[dict(cameras=[0,len(features)-1],refs=[[j,int(x),int(y)] for j,(x,y) in enumerate(zip(ia,ib))])]
            orig_before=r.evaluate(features,original,initial,K);orig_after=r.evaluate(features,original,poses,K)
            comparisons=list(zip(patch_before,patch_after))+list(zip(sift_before,sift_after))+list(zip(orig_before,orig_after))
            measured=[(a,b) for a,b in comparisons if a['matches']>0]
            noninferior=all(all(y<=x+.01 for x,y in zip(a['separation_m'],b['separation_m'])) for a,b in measured)
            gain=bool(patch_before[0]['separation_m'] and any(x-y>=.01 for x,y in zip(patch_before[0]['separation_m'],patch_after[0]['separation_m'])))
            evidence=ba['success'] and ba['finite'] and ba['all_fit_points_in_front']
            # Original sufficiently supported checks remain vetoes; lack of old support
            # may be addressed by the newly frozen patch checks, never called an old pass.
            original_supported_ok=all(x['matches']<20 or (x['separation_m'][0]<.05 and x['separation_m'][1]<.12) for x in orig_after)
            summary=dict(label=case['label'],boundary=case['boundary'],patch_tracking=tracking,
                chain_patch_checks=patch_before,ba_patch_checks=patch_after,
                chain_sift_checks=sift_before,ba_sift_checks=sift_after,
                original_check_chain=orig_before,original_check_ba=orig_after,
                chain_patch_support_pass=r.support_pass(patch_before),ba_patch_support_pass=r.support_pass(patch_after) and evidence,
                original_sufficient_support_gate_pass=original_supported_ok,
                noninferior_all_measured_checks=noninferior,meaningful_endpoint_pose_gain=gain,
                bounded_candidate_pass=r.support_pass(patch_after) and evidence and noninferior and gain and original_supported_ok,
                physical_structures_accepted=False,whole_scene_accepted=False,training_authorized=False)
            m.write(dest/'result.json',summary);results.append(summary)
            print('FOLLOWUP',case['label'],'checks',[x['matches'] for x in patch_after],
                  'support',summary['ba_patch_support_pass'],'noninferior',noninferior,'gain',gain,'solver',ba['status'],flush=True)
    m.write(out/'result.json',dict(cases=results,training_authorized=False,whole_scene_accepted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['prepare','run']);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();(prepare if a.action=='prepare' else run)(a.output_dir)
