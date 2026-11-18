"""Predeclared fitting-track follow-up to moving source holdouts.

New fitting corners are tracked, not descriptor-matched. The previous frozen
validation trajectories, depth rules and acceptance thresholds are reused.
"""
import argparse
import importlib.util
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

spec=importlib.util.spec_from_file_location('moving',Path(__file__).with_name('db3_moving_holdout.py'))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
r,m=p.r,p.m
PARENT=r.ROOT/'results/db3_moving_holdout_20261111'


def fitting_tracks(images,depths,reserved):
    gray=[cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) for im in images];h,w=gray[0].shape
    trees=[cKDTree(x) if len(x) else None for x in reserved]
    def allowed(c,xy):
        valid=np.isfinite(xy).all(1)&(xy[:,0]>=12)&(xy[:,0]<w-12)&(xy[:,1]>=12)&(xy[:,1]<h-12)&~m.check_tile(xy)
        if trees[c] is not None:valid &=trees[c].query(xy)[0]>12
        return valid
    yy,xx=np.mgrid[:h,:w];grid=np.c_[xx.ravel(),yy.ravel()];all_tracks=[];seeding=[]
    settings=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
    for seed in range(0,len(images)-2,3):
        mask=allowed(seed,grid).reshape(h,w).astype('u1')*255
        # Previously tracked observations prevent duplicate landmark seeds.
        existing=[xy for tr in all_tracks for c,xy in tr if c==seed]
        for xy in existing:cv2.circle(mask,tuple(np.rint(xy).astype(int)),7,0,-1)
        corners=cv2.goodFeaturesToTrack(gray[seed],1500,.005,7,mask=mask,blockSize=7)
        if corners is None:continue
        q=corners.reshape(-1,2);alive=np.ones(len(q),bool);tracks=[[(seed,x.copy())] for x in q];blocked=np.zeros(len(q),bool)
        for c in range(seed+1,len(images)):
            nxt,fs,_=cv2.calcOpticalFlowPyrLK(gray[c-1],gray[c],q[:,None].astype('f4'),None,**settings)
            back,bs,_=cv2.calcOpticalFlowPyrLK(gray[c],gray[c-1],nxt,None,**settings)
            nxt=nxt.reshape(-1,2);back=back.reshape(-1,2)
            flow=(fs.ravel()>0)&(bs.ravel()>0)&np.isfinite(nxt).all(1)&(np.linalg.norm(back-q,axis=1)<.75)
            permitted=allowed(c,nxt)
            # Entire physical fitting track is discarded if a tracked observation
            # is seen entering a reserved area; no earlier segment is retained.
            blocked |=alive&flow&~permitted
            good=alive&flow&permitted
            for k in np.nonzero(good)[0]:
                a=cv2.getRectSubPix(gray[c-1],(11,11),tuple(q[k])).astype(float).ravel()
                b=cv2.getRectSubPix(gray[c],(11,11),tuple(nxt[k])).astype(float).ravel()
                a-=a.mean();b-=b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
                if den<=1e-6 or np.dot(a,b)/den<.85:good[k]=False
            for k in np.nonzero(good)[0]:tracks[k].append((c,nxt[k].copy()))
            alive=good;q=nxt
        kept=[tr for tr,b in zip(tracks,blocked) if not b and len(tr)>=3]
        all_tracks.extend(kept);seeding.append(dict(camera=seed,seeds=len(corners),kept=len(kept),touching_reservation=int(blocked.sum())))
    all_tracks.sort(key=lambda tr:(-len(tr),tr[0][0],float(tr[0][1][0]),float(tr[0][1][1])))
    features=[dict(uv=[],depth=z) for z in depths];tracks=[]
    for track in all_tracks:
        nodes=[]
        for c,xy in track:
            idx=len(features[c]['uv']);features[c]['uv'].append(xy);nodes.append((c,idx))
        tracks.append(nodes)
    for c,f in enumerate(features):
        f['uv']=np.array(f['uv'],dtype='f4').reshape(-1,2);f['check']=np.zeros(len(f['uv']),bool)
        assert allowed(c,f['uv']).all()
    return features,tracks,seeding


def initializer(features,tracks):
    chain=[np.eye(4)];rows=[]
    for c in range(len(features)-1):
        pairs=[]
        for tr in tracks:
            ids=dict(tr)
            if c in ids and c+1 in ids:
                a,b=ids[c],ids[c+1]
                if features[c]['valid'][a] and features[c+1]['valid'][b]:pairs.append((a,b))
        pairs=np.array(pairs,int).reshape(-1,2)
        x,y=features[c]['xyz'][pairs[:,0]],features[c+1]['xyz'][pairs[:,1]]
        fit=r.t.recovery.robust_fit(x,y,np.random.default_rng(11010))
        if fit is None:return None,rows+[dict(camera=c,matches=len(x),fit=False)]
        T,nin,error=fit;chain.append(T@chain[-1]);rows.append(dict(camera=c,matches=len(x),inliers=nin,median_fit_error_m=error,transform=T.tolist()))
    return np.array(chain),rows


def prepare(out):
    out.mkdir(parents=True,exist_ok=False);prior=m.read(PARENT/'protocol.json')
    m.write(out/'protocol.json',dict(date='2026-11-11',timezone='Asia/Saigon',cases=prior['cases'],
        parent_protocol_sha256=m.sha(PARENT/'protocol.json'),parent_result_sha256=m.sha(PARENT/'result.json'),
        purpose='Moving holdouts restore evidence; replace potentially false SIFT fitting correspondences with source-tracked image corners.',
        fitting='Seed every third native image, max1500 corners, quality .005, minDistance7; same FB/NCC tracking rules; original check tiles and 12px moving holdouts excluded. Discard entire tracks touching reservations. Deduplicate seeds within7px of prior tracked observations; keep>=3 views.',
        initialization='Fit adjacent rigid transforms only from stable-depth shared physical fitting tracks, existing robust-fit thresholds. No old pose initialization.',
        arms=prior['fixed_arms'],gates=prior['gates'],solver='Same tested analytic Jacobian, fixed240 cap and robust objectives as parent.',
        evaluation='Reuse exact frozen moving and original checks, no filtering/deletion. Adaptive exposed development follow-up, not untouched test.',
        training_authorized=False,whole_scene_accepted=False))
    (out/'source_snapshot').mkdir()
    for path in (Path(__file__),Path(p.__file__),Path(r.__file__)):
        shutil.copyfile(path,out/'source_snapshot'/path.name)
    m.write(out/'source_hashes.json',{f.name:m.sha(f) for f in (out/'source_snapshot').iterdir()})
    print('TRACKED FIT PROTOCOL FROZEN',flush=True)


def run(out):
    protocol=m.read(out/'protocol.json')
    assert m.sha(PARENT/'protocol.json')==protocol['parent_protocol_sha256']
    assert m.sha(PARENT/'result.json')==protocol['parent_result_sha256']
    for path in (Path(__file__),Path(p.__file__),Path(r.__file__)):
        assert m.sha(path)==m.read(out/'source_hashes.json')[path.name]
    K=np.array(m.read(m.INPUT/'calibration_clean.json')['K']);cv2.setNumThreads(6);results=[]
    for case in protocol['cases']:
        prior=PARENT/case['label'];dest=out/case['label'];dest.mkdir(exist_ok=False)
        images=[];original_features=[]
        for f in m.read(prior/'inputs.json'):
            path=prior/'inputs'/f'{f["source_id"]}.npz';assert m.sha(path)==f['feature_sha256']
            with np.load(path) as a:original_features.append({k:a[k].copy() for k in a.files})
            path=path.with_suffix('.png');assert m.sha(path)==f['image_sha256'];images.append(np.array(Image.open(path).convert('RGB')))
        with np.load(prior/'moving_checks.npz') as data:
            uv=data['uv'];reserved=[data[f'reserved_{c}'] for c in range(len(images))]
        fs,tracks,seeding=fitting_tracks(images,[f['depth'] for f in original_features],reserved)
        r.observations(fs,K);r.observations(original_features,K)
        np.savez_compressed(dest/'fit_features.npz',**{f'uv_{c}':f['uv'] for c,f in enumerate(fs)})
        m.write(dest/'tracks.json',dict(tracks=tracks,seeding=seeding))
        print('TRACKED FIT',case['label'],'tracks',len(tracks),'observations',[len(f['uv']) for f in fs],flush=True)
        initial,rows=initializer(fs,tracks);m.write(dest/'initialization.json',rows)
        if initial is None:
            result=dict(label=case['label'],complete=False,reason='insufficient independent tracked initializer',training_authorized=False)
            m.write(dest/'result.json',result);results.append(result);continue
        m.write(dest/'initial_poses.json',initial.tolist())
        checkfs=[dict(uv=xy,depth=f['depth']) for xy,f in zip(uv,original_features)];r.observations(checkfs,K)
        checks=m.read(prior/'frozen_checks.json');original=m.read(prior/'original_checks.json')
        oldposes=np.array(m.read(prior/'old_chain_poses.json'))
        old=r.evaluate(checkfs,checks,oldposes,K);oldorig=r.evaluate(original_features,original,oldposes,K)
        arms={}
        for mode in protocol['arms']:
            poses,ba=p.refine(fs,tracks,initial,K,mode)
            m.write(dest/f'{mode}_poses.json',poses.tolist());m.write(dest/f'{mode}_optimization.json',ba)
            measured=r.evaluate(checkfs,checks,poses,K);orig=r.evaluate(original_features,original,poses,K)
            noninferior=all(a['separation_m'] is not None and all(y<=x+.01 for x,y in zip(a['separation_m'],b['separation_m'])) for a,b in list(zip(old,measured))+list(zip(oldorig,orig)))
            gain=bool(old[0]['separation_m'] and any(x-y>=.01 for x,y in zip(old[0]['separation_m'],measured[0]['separation_m'])))
            support=r.support_pass(measured) and ba['success'] and ba['finite'] and ba['all_fit_points_in_front']
            origok=all(x['matches']<20 or (x['separation_m'][0]<.05 and x['separation_m'][1]<.12) for x in orig)
            arms[mode]=dict(checks=measured,original_checks=orig,local_consistency_pass=support,noninferior=noninferior,
                            meaningful_metric_gain=gain,original_supported_checks_pass=origok,pose_improvement_pass=support and noninferior and gain and origok)
            print('FIT RESULT',case['label'],mode,'solver',ba['status'],'support',support,'noninferior',noninferior,'gain',gain,flush=True)
        result=dict(label=case['label'],complete=True,old_chain_checks=old,candidates=arms,training_authorized=False,whole_scene_accepted=False)
        m.write(dest/'result.json',result);results.append(result)
    m.write(out/'result.json',dict(cases=results,training_authorized=False,whole_scene_accepted=False))


if __name__=='__main__':
    a=argparse.ArgumentParser(__doc__);a.add_argument('action',choices=['prepare','run']);a.add_argument('--output-dir',type=Path,required=True)
    args=a.parse_args();(prepare if args.action=='prepare' else run)(args.output_dir)
