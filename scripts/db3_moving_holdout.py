"""Source-track holdouts, refitted initialization and bounded local reconstruction.

Checks follow physical image tracks. Original check tiles and source reservations
stay excluded. New check trajectories exclude fitting features in every frame,
and exclusions propagate through descriptor components before any pose fitting.
"""
import argparse
import importlib.util
from pathlib import Path
import shutil
import sqlite3
import time

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

spec=importlib.util.spec_from_file_location('recovery',Path(__file__).with_name('db3_r4_track_recovery.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
m=r.m


def follow_checks(images):
    gray=[cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) for im in images]
    h,w=gray[0].shape;yy,xx=np.mgrid[:h,:w]
    mask=m.check_tile(np.c_[xx.ravel(),yy.ravel()]).reshape(h,w)
    mask[:12]=False;mask[-12:]=False;mask[:,:12]=False;mask[:,-12:]=False
    p=cv2.goodFeaturesToTrack(gray[0],3000,.005,7,mask=(255*mask).astype('u1'),blockSize=7)
    if p is None:return np.empty((len(images),0,2)),[np.empty((0,2)) for _ in images],dict(seeds=0,survivors=0)
    p=p.reshape(-1,2);alive=np.ones(len(p),bool);trajectory=[p.copy()];reserved=[p.copy()];stages=[]
    settings=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
    for c in range(1,len(images)):
        q,fs,_=cv2.calcOpticalFlowPyrLK(gray[c-1],gray[c],p[:,None].astype('f4'),None,**settings)
        back,bs,_=cv2.calcOpticalFlowPyrLK(gray[c],gray[c-1],q,None,**settings)
        q=q.reshape(-1,2);back=back.reshape(-1,2)
        good=(fs.ravel()>0)&(bs.ravel()>0)&np.isfinite(q).all(1)&(np.linalg.norm(back-p,axis=1)<.75)
        good &=(q[:,0]>=12)&(q[:,0]<w-12)&(q[:,1]>=12)&(q[:,1]<h-12)
        survived_flow=alive&good
        # Reserve every geometrically untested flow-valid observation, including
        # tracks later rejected by appearance or lifespan criteria.
        reserved.append(q[survived_flow].copy())
        for k in np.nonzero(survived_flow)[0]:
            a=cv2.getRectSubPix(gray[c-1],(11,11),tuple(p[k])).astype(float).ravel()
            b=cv2.getRectSubPix(gray[c],(11,11),tuple(q[k])).astype(float).ravel()
            a-=a.mean();b-=b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
            if den<=1e-6 or np.dot(a,b)/den<.85:good[k]=False
        alive &= good
        stages.append(dict(camera=c,flow_valid=int(survived_flow.sum()),appearance_valid=int(alive.sum()),
                           outside_original_tiles=int((alive&~m.check_tile(q)).sum())))
        trajectory.append(q.copy());p=q
    uv=np.array(trajectory)[:,alive];keep=[]
    for i in range(uv.shape[1]):
        if all(np.linalg.norm(uv[-1,i]-uv[-1,j])>=3 for j in keep):keep.append(i)
    uv=uv[:,keep]
    return uv,reserved,dict(seeds=len(trajectory[0]),survivors=len(keep),stages=stages,
                            crosses_original_tile_boundaries=int(sum(not m.check_tile(uv[:,j]).all() for j in range(uv.shape[1]))))


def propagate_exclusions(features,links,reserved,radius=12.):
    """Exclude whole descriptor components touching any old/new check observation."""
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    for a,b in links:
        a,b=tuple(a),tuple(b);ra,rb=find(a),find(b)
        if ra!=rb:parent[rb]=ra
    blocked=[]
    for f,uv in zip(features,reserved):
        near=cKDTree(uv).query(f['uv'])[0]<=radius if len(uv) else np.zeros(len(f['uv']),bool)
        blocked.append(f['check']|near)
    bad={find(node) for node in parent if blocked[node[0]][node[1]]}
    for c,i in parent:
        if find((c,i)) in bad:blocked[c][i]=True
    return blocked


def fit_problem(features,tracks,initial,K,mode):
    tracks=[tr for tr in tracks if sum(features[c]['valid'][i] for c,i in tr)>=2][:800]
    if len(tracks)<30:raise ValueError('Fewer than 30 independent fitting tracks')
    ci=[];pi=[];uv=[];depth=[];sigma=[];points=[]
    for j,tr in enumerate(tracks):
        world=[r.transform_points(np.linalg.inv(initial[c]),features[c]['xyz'][i]) for c,i in tr if features[c]['valid'][i]]
        points.append(np.median(world,axis=0))
        for c,i in tr:
            assert not features[c]['check'][i]
            if mode=='metric' and not features[c]['valid'][i]:continue
            ci.append(c);pi.append(j);uv.append(features[c]['uv'][i]);depth.append(features[c]['z'][i] if features[c]['valid'][i] else 0.)
            sigma.append(features[c]['sigma'][i])
    ci,pi,uv,depth,sigma=map(np.array,(ci,pi,uv,depth,sigma))
    cams=np.array([np.r_[Rotation.from_matrix(T[:3,:3]).as_rotvec(),T[:3,3]] for T in initial])
    nc=len(cams);offset=6*(nc-1);n=len(ci)
    x0=np.r_[cams[1:].ravel(),np.array(points).ravel()]
    observed=m.backproject(uv,depth,K)
    def unpack(x):return np.vstack((cams[0],x[:offset].reshape(-1,6))),x[offset:].reshape(-1,3)
    def calc(x,want_jac=False):
        c,p=unpack(x);rot=[];drot=[]
        for camera in c:
            R,J=cv2.Rodrigues(camera[:3]);rot.append(R);drot.append(J.reshape(3,3,3))
        rot=np.array(rot);drot=np.array(drot)
        q=np.einsum('nij,nj->ni',rot[ci],p[pi])+c[ci,3:]
        A=np.zeros((n,3,3))
        if mode=='metric':
            # Isotropic conservative range scale, distinct frozen objective arm.
            scale=np.maximum(sigma,.01)
            residual=(q-observed)/scale[:,None]
            A[:,range(3),range(3)]=1/scale[:,None]
        else:
            z=np.maximum(q[:,2],.01)
            predicted=q[:,:2]/z[:,None]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
            ds=np.where(depth>0,1/np.maximum(sigma,.01),0)
            residual=np.column_stack(((predicted-uv)/.75,(q[:,2]-depth)*ds))
            A[:,0,0]=K[0,0]/z/.75;A[:,1,1]=K[1,1]/z/.75
            A[:,0,2]=np.where(q[:,2]>.01,-K[0,0]*q[:,0]/z**2/.75,0)
            A[:,1,2]=np.where(q[:,2]>.01,-K[1,1]*q[:,1]/z**2/.75,0)
            A[:,2,2]=ds
        if not want_jac:return residual.ravel()
        # Rodrigues derivative axes: parameter, output row, input column.
        dq=np.einsum('nkrj,nj->nrk',drot[ci],p[pi])
        Jcam=np.concatenate((np.einsum('nij,njk->nik',A,dq),A),axis=2)
        Jpoint=np.einsum('nij,njk->nik',A,rot[ci])
        obsrow=np.arange(n)[:,None,None]*3+np.arange(3)[None,:,None]
        active=ci>0
        camrows=np.broadcast_to(obsrow,(n,3,6))[active].ravel()
        camcols=np.broadcast_to((ci[:,None,None]-1)*6+np.arange(6)[None,None,:],(n,3,6))[active].ravel()
        ptrows=np.broadcast_to(obsrow,(n,3,3)).ravel()
        ptcols=np.broadcast_to(offset+pi[:,None,None]*3+np.arange(3)[None,None,:],(n,3,3)).ravel()
        return coo_matrix((np.r_[Jcam[active].ravel(),Jpoint.ravel()],(np.r_[camrows,ptrows],np.r_[camcols,ptcols])),shape=(n*3,len(x))).tocsr()
    return x0,calc,unpack,dict(fitting_tracks=tracks,camera_index=ci.tolist(),point_index=pi.tolist(),depth_factors=int(sum(depth>0)))


def refine(features,tracks,initial,K,mode):
    x0,calc,unpack,meta=fit_problem(features,tracks,initial,K,mode)
    start=time.time();before=calc(x0)
    sol=least_squares(calc,x0,jac=lambda x:calc(x,True),loss='soft_l1',f_scale=1.,x_scale='jac',
                      max_nfev=240,ftol=1e-5,xtol=1e-6,gtol=1e-5,tr_solver='lsmr',tr_options={'maxiter':400})
    cams,points=unpack(sol.x);poses=np.repeat(np.eye(4)[None],len(initial),axis=0)
    poses[:,:3,:3]=Rotation.from_rotvec(cams[:,:3]).as_matrix();poses[:,:3,3]=cams[:,3:]
    ci=np.array(meta['camera_index']);pi=np.array(meta['point_index'])
    q=np.einsum('nij,nj->ni',poses[ci,:3,:3],points[pi])+poses[ci,:3,3]
    report=dict(**meta,mode=mode,status=int(sol.status),success=bool(sol.success),message=sol.message,nfev=sol.nfev,
                finite=bool(np.isfinite(sol.x).all()),all_fit_points_in_front=bool((q[:,2]>.01).all()),
                fitted_rms_before=float(np.sqrt(np.mean(before**2))),fitted_rms_after=float(np.sqrt(np.mean(calc(sol.x)**2))),
                points_world=points.tolist(),elapsed_seconds=time.time()-start)
    return poses,report


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    manifest=m.read(m.INPUT/'capture_manifest.json');frames=manifest['frames']
    native=m.read(m.R1/'prospective_checks.json')['source_frames'];byid={f['rgb_message_id']:f for f in native}
    names=set(m.read(m.INPUT/'reconstruction_split.json')['historical_eval'])
    historical={f['message_id'] for f in frames if f['name'] in names}
    cases=[]
    for label,boundary in [('target',362),('matte_control',36)]:
        a,b=frames[boundary:boundary+2];roster=[f for f in native if a['timestamp_ns']<=f['timestamp_ns']<=b['timestamp_ns']]
        assert all(r.route.eligible(f,byid,historical) for f in roster)
        n=len(roster);cases.append(dict(label=label,boundary=boundary,source_ids=[f['rgb_message_id'] for f in roster],
                                      seconds=[roster[0]['replay_seconds'],roster[-1]['replay_seconds']],check_pairs=[[0,n-1],[0,n//2],[n//2,n-1]]))
    protocol=dict(date='2026-11-11',timezone='Asia/Saigon',cases=cases,
        selection='Same failed target 362 and matte-region passing boundary 36, chosen from exposed R4 results before new moving-holdout outputs.',
        check_tracking='Seed original check tiles in first frame, then track source identities across any image tiles. Preserve forward/backward <0.75px, NCC>=0.85, 11px patch, 12px image border, endpoint dedup>=3px. No pose-based filtering.',
        fit_exclusion='Original check tiles OR distance<=12px from every flow-valid reserved trajectory; propagate exclusions through ALL descriptor-linked components. Refit every native initializer with remaining fit-only descriptors. Old poses are reference only.',
        fixed_arms=['pixel_depth','metric'],optimization='Same frozen fitting tracks and initializer; up to 800 tracks, >=30 with two stable depths; first camera/calibration fixed; analytic sparse Jacobian, x_scale=jac, soft_l1, max_nfev240, lsmr maxiter400.',
        gates=dict(min_matches=20,min_hull=.05,median_m=.05,p90_m=.12,median_px=2.,p90_px=4.,max_worsening_m=.01,meaningful_metric_gain_m=.01),
        comparison='Same moving check identities for old chain, excluded refit chain and both BA arms; retain original direct descriptor endpoint checks with zero deletions. Support recovery is separate from measured pose gain.',
        advancement='Only successful local geometry may justify extending to a usable local training window and static-structure/appearance comparisons. Short-window consistency alone is not full-scene acceptance.',
        limits='Exposed development; shared sensor/calibration, no external physical reference. Track exclusions establish observation separation, not statistical independence.',
        source_hashes={str(p.relative_to(r.ROOT)):m.sha(p) for p in (m.INPUT/'capture_manifest.json',m.INPUT/'calibration_clean.json',m.INPUT/'reconstruction_split.json',m.R1/'prospective_checks.json',r.R4/'inputs_manifest.json')})
    m.write(out/'protocol.json',protocol);(out/'source_snapshot').mkdir()
    for p in (Path(__file__),Path(r.__file__),r.ROOT/'scripts/db3_transition_revisit.py',r.ROOT/'scripts/db3_local_geometry.py'):
        shutil.copyfile(p,out/'source_snapshot'/p.name)
    m.write(out/'source_hashes.json',{p.name:m.sha(p) for p in (out/'source_snapshot').iterdir()})
    print('MOVING HOLDOUT PROTOCOL FROZEN',cases,flush=True)


def run(out):
    protocol=m.read(out/'protocol.json')
    for p,h in protocol['source_hashes'].items():assert m.sha(r.ROOT/p)==h
    assert m.sha(Path(__file__))==m.read(out/'source_hashes.json')[Path(__file__).name]
    cal=m.read(m.INPUT/'calibration_clean.json');K=np.array(cal['K'])
    maps,_=m.projection.remap(cal['raw_color'],K,cal['width'],cal['height'])
    source=Path(m.read(m.INPUT/'capture_manifest.json')['source']).resolve();stat=source.stat()
    cached={f['source_id']:f for f in m.read(r.R4/'inputs_manifest.json')['frames']}
    native={f['rgb_message_id']:f for f in m.read(m.R1/'prospective_checks.json')['source_frames']}
    cv2.setNumThreads(6);cv2.setRNGSeed(11010);sift=cv2.SIFT_create(nfeatures=3000,contrastThreshold=.02)
    results=[]
    with sqlite3.connect(source.as_uri()+'?mode=ro',uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        tid=db.execute("select id from topics where name like '%/Depth_0/image/data'").fetchone()[0]
        depths=db.execute('select id,timestamp from messages where topic_id=? order by timestamp',(tid,)).fetchall();stamps=np.array([s for _,s in depths])
        for case in protocol['cases']:
            dest=out/case['label'];dest.mkdir(exist_ok=False);(dest/'inputs').mkdir();features=[];images=[];provenance=[]
            for ci,mid in enumerate(case['source_ids']):
                ts,blob=db.execute('select timestamp,data from messages where id=?',(mid,)).fetchone();assert ts==native[mid]['timestamp_ns']
                rgb,rh=m.clean.capture.CDR(blob).image();im=cv2.remap(rgb,maps[:,:,0],maps[:,:,1],cv2.INTER_LINEAR)
                if mid in cached:
                    p=r.R4/'inputs'/f'{mid}.npz';assert m.sha(p)==cached[mid]['sha256']
                    with np.load(p) as z:f={k:z[k].copy() for k in z.files}
                    did=cached[mid]['depth_source_id']
                else:
                    j=int(np.searchsorted(stamps,ts));j=min([max(0,j-1),min(len(stamps)-1,j)],key=lambda k:abs(int(stamps[k])-ts));did=depths[j][0]
                    raw,dh=m.clean.adapter.decode_depth(db.execute('select data from messages where id=?',(did,)).fetchone()[0]);assert abs(dh-rh['header_seconds'])<.02
                    *_,e=m.projection.align_depth(raw,cal['depth_units'],cal['raw_depth'],cal['depth_to_color_R'],cal['depth_to_color_t_m'],K,cal['width'],cal['height'])
                    kp,desc=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY),None);uv=np.array([k.pt for k in kp],dtype='f4').reshape(-1,2)
                    f=dict(uv=uv,desc=desc if desc is not None else np.empty((0,128),'f4'),check=m.check_tile(uv),depth=e['depth_m'])
                np.savez_compressed(dest/'inputs'/f'{mid}.npz',**f);Image.fromarray(im).save(dest/'inputs'/f'{mid}.png')
                provenance.append(dict(source_id=mid,depth_source_id=did,feature_sha256=m.sha(dest/'inputs'/f'{mid}.npz'),image_sha256=m.sha(dest/'inputs'/f'{mid}.png')))
                features.append(f);images.append(im)
            m.write(dest/'inputs.json',provenance)
            uv,reserved,tracking=follow_checks(images);m.write(dest/'tracking.json',tracking)
            np.savez_compressed(dest/'moving_checks.npz',uv=uv,**{f'reserved_{c}':x for c,x in enumerate(reserved)})
            _,_,pair_counts,links=r.build_tracks(features)
            blocked=propagate_exclusions(features,links,reserved)
            fitting=[dict(f,check=b) for f,b in zip(features,blocked)]
            tracks,rejected=r.components(links,fitting)
            m.write(dest/'tracks.json',dict(tracks=tracks,rejected=rejected,links=links,pair_counts=pair_counts))
            np.savez_compressed(dest/'exclusions.npz',**{f'camera_{c}':x for c,x in enumerate(blocked)})
            pf=[dict(uv=xy,depth=f['depth']) for xy,f in zip(uv,features)];r.observations(pf,K)
            checktracks=[[(c,i) for c in range(len(features))] for i in range(uv.shape[1])]
            checks=r.fixed_checks(pf,checktracks,case['check_pairs']);m.write(dest/'frozen_checks.json',checks)
            m.write(dest/'pre_fit_seal.json',{name:m.sha(dest/name) for name in ('moving_checks.npz','tracks.json','exclusions.npz','frozen_checks.json')})
            print('FROZEN',case['label'],'tracks',tracking['survivors'],'crossing',tracking['crosses_original_tile_boundaries'],
                  'depth checks',[len(x['refs']) for x in checks],'fit tracks',len(tracks['fit']),flush=True)
            chains={};pairreports={}
            for name,fs in [('old_chain',features),('excluded_chain',fitting)]:
                chain=[np.eye(4)];rows=[]
                for a,b in zip(fs,fs[1:]):
                    row,T,_=r.t.pair(a,b,K);rows.append(row)
                    if T is None:break
                    chain.append(T@chain[-1])
                pairreports[name]=rows
                if len(chain)==len(features):chains[name]=np.array(chain)
            m.write(dest/'pair_reports.json',pairreports)
            if len(chains)!=2:
                results.append(dict(label=case['label'],complete=False,reason='fit-only initialization incomplete',training_authorized=False));continue
            for name,poses in chains.items():m.write(dest/f'{name}_poses.json',poses.tolist())
            r.observations(features,K);r.observations(fitting,K)
            a,b=features[0],features[-1];ia=np.nonzero(a['check'])[0];ib=np.nonzero(b['check'])[0]
            pm=m.mutual(a['desc'][ia],b['desc'][ib]);ia,ib=ia[pm[:,0]],ib[pm[:,1]];valid=a['valid'][ia]&b['valid'][ib]
            original=[dict(cameras=[0,len(features)-1],refs=[[j,int(x),int(y)] for j,(x,y) in enumerate(zip(ia[valid],ib[valid]))])]
            m.write(dest/'original_checks.json',original)
            old=r.evaluate(pf,checks,chains['old_chain'],K);oldorig=r.evaluate(features,original,chains['old_chain'],K)
            candidates={}
            for mode in protocol['fixed_arms']:
                try:poses,ba=refine(fitting,tracks['fit'],chains['excluded_chain'],K,mode)
                except ValueError as error:
                    candidates[mode]=dict(complete=False,reason=str(error));continue
                m.write(dest/f'{mode}_poses.json',poses.tolist());m.write(dest/f'{mode}_optimization.json',ba)
                measured=r.evaluate(pf,checks,poses,K);orig=r.evaluate(features,original,poses,K)
                comparisons=list(zip(old,measured))+list(zip(oldorig,orig))
                noninferior=all(x['separation_m'] is not None and all(y<=v+.01 for v,y in zip(x['separation_m'],z['separation_m'])) for x,z in comparisons)
                gain=bool(old[0]['separation_m'] and measured[0]['separation_m'] and any(x-y>=.01 for x,y in zip(old[0]['separation_m'],measured[0]['separation_m'])))
                original_ok=all(x['matches']<20 or (x['separation_m'][0]<.05 and x['separation_m'][1]<.12) for x in orig)
                consistent=r.support_pass(measured) and ba['success'] and ba['finite'] and ba['all_fit_points_in_front']
                candidates[mode]=dict(complete=True,checks=measured,original_checks=orig,local_consistency_pass=consistent,
                                      noninferior=noninferior,meaningful_metric_gain=gain,original_supported_checks_pass=original_ok,
                                      pose_improvement_pass=consistent and noninferior and gain and original_ok)
                print('CANDIDATE',case['label'],mode,'solver',ba['status'],'consistency',consistent,'noninferior',noninferior,'gain',gain,flush=True)
            result=dict(label=case['label'],boundary=case['boundary'],complete=True,tracking=tracking,
                        old_chain_checks=old,excluded_chain_checks=r.evaluate(pf,checks,chains['excluded_chain'],K),
                        original_old_chain=oldorig,candidates=candidates,training_authorized=False,whole_scene_accepted=False)
            m.write(dest/'result.json',result);results.append(result)
    assert (source.stat().st_size,source.stat().st_mtime_ns)==(stat.st_size,stat.st_mtime_ns)
    m.write(out/'result.json',dict(cases=results,training_authorized=False,whole_scene_accepted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['prepare','run']);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();(prepare if args.action=='prepare' else run)(args.output_dir)
