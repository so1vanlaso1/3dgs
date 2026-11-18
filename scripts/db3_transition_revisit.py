"""R2 bounded native-rate bridge/revisit audit, with measured information matrices.

These candidate constraints are diagnostic. They never alter the baseline or
merge the local submaps. Fitting/check tiles and native reservations are enforced.
"""
import importlib.util
from pathlib import Path
import sqlite3
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

s=importlib.util.spec_from_file_location('local',Path(__file__).with_name('db3_local_geometry.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
recovery=m.module('recover-rgbd-scene.py')
OUT=m.RUN/'transitions_revisits_v2'


def skew(x):
    a,b,c=x;return np.array([[0,-c,b],[c,0,-a],[-b,a,0.]])


def pair(a,b,K):
    rows={}
    for role,check in [('fit',False),('check',True)]:
        ka=a['check']==check;kb=b['check']==check
        ia=np.nonzero(ka)[0];ib=np.nonzero(kb)[0]
        pairs=m.mutual(a['desc'][ka],b['desc'][kb]);uv=a['uv'][ka][pairs[:,0]];vv=b['uv'][kb][pairs[:,1]]
        da,va,sa=m.sample_depth(a['depth'],uv);db,vb,sb=m.sample_depth(b['depth'],vv);valid=va&vb
        rows[role]=(m.backproject(uv[valid],da[valid],K),m.backproject(vv[valid],db[valid],K),
            uv[valid],vv[valid],sa[valid],sb[valid],np.c_[ia[pairs[valid,0]],ib[pairs[valid,1]]])
    x,y,uv,vv,sa,sb,indices=rows['fit']
    record=dict(fit_matches=len(x),check_matches=len(rows['check'][0]),accepted=False)
    fit=recovery.robust_fit(x,y,np.random.default_rng(11010))
    if fit is None:record['reason']='insufficient robust fit support';return record,None,[]
    T,nin,error=fit;pred=x@T[:3,:3].T+T[:3,3];residual=np.linalg.norm(pred-y,axis=1)
    inlier=residual<(.025+.012*np.minimum(x[:,2],y[:,2]))
    H=np.zeros((6,6))
    for q,p,sd,td in zip(x[inlier],y[inlier],sa[inlier],sb[inlier]):
        # Order is rotation radians, translation meters. Pixel ray uncertainty and
        # range uncertainty explicitly produce an anisotropic metric covariance.
        def covariance(q,sigma):
            J=np.array([[q[2]/K[0,0],0,q[0]/q[2]],[0,q[2]/K[1,1],q[1]/q[2]],[0,0,1.]])
            return J@np.diag([.75**2,.75**2,sigma**2])@J.T
        cov=T[:3,:3]@covariance(q,sd)@T[:3,:3].T+covariance(p,td)
        # Left SE(3) perturbation: delta prediction = -[T*q]x*dtheta + dt.
        # Use the same tangent convention when propagating chained covariance.
        J=np.c_[-skew(T[:3,:3]@q+T[:3,3]),np.eye(3)]
        H+=J.T@np.linalg.solve(cov,J)
    eig=np.linalg.eigvalsh(H);condition=float(eig[-1]/max(eig[0],1e-20))
    singular=np.linalg.svd(np.cov(x[inlier].T),compute_uv=False)
    hull=float(cv2.contourArea(cv2.convexHull(uv[inlier].astype('f4')))/(834*463)) if nin>=3 else 0.
    cx,cy,cu,cv,*_=rows['check'];checkerror=np.linalg.norm(cx@T[:3,:3].T+T[:3,3]-cy,axis=1)
    q=lambda x:[float(v) for v in np.percentile(x,[50,90])] if len(x) else None
    # Strict candidate gate, never waive missing distributed support for a loop.
    passed=len(checkerror)>=20 and np.median(checkerror)<.05 and np.percentile(checkerror,90)<.12 and hull>.05 and eig[0]>1 and condition<1e7
    record.update(accepted=bool(passed),reason='candidate only' if passed else 'check/coverage/conditioning gate',
        inliers=int(inlier.sum()),fit_error_m=q(residual[inlier]),check_error_m=q(checkerror),
        fit_hull_fraction=hull,point_covariance_eigenvalues_m2=singular.tolist(),
        planar_or_degenerate=bool(singular[-1]/max(singular[0],1e-12)<.005),
        information=H.tolist(),information_eigenvalues=eig.tolist(),information_condition=condition,
        uncertainty='Anisotropic pixel/range model; sigma is conservative assumed model, not independently calibrated covariance',transform=T.tolist())
    return record,T,indices[inlier].tolist()


def main():
    OUT.mkdir(exist_ok=False);(OUT/'inputs').mkdir()
    original=m.read(m.INPUT/'capture_manifest.json');frames=original['frames'];source=Path(original['source']).resolve()
    reservations=m.read(m.R1/'prospective_checks.json')['source_frames'];byid={r['rgb_message_id']:r for r in reservations}
    historical=set(m.read(m.INPUT/'reconstruction_split.json')['historical_eval']);forbidden={f['message_id'] for f in frames if f['name'] in historical}
    def permitted(mid):return mid not in forbidden and byid[mid]['role']=='fit_development'
    # Frozen selected historical failed boundaries and ordinary controls. Regions
    # crossing reserved blocks are reported excluded rather than silently bridged.
    boundaries=[98,345,373,451];controls=[30,204]
    revisit_pairs=[(58,154),(58,170),(90,202),(90,250),(202,234),(202,250),(202,266),(266,314)]
    m.write(OUT/'protocol.json',dict(boundary_indices=boundaries,control_indices=controls,revisit_pairs=revisit_pairs,
        rates=[3,10,30],check_protocol='spatial tiles excluded from fitting; no pose residual selection of check matches',
        global_merge_gate='Require independently confirmed distributed physical structures and checks, not only feature fits. Diagnostic loops never merged.',
        uncertainty_order=['rx_rad','ry_rad','rz_rad','tx_m','ty_m','tz_m'],
        track_policy='real per-feature source-ID tracks across accepted adjacent links; no independent-edge proxy confidence'))
    cal=m.read(m.DATA/'calibration_clean.json');K=np.array(cal['K']);maps,_=m.projection.remap(cal['raw_color'],K,cal['width'],cal['height'])
    cache={};sift=cv2.SIFT_create(nfeatures=3000,contrastThreshold=.02)
    db=sqlite3.connect(source.as_uri()+'?mode=ro',uri=True);db.execute('PRAGMA query_only=ON')
    tid=db.execute("select id from topics where name like '%/Depth_0/image/data'").fetchone()[0]
    depth_rows=db.execute('select id,timestamp from messages where topic_id=? order by timestamp',(tid,)).fetchall();stamps=np.array([s for _,s in depth_rows])
    def load(mid):
        if mid in cache:return cache[mid]
        assert permitted(mid),'Reserved source observation must never enter fits'
        previous=m.RUN/'transitions_revisits'/'inputs'/f'{mid}.npz'
        if previous.exists():
            with np.load(previous) as data:obj={k:data[k].copy() for k in data.files}
            cache[mid]=obj
            import os
            os.link(previous,OUT/'inputs'/f'{mid}.npz')
            return obj
        stamp,blob=db.execute('select timestamp,data from messages where id=?',(mid,)).fetchone();rgb,rh=m.clean.capture.CDR(blob).image()
        j=int(np.searchsorted(stamps,stamp));j=min([max(0,j-1),min(len(stamps)-1,j)],key=lambda k:abs(int(stamps[k])-stamp))
        did,dstamp=depth_rows[j];raw,dh=m.clean.adapter.decode_depth(db.execute('select data from messages where id=?',(did,)).fetchone()[0]);assert abs(dh-rh['header_seconds'])<.02
        image=cv2.remap(rgb,maps[:,:,0],maps[:,:,1],cv2.INTER_LINEAR)
        _,_,_,e=m.projection.align_depth(raw,cal['depth_units'],cal['raw_depth'],cal['depth_to_color_R'],cal['depth_to_color_t_m'],K,cal['width'],cal['height'])
        kp,desc=sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),None);uv=np.array([k.pt for k in kp],dtype='f4').reshape(-1,2)
        obj=dict(uv=uv,desc=desc,check=m.check_tile(uv),depth=e['depth_m']);cache[mid]=obj
        np.savez_compressed(OUT/'inputs'/f'{mid}.npz',**obj)
        return obj
    bridge_results=[]
    for i in boundaries+controls:
        a,b=frames[i],frames[i+1];interval=[r for r in reservations if a['timestamp_ns']<=r['timestamp_ns']<=b['timestamp_ns']]
        for rate in (3,10,30):
            selected=interval if rate==30 else ([interval[0],*interval[3:-1:3],interval[-1]] if rate==10 else [interval[0],interval[-1]])
            mids=[r['rgb_message_id'] for r in selected];row=dict(boundary=i,kind='historical_failed' if i in boundaries else 'control',rate=rate,source_ids=mids)
            if not all(permitted(mid) for mid in mids):
                row.update(accepted=False,reason='native or historical-evaluation reservation; do not bridge across');bridge_results.append(row);continue
            transforms=[];edges=[];tracks={};total=np.eye(4);covariances=[];chain_cov=np.zeros((6,6))
            for x,y in zip(mids,mids[1:]):
                record,T,links=pair(load(x),load(y),K);record.update(source_ids=[x,y]);edges.append(record)
                if not record['accepted']:break
                total=T@total;transforms.append(T);edge_cov=np.linalg.pinv(np.array(record['information']));covariances.append(edge_cov)
                adjoint=np.zeros((6,6));adjoint[:3,:3]=T[:3,:3];adjoint[3:,3:]=T[:3,:3];adjoint[3:,:3]=skew(T[:3,3])@T[:3,:3]
                chain_cov=adjoint@chain_cov@adjoint.T+edge_cov
                for u,v in links:
                    key=(x,u);track=tracks.get(key,[(x,u)]);track.append((y,v));tracks[(y,v)]=track
            row.update(accepted=len(transforms)==len(mids)-1,edge_results=edges,composed_transform=total.tolist())
            if row['accepted']:
                first,last=load(mids[0]),load(mids[-1]);ua=first['uv'][first['check']];ub=last['uv'][last['check']]
                match=m.mutual(first['desc'][first['check']],last['desc'][last['check']]);ua,ub=ua[match[:,0]],ub[match[:,1]]
                da,va,_=m.sample_depth(first['depth'],ua);d,valid,_=m.sample_depth(last['depth'],ub);valid &= va
                x=m.backproject(ua[valid],da[valid],K);y=m.backproject(ub[valid],d[valid],K);err=np.linalg.norm(x@total[:3,:3].T+total[:3,3]-y,axis=1)
                row.update(endpoint_check_matches=len(err),endpoint_check_error_m=[float(v) for v in np.percentile(err,[50,90])] if len(err) else None,
                    endpoint_checked=bool(len(err)>=20 and np.median(err)<.05 and np.percentile(err,90)<.12),
                    covariance_conservative_chain_bound=(len(covariances)*chain_cov).tolist(),
                    covariance_caveat='Left SE3 tangent with adjoint propagation; frame-correlated cross terms bounded conservatively by multiplying propagated covariance sum by chain length; assumed sensor model, no global accuracy certificate')
            unique={tuple(track) for track in tracks.values() if len(track)>=3}
            m.write(OUT/f'boundary_{i}_rate{rate}_tracks.json',[list(t) for t in sorted(unique)])
            row['real_tracks_length_ge3']=len(unique);bridge_results.append(row)
            print('BOUNDARY',i,rate,row['accepted'],row.get('endpoint_check_error_m'),flush=True)
    revisits=[]
    for a,b in revisit_pairs:
        fa,fb=frames[a],frames[b];ids=[fa['message_id'],fb['message_id']]
        if not all(permitted(mid) for mid in ids):
            revisits.append(dict(a=a,b=b,accepted=False,reason='reserved source observation'));continue
        row,T,links=pair(load(ids[0]),load(ids[1]),K);row.update(a=a,b=b,source_ids=ids,
            globally_accepted=False,global_reason='No independent distributed physical-structure verification; do not merge solely from feature residuals')
        revisits.append(row);print('REVISIT',a,b,row['accepted'],row.get('check_error_m'),flush=True)
    db.close()
    m.write(OUT/'report.json',dict(bridges=bridge_results,revisits=revisits,
        globally_accepted_revisits=0,whole_recording_geometry_accepted=False,
        decision='D8: retain separate checked local submaps and contextual baseline; no forced global merge. Existing recording remains available for further physically validated constraints.',
        source_ids_loaded=sorted(cache),raw_and_baseline_unchanged=True,
        supersedes='transitions_revisits/report.json information matrices: corrected consistent left-SE3 tangent Jacobian and adjoint propagation; original evidence preserved'))


if __name__=='__main__':main()
