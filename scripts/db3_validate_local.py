"""R2 independent spatial-feature checks and frozen-map evaluation localization."""
import argparse
import importlib.util
from pathlib import Path
import time
import cv2
import numpy as np
import pycolmap
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

spec=importlib.util.spec_from_file_location('local',Path(__file__).with_name('db3_local_geometry.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def pose(im):
    W=np.eye(4);W[:3]=im.cam_from_world().matrix();return W


def quantile(x):
    return [float(v) for v in np.percentile(x,[50,90])] if len(x) else None


def metrics(Wa,Wb,ua,ub,da,db,K):
    a=m.backproject(ua,da,K);b=m.backproject(ub,db,K)
    T=Wb@np.linalg.inv(Wa);pred=a@T[:3,:3].T+T[:3,3]
    pix=pred[:,:2]/np.maximum(pred[:,2:3],.01)*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
    sep=np.linalg.norm(pred-b,axis=1);repro=np.linalg.norm(pix-ub,axis=1)
    return sep,repro


def legacy_pixels(uv,K,cal):
    # Common raw RGB correspondence: SDK ideal -> recorded raw -> old ideal.
    xyz=m.backproject(uv,np.ones(len(uv)),K)
    raw=m.projection.project(xyz,cal['raw_color'],m.projection.SDK)
    kr=m.projection.intrinsic(cal['raw_color']);coeff=m.projection.coefficients(cal['raw_color'])
    old=m.projection.brown((raw-[kr[0,2],kr[1,2]])/[kr[0,0],kr[1,1]],coeff)
    return old*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]


def check_matches(fa,fb):
    a=np.load(m.DATA/'features'/(fa['name']+'.npz'));b=np.load(m.DATA/'features'/(fb['name']+'.npz'))
    ua=a['uv'][a['check']];ub=b['uv'][b['check']]
    pairs=m.mutual(a['desc'][a['check']],b['desc'][b['check']])
    if len(pairs)<20:return np.empty((0,2)),np.empty((0,2)),dict(raw_matches=len(pairs),reason='too few descriptor matches')
    ua,ub=ua[pairs[:,0]],ub[pairs[:,1]]
    # Split check correspondences again: F verification never uses check residuals
    # to tune the candidate map. Remaining half assesses separately fit epipolar F.
    rng=np.random.default_rng(fa['rgb_message_id']+fb['rgb_message_id']);order=rng.permutation(len(ua))
    fit=order[:len(order)//2];check=order[len(order)//2:]
    cv2.setRNGSeed(11010);F,mask=cv2.findFundamentalMat(ua[fit],ub[fit],cv2.FM_RANSAC,1.5,.999)
    if F is None or F.shape!=(3,3):return np.empty((0,2)),np.empty((0,2)),dict(raw_matches=len(pairs),reason='separate F verification failed')
    x=np.c_[ua[check],np.ones(len(check))];y=np.c_[ub[check],np.ones(len(check))]
    line=x@F.T;line2=y@F
    distance=np.abs(np.sum(y*line,axis=1))/np.sqrt(line[:,0]**2+line[:,1]**2+line2[:,0]**2+line2[:,1]**2).clip(1e-9)
    check=check[distance<1.5]
    return ua[check],ub[check],dict(raw_matches=len(pairs),F_fit=len(fit),F_verified_check=len(check),
        selection='mutual ratio 0.75; separate-half RGB F verification; no candidate-pose residual filtering')


def localize_eval(model,region,out,K):
    frames=[f for f in m.read(m.DATA/'manifest.json')['frames'] if f['region']==region and f['role']=='eval']
    descriptor_by_point={}
    for im in model.images.values():
        features=np.load(m.DATA/'features'/(im.name+'.npz'))
        xy=features['uv'];fit=~features['check'];fitidx=np.nonzero(fit)[0]
        if not len(fitidx):continue
        pts=[pt for pt in im.points2D if pt.has_point3D()]
        if not pts:continue
        distance,nn=cKDTree(xy[fit]).query(np.array([pt.xy for pt in pts]))
        for j in np.nonzero(distance<1.2)[0]:
            pid=pts[j].point3D_id
            if pid not in descriptor_by_point:
                descriptor_by_point[pid]=features['desc'][fitidx[nn[j]]]
    pids=sorted(descriptor_by_point)
    descriptors=np.array([descriptor_by_point[pid] for pid in pids]);xyz=np.array([model.points3D[pid].xyz for pid in pids])
    poses={};reports=[]
    for f in frames:
        features=np.load(m.DATA/'features'/(f['name']+'.npz'));fit=~features['check']
        pairs=m.mutual(features['desc'][fit],descriptors);uv=features['uv'][fit][pairs[:,0]];q=xyz[pairs[:,1]]
        row=dict(name=f['name'],source_id=f['rgb_message_id'],matches=len(pairs),accepted=False)
        if len(pairs)>=25:
            cv2.setRNGSeed(11010)
            ok,rv,tv,inliers=cv2.solvePnPRansac(q,uv,K,None,iterationsCount=500,reprojectionError=2.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
            if ok and inliers is not None and len(inliers)>=25:
                ii=inliers[:,0];ok,rv,tv=cv2.solvePnP(q[ii],uv[ii],K,None,rv,tv,True)
                W=np.eye(4);W[:3,:3]=cv2.Rodrigues(rv)[0];W[:3,3]=tv[:,0]
                poses[f['name']]=W
                row.update(accepted=bool(ok),inliers=len(ii),world_to_camera=W.tolist())
        reports.append(row)
    m.write(out/'evaluation_localization.json',dict(frames=reports,
        convention='PnP localization against fixed training landmarks with non-check tiles only. No landmarks, training poses, depth factors or Gaussian parameters fitted to evaluation images. Historical evaluation is exposed development, not untouched.'))
    return poses


def validate(region,rate):
    out=m.RUN/region/f'rate{rate}'
    if (out/'validation.json').exists():raise FileExistsError('Preserve validation; use new namespace for revised protocol')
    model=pycolmap.Reconstruction(out/'metric_joint');rgb=pycolmap.Reconstruction(out/'metric_rgb')
    cal=m.read(m.DATA/'calibration_clean.json');K=np.array(cal['K']);protocol=m.read(m.RUN/'protocol.json')
    rows=m.read(m.DATA/'manifest.json')['frames'];byname={f['name']:f for f in rows}
    new={im.name:pose(im) for im in model.images.values()};pre={im.name:pose(im) for im in rgb.images.values()}
    eval_poses=localize_eval(model,region,out,K);new.update(eval_poses)
    baseline=pycolmap.Reconstruction(m.ROOT/'data/realsense_20261030_134523_clean/rgbd/dataset/sparse')
    old={im.name:pose(im) for im in baseline.images.values()}
    # Original identities provide comparable geometry checks for both rates.
    original=sorted([f for f in rows if f['region']==region and f['selected_name'] and f['name'] in new],key=lambda f:f['timestamp_ns'])
    pairs=[]
    for i,a in enumerate(original):
        for delta in (1,3,6):
            if i+delta<len(original):pairs.append((a,original[i+delta]))
    reports=[];all_sep=[];all_repro=[];old_sep=[];old_repro=[];panels=[]
    for fa,fb in pairs:
        ua,ub,row=check_matches(fa,fb)
        za=np.load(m.DATA/'depth'/(fa['name']+'.npz'))['depth_m'];zb=np.load(m.DATA/'depth'/(fb['name']+'.npz'))['depth_m']
        da,va,_=m.sample_depth(za,ua);db,vb,_=m.sample_depth(zb,ub)
        oa,ob=legacy_pixels(ua,K,cal),legacy_pixels(ub,K,cal)
        doa,voa,_=m.sample_depth(za,oa);dob,vob,_=m.sample_depth(zb,ob)
        good=va&vb&voa&vob
        ua,ub,oa,ob=ua[good],ub[good],oa[good],ob[good]
        da,db,doa,dob=da[good],db[good],doa[good],dob[good]
        row.update(a=fa['name'],b=fb['name'],a_original=fa['selected_name'],b_original=fb['selected_name'],
            seconds=[fa['replay_seconds'],fb['replay_seconds']],stable_common_depth_matches=int(good.sum()),
            contains_localized_eval=fa['role']=='eval' or fb['role']=='eval')
        if len(ua)<10:row['reason']='fewer than 10 stable common check observations';reports.append(row);continue
        sep,repro=metrics(new[fa['name']],new[fb['name']],ua,ub,da,db,K)
        os,orr=metrics(old[fa['selected_name']],old[fb['selected_name']],oa,ob,doa,dob,K)
        hull=float(cv2.contourArea(cv2.convexHull(ua.astype('f4')))/(cal['width']*cal['height']))
        # 3D support degeneracy is reported independently from connectivity.
        singular=np.linalg.svd(np.cov(m.backproject(ua,da,K).T),compute_uv=False)
        row.update(candidate_separation_m=quantile(sep),baseline_separation_m=quantile(os),
                   candidate_reprojection_px=quantile(repro),baseline_reprojection_px=quantile(orr),
                   hull_fraction=hull,point_covariance_eigenvalues_m2=singular.tolist(),
                   planar_or_degenerate=bool(singular[-1]/max(singular[0],1e-9)<.005))
        if fa['name'] in pre and fb['name'] in pre:
            ps,pr=metrics(pre[fa['name']],pre[fb['name']],ua,ub,da,db,K)
            row.update(rgb_only_scaled_separation_m=quantile(ps),rgb_only_scaled_reprojection_px=quantile(pr))
        reports.append(row)
        all_sep.extend(sep);all_repro.extend(repro);old_sep.extend(os);old_repro.extend(orr)
        if len(panels)<6 and hull>.025:
            left=np.array(Image.open(m.DATA/'images'/fa['name']));right=np.array(Image.open(m.DATA/'images'/fb['name']))
            sheet=Image.fromarray(np.concatenate((left,right),axis=1));draw=ImageDraw.Draw(sheet)
            for a,b in zip(ua[::max(1,len(ua)//25)],ub[::max(1,len(ub)//25)]):
                draw.line((tuple(a),tuple(b+[cal['width'],0])),fill=(255,90,0),width=1)
            sheet.thumbnail((1000,300));panels.append(sheet)
    goodrows=[r for r in reports if 'candidate_separation_m' in r]
    qsep=quantile(all_sep);qr=quantile(all_repro);baseq=quantile(old_sep)
    registered=model.num_reg_images()/len(m.frames_for(region,rate));gate=protocol['gate']
    controls=[r for r in goodrows if r['baseline_separation_m'][0]<.05]
    control_ok=all(r['candidate_separation_m'][0]<=r['baseline_separation_m'][0]+gate['max_control_worsening_m'] for r in controls)
    checks=dict(registration=registered>=gate['min_registered_fraction'],enough_checks=len(goodrows)>=gate['min_check_pairs'] and len(all_sep)>=gate['min_check_matches'],
        reprojection=bool(qr and qr[0]<=gate['reprojection_median_px'] and qr[1]<=gate['reprojection_p90_px']),
        separation=bool(qsep and qsep[0]<=gate['separation_median_m'] and qsep[1]<=gate['separation_p90_m']),
        controls_not_worse=control_ok)
    result=dict(region=region,rate=rate,registered_fraction=registered,registered_frames=model.num_reg_images(),
        real_points=model.num_points3D(),mean_track_length=model.compute_mean_track_length(),fitted_reprojection_px=model.compute_mean_reprojection_error(),
        check_pair_count=len(goodrows),check_matches=len(all_sep),candidate_separation_m=qsep,baseline_separation_m=baseq,
        candidate_reprojection_px=qr,baseline_reprojection_px=quantile(old_repro),
        control_pairs=len(controls),gates=checks,local_consistency_pass=all(checks.values()),
        meaningful_gain=bool(qsep and baseq and qsep[0]<baseq[0]-gate['meaningful_separation_gain_m']),
        pair_results=reports,absolute_geometry_accuracy_established=False,
        warning='Correlated feature/depth consistency on exposed development; no independent calibrated reference or whole-recording placement. RGB F filtering and depth stability limit the evaluated support.')
    m.write(out/'validation.json',result)
    if panels:
        sheet=Image.new('RGB',(1000,sum(im.height for im in panels)),(255,255,255));y=0
        for im in panels:sheet.paste(im,(0,y));y+=im.height
        sheet.save(out/'check_matches.jpg')
    # Trajectory in local coordinates, retained as data without arbitrary global merge.
    names=list(new);np.savez_compressed(out/'checked_poses.npz',names=np.array(names),world_to_camera=np.stack([new[n] for n in names]))
    print('VALIDATION',region,rate,{k:v for k,v in result.items() if k!='pair_results'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--region',choices=['matte','exhibit'],required=True);p.add_argument('--rate',type=int,choices=[3,10],required=True)
    a=p.parse_args();validate(a.region,a.rate)
