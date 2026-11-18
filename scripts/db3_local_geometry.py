"""Bounded R2 research: fresh tracks, fixed-calibration SfM, metric local checks.

Never changes R0/R1 inputs or the contextual baseline. Run `prepare`, then
`map --region matte|exhibit --rate 3|10`, then `validate` separately.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '6')
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3
import time

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'results/db3_r2_r3_20261110'
DATA = ROOT/'data/db3_r2_local_20261110'
R1 = ROOT/'results/db3_r0_r1_20261110'
INPUT = ROOT/'data/db3_r1_sdk_20261110'


def module(name):
    s = importlib.util.spec_from_file_location(name, ROOT/'scripts'/name)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


clean = module('clean-realsense-scene.py')
projection = clean.projection


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p, x): clean.write(Path(p), x)
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def check_tile(uv):
    """Reserve disjoint spatial tiles before mapping, not residual-selected points."""
    return ((np.floor(uv[:, 0]/70).astype(int) +
             2*np.floor(uv[:, 1]/70).astype(int)) % 5) == 0


def sample_depth(z, uv):
    xy = np.rint(uv).astype(int)
    inside = (xy[:, 0] >= 2)&(xy[:, 0] < z.shape[1]-2)&(xy[:, 1] >= 2)&(xy[:, 1] < z.shape[0]-2)
    x, y = np.clip(xy[:, 0], 0, z.shape[1]-1), np.clip(xy[:, 1], 0, z.shape[0]-1)
    lo = cv2.erode(z, np.ones((3,3), 'u1')); hi = cv2.dilate(z, np.ones((3,3), 'u1'))
    d = z[y,x]
    valid = inside & (lo[y,x] > .3)&(hi[y,x] < 6)&((hi[y,x]-lo[y,x]) < np.maximum(.03, .02*d))
    # Conservative uncertainty model, explicitly not a calibrated sensor variance.
    sigma = np.maximum(.01 + .005*d*d, (hi[y,x]-lo[y,x])/2)
    return d, valid, sigma


def backproject(uv, z, K):
    return np.column_stack(((uv[:,0]-K[0,2])/K[0,0]*z,
                            (uv[:,1]-K[1,2])/K[1,1]*z, z))


def mutual(a, b):
    if a is None or b is None or min(len(a),len(b)) < 2: return np.empty((0,2),int)
    bf = cv2.BFMatcher()
    fw = bf.knnMatch(a.astype('f4'), b.astype('f4'), k=2)
    bw = bf.knnMatch(b.astype('f4'), a.astype('f4'), k=2)
    back = {m.queryIdx:m.trainIdx for m,n in bw if m.distance < .75*n.distance}
    return np.array([(m.queryIdx,m.trainIdx) for m,n in fw
                     if m.distance < .75*n.distance and back.get(m.trainIdx) == m.queryIdx],int).reshape(-1,2)


def prepare():
    RUN.mkdir(parents=True,exist_ok=False); DATA.mkdir(parents=True,exist_ok=False)
    cal = read(INPUT/'calibration_clean.json'); manifest = read(INPUT/'capture_manifest.json')
    prospective = read(R1/'prospective_checks.json'); split = read(INPUT/'reconstruction_split.json')
    byid = {f['message_id']:f for f in manifest['frames']}
    historical = {f['message_id'] for f in manifest['frames'] if f['name'] in split['historical_eval']}
    windows = {'matte':[6.,14.], 'exhibit':[64.,72.]}
    protocol = dict(source_sha256=manifest['source_sha256'], source=str(Path(manifest['source']).resolve()),
        windows_seconds=windows, tracking_rates_fps=[3,10], training_rate='original selected train IDs only',
        pixel_sigma_model=.75, depth_sigma_model_m='max(0.01+0.005*z^2, local 3x3 range/2); uncertainty model, not measured certainty',
        intrinsics_fixed=True, original_poses_reused=False, fitted_tracks='spatial tile modulo 5 != 0',
        checks='tile modulo 5 == 0: never inserted into mapping database or metric BA',
        gate=dict(min_registered_fraction=.9,min_check_pairs=3,min_check_matches=100,
                  reprojection_median_px=2.,reprojection_p90_px=4.,separation_median_m=.05,separation_p90_m=.12,
                  max_control_worsening_m=.01,meaningful_separation_gain_m=.01),
        R3=dict(steps=7000,seeds=[42],arms=['off','current','confidence'],depth_lambda=.01,
                initialization='identical support-aware frozen cloud in all depth arms',
                train_geometry='fixed local poses; no pose optimization during training',
                promotion='target depth/duplicate-surface consistency improves by >=0.01m; every checked structure <=0.01m worse; appearance PSNR >= control-0.2dB, SSIM >= control-0.005, LPIPS <= control+0.01; visual target/control review'),
        limits='Exposed development research. No external geometric reference; consistency only, not absolute accuracy. No global merge without distributed verified revisits. R1 precise physical calibration remains partial.')
    write(RUN/'protocol.json',protocol)
    shutil.copyfile(INPUT/'calibration_clean.json',DATA/'calibration_clean.json')
    for d in ('images','depth','features'): (DATA/d).mkdir()
    K=np.array(cal['K']); maps,valid=projection.remap(cal['raw_color'],K,cal['width'],cal['height']); assert valid.all()
    # Native stride 3 (~10 Hz). Original selected identities are added explicitly.
    roles=prospective['source_frames']; native_index={r['rgb_message_id']:j for j,r in enumerate(roles)}
    selected=[]
    for region,(lo,hi) in windows.items():
        for r in roles:
            mid=r['rgb_message_id']; t=r['replay_seconds']
            if lo <= t <= hi and (native_index[mid]%3==0 or mid in byid):
                # Exposed historical eval is localization/evaluation only, never mapping/training.
                role='eval' if mid in historical else ('fit' if r['role']=='fit_development' else r['role'])
                assert role not in ('geometry_check','final_prospective'), 'Window intersects frozen reservation'
                selected.append(dict(r,region=region,name=f'rgb_{mid:07d}.png',role=role,
                    selected_name=byid[mid]['name'] if mid in byid else None,
                    original_train=mid in byid and byid[mid]['name'] in split['train']))
    start=time.time(); source=Path(manifest['source']).resolve(); stat=source.stat()
    sift=cv2.SIFT_create(nfeatures=3500,contrastThreshold=.02)
    with sqlite3.connect(source.as_uri()+'?mode=ro',uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        tid=db.execute("select id from topics where name like '%/Depth_0/image/data'").fetchone()[0]
        dep=db.execute('select id,timestamp from messages where topic_id=? order by timestamp',(tid,)).fetchall()
        stamps=np.array([s for _,s in dep],dtype=np.int64)
        for j,r in enumerate(selected):
            mid=r['rgb_message_id']; ts,blob=db.execute('select timestamp,data from messages where id=?',(mid,)).fetchone()
            assert ts==r['timestamp_ns']
            rgb,rh=clean.capture.CDR(blob).image()
            k=int(np.searchsorted(stamps,ts)); k=min([max(0,k-1),min(len(stamps)-1,k)],key=lambda k:abs(int(stamps[k])-ts))
            did,dts=dep[k]; raw,dh=clean.adapter.decode_depth(db.execute('select data from messages where id=?',(did,)).fetchone()[0])
            assert abs(dh-rh['header_seconds']) < .02
            image=cv2.remap(rgb,maps[:,:,0],maps[:,:,1],cv2.INTER_LINEAR)
            _,_,_,e=projection.align_depth(raw,cal['depth_units'],cal['raw_depth'],cal['depth_to_color_R'],cal['depth_to_color_t_m'],K,cal['width'],cal['height'])
            Image.fromarray(image).save(DATA/'images'/r['name'],compress_level=1)
            np.savez_compressed(DATA/'depth'/(r['name']+'.npz'),depth_m=e['depth_m'],collision=e['multiple_projected_samples'])
            kp,desc=sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),None)
            uv=np.array([k.pt for k in kp],dtype='f4').reshape(-1,2)
            np.savez_compressed(DATA/'features'/(r['name']+'.npz'),uv=uv,desc=desc,check=check_tile(uv))
            r.update(depth_message_id=did,depth_timestamp_ns=dts,rgb_sha256=sha(DATA/'images'/r['name']),
                     depth_sha256=sha(DATA/'depth'/(r['name']+'.npz')),quality=clean.rgb_quality(image))
            if j%30==0: print('PREPARE',j,'/',len(selected),flush=True)
    assert (stat.st_size,stat.st_mtime_ns)==(source.stat().st_size,source.stat().st_mtime_ns)
    write(DATA/'manifest.json',dict(source=str(source),source_sha256=protocol['source_sha256'],frames=selected,
        exclusion_counts={k:sum(r['role']==k for r in selected) for k in set(r['role'] for r in selected)},
        fitting_has_no_reserved_native_or_historical_eval=True,elapsed_seconds=time.time()-start))
    print('PREPARED',len(selected),time.time()-start,flush=True)


def frames_for(region,rate):
    frames=[f for f in read(DATA/'manifest.json')['frames'] if f['region']==region and f['role']=='fit']
    return [f for f in frames if f['original_train']] if rate==3 else frames


def map_region(region,rate):
    import pycolmap as p
    out=RUN/region/f'rate{rate}';out.mkdir(parents=True,exist_ok=False)
    frames=frames_for(region,rate); names=[f['name'] for f in frames]
    cal=read(DATA/'calibration_clean.json');K=np.array(cal['K'])
    write(out/'input.json',dict(frames=frames,calibration_sha256=sha(DATA/'calibration_clean.json')))
    reader=p.ImageReaderOptions(camera_model='PINHOLE',camera_params=','.join(map(str,[K[0,0],K[1,1],K[0,2],K[1,2]])))
    ext=p.FeatureExtractionOptions(num_threads=6);ext.sift.max_num_features=3500
    p.extract_features(out/'colmap.db',DATA/'images',image_names=names,camera_mode=p.CameraMode.SINGLE,reader_options=reader,extraction_options=ext,device=p.Device.cpu)
    # Erase the prospective spatial checks before any matching or mapping.
    counts=[]
    with sqlite3.connect(out/'colmap.db') as db:
        keys=db.execute('select image_id,rows,cols,data from keypoints').fetchall()
        for iid,n,c,b in keys:
            kp=np.frombuffer(b,'f4').reshape(n,c); keep=~check_tile(kp[:,:2])
            dn,dc,dd=db.execute('select rows,cols,data from descriptors where image_id=?',(iid,)).fetchone();assert dn==n
            desc=np.frombuffer(dd,'u1').reshape(dn,dc)
            db.execute('update keypoints set rows=?,data=? where image_id=?',(int(keep.sum()),kp[keep].tobytes(),iid))
            db.execute('update descriptors set rows=?,data=? where image_id=?',(int(keep.sum()),desc[keep].tobytes(),iid))
            counts.append(dict(image_id=iid,fit=int(keep.sum()),excluded=int((~keep).sum())))
    write(out/'feature_exclusions.json',counts)
    matching=p.FeatureMatchingOptions(num_threads=6)
    pairing=p.SequentialPairingOptions(overlap=8,quadratic_overlap=True)
    p.match_sequential(out/'colmap.db',matching_options=matching,pairing_options=pairing,device=p.Device.cpu)
    options=p.IncrementalPipelineOptions(num_threads=6,random_seed=11010,min_model_size=4,
        ba_refine_focal_length=False,ba_refine_principal_point=False,ba_refine_extra_params=False,
        ba_refine_sensor_from_rig=False,max_num_models=5,max_runtime_seconds=900)
    options.mapper.abs_pose_refine_focal_length=False;options.mapper.abs_pose_refine_extra_params=False
    # Local small-baseline pilot; accept weak geometry only through independent checks.
    options.mapper.init_min_tri_angle=4.;options.mapper.filter_min_tri_angle=.5
    start=time.time();models=p.incremental_mapping(out/'colmap.db',DATA/'images',out/'models',options=options)
    summary=[]
    for i,m in models.items():
        summary.append(dict(model=i,images=m.num_reg_images(),points=m.num_points3D(),mean_track_length=m.compute_mean_track_length(),mean_reprojection_error=m.compute_mean_reprojection_error()))
    write(out/'mapping.json',dict(requested=len(frames),models=summary,elapsed_seconds=time.time()-start,
                                  source='fresh RGB feature tracks; fixed calibration; no old poses'))
    if not models: print('NO_MODEL',region,rate,flush=True);return
    best=max(models.values(),key=lambda m:m.num_reg_images());metric_model(best,out,K)


def metric_model(model,out,K):
    """Robust metric scale from fitting landmarks only; full joint depth BA follows."""
    ratios=[]; obs=[]
    for im in model.images.values():
        W=np.asarray(im.cam_from_world().matrix())
        z=np.load(DATA/'depth'/(im.name+'.npz'))['depth_m']
        pts=[pt for pt in im.points2D if pt.has_point3D()]
        if not pts:continue
        uvs=np.array([pt.xy for pt in pts]);ds,vs,ss=sample_depth(z,uvs)
        for pt,uv,d,v,s in zip(pts,uvs,ds,vs,ss):
            if not v:continue
            world=model.points3D[pt.point3D_id].xyz
            pred=(W[:,:3]@world+W[:,3])[2]
            if pred>0:
                ratios.append(d/pred);obs.append((im.image_id,pt.point3D_id,*uv,d,s))
    if len(ratios)<100:write(out/'metric_failure.json',dict(reason='fewer than 100 fitting depth samples'));return
    scale=float(np.median(ratios));disp=float(np.median(np.abs(np.array(ratios)/scale-1)))
    transform=__import__('pycolmap').Sim3d(scale=scale)
    model.transform(transform)
    (out/'metric_rgb').mkdir();model.write(out/'metric_rgb')
    write(out/'scale.json',dict(scale=scale,samples=len(ratios),median_relative_dispersion=disp,
        rule='metric scale from fitting tracks only; no absolute accuracy claim'))
    joint_depth_ba(model,out,K)


def joint_depth_ba(model,out,K):
    """Explicit whitened pixel and metric depth factors on real multi-view tracks."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    import pycolmap as p
    images=sorted(model.images.values(),key=lambda im:im.name)
    # Deterministic support-ranked bounded BA, maintaining the real tracks in export.
    items=sorted(model.points3D.items(),key=lambda item:(-item[1].track.length(),item[0]))[:1800]
    points=[pt for _,pt in items]
    ids={pid:j for j,(pid,_) in enumerate(items)}
    cams=[];ci=[];pi=[];uvs=[];ds=[];sigmas=[]
    for i,im in enumerate(images):
        W=np.asarray(im.cam_from_world().matrix());cams.append(np.r_[Rotation.from_matrix(W[:,:3]).as_rotvec(),W[:,3]])
        z=np.load(DATA/'depth'/(im.name+'.npz'))['depth_m']
        pts=[pt for pt in im.points2D if pt.has_point3D() and pt.point3D_id in ids]
        if not pts:continue
        xy=np.array([pt.xy for pt in pts]);d,v,s=sample_depth(z,xy)
        for pt,uv,dep,valid,sigma in zip(pts,xy,d,v,s):
            ci.append(i);pi.append(ids[pt.point3D_id]);uvs.append(uv);ds.append(dep if valid else 0);sigmas.append(sigma)
    cams=np.array(cams);ci=np.array(ci);pi=np.array(pi);uvs=np.array(uvs);ds=np.array(ds);sigmas=np.array(sigmas)
    points0=np.array([pt.xyz for pt in points]);ncam=len(cams);npt=len(points0);n=len(ci)
    x0=np.r_[cams[1:].ravel(),points0.ravel()];depth_valid=ds>0
    def unpack(x):return np.vstack((cams[0],x[:6*(ncam-1)].reshape(-1,6))),x[6*(ncam-1):].reshape(-1,3)
    def residual(x):
        c,xyz=unpack(x);R=Rotation.from_rotvec(c[:,:3]).as_matrix()
        q=np.einsum('nij,nj->ni',R[ci],xyz[pi])+c[ci,3:]
        z=np.maximum(q[:,2],.01);uv=q[:,:2]/z[:,None]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
        pixels=(uv-uvs)/.75
        depth=np.where(depth_valid,(q[:,2]-ds)/np.maximum(sigmas,.01),0)
        return np.column_stack((pixels,depth)).ravel()
    sparsity=lil_matrix((n*3,len(x0)),dtype=int)
    for j,(c,pt) in enumerate(zip(ci,pi)):
        if c>0:sparsity[j*3:j*3+3,(c-1)*6:c*6]=1
        sparsity[j*3:j*3+3,6*(ncam-1)+pt*3:6*(ncam-1)+(pt+1)*3]=1
    start=time.time();before=residual(x0)
    result=least_squares(residual,x0,jac_sparsity=sparsity.tocsr(),loss='soft_l1',f_scale=1.,max_nfev=45,
        ftol=1e-5,xtol=1e-6,gtol=1e-5,tr_solver='lsmr',tr_options={'maxiter':80},verbose=1)
    c,xyz=unpack(result.x)
    for im,v in zip(images,c):
        im.frame.rig_from_world=p.Rigid3d(p.Rotation3d(Rotation.from_rotvec(v[:3]).as_matrix()),v[3:])
    for pt,q in zip(points,xyz):pt.xyz=q
    # Unoptimized landmarks must be retriangulated/filtered under the changed cameras.
    # They are excluded from the exported metric model, avoiding stale landmarks.
    keep=set(ids)
    for pid in list(model.points3D):
        if pid not in keep:model.delete_point3D(pid)
    model.update_point_3d_errors()
    (out/'metric_joint').mkdir();model.write(out/'metric_joint')
    np.savez_compressed(out/'joint_observations.npz',camera_index=ci,point_index=pi,uv=uvs,depth_m=ds,sigma_m=sigmas,
                        before=before.reshape(-1,3),after=residual(result.x).reshape(-1,3))
    write(out/'joint_ba.json',dict(images=ncam,points=npt,observations=n,depth_factors=int(depth_valid.sum()),
        first_camera_fixed=True,calibration_fixed=True,real_tracks=True,solver='scipy sparse robust pixel+metric-depth BA',
        max_nfev=45,nfev=result.nfev,status=result.status,message=result.message,
        fitted_rms_before=float(np.sqrt(np.mean(before**2))),fitted_rms_after=float(np.sqrt(np.mean(residual(result.x)**2))),
        elapsed_seconds=time.time()-start,warning='Fitted residuals do not establish acceptance. Separate spatial checks required.'))
    print('JOINT_BA',out,ncam,npt,result.status,time.time()-start,flush=True)


def main():
    a=argparse.ArgumentParser(__doc__);a.add_argument('stage',choices=['prepare','map','refine']);a.add_argument('--region',choices=['matte','exhibit']);a.add_argument('--rate',type=int,choices=[3,10],default=3)
    args=a.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='map':map_region(args.region,args.rate)
    else:
        import pycolmap
        out=RUN/args.region/f'rate{args.rate}'
        joint_depth_ba(pycolmap.Reconstruction(out/'metric_rgb'),out,np.array(read(DATA/'calibration_clean.json')['K']))


if __name__=='__main__':main()
