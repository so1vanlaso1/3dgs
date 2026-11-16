"""R3 matched local depth-policy pilots, gated by R2 spatial checks."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import cv2
import numpy as np
import pycolmap
from PIL import Image
from scipy.spatial.transform import Rotation

s=importlib.util.spec_from_file_location('local',Path(__file__).with_name('db3_local_geometry.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def matrix(im):
    W=np.eye(4);W[:3]=im.cam_from_world().matrix();return W


def support_observations(f,neighbors,poses,K,cal,max_support_error_m=None):
    """Observed depth only. Missing and occluded samples are never free space."""
    z=np.load(m.DATA/'depth'/(f['name']+'.npz'))['depth_m']
    yy,xx=np.mgrid[2:z.shape[0]-2:3,2:z.shape[1]-2:3];uv=np.c_[xx.ravel(),yy.ravel()].astype('f4')
    dep,valid,sigma=m.sample_depth(z,uv)
    valid &= ~m.check_tile(uv)
    uv,dep,sigma=uv[valid],dep[valid],sigma[valid]
    xyz=m.backproject(uv,dep,K);W=poses[f['name']];C=np.linalg.inv(W)
    world=xyz@C[:3,:3].T+C[:3,3]
    support=np.zeros(len(uv),'i2');free=np.zeros(len(uv),'i2');occluded=np.zeros(len(uv),'i2');missing=np.zeros(len(uv),'i2');relative=[]
    for other in neighbors:
        T=poses[other['name']]@C;q=xyz@T[:3,:3].T+T[:3,3]
        p=q[:,:2]/np.maximum(q[:,2:3],.01)*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
        oz=np.load(m.DATA/'depth'/(other['name']+'.npz'))['depth_m'];d,v,sg=m.sample_depth(oz,p)
        v &= q[:,2]>.3; tolerance=np.maximum(.03,2*np.sqrt(sigma*sigma+sg*sg))
        if max_support_error_m is not None:
            if not np.isfinite(max_support_error_m) or max_support_error_m<=0:
                raise ValueError('Support error limit must be finite and positive')
            tolerance=np.minimum(tolerance,max_support_error_m)
        error=q[:,2]-d;agree=v&(np.abs(error)<tolerance)
        support+=agree;free+=v&(error < -tolerance);occluded+=v&(error > tolerance);missing+=~v
        relative.append(Rotation.from_matrix(T[:3,:3]).magnitude()/max(abs(f['replay_seconds']-other['replay_seconds']),.1))
    xy=np.rint(uv).astype(int);image=np.array(Image.open(m.DATA/'images'/f['name']))
    color=image[xy[:,1],xy[:,0]]
    # Incidence model from observed local depth derivatives; gaps remain uncertain.
    dzx=(z[xy[:,1],xy[:,0]+1]-z[xy[:,1],xy[:,0]-1])/2
    dzy=(z[xy[:,1]+1,xy[:,0]]-z[xy[:,1]-1,xy[:,0]])/2
    incidence=1/np.sqrt(1+(K[0,0]*dzx/dep)**2+(K[1,1]*dzy/dep)**2)
    pair_gap=abs(f['timestamp_ns']-f['depth_timestamp_ns'])/1e9
    motion_px=K[0,0]*(np.median(relative) if relative else 0)*pair_gap
    motion_weight=float(np.exp(-.5*motion_px*motion_px))
    saturation=(color.max(axis=1)>249)|(color.max(axis=1)<8)
    stability=np.minimum(1,.025/np.maximum(sigma,.01))
    confidence=np.minimum(1,support/3)*np.exp(-free)*stability*incidence*motion_weight
    supported=(support>=2)&(free==0)&(~saturation)&(incidence>.25)&(motion_px<2.)
    confidence[~supported]=0
    flags=(np.where(support>=2,1,0)|np.where(missing>0,2,0)|np.where(occluded>0,4,0)|
           np.where(free>0,8,0)|np.where(motion_px>=2,16,0)|np.where(incidence<=.25,32,0)|
           np.where(saturation,64,0)|np.where(support<2,128,0)).astype('u2')
    return dict(points=uv,depths=dep,confidence=confidence.astype('f4'),world=world,color=color,
                support=support,free_conflicts=free,occluded=occluded,missing=missing,flags=flags,
                sigma_m=sigma.astype('f4'),incidence=incidence.astype('f4'),motion_bound_px=np.full(len(uv),motion_px,'f4'))


def prepare(region,rate):
    geo=m.RUN/region/f'rate{rate}';check=m.read(geo/'validation.json')
    if not check['local_consistency_pass']:raise RuntimeError('R2 local gate failed: no R3 training permitted')
    out=m.RUN/region/'pilot';out.mkdir(exist_ok=False)
    dataset=out/'dataset';dataset.mkdir();(dataset/'images').mkdir();(dataset/'depth').mkdir();(dataset/'evidence').mkdir()
    model=pycolmap.Reconstruction(geo/'metric_joint');poses={im.name:matrix(im) for im in model.images.values()}
    rows=m.read(m.DATA/'manifest.json')['frames'];frames=sorted([f for f in rows if f['region']==region and f['original_train'] and f['name'] in poses],key=lambda f:f['replay_seconds'])
    cal=m.read(m.DATA/'calibration_clean.json');K=np.array(cal['K']);cloud=[];colors=[];weights=[];cloud_source=[];stats=[]
    for i,f in enumerate(frames):
        candidates=sorted([g for g in frames if .5<=abs(g['replay_seconds']-f['replay_seconds'])<=2.5],key=lambda g:abs(g['replay_seconds']-f['replay_seconds']))[:6]
        record=support_observations(f,candidates,poses,K,cal)
        np.savez_compressed(dataset/'evidence'/(f['name']+'.npz'),**record)
        rng=np.random.default_rng(11010+f['rgb_message_id']);indices=rng.choice(len(record['depths']),min(4096,len(record['depths'])),replace=False)
        np.savez_compressed(dataset/'depth'/(f['name']+'.npz'),**{key:record[key][indices] for key in ('points','depths','confidence')})
        supported=record['confidence']>0
        cloud.extend(record['world'][supported]);colors.extend(record['color'][supported]);weights.extend(record['confidence'][supported]);cloud_source.extend([f['rgb_message_id']]*int(supported.sum()))
        stats.append(dict(name=f['name'],source_id=f['rgb_message_id'],candidate_samples=len(record['depths']),supported=int(supported.sum()),
                          positive_weight_samples=int((record['confidence'][indices]>0).sum()),neighbors=[g['rgb_message_id'] for g in candidates],
                          confidence_mean=float(record['confidence'].mean()),motion_bound_px=float(record['motion_bound_px'][0])))
    xyz=np.asarray(cloud);rgb=np.asarray(colors);confidence=np.asarray(weights);source=np.asarray(cloud_source)
    if len(xyz)<1000:raise RuntimeError('Insufficient supported initialization')
    # Highest-evidence representative in 1 cm voxels, no hole filling.
    order=np.argsort(-confidence,kind='stable');voxel=np.floor(xyz/.01).astype('i4')
    _,first=np.unique(voxel[order],axis=0,return_index=True);keep=order[first]
    if len(keep)>90000:keep=keep[np.random.default_rng(42).choice(len(keep),90000,replace=False)]
    xyz,rgb=xyz[keep],rgb[keep]
    np.savez_compressed(out/'supported_initialization.npz',xyz=xyz,rgb=rgb,confidence=confidence[keep],source_id=source[keep])
    # This initialization is a fused sensor cloud, not the real landmark model.
    # The original tracked model stays in the geometry folder for geometry work.
    for pid in list(model.points3D):model.delete_point3D(pid)
    for q,c in zip(xyz,rgb):model.add_point3D(q,pycolmap.Track(),c)
    eval_rows=m.read(geo/'evaluation_localization.json')['frames'];eval_names=[]
    for f in eval_rows:
        if not f['accepted']:continue
        W=np.array(f['world_to_camera']);iid=max(model.images)+1
        im=pycolmap.Image(name=f['name'],camera_id=next(iter(model.cameras)),image_id=iid)
        model.add_image_with_trivial_frame(im,pycolmap.Rigid3d(pycolmap.Rotation3d(W[:3,:3]),W[:3,3]));eval_names.append(f['name'])
    if len(eval_names)<2:raise RuntimeError('Need at least two localized evaluation frames')
    (dataset/'sparse').mkdir();model.write(dataset/'sparse')
    for im in model.images.values():os.link(m.DATA/'images'/im.name,dataset/'images'/im.name)
    train_names=sorted(f['name'] for f in frames);m.write(dataset/'experiment_split.json',dict(train=train_names,val=sorted(eval_names)))
    sequence=[];rng=np.random.default_rng(42)
    while len(sequence)<7000:sequence.extend(rng.permutation(len(train_names)).tolist())
    m.write(out/'camera_sequence.json',sequence[:7000])
    m.write(out/'initialization_report.json',dict(geometry=str(geo),geometry_validation_sha256=m.sha(geo/'validation.json'),
        initialization_sha256=m.sha(out/'supported_initialization.npz'),initial_points=len(xyz),train_count=len(train_names),eval_count=len(eval_names),
        stats=stats,flags={'1':'>=2 separated supporting views','2':'missing elsewhere, unknown','4':'occluded elsewhere, unknown',
                         '8':'measured free-space conflict','16':'timing/motion bound >=2px','32':'grazing/unstable local normal','64':'clipped/dark RGB','128':'insufficient cross-view support'},
        uncertainty_policy='Only supported opaque-compatible points initialize. Other raw observations and all RGB remain preserved in source/evidence. Reflection/dynamics not automatically confirmed. No hole filling.',
        depth_policy='4096 shared valid fitting-tile observations/frame. Current arm uses same samples with unweighted historical disparity loss. Confidence arm normalizes positive evidence weights. Off is appearance optimization on RGB-D-derived fixed geometry.',
        metric_accuracy='Local consistency only, no independent reference',scene_merge='No global placement assigned'))
    print('INITIALIZATION',region,len(xyz),len(train_names),len(eval_names),flush=True)


def train(region,arms,steps=7000):
    out=m.RUN/region/'pilot';report=m.read(out/'initialization_report.json');dataset=out/'dataset'
    env=os.environ.copy();env['PYTHONPATH']=str(m.ROOT)+os.pathsep+str(m.ROOT/'examples')+os.pathsep+env.get('PYTHONPATH','')
    replay=out/'camera_sequence.json'
    if steps!=7000:
        sequence=[];rng=np.random.default_rng(42)
        while len(sequence)<steps:sequence.extend(rng.permutation(report['train_count']).tolist())
        assert sequence[:7000]==m.read(replay)
        replay=out/f'camera_sequence_{steps}.json'
        if replay.exists():assert m.read(replay)==sequence[:steps]
        else:m.write(replay,sequence[:steps])
    for arm in arms:
        target=out/arm
        if target.exists():raise FileExistsError(f'Preserving {target}')
        args=[sys.executable,'-u','examples/simple_trainer.py','budget','--seed','42','--independent-sampling','--record-experiment',
            '--data-dir',str(dataset),'--data-factor','1','--result-dir',str(target),
            '--max-steps',str(steps),'--eval-steps','1',str(steps//2),str(steps),'--save-steps',str(steps),'--disable-viewer','--disable-video',
            '--batch-size','1','--packed','--sh-degree','3','--sh-degree-interval','1000','--cache-images','--data-workers','0',
            '--camera-sequence-path',str(replay),'--strategy.cap-max','180000',
            '--strategy.refine-start-iter','500','--strategy.refine-stop-iter','6000','--strategy.refine-every','100','--strategy.reset-every','3000']
        if arm!='off':args+=['--depth-loss','--sensor-depth-dir',str(dataset/'depth'),'--depth-lambda','.05' if arm=='confidence_supported_15k' else '.01']
        if arm.startswith('confidence'):args+=['--depth-confidence']
        if arm=='confidence_supported_15k':args+=['--sensor-alpha-lambda','.02']
        m.write(out/(arm+'_command.json'),dict(command=args,geometry_and_initialization=report,steps=steps,
            source_sha256={name:m.sha(m.ROOT/name) for name in ('examples/simple_trainer.py','examples/datasets/colmap.py','gsplat/losses.py','scripts/db3_local_pilot.py')}))
        m.write(out/'status.json',dict(state='training',arm=arm));start=time.time()
        print('TRAIN_START',region,arm,flush=True)
        with (out/(arm+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(args,cwd=m.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            m.write(out/'status.json',dict(state='failed',arm=arm,returncode=result.returncode));raise RuntimeError(f'{arm} training failed; inspect log')
        m.write(out/(arm+'_runtime.json'),dict(wall_seconds=time.time()-start))
        print('TRAIN_DONE',region,arm,time.time()-start,flush=True)
    m.write(out/'status.json',dict(state='requested_arms_finished',arms=arms))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['prepare','train']);p.add_argument('--region',choices=['matte','exhibit'],required=True);p.add_argument('--rate',type=int,default=10);p.add_argument('--steps',type=int,default=7000);p.add_argument('--arms',nargs='+',choices=['off','current','current_safe','confidence','confidence_supported_15k'],default=['off','current','confidence'])
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.region,a.rate)
    else:train(a.region,a.arms,a.steps)
