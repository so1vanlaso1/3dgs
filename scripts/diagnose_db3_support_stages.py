"""Frozen-pose diagnostic: locate lost support without fitting evaluation depth.

Raw clouds and stricter support candidates are diagnostics, not promoted models.
No source images, geometry, training checkpoints or old evidence are overwritten.
"""
import argparse
import importlib.util
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.spatial import cKDTree

s=importlib.util.spec_from_file_location('pilot',Path(__file__).with_name('db3_local_pilot.py'))
p=importlib.util.module_from_spec(s);s.loader.exec_module(p);m=p.m


def summarize_cloud(cloud, poses, K, boxes, cal):
    records=[]
    for name,regions in boxes.items():
        W=poses[name];q=cloud@W[:3,:3].T+W[:3,3]
        q=q[(q[:,2]>.3)&(q[:,2]<6)]
        uv=q[:,:2]/q[:,2:3]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
        pixels,z,_=m.projection.zbuffer(uv,q[:,2],cal['width'],cal['height'])
        tree=cKDTree(pixels)
        gt=np.load(m.DATA/'depth'/(name+'.npz'))['depth_m']
        lo=cv2.erode(gt,np.ones((3,3),'u1'));hi=cv2.dilate(gt,np.ones((3,3),'u1'))
        stable=(lo>.3)&(hi<6)&((hi-lo)<np.maximum(.03,.02*gt))
        for label,(x0,y0,x1,y1) in regions.items():
            yy,xx=np.mgrid[y0:y1:2,x0:x1:2];xy=np.c_[xx.ravel(),yy.ravel()]
            xy=xy[stable[xy[:,1],xy[:,0]]]
            distance,nn=tree.query(xy);covered=distance<=2
            delta=z[nn[covered]]-gt[xy[covered,1],xy[covered,0]]
            records.append(dict(view=name,region=label,source_samples=len(xy),covered=int(covered.sum()),
                coverage=float(covered.mean()) if len(xy) else 0,
                absolute_median_p90_m=np.percentile(abs(delta),[50,90]).tolist() if len(delta) else None,
                signed_median_m=float(np.median(delta)) if len(delta) else None,
                fraction_all_source_within_5cm=float((abs(delta)<=.05).sum()/len(xy)) if len(xy) else 0))
    return records


def main(out):
    out.mkdir(parents=True,exist_ok=False);start=time.time()
    # Freeze recipe before computing new diagnostic results. Thresholds here are
    # support diagnostics, never replacements for the original R1/R2/R3 exits.
    m.write(out/'protocol.json',dict(region='exhibit',geometry='frozen rate10',fit='all existing fit identities, no eval/reserved frames',
        variants=['selected_90k','supported_original_precap','supported_original_voxels','raw_original','supported_dense','supported_dense_5cm','raw_dense'],
        temporal_support_seconds=[.5,2.5],neighbors=6,min_support=2,max_free_conflicts=0,
        tighter_agreement_m=.05,check='unchanged 15 source-specific regions; stable source samples every 2 pixels; nearest Z within 2px',
        limitation='Exposed development diagnostic, conditional point-cloud comparison, not surface rendering or independent physical metrology. No pose fitting or training on evaluation observations.'))
    geo=m.RUN/'exhibit/rate10';archive=np.load(geo/'checked_poses.npz')
    poses=dict(zip(archive['names'],archive['world_to_camera']))
    rows=m.read(m.DATA/'manifest.json')['frames']
    frames=sorted([f for f in rows if f['region']=='exhibit' and f['role']=='fit' and f['name'] in poses],key=lambda f:f['replay_seconds'])
    reservations={r['rgb_message_id']:r['role'] for r in m.read(m.R1/'prospective_checks.json')['source_frames']}
    assert all(reservations[f['rgb_message_id']]=='fit_development' for f in frames)
    cal=m.read(m.DATA/'calibration_clean.json');K=np.array(cal['K'])
    boxes=m.read(m.RUN/'confirmation_structure_regions.json')['regions_xyxy']
    original=[f for f in frames if f['original_train']]
    clouds={k:[] for k in ('supported_original_precap','raw_original','supported_dense','supported_dense_5cm','raw_dense')}
    stats=[];original_weights=[]
    for f in original:
        with np.load(m.RUN/'exhibit/pilot/dataset/evidence'/(f['name']+'.npz')) as ev:
            clouds['raw_original'].append(ev['world'])
            clouds['supported_original_precap'].append(ev['world'][ev['confidence']>0])
            original_weights.append(ev['confidence'][ev['confidence']>0])
    for i,f in enumerate(frames):
        neighbors=sorted([g for g in frames if .5<=abs(g['replay_seconds']-f['replay_seconds'])<=2.5],key=lambda g:abs(g['replay_seconds']-f['replay_seconds']))[:6]
        broad=p.support_observations(f,neighbors,poses,K,cal)
        tight=p.support_observations(f,neighbors,poses,K,cal,max_support_error_m=.05)
        clouds['raw_dense'].append(broad['world'])
        clouds['supported_dense'].append(broad['world'][broad['confidence']>0])
        clouds['supported_dense_5cm'].append(tight['world'][tight['confidence']>0])
        stats.append(dict(name=f['name'],source_id=f['rgb_message_id'],neighbors=[g['rgb_message_id'] for g in neighbors],
            raw=len(broad['world']),supported=int((broad['confidence']>0).sum()),tight=int((tight['confidence']>0).sum()),
            motion_bound_px=float(broad['motion_bound_px'][0]),
            rejection_counts={str(bit):int(((tight['flags']&bit)>0).sum()) for bit in (8,16,32,64,128)}))
        if (i+1)%20==0:print('SUPPORT',i+1,'/',len(frames),flush=True)
    clouds={k:np.concatenate(v) for k,v in clouds.items()}
    original_cloud=clouds['supported_original_precap'];weights=np.concatenate(original_weights)
    order=np.argsort(-weights,kind='stable');voxel=np.floor(original_cloud/.01).astype('i4')
    _,first=np.unique(voxel[order],axis=0,return_index=True);keep=order[first]
    clouds['supported_original_voxels']=original_cloud[keep]
    clouds['selected_90k']=np.load(m.RUN/'exhibit/pilot/supported_initialization.npz')['xyz']
    capped=keep[np.random.default_rng(42).choice(len(keep),90000,replace=False)] if len(keep)>90000 else keep
    assert np.array_equal(original_cloud[capped],clouds['selected_90k']),'Frozen initializer recipe drift'
    results={}
    for key,cloud in clouds.items():
        results[key]=dict(points=len(cloud),regions=summarize_cloud(cloud,poses,K,boxes,cal))
        print('DIAGNOSED',key,len(cloud),flush=True)
    m.write(out/'results.json',dict(variants=results,fit_frame_stats=stats,fit_count=len(frames),original_train_count=len(original),
        evaluated_source_ids=[f['rgb_message_id'] for f in rows if f['name'] in boxes],
        no_eval_depth_fit=True,elapsed_seconds=time.time()-start,
        input_hashes={str(path.relative_to(m.ROOT)):m.sha(path) for path in (geo/'checked_poses.npz',m.DATA/'manifest.json',m.RUN/'confirmation_structure_regions.json',m.RUN/'exhibit/pilot/supported_initialization.npz')},
        code_hashes={str(path.relative_to(m.ROOT)):m.sha(path) for path in (Path(__file__),Path(p.__file__),Path(m.__file__))}))
    print('DONE',out,round(time.time()-start,1),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    main(parser.parse_args().output_dir)
