"""Experimental sparse keyframe correction; never overwrites source poses."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path('data/realsense_20261030_111354_all30/rgbd')
OUT = Path('results/courtyard_revisit_repair_20261102')


def residual(p, i, j, T):
    d = np.linalg.inv(T) @ np.linalg.inv(p[j]) @ p[i]
    return dict(translation_m=float(np.linalg.norm(d[:3,3])),
                rotation_deg=float(np.degrees(Rotation.from_matrix(d[:3,:3]).magnitude())))


def main():
    source=np.load(ROOT/'poses.npz')['camtoworlds']
    rows=json.loads((OUT/'static_revisit_pairs.json').read_text())['pairs']
    train=[r for r in rows if r.get('accepted') and r.get('validation')=='candidate']
    held=[r for r in rows if r.get('accepted') and r.get('validation')=='held-out']
    if len(train)<2 or not held: raise RuntimeError('Insufficient independent revisit checks')
    ids=np.unique(np.r_[np.arange(0,len(source),30),len(source)-1,
                        [x for r in train+held for x in (r['i'],r['j'])]])
    lookup={v:i for i,v in enumerate(ids)}
    loops=[SimpleNamespace(source_node_id=lookup[r['i']],target_node_id=lookup[r['j']],
           transformation=np.array(r['transform']),information=np.eye(6)*100) for r in train]
    edges=[SimpleNamespace(source_node_id=k,target_node_id=k+1,
        transformation=np.linalg.inv(source[j])@source[i],information=np.eye(6)*100)
        for k,(i,j) in enumerate(zip(ids[:-1],ids[1:]))] + loops
    # Initialize by smoothly distributing the measured global correction after
    # the first revisit; a robust loss initialized at the old graph can ignore
    # the large, valid loop as an outlier. This is an experiment, not ground truth.
    ref=train[1]; i,j=ref['i'],ref['j']; T=np.array(ref['transform'])
    D=source[i]@np.linalg.inv(T)@np.linalg.inv(source[j])
    interp=Slerp([0,1],Rotation.from_matrix(np.stack([np.eye(3),D[:3,:3]])))
    initial=source[ids].copy()
    for k,n in enumerate(ids):
        a=float(np.clip((n-i)/(j-i),0,1)); correction=np.eye(4)
        correction[:3,:3]=interp(a).as_matrix(); correction[:3,3]=a*D[:3,3]
        initial[k]=correction@initial[k]
    spec=importlib.util.spec_from_file_location('solver',Path('scripts/sparse-rgbd-pose-graph.py'))
    solver=importlib.util.module_from_spec(spec);spec.loader.exec_module(solver)
    result,stats=solver.optimize(initial,edges,max_nfev=800)
    # Interpolate the estimated world-frame correction, preserving every original
    # high-frequency local pose rather than interpolating the camera trajectory.
    corrections=result @ np.linalg.inv(source[ids])
    rotations=Slerp(ids, Rotation.from_matrix(corrections[:,:3,:3]))
    all_ids=np.arange(len(source)); corrected=source.copy()
    corrR=rotations(all_ids).as_matrix()
    corrT=np.column_stack([np.interp(all_ids,ids,corrections[:,k,3]) for k in range(3)])
    corrected[:,:3,:3]=corrR@source[:,:3,:3]
    corrected[:,:3,3]=np.einsum('nij,nj->ni',corrR,source[:,:3,3])+corrT
    before=[residual(source,r['i'],r['j'],np.array(r['transform'])) for r in held]
    after=[residual(corrected,r['i'],r['j'],np.array(r['transform'])) for r in held]
    local_before=np.linalg.norm(np.diff(source[:,:3,3],axis=0),axis=1)
    local_after=np.linalg.norm(np.diff(corrected[:,:3,3],axis=0),axis=1)
    report=dict(method='experimental keyframe pose graph with two neighboring revisit constraints; original graph poses define local relative edges',
        keyframes=len(ids),solver=stats,training_loop_count=len(train),heldout_loop_count=len(held),
        heldout_before=before,heldout_after=after,
        frame_motion_change_percentiles_m=np.percentile(np.abs(local_after-local_before),[50,90,99,100]).tolist(),
        caveat='Only one revisit area; heldout pair is nearby and not independent of the local trajectory. Not ground truth.',
        passed=bool(stats['status']>0 and all(x['translation_m']<.5 and x['rotation_deg']<8 for x in after)))
    np.savez_compressed(OUT/'candidate_poses.npz',indices=np.arange(len(source)),camtoworlds=corrected)
    (OUT/'candidate_pose_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
