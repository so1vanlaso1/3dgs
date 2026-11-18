"""Isolated sparse keyframe graph trial using one exploratory room revisit."""
import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/room_improve_sdk_rectified_20261103/rgbd';OUT=ROOT/'results/room_improve_sdk_rigid_repair_20261103'
def main():
 poses=np.load(DATA/'poses.npz')['camtoworlds'];report=json.loads((OUT/'report.json').read_text()); rows=report['pairs'];loop=next(r for r in rows if (r['a'],r['b'])==(202,250));tests=[r for r in rows if r.get('reason')=='exploratory' and (r['a'],r['b'])!=(202,250)]
 ids=np.unique(np.r_[np.arange(0,len(poses),8),len(poses)-1,202,250,[v for r in tests for v in (r['a'],r['b'])]]); ix={v:i for i,v in enumerate(ids)}
 edges=[SimpleNamespace(source_node_id=k,target_node_id=k+1,transformation=np.linalg.inv(poses[j])@poses[i],information=np.eye(6)*100) for k,(i,j) in enumerate(zip(ids[:-1],ids[1:]))]
 edges.append(SimpleNamespace(source_node_id=ix[202],target_node_id=ix[250],transformation=np.array(loop['transform']),information=np.eye(6)*100))
 spec=importlib.util.spec_from_file_location('solver',ROOT/'scripts/sparse-rgbd-pose-graph.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s); result,stats=s.optimize(poses[ids],edges,max_nfev=200)
 correction=result@np.linalg.inv(poses[ids]);all_ids=np.arange(len(poses));rot=Slerp(ids,Rotation.from_matrix(correction[:,:3,:3]))(all_ids).as_matrix();t=np.column_stack([np.interp(all_ids,ids,correction[:,k,3]) for k in range(3)]);new=poses.copy();new[:,:3,:3]=rot@poses[:,:3,:3];new[:,:3,3]=np.einsum('nij,nj->ni',rot,poses[:,:3,3])+t
 def error(p,r):
  d=np.linalg.inv(np.array(r['transform']))@np.linalg.inv(p[r['b']])@p[r['a']];return {'translation_m':float(np.linalg.norm(d[:3,3])),'rotation_deg':float(np.degrees(Rotation.from_matrix(d[:3,:3]).magnitude()))}
 checks=[dict(a=r['a'],b=r['b'],before=error(poses,r),after=error(new,r)) for r in tests]; dis=np.linalg.norm(new[:,:3,3]-poses[:,:3,3],axis=1)
 np.savez_compressed(OUT/'graph_trial_poses.npz',camtoworlds=new,indices=all_ids)
 (OUT/'graph_trial_report.json').write_text(json.dumps(dict(method='Development-only keyframe graph: adjacent edges preserve existing SDK poses; one 202/250 exploratory depth match fit as loop. Other fitted pair transforms as checks. Keyframe correction interpolated to full trajectory.',loop=[202,250],solver=stats,loop_before=error(poses,loop),loop_after=error(new,loop),other_pairs=checks,camera_displacement_percentiles_m=np.percentile(dis,[50,90,100]).tolist(),caveat='This is a simplified graph, not optimization of all original measurements. Fitted constraints are sparse, correlated, planar or repeated and were not manually verified as physical matches; check pair transforms fit their own matches and share recording and graph. The output is an exploratory trajectory only; no training, no geometry accuracy claim.'),indent=2),encoding='utf8');print('checks',checks)
if __name__=='__main__':main()
