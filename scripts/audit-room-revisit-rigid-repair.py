"""Fit exploratory local RGB-D revisit corrections and evaluate held matches."""
import importlib.util,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('revisit',ROOT/'scripts/audit-room-sdk-3d-revisits.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
SRC=ROOT/'results/room_improve_sdk_distributed_revisits_20261103';OUT=ROOT/'results/room_improve_sdk_rigid_repair_20261103'
def fit(a,b):
 ac=a.mean(0);bc=b.mean(0);u,_,vt=np.linalg.svd((a-ac).T@(b-bc));D=np.eye(3);D[-1,-1]=np.linalg.det(vt.T@u.T);R=vt.T@D@u.T;T=np.eye(4);T[:3,:3]=R;T[:3,3]=bc-R@ac;return T
def err(T,a,b):return np.linalg.norm(a@T[:3,:3].T+T[:3,3]-b,axis=1)
def main():
 OUT.mkdir(parents=True,exist_ok=False);images=m.model(m.NEW);K=np.array(json.loads((m.NEW/'calibration_clean.json').read_text())['K']); rows=[]
 for f in sorted(SRC.glob('*.npz')):
  a,b=map(int,f.stem.split('_')); n1=f'frame_{a:06}.png';n2=f'frame_{b:06}.png'
  with np.load(f) as d:x1=d['xa'];x2=d['xb']
  with np.load(m.ROOT/'results/room_improve_full_cloud_dense_checks_20261103'/(n1+'.dense_depth.npz')) as d:z1=d['depth_m']
  with np.load(m.ROOT/'results/room_improve_full_cloud_dense_checks_20261103'/(n2+'.dense_depth.npz')) as d:z2=d['depth_m']
  p,v1=m.sample(z1,x1,K);q,v2=m.sample(z2,x2,K);assert np.all(v1&v2)
  rng=np.random.default_rng(10000*a+b);ids=rng.permutation(len(p));n=len(ids)//2;tr=ids[:n];te=ids[n:];row=dict(a=a,b=b,train=len(tr),check=len(te))
  if n<6:row['reason']='insufficient training correspondences';rows.append(row);continue
  best=np.zeros(n,dtype=bool)
  for k in range(1500):
   pick=rng.choice(n,3,replace=False);pa=p[tr][pick]
   if np.linalg.norm(np.cross(pa[1]-pa[0],pa[2]-pa[0]))<.0002:continue
   T=fit(pa,q[tr][pick]);good=err(T,p[tr],q[tr])<.06
   if good.sum()>best.sum():best=good
  if best.sum()<4:row['reason']='insufficient rigid inliers';rows.append(row);continue
  T=fit(p[tr][best],q[tr][best]);prior=m.base.pose(images[n2])@np.linalg.inv(m.base.pose(images[n1]));before=err(prior,p[te],q[te]);after=err(T,p[te],q[te]);spread=np.linalg.svd(p[tr][best]-p[tr][best].mean(0),compute_uv=False)
  row.update(reason='exploratory',train_inliers=int(best.sum()),train_spread_singular_values_m=spread.tolist(),check_prior_median_m=float(np.median(before)),check_repaired_median_m=float(np.median(after)),check_prior_p90_m=float(np.percentile(before,90)),check_repaired_p90_m=float(np.percentile(after,90)),relative_correction_translation_m=float(np.linalg.norm((T@np.linalg.inv(prior))[:3,3])),relative_correction_rotation_deg=float(np.degrees(Rotation.from_matrix((T@np.linalg.inv(prior))[:3,:3]).magnitude())),transform=T.tolist());rows.append(row)
 (OUT/'report.json').write_text(json.dumps(dict(method='Fixed-seed random half split of previously F-filtered, jointly stable depth feature matches. Fit 3D rigid transform to training half by 6 cm RANSAC and SVD, compare existing SDK graph relative pose to fitted transform on check half. No graph changed.',pairs=rows,caveat='Matches selected by candidate F and same frames contributed to graph. Check split is feature-level, not independent physical structures or scenes; matches often spatially concentrated or planar. Relative translation depends on rotation and camera coordinates. This diagnostic does not establish a reliable graph repair.'),indent=2),encoding='utf8')
 for r in rows:print(r['a'],r['b'],r['reason'],r.get('train_inliers'),r.get('check_prior_median_m'),r.get('check_repaired_median_m'))
if __name__=='__main__':main()
