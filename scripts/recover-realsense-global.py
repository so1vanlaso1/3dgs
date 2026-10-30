"""Add SIFT visual-word loop candidates, then remap with lower-baseline settings."""
import json
import os
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import sqlite3
from pathlib import Path
import cv2
import numpy as np
import pycolmap as p
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

root = Path('data/realsense_20261029_140739')
dbpath = root/'colmap.db'
rng = np.random.default_rng(42)
with sqlite3.connect(dbpath) as db:
    rows = db.execute('select i.image_id,i.name,d.rows,d.cols,d.data from images i join descriptors d on i.image_id=d.image_id order by i.name').fetchall()
features = []
for _,_,n,c,b in rows:
    a = np.frombuffer(b,np.uint8).reshape(n,c).astype('f4')
    a = np.sqrt(a / np.maximum(a.sum(axis=1,keepdims=True),1e-8))
    features.append(a)
samples = np.concatenate([a[rng.choice(len(a),min(80,len(a)),replace=False)] for a in features])
print('Learning visual words', samples.shape, flush=True)
with threadpool_limits(limits=8):
    km = MiniBatchKMeans(n_clusters=512, batch_size=4096, random_state=42, n_init=1, max_iter=50).fit(samples)
    hist = np.stack([np.bincount(km.predict(a),minlength=512) for a in features]).astype('f4')
hist *= np.log((len(rows)+1)/(1+(hist>0).sum(axis=0)))
hist = np.sqrt(hist)
hist /= np.maximum(np.linalg.norm(hist,axis=1,keepdims=True),1e-8)
similarity = hist @ hist.T
pairs = set()
for i in range(len(rows)):
    for j in np.argsort(-similarity[i])[:31]:
        if abs(i-int(j)) > 12:
            pairs.add(tuple(sorted((rows[i][1],rows[int(j)][1]))))
pairpath = root/'retrieval_pairs.txt'
pairpath.write_text('\n'.join(' '.join(v) for v in sorted(pairs))+'\n')
print('Matching extra pairs',len(pairs),flush=True)
mo = p.FeatureMatchingOptions()
mo.num_threads = 8
po = p.ImportedPairingOptions()
po.match_list_path = str(pairpath)
p.match_image_pairs(dbpath, matching_options=mo, pairing_options=po, device=p.Device.cpu)
options = p.IncrementalPipelineOptions()
options.num_threads = 8
options.random_seed = 42
options.max_num_models = 12
options.min_model_size = 15
options.init_num_trials = 50
options.mapper.init_min_tri_angle = 4
options.mapper.filter_min_tri_angle = .5
options.triangulation.min_angle = .5
options.max_runtime_seconds = 600
out = root/'reconstruction_global'
out.mkdir(exist_ok=True)
models = p.incremental_mapping(dbpath,root/'raw_images',out,options=options)
report = dict(selected_frames=len(rows), depth_used=False, matching='sequential plus top-30 TF-IDF RootSIFT visual words',
              models=[dict(id=i,registered_images=m.num_reg_images(),points3D=m.num_points3D(),mean_reprojection_error=m.compute_mean_reprojection_error()) for i,m in models.items()])
(root/'global_reconstruction_report.json').write_text(json.dumps(report,indent=2))
print('GLOBAL_COMPLETE',report,flush=True)
