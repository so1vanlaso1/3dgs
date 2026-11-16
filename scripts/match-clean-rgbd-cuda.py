"""Exact CUDA SIFT descriptor matches for the metric RGB-D pose branch.

L2 distance <=0.7, Lowe ratio <0.8, reciprocal nearest neighbors. These are
candidate correspondences: robust metric pose verification remains mandatory.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import torch


@torch.no_grad()
def match_descriptors(a, b):
    if len(a)==0 or len(b)<2:
        return np.empty((0,2),dtype=np.uint32)
    distances=torch.cdist(a,b)
    values,nearest=torch.topk(distances,2,dim=1,largest=False)
    source=torch.arange(len(a),device=a.device)
    keep=(values[:,0]<.7)&(values[:,0]<.8*values[:,1])&(distances.argmin(dim=0)[nearest[:,0]]==source)
    return torch.stack([source[keep],nearest[keep,0]],dim=1).cpu().numpy().astype(np.uint32)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    a=ap.parse_args();torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA matching requires the verified PyTorch GPU environment')
    cache=OrderedDict();times=[];new=0;skipped=0;matches=0
    with sqlite3.connect(a.root/'colmap.db') as db:
        rows=db.execute('select image_id,name from images order by name').fetchall()
        existing={pid for pid, in db.execute('select pair_id from matches')}
        def descriptor(i):
            if i in cache:
                cache.move_to_end(i);return cache[i]
            n,c,blob=db.execute('select rows,cols,data from descriptors where image_id=?',(i,)).fetchone()
            value=torch.from_numpy(np.frombuffer(blob,dtype=np.uint8).reshape(n,c).copy()).float().cuda()/512.
            cache[i]=value
            while len(cache)>10:cache.popitem(last=False)
            return value
        for j,(iid,name) in enumerate(rows):
            for delta in (1,2,3,4):
                if j+delta>=len(rows):continue
                other=rows[j+delta][0];lo,hi=sorted((iid,other));pid=lo*2147483647+hi
                if pid in existing:
                    skipped+=1;continue
                x,y=descriptor(lo),descriptor(hi)
                started=time.perf_counter();pairs=match_descriptors(x,y);times.append(time.perf_counter()-started)
                db.execute('insert into matches(pair_id,rows,cols,data) values(?,?,?,?)',(pid,len(pairs),2,pairs.tobytes()))
                existing.add(pid);new+=1;matches+=len(pairs)
            if j%100==0:
                db.commit()
                print('CUDA_MATCH',j+1,'/',len(rows),'new_pairs',new,'cached_pairs',skipped,flush=True)
        db.commit()
        total=db.execute('select count(*) from matches').fetchone()[0]
    report=dict(method='Exact L2 SIFT matching on PyTorch CUDA; uint8 descriptors / 512',overlap=4,
                maximum_l2_distance=.7,lowe_ratio=.8,reciprocal_nearest_neighbor=True,
                new_pairs=new,cached_pairs=skipped,total_database_pairs=total,new_correspondences=matches,
                median_seconds_per_new_pair=float(np.median(times)) if times else None,
                peak_torch_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
                geometric_verification='Mandatory downstream metric 3D RANSAC/PnP, gated odometry and pose graph',
                cached_policy='Preserved existing COLMAP CPU matches and any stored epipolar inliers')
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2))
    print('CUDA_MATCH_COMPLETE',report,flush=True)


if __name__=='__main__':main()
