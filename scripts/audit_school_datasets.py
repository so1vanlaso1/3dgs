"""Read-only dataset QA; flags are candidates and never delete source frames."""
import argparse
import hashlib
from pathlib import Path
import sys
import cv2
import numpy as np
from research_campaign import REPO, write
sys.path.insert(0,str(REPO/'examples'))
from datasets.colmap import Parser


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,nargs='+',required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    summary={}
    for folder in a.data:
        parser=Parser(str(folder),factor=1,normalize=True,test_every=8,load_exposure=False)
        poses=parser.camtoworlds
        assert np.isfinite(poses).all() and np.isfinite(parser.points).all()
        rotations=poses[:,:3,:3]
        rotation_error=np.max(np.abs(rotations.transpose(0,2,1)@rotations-np.eye(3)))
        assert rotation_error<1e-3 and np.all(np.linalg.det(rotations)>.99)
        rows=[]
        hashes={}
        for i,(name,path) in enumerate(zip(parser.image_names,parser.image_paths)):
            image=cv2.imread(path)
            if image is None: raise RuntimeError('Unreadable image '+path)
            h,w=image.shape[:2]
            gray=cv2.cvtColor(cv2.resize(image,(320,240)),cv2.COLOR_BGR2GRAY)
            digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
            hashes.setdefault(digest,[]).append(name)
            rows.append(dict(name=name,split='heldout' if i%8==0 else 'train',
                width=w,height=h,sha256=digest,laplacian_variance=float(cv2.Laplacian(gray,cv2.CV_64F).var())))
        scores=np.array([r['laplacian_variance'] for r in rows])
        duplicate_groups=[names for names in hashes.values() if len(names)>1]
        rec=dict(data=str(folder.resolve()),images=len(rows),train=sum(r['split']=='train' for r in rows),
            heldout=sum(r['split']=='heldout' for r in rows),points=len(parser.points),
            rotation_orthogonality_max_error=float(rotation_error),
            exact_duplicate_groups=duplicate_groups,
            blur_score_percentiles=np.percentile(scores,[0,5,50,95,100]).tolist(),
            low_texture_or_blur_candidates=[rows[i]['name'] for i in np.argsort(scores)[:8]],
            rows=rows,limits='Laplacian score is affected by texture/resizing. Candidates require visual review; no frames removed. Rigid/finite poses are not physical accuracy. Camera coverage is not surface completeness.')
        write(a.output/(folder.name+'.json'),rec)
        summary[folder.name]={k:v for k,v in rec.items() if k!='rows'}
        write(a.output/'summary.json',summary)
        print('AUDITED',folder.name,len(rows),flush=True)


if __name__=='__main__': main()
