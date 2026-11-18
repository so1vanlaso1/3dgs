"""Validate static RGB-D revisit candidates without changing the source graph."""
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path('data/realsense_20261030_111354_all30')
OUT = Path('results/courtyard_revisit_repair_20261102')
EARLY = [3176, 3206, 3236]
LATE = [6440, 6470, 6500, 6530]


def fit(a, b):
    ac, bc = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a-ac).T @ (b-bc))
    if np.linalg.det(vt.T @ u.T) < 0:
        vt[-1] *= -1
    R = vt.T @ u.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, bc - R @ ac
    return T


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    K = np.array(json.loads((ROOT/'calibration_clean.json').read_text())['K'])
    poses = np.load(ROOT/'rgbd/poses.npz')['camtoworlds']
    sift = cv2.SIFT_create(nfeatures=4000)
    features = {}
    for frame in EARLY + LATE:
        im = cv2.imread(str(ROOT/'rgbd/dataset/images'/f'frame_{frame:06}.png'), 0)
        mask = np.full(im.shape, 255, np.uint8)
        if frame in EARLY:
            # Broad manual exclusion around the group in the early revisit.
            mask[130:340, 325:525] = 0
        kp, des = sift.detectAndCompute(im, mask)
        depth = np.load(ROOT/'aligned_depth_meters'/f'frame_{frame:06}.png.npz')
        features[frame] = (kp, des, cKDTree(depth['points']), depth['points'], depth['depths'])
    rows = []
    matcher = cv2.BFMatcher()
    for i in EARLY:
        for j in LATE:
            ka, da, ta, ua, za = features[i]
            kb, db, tb, ub, zb = features[j]
            matches = [a for a,b in matcher.knnMatch(da, db, k=2) if a.distance < .72*b.distance]
            xy_a = np.array([ka[m.queryIdx].pt for m in matches])
            xy_b = np.array([kb[m.trainIdx].pt for m in matches])
            if len(matches) < 8:
                continue
            F, fmask = cv2.findFundamentalMat(xy_a, xy_b, cv2.FM_RANSAC, 1.5, .999)
            if F is None or fmask is None:
                continue
            d1, ia = ta.query(xy_a); d2, ib = tb.query(xy_b)
            valid = (d1 < 2) & (d2 < 2) & fmask.ravel().astype(bool)
            def xyz(uv, z):
                return np.column_stack(((uv[:,0]-K[0,2])*z/K[0,0], (uv[:,1]-K[1,2])*z/K[1,1], z))
            a, b = xyz(ua[ia], za[ia])[valid], xyz(ub[ib], zb[ib])[valid]
            if len(a) < 15:
                rows.append(dict(i=i,j=j,matches=len(matches),depth_pairs=len(a),accepted=False,reason='insufficient depth'))
                continue
            rng = np.random.default_rng(i*10000+j)
            best = np.zeros(len(a), bool)
            for _ in range(1500):
                idx = rng.choice(len(a), 3, replace=False)
                if np.linalg.norm(np.cross(a[idx[1]]-a[idx[0]], a[idx[2]]-a[idx[0]])) < .0002:
                    continue
                T = fit(a[idx], b[idx])
                err = np.linalg.norm(a@T[:3,:3].T+T[:3,3]-b, axis=1)
                keep = err < .025+.012*np.minimum(a[:,2], b[:,2])
                if keep.sum() > best.sum(): best = keep
            if best.sum() < 12:
                rows.append(dict(i=i,j=j,matches=len(matches),depth_pairs=len(a),accepted=False,reason='insufficient rigid inliers'))
                continue
            T=fit(a[best], b[best]); err=np.linalg.norm(a@T[:3,:3].T+T[:3,3]-b,axis=1)
            np.savez_compressed(OUT/f'pair_{i}_{j}_depth.npz',a=a,b=b,rigid_inliers=best)
            canonical=np.linalg.inv(poses[6470])@poses[j]@T@np.linalg.inv(poses[i])@poses[3206]
            singular=np.linalg.svd(a[best]-a[best].mean(axis=0),compute_uv=False)
            row=dict(i=i,j=j,matches=len(matches),fundamental_inliers=int(fmask.sum()),depth_pairs=len(a),
                inliers=int(best.sum()),fraction=float(best.mean()),median_m=float(np.median(err[best])),
                p90_m=float(np.percentile(err[best],90)),spread_singular_values=singular.tolist(),
                transform=T.tolist(),canonical_transform=canonical.tolist(),
                validation='held-out' if (i==3236 or j==6530) else 'candidate',
                accepted=bool(best.sum()>=20 and best.mean()>=.45 and np.median(err[best])<.06))
            rows.append(row)
    good=[r for r in rows if r.get('accepted') and r['validation']=='candidate']
    if good:
        ref=np.array(good[0]['canonical_transform'])
        for row in rows:
            if 'canonical_transform' not in row: continue
            d=np.linalg.inv(ref)@np.array(row['canonical_transform'])
            row['canonical_disagreement_m']=float(np.linalg.norm(d[:3,3]))
            row['canonical_disagreement_deg']=float(np.degrees(Rotation.from_matrix(d[:3,:3]).magnitude()))
    report=dict(scope='manual people exclusion is approximate; not full semantic masking',pairs=rows)
    (OUT/'static_revisit_pairs.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for r in rows:
        print(r['i'],r['j'],'inliers',r.get('inliers',0),'depth',r['depth_pairs'], 'accepted',r['accepted'],
              'heldout',r.get('validation'), 'canonical_m',round(r.get('canonical_disagreement_m',-1),3),
              'canonical_deg',round(r.get('canonical_disagreement_deg',-1),2),flush=True)


if __name__=='__main__': main()
