"""Separate RGB correspondence drift from measured-depth geometry disagreement.

Diagnostic only: no old checks, models or acceptance decisions are modified.
"""
import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location('direct', Path(__file__).with_name('db3_direct_depth_refine.py'))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
t, r, m = d.t, d.r, d.m


def quantiles(values):
    values = np.asarray(values)
    return np.percentile(values, [50, 90, 100]).tolist() if len(values) else None


def anchor_flow(first, last, start, initial):
    """Match the original patch directly, using the chain only as initialization."""
    settings = dict(winSize=(21, 21), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .01),
                    flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    q, fs, _ = cv2.calcOpticalFlowPyrLK(first, last, start[:, None].astype('f4'),
                                      initial[:, None].astype('f4').copy(), **settings)
    back, bs, _ = cv2.calcOpticalFlowPyrLK(last, first, q,
                                         start[:, None].astype('f4').copy(), **settings)
    return q.reshape(-1, 2), (fs.ravel() > 0) & (bs.ravel() > 0), np.linalg.norm(back.reshape(-1, 2)-start, axis=1)


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(6)
    cv2.setRNGSeed(11012)
    K = np.array(m.read(m.INPUT/'calibration_clean.json')['K'])
    protocol = m.read(d.PARENT/'protocol.json')
    paths = [Path(__file__), m.INPUT/'calibration_clean.json']
    for base in (d.PARENT, d.INPUTS):
        paths.extend(f for f in base.rglob('*') if f.is_file())
    m.write(out/'protocol.json', dict(date='2026-11-12',
        question='Does incremental patch drift explain the target reprojection tail?',
        cases=protocol['cases'],
        method='Fit RGB-only homography on existing disjoint fitting tracks (RANSAC 1px); diagnose all original check identities. Direct first-to-last LK uses same 21px window and 3 levels, initialized by old track, no camera/depth or validation residual filtering. Report every identity, failures and drift. No promotion.',
        input_hashes={str(f.relative_to(r.ROOT)): m.sha(f) for f in paths},
        reference='https://docs.opencv.org/4.7.0/dc/d6b/group__video__track.html'))
    results = []
    for case in protocol['cases']:
        label = case['label']
        parent = d.INPUTS/label
        rows = m.read(parent/'inputs.json')
        images = [cv2.imread(str(parent/'inputs'/f'{x["source_id"]}.png'), 0) for x in rows]
        original = []
        for row in rows:
            with np.load(parent/'inputs'/f'{row["source_id"]}.npz') as z:
                original.append({k: z[k].copy() for k in z.files})
        with np.load(parent/'moving_checks.npz') as z:
            uv = z['uv'].copy()
        with np.load(d.PARENT/label/'fit_features.npz') as z:
            fit = [z[f'uv_{c}'].copy() for c in range(len(rows))]
        tracks = m.read(d.PARENT/label/'tracks.json')['tracks']
        poses = np.array(m.read(r.ROOT/'results/db3_direct_depth_20261111'/label/'poses.json'))
        checkfs = [dict(uv=q, depth=f['depth']) for q, f in zip(uv, original)]
        r.observations(checkfs, K)
        pairs = []
        for a, b in case['check_pairs']:
            refs = [dict(tr) for tr in tracks if a in dict(tr) and b in dict(tr)]
            x = np.array([fit[a][tr[a]] for tr in refs])
            y = np.array([fit[b][tr[b]] for tr in refs])
            H, mask = cv2.findHomography(x, y, cv2.RANSAC, 1.)
            pred = cv2.perspectiveTransform(uv[a][:, None], H).reshape(-1, 2)
            q, ok, fb = anchor_flow(images[a], images[b], uv[a], uv[b])
            T = poses[b]@np.linalg.inv(poses[a])
            xyz = r.transform_points(T, checkfs[a]['xyz'])
            projected = xyz[:, :2]/xyz[:, 2:]*[K[0, 0], K[1, 1]]+[K[0, 2], K[1, 2]]
            row = dict(cameras=[a, b], fitting_matches=len(x), homography_inliers=int(mask.sum()),
                       check_count=len(q), anchor_status_ok=int(ok.sum()), anchor_fb_lt075=int((ok & (fb < .75)).sum()),
                       anchor_fb_px=quantiles(fb), chain_to_anchor_px=quantiles(np.linalg.norm(q-uv[b], axis=1)),
                       homography_chain_px=quantiles(np.linalg.norm(pred-uv[b], axis=1)),
                       homography_anchor_px=quantiles(np.linalg.norm(pred-q, axis=1)),
                       sensor_depth_m=quantiles(checkfs[a]['z']),
                       observations=[dict(id=i, old=uv[b, i].tolist(), anchored=q[i].tolist(),
                                          homography=pred[i].tolist(), depth_projection=projected[i].tolist(),
                                          anchor_ok=bool(ok[i]), fb_px=float(fb[i])) for i in range(len(q))])
            pairs.append(row)
        result = dict(label=label, pairs=pairs, training_authorized=False)
        m.write(out/f'{label}.json', result)
        results.append(result)
        print(label, [{k: v for k, v in p.items() if k != 'observations'} for p in pairs], flush=True)
    m.write(out/'result.json', dict(cases=results, diagnostic_only=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    main(parser.parse_args().output_dir)
