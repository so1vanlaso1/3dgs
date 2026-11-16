"""Recover cleaned-frame metric poses when RGB-only SfM splits into fragments.

Uses already rectified RGB, aligned metric depth, COLMAP matches and scale-gated
SfM relative-pose anchors. Never reopens or hard-codes a particular recording.
"""
import argparse
import json
from pathlib import Path
import sqlite3

import cv2
import numpy as np
from scipy.spatial import cKDTree

import importlib.util
spec = importlib.util.spec_from_file_location('clean_pipeline', Path(__file__).with_name('clean-realsense-scene.py'))
clean = importlib.util.module_from_spec(spec); spec.loader.exec_module(clean)
recovery = clean.module('recover-rgbd-scene.py')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('data/realsense_20261030_111354_clean'))
    ap.add_argument('--anchor-root', type=Path, help='Verified lower-rate reconstruction of the exact same recording; reuse metric relative poses')
    a = ap.parse_args(); root = a.root.resolve()
    manifest = clean.check_source(root)
    cal = json.loads((root/'calibration_clean.json').read_text())
    K = np.array(cal['K'])
    out = root/'rgbd'
    for d in ('features', 'depth_all', 'images'):
        (out/d).mkdir(parents=True, exist_ok=True)
    with sqlite3.connect((root/'colmap.db').as_uri()+'?mode=ro', uri=True) as db:
        keys = {i: np.frombuffer(b, 'f4').reshape(n, c)[:, :2].copy()
                for i, n, c, b in db.execute('select image_id,rows,cols,data from keypoints')}
        ids = {n: i for i, n in db.execute('select image_id,name from images')}
        matches = {pid: (pid, n, b) for pid, n, b in db.execute('select pair_id,rows,data from matches where rows>=30')}
        matches.update({pid: (pid, n, b) for pid, n, b in db.execute('select pair_id,rows,data from two_view_geometries where rows>=30')})
    records, xyz = [], []
    for j, f in enumerate(manifest['frames']):
        name = f['name']; iid = ids[name]
        records.append(dict(name=name, image_id=iid))
        with np.load(root/'aligned_depth_meters'/(name+'.npz')) as dep:
            pixels, z = dep['points'], dep['depths']
        dist, nn = cKDTree(pixels).query(keys[iid])
        p = pixels[nn]; z = z[nn]
        # Deproject at the matched depth sample, not at a fictitious keypoint ray.
        points = np.column_stack(((p[:, 0]-K[0, 2])/K[0, 0]*z,
                                  (p[:, 1]-K[1, 2])/K[1, 1]*z, z)).astype('f4')
        points[dist > 2] = np.nan
        xyz.append(points)
        np.savez_compressed(out/'features'/(name+'.npz'), xyz=points)
        for src, dst in ((root/'images'/name, out/'images'/name),
                         (root/'aligned_depth_meters'/(name+'.npz'), out/'depth_all'/(name+'.npz'))):
            if not dst.exists():
                import os
                os.link(src, dst)
    clean.write(out/'calibration.json', dict(K=cal['K'], width=cal['width'], height=cal['height'], frames=records))
    index = {f['image_id']: j for j, f in enumerate(records)}
    edges = []
    rng = np.random.default_rng(42)
    ordered = sorted(matches.values(), key=lambda r: abs(index[r[0]//2147483647]-index[r[0]%2147483647]))
    for count, (pid, n, blob) in enumerate(ordered):
        ia, ib = pid//2147483647, pid%2147483647
        i, j = index[ia], index[ib]
        pairs = np.frombuffer(blob, 'u4').reshape(n, 2)
        x, y = xyz[i][pairs[:, 0]], xyz[j][pairs[:, 1]]
        valid = np.isfinite(x).all(axis=1)&np.isfinite(y).all(axis=1)
        result = recovery.robust_fit(x[valid], y[valid], rng)
        method = 'metric_3d_ransac'
        if result is None:
            good = np.isfinite(x).all(axis=1)
            if good.sum() >= 25:
                pix = keys[ib][pairs[:, 1]][good]
                ok, rv, tv, inliers = cv2.solvePnPRansac(x[good], pix, K, None,
                        iterationsCount=300, reprojectionError=3, confidence=.999, flags=cv2.SOLVEPNP_EPNP)
                if ok and inliers is not None and len(inliers) >= 25 and len(inliers)/good.sum() > .45:
                    ii = inliers[:, 0]
                    ok, rv, tv = cv2.solvePnP(x[good][ii], pix[ii], K, None, rv, tv, True)
                    T = np.eye(4); T[:3, :3] = cv2.Rodrigues(rv)[0]; T[:3, 3] = tv[:, 0]
                    error = np.linalg.norm(x[valid]@T[:3, :3].T+T[:3, 3]-y[valid], axis=1)
                    if ok and len(error) >= 10 and np.median(error) < .1:
                        result = T, len(ii), float(np.median(error)); method = 'depth_pnp'
        if result is not None:
            T, nin, err = result
            edges.append(dict(i=i, j=j, T=T.tolist(), inliers=nin, error=err, method=method))
        if count % 500 == 0:
            print('CLEAN_RGBD', count, '/', len(ordered), 'edges', len(edges), flush=True)
    # SfM fragments provide reliable relative constraints only after metric-scale
    # consistency checks. Fit each scale from training frames, not heldout depth.
    import pycolmap as p
    name_index = {f['name']: j for j, f in enumerate(records)}
    train = {f['name'] for j, f in enumerate(records) if j % 8 != 0}
    anchors = []
    directories = sorted((root/'reconstruction').iterdir()) if (root/'reconstruction').exists() else []
    for directory in directories:
        if not (directory/'images.bin').exists():
            continue
        model = p.Reconstruction(directory)
        images = sorted(model.images.values(), key=lambda im: im.name)
        ratios = []
        for im in images:
            if im.name not in train:
                continue
            samples = xyz[name_index[im.name]]
            pairs = [(j, pt.point3D_id) for j, pt in enumerate(im.points2D) if pt.has_point3D() and np.isfinite(samples[j]).all()]
            if not pairs:
                continue
            world = np.array([model.points3D[pid].xyz for j, pid in pairs])
            W = np.asarray(im.cam_from_world().matrix())
            z = (world@W[:, :3].T+W[:, 3])[:, 2]
            depth = np.array([samples[j, 2] for j, pid in pairs])
            good = (z > 0)&(depth > .3)
            ratios.extend((z[good]/depth[good]).tolist())
        scale = float(np.median(ratios)) if ratios else 0.
        error = float(np.median(np.abs(np.array(ratios)/scale-1))) if scale > 0 else 1.
        passed = len(ratios) > 100 and error < .08
        anchors.append(dict(model=directory.name, registered_frames=len(images), scale=scale, training_pairs=len(ratios), median_relative_error=error, passed=passed))
        if not passed:
            continue
        poses = []
        for im in images:
            W = np.eye(4); W[:3] = im.cam_from_world().matrix()
            pose = np.linalg.inv(W); pose[:3, 3] /= scale; poses.append(pose)
        for j in range(len(images)):
            for delta in (1, 4):
                if j+delta >= len(images):
                    continue
                T = np.linalg.inv(poses[j+delta])@poses[j]
                edges.append(dict(i=name_index[images[j].name], j=name_index[images[j+delta].name],
                                  T=T.tolist(), inliers=1000, error=max(.005, error), method='scale_gated_sfm_'+directory.name))
    if a.anchor_root:
        base = a.anchor_root.resolve()
        baseline = clean.check_source(base)
        if (baseline['source'], baseline['source_bytes'], baseline['source_mtime_ns']) != (manifest['source'], manifest['source_bytes'], manifest['source_mtime_ns']):
            raise ValueError('Anchor recording differs from current recording')
        basecal = json.loads((base/'calibration_clean.json').read_text())
        if not np.allclose(basecal['K'], cal['K']) or (basecal['width'], basecal['height']) != (cal['width'], cal['height']):
            raise ValueError('Anchor and full-rate image calibrations differ')
        current = {f['message_id']: j for j, f in enumerate(manifest['frames'])}
        original = {f['name']: f for f in baseline['frames']}
        model = p.Reconstruction(base/'rgbd/dataset/sparse')
        images = sorted(model.images.values(), key=lambda im: im.name)
        mapped, poses = [], []
        for im in images:
            f = original[im.name]
            if f['message_id'] not in current:
                continue
            W = np.eye(4); W[:3] = im.cam_from_world().matrix()
            mapped.append(current[f['message_id']]); poses.append(np.linalg.inv(W))
        for j in range(len(mapped)):
            for delta in (1, 4):
                if j+delta < len(mapped):
                    T = np.linalg.inv(poses[j+delta])@poses[j]
                    edges.append(dict(i=mapped[j], j=mapped[j+delta], T=T.tolist(), inliers=1000,
                                      error=.005, method='verified_same_recording_anchor'))
        anchors.append(dict(method='verified lower-rate same-recording metric relative poses', source=str(base), retained_anchors=len(mapped),
                            independent_reconstruction=False))
    clean.write(out/'anchor_report.json', anchors)
    clean.write(out/'edges.json', edges)
    print('CLEAN_RGBD_OPTIMIZE', len(edges), 'constraints', flush=True)
    if not edges:
        raise RuntimeError('No valid metric pose constraints')
    recovery.optimize(root)


if __name__ == '__main__':
    main()
