"""Align D455 depth to undistorted COLMAP images and validate scene scale.

Read-only recording access; no ROS installation or connected camera needed.
Samples are camera Z in original COLMAP units, not Euclidean range.
"""
import argparse
import importlib.util
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from scipy.spatial import cKDTree

spec = importlib.util.spec_from_file_location("capture", Path(__file__).with_name("prepare-realsense-scene.py"))
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def fields(text):
    return dict(f.split("=", 1) for f in text.split(";"))


def decode_depth(blob):
    c = capture.CDR(blob)
    sec, ns, frame = c.uint32(), c.uint32(), c.string()
    height, width, encoding = c.uint32(), c.uint32(), c.string()
    endian = blob[c.offset]
    c.offset += 1
    step, size = c.uint32(), c.uint32()
    if encoding not in ("16UC1", "mono16", "Z16") or size != height * step:
        raise ValueError((encoding, size, step, height))
    a = np.frombuffer(blob, dtype=">u2" if endian else "<u2", count=size//2, offset=c.offset)
    return a.reshape(height, step//2)[:, :width].astype(np.float32), sec + ns/1e9


def align(blob, d, c, rotation, translation, rawcam, targetcam, units):
    depth, header = decode_depth(blob)
    z = depth * units
    # Exclude invalid pixels, distant returns and discontinuities, with no filling.
    low = cv2.erode(z, np.ones((3, 3), np.uint8))
    high = cv2.dilate(z, np.ones((3, 3), np.uint8))
    valid = (low > .3) & (high < 6) & ((high-low) < np.maximum(.03, z*.02))
    yy, xx = np.nonzero(valid)
    z = z[valid]
    xyz = np.stack(((xx-float(d['ppx']))/float(d['fx'])*z,
                    (yy-float(d['ppy']))/float(d['fy'])*z, z), axis=1)
    xyz = xyz @ rotation.T + translation
    normalized = xyz[:, :2] / xyz[:, 2:3]
    # Stored color distortion maps raw pixels to ideal rays (inverse Brown).
    # Invert that mapping iteratively to recover raw RGB pixel coordinates.
    coeff = np.array([float(v) for v in c['coeffs'].split(',')])
    raw = normalized.copy()
    k1,k2,p1,p2,k3 = coeff
    for _ in range(20):
        x,y = raw.T
        r2 = x*x+y*y
        radial = 1+k1*r2+k2*r2*r2+k3*r2*r2*r2
        tangential = np.stack((2*p1*x*y+p2*(r2+2*x*x),
                               p1*(r2+2*y*y)+2*p2*x*y),axis=1)
        raw = (normalized-tangential)/radial[:,None]
    raw = raw * [float(c['fx']), float(c['fy'])] + [float(c['ppx']), float(c['ppy'])]
    rays = rawcam.cam_from_img(np.ascontiguousarray(raw))
    K = np.asarray(targetcam.calibration_matrix())
    pixels = rays * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    valid = (np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= 1)
             & (pixels[:, 0] < targetcam.width-1) & (pixels[:, 1] >= 1)
             & (pixels[:, 1] < targetcam.height-1) & (xyz[:, 2] > 0))
    pixels, z = pixels[valid], xyz[valid, 2]
    # Z-buffer competing depth samples in each target pixel; nearest wins.
    key = np.rint(pixels[:, 1]).astype(int)*targetcam.width + np.rint(pixels[:, 0]).astype(int)
    order = np.lexsort((z, key))
    _, first = np.unique(key[order], return_index=True)
    keep = order[first]
    return pixels[keep], z[keep], header


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('data/realsense_20261029_140739'))
    ap.add_argument('--recording', type=Path, default=Path('20261029_140739.db3'))
    ap.add_argument('--samples', type=int, default=4096)
    ap.add_argument('--model-dir', default='reconstruction')
    ap.add_argument('--dataset-dir', default='dataset')
    ap.add_argument('--depth-output', default='sensor_depth')
    args = ap.parse_args()
    root = args.root
    report = json.loads((root/'reconstruction_report.json').read_text())
    rawmodel = pycolmap.Reconstruction(root/args.model_dir/str(report['best_model']))
    target = pycolmap.Reconstruction(root/args.dataset_dir/'sparse')
    calibration = json.loads((root/'calibration.json').read_text())
    d = fields(next(v for k,v in calibration.items() if k.endswith('/Depth_0/camera_info')))
    c = fields(next(v for k,v in calibration.items() if k.endswith('/Color_0/camera_info')))
    if any(float(v) != 0 for v in d['coeffs'].split(',')):
        raise ValueError('Nonzero depth distortion requires additional deprojection')
    units = float(next(v for k,v in calibration.items() if k.endswith('/Depth_Units/value')))
    frames = {f['name']: f for f in json.loads((root/'capture_manifest.json').read_text())['frames']}
    target_images = {im.name: im for im in target.images.values()}
    output = root/args.depth_output
    output.mkdir(exist_ok=True)
    cache = root/(args.depth_output+'_aligned_cache')
    cache.mkdir(exist_ok=True)
    records, ratios_all = [], []
    with sqlite3.connect(args.recording.resolve().as_uri()+'?mode=ro', uri=True) as db:
        topics = {n:i for i,n in db.execute('select id,name from topics')}
        def string(suffix):
            tid = next(i for n,i in topics.items() if n.endswith(suffix))
            return capture.CDR(db.execute('select data from messages where topic_id=? limit 1', (tid,)).fetchone()[0]).string()
        ext = fields(string('/Color_0/tf/ref_0'))
        # librealsense extrinsics are reference -> stream, column-major rotation.
        rotation = np.array([float(v) for v in ext['rotation'].split(',')]).reshape(3,3,order='F')
        translation = np.array([float(v) for v in ext['translation'].split(',')])
        deptext = fields(string('/Depth_0/tf/ref_0'))
        if not np.allclose(np.array([float(v) for v in deptext['rotation'].split(',')]).reshape(3,3), np.eye(3)) or any(float(v) for v in deptext['translation'].split(',')):
            raise ValueError('Depth must be reference stream')
        tid = next(i for n,i in topics.items() if n.endswith('/Depth_0/image/data'))
        times = db.execute('select id,timestamp from messages where topic_id=? order by timestamp', (tid,)).fetchall()
        stamps = np.array([t for i,t in times], dtype=np.int64)
        for index, im in enumerate(sorted(rawmodel.images.values(), key=lambda im: im.name)):
            f = frames[im.name]
            j = np.searchsorted(stamps, f['timestamp_ns'])
            j = min([max(0,j-1), min(j,len(times)-1)], key=lambda k: abs(int(stamps[k])-f['timestamp_ns']))
            delta = abs(int(stamps[j])-f['timestamp_ns'])/1e9
            if delta > .020:
                raise ValueError(f'RGB/depth pair too far apart: {im.name}: {delta}')
            blob = db.execute('select data from messages where id=?', (times[j][0],)).fetchone()[0]
            tc = target.cameras[target_images[im.name].camera_id]
            pixels, z, header = align(blob, d, c, rotation, translation, rawmodel.cameras[im.camera_id], tc, units)
            np.savez_compressed(cache/(im.name+'.npz'), points=pixels.astype('f4'), meters=z.astype('f4'))
            # Sparse points seen by this image provide a scale cross-check.
            ids = [p.point3D_id for p in im.points2D if p.has_point3D()]
            world = np.array([rawmodel.points3D[i].xyz for i in ids])
            w2c = np.asarray(im.cam_from_world().matrix())
            cam = world @ w2c[:3,:3].T + w2c[:3,3]
            projected = cam[:,:2]/cam[:,2:3] * [tc.focal_length_x,tc.focal_length_y] + [tc.principal_point_x,tc.principal_point_y]
            dist, nearest = cKDTree(pixels).query(projected)
            good = (dist < 1.5) & (cam[:,2] > 0)
            ratios = cam[good,2] / z[nearest[good]]
            ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
            # Fit scale from training images only (same every-eighth split).
            if index % 8 != 0:
                ratios_all.extend(ratios.tolist())
            records.append(dict(name=im.name, timestamp_delta_seconds=delta,
                                header_delta_seconds=abs(header-f['header_seconds']),
                                valid_samples=len(z), scale_pairs=len(ratios),
                                scale_median=float(np.median(ratios)) if len(ratios) else None))
            if index % 50 == 0:
                print(f'Aligned {index+1}/{len(rawmodel.images)}', flush=True)
    ratios = np.array(ratios_all)
    scale = float(np.median(ratios))
    rel = np.abs(ratios/scale-1)
    frame_scales = np.array([r['scale_median'] for r in records if r['scale_pairs'] >= 20])
    result = dict(colmap_units_per_meter=scale, training_scale_pairs=len(ratios),
                  scale_inlier_fraction_10pct=float(np.mean(rel < .10)),
                  scale_relative_median_error=float(np.median(rel)),
                  frame_scale_relative_p90=float(np.percentile(np.abs(frame_scales/scale-1),90)),
                  extrinsic_rotation=rotation.tolist(), extrinsic_translation=translation.tolist(),
                  filtering='0.3-6m; valid 3x3 neighborhood; depth discontinuity < max(3cm,2%); z-buffer',
                  scale_fit='training frames only; median sparse Z / sensor Z', frames=records)
    result['passed'] = bool(len(ratios) >= 500 and result['scale_inlier_fraction_10pct'] > .65
                            and result['frame_scale_relative_p90'] < .20)
    capture.save_json(root/(args.depth_output+'_report.json'), result)
    if not result['passed']:
        raise RuntimeError('Sensor scale gate failed; inspect sensor_depth_report.json before training')
    rng = np.random.default_rng(42)
    for r in records:
        with np.load(cache/(r['name']+'.npz')) as a:
            keep = rng.choice(len(a['meters']), min(args.samples,len(a['meters'])), replace=False)
            np.savez_compressed(output/(r['name']+'.npz'), points=a['points'][keep], depths=a['meters'][keep]*scale)
    print('SENSOR_DEPTH_READY', {k:v for k,v in result.items() if k != 'frames'}, flush=True)


if __name__ == '__main__':
    main()
