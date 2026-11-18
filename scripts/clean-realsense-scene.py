"""Offline RealSense DB3 -> cleaned calibrated RGB-D -> COLMAP -> gsplat inputs.

Raw recording is read-only. All rejection decisions and numerical checks are saved.
Factory inverse-Brown RGB distortion is removed BEFORE feature/pose estimation.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import struct
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '8')
import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


capture = module('prepare-realsense-scene.py')
adapter = module('prepare-sensor-depth.py')
projection = module('realsense_projection.py')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def fields(s):
    return dict(f.split('=', 1) for f in s.split(';'))


# Compatibility names for historical diagnostics; implementation has one owner.
brown = projection.brown
invert_brown = projection.invert_brown
intrinsic = projection.intrinsic
relative_extrinsics = projection.relative_extrinsics


def clean_depth(depth, units, d, R, t, K, width, height, near=.3, far=6.):
    p, z, fraction, _ = projection.align_depth(depth, units, d, R, t, K, width, height, near, far)
    return p, z, fraction


def rgb_quality(rgb):
    gray = cv2.cvtColor(cv2.resize(rgb, (640, 360)), cv2.COLOR_RGB2GRAY)
    return dict(sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                mean_luma=float(gray.mean()), dark_fraction=float((gray <= 5).mean()),
                bright_fraction=float((gray >= 250).mean()))


def rejection(q, min_sharpness):
    reasons = []
    if q['sharpness'] < min_sharpness:
        reasons.append('blur_or_low_texture')
    if q['mean_luma'] < 18 or q['dark_fraction'] > .65:
        reasons.append('underexposed')
    if q['mean_luma'] > 240 or q['bright_fraction'] > .45:
        reasons.append('overexposed')
    return reasons


def frame_groups(rows, fps, all_frames=False):
    """All-frame mode must never drop frames due to timestamp-bin collisions."""
    if all_frames:
        return {j: [row] for j, row in enumerate(rows)}
    bins = {}
    for mid, stamp in rows:
        bins.setdefault(int((stamp-rows[0][1])*fps/1e9), []).append((mid, stamp))
    return bins


def header(blob):
    c = capture.CDR(blob)
    sec, ns, frame = c.uint32(), c.uint32(), c.string()
    return c, sec+ns/1e9, frame


def imu_export(db, topics, output):
    report = {}
    for tid, name, typ in topics:
        if not name.endswith('/imu/data'):
            continue
        kind = 'gyro' if '/Gyro_' in name else 'accel'
        records = []
        with (output/f'{kind}.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['message_id', 'timestamp_ns', 'header_seconds', 'frame_id', 'x', 'y', 'z'])
            for mid, stamp, blob in db.execute('select id,timestamp,data from messages where topic_id=? order by timestamp', (tid,)):
                c, seconds, frame = header(blob)
                c.offset = 4 + ((c.offset-4+7)//8)*8
                # sensor_msgs/Imu: quaternion, orientation covariance, angular
                # velocity, angular covariance, linear acceleration, covariance.
                vals = struct.unpack_from('<37d', blob, c.offset)
                xyz = vals[13:16] if kind == 'gyro' else vals[25:28]
                if not np.isfinite(xyz).all():
                    raise ValueError('Nonfinite IMU measurement')
                writer.writerow([mid, stamp, seconds, frame, *xyz])
                records.append((stamp, seconds, *xyz))
        a = np.array(records)
        if not len(a):
            raise ValueError(f'Empty IMU topic {name}')
        report[kind] = dict(count=len(a), units='rad/s' if kind == 'gyro' else 'm/s^2',
                            strictly_increasing=bool(np.all(np.diff(a[:, 0]) > 0)),
                            duration_seconds=float((a[-1, 0]-a[0, 0])/1e9),
                            median_sample_interval_seconds=float(np.median(np.diff(a[:, 0]))/1e9),
                            magnitude_percentiles=np.percentile(np.linalg.norm(a[:, 2:], axis=1), [5, 50, 95]).tolist())
    report['pose_fusion'] = 'Not used; exported and checked, no inertial integration or IMU calibration fit'
    return report


def clean(args):
    root = args.output.resolve()
    if (root/'capture_manifest.json').exists():
        old = json.loads((root/'capture_manifest.json').read_text())
        previous_parameters = dict(old['parameters'])
        previous_parameters.setdefault('all_frames', False)
        previous_parameters.setdefault('projection_version', projection.LEGACY)
        if old['source'] != str(args.recording.resolve()) or previous_parameters != vars_for_manifest(args):
            raise ValueError('Existing extraction has different source/settings; choose another output')
        check_source(root)
        print('CLEAN already complete', flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    for name in ('images', 'aligned_depth_meters', 'imu'):
        (root/name).mkdir(exist_ok=True)
    source = args.recording.resolve()
    before = source.stat()
    with sqlite3.connect(source.as_uri()+'?mode=ro', uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        # Full SQLite page integrity, not merely file presence.
        integrity = db.execute('PRAGMA quick_check').fetchall()
        if integrity != [('ok',)]:
            raise ValueError(integrity)
        topics = db.execute('select id,name,type from topics').fetchall()
        def topic(suffix):
            return next(i for i, n, _ in topics if n.endswith(suffix))
        def string(suffix):
            return capture.CDR(db.execute('select data from messages where topic_id=? limit 1', (topic(suffix),)).fetchone()[0]).string()
        calibration = {n: capture.CDR(db.execute('select data from messages where topic_id=? limit 1', (i,)).fetchone()[0]).string()
                       for i, n, typ in topics if typ == 'std_msgs/msg/String' and n.endswith(('/camera_info', '/tf/ref_0', '/imu_intrinsic', '/Depth_Units/value'))}
        write(root/'calibration_raw.json', calibration)
        c, d = fields(string('/Color_0/camera_info')), fields(string('/Depth_0/camera_info'))
        if any(float(v) != 0 for v in d['coeffs'].split(',')):
            raise ValueError('Nonzero depth distortion requires explicit deprojection support')
        if c['model'] not in ('Inverse Brown Conrady', 'None', 'Brown Conrady'):
            raise ValueError(f"Unsupported color model: {c['model']}")
        units = float(string('/Depth_Units/value'))
        if not 0 < units < .1:
            raise ValueError('Invalid depth unit scale')
        R, t = relative_extrinsics(string('/Depth_0/tf/ref_0'), string('/Color_0/tf/ref_0'))
        K = intrinsic(c)
        width = args.width
        height = round(int(c['height'])*width/int(c['width']))
        targetK = K.copy()
        targetK[0] *= width/int(c['width'])
        targetK[1] *= height/int(c['height'])
        maps, validmap = projection.remap(c, targetK, width, height, args.projection_version)
        # Crop to a fully valid rectangle: no black borders enter SfM/training.
        mx, my = 0, 0
        while not validmap[my:height-my or None, mx:width-mx or None].all():
            mx += 1
            my += 1
            if mx > width//4 or my > height//4:
                raise ValueError('Undistortion loses too much image area')
        maps = maps[my:height-my or None, mx:width-mx or None]
        targetK[0, 2] -= mx
        targetK[1, 2] -= my
        height, width = maps.shape[:2]
        write(root/'calibration_clean.json', dict(K=targetK.tolist(), width=width, height=height,
              model='PINHOLE', depth_units=units, depth_to_color_R=R.tolist(), depth_to_color_t_m=t.tolist(),
              crop_xy=[mx, my], raw_color=c, raw_depth=d, projection_version=args.projection_version))
        rgbrows = db.execute('select id,timestamp from messages where topic_id=? order by timestamp', (topic('/Color_0/image/data'),)).fetchall()
        drows = db.execute('select id,timestamp from messages where topic_id=? order by timestamp', (topic('/Depth_0/image/data'),)).fetchall()
        stamps = np.array([r[1] for r in drows], dtype=np.int64)
        if not np.all(np.diff(stamps) > 0):
            raise ValueError('Depth timestamps must be strictly increasing')
        bins = frame_groups(rgbrows, args.fps, args.all_frames)
        frames, rejected = [], []
        rng = np.random.default_rng(42)
        previous = None
        for b, choices in sorted(bins.items()):
            # Three candidates cover each interval, keeping IO bounded on large bags.
            candidates = []
            for j in np.unique(np.linspace(0, len(choices)-1, min(3, len(choices)), dtype=int)):
                mid, stamp = choices[j]
                blob = db.execute('select data from messages where id=?', (mid,)).fetchone()[0]
                rgb, h = capture.CDR(blob).image()
                q = rgb_quality(rgb)
                reasons = rejection(q, args.min_sharpness)
                rec = dict(message_id=mid, timestamp_ns=stamp, **h, **q)
                if reasons:
                    rejected.append(dict(**rec, reasons=reasons))
                else:
                    candidates.append((q['sharpness'], rec, rgb))
            if not candidates:
                continue
            _, rec, rgb = max(candidates, key=lambda a: a[0])
            stamp = rec['timestamp_ns']
            j = int(np.searchsorted(stamps, stamp))
            j = min([max(0, j-1), min(len(stamps)-1, j)], key=lambda k: abs(int(stamps[k])-stamp))
            dt = abs(int(stamps[j])-stamp)/1e9
            if dt > args.max_delta:
                rejected.append(dict(**rec, reasons=['timestamp_pair_gap'], pair_delta_seconds=dt))
                continue
            blob = db.execute('select data from messages where id=?', (drows[j][0],)).fetchone()[0]
            dep, dh = adapter.decode_depth(blob)
            hd = abs(dh-rec['header_seconds'])
            if hd > args.max_delta:
                rejected.append(dict(**rec, reasons=['header_pair_gap'], header_delta_seconds=hd))
                continue
            pixels, z, valid_fraction = clean_depth(dep, units, d, R, t, targetK, width, height, args.near, args.far)
            if len(z) < 1500:
                rejected.append(dict(**rec, reasons=['insufficient_valid_depth'], aligned_samples=len(z)))
                continue
            cleaned = cv2.remap(rgb, maps[:, :, 0], maps[:, :, 1], cv2.INTER_LINEAR)
            small = cv2.resize(cleaned, (160, 90))
            difference = float(np.abs(small.astype(float)-previous).mean()) if previous is not None else None
            if not args.all_frames and difference is not None and difference < .5:
                rejected.append(dict(**rec, reasons=['near_duplicate'], previous_mean_abs_difference=difference))
                continue
            name = f'frame_{len(frames):06d}.png'
            Image.fromarray(cleaned).save(root/'images'/name)
            keep = rng.choice(len(z), min(16000, len(z)), replace=False)
            np.savez_compressed(root/'aligned_depth_meters'/(name+'.npz'), points=pixels[keep], depths=z[keep])
            frames.append(dict(**rec, name=name, depth_message_id=drows[j][0], depth_timestamp_ns=int(stamps[j]),
                               depth_header_seconds=dh, pair_delta_seconds=dt, header_delta_seconds=hd,
                               depth_valid_fraction=valid_fraction, aligned_samples=len(z),
                               previous_mean_abs_difference=difference))
            previous = small.astype(float)
            if b % 25 == 0:
                print('CLEAN', b, '/', len(bins), 'accepted', len(frames), flush=True)
        imu = imu_export(db, topics, root/'imu')
        topic_report = [dict(name=n, type=typ, count=db.execute('select count(*) from messages where topic_id=?', (tid,)).fetchone()[0]) for tid, n, typ in topics]
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Source changed during extraction')
    if len(frames) < 20:
        raise ValueError(f'Only {len(frames)} clean frames; inspect rejected_frames.json')
    write(root/'rejected_frames.json', rejected)
    write(root/'imu/report.json', imu)
    manifest = dict(source=str(source), source_bytes=before.st_size, source_mtime_ns=before.st_mtime_ns,
                    source_open_mode='read-only', sqlite_quick_check='ok', parameters=vars_for_manifest(args),
                    raw_rgb_frames=len(rgbrows), raw_depth_frames=len(drows), topics=topic_report,
                    duration_seconds=(rgbrows[-1][1]-rgbrows[0][1])/1e9, frames=frames)
    write(root/'capture_manifest.json', manifest)
    sheet = Image.new('RGB', (1280, 800), '#222222')
    draw = ImageDraw.Draw(sheet)
    for j, i in enumerate(np.linspace(0, len(frames)-1, 16, dtype=int)):
        f = frames[i]
        im = Image.open(root/'images'/f['name']); im.thumbnail((320, 175))
        x, y = j%4*320, j//4*200
        sheet.paste(im, (x, y+25))
        draw.text((x+3, y+3), f"{f['timestamp_ns']/1e9:.2f}s | sharp {f['sharpness']:.0f}", fill='white')
    sheet.save(root/'clean_preview.jpg')
    write(root/'cleaning_report.json', dict(accepted_frames=len(frames), rejected_candidates=len(rejected),
          raw_rgb_frames=len(rgbrows), sampled_bins=len(bins),
          selection_mode='Every source frame checked exactly once; no FPS binning or near-duplicate removal' if args.all_frames else 'Sharpest candidate per sampling interval',
          pair_delta_seconds_percentiles=np.percentile([f['pair_delta_seconds'] for f in frames], [0, 50, 95, 100]).tolist(),
          header_delta_seconds_max=max(f['header_delta_seconds'] for f in frames),
          sharpness_percentiles=np.percentile([f['sharpness'] for f in frames], [0, 50, 100]).tolist(),
          depth_valid_fraction_percentiles=np.percentile([f['depth_valid_fraction'] for f in frames], [0, 50, 100]).tolist(),
          operations=['exposure rejection', 'sharpness selection and rejection', 'near-duplicate rejection',
              'inverse-Brown RGB undistortion and valid crop', 'timestamp and header pairing gates',
              'depth range and discontinuity masking', 'valid-neighborhood median depth denoise',
              'calibrated depth-to-color transform', 'pinhole projection and nearest-Z occlusion handling'],
          photometric_policy='Reject gross exposure failures; preserve observed RGB, no synthetic exposure equalization',
          imu=imu, limitations=['No dynamic-object segmentation', 'Factory calibration reused, not independently recalibrated',
              'Nearest timestamp pairing is not hardware synchronization', 'No IMU pose fusion',
              'Thresholds do not prove every pixel is artifact-free'] ))
    print('CLEAN_COMPLETE', len(frames), 'frames', flush=True)


def vars_for_manifest(a):
    return {k: getattr(a, k) for k in ('fps', 'width', 'min_sharpness', 'max_delta', 'near', 'far', 'all_frames', 'projection_version')}


def check_source(root):
    m = json.loads((root/'capture_manifest.json').read_text())
    stat = Path(m['source']).stat()
    if (stat.st_size, stat.st_mtime_ns) != (m['source_bytes'], m['source_mtime_ns']):
        raise ValueError('Source recording changed since extraction')
    cal_path = root/'calibration_clean.json'
    if cal_path.exists():
        cal = json.loads(cal_path.read_text())
        mode = projection.calibration_mode(cal)
        declared = m.get('parameters', {}).get('projection_version', projection.LEGACY)
        if mode != declared:
            raise ValueError('Capture/calibration projection versions disagree; regenerate inputs')
    return m


def reconstruct(args):
    import pycolmap as p
    root = args.output.resolve()
    m = check_source(root)
    if (root/'reconstruction_report.json').exists():
        print('RECONSTRUCT already complete', flush=True)
        return
    cal = json.loads((root/'calibration_clean.json').read_text())
    K = np.array(cal['K'])
    db = root/'colmap.db'
    reader = p.ImageReaderOptions()
    reader.camera_model = 'PINHOLE'
    reader.camera_params = ','.join(map(str, [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))
    features = p.FeatureExtractionOptions()
    features.num_threads = 8
    features.max_image_size = args.width
    features.sift.max_num_features = 6000
    features.sift.first_octave = 0
    p.extract_features(db, root/'images', camera_mode=p.CameraMode.SINGLE,
                       reader_options=reader, extraction_options=features, device=p.Device.cpu)
    matching = p.FeatureMatchingOptions()
    matching.num_threads = 8
    pairs = p.SequentialPairingOptions()
    pairs.overlap = 12
    pairs.quadratic_overlap = True
    p.match_sequential(db, matching_options=matching, pairing_options=pairs, device=p.Device.cpu)
    options = p.IncrementalPipelineOptions()
    options.num_threads = 8
    options.random_seed = 42
    options.max_num_models = 8
    options.min_model_size = 15
    # RGB was rectified with measured K; refining K would invalidate stored depth pixels.
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    sparse = root/'reconstruction'
    sparse.mkdir(exist_ok=True)
    models = p.incremental_mapping(db, root/'images', sparse, options=options)
    if not models:
        raise RuntimeError('No coherent COLMAP model; examine capture and feature logs')
    best_id = max(models, key=lambda i: models[i].num_reg_images())
    model = models[best_id]
    dataset = root/'dataset'
    (dataset/'sparse').mkdir(parents=True, exist_ok=True)
    (dataset/'images').mkdir(exist_ok=True)
    model.write(dataset/'sparse')
    names = {im.name for im in model.images.values()}
    for name in names:
        dst = dataset/'images'/name
        if not dst.exists():
            os.link(root/'images'/name, dst)
    report = dict(selected_frames=len(m['frames']), registered_frames=model.num_reg_images(),
                  coverage_fraction=model.num_reg_images()/len(m['frames']), points3D=model.num_points3D(),
                  mean_reprojection_error=model.compute_mean_reprojection_error(), best_model=int(best_id),
                  models=[dict(id=int(i), registered_frames=v.num_reg_images(), points=v.num_points3D()) for i, v in models.items()],
                  unregistered_frames=[f['name'] for f in m['frames'] if f['name'] not in names],
                  precolmap_cleaning=True, calibration_fixed=True, pose_method='COLMAP RGB SfM')
    write(root/'reconstruction_report.json', report)
    print('RECONSTRUCTION_COMPLETE', report, flush=True)


def depth_dataset(args):
    import pycolmap as p
    from scipy.spatial import cKDTree
    root = args.output.resolve()
    check_source(root)
    model = p.Reconstruction(root/'dataset/sparse')
    names = sorted(im.name for im in model.images.values())
    train = set(names)-set(names[::8])
    ratios, rows = [], []
    for im in model.images.values():
        with np.load(root/'aligned_depth_meters'/(im.name+'.npz')) as a:
            pixels, meters = a['points'], a['depths']
        ids = [pt.point3D_id for pt in im.points2D if pt.has_point3D()]
        if not ids:
            continue
        xyz = np.array([model.points3D[i].xyz for i in ids])
        w2c = np.asarray(im.cam_from_world().matrix())
        cam = xyz@w2c[:, :3].T+w2c[:, 3]
        K = np.asarray(model.cameras[im.camera_id].calibration_matrix())
        projected = cam[:, :2]/cam[:, 2:3]*[K[0, 0], K[1, 1]]+[K[0, 2], K[1, 2]]
        dist, nn = cKDTree(pixels).query(projected)
        good = (dist < 1.5) & (cam[:, 2] > 0)
        r = cam[good, 2]/meters[nn[good]]
        r = r[np.isfinite(r) & (r > 0)]
        if im.name in train:
            ratios.extend(r.tolist())
        rows.append(dict(name=im.name, scale_pairs=len(r), median_scale=float(np.median(r)) if len(r) else None))
    scale = float(np.median(ratios)) if ratios else 0
    err = np.abs(np.array(ratios)/scale-1) if scale > 0 else np.array([1.])
    per_frame = [abs(r['median_scale']/scale-1) for r in rows if r['scale_pairs'] >= 20] if scale > 0 else [1.]
    report = dict(colmap_units_per_meter=scale, training_scale_pairs=len(ratios),
                  scale_inlier_fraction_10pct=float(np.mean(err < .1)),
                  median_relative_error=float(np.median(err)), frame_scale_relative_p90=float(np.percentile(per_frame, 90)) if per_frame else 1.,
                  fit_scope='training frames only; median sparse camera Z / aligned sensor camera Z', frames=rows)
    report['passed'] = bool(len(ratios) >= 500 and report['scale_inlier_fraction_10pct'] > .65 and report['frame_scale_relative_p90'] < .2)
    write(root/'sensor_depth_report.json', report)
    if not report['passed']:
        raise ValueError('Depth/SfM scale consistency failed; do not train with misaligned sensor depth')
    output = root/'dataset/sensor_depth'
    output.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    for name in names:
        with np.load(root/'aligned_depth_meters'/(name+'.npz')) as a:
            keep = rng.choice(len(a['depths']), min(4096, len(a['depths'])), replace=False)
            np.savez_compressed(output/(name+'.npz'), points=a['points'][keep], depths=a['depths'][keep]*scale)
    print('DEPTH_DATASET_COMPLETE', {k: v for k, v in report.items() if k != 'frames'}, flush=True)


def validate(args):
    sys.path.insert(0, str(REPO/'examples'))
    from datasets.colmap import Parser, Dataset
    root = args.output.resolve()
    manifest = check_source(root)
    dataset = root/args.dataset_name
    p = Parser(str(dataset), factor=1, normalize=True, load_exposure=False)
    if manifest['parameters'].get('all_frames', False):
        expected={f['name'] for f in manifest['frames']}
        missing=expected-set(p.image_names)
        if missing:
            raise ValueError(f'{len(missing)} qualified full-rate frames have no registered pose; recover them before training')
    train = Dataset(p, split='train', load_depths=True, sensor_depth_dir=str(dataset/'sensor_depth'))
    val = Dataset(p, split='val')
    if len(train) < 15 or len(val) < 2:
        raise ValueError('Insufficient training/holdout views')
    for i in range(len(train)):
        sample = train[i]
        for key in ('image', 'depths', 'points'):
            if key in sample and not np.isfinite(np.asarray(sample[key])).all():
                raise ValueError(f'Nonfinite {key} in sample {i}')
        if len(sample['depths']) == 0 or not (np.asarray(sample['depths']) > 0).all():
            raise ValueError('Empty or invalid depth sample')
    assert not set(train.indices)&set(val.indices)
    report = dict(registered_views=len(p.image_names), train_views=len(train), heldout_views=len(val),
                  all_training_samples_finite=True, depth_positive=True, source_stat_unchanged=True,
                  source=manifest['source'], normalization_transform=p.transform.tolist(),
                  evaluation_scope='Every eighth registered view; poses estimated jointly, interpolation test, not independent geometry ground truth',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(dataset/'dataset_validation.json', report)
    if args.dataset_name == 'dataset':
        write(root/'dataset_validation.json', report)
    print('DATASET_VALIDATED', report, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['clean', 'reconstruct', 'depth', 'validate', 'all'])
    ap.add_argument('--recording', type=Path, default=Path('20261030_111354.db3'))
    ap.add_argument('--output', type=Path, default=Path('data/realsense_20261030_111354_clean'))
    ap.add_argument('--dataset-name', choices=['dataset', 'rgbd/dataset'], default='dataset')
    ap.add_argument('--projection-version', choices=projection.MODES, default=projection.SDK)
    ap.add_argument('--fps', type=float, default=3.)
    ap.add_argument('--all-frames', action='store_true', help='Check every source RGB frame; keep every qualifying frame without binning/deduplication')
    ap.add_argument('--width', type=int, default=848)
    ap.add_argument('--min-sharpness', type=float, default=25.)
    ap.add_argument('--max-delta', type=float, default=.020)
    ap.add_argument('--near', type=float, default=.3)
    ap.add_argument('--far', type=float, default=6.)
    a = ap.parse_args()
    if a.fps <= 0 or a.width < 64 or a.near <= 0 or a.far <= a.near or a.max_delta <= 0:
        ap.error('Invalid sampling, resolution, depth range or timestamp tolerance')
    stages = dict(clean=clean, reconstruct=reconstruct, depth=depth_dataset, validate=validate)
    for name in stages if a.stage == 'all' else [a.stage]:
        write(a.output/'pipeline_status.json', dict(stage=name, status='running', updated_epoch=time.time()))
        try:
            stages[name](a)
        except Exception as exc:
            write(a.output/'pipeline_status.json', dict(stage=name, status='failed', error=str(exc), updated_epoch=time.time()))
            raise
        write(a.output/'pipeline_status.json', dict(stage=name, status='complete', updated_epoch=time.time()))


if __name__ == '__main__':
    main()
