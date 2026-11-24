"""Read the saved D455 motion streams and freeze device-clock interval support."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import db3_local_geometry as db3

ROOT = db3.ROOT
HISTORICAL = ROOT / 'data/realsense_20261030_134523_clean'
R4 = ROOT / 'results/db3_r4_20261111_v2'
LOCAL = ROOT / 'results/db3_direct_depth_20261111'
METADATA = db3.R1 / 'timing_metadata.json'
GAPS = {'gyro': .0075, 'accel': .015}
OFFSET_GRID = np.array([-20, -10, -5, 0, 5, 10, 20]) / 1000


def quantiles(values):
    return np.percentile(values, [0, 50, 95, 100]).tolist()


def metadata():
    return {
        path.split('/')[3].split('_')[0].lower(): {
            int(row['timestamp_ns']): db3.clean.fields(row['text'].rstrip(';'))
            for row in records
        }
        for path, records in db3.read(METADATA).items()
    }


def device_clock(fields, name='Frame Timestamp'):
    return int(fields[name]) / 1e6


def motion(meta, kind):
    with (HISTORICAL / 'imu' / f'{kind}.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    playback = np.array([int(row['timestamp_ns']) for row in rows], dtype=np.int64)
    if len(meta) != len(rows) or set(playback) != meta.keys():
        raise ValueError(f'{kind} metadata and sample rosters differ')
    hardware = np.array([device_clock(meta[t]) for t in playback])
    sequence = np.array([int(meta[t]['Frame number']) for t in playback])
    xyz = np.array([[float(row[axis]) for axis in 'xyz'] for row in rows])
    if not (np.all(np.diff(hardware) > 0) and np.isfinite(xyz).all()
            and np.all(np.diff(playback) > 0) and np.all(np.diff(sequence) > 0)):
        raise ValueError(f'Invalid {kind} times, frame numbers or measurements')
    report = dict(samples=len(rows), units='rad/s' if kind == 'gyro' else 'm/s²',
                  stream_frame_ids=sorted({row['frame_id'] for row in rows}),
                  header_playback_max_difference_ns=float(np.max(np.abs(
                      playback - np.rint([float(row['header_seconds']) * 1e9 for row in rows])))),
                  device_timing_ms=quantiles(1000 * np.diff(hardware)),
                  playback_timing_ms=quantiles(1e-6 * np.diff(playback)),
                  missing_between_frame_numbers=int(np.sum(np.diff(sequence) - 1)),
                  discontinuities=(np.flatnonzero(np.diff(sequence) != 1) + 1).tolist(),
                  magnitude=quantiles(np.linalg.norm(xyz, axis=1)))
    return hardware, xyz, report


def interval(stream, start, stop, max_gap):
    time, xyz = stream
    left = np.searchsorted(time, start, side='right') - 1
    right = np.searchsorted(time, stop, side='left')
    if left < 0 or right >= len(time) or stop <= start:
        return None, 'unbracketed'
    if np.max(np.diff(time[left:right + 1])) > max_gap:
        return None, 'sample_gap'
    knots = np.r_[start, time[left + 1:right], stop]
    return (knots, np.column_stack([np.interp(knots, time, axis) for axis in xyz.T])), 'valid'


def rotation(increments, bias):
    knots, omega = increments
    dt = np.diff(knots)
    steps = Rotation.from_rotvec((.5 * (omega[:-1] + omega[1:]) - bias) * dt[:, None])
    result = Rotation.identity()
    for step in steps:
        result = result * step
    return result


def calibration(gyro, camera_times, imu_to_camera, edges):
    # Use only statistics of the fit points. The R4 check counts, check errors,
    # and accept/reject labels are unused. Keep both scored windows out of calibration.
    fit, check = [], []
    for edge in edges:
        i = edge['boundary']
        start, stop = camera_times[i:i + 2]
        relative = start - camera_times[0]
        if (8 <= relative <= 16 or 116 <= relative <= 125
                or edge.get('inliers', 0) < 35 or edge['fit_hull_fraction'] <= .05
                or edge['information_condition'] >= 1e7):
            continue
        batches = [interval(gyro, start + lag, stop + lag, GAPS['gyro'])[0] for lag in OFFSET_GRID]
        if any(batch is None for batch in batches):
            continue
        # R4 T maps camera i points into j. The gyro integrates R_i^T R_j.
        expected = imu_to_camera.inv() * Rotation.from_matrix(np.array(edge['transform'])[:3, :3]).inv() * imu_to_camera
        group = check if int(relative // 8) % 3 == 2 else fit
        group.append((i, batches, expected))
    if min(len(fit), len(check)) < 12:
        raise ValueError('Insufficient RGB-D calibration or validation intervals')

    def errors(group, offset, bias):
        return np.array([(rot.inv() * rotation(batches[offset], bias)).as_rotvec()
                         for _, batches, rot in group])

    scores = []
    for n, lag in enumerate(OFFSET_GRID):
        sol = least_squares(lambda bias: errors(fit, n, bias).ravel(), np.zeros(3),
                            loss='soft_l1', f_scale=np.deg2rad(.5))
        e = errors(check, n, sol.x)
        scores.append(dict(offset_ms=float(lag * 1000), fitted_bias_rad_s=sol.x.tolist(),
                           fit_error_deg=quantiles(np.rad2deg(np.linalg.norm(errors(fit, n, sol.x), axis=1))),
                           reserved_calibration_error_deg=quantiles(np.rad2deg(np.linalg.norm(e, axis=1)))))
    # Hardware timestamp is fixed before scoring. The profile diagnoses timing
    # sensitivity without choosing the best offset on the calibration check set.
    zero = int(np.flatnonzero(OFFSET_GRID == 0)[0])
    bias = np.array(scores[zero]['fitted_bias_rad_s'])
    e = errors(check, zero, bias)
    # Conservative combined camera/IMU repeatability, not isolated sensor noise.
    # Scale the rate estimate with interval duration and keep a floor for short links.
    duration = np.array([batch[zero][0][-1] - batch[zero][0][0] for _, batch, _ in check])
    rate_sigma = np.maximum(.02, np.median(np.abs(e / duration[:, None]), axis=0) / .67449)
    return dict(fit_boundary_indices=[i for i, *_ in fit], check_boundary_indices=[i for i, *_ in check],
                offset_profile=scores, selected_offset_ms=0, gyro_bias_rad_s=bias.tolist(),
                effective_rate_sigma_rad_s=rate_sigma.tolist(),
                uncorrected_at_zero_check_error_deg=quantiles(np.rad2deg(np.linalg.norm(
                    errors(check, zero, np.zeros(3)), axis=1))))


def audit(out):
    out.mkdir(parents=True, exist_ok=False)
    raw_cal = db3.read(db3.INPUT / 'calibration_raw.json')
    archives = [HISTORICAL / 'capture_manifest.json', HISTORICAL / 'rgbd/poses.npz',
                HISTORICAL / 'rgbd/pose_graph.json', HISTORICAL / 'rgbd/edges.json',
                HISTORICAL / 'rgbd/calibration.json', HISTORICAL / 'rgbd/report.json',
                HISTORICAL / 'imu/gyro.csv', HISTORICAL / 'imu/accel.csv', METADATA,
                db3.INPUT / 'capture_manifest.json', db3.INPUT / 'calibration_raw.json',
                db3.INPUT / 'calibration_clean.json', db3.INPUT / 'reconstruction_split.json',
                db3.R1 / 'prospective_checks.json', db3.R1 / 'baseline_manifest.json',
                R4 / 'protocol.json', R4 / 'selected_edges.json', R4 / 'native_retries.json',
                R4 / 'inputs_manifest.json', LOCAL / 'protocol.json', LOCAL / 'result.json']
    shutil.copyfile(__file__, out / 'db3_imu_audit.py')
    db3.write(out / 'audit_protocol.json', dict(
        inputs={str(p.relative_to(ROOT)): db3.sha(p) for p in archives},
        source_sha256=db3.sha(Path(__file__)), maximum_sample_gap_ms={k: v * 1000 for k, v in GAPS.items()},
        camera_clock='Recorded RGB device exposure midpoint in microseconds. Preserve RGB/depth pair IDs.',
        motion_clock='Recorded device Frame Timestamp. Exact matching to each exported playback timestamp.',
        profile='Fixed zero device-clock lag for fitting; compare seven offsets on disjoint RGB-D fitting links before testing.',
        bias='Robust constant estimated from R4 fit-only relative rotations with >=35 inliers, >5% hull, condition <1e7. Exclude both tested windows. Reserve every third 8 second block.'))
    meta = metadata()
    clock = {}
    for kind, records in meta.items():
        rows = list(records.values())
        if {r['timestamp_domain'] for r in rows} != {'Global Time'}:
            raise ValueError(f'Mixed or undocumented time domain in {kind}')
        clock_residual = np.array([float(r['timestamp']) / 1000 - device_clock(r) for r in rows])
        clock[kind] = 1000 * (clock_residual - 1793342673.445)
    frames = db3.read(db3.INPUT / 'capture_manifest.json')['frames']
    baseline = db3.read(HISTORICAL / 'capture_manifest.json')['frames']
    assert all([a[k] for k in ('message_id', 'depth_message_id', 'timestamp_ns', 'name')] ==
               [b[k] for k in ('message_id', 'depth_message_id', 'timestamp_ns', 'name')]
               for a, b in zip(frames, baseline, strict=True))
    seconds = np.array([device_clock(meta['color'][f['timestamp_ns']], 'Sensor Timestamp') for f in frames])
    streams, reports = {}, {}
    for kind in GAPS:
        t, xyz, reports[kind] = motion(meta[kind], kind)
        streams[kind] = t, xyz
    imu_to_color = {}
    for sensor in ('Gyro', 'Accel'):
        rot, shift = db3.projection.relative_extrinsics(
            raw_cal[f'/device_0/sensor_2/{sensor}_0/tf/ref_0'],
            raw_cal['/device_0/sensor_1/Color_0/tf/ref_0'])
        imu_to_color[sensor.lower()] = dict(rotation=Rotation.from_matrix(rot).as_matrix().tolist(),
                                             translation_m=shift.tolist())
    cal = calibration(streams['gyro'], seconds, Rotation.from_matrix(imu_to_color['gyro']['rotation']),
                      db3.read(R4 / 'selected_edges.json')['edges'])
    records = []
    for i, (a, b) in enumerate(zip(frames, frames[1:])):
        start, stop = seconds[i:i + 2]
        row = dict(boundary=i, from_rgb_id=a['message_id'], to_rgb_id=b['message_id'],
                   start_device_s=start, end_device_s=stop)
        for kind, (timestamp, _) in streams.items():
            row[f'{kind}_samples'] = int(np.sum((timestamp > start) & (timestamp <= stop)))
            _, row[f'{kind}_status'] = interval(streams[kind], start, stop, GAPS[kind])
        records.append(row)
    with (out / 'selected_intervals.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    # Fastest available low-motion proxy is still moving. Do not assign
    # accelerometer bias or absolute gravity using a presumed stationary sample.
    gyro_t, omega = streams['gyro']
    starts = np.arange(gyro_t[0], gyro_t[-1] - .5, .05)
    indices = np.searchsorted(gyro_t, starts[:, None] + [0, .5])
    speed = np.linalg.norm(omega - cal['gyro_bias_rad_s'], axis=1)
    best, lo, hi = min((float(np.percentile(speed[a:b], 90)), a, b) for a, b in indices)
    db3.write(out / 'audit.json', dict(
        selected_frames=len(frames), selected_intervals=len(records),
        coverage={kind: sum(row[f'{kind}_status'] == 'valid' for row in records) for kind in GAPS},
        both_valid=sum(all(row[f'{kind}_status'] == 'valid' for kind in GAPS) for row in records),
        streams=reports, factory_imu_to_color=imu_to_color,
        imu_intrinsics={k: v for k, v in raw_cal.items() if k.endswith('/imu_intrinsic')},
        global_minus_device_offset_ms_relative_to_common_epoch={k: quantiles(v) for k, v in clock.items()},
        nominal_near_stationary_window=dict(start_device_s=float(gyro_t[lo]), end_device_s=float(gyro_t[hi-1]),
                                             corrected_gyro_p90_rad_s=best, certified_stationary=False),
        gyro_calibration=cal, calibrated_accel_bias_noise=False))
    np.savez_compressed(out / 'synchronized_streams.npz', camera_device_s=seconds,
                        rgb_message_id=[f['message_id'] for f in frames],
                        depth_message_id=[f['depth_message_id'] for f in frames],
                        gyro_device_s=streams['gyro'][0], gyro_rad_s=streams['gyro'][1],
                        accel_device_s=streams['accel'][0], accel_m_s2=streams['accel'][1])
    print(f'Audited {len(frames)} RGB-D frames and {len(records)} consecutive intervals', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    audit(parser.parse_args().output_dir)
