"""Matched RGB-D / inertial experiments on the archived D455 graph and windows."""
import argparse
from contextlib import ExitStack
import csv
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image
from scipy.linalg import null_space
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.spatial.transform import Rotation

import db3_direct_depth_refine as direct
import db3_imu_audit as imu

db3 = imu.db3
ROOT = imu.ROOT
ARCHIVE = imu.HISTORICAL / 'rgbd'
MASKS = ROOT / 'results/db3_observation_masks_narrow_20261117'
REVISITS = db3.RUN / 'transitions_revisits_v2'


def sparsity(refs, block, columns):
    rows, cols = [], []
    for k, variables in enumerate(refs):
        rows.extend(np.repeat(np.arange(k * block, (k + 1) * block), len(variables)))
        cols.extend(np.tile(variables, block))
    return coo_matrix((np.ones(len(rows)), (np.array(rows, int), np.array(cols, int))),
                      shape=(len(refs) * block, columns)).tocsr()


def camera_variables(indices):
    return [column for i in indices if i for column in range((i - 1) * 6, i * 6)]


def solve(initial, function, structure, evaluations):
    before = function(initial)
    start = time.monotonic()
    result = least_squares(function, initial, jac_sparsity=structure, loss='soft_l1',
                           x_scale='jac', tr_solver='lsmr', tr_options={'maxiter': 400},
                           max_nfev=evaluations, ftol=1e-5, xtol=1e-6, gtol=1e-5)
    after = function(result.x)
    report = dict(success=bool(result.success), message=result.message, evaluations=result.nfev,
                  rms_initial=float(np.sqrt(np.mean(before ** 2))),
                  rms_final=float(np.sqrt(np.mean(after ** 2))), optimality=float(result.optimality),
                  elapsed_s=time.monotonic() - start)
    return result.x, report


def motion_factors(stamps, session):
    gyro = session['gyro']
    bias = session['gyro_bias']
    rc = session['imu_to_color'][:3, :3]
    valid, steps, integrated = [], [], []
    for i, (a, b) in enumerate(zip(stamps, stamps[1:])):
        readings, status = imu.interval(gyro, a, b, imu.GAPS['gyro'])
        if status == 'valid':
            valid.append(i)
            steps.append(readings)
            integrated.append(rc @ imu.rotation(readings, bias).as_matrix() @ rc.T)
    i = np.array(valid, dtype=int)
    duration = stamps[i + 1] - stamps[i]
    sigma = np.hypot(duration[:, None] * session['rate_sigma'], np.deg2rad(.05))

    def residual(c2w):
        r = c2w[:, :3, :3]
        predicted = r[i].transpose(0, 2, 1) @ r[i + 1]
        error = np.array(integrated).transpose(0, 2, 1) @ predicted
        return (Rotation.from_matrix(error).as_rotvec() / sigma).ravel()

    return i, steps, residual


def trajectory_change(reference, candidate):
    delta = np.linalg.inv(reference) @ candidate
    return dict(position_cm=imu.quantiles(np.linalg.norm(delta[:, :3, 3], axis=1) * 100),
                rotation_deg=imu.quantiles(np.rad2deg(Rotation.from_matrix(delta[:, :3, :3]).magnitude())))


def full_graph(out, stamps, session):
    saved = db3.read(ARCHIVE / 'pose_graph.json')
    with np.load(ARCHIVE / 'poses.npz') as data:
        a0 = data['camtoworlds']
        assert np.array_equal(data['indices'], np.arange(len(stamps)))
    assert np.array_equal(a0, np.array([n['pose'] for n in saved['nodes']]).reshape(-1, 4, 4).transpose(0, 2, 1))
    edges = saved['edges']
    source = np.array([e['source_node_id'] for e in edges])
    target = np.array([e['target_node_id'] for e in edges])
    measurement_inv = np.linalg.inv(np.array([e['transformation'] for e in edges]).reshape(-1, 4, 4).transpose(0, 2, 1))
    weight = np.array([np.asarray(e['information']).reshape(6, 6) * e['confidence'] for e in edges])
    whiten = np.linalg.cholesky(weight / 100).transpose(0, 2, 1) / .03
    initial = np.c_[Rotation.from_matrix(a0[1:, :3, :3]).as_rotvec(), a0[1:, :3, 3]].ravel()

    def unpack(x):
        q = x.reshape(-1, 6)
        poses = a0.copy()
        poses[1:, :3, :3] = Rotation.from_rotvec(q[:, :3]).as_matrix()
        poses[1:, :3, 3] = q[:, 3:]
        return poses

    def rgbd(poses):
        delta = measurement_inv @ np.linalg.inv(poses[target]) @ poses[source]
        errors = np.c_[Rotation.from_matrix(delta[:, :3, :3]).as_rotvec(), delta[:, :3, 3]]
        return np.einsum('nij,nj->ni', whiten, errors).ravel()

    indexes, batches, gyro = motion_factors(stamps, session)
    rgbd_structure = sparsity([camera_variables((a, b)) for a, b in zip(source, target)], 6, len(initial))
    gyro_structure = sparsity([camera_variables((a, a + 1)) for a in indexes], 3, len(initial))
    arms = {'A0': a0}
    statistics = {}
    for label in ('A1', 'B', 'C'):
        print('Full route start', label, flush=True)
        if label == 'A1':
            x, report = solve(initial, lambda x: rgbd(unpack(x)), rgbd_structure, 160)
        elif label == 'B':
            x, report = solve(initial, lambda x: np.r_[rgbd(unpack(x)), gyro(unpack(x))],
                              vstack([rgbd_structure, gyro_structure], format='csr'), 160)
        else:
            w2c = lambda x: np.linalg.inv(unpack(x))
            x, report = fit_full_imu(stamps, session, np.linalg.inv(a0), w2c, indexes, batches,
                                     initial, lambda x: rgbd(unpack(x)), rgbd_structure, 160)
        arms[label] = unpack(x)
        statistics[label] = report
        print('Full route solved', label, report['success'], report['evaluations'], flush=True)
    for label, poses in arms.items():
        statistics.setdefault(label, {}).update(
            rgbd_normalized_rms=float(np.sqrt(np.mean(rgbd(poses) ** 2))),
            gyro_normalized_rms=float(np.sqrt(np.mean(gyro(poses) ** 2))),
            versus_A0=trajectory_change(a0, poses), versus_A1=trajectory_change(arms['A1'], poses))
    np.savez_compressed(out / 'full_route_camtoworlds.npz', **arms)
    db3.write(out / 'full_route_optimization.json', dict(frames=len(a0), archived_edges=len(edges),
              gyro_factors=len(indexes), unsupported_gyro_boundaries=sorted(set(range(len(a0) - 1)) - set(indexes.tolist())),
              arms=statistics, C_status='Exploratory treatment with assumed accelerometer noise and bias evolution.'))
    return arms


def preintegrate(knots, gyro, accel, bias, propagate_covariance=True):
    """Midpoint body-frame integration with tangent covariance in (R, v, p)."""
    rot = Rotation.identity()
    vel, pos = np.zeros(3), np.zeros(3)
    covariance = np.zeros((9, 9))
    for k, dt in enumerate(np.diff(knots)):
        angular = .5 * (gyro[k] + gyro[k + 1]) - bias[:3]
        force = .5 * (accel[k] + accel[k + 1]) - bias[3:]
        midpoint = (rot * Rotation.from_rotvec(angular * dt * .5)).as_matrix()
        a = midpoint @ force
        pos += vel * dt + .5 * a * dt ** 2
        vel += a * dt
        if propagate_covariance:
            cross = np.cross(np.eye(3), force).T
            transition = np.eye(9)
            transition[:3, :3] = Rotation.from_rotvec(-angular * dt).as_matrix()
            transition[3:6, :3] = -midpoint @ cross * dt
            transition[6:9, :3] = -.5 * midpoint @ cross * dt ** 2
            transition[6:9, 3:6] = np.eye(3) * dt
            density = np.zeros((9, 6))
            density[:3, :3] = np.eye(3) * .005
            density[3:6, 3:] = midpoint * .10
            density[6:9, 3:] = midpoint * .10 * .5 * dt
            covariance = transition @ covariance @ transition.T + density @ density.T * dt
            # Integrated white acceleration also has an independent position term.
            covariance[6:9, 6:9] += np.eye(3) * .10 ** 2 * dt ** 3 / 12
        rot = rot * Rotation.from_rotvec(angular * dt)
    return rot, vel, pos, covariance


def full_imu_problem(stamps, session, initial_w2c, unpack, gyro_indices, gyro_batches, camera_size):
    bias0 = np.r_[session['gyro_bias'], np.zeros(3)]
    records = []
    for i, (gyro_t, omega) in zip(gyro_indices, gyro_batches):
        readings, status = imu.interval(session['accel'], stamps[i], stamps[i + 1], imu.GAPS['accel'])
        if status != 'valid':
            continue
        accel_t, accel = readings
        knots = np.union1d(gyro_t, accel_t)
        angular = np.column_stack([np.interp(knots, gyro_t, axis) for axis in omega.T])
        force = np.column_stack([np.interp(knots, accel_t, axis) for axis in accel.T])
        rot, vel, pos, cov = preintegrate(knots, angular, force, bias0)
        jacobian = []
        for direction in np.eye(6) * 1e-5:
            r1, v1, p1, _ = preintegrate(knots, angular, force, bias0 + direction, propagate_covariance=False)
            jacobian.append(np.r_[(rot.inv() * r1).as_rotvec(), v1 - vel, p1 - pos] / 1e-5)
        # Assumed model discrepancy for imperfect calibration and short windows.
        floor = np.r_[np.repeat(np.deg2rad(.05), 3), np.repeat(.015, 3), np.repeat(.002, 3)]
        white = np.linalg.inv(np.linalg.cholesky(cov + np.diag(floor ** 2)))
        records.append((i, rot.as_matrix(), vel, pos, np.array(jacobian).T, white))
    if not records:
        raise ValueError('No supported accelerometer factors in window')
    i, dr, dv, dp, jac, white = map(np.array, zip(*records))
    delta = (stamps[i + 1] - stamps[i])[:, None]
    initial_body = np.linalg.inv(initial_w2c) @ session['imu_to_color']
    velocities = np.gradient(initial_body[:, :3, 3], stamps, axis=0, edge_order=2)
    accel_t, accel = session['accel']
    samples = np.column_stack([np.interp(stamps, accel_t, axis) for axis in accel.T])
    apparent = np.einsum('nij,nj->ni', initial_body[:, :3, :3], samples)
    direction = -np.mean(apparent, axis=0)
    gravity0 = 9.80665 * direction / np.linalg.norm(direction)
    tilt_basis = null_space(gravity0[None])
    duration = stamps[-1] - stamps[0]
    bias_times = np.linspace(stamps[0], stamps[-1], int(np.ceil(duration / 16)) + 1) if duration > 16 else stamps[:1]
    bias_index = camera_size + velocities.size
    bias_columns = np.arange(bias_index, bias_index + len(bias_times) * 6).reshape(-1, 6)
    state_size = bias_columns[-1, -1] + 3
    gyro_walk, accel_walk = .001, .025
    bias_walk = np.sqrt(np.diff(bias_times))[:, None] * np.repeat([gyro_walk, accel_walk], 3)
    midpoints = stamps[i] + .5 * delta[:, 0]
    bias_left = np.clip(np.searchsorted(bias_times, midpoints, side='right') - 1, 0, max(len(bias_times) - 2, 0))
    bias_right = np.minimum(bias_left + 1, len(bias_times) - 1)
    fraction = (midpoints - bias_times[bias_left]) / np.maximum(bias_times[bias_right] - bias_times[bias_left], 1e-6)
    fraction = np.clip(fraction, 0, 1)[:, None]

    def state(x):
        body = np.linalg.inv(unpack(x[:camera_size])) @ session['imu_to_color']
        corrected_bias = x[bias_columns]
        gravity = Rotation.from_rotvec(tilt_basis @ x[-2:]).apply(gravity0)
        return body, x[camera_size:bias_index].reshape(-1, 3), corrected_bias, gravity

    def imu_residual(x):
        body, v, bias, g = state(x)
        r, p = body[:, :3, :3], body[:, :3, 3]
        inverse_ri = r[i].transpose(0, 2, 1)
        correction = np.einsum('nij,nj->ni', jac,
                                bias[bias_left] * (1 - fraction) + bias[bias_right] * fraction)
        measurement_r = dr @ Rotation.from_rotvec(correction[:, :3]).as_matrix()
        predicted_r = inverse_ri @ r[i + 1]
        velocity = np.einsum('nij,nj->ni', inverse_ri, v[i + 1] - v[i] - g * delta)
        position = np.einsum('nij,nj->ni', inverse_ri, p[i + 1] - p[i] - v[i] * delta - .5 * g * delta ** 2)
        error = np.c_[Rotation.from_matrix(measurement_r.transpose(0, 2, 1) @ predicted_r).as_rotvec(),
                      velocity - dv - correction[:, 3:6], position - dp - correction[:, 6:9]]
        weighted = np.einsum('nij,nj->ni', white, error)
        priors = np.r_[bias[0] / [.03, .03, .03, .5, .5, .5],
                       (np.diff(bias, axis=0) / bias_walk).ravel(), x[-2:] / np.deg2rad(15)]
        return np.r_[weighted.ravel(), priors]

    variables = [camera_variables((a, a + 1)) + list(range(camera_size + 3 * a, camera_size + 3 * (a + 2)))
                 + np.unique(bias_columns[[left, right]]).tolist()
                 + [state_size - 2, state_size - 1] for a, left, right in zip(i, bias_left, bias_right)]
    bias_priors = [bias_columns[0].tolist()]
    bias_priors.extend(np.r_[a, b].tolist() for a, b in zip(bias_columns, bias_columns[1:]))
    structure = vstack([sparsity(variables, 9, state_size),
                        sparsity(bias_priors, 6, state_size),
                        sparsity([[state_size - 2, state_size - 1]], 2, state_size)], format='csr')
    return velocities, imu_residual, structure, state, bias0, bias_times, len(i)


def fit_full_imu(stamps, session, initial_w2c, unpack, ids, batches, x0, rgbd, rgbd_structure, evaluations):
    velocity, inertial, structure, state, bias0, bias_times, factors = full_imu_problem(
        stamps, session, initial_w2c, unpack, ids, batches, len(x0))
    extra = structure.shape[1] - len(x0)
    common_structure = csr_matrix((rgbd_structure.data, rgbd_structure.indices, rgbd_structure.indptr),
                                  shape=(rgbd_structure.shape[0], structure.shape[1]))
    x, report = solve(np.r_[x0, velocity.ravel(), np.zeros(extra - velocity.size)],
                      lambda x: np.r_[rgbd(x[:len(x0)]), inertial(x)],
                      vstack([common_structure, structure], format='csr'), evaluations)
    _, v, bias, gravity = state(x)
    report.update(full_imu_factors=factors, bias_rad_s_and_m_s2=(bias0 + bias).tolist(),
                  bias_device_s=bias_times.tolist(), gravity_m_s2=gravity.tolist(),
                  imu_velocity_m_s=v.tolist(),
                  status_of_calibration='Exploratory assumed covariance and bias priors.')
    return x[:len(x0)], report


def stable_depth(depth):
    kernel = np.ones((3, 3), np.uint8)
    lo, hi = cv2.erode(depth, kernel), cv2.dilate(depth, kernel)
    return (lo >= .3) & (hi <= 6.) & (hi - lo < np.maximum(.03, .015 * depth))


def dense_checks(features, checks, arms, masks, K):
    depths = [f['depth'] for f in features]
    stable = [stable_depth(z) & ~cv2.dilate(mask.astype('u1'), np.ones((7, 7), 'u1')).astype(bool)
              for z, mask in zip(depths, masks)]
    yy, xx = np.mgrid[:depths[0].shape[0]:4, :depths[0].shape[1]:4]
    pixels = np.c_[xx.ravel(), yy.ravel()]
    pixels = pixels[db3.check_tile(pixels)]
    results = []
    for row in checks:
        a, b = row['cameras']
        for i, j in ((a, b), (b, a)):
            uv = pixels[stable[i][pixels[:, 1], pixels[:, 0]]]
            x = db3.backproject(uv, depths[i][uv[:, 1], uv[:, 0]], K)
            errors = {}
            for label, w in arms.items():
                transformed = direct.r.transform_points(w[j] @ np.linalg.inv(w[i]), x)
                z = transformed[:, 2]
                projected = transformed[:, :2] / np.maximum(z[:, None], .01) * [K[0, 0], K[1, 1]] + K[:2, 2]
                q = np.rint(projected).astype(int)
                h, width = depths[j].shape
                keep = (z > .3) & (q[:, 0] >= 0) & (q[:, 0] < width) & (q[:, 1] >= 0) & (q[:, 1] < h)
                ids = np.flatnonzero(keep)
                q = q[ids]
                ids = ids[stable[j][q[:, 1], q[:, 0]]]
                q = np.rint(projected[ids]).astype(int)
                diff = z[ids] - depths[j][q[:, 1], q[:, 0]]
                # A nearer target measurement occludes the predicted sample.
                visible = diff <= np.maximum(.10, .03 * z[ids])
                error = np.full(len(x), np.nan)
                error[ids[visible]] = np.abs(diff[visible])
                errors[label] = error
            common = np.isfinite(list(errors.values())).all(axis=0)
            results.append(dict(cameras=[i, j], source_count=len(x), shared_count=int(common.sum()),
                                arms={label: dict(coverage=float(np.isfinite(e).mean()),
                                                  shared_median_p90_cm=(100 * np.percentile(e[common], [50, 90])).tolist()
                                                  if common.any() else None)
                                      for label, e in errors.items()}))
    return results


def clouds(directory, arms, samples, count):
    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                      ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    header = ('ply\nformat binary_little_endian 1.0\nelement vertex '
              f'{count}\nproperty float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with ExitStack() as stack:
        outputs = {k: stack.enter_context((directory / f'{k}_same_measured_points.ply').open('wb')) for k in arms}
        for stream in outputs.values():
            stream.write(header.encode('ascii'))
        written = 0
        for i, (xyz, rgb) in enumerate(samples):
            points = np.empty(len(xyz), dtype=dtype)
            points['red'], points['green'], points['blue'] = rgb.T
            for k, w in arms.items():
                points['x'], points['y'], points['z'] = direct.r.transform_points(w[i], xyz).T
                points.tofile(outputs[k])
            written += len(xyz)
        assert written == count


def local_windows(out, session):
    meta = imu.metadata()['color']
    timestamps = {r['rgb_message_id']: r['timestamp_ns']
                  for r in db3.read(db3.R1 / 'prospective_checks.json')['source_frames']}
    K = np.array(db3.read(db3.INPUT / 'calibration_clean.json')['K'])
    for case in db3.read(imu.LOCAL / 'protocol.json')['cases']:
        label = case['label']
        print('Local window start', label, flush=True)
        dest = out / label
        dest.mkdir()
        prior, inputs = direct.PARENT / label, direct.INPUTS / label
        rows = db3.read(inputs / 'inputs.json')
        assert [x['source_id'] for x in rows] == case['source_ids']
        originals = []
        stamps = np.array([imu.device_clock(meta[timestamps[r['source_id']]], 'Sensor Timestamp') for r in rows])
        for row in rows:
            path = inputs / 'inputs' / f"{row['source_id']}.npz"
            assert db3.sha(path) == row['feature_sha256']
            with np.load(path) as data:
                originals.append(dict(data))
        with np.load(prior / 'fit_features.npz') as data:
            fit = [dict(uv=data[f'uv_{c}'], depth=f['depth'],
                        check=np.zeros(len(data[f'uv_{c}']), bool)) for c, f in enumerate(originals)]
        with np.load(inputs / 'moving_checks.npz') as data:
            checkfs = [dict(uv=uv, depth=f['depth']) for uv, f in zip(data['uv'], originals)]
        for features in (fit, originals, checkfs):
            direct.r.observations(features, K)
        tracks = db3.read(prior / 'tracks.json')['tracks']
        initial = np.array(db3.read(prior / 'initial_poses.json'))
        x0, rgbd, unpack, transformed, rgbd_structure, refs = direct.fit_problem(fit, tracks, initial, K)
        archived_fit = db3.read(imu.LOCAL / label / 'optimization.json')
        assert np.array_equal(refs, archived_fit['fitting_pair_refs'])
        ids, batches, gyro = motion_factors(stamps, session)
        gyro_structure = sparsity([camera_variables((a, a + 1)) for a in ids], 3, len(x0))
        local = {'A0': np.array(db3.read(imu.LOCAL / label / 'poses.json'))}
        results = {}
        support = {}
        for arm in ('A1', 'B', 'C'):
            if arm == 'A1':
                x, report = solve(x0, rgbd, rgbd_structure, 240)
            elif arm == 'B':
                x, report = solve(x0, lambda x: np.r_[rgbd(x), gyro(np.linalg.inv(unpack(x)))],
                                  vstack([rgbd_structure, gyro_structure], format='csr'), 240)
            else:
                x, report = fit_full_imu(stamps, session, initial, unpack, ids, batches, x0, rgbd, rgbd_structure, 240)
            local[arm] = unpack(x)
            front = all(np.all(q[:, 2] > .01) for q in transformed(x))
            support[arm] = front and report['success']
            results[arm] = dict(optimization=report, fitted_measurements_in_front=bool(front))
            print('Local window solved', label, arm, report['success'], report['evaluations'], flush=True)
        checks = db3.read(inputs / 'frozen_checks.json')
        orig_checks = db3.read(inputs / 'original_checks.json')
        for arm, w in local.items():
            measured = direct.r.evaluate(checkfs, checks, w, K)
            original = direct.r.evaluate(originals, orig_checks, w, K)
            results.setdefault(arm, {}).update(moving=measured, descriptors=original,
                moving_gates_pass=direct.r.support_pass(measured) and support.get(arm, True),
                versus_A0=trajectory_change(np.linalg.inv(local['A0']), np.linalg.inv(w)))
        with np.load(MASKS / label / 'masks.npz') as masks:
            exclusions = (masks['people_candidate'] | masks['reflection_glare']).astype(bool)
        depths = dense_checks(originals, checks, local, exclusions, K)
        np.savez_compressed(dest / 'w2c.npz', source_ids=case['source_ids'], **local)
        db3.write(dest / 'metrics.json', dict(label=label, measured_rgbd_factors=len(refs),
                  gyro_factors=len(ids), arms=results, stable_depth_checks=depths))
        samples = []
        for c, (row, feature) in enumerate(zip(rows, originals)):
            z = feature['depth']
            y, x = np.nonzero((z >= .3) & (z <= 6.))
            x, y = x[::16], y[::16]
            path = inputs / 'inputs' / f"{row['source_id']}.png"
            assert db3.sha(path) == row['image_sha256']
            rgb = np.asarray(Image.open(path).convert('RGB'))
            samples.append((db3.backproject(np.c_[x, y], z[y, x], K), rgb[y, x]))
        clouds(dest, {arm: np.linalg.inv(w) for arm, w in local.items()}, samples, sum(len(x) for x, _ in samples))


def frozen_pair(a, b, K):
    with np.load(a) as fa, np.load(b) as fb:
        ca, cb = fa['check'], fb['check']
        matches = db3.mutual(fa['desc'][ca], fb['desc'][cb])
        uv, vv = fa['uv'][ca][matches[:, 0]], fb['uv'][cb][matches[:, 1]]
        da, va, _ = db3.sample_depth(fa['depth'], uv)
        db, vb, _ = db3.sample_depth(fb['depth'], vv)
        valid = va & vb
        return db3.backproject(uv[valid], da[valid], K), db3.backproject(vv[valid], db[valid], K)


def score_route(out, arms, frames):
    K = np.array(db3.read(db3.INPUT / 'calibration_clean.json')['K'])
    route, revisit = [], []
    archive = db3.read(imu.R4 / 'selected_edges.json')['edges']
    native = {r['boundary']: r for r in db3.read(imu.R4 / 'native_retries.json')['retries']}
    by_id = {f['message_id']: i for i, f in enumerate(frames)}

    def metrics(x, y, transform):
        predicted_y = direct.r.transform_points(transform, x)
        e = np.linalg.norm(predicted_y - y, axis=1)
        stats = np.percentile(e, [50, 90]).tolist() if len(e) else None
        predicted_x = direct.r.transform_points(np.linalg.inv(transform), y)
        front = (predicted_y[:, 2] > .01) & (predicted_x[:, 2] > .01)

        def project(points):
            return points[:, :2] / points[:, 2:3] * [K[0, 0], K[1, 1]] + K[:2, 2]

        pixel = np.maximum(np.linalg.norm(project(predicted_x[front]) - project(x[front]), axis=1),
                           np.linalg.norm(project(predicted_y[front]) - project(y[front]), axis=1))
        return dict(matches=len(e), median_p90_m=stats,
                    reprojection_matches=len(pixel),
                    median_p90_px=np.percentile(pixel, [50, 90]).tolist() if len(pixel) else None,
                    endpoint_pass=bool(len(e) >= 20 and stats[0] < .05 and stats[1] < .12))

    hashes = {record['source_id']: record['sha256']
              for record in db3.read(imu.R4 / 'inputs_manifest.json')['frames']}
    paths = {}
    for edge in archive:
        i = edge['boundary']
        a, b = edge['source_ids']
        assert [by_id[a], by_id[b]] == [i, i + 1]
        for key in (a, b):
            if key not in paths:
                paths[key] = imu.R4 / 'inputs' / f'{key}.npz'
                assert db3.sha(paths[key]) == hashes[key]
        x, y = frozen_pair(paths[a], paths[b], K)
        assert len(x) == edge['check_matches']
        support = (edge.get('fit_hull_fraction', 0) > .05
                   and edge.get('information_eigenvalues', [0])[0] > 1
                   and edge.get('information_condition', np.inf) < 1e7)
        original_pass = bool(support and metrics(x, y, np.array(edge['transform']))['endpoint_pass'])
        assert original_pass == edge['accepted'], i
        scores = {k: metrics(x, y, np.linalg.inv(p[i + 1]) @ p[i]) for k, p in arms.items()}
        route.append(dict(boundary=i, source_ids=[a, b], captures_s=edge['seconds'],
                          archived_pass=bool(edge['accepted'] or native.get(i, {}).get('accepted')),
                          same_direct_fit_support=bool(support), arms=scores))
    for row in db3.read(REVISITS / 'report.json')['revisits']:
        source_ids = row.get('source_ids')
        if source_ids is None:
            revisit.append(dict(a=row['a'], b=row['b'], status=row['reason']))
            continue
        a, b = source_ids
        paths = [REVISITS / 'inputs' / f'{key}.npz' for key in (a, b)]
        if a not in by_id or b not in by_id or not all(p.is_file() for p in paths):
            revisit.append(dict(source_ids=[a, b], status='Archived graph camera or check data unavailable.'))
            continue
        x, y = frozen_pair(*paths, K)
        assert len(x) == row['check_matches']
        revisit.append(dict(source_ids=[a, b], archived_globally_accepted=row['globally_accepted'],
                            arms={k: metrics(x, y, np.linalg.inv(p[by_id[b]]) @ p[by_id[a]])
                                  for k, p in arms.items()}))
    def summarize(rows, label):
        checks = [r['arms'][label] for r in rows]
        eligible = [r for r in checks if r['matches'] >= 20]
        return dict(boundaries=len(rows), supported_boundaries=len(eligible),
                    across_pair_medians_m=np.median([r['median_p90_m'] for r in eligible], axis=0).tolist()
                    if eligible else None,
                    across_pair_medians_px=np.median([r['median_p90_px'] for r in eligible], axis=0).tolist()
                    if eligible and all(r['reprojection_matches'] >= 20 for r in eligible) else None)

    groups = {}
    for row in route:
        # Group by capture time to expose regional variation without pooling pixels.
        group = int(row['captures_s'][0] // 20)
        groups.setdefault(group, []).append(row)
    db3.write(out / 'route_checks.json', dict(
        original_pass=sum(r['archived_pass'] for r in route), boundaries=len(route),
        arms={k: dict(same_direct_gates_pass=sum(r['same_direct_fit_support'] and r['arms'][k]['endpoint_pass'] for r in route),
                      previously_failed_now_direct_pass=sum(not r['archived_pass'] and r['same_direct_fit_support']
                                                           and r['arms'][k]['endpoint_pass'] for r in route),
                      retained_prior_pass=sum(r['archived_pass'] and r['same_direct_fit_support']
                                              and r['arms'][k]['endpoint_pass'] for r in route),
                      **summarize(route, k))
              for k in arms},
        twenty_second_groups=[dict(start_s=index * 20, arms={k: summarize(rows, k) for k in arms})
                              for index, rows in groups.items()],
        qualification='Re-score archived direct checks using global endpoint poses and fixed direct fit gates. The native retry still needs native camera poses, so count these as direct-gate results. Historical graph observation overlap is possible.',
        adjacent=route, revisits=revisit))


def selected_clouds(out, arms, frames):
    K = np.array(db3.read(db3.INPUT / 'calibration_clean.json')['K'])

    def samples():
        for f in frames:
            name = f['name']
            with np.load(db3.INPUT / 'aligned_depth_meters' / (name + '.npz')) as data:
                uv, z = data['points'][::16], data['depths'][::16]
            rgb = np.asarray(Image.open(db3.INPUT / 'images' / name).convert('RGB'))
            x, y = np.rint(uv).astype(int).T
            yield db3.backproject(uv, z, K), rgb[y, x]

    count = sum((f['aligned_samples'] + 15) // 16 if f['aligned_samples'] < 16000 else 1000 for f in frames)
    clouds(out, arms, samples(), count)


def run(audit, out, stage, route_poses=None):
    out.mkdir(parents=True, exist_ok=False)
    config = db3.read(audit / 'audit.json')
    protocol = db3.read(audit / 'audit_protocol.json')
    for name, checksum in protocol['inputs'].items():
        assert db3.sha(ROOT / name) == checksum, name
    imu_to_camera = np.eye(4)
    imu_to_camera[:3, :3] = config['factory_imu_to_color']['gyro']['rotation']
    imu_to_camera[:3, 3] = config['factory_imu_to_color']['gyro']['translation_m']
    assert config['factory_imu_to_color']['gyro'] == config['factory_imu_to_color']['accel']
    cv2.setNumThreads(4)
    extra_inputs = [REVISITS / 'report.json']
    for case in db3.read(imu.LOCAL / 'protocol.json')['cases']:
        label = case['label']
        extra_inputs.extend(d / name for d, names in [
            (direct.PARENT / label, ['fit_features.npz', 'tracks.json', 'initial_poses.json']),
            (direct.INPUTS / label, ['inputs.json', 'moving_checks.npz', 'frozen_checks.json', 'original_checks.json']),
            (imu.LOCAL / label, ['poses.json', 'optimization.json']),
            (MASKS / label, ['masks.npz'])] for name in names)
    db3.write(out / 'protocol.json', dict(
        stage=stage, inputs={str(f.relative_to(ROOT)): db3.sha(f) for f in extra_inputs},
        route_poses_sha256=db3.sha(route_poses) if route_poses is not None else None,
        audit_inputs_sha256={p: db3.sha(audit / p) for p in ('audit_protocol.json', 'audit.json', 'synchronized_streams.npz')},
        method='SciPy sparse robust least squares, common RGB-D factors and fixed first camera. Route reuses archive edge information/confidence, residual scaling /0.03 and /sqrt(100). Local replays all exact measured RGB-D factors from direct-depth fit.',
        gyro='Only supported adjacent intervals. Fixed recorded frame rotation and 0 ms lag. Robust fitted constant bias and /sqrt((duration*rate_sigma)^2+(0.05 degree)^2).',
        C='Exploratory midpoint preintegration with position, velocity, fixed-magnitude optimized gravity, and constant biases per short window. For a route longer than 16 seconds interpolate bias knots at no more than 16 seconds with random walk 0.001 rad/s/sqrt(s), 0.025 m/s²/sqrt(s). Assumed noise densities 0.005 rad/s/sqrt(Hz), 0.1 m/s²/sqrt(Hz), covariance floors 0.05 degree, 0.015 m/s, 0.002 m. Initial bias priors 0.03 rad/s, 0.5 m/s² and gravity tilt prior 15 degrees.',
        gates='Keep R4 thresholds. Local promotion needs original moving support and meaningful 1 cm held-out endpoint improvement with no >1 cm regression on retained geometric checks. Route revisit promotion requires independent verified distributed structures. No training without validation of a treatment.',
        dense='Spatially reserved check tiles, stride 4, stable 3x3 raw depth, archived candidate person/glare exclusions expanded three pixels, reject predicted samples occluded by closer observed depth. Score common source pixels across arms and report each arm coverage.',
        source_sha256=db3.sha(Path(__file__))))
    shutil.copyfile(__file__, out / Path(__file__).name)
    with np.load(audit / 'synchronized_streams.npz') as data:
        session = dict(gyro=(data['gyro_device_s'], data['gyro_rad_s']),
                       accel=(data['accel_device_s'], data['accel_m_s2']),
                       gyro_bias=np.array(config['gyro_calibration']['gyro_bias_rad_s']),
                       rate_sigma=np.array(config['gyro_calibration']['effective_rate_sigma_rad_s']),
                       imu_to_color=imu_to_camera)
        if stage == 'local':
            local_windows(out, session)
        else:
            frames = db3.read(db3.INPUT / 'capture_manifest.json')['frames']
            assert np.array_equal(data['rgb_message_id'], [f['message_id'] for f in frames])
            if route_poses is None:
                arms = full_graph(out, data['camera_device_s'], session)
            else:
                with np.load(route_poses) as archive:
                    arms = {k: archive[k] for k in ('A0', 'A1', 'B', 'C')}
                for poses in arms.values():
                    assert poses.shape == (len(frames), 4, 4) and np.isfinite(poses).all()
                shutil.copyfile(route_poses, out / 'full_route_camtoworlds.npz')
            score_route(out, arms, frames)
            selected_clouds(out, arms, frames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['local', 'route'])
    parser.add_argument('audit', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--route-poses', type=Path, help='Rescore an already completed route optimization.')
    args = parser.parse_args()
    if args.route_poses is not None and args.stage != 'route':
        parser.error('--route-poses requires route')
    run(args.audit, args.output_dir, args.stage, args.route_poses)
