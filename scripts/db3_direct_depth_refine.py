"""Bounded R4 follow-up: fit measured-depth projections with fixed holdouts.

Camera-only optimization avoids substituting fitted landmark depths for the
sensor depths used by acceptance. This is exposed development, not truth data.
"""
import argparse
import importlib.util
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

spec = importlib.util.spec_from_file_location('tracked', Path(__file__).with_name('db3_tracked_fit.py'))
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
p, r, m = t.p, t.r, t.m
PARENT = r.ROOT / 'results/db3_tracked_fit_20261111'
INPUTS = t.PARENT


def fit_problem(features, tracks, initial, K):
    """Seven residuals per fitting pair: two projections and metric separation."""
    nc = len(initial)
    pairs = {(a, b) for a in range(nc) for b in range(a + 1, nc)
             if b - a in (1, 2, 3, 5)}
    pairs |= {(0, nc - 1), (0, nc // 2), (nc // 2, nc - 1)}
    refs = []
    for tid, tr in enumerate(tracks[:800]):
        ids = dict(tr)
        for a, b in sorted(pairs):
            if a in ids and b in ids:
                ia, ib = ids[a], ids[b]
                if features[a]['valid'][ia] and features[b]['valid'][ib]:
                    assert not features[a]['check'][ia] and not features[b]['check'][ib]
                    refs.append((tid, a, ia, b, ib))
    if len({x[0] for x in refs}) < 30:
        raise ValueError('Fewer than 30 stable-depth fitting tracks')
    refs = np.array(refs, dtype=int)
    a, b = refs[:, 1], refs[:, 3]
    xa = np.array([features[c]['xyz'][i] for c, i in refs[:, 1:3]])
    xb = np.array([features[c]['xyz'][i] for c, i in refs[:, 3:5]])
    ua = np.array([features[c]['uv'][i] for c, i in refs[:, 1:3]])
    ub = np.array([features[c]['uv'][i] for c, i in refs[:, 3:5]])
    sigma = np.maximum(.01, np.sqrt(
        np.array([features[c]['sigma'][i] for c, i in refs[:, 1:3]]) ** 2 +
        np.array([features[c]['sigma'][i] for c, i in refs[:, 3:5]]) ** 2))
    cams = np.c_[Rotation.from_matrix(initial[:, :3, :3]).as_rotvec(), initial[:, :3, 3]]
    x0 = cams[1:].ravel()

    def unpack(x):
        c = np.vstack((cams[0], x.reshape(-1, 6)))
        poses = np.repeat(np.eye(4)[None], nc, axis=0)
        poses[:, :3, :3] = Rotation.from_rotvec(c[:, :3]).as_matrix()
        poses[:, :3, 3] = c[:, 3:]
        return poses

    def transformed(x):
        poses = unpack(x)
        R, trans = poses[:, :3, :3], poses[:, :3, 3]
        world_a = np.einsum('nji,nj->ni', R[a], xa - trans[a])
        world_b = np.einsum('nji,nj->ni', R[b], xb - trans[b])
        return (np.einsum('nij,nj->ni', R[b], world_a) + trans[b],
                np.einsum('nij,nj->ni', R[a], world_b) + trans[a])

    def residual(x):
        qb, qa = transformed(x)
        def project(q):
            return q[:, :2] / np.maximum(q[:, 2:3], .01) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        return np.c_[(project(qb) - ub) / .75,
                     (project(qa) - ua) / .75,
                     (qb - xb) / sigma[:, None]].ravel()

    sparsity = lil_matrix((len(refs) * 7, len(x0)), dtype=int)
    for j, (_, ca, _, cb, _) in enumerate(refs):
        for c in (ca, cb):
            if c:
                sparsity[j * 7:(j + 1) * 7, (c - 1) * 6:c * 6] = 1
    return x0, residual, unpack, transformed, sparsity.tocsr(), refs


def refine(features, tracks, initial, K):
    x0, residual, unpack, transformed, sparsity, refs = fit_problem(features, tracks, initial, K)
    start = time.time()
    sol = least_squares(residual, x0, jac_sparsity=sparsity, loss='soft_l1',
                        f_scale=1., x_scale='jac', max_nfev=240,
                        ftol=1e-5, xtol=1e-6, gtol=1e-5,
                        tr_solver='lsmr', tr_options={'maxiter': 400})
    qb, qa = transformed(sol.x)
    return unpack(sol.x), dict(success=bool(sol.success), status=int(sol.status),
        message=sol.message, nfev=sol.nfev, finite=bool(np.isfinite(sol.x).all()),
        all_fit_points_in_front=bool((qa[:, 2] > .01).all() and (qb[:, 2] > .01).all()),
        fitting_pair_refs=refs.tolist(), fitting_tracks=len(np.unique(refs[:, 0])),
        fitting_pairs=len(refs), fitted_rms_before=float(np.sqrt(np.mean(residual(x0) ** 2))),
        fitted_rms_after=float(np.sqrt(np.mean(residual(sol.x) ** 2))),
        elapsed_seconds=time.time() - start)


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    prior = m.read(PARENT / 'protocol.json')
    dependencies = [Path(__file__), Path(t.__file__), Path(p.__file__), Path(r.__file__),
                    Path(m.__file__), r.ROOT / 'scripts/db3_transition_revisit.py']
    artifacts = [f for root in (PARENT, INPUTS) for f in root.rglob('*') if f.is_file()]
    artifacts.append(m.INPUT / 'calibration_clean.json')
    m.write(out / 'protocol.json', dict(date='2026-11-11', timezone='Asia/Saigon',
        cases=prior['cases'], gates=prior['gates'],
        hypothesis='Fitted landmarks can absorb image/depth disagreement; direct measured-depth projection fitting may improve unchanged validation alignment.',
        method='One camera-only arm; fixed first camera/intrinsics/depth. Same up-to-800 tracked fitting tracks. Both directional pixel residuals /0.75px and 3D separation /sqrt(sigma_a^2+sigma_b^2), floor .01m. Pair gaps 1,2,3,5 plus frozen endpoint/half pairs.',
        solver='soft_l1 scale1, sparse numerical Jacobian, x_scale jac, max_nfev240, LSMR maxiter400; ftol1e-5 xtol1e-6 gtol1e-5. No hyperparameter sweep.',
        initialization='Previously saved tracked-fit-only initial poses, no previous optimized or validation-fitted camera initialization.',
        exclusions='Retain all original tiles, 12px moving reservations, whole-track exclusions and native source reservations. Check every loaded fitting observation before fitting.',
        evaluation='Same frozen moving and original checks, no deletion. Support/noninferiority/gain reported separately. Exposed adaptive development evidence, not external accuracy.',
        advancement='Only a local support pass permits further regional validation; short-window success cannot authorize full-scene training or R5 final acceptance.',
        stop='One fixed arm on target and matte control. Failure retains models and blocks final R5; no training on failed geometry.',
        input_hashes={str(f.relative_to(r.ROOT)): m.sha(f) for f in artifacts},
        source_hashes={str(f.relative_to(r.ROOT)): m.sha(f) for f in dependencies}))
    (out / 'source_snapshot').mkdir()
    for f in dependencies:
        shutil.copyfile(f, out / 'source_snapshot' / f.name)
    print('DIRECT DEPTH PROTOCOL FROZEN', flush=True)


def run(out):
    protocol = m.read(out / 'protocol.json')
    for group in ('input_hashes', 'source_hashes'):
        for path, digest in protocol[group].items():
            assert m.sha(r.ROOT / path) == digest, path
    K = np.array(m.read(m.INPUT / 'calibration_clean.json')['K'])
    results = []
    for case in protocol['cases']:
        label = case['label']
        prior, inputs, dest = PARENT / label, INPUTS / label, out / label
        dest.mkdir(exist_ok=False)
        rows = m.read(inputs / 'inputs.json')
        assert [x['source_id'] for x in rows] == case['source_ids']
        original = []
        for row in rows:
            with np.load(inputs / 'inputs' / f'{row["source_id"]}.npz') as data:
                original.append({k: data[k].copy() for k in data.files})
        with np.load(prior / 'fit_features.npz') as data:
            features = [dict(uv=data[f'uv_{c}'].copy(), depth=f['depth'],
                             check=np.zeros(len(data[f'uv_{c}']), bool)) for c, f in enumerate(original)]
        with np.load(inputs / 'moving_checks.npz') as data:
            uv = data['uv'].copy()
            for c, f in enumerate(features):
                assert not m.check_tile(f['uv']).any()
                reserved = data[f'reserved_{c}']
                assert not len(reserved) or (cKDTree(reserved).query(f['uv'])[0] > 12).all()
        tracks = m.read(prior / 'tracks.json')['tracks']
        for tr in tracks:
            assert len(dict(tr)) == len(tr)
        checkfs = [dict(uv=xy, depth=f['depth']) for xy, f in zip(uv, original)]
        for fs in (features, original, checkfs):
            r.observations(fs, K)
        initial = np.array(m.read(prior / 'initial_poses.json'))
        # Rebuild the fit-only initializers as an additional exclusion check.
        rebuilt, _ = t.initializer(features, tracks)
        assert rebuilt is not None and np.allclose(rebuilt, initial, atol=1e-10, rtol=0)
        checks = m.read(inputs / 'frozen_checks.json')
        original_checks = m.read(inputs / 'original_checks.json')
        oldposes = np.array(m.read(inputs / 'old_chain_poses.json'))
        print('DIRECT DEPTH START', label, len(tracks), 'fitting tracks', flush=True)
        poses, fit = refine(features, tracks, initial, K)
        m.write(dest / 'poses.json', poses.tolist())
        m.write(dest / 'optimization.json', fit)
        measured = r.evaluate(checkfs, checks, poses, K)
        orig = r.evaluate(original, original_checks, poses, K)
        old = r.evaluate(checkfs, checks, oldposes, K)
        oldorig = r.evaluate(original, original_checks, oldposes, K)
        noninferior = all(a['separation_m'] is not None and b['separation_m'] is not None and
            all(y <= x + .01 for x, y in zip(a['separation_m'], b['separation_m']))
            for a, b in list(zip(old, measured)) + list(zip(oldorig, orig)))
        gain = bool(old[0]['separation_m'] and measured[0]['separation_m'] and
            any(x - y >= .01 for x, y in zip(old[0]['separation_m'], measured[0]['separation_m'])))
        support = bool(r.support_pass(measured) and fit['success'] and fit['finite'] and fit['all_fit_points_in_front'])
        origok = all(x['matches'] < 20 or (x['separation_m'][0] < .05 and x['separation_m'][1] < .12) for x in orig)
        result = dict(label=label, complete=True, old_chain_checks=old, checks=measured,
            original_checks=orig, original_old_chain=oldorig, local_consistency_pass=support,
            noninferior=noninferior, meaningful_metric_gain=gain, original_supported_checks_pass=origok,
            pose_improvement_pass=bool(support and noninferior and gain and origok),
            training_authorized=False, whole_scene_accepted=False)
        m.write(dest / 'result.json', result)
        results.append(result)
        print('DIRECT DEPTH RESULT', label, 'support', support, 'noninferior', noninferior,
              'gain', gain, 'endpoint px', measured[0]['symmetric_reprojection_px'], flush=True)
    m.write(out / 'result.json', dict(cases=results, training_authorized=False,
        whole_scene_accepted=False, r5_final_acceptance_authorized=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else run)(args.output_dir)
