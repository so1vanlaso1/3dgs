"""Independent residual/provenance checks for the bounded direct-depth arm."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


d = module('db3_direct_depth_refine')
independent = module('audit_db3_moving_holdout')
r, m = d.r, d.m


def support(rows):
    return bool(rows) and all(x['matches'] >= 20 and min(x['hull_fraction']) > .05
        and x['all_in_front'] and x['separation_m'][0] < .05 and x['separation_m'][1] < .12
        and x['symmetric_reprojection_px'][0] < 2 and x['symmetric_reprojection_px'][1] < 4 for x in rows)


def audit(run, out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = m.read(run / 'protocol.json')
    for group in ('input_hashes', 'source_hashes'):
        for path, digest in protocol[group].items():
            assert m.sha(r.ROOT / path) == digest, path
            if group == 'source_hashes':
                assert m.sha(run / 'source_snapshot' / Path(path).name) == digest
    K = np.array(m.read(m.INPUT / 'calibration_clean.json')['K'])
    native = {f['rgb_message_id']: f for f in m.read(m.R1 / 'prospective_checks.json')['source_frames']}
    excluded_names = set(m.read(m.INPUT / 'reconstruction_split.json')['historical_eval'])
    historical = {f['message_id'] for f in m.read(m.INPUT / 'capture_manifest.json')['frames'] if f['name'] in excluded_names}
    summaries, observation_count, pair_count = [], 0, 0
    for case in protocol['cases']:
        label = case['label']
        inputs, prior, dest = d.INPUTS / label, d.PARENT / label, run / label
        rows = m.read(inputs / 'inputs.json')
        assert [x['source_id'] for x in rows] == case['source_ids']
        assert all(r.route.eligible(dict(message_id=x['source_id']), native, historical) for x in rows)
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
                observation_count += len(f['uv'])
        checkfs = [dict(uv=xy, depth=f['depth']) for xy, f in zip(uv, original)]
        for fs in (original, features, checkfs):
            r.observations(fs, K)
        tracks = m.read(prior / 'tracks.json')['tracks']
        poses = np.array(m.read(dest / 'poses.json'))
        assert np.isfinite(poses).all() and np.array_equal(poses[0], np.eye(4))
        for T in poses:
            assert np.allclose(T[3], [0, 0, 0, 1])
            assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-10)
            assert abs(np.linalg.det(T[:3, :3]) - 1) < 1e-10
        fit = m.read(dest / 'optimization.json')
        initial = np.array(m.read(prior / 'initial_poses.json'))
        rebuilt, _ = d.t.initializer(features, tracks)
        assert np.allclose(rebuilt, initial, atol=1e-10, rtol=0)
        _, residual, _, _, _, refs = d.fit_problem(features, tracks, initial, K)
        assert np.array_equal(refs, fit['fitting_pair_refs'])
        for tid, a, ia, b, ib in refs:
            assert dict(tracks[tid])[a] == ia and dict(tracks[tid])[b] == ib
        x = np.c_[Rotation.from_matrix(poses[1:, :3, :3]).as_rotvec(), poses[1:, :3, 3]].ravel()
        assert np.isclose(np.sqrt(np.mean(residual(x) ** 2)), fit['fitted_rms_after'], atol=1e-10)
        pair_count += len(refs)
        result = m.read(dest / 'result.json')
        check_groups = [(checkfs, m.read(inputs / 'frozen_checks.json'), result['checks']),
                        (original, m.read(inputs / 'original_checks.json'), result['original_checks'])]
        for fs, checks, saved in check_groups:
            assert len(checks) == len(saved)
            independent.independent_errors(fs, checks, poses, K, saved)
            for reference, measured in zip(checks, saved):
                assert measured['matches'] == len(reference['refs'])
                assert measured['track_ids'] == [x[0] for x in reference['refs']]
                for key, errors in [('separation_m', 'errors_m'), ('symmetric_reprojection_px', 'errors_px')]:
                    assert np.allclose(measured[key], np.percentile(measured[errors], [50, 90]))
        consistent = support(result['checks']) and fit['success'] and fit['finite'] and fit['all_fit_points_in_front']
        assert consistent == result['local_consistency_pass']
        noninferior = all(all(new <= old + .01 for old, new in zip(a['separation_m'], b['separation_m']))
            for a, b in list(zip(result['old_chain_checks'], result['checks'])) +
                        list(zip(result['original_old_chain'], result['original_checks'])))
        assert noninferior == result['noninferior']
        assert not result['training_authorized'] and not result['whole_scene_accepted']
        summaries.append(dict(label=label, fitting_tracks=fit['fitting_tracks'], fitting_pairs=len(refs),
            solver_converged=fit['success'], evaluations=fit['nfev'], elapsed_seconds=fit['elapsed_seconds'],
            local_consistency_pass=consistent, noninferior=noninferior,
            meaningful_metric_gain=result['meaningful_metric_gain'],
            endpoint_checks=result['checks'][0]['matches'],
            endpoint_median_p90_m=result['checks'][0]['separation_m'],
            endpoint_median_p90_px=result['checks'][0]['symmetric_reprojection_px']))
    delivery = m.read(r.R4 / 'delivery/manifest.json')
    for label in ('whole_route', 'matte'):
        assert m.sha(r.ROOT / delivery[label]['checkpoint']) == delivery[label]['checkpoint_sha256']
    hashes = {str(f.relative_to(run)): m.sha(f) for f in run.rglob('*') if f.is_file()}
    m.write(out / 'artifact_hashes.json', hashes)
    m.write(out / 'audit.json', dict(passed=True, input_hashes_checked=len(protocol['input_hashes']),
        source_hashes_checked=len(protocol['source_hashes']), fitting_observations_checked=observation_count,
        fitting_pairs_checked=pair_count, independent_validation_residuals_reproduced=True,
        all_check_identities_retained=True, initializers_rebuilt=True, original_models_unchanged=True,
        cases=summaries, sealed_artifacts=len(hashes), physical_accuracy_established=False))
    print('DIRECT DEPTH AUDIT PASS', summaries, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    audit(args.run_dir, args.output_dir)
