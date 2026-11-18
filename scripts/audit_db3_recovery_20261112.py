"""Reproduce new correspondence metrics and verify retained delivery integrity."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

spec = importlib.util.spec_from_file_location('learned', Path(__file__).with_name('db3_learned_correspondence.py'))
l = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l)
m, r, d, ROOT = l.m, l.r, l.d, l.ROOT


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    run = ROOT/'results/db3_learned_correspondence_v2_20261112'
    protocol = m.read(run/'protocol.json')
    for path, digest in protocol['input_hashes'].items():
        assert m.sha(ROOT/path) == digest, path
    K = np.array(m.read(m.INPUT/'calibration_clean.json')['K'])
    counts = []
    for case in protocol['cases']:
        label = case['label']; dest = run/label; parent = d.INPUTS/label
        assert m.sha(dest/'matches.npz') == m.read(dest/'pre_fit_seal.json')['matches_sha256']
        rows = m.read(parent/'inputs.json'); selected = [0, len(rows)//2, len(rows)-1]
        features = []
        for c in selected:
            with np.load(parent/'inputs'/f'{rows[c]["source_id"]}.npz') as z:
                features.append(dict(uv=[], depth=z['depth'].copy()))
        with np.load(parent/'moving_checks.npz') as z:
            reserved = [z[f'reserved_{c}'].copy().tolist() for c in selected]
            old_uv = z['uv'].copy()
        checks = []; fits = []
        with np.load(dest/'matches.npz') as z:
            for j, (a, b) in enumerate(((0, 2), (0, 1), (1, 2))):
                x, y = z[f'uv_{j}_0'], z[f'uv_{j}_1']
                ischeck = m.check_tile(x) | m.check_tile(y)
                assert np.array_equal(ischeck, z[f'check_{j}'])
                assert not (ischeck & z[f'fit_{j}']).any()
                assert (z[f'confidence_{j}'] >= .2).all()
                reserved[a].extend(x[ischeck]); reserved[b].extend(y[ischeck])
                refs = []
                for i, (u, v) in enumerate(zip(x, y)):
                    ia, ib = len(features[a]['uv']), len(features[b]['uv'])
                    features[a]['uv'].append(u); features[b]['uv'].append(v)
                    if ischeck[i]: refs.append([i, ia, ib])
                    if z[f'fit_{j}'][i]: fits.append([(a, ia), (b, ib)])
                checks.append(dict(cameras=[a, b], refs=refs))
        for f in features: f['uv'] = np.array(f['uv']).reshape(-1, 2)
        r.observations(features, K)
        trees = [cKDTree(x) for x in reserved]
        for tr in fits:
            for c, i in tr:
                assert not m.check_tile(features[c]['uv'][i:i+1])[0]
                assert trees[c].query(features[c]['uv'][i])[0] > 12
        for row in checks:
            a, b = row['cameras']
            row['refs'] = [ref for ref in row['refs'] if features[a]['valid'][ref[1]] and features[b]['valid'][ref[2]]]
        assert checks == m.read(dest/'frozen_checks.json')
        chosen = m.read(dest/'fit_tracks.json')
        allowed = {tuple(tuple(x) for x in tr) for tr in fits}
        assert all(tuple(tuple(x) for x in tr) in allowed for tr in chosen)
        poses = np.array(m.read(dest/'poses.json'))
        assert np.isfinite(poses).all() and np.allclose(np.linalg.det(poses[:, :3, :3]), 1)
        result = m.read(dest/'result.json')
        reproduced = r.evaluate(features, checks, poses, K)
        for a, b in zip(reproduced, result['new_checks_refined_camera']):
            for key in ('errors_m', 'errors_px', 'hull_fraction'):
                assert np.allclose(a[key], b[key], atol=1e-10, rtol=0), (label, key)
        assert result['refined_camera_new_support'] == r.support_pass(reproduced)
        assert result['training_authorized'] is False and result['whole_scene_accepted'] is False
        # Independently reproduce ALL retained original moving checks.
        oldfs = []
        for c, row in enumerate(rows):
            with np.load(parent/'inputs'/f'{row["source_id"]}.npz') as z:
                oldfs.append(dict(uv=old_uv[c], depth=z['depth'].copy()))
        r.observations(oldfs, K)
        extended = np.array(m.read(ROOT/'results/db3_direct_depth_20261111'/label/'poses.json'))
        extended[selected] = poses
        old = r.evaluate(oldfs, m.read(parent/'frozen_checks.json'), extended, K)
        for a, b in zip(old, result['original_moving_checks']):
            assert a['track_ids'] == b['track_ids']
            assert np.allclose(a['errors_px'], b['errors_px'], atol=1e-10, rtol=0)
        counts.append(dict(label=label, checks=[len(x['refs']) for x in checks], fit_pairs=len(chosen), original_checks_reproduced=True))
    delivery = ROOT/'results/db3_r4_20261111_v2/delivery'
    manifest = m.read(delivery/'manifest.json')
    for path, digest in m.read(delivery/'artifact_hashes.json').items():
        assert m.sha(delivery/path) == digest, path
    for model in ('whole_route', 'matte'):
        assert m.sha(ROOT/manifest[model]['checkpoint']) == manifest[model]['checkpoint_sha256']
    m.write(out/'audit.json', dict(passed=True, cases=counts, input_hashes_verified=len(protocol['input_hashes']),
        original_delivery_and_models_unchanged=True, scientific_acceptance=False,
        limits='Reconstructs refs and exclusions separately; numerical geometry evaluation shares the tested repository evaluator. No external physical reference. Full raw database hash not rerun.'))
    print('RECOVERY AUDIT PASS', counts, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--output-dir', type=Path, required=True)
    main(parser.parse_args().output_dir)
