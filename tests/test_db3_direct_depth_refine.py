import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

spec = importlib.util.spec_from_file_location('direct', Path(__file__).resolve().parents[1] / 'scripts/db3_direct_depth_refine.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def fixture():
    rng = np.random.default_rng(11)
    points = rng.uniform([-.6, -.4, 1.2], [.6, .4, 3.], (70, 3))
    K = np.array([[500., 0, 320.], [0, 500., 240.], [0, 0, 1.]])
    truth = np.repeat(np.eye(4)[None], 4, axis=0)
    features = []
    for c in range(4):
        truth[c, :3, :3] = Rotation.from_rotvec(np.array([.007, -.009, .005]) * c).as_matrix()
        truth[c, :3, 3] = np.array([.015, -.006, .002]) * c
        q = points @ truth[c, :3, :3].T + truth[c, :3, 3]
        uv = q[:, :2] / q[:, 2:] * 500 + [320, 240]
        features.append(dict(xyz=q, uv=uv, sigma=np.full(70, .02), valid=np.ones(70, bool), check=np.zeros(70, bool)))
    tracks = [[(c, i) for c in range(4)] for i in range(70)]
    initial = truth.copy()
    initial[1:, :3, 3] += [.01, -.01, .005]
    return features, tracks, initial, K, truth


def test_recovers_metric_camera_motion_without_fitted_landmarks():
    fs, tracks, initial, K, truth = fixture()
    poses, report = d.refine(fs, tracks, initial, K)
    assert report['success'] and report['all_fit_points_in_front']
    assert np.allclose(poses, truth, atol=2e-5)
    assert np.array_equal(poses[0], truth[0])
    assert report['fitted_rms_after'] < 1e-4


def test_reserved_observations_and_missing_support_fail_closed():
    fs, tracks, initial, K, _ = fixture()
    fs[1]['check'][0] = True
    with pytest.raises(AssertionError):
        d.fit_problem(fs, tracks, initial, K)
    fs[1]['check'][0] = False
    with pytest.raises(ValueError, match='30 stable-depth'):
        d.fit_problem(fs, tracks[:29], initial, K)
