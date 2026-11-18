import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_anchor_patch_recovers_known_small_affine_motion():
    a = load('db3_anchor_patches')
    rng = np.random.default_rng(12)
    first = cv2.GaussianBlur(rng.integers(0, 256, (240, 320), dtype='u1'), (5, 5), 1.)
    transform = cv2.getRotationMatrix2D((160, 120), 3., 1.03)
    transform[:, 2] += [3., -2.]
    second = cv2.warpAffine(first, transform, (320, 240))
    source = np.array([150., 110.]); truth = transform@np.r_[source, 1.]
    q, report = a.correct(first, second, source, truth+[1., -.5])
    assert report['ok'], report
    assert np.linalg.norm(q-truth) < .2


def test_blank_anchor_does_not_fabricate_valid_measurement():
    a = load('db3_anchor_patches')
    im = np.zeros((160, 200), dtype='u1')
    _, report = a.correct(im, im, np.array([90., 80.]), np.array([90., 80.]))
    assert not report['ok']


def test_learned_empty_matches_and_transitive_check_exclusion():
    sys.path.insert(0, str(ROOT/'.tools/db3-matcher'))
    pytest.importorskip('kornia', reason='Optional learned-correspondence experiment dependency')
    a = load('db3_learned_correspondence')
    # (10,10) is in check tile; its source moves outside that tile to (100,10).
    # A near-coincident observation in camera1 links to a third camera and must
    # also be excluded even though that pair's endpoints lie outside old tiles.
    matches = [dict(cameras=[0, 1], uv=[np.array([[10., 10.]]), np.array([[100., 10.]])], confidence=np.ones(1)),
               dict(cameras=[1, 2], uv=[np.array([[101., 10.]]), np.array([[180., 10.]])], confidence=np.ones(1)),
               dict(cameras=[0, 2], uv=[np.empty((0, 2)), np.empty((0, 2))], confidence=np.empty(0))]
    a.split_matches(matches, [np.empty((0, 2)) for _ in range(3)])
    assert matches[0]['check'][0]
    assert not matches[1]['fit'][0]
    assert matches[2]['fit'].dtype == bool
    assert not matches[2]['fit'].size
