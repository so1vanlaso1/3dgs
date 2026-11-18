"""Calibration, cleaning, synchronization boundaries and CDR IMU regression tests."""
import importlib.util
from pathlib import Path
import sqlite3
import struct

import numpy as np


def load():
    path = Path(__file__).parents[1]/'scripts/clean-realsense-scene.py'
    spec = importlib.util.spec_from_file_location('clean_realsense', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_inverse_distortion_full_image_roundtrip():
    m = load()
    xy = np.random.default_rng(19).uniform([-1.05, -.6], [1.05, .6], (10000, 2))
    coeff = np.array([-.0559454858, .0645855516, -.000252637488, .000818350061, -.021141313])
    np.testing.assert_allclose(m.brown(m.invert_brown(xy, coeff), coeff), xy, atol=1e-7)


def test_depth_mask_denoise_and_metric_extrinsic():
    m = load()
    dep = np.full((30, 40), 1000, dtype='f4')
    dep[8, 8] = 0
    dep[20, 20] = 8000
    dep[15, 15] = 1008  # locally continuous noise removed by median
    d = dict(fx='30', fy='30', ppx='20', ppy='15')
    K = m.intrinsic(d)
    pixels, z, fraction = m.clean_depth(dep, .001, d, np.eye(3), np.array([0., 0., .1]), K, 40, 30)
    assert .8 < fraction < 1.
    np.testing.assert_allclose(z, 1.1, atol=1e-6)
    keys = np.rint(pixels[:, 1]).astype(int)*40+np.rint(pixels[:, 0]).astype(int)
    assert len(np.unique(keys)) == len(keys)
    # No filling around invalid/far samples; source invalid neighborhoods stay absent.
    source = (pixels-[20, 15])*z[:, None]/30
    source_pixels = source*30/(z[:, None]-.1)+[20, 15]
    assert not np.any(np.linalg.norm(source_pixels-[8, 8], axis=1) < 1.1)


def test_reference_extrinsics_compose_not_assume_identity():
    m = load()
    rd = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rc = np.eye(3)
    td, tc = np.array([.1, .2, .3]), np.array([-.1, 0., .02])
    def encode(r, t):
        return 'rotation='+','.join(map(str, r.ravel(order='F')))+';translation='+','.join(map(str, t))
    r, t = m.relative_extrinsics(encode(rd, td), encode(rc, tc))
    x = np.array([.2, .3, 2.])
    np.testing.assert_allclose(r@(rd@x+td)+t, rc@x+tc)


def test_exposure_and_blur_rejection():
    m = load()
    assert 'underexposed' in m.rejection(m.rgb_quality(np.zeros((72, 128, 3), dtype='u1')), 25)
    assert 'overexposed' in m.rejection(m.rgb_quality(np.full((72, 128, 3), 255, dtype='u1')), 25)
    assert 'blur_or_low_texture' in m.rejection(m.rgb_quality(np.full((72, 128, 3), 128, dtype='u1')), 25)


def test_all_frames_keeps_timestamp_bin_collisions_and_order():
    m = load()
    rows = [(10, 0), (11, 32_000_000), (12, 33_000_000), (13, 66_000_000)]
    assert len(m.frame_groups(rows, 30)) < len(rows)
    assert [r for group in m.frame_groups(rows, 30, True).values() for r in group] == rows


def test_imu_cdr_layout_and_units(tmp_path):
    m = load()
    db = sqlite3.connect(':memory:')
    db.execute('create table messages(id integer, timestamp integer, data blob, topic_id integer)')
    for j in range(2):
        blob = bytearray(b'\x00\x01\x00\x00')
        def u32(v):
            blob.extend(b'\0'*((-(len(blob)-4))%4))
            blob.extend(struct.pack('<I', v))
        u32(12); u32(j*10000000)
        u32(4); blob.extend(b'imu\0')
        blob.extend(b'\0'*((-(len(blob)-4))%8))
        vals = np.zeros(37)
        vals[13:16] = [1., 2., 3.]
        vals[25:28] = [0., 9.81, 0.]
        blob.extend(struct.pack('<37d', *vals))
        db.execute('insert into messages values(?,?,?,?)', (j, j*10000000, bytes(blob), 1))
    report = m.imu_export(db, [(1, '/Accel_0/imu/data', 'sensor_msgs/msg/Imu')], tmp_path)
    assert report['accel']['count'] == 2
    assert report['accel']['strictly_increasing']
    np.testing.assert_allclose(report['accel']['magnitude_percentiles'], 9.81)
