"""Independent pinned SDK oracle plus crop/occlusion/version regression controls."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('projection', ROOT/'scripts/realsense_projection.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def test_zbuffer_keeps_nearest_and_missing():
    uv, z, e = p.zbuffer([[2, 3], [2.1, 3.1], [5, 4], [-1, 3], [6, 4]],
                         [2, 1, 3, 4, float('nan')], 8, 6)
    assert e['depth_m'][3, 2] == 1 and e['depth_m'][4, 5] == 3
    assert e['discarded_collisions'] == 1 and e['multiple_projected_samples'][3, 2]
    assert e['missing'].sum() == 46 and e['observed'].sum() == 2
    assert len(z) == 2


def test_border_gate_precedes_float32_quantization():
    # This real-data failure at frame 49 changed RNG sampling of all later frames.
    # The value is valid before quantization and becomes exactly width-1 in f32.
    _, z, _ = p.zbuffer([[6.99999999, 3.]], [1.], 8, 6, margin=1)
    assert len(z) == 1
    _, z, e = p.zbuffer([[7.49999999, 3.]], [1.], 8, 6)
    assert len(z) == 0 and not e['observed'].any()


def test_cleaner_refuses_silent_legacy_reuse(tmp_path):
    from argparse import Namespace
    source=tmp_path/'source.db3';source.write_bytes(b'identity fixture')
    args=Namespace(output=tmp_path,recording=source,fps=3.,width=848,min_sharpness=25.,
                   max_delta=.02,near=.3,far=6.,all_frames=False,projection_version=p.SDK)
    old=dict(source=str(source.resolve()),source_bytes=source.stat().st_size,
             source_mtime_ns=source.stat().st_mtime_ns,
             parameters={k:v for k,v in vars(args).items() if k not in ('output','recording','projection_version')})
    (tmp_path/'capture_manifest.json').write_text(json.dumps(old))
    sp=importlib.util.spec_from_file_location('cleaner_reuse',ROOT/'scripts/clean-realsense-scene.py')
    cleaner=importlib.util.module_from_spec(sp);sp.loader.exec_module(cleaner)
    with pytest.raises(ValueError,match='different source/settings'):
        cleaner.clean(args)
    args.projection_version=p.LEGACY
    cleaner.clean(args)
    (tmp_path/'calibration_clean.json').write_text(json.dumps(dict(projection_version=p.SDK)))
    with pytest.raises(ValueError,match='versions disagree'):
        cleaner.check_source(tmp_path)


def test_versions_fail_closed_and_old_calibration_is_legacy():
    assert p.calibration_mode({}) == p.LEGACY
    with pytest.raises(ValueError):
        p.calibration_mode({'projection_version': 'future_unknown'})
    with pytest.raises(ValueError):
        p.project([[0, 0, -1]], {}, p.SDK)


def test_crop_and_scale_represent_same_ray():
    info = dict(width='1280', height='720', fx='650', fy='650', ppx='640', ppy='360',
                model='None', coeffs='0,0,0,0,0')
    K = p.intrinsic(info); K[:2] *= .6625; K[:2, 2] -= [7, 7]
    maps, valid = p.remap(info, K, 834, 463)
    np.testing.assert_allclose(maps[200, 300], [(300+7)/.6625, (200+7)/.6625], atol=1e-4)
    assert valid.all()


def test_pinned_sdk_projection_and_deprojection():
    sys.path.insert(0, str(ROOT/'.cache/room_sdk_audit/install'))
    rs = pytest.importorskip('pyrealsense2')
    assert rs.__version__ == '2.59.1'
    cal = json.loads((ROOT/'data/realsense_20261030_134523_clean/calibration_clean.json').read_text())
    for key in ('raw_color', 'raw_depth'):
        info = cal[key]
        for model in ('None', 'Brown Conrady', 'Inverse Brown Conrady'):
            info = dict(info, model=model)
            intr = rs.intrinsics()
            for attr in ('width', 'height'):
                setattr(intr, attr, int(info[attr]))
            for attr in ('fx', 'fy', 'ppx', 'ppy'):
                setattr(intr, attr, float(info[attr]))
            intr.model = {0: rs.distortion.none, 2: rs.distortion.inverse_brown_conrady,
                          4: rs.distortion.brown_conrady}[p.MODEL_IDS[model]]
            intr.coeffs = p.coefficients(info).tolist()
            pixels = np.array([(x, y) for x in np.linspace(0, intr.width-1, 17)
                                for y in np.linspace(0, intr.height-1, 11)])
            depths = np.resize([.3, .7, 2., 5.9], len(pixels))
            oracle = np.array([rs.rs2_deproject_pixel_to_point(intr, uv.tolist(), float(z))
                               for uv, z in zip(pixels, depths)])
            local = p.deproject(pixels, depths, info)
            assert np.max(np.linalg.norm(local-oracle, axis=1)) < 5e-6
            oracle_uv = np.array([rs.rs2_project_point_to_pixel(intr, xyz.tolist()) for xyz in local])
            assert np.max(np.linalg.norm(p.project(local, info)-oracle_uv, axis=1)) < .001
            if model != 'None' or not np.any(p.coefficients(info)):
                assert np.max(np.linalg.norm(p.project(local, info)-pixels, axis=1)) < .001
