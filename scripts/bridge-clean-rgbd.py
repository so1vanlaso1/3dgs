"""Try original higher-rate cleaned RGB-D frames at disconnected pose boundaries."""
import argparse
import importlib.util
import json
from pathlib import Path
import sqlite3
import time

import cv2
import numpy as np
import open3d as o3d
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation

spec = importlib.util.spec_from_file_location('clean', Path(__file__).with_name('clean-realsense-scene.py'))
clean = importlib.util.module_from_spec(spec); spec.loader.exec_module(clean)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('data/realsense_20261030_111354_clean'))
    a = ap.parse_args(); root = a.root.resolve()
    m = clean.check_source(root); frames = m['frames']; out = root/'rgbd'
    edges = json.loads((out/'edges.json').read_text())
    if not (out/'edges_before_bridge.json').exists():
        clean.write(out/'edges_before_bridge.json', edges)
        clean.write(out/'report_before_bridge.json', json.loads((out/'report.json').read_text()))
    g = np.zeros((len(frames), len(frames)))
    for e in edges:
        g[e['i'], e['j']] = g[e['j'], e['i']] = 1
    _, labels = connected_components(csr_matrix(g), directed=False)
    boundaries = [i for i in range(len(frames)-1) if labels[i] != labels[i+1]]
    if not boundaries:
        clean.write(out/'bridge_report.json',dict(boundaries=0,accepted=0,results=[],
                    pose_graph_reoptimized=False,reason='Every qualified frame already belongs to one connected optimized graph'))
        print('BRIDGE_NOT_NEEDED: all frames already connected; preserved verified optimized graph',flush=True)
        return
    cal = json.loads((root/'calibration_clean.json').read_text())
    c, d = cal['raw_color'], cal['raw_depth']
    K = np.array(cal['K']); w, h = round(cal['width']/4), round(cal['height']/4)
    K[0] *= w/cal['width']; K[1] *= h/cal['height']
    R, t = np.array(cal['depth_to_color_R']), np.array(cal['depth_to_color_t_m'])
    mode = clean.projection.calibration_mode(cal)
    if m.get('parameters', {}).get('projection_version', mode) != mode:
        raise ValueError('Capture/calibration projection versions disagree')
    maps, validmap = clean.projection.remap(c, K, w, h, mode)
    if not validmap.all():
        raise ValueError('Bridge RGB map has invalid sampling pixels')
    camera = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    opt = o3d.pipelines.odometry.OdometryOption()
    opt.depth_min=.3; opt.depth_max=6; opt.depth_diff_max=.08
    opt.iteration_number_per_pyramid_level=o3d.utility.IntVector([30, 20, 10])
    jac = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
    def consistency(T, src, dst):
        y, x = np.nonzero(src > 0); z = src[y, x]
        xyz = np.column_stack(((x-K[0, 2])/K[0, 0]*z, (y-K[1, 2])/K[1, 1]*z, z))@T[:3, :3].T+T[:3, 3]
        valid = xyz[:, 2] > .3; xyz = xyz[valid]
        p = np.rint(xyz[:, :2]/xyz[:, 2:3]*[K[0, 0], K[1, 1]]+[K[0, 2], K[1, 2]]).astype(int)
        valid = (p[:, 0] >= 0)&(p[:, 0] < w)&(p[:, 1] >= 0)&(p[:, 1] < h)
        p, xyz = p[valid], xyz[valid]
        measured = dst[p[:, 1], p[:, 0]]
        valid = measured > 0
        errors = np.abs(xyz[valid, 2]-measured[valid])
        return len(errors) >= 200 and np.mean(errors < .08) > .65 and np.median(errors) < .05
    results = []
    with sqlite3.connect(Path(m['source']).as_uri()+'?mode=ro', uri=True) as db:
        topics = {n: i for i, n in db.execute('select id,name from topics')}
        def rows(suffix):
            tid = next(i for n, i in topics.items() if n.endswith(suffix))
            return db.execute('select id,timestamp from messages where topic_id=? order by timestamp', (tid,)).fetchall()
        rgbrows, drows = rows('/Color_0/image/data'), rows('/Depth_0/image/data')
        dstamps = np.array([s for _, s in drows], dtype=np.int64)
        cache = {}
        def load(row):
            mid, stamp = row
            if mid in cache:
                return cache[mid]
            rgb, rh = clean.capture.CDR(db.execute('select data from messages where id=?', (mid,)).fetchone()[0]).image()
            j = int(np.searchsorted(dstamps, stamp))
            j = min([max(0, j-1), min(len(dstamps)-1, j)], key=lambda k: abs(int(dstamps[k])-stamp))
            if abs(int(dstamps[j])-stamp) > 20_000_000:
                raise ValueError('Bridge timestamp gap')
            dep, dh = clean.adapter.decode_depth(db.execute('select data from messages where id=?', (drows[j][0],)).fetchone()[0])
            if abs(dh-rh['header_seconds']) > .02:
                raise ValueError('Bridge header gap')
            p, z, _ = clean.clean_depth(dep, cal['depth_units'], d, R, t, K, w, h)
            image = cv2.remap(rgb, maps[:, :, 0], maps[:, :, 1], cv2.INTER_LINEAR)
            depth = np.zeros((h, w), dtype='f4')
            xy = np.rint(p).astype(int); depth[xy[:, 1], xy[:, 0]] = z
            pair = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(image), o3d.geometry.Image(depth), depth_scale=1, depth_trunc=6, convert_rgb_to_intensity=True)
            cache[mid] = pair, depth
            return pair, depth
        for i in boundaries:
            start, end = frames[i]['timestamp_ns'], frames[i+1]['timestamp_ns']
            interior = [r for r in rgbrows if start < r[1] < end]
            selected = [(frames[i]['message_id'], start)]+interior+[(frames[i+1]['message_id'], end)]
            total = np.eye(4); ok_all = True; successful = 0; reason = None
            try:
                source, adepth = load(selected[0])
                for row in selected[1:]:
                    target, bdepth = load(row)
                    ok, T, _ = o3d.pipelines.odometry.compute_rgbd_odometry(source, target, camera, np.eye(4), jac, opt)
                    good = ok and np.linalg.norm(T[:3, 3]) < .3 and Rotation.from_matrix(T[:3, :3]).magnitude() < .35 and consistency(T, adepth, bdepth)
                    if good:
                        ok, back, _ = o3d.pipelines.odometry.compute_rgbd_odometry(target, source, camera, np.linalg.inv(T), jac, opt)
                        delta = back@T
                        good = ok and np.linalg.norm(delta[:3, 3]) < .03 and Rotation.from_matrix(delta[:3, :3]).magnitude() < .035 and consistency(back, bdepth, adepth)
                    if not good:
                        ok_all = False; reason = 'bidirectional_or_depth_consistency_gate'; break
                    total = T@total; successful += 1
                    source, adepth = target, bdepth
            except ValueError as exc:
                ok_all = False; reason = str(exc)
            if ok_all:
                edges.append(dict(i=i, j=i+1, T=total.tolist(), inliers=100, error=.03, method='high_rate_cleaned_rgbd_bridge'))
            results.append(dict(i=i, j=i+1, accepted=bool(ok_all), transitions=successful, sampled_frames=len(selected), reason=reason))
            print('BRIDGE', results[-1], flush=True)
    if (out/'bridge_report.json').exists():
        clean.write(out/f'bridge_report_previous_{int(time.time())}.json', json.loads((out/'bridge_report.json').read_text()))
    clean.write(out/'bridge_report.json', dict(boundaries=len(boundaries), accepted=sum(r['accepted'] for r in results), rgb_sampling='Every recorded RGB frame between selected endpoints',
                filtering='Same range/discontinuity/median/rectification/alignment as cleaned capture, no depth filling', results=results))
    clean.write(out/'edges.json', edges)
    recovery = clean.module('recover-rgbd-scene.py')
    recovery.optimize(root)


if __name__ == '__main__':
    main()
