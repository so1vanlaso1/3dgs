"""Frozen LoFTR RGB-D correspondence experiment on exposed R4 windows.

New checks live in original excluded tiles in both images. Old failed checks
remain reported. A pretrained matcher supplies observations, not geometry truth.
"""
import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import time
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'.tools/db3-matcher'))
import cv2
import numpy as np
from scipy.spatial import cKDTree
import torch
import kornia
from kornia.feature import LoFTR
from kornia.feature.loftr.loftr import default_cfg

spec = importlib.util.spec_from_file_location('direct', Path(__file__).with_name('db3_direct_depth_refine.py'))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
m, r = d.m, d.r
WEIGHTS = ROOT/'.cache/db3-models/loftr_indoor_ds_new.ckpt'


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    parent = m.read(d.PARENT/'protocol.json')
    paths = [Path(__file__), Path(d.__file__), WEIGHTS, m.INPUT/'calibration_clean.json']
    for folder in (d.INPUTS, d.PARENT, ROOT/'results/db3_direct_depth_20261111'):
        paths.extend(p for p in folder.rglob('*') if p.is_file())
    paths.extend((ROOT/'.tools/db3-matcher/kornia/feature/loftr').rglob('*.py'))
    m.write(out/'protocol.json', dict(date='2026-11-12', cases=parent['cases'],
        hypothesis='Learned coarse-to-fine matching can supply static correspondences where local incremental trackers drift or confuse repeated texture.',
        matcher=dict(kornia=kornia.__version__, weights=str(WEIGHTS.relative_to(ROOT)),
                     weights_sha256=m.sha(WEIGHTS), model='indoor_new', resize=[640, 352],
                     confidence_min=.2, device='cuda', fine_coordinates='Scale x/y back to original dimensions; no half-pixel offset, matching the reference LoFTR convention.'),
        validation='Either endpoint in original excluded check tiles; follow its source identity into the other image and exclude both locations plus whole connected components from fit. Unchanged stable-depth rule; no geometric residual or RANSAC filters checks. Freeze all correspondences before optimization. Retain old moving and descriptor checks with unchanged coordinates.',
        adaptive_change='First .8-confidence/intersection split produced only 1/30/1 stable checks on target; use pretrained model default .2 and source-following union split. Fix empty-match boolean dtype. These changes are disclosed, not untouched confirmation.',
        fitting='New pair matches outside ALL original tiles, >12px from every old moving reservation and every new check point; propagate exclusions through cross-pair near-coincident observations (3px). Fixed first camera/intrinsics/sensor depths. Same direct-depth objective and 240-evaluation cap; fit-only prior initializer.',
        gates=parent['gates'],
        advancement='Report new-check support separately from original-check regression and true pose improvement. Never automatically promote a model or declare whole-room acceptance. This is exposed development with shared sensor/image and learned receptive fields, not metrology.',
        reference='https://www.kornia.org/tutorials/nbs/image_matching.html',
        input_hashes={str(p.relative_to(ROOT)): m.sha(p) for p in paths}))
    (out/'source_snapshot').mkdir()
    shutil.copyfile(__file__, out/'source_snapshot'/Path(__file__).name)
    print('LEARNED CORRESPONDENCE PROTOCOL FROZEN', flush=True)


def split_matches(matches, original_reserved):
    """Whole components touching any old/new check cannot enter camera fitting."""
    nc = len(original_reserved)
    reserved = [list(x) for x in original_reserved]
    for pair in matches:
        a, b = pair['cameras']; x, y = pair['uv']
        pair['check'] = m.check_tile(x) | m.check_tile(y)
        reserved[a].extend(x[pair['check']]); reserved[b].extend(y[pair['check']])
    trees = [cKDTree(np.asarray(x)) if len(x) else None for x in reserved]
    nodes = [[] for _ in range(nc)]; edges = []; blocked = []; parent = []
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[b] = a
    for pair in matches:
        refs = []
        for i in range(len(pair['confidence'])):
            ids = []
            for c, uv in zip(pair['cameras'], pair['uv']):
                q = uv[i]; j = len(parent); parent.append(j); ids.append(j)
                bad = bool(m.check_tile(q[None])[0]) or (trees[c] is not None and trees[c].query(q)[0] <= 12)
                blocked.append(bad); nodes[c].append((j, q)); refs.append(j)
            edges.append(ids); union(*ids)
        pair['nodes'] = np.array(refs).reshape(-1, 2)
    for rows in nodes:
        if not rows: continue
        for i, j in cKDTree(np.array([x[1] for x in rows])).query_pairs(3.):
            union(rows[i][0], rows[j][0])
    badroots = {find(i) for i, b in enumerate(blocked) if b}
    for pair in matches:
        pair['fit'] = np.array([find(int(a)) not in badroots for a, b in pair['nodes']], dtype=bool)
        assert not np.any(pair['fit'] & pair['check'])
    return reserved


def match_pair(model, images, a, b):
    tensors = [torch.from_numpy(cv2.resize(images[c], (640, 352), interpolation=cv2.INTER_AREA)).float()[None, None].cuda()/255 for c in (a, b)]
    with torch.inference_mode():
        result = model(dict(image0=tensors[0], image1=tensors[1]))
    confidence = result['confidence'].cpu().numpy()
    uv = [result[f'keypoints{j}'].cpu().numpy()*[images[c].shape[1]/640, images[c].shape[0]/352] for j, c in enumerate((a, b))]
    keep = confidence >= .2
    return dict(cameras=[a, b], uv=[q[keep] for q in uv], confidence=confidence[keep])


def run(out):
    protocol = m.read(out/'protocol.json')
    for path, expected in protocol['input_hashes'].items():
        assert m.sha(ROOT/path) == expected, path
    cv2.setNumThreads(6); torch.set_num_threads(6); torch.manual_seed(11012)
    cfg = deepcopy(default_cfg); cfg['coarse']['temp_bug_fix'] = True
    model = LoFTR(pretrained=None, config=cfg)
    model.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True)['state_dict'])
    model = model.cuda().eval()
    K = np.array(m.read(m.INPUT/'calibration_clean.json')['K']); results = []
    for case in protocol['cases']:
        started = time.time(); label = case['label']; dest = out/label; dest.mkdir(exist_ok=False)
        parent = d.INPUTS/label; rows = m.read(parent/'inputs.json'); n = len(rows)
        selected = [0, n//2, n-1]
        images = [cv2.imread(str(parent/'inputs'/f'{rows[c]["source_id"]}.png'), 0) for c in selected]
        original = []
        for row in rows:
            with np.load(parent/'inputs'/f'{row["source_id"]}.npz') as z:
                original.append({k: z[k].copy() for k in z.files})
        with np.load(parent/'moving_checks.npz') as z:
            reserved = [z[f'reserved_{c}'].copy() for c in selected]; old_uv = z['uv'].copy()
        matches = [match_pair(model, images, a, b) for a, b in ((0, 2), (0, 1), (1, 2))]
        split_matches(matches, reserved)
        arrays = {}
        for j, pair in enumerate(matches):
            for key in ('confidence', 'check', 'fit', 'nodes'): arrays[f'{key}_{j}'] = pair[key]
            for c, uv in enumerate(pair['uv']): arrays[f'uv_{j}_{c}'] = uv
        np.savez_compressed(dest/'matches.npz', **arrays)
        m.write(dest/'pre_fit_seal.json', dict(matches_sha256=m.sha(dest/'matches.npz'), selected_native_cameras=selected))
        fs = [dict(uv=[], depth=original[c]['depth']) for c in selected]; fit_tracks = []; checks = []
        for pair in matches:
            a, b = pair['cameras']; refs = []
            for i, (x, y) in enumerate(zip(*pair['uv'])):
                ia, ib = len(fs[a]['uv']), len(fs[b]['uv'])
                fs[a]['uv'].append(x); fs[b]['uv'].append(y)
                if pair['fit'][i]: fit_tracks.append([(a, ia), (b, ib)])
                if pair['check'][i]: refs.append([i, ia, ib])
            checks.append(dict(cameras=[a, b], refs=refs))
        for f in fs:
            f['uv'] = np.array(f['uv']).reshape(-1, 2); f['check'] = m.check_tile(f['uv'])
        r.observations(fs, K)
        for row in checks:
            a, b = row['cameras']
            row['refs'] = [ref for ref in row['refs'] if fs[a]['valid'][ref[1]] and fs[b]['valid'][ref[2]]]
        m.write(dest/'frozen_checks.json', checks)
        # Fix cap evenly across the three pairs, not just the first pair.
        stable = [tr for tr in fit_tracks if all(fs[c]['valid'][i] for c, i in tr)]
        groups = [[tr for tr in stable if [tr[0][0], tr[1][0]] == pair['cameras']] for pair in matches]
        chosen = [group[i] for i in range(max(map(len, groups), default=0)) for group in groups if i < len(group)][:800]
        m.write(dest/'fit_tracks.json', chosen)
        initial_all = np.array(m.read(d.PARENT/label/'initial_poses.json'))
        before_all = np.array(m.read(ROOT/'results/db3_direct_depth_20261111'/label/'poses.json'))
        before = before_all[selected]
        old_measure = r.evaluate(fs, checks, before, K)
        print('LEARNED', label, 'confident', [len(p['confidence']) for p in matches], 'check', [len(p['refs']) for p in checks], 'fit', len(chosen), flush=True)
        if len(chosen) < 30:
            row = dict(label=label, complete=False, reason='insufficient fit', new_checks_existing_camera=old_measure)
        else:
            poses, fit = d.refine(fs, chosen, initial_all[selected], K)
            m.write(dest/'poses.json', poses.tolist()); m.write(dest/'optimization.json', fit)
            measured = r.evaluate(fs, checks, poses, K)
            oldfs = [dict(uv=xy, depth=f['depth']) for xy, f in zip(old_uv, original)]
            r.observations(oldfs, K); r.observations(original, K)
            extended = before_all.copy(); extended[selected] = poses
            oldchecks = r.evaluate(oldfs, m.read(parent/'frozen_checks.json'), extended, K)
            origchecks = r.evaluate(original, m.read(parent/'original_checks.json'), extended, K)
            row = dict(label=label, complete=True, new_checks_existing_camera=old_measure,
                       new_checks_refined_camera=measured, original_moving_checks=oldchecks,
                       original_descriptor_checks=origchecks,
                       existing_camera_new_support=r.support_pass(old_measure),
                       refined_camera_new_support=r.support_pass(measured),
                       original_moving_support=r.support_pass(oldchecks),
                       elapsed_seconds=time.time()-started, training_authorized=False,
                       whole_scene_accepted=False)
            print('LEARNED RESULT', label, 'existing/refined/old support', row['existing_camera_new_support'], row['refined_camera_new_support'], row['original_moving_support'],
                  'endpoint', measured[0]['symmetric_reprojection_px'], flush=True)
        m.write(dest/'result.json', row); results.append(row)
    m.write(out/'result.json', dict(cases=results, training_authorized=False, whole_scene_accepted=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); (prepare if args.action == 'prepare' else run)(args.output_dir)
