"""Bounded, preregistered persistent-track recovery; never modifies prior runs.

Fit and check tracks have disjoint feature observations and no mixed-role track
component is used. Camera/landmark BA uses fitting tracks only. Checks never
select residual inliers. This is exposed local consistency, not physical truth.
"""
import argparse
from collections import Counter
import importlib.util
from pathlib import Path
import shutil
import sqlite3
import time

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('route', ROOT/'scripts/db3_r4_route.py')
route = importlib.util.module_from_spec(spec)
spec.loader.exec_module(route)
m, t = route.m, route.t
R4 = ROOT/'results/db3_r4_20261111_v2'


def components(links, features, min_length=3):
    """Discard whole ambiguous/mixed components rather than splitting to hide leakage."""
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    for a, b in links:
        a, b = tuple(a), tuple(b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    groups = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    kept = {'fit': [], 'check': []}
    rejected = Counter()
    for nodes in groups.values():
        nodes = sorted(nodes)
        if len({c for c, _ in nodes}) != len(nodes):
            rejected['ambiguous_multiple_features_in_one_camera'] += 1
        elif len(nodes) < min_length:
            rejected['shorter_than_three_views'] += 1
        else:
            roles = {bool(features[c]['check'][i]) for c, i in nodes}
            if len(roles) != 1:
                rejected['mixed_fit_check_component'] += 1
            else:
                kept['check' if roles.pop() else 'fit'].append(nodes)
    for name in kept:
        kept[name].sort(key=lambda track: (-len(track), track))
    return kept, dict(rejected)


def build_tracks(features):
    n = len(features)
    pairs = sorted({(a, b) for a in range(n) for b in range(a+1, n)
                    if b-a in (1, 2, 3, 5) or (a == 0 and b == n-1)})
    links, counts = [], []
    # Match all descriptors first, so mixed-role components are visible and rejected.
    for a, b in pairs:
        matches = m.mutual(features[a]['desc'], features[b]['desc'])
        links.extend([((a, int(i)), (b, int(j))) for i, j in matches])
        counts.append(dict(cameras=[a, b], matches=len(matches)))
    tracks, rejected = components(links, features)
    return tracks, rejected, counts, links


def transform_points(T, xyz):
    return xyz @ T[:3, :3].T + T[:3, 3]


def observations(features, K):
    for f in features:
        d, valid, sigma = m.sample_depth(f['depth'], f['uv'])
        f.update(z=d, valid=valid, sigma=sigma, xyz=m.backproject(f['uv'], d, K))


def optimize(features, tracks, initial, K, max_nfev=60):
    tracks = [tr for tr in tracks if sum(features[c]['valid'][i] for c, i in tr) >= 2][:800]
    if len(tracks) < 30:
        return None, dict(reason='fewer than 30 fitting tracks with two stable depths', tracks=len(tracks))
    cams = np.array([np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]] for T in initial])
    points, ci, pi, uv, ds, sigmas = [], [], [], [], [], []
    for j, tr in enumerate(tracks):
        world = [transform_points(np.linalg.inv(initial[c]), features[c]['xyz'][i])
                 for c, i in tr if features[c]['valid'][i]]
        points.append(np.median(world, axis=0))
        for c, i in tr:
            assert not features[c]['check'][i]
            ci.append(c); pi.append(j); uv.append(features[c]['uv'][i])
            ds.append(features[c]['z'][i] if features[c]['valid'][i] else 0.)
            sigmas.append(features[c]['sigma'][i])
    ci, pi, uv, ds, sigmas = map(np.array, (ci, pi, uv, ds, sigmas))
    nc = len(cams)
    x0 = np.r_[cams[1:].ravel(), np.array(points).ravel()]
    def unpack(x):
        return np.vstack((cams[0], x[:6*(nc-1)].reshape(-1, 6))), x[6*(nc-1):].reshape(-1, 3)
    def residual(x):
        c, p = unpack(x)
        rotations = Rotation.from_rotvec(c[:, :3]).as_matrix()
        q = np.einsum('nij,nj->ni', rotations[ci], p[pi]) + c[ci, 3:]
        projected = q[:, :2]/np.maximum(q[:, 2:3], .01)*[K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        return np.column_stack(((projected-uv)/.75, np.where(ds > 0, (q[:, 2]-ds)/np.maximum(sigmas, .01), 0))).ravel()
    sparsity = lil_matrix((len(ci)*3, len(x0)), dtype=int)
    for j, (c, p) in enumerate(zip(ci, pi)):
        if c:
            sparsity[j*3:j*3+3, (c-1)*6:c*6] = 1
        sparsity[j*3:j*3+3, 6*(nc-1)+p*3:6*(nc-1)+(p+1)*3] = 1
    before = residual(x0)
    start = time.time()
    result = least_squares(residual, x0, jac_sparsity=sparsity.tocsr(), loss='soft_l1',
                           f_scale=1., max_nfev=max_nfev, ftol=1e-5, xtol=1e-6,
                           gtol=1e-5, tr_solver='lsmr', tr_options={'maxiter':80})
    c, points = unpack(result.x)
    poses = np.repeat(np.eye(4)[None], nc, axis=0)
    poses[:, :3, :3] = Rotation.from_rotvec(c[:, :3]).as_matrix()
    poses[:, :3, 3] = c[:, 3:]
    q = np.einsum('nij,nj->ni', poses[ci, :3, :3], points[pi])+poses[ci, :3, 3]
    report = dict(tracks=len(tracks), observations=len(ci), depth_factors=int(sum(ds > 0)),
                  fitting_tracks=tracks, points_world=points.tolist(), camera_index=ci.tolist(),
                  point_index=pi.tolist(), status=int(result.status), success=bool(result.success),
                  message=result.message, nfev=result.nfev, elapsed_seconds=time.time()-start,
                  finite=bool(np.isfinite(result.x).all()), all_fit_points_in_front=bool((q[:, 2] > .01).all()),
                  fitted_rms_before=float(np.sqrt(np.mean(before**2))),
                  fitted_rms_after=float(np.sqrt(np.mean(residual(result.x)**2))))
    return poses, report


def fixed_checks(features, tracks, pairs):
    rows = []
    for a, b in pairs:
        refs = []
        for tid, track in enumerate(tracks):
            obs = dict(track)
            if a in obs and b in obs and features[a]['valid'][obs[a]] and features[b]['valid'][obs[b]]:
                refs.append([tid, obs[a], obs[b]])
        rows.append(dict(cameras=[a, b], refs=refs))
    return rows


def evaluate(features, checks, poses, K):
    rows = []
    for row in checks:
        a, b = row['cameras']
        refs = np.array(row['refs'], dtype=int).reshape(-1, 3)
        ia, ib = refs[:, 1], refs[:, 2]
        T = poses[b] @ np.linalg.inv(poses[a])
        x, y = features[a]['xyz'][ia], features[b]['xyz'][ib]
        pred, reverse = transform_points(T, x), transform_points(np.linalg.inv(T), y)
        error = np.linalg.norm(pred-y, axis=1)
        proj = lambda q: q[:, :2]/np.maximum(q[:, 2:3], .01)*[K[0, 0], K[1, 1]]+[K[0, 2], K[1, 2]]
        pixel = np.maximum(np.linalg.norm(proj(pred)-features[b]['uv'][ib], axis=1),
                           np.linalg.norm(proj(reverse)-features[a]['uv'][ia], axis=1))
        hulls = [float(cv2.contourArea(cv2.convexHull(f['uv'][idx].astype('f4')))/f['depth'].size) if len(idx) >= 3 else 0.
                 for f, idx in ((features[a], ia), (features[b], ib))]
        rows.append(dict(cameras=[a, b], matches=len(refs), track_ids=refs[:, 0].tolist(),
                         separation_m=np.percentile(error, [50, 90]).tolist() if len(error) else None,
                         symmetric_reprojection_px=np.percentile(pixel, [50, 90]).tolist() if len(pixel) else None,
                         errors_m=error.tolist(), errors_px=pixel.tolist(), hull_fraction=hulls,
                         all_in_front=bool((pred[:, 2] > .01).all() and (reverse[:, 2] > .01).all())))
    return rows


def support_pass(rows):
    return bool(rows) and all(r['matches'] >= 20 and min(r['hull_fraction']) > .05 and r['all_in_front']
                             and r['separation_m'][0] < .05 and r['separation_m'][1] < .12
                             and r['symmetric_reprojection_px'][0] < 2 and r['symmetric_reprojection_px'][1] < 4
                             for r in rows)


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    frames = m.read(m.INPUT/'capture_manifest.json')['frames']
    native = m.read(m.R1/'prospective_checks.json')['source_frames']
    reservations = {f['rgb_message_id']: f for f in native}
    hist_names = set(m.read(m.INPUT/'reconstruction_split.json')['historical_eval'])
    historical = {f['message_id'] for f in frames if f['name'] in hist_names}
    cases = []
    for label, boundary in [('count_limited_target', 362), ('passing_control', 204), ('large_tail_diagnostic', 449)]:
        a, b = frames[boundary:boundary+2]
        roster = [f for f in native if a['timestamp_ns'] <= f['timestamp_ns'] <= b['timestamp_ns']]
        assert all(route.eligible(f, reservations, historical) for f in roster)
        cases.append(dict(label=label, boundary=boundary, source_ids=[f['rgb_message_id'] for f in roster],
                          seconds=[roster[0]['replay_seconds'], roster[-1]['replay_seconds']],
                          check_pairs=[[0, len(roster)-1], [0, len(roster)//2], [len(roster)//2, len(roster)-1]]))
    protocol = dict(date='2026-11-11', timezone='Asia/Saigon', cases=cases,
        selection='Target 362 has a complete native chain but only 12 original endpoint checks; control 204 passes original R4; boundary 449 has a large residual tail. Selected from exposed historical results before new track results.',
        track_recipe='Mutual SIFT ratio 0.75 across gaps 1,2,3,5 and full endpoints; union components; discard components with repeated camera or mixed fit/check roles; keep length >=3. No check residual filtering.',
        additional_support='Persistent check tracks connect endpoints through intermediate observations, even if direct endpoint descriptor matching failed. Same sensor and exposed frames: additional correspondence support, not independent physical truth.',
        fitting='At most 800 tracks ordered by length then identity, each with at least two stable depth measurements; first camera and calibration fixed; robust joint pixel+depth camera/landmark BA; at least 30 tracks; max_nfev=60, no tuning on checks.',
        validation='Freeze tracks and first-last/first-middle/middle-last check identities before optimization. Original direct descriptor checks also retained. Compare direct endpoint, native chain and BA on identical new endpoint checks.',
        gate=dict(min_matches_per_pair=20,min_check_hull_fraction=.05,median_m=.05,p90_m=.12,median_px=2.,p90_px=4.,
                  max_worsening_m=.01,meaningful_pose_gain_m=.01,optimizer_must_converge=True),
        advancement='Local support and pose improvement reported separately. Training/whole-scene promotion additionally requires unresolved fixed physical structures/global constraints; no automatic training on local consistency alone.',
        source_hashes={str(p.relative_to(ROOT)):m.sha(p) for p in (m.INPUT/'capture_manifest.json',m.INPUT/'calibration_clean.json',m.INPUT/'reconstruction_split.json',m.R1/'prospective_checks.json',R4/'selected_edges.json',R4/'native_retries.json',R4/'inputs_manifest.json')})
    m.write(out/'protocol.json', protocol)
    (out/'source_snapshot').mkdir()
    for path in (Path(__file__), ROOT/'scripts/db3_r4_route.py', ROOT/'scripts/db3_transition_revisit.py', ROOT/'scripts/db3_local_geometry.py'):
        shutil.copyfile(path, out/'source_snapshot'/path.name)
    m.write(out/'source_hashes.json', {p.name:m.sha(p) for p in (out/'source_snapshot').iterdir()})
    print('PROTOCOL FROZEN', [(c['label'], c['boundary'], len(c['source_ids'])) for c in cases], flush=True)


def execute(out):
    protocol = m.read(out/'protocol.json')
    for name, expected in protocol['source_hashes'].items():
        assert m.sha(ROOT/name) == expected
    assert m.sha(Path(__file__)) == m.read(out/'source_hashes.json')[Path(__file__).name]
    cal = m.read(m.INPUT/'calibration_clean.json'); K = np.array(cal['K'])
    source = Path(m.read(m.INPUT/'capture_manifest.json')['source']).resolve()
    source_stat = source.stat()
    cached = {f['source_id']:f for f in m.read(R4/'inputs_manifest.json')['frames']}
    native = {f['rgb_message_id']:f for f in m.read(m.R1/'prospective_checks.json')['source_frames']}
    cv2.setNumThreads(6); cv2.setRNGSeed(11010)
    maps, _ = m.projection.remap(cal['raw_color'], K, cal['width'], cal['height'])
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=.02)
    results = []
    with sqlite3.connect(source.as_uri()+'?mode=ro', uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        topic = db.execute("select id from topics where name like '%/Depth_0/image/data'").fetchone()[0]
        depth_rows = db.execute('select id,timestamp from messages where topic_id=? order by timestamp', (topic,)).fetchall()
        stamps = np.array([stamp for _, stamp in depth_rows], dtype=np.int64)
        for case in protocol['cases']:
            dest = out/case['label']; dest.mkdir(exist_ok=False)
            (dest/'inputs').mkdir()
            features, provenance = [], []
            for ci, mid in enumerate(case['source_ids']):
                path = R4/'inputs'/f'{mid}.npz'
                stamp, blob = db.execute('select timestamp,data from messages where id=?', (mid,)).fetchone()
                assert stamp == native[mid]['timestamp_ns']
                rgb, rh = m.clean.capture.CDR(blob).image()
                image = cv2.remap(rgb, maps[:,:,0], maps[:,:,1], cv2.INTER_LINEAR)
                if mid in cached:
                    assert m.sha(path) == cached[mid]['sha256']
                    with np.load(path) as data:
                        f = {k:data[k].copy() for k in data.files}
                    did = cached[mid]['depth_source_id']
                    origin = str(path.relative_to(ROOT))
                else:
                    j = int(np.searchsorted(stamps, stamp))
                    j = min([max(0,j-1),min(len(stamps)-1,j)],key=lambda k:abs(int(stamps[k])-stamp))
                    did = depth_rows[j][0]
                    raw, dh = m.clean.adapter.decode_depth(db.execute('select data from messages where id=?',(did,)).fetchone()[0])
                    assert abs(dh-rh['header_seconds']) < .02
                    *_, evidence = m.projection.align_depth(raw,cal['depth_units'],cal['raw_depth'],cal['depth_to_color_R'],cal['depth_to_color_t_m'],K,cal['width'],cal['height'])
                    kp, desc = sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),None)
                    uv = np.array([k.pt for k in kp], dtype='f4').reshape(-1,2)
                    f = dict(uv=uv,desc=desc if desc is not None else np.empty((0,128),'f4'),check=m.check_tile(uv),depth=evidence['depth_m'])
                    origin = 'read-only raw source, frozen R4 feature recipe'
                assert np.array_equal(f['check'], m.check_tile(f['uv']))
                np.savez_compressed(dest/'inputs'/f'{mid}.npz', **f)
                provenance.append(dict(source_id=mid,depth_source_id=did,origin=origin,sha256=m.sha(dest/'inputs'/f'{mid}.npz')))
                if ci in (0,len(case['source_ids'])//2,len(case['source_ids'])-1):
                    Image.fromarray(image).save(dest/f'camera_{ci:02d}.png')
                features.append(f)
            m.write(dest/'inputs.json',provenance)
            observations(features,K)
            tracks,rejected,pair_counts,links = build_tracks(features)
            m.write(dest/'tracks.json', dict(tracks=tracks,rejected=rejected,pair_counts=pair_counts,links=links))
            checks=fixed_checks(features,tracks['check'],case['check_pairs'])
            m.write(dest/'frozen_checks.json', checks)
            chain=[np.eye(4)]; pair_reports=[]
            for a,b in zip(features,features[1:]):
                row,T,_=t.pair(a,b,K);pair_reports.append(row)
                if T is None:break
                chain.append(T@chain[-1])
            m.write(dest/'initial_pair_reports.json', pair_reports)
            if len(chain)!=len(features):
                results.append(dict(label=case['label'],complete=False,reason='cannot initialize every native camera'))
                continue
            direct, Td, _ = t.pair(features[0],features[-1],K)
            m.write(dest/'direct_pair.json',direct)
            chain=np.array(chain)
            baseline=evaluate(features,checks,chain,K)
            if Td is not None:
                direct_poses=chain.copy();direct_poses[-1]=Td
                direct_new=evaluate(features,checks[:1],direct_poses,K)
            else:direct_new=None
            print('TRACKS',case['label'],'fit',len(tracks['fit']),'check',len(tracks['check']),
                  'fixed checks',[len(r['refs']) for r in checks],flush=True)
            poses,ba=optimize(features,tracks['fit'],chain,K)
            m.write(dest/'optimization.json',ba)
            m.write(dest/'initial_poses.json',chain.tolist())
            if poses is None:
                results.append(dict(label=case['label'],complete=False,reason=ba['reason']))
                continue
            m.write(dest/'optimized_poses.json',poses.tolist())
            candidate=evaluate(features,checks,poses,K)
            # Preserve original direct check matches as another fixed evidence set.
            ca,cb=features[0]['check'],features[-1]['check']
            pairs=m.mutual(features[0]['desc'][ca],features[-1]['desc'][cb])
            ia,ib=np.nonzero(ca)[0][pairs[:,0]],np.nonzero(cb)[0][pairs[:,1]]
            valid=features[0]['valid'][ia]&features[-1]['valid'][ib]
            original=[dict(cameras=[0,len(features)-1],refs=[[j,int(a),int(b)] for j,(a,b) in enumerate(zip(ia[valid],ib[valid]))])]
            original_before=evaluate(features,original,chain,K)
            original_after=evaluate(features,original,poses,K)
            comparisons=list(zip(baseline,candidate))+list(zip(original_before,original_after))
            noninferior=all(a['separation_m'] is not None and b['separation_m'] is not None and
                           all(y<=x+.01 for x,y in zip(a['separation_m'],b['separation_m'])) for a,b in comparisons)
            gain=bool(baseline[0]['separation_m'] and candidate[0]['separation_m'] and
                      any(x-y>=.01 for x,y in zip(baseline[0]['separation_m'],candidate[0]['separation_m'])))
            evidence_ok=ba['success'] and ba['finite'] and ba['all_fit_points_in_front']
            endpoint_direct = {(int(a),int(b)) for a,b in zip(ia[valid],ib[valid])}
            additional=sum((r[1],r[2]) not in endpoint_direct for r in checks[0]['refs'])
            summary=dict(label=case['label'],boundary=case['boundary'],complete=True,
                         native_edges_all_pass=all(r['accepted'] for r in pair_reports),
                         fit_tracks=len(tracks['fit']),check_tracks=len(tracks['check']),
                         additional_endpoint_correspondences=additional,
                         direct_on_new_endpoints=direct_new,chain_on_new_checks=baseline,
                         ba_on_new_checks=candidate,original_check_chain=original_before,original_check_ba=original_after,
                         chain_support_pass=support_pass(baseline),ba_support_pass=support_pass(candidate) and evidence_ok,
                         ba_noninferior_on_fixed_checks=noninferior,meaningful_endpoint_pose_gain=gain,
                         bounded_pose_candidate_pass=support_pass(candidate) and evidence_ok and noninferior and gain,
                         physical_structures_accepted=False,whole_scene_accepted=False,training_authorized=False)
            m.write(dest/'result.json',summary)
            results.append(summary)
            print('RESULT',case['label'],'chain support',summary['chain_support_pass'],'BA support',summary['ba_support_pass'],
                  'pose gain',gain,'noninferior',noninferior,'solver',ba['status'],flush=True)
    assert (source.stat().st_size,source.stat().st_mtime_ns)==(source_stat.st_size,source_stat.st_mtime_ns)
    m.write(out/'result.json',dict(cases=results,whole_scene_accepted=False,training_authorized=False,
        caveat='All data exposed; disjoint feature tracks share sensor/calibration. No physical accuracy or whole-route acceptance. Existing models unchanged.'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    (prepare if args.action=='prepare' else execute)(args.output_dir)
