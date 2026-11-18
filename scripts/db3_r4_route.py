"""R4 full-route expansion screen. Never promotes a pair graph as a global model.

Uses corrected R1 images, fitting-only native retries and disjoint feature tiles.
All outputs are fresh. A failed gate selects the documented D8 partial delivery.
"""
import argparse
from collections import Counter, OrderedDict
import importlib.util
from pathlib import Path
import shutil
import sqlite3
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('transition', ROOT/'scripts/db3_transition_revisit.py')
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
m = t.m


def eligible(frame, reservations, historical):
    mid = frame.get('message_id', frame.get('rgb_message_id'))
    return mid not in historical and reservations[mid]['role'] == 'fit_development'


def expansion_decision(edges, expected_edges, *, structures_pass, global_constraints_pass):
    """Missing/duplicate checks and a connected but inaccurate graph fail closed."""
    ids = [e['boundary'] for e in edges]
    roster_ok = len(ids) == len(set(ids)) and set(ids) == set(expected_edges) and bool(ids)
    overlap_ok = roster_ok and all(e.get('accepted') is True for e in edges)
    passed = bool(overlap_ok and structures_pass and global_constraints_pass)
    return dict(edge_roster_complete=roster_ok, all_checked_overlaps_pass=overlap_ok,
                difficult_structures_pass=bool(structures_pass),
                distributed_global_constraints_pass=bool(global_constraints_pass),
                whole_scene_training_authorized_by_evidence=passed,
                whole_scene_scientifically_accepted=False,
                next_action='bounded_global_reconstruction' if passed else 'D8_partial_delivery')


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    manifest = m.read(m.INPUT/'capture_manifest.json'); frames = manifest['frames']
    native = m.read(m.R1/'prospective_checks.json')['source_frames']
    reservations = {f['rgb_message_id']: f for f in native}
    historical_names = set(m.read(m.INPUT/'reconstruction_split.json')['historical_eval'])
    historical = {f['message_id'] for f in frames if f['name'] in historical_names}
    original = {f['message_id']: f for f in frames}
    origin = native[0]['timestamp_ns']
    seconds = lambda f: (f['timestamp_ns']-origin)/1e9
    allowed = lambda f: eligible(f, reservations, historical)
    expected = [i for i in range(len(frames)-1) if allowed(frames[i]) and allowed(frames[i+1])
                and all(allowed(f) for f in native if frames[i]['timestamp_ns'] <= f['timestamp_ns'] <= frames[i+1]['timestamp_ns'])]
    protocol = dict(source_sha256=manifest['source_sha256'], selected_frames=len(frames), native_rgb_frames=len(native),
        candidate_boundaries=expected, selected_pairing='every adjacent original R1 identity; no bridging any native/historical reservation',
        failure_retry='all eligible failed boundaries, actual native observations, stop at first failed edge; endpoint check required for success',
        feature_recipe=dict(features=3000, contrastThreshold=.02, seed=11010, fit_tiles='modulo5 != 0', check_tiles='modulo5 == 0'),
        edge_gate=dict(min_check_matches=20, median_m=.05, p90_m=.12, min_fit_hull=.05, min_information_eigenvalue=1., max_condition=1e7),
        global_gate='All required overlaps, fixed difficult structures and independent distributed global constraints must pass. Local edges are not global accuracy.',
        training_gate='Do not expand/extend Gaussian training on rejected geometry. Preserve R3 matte; do not merge independent worlds.',
        evaluation='Exposed development screen, not untouched test or physical metrology. No evaluation depth fits any pose.',
        coverage='Temporal observation/support inventory only, never a surface-area or accuracy fraction.',
        raw_open_mode='read only', date='2026-11-11', timezone='Asia/Saigon')
    m.write(out/'protocol.json', protocol)
    (out/'inputs').mkdir(); (out/'edges').mkdir()
    cal = m.read(m.INPUT/'calibration_clean.json'); K = np.array(cal['K'])
    maps, _ = m.projection.remap(cal['raw_color'], K, cal['width'], cal['height'])
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=.02)
    cv2.setNumThreads(6); cv2.setRNGSeed(11010)
    source = Path(manifest['source']).resolve(); stat = source.stat()
    db = sqlite3.connect(source.as_uri()+'?mode=ro', uri=True); db.execute('PRAGMA query_only=ON')
    tid = db.execute("select id from topics where name like '%/Depth_0/image/data'").fetchone()[0]
    dep = db.execute('select id,timestamp from messages where topic_id=? order by timestamp', (tid,)).fetchall()
    stamps = np.array([v for _,v in dep], dtype=np.int64)
    cache = OrderedDict(); loaded = {}; input_hashes = []

    def load(mid):
        assert allowed(reservations[mid]), 'Reserved observation in a fit'
        if mid in cache:
            cache.move_to_end(mid); return cache[mid]
        path = out/'inputs'/f'{mid}.npz'
        if path.exists():
            with np.load(path) as a: obj = {k:a[k].copy() for k in a.files}
        else:
            f = original.get(mid)
            if f is not None:
                image_path = m.INPUT/'images'/f['name']
                depth_path = m.INPUT/'aligned_depth_meters'/(f['name']+'.npz')
                assert m.sha(image_path) == f['rgb_sha256'] and m.sha(depth_path) == f['depth_sha256']
                image = np.array(Image.open(image_path).convert('RGB'))
                did = f['depth_message_id']
                # R1 archives hold 16k sampled points/depths, not dense depth.
                # Reproject the original raw measurement without filling holes.
                raw, _ = m.clean.adapter.decode_depth(db.execute('select data from messages where id=?', (did,)).fetchone()[0])
                *_, evidence = m.projection.align_depth(raw, cal['depth_units'], cal['raw_depth'], cal['depth_to_color_R'], cal['depth_to_color_t_m'], K, cal['width'], cal['height'])
                depth = evidence['depth_m']
                input_hashes.append(dict(source_id=mid, rgb_sha256=f['rgb_sha256'], depth_sha256=f['depth_sha256']))
            else:
                stamp, blob = db.execute('select timestamp,data from messages where id=?', (mid,)).fetchone()
                assert stamp == reservations[mid]['timestamp_ns']
                rgb, rh = m.clean.capture.CDR(blob).image()
                j = int(np.searchsorted(stamps, stamp))
                j = min([max(0,j-1), min(len(stamps)-1,j)], key=lambda k:abs(int(stamps[k])-stamp))
                did, _ = dep[j]
                raw, dh = m.clean.adapter.decode_depth(db.execute('select data from messages where id=?', (did,)).fetchone()[0])
                assert abs(dh-rh['header_seconds']) < .02
                image = cv2.remap(rgb, maps[:,:,0], maps[:,:,1], cv2.INTER_LINEAR)
                *_, e = m.projection.align_depth(raw, cal['depth_units'], cal['raw_depth'], cal['depth_to_color_R'], cal['depth_to_color_t_m'], K, cal['width'], cal['height'])
                depth = e['depth_m']
            kp, desc = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None)
            uv = np.array([k.pt for k in kp], dtype='f4').reshape(-1,2)
            if desc is None: desc = np.empty((0,128), dtype='f4')
            obj = dict(uv=uv, desc=desc, check=m.check_tile(uv), depth=depth)
            np.savez_compressed(path, **obj)
            loaded[mid] = dict(source_id=mid, depth_source_id=did, seconds=seconds(reservations[mid]),
                               selected_name=f['name'] if f else None, sha256=m.sha(path))
        cache[mid] = obj
        if len(cache)>32: cache.popitem(last=False)
        return obj

    edges = []; exclusions = []
    for i in range(len(frames)-1):
        a,b = frames[i:i+2]
        if i not in expected:
            exclusions.append(dict(boundary=i, seconds=[seconds(a),seconds(b)], reason='native/historical reservation; not a measured geometry failure'))
            continue
        record, _, _ = t.pair(load(a['message_id']), load(b['message_id']), K)
        record.update(boundary=i, source_ids=[a['message_id'],b['message_id']], seconds=[seconds(a),seconds(b)])
        m.write(out/'edges'/f'{i:04d}.json', record); edges.append(record)
        if len(edges)%25 == 0: print('ROUTE', len(edges), '/', len(expected), flush=True)
    m.write(out/'selected_edges.json', dict(edges=edges, excluded_boundaries=exclusions))

    retries = []
    for row in edges:
        if row['accepted']: continue
        a,b = frames[row['boundary']:row['boundary']+2]
        interval = [r for r in native if a['timestamp_ns']<=r['timestamp_ns']<=b['timestamp_ns']]
        mids = [r['rgb_message_id'] for r in interval]
        assert mids[0] == a['message_id'] and mids[-1] == b['message_id'] and all(allowed(r) for r in interval)
        total = np.eye(4); parts = []
        for x,y in zip(mids,mids[1:]):
            record,T,_ = t.pair(load(x),load(y),K); record.update(source_ids=[x,y]); parts.append(record)
            if not record['accepted']: break
            total = T@total
        endpoint = None
        chain_ok = len(parts)==len(mids)-1 and all(p['accepted'] for p in parts)
        if chain_ok:
            fa,fb = load(mids[0]),load(mids[-1]); ca,cb=fa['check'],fb['check']
            matches=m.mutual(fa['desc'][ca],fb['desc'][cb]); ua,ub=fa['uv'][ca][matches[:,0]],fb['uv'][cb][matches[:,1]]
            da,va,_=m.sample_depth(fa['depth'],ua); dd,vb,_=m.sample_depth(fb['depth'],ub); valid=va&vb
            x=m.backproject(ua[valid],da[valid],K); y=m.backproject(ub[valid],dd[valid],K)
            error=np.linalg.norm(x@total[:3,:3].T+total[:3,3]-y,axis=1)
            endpoint=dict(matches=len(error), median_p90_m=np.percentile(error,[50,90]).tolist() if len(error) else None,
                          passed=bool(len(error)>=20 and np.median(error)<.05 and np.percentile(error,90)<.12))
        retry=dict(boundary=row['boundary'], native_source_ids=mids, edges=parts, total_native_edges=len(mids)-1,
                   accepted=bool(chain_ok and endpoint['passed']), endpoint_check=endpoint,
                   composed_transform=total.tolist() if chain_ok else None)
        m.write(out/'edges'/f'{row["boundary"]:04d}_native.json',retry); retries.append(retry)
        print('RETRY', row['boundary'], retry['accepted'], len(parts), '/',len(mids)-1, flush=True)
    db.close()
    assert (source.stat().st_size,source.stat().st_mtime_ns)==(stat.st_size,stat.st_mtime_ns)
    m.write(out/'native_retries.json',dict(retries=retries, all_eligible_failures_attempted=True))
    retry_by = {r['boundary']:r for r in retries}
    final = [dict(boundary=e['boundary'], accepted=e['accepted'] or retry_by[e['boundary']]['accepted']) if not e['accepted']
             else dict(boundary=e['boundary'],accepted=True) for e in edges]
    previous = m.read(ROOT/'results/db3_r0_r3_cleanup_20261111/phase_decision.json')
    structure = m.read(m.RUN/'exhibit/pilot/selected_policy_confirmation_result_supported_15k.json')
    global_report = m.read(m.RUN/'transitions_revisits_v2/report.json')
    decision = expansion_decision(final, expected, structures_pass=structure['manual_structure_quantitative_pass'],
                                  global_constraints_pass=global_report['whole_recording_geometry_accepted'])
    m.write(out/'phase_decision.json',dict(**decision, prior_all_R0_R3_accepted=previous['all_R0_R3_scientific_acceptance_complete'],
        R4_execution_complete=True, R4_whole_scene_reconstruction_accepted=False,
        selected_delivery='Preserved full-route 120k context plus separate accepted matte local candidate',
        new_global_model=False, new_gaussian_training=False,
        reason='Full-route screen and bounded native retries do not replace failed physical structures or missing distributed placement checks.',
        R5='pending final interactive verification; no claim made by R4',
        observed_failure_is_proof_of_impossibility=False))
    counts=dict(selected_observations=len(frames), native_observations=len(native), fitting_selected_observations=sum(allowed(f) for f in frames),
        candidate_boundaries=len(expected), excluded_boundaries=len(exclusions), selected_pass=sum(e['accepted'] for e in edges),
        selected_fail=sum(not e['accepted'] for e in edges), native_recovered=sum(r['accepted'] for r in retries),
        unresolved_eligible_boundaries=sum(not e['accepted'] for e in final), fitting_inputs_loaded=len(loaded))
    m.write(out/'route_summary.json',dict(counts=counts, selected_extent_seconds=[seconds(frames[0]),seconds(frames[-1])],
        source_sha256=manifest['source_sha256'], raw_hash_reverification='integrity/integrity_audit.json',
        assumptions='Pairwise local consistency and assumed sensor uncertainty; no global camera/landmark BA or physical accuracy claim.',
        temporal_coverage_is_not_surface_coverage=True, elapsed_seconds=time.time()-started))
    m.write(out/'inputs_manifest.json',dict(frames=list(loaded.values()), selected_input_hashes=input_hashes,
        calibration_sha256=m.sha(m.INPUT/'calibration_clean.json'), reservation_sha256=m.sha(m.R1/'prospective_checks.json'),
        all_loaded_frames_are_fit_only=all(allowed(reservations[mid]) for mid in loaded),
        sources={str(p.relative_to(ROOT)):m.sha(p) for p in (m.INPUT/'capture_manifest.json',m.INPUT/'reconstruction_split.json',
            m.RUN/'exhibit/pilot/selected_policy_confirmation_result_supported_15k.json',m.RUN/'transitions_revisits_v2/report.json')}))
    coverage=[]
    for i,f in enumerate(frames):
        sec=seconds(f); role='historical_eval' if f['message_id'] in historical else reservations[f['message_id']]['role']
        coverage.append(dict(name=f['name'],source_id=f['message_id'],seconds=sec,role=role,
            selected_context='preserved full-room baseline; uncertain geometry',
            accepted_local='matte, separate coordinate frame' if 6<=sec<=14 else None,
            surface_coverage='unknown; no validated full-route surface reference',
            adjacent_pair_status=next((('local_pair_pass' if r['accepted'] else 'unresolved') for r in final if r['boundary']==i), 'excluded_or_end')))
    m.write(out/'coverage_uncertainty.json',dict(frames=coverage, independent_submaps=True, global_alignment=None,
        unknown_regions=['far depth','glass/reflections','occluded/unseen backsides','exhibit backdrop/floor/wall/post failures'],
        limits='Temporal evidence inventory; matte acceptance is confined to frozen checked views/patches, not every surface in its time window.'))
    draw_route(out, frames, edges, retries, exclusions, origin)
    for path in (Path(__file__), ROOT/'scripts/db3_transition_revisit.py',ROOT/'scripts/db3_local_geometry.py'):
        dst=out/'source_snapshot'/path.name;dst.parent.mkdir(exist_ok=True);shutil.copyfile(path,dst)
    m.write(out/'source_manifest.json',{p.name:m.sha(p) for p in (out/'source_snapshot').iterdir()})
    print('R4 SCREEN COMPLETE',counts,flush=True)


def draw_route(out, frames, edges, retries, exclusions, origin):
    image=Image.new('RGB',(1400,420),'white');d=ImageDraw.Draw(image)
    d.text((25,15),'R4 temporal support inventory | green = pair checks only; red = unresolved; gray = excluded',fill='black')
    d.text((25,38),'No color certifies global geometry or physical surface coverage. Whole-room baseline retained as uncertain context.',fill='black')
    duration=(frames[-1]['timestamp_ns']-origin)/1e9
    x=lambda sec:30+int(sec/duration*1340)
    byretry={r['boundary']:r for r in retries}
    for e in edges:
        ok=e['accepted'] or byretry.get(e['boundary'],{}).get('accepted',False)
        d.rectangle((x(e['seconds'][0]),95,x(e['seconds'][1])+1,155),fill=(40,150,85) if ok else (205,65,60))
    for e in exclusions:d.rectangle((x(e['seconds'][0]),95,x(e['seconds'][1])+1,155),fill=(155,155,155))
    for sec,label in [(0,'start/kiosk'),(10,'matte'),(43,'wide room'),(61,'doorway'),(68,'exhibit'),(110,'glass'),(157,'end/revisit')]:
        xx=x(sec);d.line((xx,160,xx,190),fill='black');d.text((min(xx,1270),195),f'{sec}s {label}',fill='black')
    d.rectangle((x(6),250,x(14),280),fill=(50,100,190))
    d.text((30,300),'Blue: time window of separately delivered matte submap; only declared checks accepted. Global relationship unknown.',fill='black')
    d.text((30,325),'Exhibit remains unpromoted. Missing depth / reserved observations / failed checks remain distinct categories.',fill='black')
    image.save(out/'route_support.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args().output_dir)
