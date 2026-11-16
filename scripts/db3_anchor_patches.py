"""Pose-independent affine template anchoring for existing source identities.

Old trajectories and evidence are immutable. Corrections are a new measurement
arm, never a deletion of the earlier failed checks.
"""
import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location('diagnostic', Path(__file__).with_name('diagnose_db3_tracking_bias.py'))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
m, r = d.m, d.r
PATCH = 41
HALF = (PATCH-1)/2


def correct(first, current, source, guess):
    """Independent forward/reverse ECC registrations; return diagnostics always."""
    report = dict(ok=False, reason='border')
    for im, xy in ((first, source), (current, guess)):
        if not (HALF+12 <= xy[0] < im.shape[1]-HALF-12 and
                HALF+12 <= xy[1] < im.shape[0]-HALF-12):
            return np.asarray(guess).copy(), report
    a = cv2.getRectSubPix(first, (PATCH, PATCH), tuple(map(float, source)))
    b = cv2.getRectSubPix(current, (PATCH, PATCH), tuple(map(float, guess)))
    center = np.array([HALF, HALF, 1.])
    try:
        settings = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
        cf, fw = cv2.findTransformECC(a, b, np.eye(2, 3, dtype='f4'), cv2.MOTION_AFFINE, settings, None, 5)
        cb, bw = cv2.findTransformECC(b, a, np.eye(2, 3, dtype='f4'), cv2.MOTION_AFFINE, settings, None, 5)
        delta = fw@center-center[:2]
        cycle = bw@np.r_[fw@center, 1.]-center[:2]
        singular = np.linalg.svd(fw[:, :2], compute_uv=False)
        ok = (min(cf, cb) >= .85 and np.linalg.norm(cycle) < .75 and
              np.linalg.norm(delta) <= 12 and min(singular) > .75 and max(singular) < 1.3334 and
              np.linalg.det(fw[:, :2]) > 0)
        report = dict(ok=bool(ok), reason='accepted' if ok else 'photometric_or_cycle_or_warp',
                      forward_correlation=float(cf), backward_correlation=float(cb),
                      cycle_px=float(np.linalg.norm(cycle)), displacement_px=float(np.linalg.norm(delta)),
                      singular_values=singular.tolist(), warp=fw.tolist())
        return np.asarray(guess)+delta, report
    except cv2.error:
        return np.asarray(guess).copy(), dict(ok=False, reason='ecc_nonconvergence')


def diagnostic(out):
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(6)
    protocol = m.read(d.d.PARENT/'protocol.json')
    paths = [Path(__file__), Path(d.__file__), m.INPUT/'calibration_clean.json']
    paths.extend(f for f in d.d.INPUTS.rglob('*') if f.is_file())
    m.write(out/'protocol.json', dict(date='2026-11-12', cases=protocol['cases'],
        method='Same 74/112 source IDs; every frame independently registered to original first-frame 41x41 patch by affine ECC. Both directions >=.85 correlation, cycle<.75px, shift<=12px, singular values (.75,1.3334). Fixed80 iterations, epsilon1e-5, Gaussian5. No depth/pose/error used. Old checks retained. Diagnostic only.',
        input_hashes={str(f.relative_to(r.ROOT)): m.sha(f) for f in paths}))
    K = np.array(m.read(m.INPUT/'calibration_clean.json')['K'])
    results = []
    for case in protocol['cases']:
        label = case['label']; parent = d.d.INPUTS/label
        dest = out/label; dest.mkdir()
        rows = m.read(parent/'inputs.json')
        images = [cv2.imread(str(parent/'inputs'/f'{x["source_id"]}.png'), 0) for x in rows]
        with np.load(parent/'moving_checks.npz') as z:
            old = z['uv'].copy()
        corrected = old.astype(float).copy(); ok = np.ones(old.shape[:2], bool); diagnostics = []
        for c in range(1, len(images)):
            for i in range(old.shape[1]):
                corrected[c, i], report = correct(images[0], images[c], old[0, i], old[c, i])
                ok[c, i] = report['ok']; diagnostics.append(dict(camera=c, id=i, **report))
        np.savez_compressed(dest/'corrected.npz', uv=corrected, old_uv=old, ok=ok)
        m.write(dest/'registration.json', diagnostics)
        fs = []
        for c, row in enumerate(rows):
            with np.load(parent/'inputs'/f'{row["source_id"]}.npz') as z:
                fs.append(dict(uv=corrected[c], depth=z['depth'].copy()))
        r.observations(fs, K)
        poses = np.array(m.read(r.ROOT/'results/db3_direct_depth_20261111'/label/'poses.json'))
        frozen = m.read(parent/'frozen_checks.json')
        # Report ALL original refs, including corrections failing independent quality.
        measured = r.evaluate(fs, frozen, poses, K)
        row = dict(label=label, original_identity_count=old.shape[1],
                   accepted_per_frame=ok.sum(axis=1).tolist(), complete_accepted_tracks=int(ok.all(axis=0).sum()),
                   all_original_refs_diagnostic=measured,
                   correction_displacement_px=d.quantiles(np.linalg.norm(corrected-old, axis=2).ravel()),
                   training_authorized=False)
        m.write(dest/'result.json', row); results.append(row)
        print(label, 'accepted', row['complete_accepted_tracks'], '/', old.shape[1],
              'old-camera corrected endpoint', measured[0]['symmetric_reprojection_px'], flush=True)
    m.write(out/'result.json', dict(cases=results, diagnostic_only=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    diagnostic(parser.parse_args().output_dir)
