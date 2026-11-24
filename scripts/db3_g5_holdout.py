"""Reserve unused raw observations and localize them against the fixed matte map."""

import argparse
from collections import defaultdict
from contextlib import closing
from pathlib import Path
import sqlite3

import cv2
import numpy as np
from PIL import Image
import pycolmap
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import db3_local_geometry as local
from week12_comparison import REPO, read, sha, write


GEO = local.RUN / "matte/rate10"
R1_CAPTURE = local.INPUT / "capture_manifest.json"
R2_CAPTURE = local.DATA / "manifest.json"
NATIVE = local.R1 / "prospective_checks.json"
ROUTE = "results/db3_r4_20261111_v2/inputs_manifest.json"
GATES = dict(inliers=25, check_matches=20, check_hull=.04, check_median_px=2.,
             check_p90_px=4., split_translation_m=.025, split_rotation_deg=.6,
             pair_gap_ms=20., min_views=6, min_capture_span_s=3.)


def previous_observations():
    manifests = (R1_CAPTURE, R2_CAPTURE, REPO / ROUTE)
    rgb, depth = set(), set()
    for path in manifests:
        for f in read(path)["frames"]:
            rgb.add(f.get("rgb_message_id", f.get("message_id", f.get("source_id"))))
            depth.add(f.get("depth_message_id", f.get("depth_source_id")))
    # Additional native-rate R4 retries and the explicit local/transition trials.
    results = REPO / "results"
    files = sorted({*results.glob("db3_r4_*/inputs/*.npz"),
                    *results.glob("db3_*/matte_control/inputs/*")})
    rgb.update(int(p.stem) for p in files if p.stem.isdigit())
    sources = [*manifests, NATIVE, local.DATA / "calibration_clean.json", GEO / "input.json",
               *sorted((GEO / "metric_joint").glob("*.bin"))]
    return rgb, depth, sources, files


def reservation(db):
    rgb, used_depth, sources, files = previous_observations()
    depth_topic = db.execute("select id from topics where name like '%/Depth_0/image/data'").fetchone()[0]
    depth = db.execute("select timestamp,id from messages where topic_id=? order by timestamp",
                       (depth_topic,)).fetchall()
    stamps = np.array([t for t, _ in depth], dtype=np.int64)

    def nearest(stamp):
        index = int(np.searchsorted(stamps, stamp))
        neighbors = depth[max(0, index - 1):index + 1]
        return min(neighbors, key=lambda row: abs(row[0] - stamp))

    used_depth.update(d for _, d in (nearest(f["timestamp_ns"]) for f in read(NATIVE)["source_frames"]
                                    if f["rgb_message_id"] in rgb))
    groups = defaultdict(list)
    reserved_depth = set()
    for f in read(NATIVE)["source_frames"]:
        t = f["replay_seconds"]
        if not (6. <= t < 14.) or f["rgb_message_id"] in rgb:
            continue
        depth_stamp, depth_id = nearest(f["timestamp_ns"])
        if depth_id in used_depth or depth_id in reserved_depth:
            continue
        if abs(depth_stamp - f["timestamp_ns"]) > GATES["pair_gap_ms"] * 1e6:
            continue
        reserved_depth.add(depth_id)
        groups[int((t - 6.) / .6)].append(dict(f, depth_message_id=depth_id, depth_stamp=depth_stamp))

    def midpoint_order(item):
        index, group = item
        center = 6. + .6 * (index + .5)
        return sorted(group, key=lambda f: (abs(f["replay_seconds"] - center), f["rgb_message_id"]))

    return dict(
        gates=GATES, trials=[midpoint_order(item) for item in sorted(groups.items())],
        exclusions_sha256={str(p.relative_to(REPO)): sha(p) for p in sources},
        historical_native_input_files=[str(p.relative_to(REPO)) for p in files],
        exclusion_rule="Exclude all R1 selected, R2 prepared, R4 inputs and recorded local native trials, "
                       "and every depth paired with those frames. The full recording was previously indexed.",
        selection="One RGB-validated pose per 0.6 s stratum, in closest-to-midpoint order. "
                  "Select without depth data or model outputs. Score every accepted stratum.",
        pose_rule="Mutual SIFT matches against the ten nearest frozen mapping cameras. Fit with "
                  "four spatial partitions; check on the fifth; compare disjoint half-fits.",
        source_sha256={p.name: sha(p) for p in (Path(__file__), Path(local.__file__))},
    )


def landmarks():
    model = pycolmap.Reconstruction(GEO / "metric_joint")
    when = {f["name"]: f["replay_seconds"] for f in read(GEO / "input.json")["frames"]}
    frames = []
    for im in sorted(model.images.values(), key=lambda item: item.name):
        with np.load(local.DATA / "features" / f"{im.name}.npz") as features:
            uv, desc = features["uv"], features["desc"]
            keep = ~features["check"]
        points = [pt for pt in im.points2D if pt.has_point3D()]
        if not points:
            continue
        distance, nn = cKDTree(uv[keep]).query(np.stack([p.xy for p in points]))
        descriptors = desc[keep]
        bank = {points[i].point3D_id: descriptors[nn[i]] for i in np.flatnonzero(distance < 1.2)}
        frames.append((im.name, when[im.name], bank))
    return model, frames


def pnp(xyz, pixels, K):
    if len(pixels) < GATES["inliers"]:
        return None, 0
    cv2.setRNGSeed(11010)
    ok, r, t, inliers = cv2.solvePnPRansac(
        xyz, pixels, K, None, iterationsCount=500, reprojectionError=2.,
        confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < GATES["inliers"]:
        return None, 0 if inliers is None else len(inliers)
    keep = inliers.ravel()
    r, t = cv2.solvePnPRefineLM(xyz[keep], pixels[keep], K, None, r, t)
    world_to_camera = np.eye(4)
    world_to_camera[:3, :3] = cv2.Rodrigues(r)[0]
    world_to_camera[:3, 3] = t[:, 0]
    return world_to_camera, len(inliers)


def localize(image, seconds, K, sift, model, reference):
    rgb_points, desc = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None)
    if desc is None:
        return {"accepted": False, "reason": "no RGB features"}, {}
    nearby = sorted(reference, key=lambda f: (abs(f[1] - seconds), f[0]))[:10]
    bank = {}
    for _, _, descriptors in reversed(nearby):
        bank.update(descriptors)
    ids = np.array(sorted(bank))
    matched = local.mutual(desc, np.stack([bank[pid] for pid in ids]))
    xyz = np.array([model.points3D[int(pid)].xyz for pid in ids[matched[:, 1]]]).reshape(-1, 3)
    uv = np.array([rgb_points[i].pt for i in matched[:, 0]]).reshape(-1, 2)
    tile = np.floor(uv / 70).astype(int)
    group = (tile[:, 0] + 2 * tile[:, 1]) % 5
    fits = [pnp(xyz[mask], uv[mask], K) for mask in
            (group != 0, np.isin(group, (1, 2)), np.isin(group, (3, 4)))]
    point_data = dict(xyz=xyz, uv=uv, tile_group=group, landmark_ids=ids[matched[:, 1]])
    primary = fits[0][0]
    check = group == 0
    row = dict(accepted=False, matched=len(uv), inliers=[n for _, n in fits],
               check_matches=int(check.sum()), reference_rgb=[name for name, _, _ in nearby])
    if any(W is None for W, _ in fits) or check.sum() < GATES["check_matches"]:
        return dict(row, reason="insufficient disjoint RGB correspondences"), point_data
    q = xyz[check] @ primary[:3, :3].T + primary[:3, 3]
    p = q @ K.T
    if not (q[:, 2] > 0.).all():
        return dict(row, reason="check points behind camera"), point_data
    error = np.linalg.norm(p[:, :2] / p[:, 2:3] - uv[check], axis=1)
    p50, p90 = np.percentile(error, (50, 90))
    hull = cv2.contourArea(cv2.convexHull(uv[check].astype("f4"))) / np.prod(image.shape[:2])
    a, b = [np.linalg.inv(W) for W, _ in fits[1:]]
    translation = np.linalg.norm(a[:3, 3] - b[:3, 3])
    rotation = np.degrees(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())
    row.update(check_median_px=float(p50), check_p90_px=float(p90), check_hull=float(hull),
               split_translation_m=float(translation), split_rotation_deg=float(rotation),
               world_to_camera=primary.tolist(),
               split_world_to_camera=[W.tolist() for W, _ in fits[1:]])
    row["accepted"] = bool(p50 <= GATES["check_median_px"] and p90 <= GATES["check_p90_px"]
                           and hull >= GATES["check_hull"] and translation <= GATES["split_translation_m"]
                           and rotation <= GATES["split_rotation_deg"])
    return row, point_data | {"check_error_px": error}


def reserve(out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = read(R2_CAPTURE)
    source = Path(manifest["source"])
    assert sha(source) == manifest["source_sha256"]
    cal = read(local.DATA / "calibration_clean.json")
    K = np.array(cal["K"])
    w, h = cal["width"], cal["height"]
    remap, valid = local.projection.remap(cal["raw_color"], K, w, h)
    assert valid.all()
    cv2.setNumThreads(4)
    sift = cv2.SIFT_create(nfeatures=3500, contrastThreshold=.02)
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as db:
        protocol = reservation(db)
        protocol.update(recording_sha256=manifest["source_sha256"], K=K.tolist())
        write(out / "reservation.json", protocol)
        model, reference = landmarks()
        attempts, accepted = [], []
        for group in protocol["trials"]:
            for f in group:
                mid = f["rgb_message_id"]
                ts, blob = db.execute("select timestamp,data from messages where id=?", (mid,)).fetchone()
                assert ts == f["timestamp_ns"]
                rgb, header = local.clean.capture.CDR(blob).image()
                image = cv2.remap(rgb, remap[:, :, 0], remap[:, :, 1], cv2.INTER_LINEAR)
                pose, evidence = localize(image, f["replay_seconds"], K, sift, model, reference)
                row = dict(f, **pose)
                attempts.append(row)
                match_file = out / f"{mid}_matches.npz"
                np.savez_compressed(match_file, **evidence)
                row["matches_sha256"] = sha(match_file)
                if pose["accepted"]:
                    rgb_path = out / f"{mid}.png"
                    Image.fromarray(image).save(rgb_path, compress_level=1)
                    row.update(rgb_sha256=sha(rgb_path), rgb_header_sec=header["header_seconds"])
                    accepted.append(row)
                    break
        # Seal the accepted RGB poses before any reserved depth is decoded.
        write(out / "localization.json", dict(attempted=attempts, accepted=accepted,
                                              reservation_sha256=sha(out / "reservation.json")))
        for row in accepted:
            did = row["depth_message_id"]
            ts, blob = db.execute("select timestamp,data from messages where id=?", (did,)).fetchone()
            assert ts == row["depth_stamp"]
            raw, header = local.clean.adapter.decode_depth(blob)
            dt_ms = 1000 * abs(header - row["rgb_header_sec"])
            if dt_ms > GATES["pair_gap_ms"]:
                row["timing_failure_ms"] = dt_ms
                continue
            *_, aligned = local.projection.align_depth(
                raw, cal["depth_units"], cal["raw_depth"], cal["depth_to_color_R"],
                cal["depth_to_color_t_m"], K, w, h)
            depth_path = out / f"{row['rgb_message_id']}_depth.npz"
            np.savez_compressed(depth_path, depth_m=aligned["depth_m"])
            row.update(depth_sha256=sha(depth_path), pair_gap_ms=dt_ms)
    frames = [f for f in accepted if "depth_sha256" in f]
    span = frames[-1]["replay_seconds"] - frames[0]["replay_seconds"] if frames else 0.
    qualified = len(frames) >= GATES["min_views"] and span >= GATES["min_capture_span_s"]
    write(out / "frames.json", dict(frames=frames, capture_span_s=span, qualified=qualified, K=K.tolist(),
                                    localization_sha256=sha(out / "localization.json"),
                                    reservation_sha256=sha(out / "reservation.json")))
    print(f"Qualified holdout: {qualified}. Validated poses with depth: {len(frames)}. "
          f"Capture span: {span:.2f} s.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    reserve(parser.parse_args().output.resolve())
