"""Refine ARKit world placement using metric, mutually verified RGB-D revisits."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from sklearn.cluster import MiniBatchKMeans

from iphone_rescan_alignment import rgb_depth
from stray_quality_preflight import SIZE, write_json
from stray_scanner_reconstruct import columns, percentiles, read_csv


def camera_points(source, frame, pixels, intrinsic):
    depth, confidence = rgb_depth(source, frame)
    x = np.clip(np.rint(pixels[:, 0] * 256 / SIZE[0]).astype(int), 0, 255)
    y = np.clip(np.rint(pixels[:, 1] * 192 / SIZE[1]).astype(int), 0, 191)
    z = depth[y, x] * .001
    lo = cv2.erode(depth, np.ones((3, 3), np.uint8))[y, x] * .001
    hi = cv2.dilate(depth, np.ones((3, 3), np.uint8))[y, x] * .001
    good = (confidence[y, x] == 2) & (lo >= .25) & (z <= 5) & (hi - lo < .045 + .017 * z)
    fx, fy, cx, cy = intrinsic
    point = np.column_stack(((pixels[:, 0] - cx) * z / fx, (pixels[:, 1] - cy) * z / fy, z))
    point[~good] = np.nan
    return point


def pnp(points, pixels, intrinsic):
    keep = np.isfinite(points[:, 2])
    p, q = points[keep], pixels[keep]
    if len(p) < 24:
        return None
    fx, fy, cx, cy = intrinsic
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    ok, rv, tv, indices = cv2.solvePnPRansac(p, q, k, None, iterationsCount=220, reprojectionError=2.,
                                             confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or indices is None or len(indices) < 20 or len(indices) < .52 * len(p):
        return None
    indices = indices.ravel()
    if np.linalg.norm(np.ptp(q[indices], axis=0)) < 70 or np.linalg.norm(np.std(p[indices], axis=0)) < .10:
        return None
    rv, tv = cv2.solvePnPRefineLM(p[indices], q[indices], k, None, rv, tv)
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = cv2.Rodrigues(rv)[0], tv.ravel()
    return result, len(indices)


def verify_pair(a, b, intrinsic_a, intrinsic_b, matcher):
    pix_a, da, a3 = a
    pix_b, db, b3 = b
    if da is None or db is None or min(len(da), len(db)) < 24:
        return None
    matched = matcher.knnMatch(da, db, k=2)
    pairs = [(m.queryIdx, m.trainIdx, m.distance) for m, n in matched if m.distance < .76 * n.distance]
    if len(pairs) < 25:
        return None
    # Keep the strongest proposal for a target feature to prevent duplicated votes.
    unique = {j: i for i, j, _ in sorted(pairs, key=lambda row: -row[2])}
    ia, ib = np.array(list(unique.values())), np.array(list(unique.keys()))
    forward = pnp(a3[ia], pix_b[ib], intrinsic_b)
    reverse = pnp(b3[ib], pix_a[ia], intrinsic_a)
    if forward is None or reverse is None:
        return None
    transform, n1 = forward
    inverse, n2 = reverse
    round_trip = inverse @ transform
    if np.linalg.norm(round_trip[:3, 3]) > .055 or Rotation.from_matrix(round_trip[:3, :3]).magnitude() > .035:
        return None
    a_to_b = a3[ia] @ transform[:3, :3].T + transform[:3, 3]
    residual = np.linalg.norm(a_to_b - b3[ib], axis=1)
    residual = residual[np.isfinite(residual)]
    if len(residual) < 17 or np.median(residual) > .055:
        return None
    return transform, min(n1, n2), float(np.median(residual))


def candidate_pairs(descriptors, time, position, rotation):
    arrays = [d for d in descriptors if d is not None]
    sample = np.concatenate([a[::max(1, len(a) // 60)] for a in arrays])
    rng = np.random.default_rng(43)
    sample = rng.permutation(sample)[:33000]
    vocabulary = MiniBatchKMeans(n_clusters=384, n_init=1, max_iter=30, batch_size=1536, random_state=43).fit(sample)
    bins = np.array([np.bincount(vocabulary.predict(d), minlength=384) if d is not None else np.zeros(384)
                     for d in descriptors], dtype=np.float32)
    inverse_frequency = np.log((1 + len(bins)) / (1 + (bins > 0).sum(0))) + 1
    features = np.sqrt(bins) * inverse_frequency
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    similarity = features @ features.T
    separation = abs(time[:, None] - time)
    distance = np.linalg.norm(position[:, None] - position, axis=-1)
    facing = rotation[:, :, 2] @ rotation[:, :, 2].T
    candidates = set()
    for i in range(len(time)):
        found = []
        for j in np.argsort(-similarity[i]):
            if separation[i, j] < 9 or similarity[i, j] < .20 or any(abs(time[j] - time[k]) < 4 for k in found):
                continue
            found.append(j)
            if len(found) == 10:
                break
        neighbors = np.flatnonzero((separation[i] >= 7) & (distance[i] <= 1.6) & (facing[i] > .66))
        found.extend(neighbors[np.argsort(-similarity[i, neighbors])[:5]])
        candidates.update(tuple(sorted((i, j))) for j in found)
    return sorted(candidates)


def refine(run, source, folder):
    folder.mkdir(exist_ok=True, parents=True)
    audit = run / "dataset_audit"
    rows = read_csv(source / "odometry.csv")
    time = columns(rows, "timestamp").ravel()
    time -= time[0]
    preflight = json.loads((audit / "preflight.json").read_text())
    width, height = preflight["native_rgb_size"]
    calibration = columns(rows, "fx", "fy", "cx", "cy") * ([SIZE[0] / width, SIZE[1] / height] * 2)
    xyz = columns(rows, "x", "y", "z")
    rotation = Rotation.from_quat(columns(rows, "qx", "qy", "qz", "qw")).as_matrix()
    fit = np.array(json.loads((run / "splits/train_frames.json").read_text()))
    sharpness = np.load(audit / "sharpness.npy")
    bin_id = np.floor(time[fit] / 1.05).astype(int)
    groups = np.split(fit, np.flatnonzero(np.diff(bin_id)) + 1)
    nodes = np.array([group[np.argmax(sharpness[group])] for group in groups])
    nodes = np.unique(np.r_[fit[0], nodes, fit[-1]])
    pose = np.broadcast_to(np.eye(4), (len(nodes), 4, 4)).copy()
    pose[:, :3, :3], pose[:, :3, 3] = rotation[nodes], xyz[nodes]
    loop_path = folder / "metric_loops.json"
    if loop_path.exists():
        data = json.loads(loop_path.read_text())
        if nodes.tolist() != data["nodes"]:
            raise ValueError("Pose loop nodes changed")
        verified, tested = data["pairs"], data["tested"]
    else:
        proxy = np.memmap(audit / "proxy_gray.bin", dtype=np.uint8, mode="r", shape=(len(time), SIZE[1], SIZE[0]))
        sift = cv2.SIFT_create(nfeatures=750, contrastThreshold=.018)
        features = []
        for i, frame in enumerate(nodes):
            k, d = sift.detectAndCompute(proxy[frame], None)
            pixels = np.array([key.pt for key in k], dtype=np.float64).reshape(-1, 2)
            if d is not None:
                d = np.sqrt(d / np.maximum(d.sum(1, keepdims=True), 1e-8))
            features.append((pixels, d, camera_points(source, int(frame), pixels, calibration[frame])))
            if (i + 1) % 180 == 0:
                print(f"Extracted RGB-D features for {i+1}/{len(nodes)} fit nodes.", flush=True)
        pairs = candidate_pairs([f[1] for f in features], time[nodes], xyz[nodes], rotation[nodes])
        print(f"Testing {len(pairs):,} potential revisits among {len(nodes)} fit nodes.", flush=True)
        matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 48})
        verified, tested = [], len(pairs)
        for k, (i, j) in enumerate(pairs):
            match = verify_pair(features[i], features[j], calibration[nodes[i]], calibration[nodes[j]], matcher)
            if match is not None:
                transform, count, error = match
                registration = np.linalg.inv(pose[j]) @ pose[i]
                if (np.linalg.norm(transform[:3, 3] - registration[:3, 3]) <= 4
                        and Rotation.from_matrix(transform[:3, :3].T @ registration[:3, :3]).magnitude() <= .55):
                    verified.append({"nodes": [int(i), int(j)], "transform": transform.tolist(), "inliers": count,
                                     "median_metric_error_m": error})
            if (k + 1) % 600 == 0:
                print(f"Pose revisits {k+1:,}/{len(pairs):,}, {len(verified):,} verified.", flush=True)
        write_json(loop_path, {"nodes": nodes.tolist(), "tested": tested, "pairs": verified})
    if len(verified) < 5:
        raise RuntimeError("Insufficient metric revisit support for trajectory refinement")
    spec = spec_from_file_location("sparse_pose_graph", Path(__file__).with_name("sparse-rgbd-pose-graph.py"))
    solver = module_from_spec(spec)
    spec.loader.exec_module(solver)

    def edge(i, j, transform, weight):
        return SimpleNamespace(source_node_id=i, target_node_id=j, transformation=transform, information=np.eye(6) * weight)

    local = []
    for stride, weight in ((1, 100), (2, 46)):
        for i in range(len(nodes) - stride):
            j = i + stride
            local.append(edge(i, j, np.linalg.inv(pose[j]) @ pose[i], weight))
    loops = [edge(*record["nodes"], np.asarray(record["transform"]), np.clip(record["inliers"] * 1.7, 65, 560))
             for record in verified]
    aligned, trace = solver.optimize(pose, local + loops, max_nfev=120)

    def errors(poses):
        return np.array([np.linalg.norm((np.linalg.inv(poses[e.target_node_id]) @ poses[e.source_node_id])[:3, 3]
                                        - e.transformation[:3, 3]) for e in loops])

    after = errors(aligned)
    trusted = [e for e, e_error in zip(loops, after) if e_error <= .12]
    if len(trusted) >= 5 and len(trusted) < len(loops):
        aligned, trace = solver.optimize(aligned, local + trusted, max_nfev=80)
    correction = aligned @ np.linalg.inv(pose)
    interpolated_rotation = Slerp(time[nodes], Rotation.from_matrix(correction[:, :3, :3]))(
        np.clip(time, time[nodes[0]], time[nodes[-1]])).as_matrix()
    offset = np.column_stack([np.interp(time, time[nodes], correction[:, c, 3]) for c in range(3)])
    corrected_positions = (interpolated_rotation @ xyz[..., None]).squeeze(-1) + offset
    corrected_rotation = interpolated_rotation @ rotation
    np.savez_compressed(folder / "poses.npz", positions_m=corrected_positions, camera_to_world_rotation=corrected_rotation)
    write_json(folder / "metrics.json", {"keyframes": len(nodes), "candidates": tested, "verified_revisits": len(verified),
                                          "trusted_revisits": len(trusted), "solver": trace,
                                          "loop_translation_before_m": percentiles(errors(pose)),
                                          "loop_translation_after_m": percentiles(errors(aligned)),
                                          "position_correction_m": percentiles(np.linalg.norm(corrected_positions - xyz, axis=1)),
                                          "rotation_correction_deg": percentiles(np.degrees(Rotation.from_matrix(interpolated_rotation).magnitude()))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    cv2.setRNGSeed(43)
    refine(args.run, args.source, args.output)
