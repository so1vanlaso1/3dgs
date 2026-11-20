"""Compare native DB3 wall thickness and inter-view separation with saved poses."""

import argparse
from contextlib import closing
import csv
import importlib.util
from itertools import combinations
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycolmap
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("baseline", ROOT / "scripts/run-db3-r0-r1.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
p = baseline.p

# Rectified SDK RGB coordinates. Freeze from RGB before decoding depth.
# Patches lie on the plain wall beneath the lit panel, away from the exhibit.
# Edge marks follow the outer jamb-to-opening silhouette, to about +/-2 pixels.
CASES = {
    200: ((548, 234, 652, 282), [(235, 699), (250, 697), (265, 696), (280, 695)]),
    208: ((500, 170, 632, 235), [(175, 676), (190, 675), (205, 674), (220, 673)]),
    240: ((396, 140, 534, 215), [(145, 576), (160, 572), (175, 569), (190, 566)]),
}


def pose_matrix(image, scale):
    w2c = np.eye(4)
    w2c[:3] = image.cam_from_world().matrix()
    c2w = np.linalg.inv(w2c)
    c2w[:3, 3] /= scale
    return c2w


def transform(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def plane_fit(points):
    # Deterministic robust fit. Report residuals for ALL observed points below.
    keep = np.ones(len(points), dtype=bool)
    for _ in range(4):
        center = points[keep].mean(axis=0)
        _, _, axes = np.linalg.svd(points[keep] - center, full_matrices=False)
        normal = axes[-1]
        residual = (points - center) @ normal
        median = np.median(residual)
        mad = np.median(np.abs(residual - median))
        keep = np.abs(residual - median) <= max(.004, 4 * 1.4826 * mad)
    if normal @ center < 0:
        normal = -normal
    return center, normal


def residual_stats(values):
    return {
        "median_signed_mm": float(1000 * np.median(values)),
        "p50_p90_p99_absolute_mm": (1000 * np.percentile(abs(values), [50, 90, 99])).tolist(),
    }


def compare(a, b, transforms):
    # Pair-local coordinates referenced to A's pose. Evaluate points only where
    # their tangential projection falls inside the other wall patch's footprint.
    rel = np.linalg.solve(transforms[a["id"]], transforms[b["id"]])
    xa, xb = a["xyz"], transform(b["xyz"], rel)
    ca, na = a["plane"]
    cb, nb = transform(b["plane"][0], rel), rel[:3, :3] @ b["plane"][1]
    horizontal = np.cross([0, 1, 0], na)
    horizontal /= np.linalg.norm(horizontal)
    basis = np.column_stack((horizontal, np.cross(na, horizontal)))
    coords = [(xyz - ca) @ basis for xyz in (xa, xb)]
    hulls = [ConvexHull(uv) for uv in coords]
    result = {
        "pair": [a["id"], b["id"]],
        "camera_translation_m": float(np.linalg.norm(rel[:3, 3])),
        "camera_rotation_deg": float(np.degrees(Rotation.from_matrix(rel[:3, :3]).magnitude())),
        "wall_normal_angle_deg": float(np.degrees(np.arccos(np.clip(na @ nb, -1, 1)))),
    }
    for label, uv, other, distances, own_distances in (
        ("b_to_a", coords[1], hulls[0], (xb - ca) @ na, (xb - cb) @ nb),
        ("a_to_b", coords[0], hulls[1], (xa - cb) @ nb, (xa - ca) @ na),
    ):
        halfspaces = other.equations
        inside = (uv @ halfspaces[:, :2].T + halfspaces[:, 2] <= -.01).all(axis=1)
        result[label] = {
            "count": int(inside.sum()),
            "fraction": float(inside.mean()),
            "separation": residual_stats(distances[inside]) if inside.sum() >= 50 else None,
            "own_plane_same_points": residual_stats(own_distances[inside]) if inside.sum() >= 50 else None,
        }
    return result, [(coords[0][:, 0], (xa - ca) @ na), (coords[1][:, 0], (xb - ca) @ na)]


def edge_profile(z, y, x):
    offsets = np.arange(-28, 36)
    # Pool three nearby rows, requiring two observed samples per column.
    stack = z[y - 1:y + 2, x + offsets]
    profile = np.zeros(len(offsets))
    for j, column in enumerate(stack.T):
        observed = column[column > 0]
        if len(observed) >= 2:
            profile[j] = np.median(observed)
    near = profile[(offsets >= -14) & (offsets <= -5) & (profile > 0)]
    far = profile[(offsets >= 7) & (offsets <= 24) & (profile > 0)]
    result = {
        "rgb_xy": [x, y],
        "near_columns": len(near),
        "far_columns": len(far),
        "near_m": float(np.median(near)) if len(near) else None,
        "far_m": float(np.median(far)) if len(far) else None,
        "transition_relative_to_rgb_px": None,
    }
    if min(len(near), len(far)) >= 4 and result["far_m"] - result["near_m"] >= .10:
        lo = profile > 0
        hi = profile > 0
        lo &= abs(profile - result["near_m"]) <= .06
        hi &= abs(profile - result["far_m"]) <= .10
        # Require adjacent supporting columns on each side of the transition.
        near_ends = offsets[1:][lo[:-1] & lo[1:] & (offsets[1:] >= -10) & (offsets[1:] <= 10)]
        far_starts = offsets[:-1][hi[:-1] & hi[1:] & (offsets[:-1] >= -10) & (offsets[:-1] <= 24)]
        if len(near_ends) and np.any(far_starts > near_ends[-1]):
            result["transition_relative_to_rgb_px"] = [int(near_ends[-1]), int(far_starts[far_starts > near_ends[-1]][0])]
    return result, offsets, np.where(profile > 0, profile, np.nan)


def read_frame(db, record, number, cal, historical_K, maps, timing, output):
    box, edges = CASES[number]
    blobs = []
    for key in ("message_id", "depth_message_id"):
        blob, timestamp = db.execute("SELECT data,timestamp FROM messages WHERE id=?", (record[key],)).fetchone()
        timestamp_key = "timestamp_ns" if key == "message_id" else "depth_timestamp_ns"
        assert timestamp == record[timestamp_key]
        blobs.append(blob)
    rgb, _ = baseline.clean.capture.CDR(blobs[0]).image()
    rgb = cv2.remap(rgb, maps[:, :, 0], maps[:, :, 1], cv2.INTER_LINEAR)
    raw, _ = baseline.clean.adapter.decode_depth(blobs[1])
    y, x = np.nonzero(raw)
    native = p.deproject(np.column_stack((x, y)), raw[y, x] * cal["depth_units"], cal["raw_depth"])
    color = native @ np.array(cal["depth_to_color_R"]).T + np.array(cal["depth_to_color_t_m"])
    K = np.array(cal["K"])
    uv = color[:, :2] / color[:, 2:3] * [K[0, 0], K[1, 1]] + K[:2, 2]
    x0, y0, x1, y1 = box
    mask = (uv[:, 0] >= x0) & (uv[:, 0] < x1) & (uv[:, 1] >= y0) & (uv[:, 1] < y1)
    xyz = color[mask]
    assert len(xyz) >= 500
    center, normal = plane_fit(xyz)
    with np.load(baseline.BASE / "rgbd/depth_all" / (record["name"] + ".npz")) as cleaned:
        rays = (cleaned["points"] - historical_K[:2, 2]) / [historical_K[0, 0], historical_K[1, 1]]
        sdk_pixels = rays * [K[0, 0], K[1, 1]] + K[:2, 2]
        inside = ((sdk_pixels >= [x0, y0]) & (sdk_pixels < [x1, y1])).all(axis=1)
        cleaned_xyz = np.column_stack((rays[inside], np.ones(inside.sum()))) * cleaned["depths"][inside, None]
    assert len(cleaned_xyz) >= 100
    clean_center, clean_normal = plane_fit(cleaned_xyz)
    _, _, evidence = p.zbuffer(uv, color[:, 2], cal["width"], cal["height"])
    z = evidence["depth_m"]
    edge_results = [edge_profile(z, yy, xx) for yy, xx in edges]
    np.savez_compressed(output / f"frame_{number:03d}.npz", native_depth_raw=raw,
                        wall_xyz_depth_camera_m=native[mask], wall_xyz_color_camera_m=xyz,
                        source_depth_pixels_xy=np.column_stack((x[mask], y[mask])),
                        projected_unfilled_z_m=z)
    patch = z[y0:y1, x0:x1]
    stats = {
        "id": number, "name": record["name"], "rgb_message_id": record["message_id"],
        "depth_message_id": record["depth_message_id"], "rgb_timestamp_ns": record["timestamp_ns"],
        "depth_timestamp_ns": record["depth_timestamp_ns"], "rgb_roi_xyxy": box,
        "wall_points": len(xyz), "projected_fill_fraction": float((patch > 0).mean()),
        "median_native_depth_m": float(np.median(native[mask, 2])),
        "signed_exposure_midpoint_gap_ms": float(timing["signed_exposure_midpoint_depth_minus_rgb_ms"]),
        "rotation_during_gap_estimate_px": float(timing["estimated_rotation_during_sensor_gap_px"]),
        "plane_center_color_m": center.tolist(), "plane_normal_color": normal.tolist(),
        **residual_stats((xyz - center) @ normal),
        "historical_cleaned_samples": {
            "count": len(cleaned_xyz), **residual_stats((cleaned_xyz - clean_center) @ clean_normal),
        },
        "edges": [result for result, _, _ in edge_results],
    }
    return dict(id=number, rgb=rgb, xyz=xyz, plane=(center, normal), z=z, edges=edge_results, stats=stats)


def plots(frames, pair_plots, output):
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), layout="constrained")
    for row, f in enumerate(frames):
        n = f["id"]
        a, b, c = axes[row]
        x0, y0, x1, y1 = CASES[n][0]
        a.imshow(f["rgb"])
        a.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="yellow", lw=1.8))
        a.scatter([e[1] for e in CASES[n][1]], [e[0] for e in CASES[n][1]], s=13, c="magenta")
        a.set_title(f"Frame {n} · wall patch and jamb marks")
        a.axis("off")
        center, normal = f["plane"]
        delta = f["xyz"] - center
        b.scatter(delta[:, 0] * 100, delta @ normal * 1000, s=2, alpha=.23)
        b.set(xlabel="Camera X from patch center (cm)", ylabel="Plane residual (mm)")
        b.set_title("Native depth points")
        for edge, offsets, profile in f["edges"]:
            c.plot(offsets, profile, ".-", ms=3, lw=.8, label=f"row {edge['rgb_xy'][1]}")
        c.axvline(0, color="magenta", linestyle="--", lw=1)
        c.set(xlabel="Offset from RGB jamb (px)", ylabel="Projected depth Z (m)")
        c.legend(fontsize=8)
    fig.savefig(output / "01_raw_depth_and_rgb_edges.png", dpi=155)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharey=True, layout="constrained")
    for row, (label, pairs) in enumerate(pair_plots.items()):
        for ax, (result, points) in zip(axes[row], pairs):
            for n, (u, z) in zip(result["pair"], points):
                ax.scatter(u * 100, z * 1000, s=2, alpha=.27, label=f"Frame {n}")
            ax.set_title(f"{label} · {result['pair'][0]} vs {result['pair'][1]}")
            ax.set(xlabel="Across reference wall (cm)", ylabel="From reference plane (mm)")
            ax.legend(markerscale=3, loc="best")
    fig.savefig(output / "02_wall_overlap_saved_poses.png", dpi=155)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New output directory")
    output = parser.parse_args().output
    manifest = baseline.clean.check_source(baseline.BASE)
    records = {f["name"]: f for f in manifest["frames"]}
    sdk_records = {f["name"]: f for f in baseline.read(baseline.DATA / "capture_manifest.json")["frames"]}
    cal = baseline.read(baseline.DATA / "calibration_clean.json")
    historical_K = np.array(baseline.read(baseline.BASE / "calibration_clean.json")["K"])
    maps, valid = p.remap(cal["raw_color"], np.array(cal["K"]), cal["width"], cal["height"])
    assert valid.all()
    with (baseline.RUN / "sensor_timing_pairs.csv").open(newline="", encoding="utf-8") as stream:
        timing = {int(f["rgb_message_id"]): f for f in csv.DictReader(stream)}
    rgbd = baseline.BASE / "rgbd"
    rgbd_path = rgbd / "dataset/sparse"
    sfm_path = baseline.BASE / "reconstruction/0"
    anchor = next(f for f in baseline.read(rgbd / "anchor_report.json") if f["model"] == "0")
    assert anchor["passed"]
    pose_inputs = {"Saved RGB-D": (rgbd_path, 1.0), "Saved COLMAP": (sfm_path, anchor["scale"])}
    matrices = {}
    for label, (path, scale) in pose_inputs.items():
        images = {im.name: im for im in pycolmap.Reconstruction(path).images.values()}
        matrices[label] = {n: pose_matrix(images[f"frame_{n:06d}.png"], scale) for n in CASES}
    with np.load(rgbd / "poses.npz") as archive:
        for n in CASES:
            i = np.flatnonzero(archive["indices"] == n).item()
            np.testing.assert_allclose(matrices["Saved RGB-D"][n], archive["camtoworlds"][i], atol=1e-6)

    output.mkdir(parents=True, exist_ok=False)
    frames = []
    with closing(baseline.connect()) as db:
        for n in CASES:
            record = records[f"frame_{n:06d}.png"]
            pair_timing = timing[record["message_id"]]
            assert int(pair_timing["depth_message_id"]) == record["depth_message_id"]
            assert sdk_records[record["name"]]["depth_message_id"] == record["depth_message_id"]
            f = read_frame(db, record, n, cal, historical_K, maps, pair_timing, output)
            f["stats"]["seconds_since_first_selected_frame"] = (
                record["timestamp_ns"] - manifest["frames"][0]["timestamp_ns"]) / 1e9
            frames.append(f)

    pair_plots = {label: [compare(a, b, poses) for a, b in combinations(frames, 2)]
                  for label, poses in matrices.items()}
    report = {
        "source": str(baseline.DB), "source_bytes": baseline.DB.stat().st_size,
        "intrinsics_and_extrinsics": cal,
        "calibration_source": str(baseline.DATA / "calibration_clean.json"),
        "selection": "Three RGB-selected views of the matte wall below the illuminated panel.",
        "native_measurement": "DB3 nonzero samples projected into RGB only for patch membership; original 3D coordinates retained, no smoothing, interpolation, range filter, or filling.",
        "historical_cleaned_measurement": "Archived sampled RGB-D depth_all, mapped into the native test's rectified ROI by its calibrated RGB rays, with a separate plane fit.",
        "overlap_measurement": "Absolute point-to-plane distances within the opposing patch's projected hull, inset 1 cm; directions are evaluated separately.",
        "edge_measurement": "Three-row projected depth profiles, at least two observed samples per column. Relative transition endpoints bound stable near and far runs and may enclose missing or mixed pixels.",
        "sfm_scale_archive": anchor,
        "poses": {label: {
            "source": str(pose_inputs[label][0]),
            "images_bin_sha256": baseline.sha(pose_inputs[label][0] / "images.bin"),
            "camera_to_world_metric": {n: mat.tolist() for n, mat in poses.items()},
            "pairs": [pair for pair, _ in pair_plots[label]],
        } for label, poses in matrices.items()},
        "frames": [f["stats"] for f in frames],
    }
    baseline.write(output / "measurements.json", report)
    plots(frames, pair_plots, output)


if __name__ == "__main__":
    main()
