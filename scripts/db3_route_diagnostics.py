"""Diagnose frozen R4 intervals and inspect difficult DB3 regions."""

import argparse
from collections import Counter
from contextlib import closing
import csv
from functools import lru_cache
from pathlib import Path
import sqlite3

import cv2
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from db3_measured_depth import CAL, IMU, RECORDING, TIMING, blob, db3, sensor_masks, world


ROOT = db3.ROOT
R4 = ROOT / "results/db3_r4_20261111_v2"
WALL = ROOT / "results/db3_wall_diagnostic_20261120/final"
MASK = ROOT / "results/db3_observation_masks_narrow_20261117"
WINDOW = ROOT / "results/db3_moving_holdout_20261111/target"
WINDOW_POSES = IMU / "local_v2/target/w2c.npz"
REGION_NAMES = ("passed", "failed_with_usable_evidence", "insufficient_or_invalid_evidence")
STATES = dict(zip(REGION_NAMES, ["#187758", "#c3442d", "#a98842"], strict=True))


def hull(uv, shape):
    return float(cv2.contourArea(cv2.convexHull(uv.astype("f4"))) / np.prod(shape)) if len(uv) >= 3 else 0.


def tiles(uv, shape):
    ij = (uv / [shape[1], shape[0]] * [8, 5]).astype(int)
    return len(np.unique(ij, axis=0))


def coverage(depth):
    valid, stable = sensor_masks(depth)
    kernel = np.ones((3, 3), dtype="u1")
    low, high = cv2.erode(depth, kernel), cv2.dilate(depth, kernel)
    full = valid & (low > .3)
    discontinuities = full & (high-low >= np.maximum(.03, .02*depth))
    return dict(valid_fraction=float(valid.mean()), stable_nonedge_fraction=float(stable.mean()),
                depth_jump_fraction=float(discontinuities.mean()),
                missing_neighbor_fraction=float((valid & ~full).mean()))


def matched(a, b, check):
    ia, ib = a["check"] == check, b["check"] == check
    links = db3.mutual(a["desc"][ia], b["desc"][ib])
    xy, uv = a["uv"][ia][links[:, 0]], b["uv"][ib][links[:, 1]]
    za, va, _ = db3.sample_depth(a["depth"], xy)
    zb, vb, _ = db3.sample_depth(b["depth"], uv)
    depth_valid = va & vb
    return xy, uv, za, zb, depth_valid


def scores(x, y, transform, k):
    q = world(x, transform)
    p = world(y, np.linalg.inv(transform))
    separation = np.linalg.norm(q-y, axis=1)
    front = (q[:, 2]>.01) & (p[:, 2]>.01)

    def pixel(pts):
        return pts[:, :2]/pts[:, 2:] * [k[0, 0], k[1, 1]] + k[:2, 2]

    error_px = np.maximum(np.linalg.norm(pixel(q[front])-pixel(y[front]), axis=1),
                          np.linalg.norm(pixel(p[front])-pixel(x[front]), axis=1))
    return dict(matches=len(separation), reprojection_matches=len(error_px),
                median_p90_m=np.percentile(separation, [50, 90]).tolist() if len(separation) else None,
                median_p90_px=np.percentile(error_px, [50, 90]).tolist() if len(error_px) else None,
                relative_translation_m=float(np.linalg.norm(transform[:3, 3])),
                relative_rotation_deg=float(Rotation.from_matrix(transform[:3, :3]).magnitude()*180/np.pi))


def native_reason(retry):
    if retry is None:
        return None
    if retry["accepted"]:
        return "native chain and endpoint passed"
    endpoint = retry["endpoint_check"]
    if endpoint is None:
        edge = retry["edges"][-1]
        return f"native chain stopped at link {len(retry['edges'])}/{retry['total_native_edges']}: {edge['reason']}"
    if endpoint["matches"] < 20:
        return "native chain completed with fewer than 20 endpoint checks"
    return "native chain completed with excessive endpoint 3D error"


def diagnose(edge, support, status, score, fit, check, timing, a, b):
    gate_reasons, evidence = [], []
    has_transform = "transform" in edge
    if not has_transform:
        gate_reasons.append("no robust direct fit")
    if not support:
        gate_reasons.append("direct fit hull or information gate failed")
    if check["stable_depth_matches"] < 20:
        gate_reasons.append("fewer than 20 frozen stable-depth checks")
    elif score["median_p90_m"][0] >= .05 or score["median_p90_m"][1] >= .12:
        gate_reasons.append("A0 3D check exceeds R4 tolerance")
    if status == REGION_NAMES[0]:
        return gate_reasons, evidence, "passes the A0 direct check"
    if fit["rgb_matches"] < 20 or min(fit["rgb_hull"]) <= .05:
        evidence.append("sparse or spatially concentrated raw RGB matching; tracking versus overlap is unresolved")
    if check["rgb_matches"] >= 20 and check["stable_depth_matches"] < 20:
        evidence.append("at least 20 RGB check matches but the frozen depth gate leaves fewer than 20")
    if min(a["valid_fraction"], b["valid_fraction"]) < .5:
        evidence.append("less than half of at least one R4 aligned depth image has measured raster support")
    if max(row["estimated_rotation_during_sensor_gap_px"] for row in timing) >= 1.:
        evidence.append("estimated rotation across RGB-depth exposure gap exceeds 1 pixel for an endpoint")
    if score["relative_rotation_deg"] >= 20:
        evidence.append("A0 rotation exceeds 20 degrees between endpoints")
    if edge["accepted"] and status == REGION_NAMES[1]:
        evidence.append("local direct transform passes the frozen checks but A0 relative transform fails")
    if edge["boundary"] == 362:
        evidence.append("display window has visible glare and mixed surface depths; see targeted diagnostic")
    return gate_reasons, evidence, "unresolved"


def route(out, cal):
    k = np.array(cal["K"])
    source = db3.read(db3.INPUT / "capture_manifest.json")["frames"]
    archived = db3.read(R4 / "selected_edges.json")
    retries = {row["boundary"]: row for row in db3.read(R4 / "native_retries.json")["retries"]}
    ref = db3.read(IMU / "route_evaluation/route_checks.json")
    frozen = {row["boundary"]: row for row in ref["adjacent"]}
    manifest = {row["source_id"]: row for row in db3.read(R4 / "inputs_manifest.json")["frames"]}
    with np.load(IMU / "route_v2/full_route_camtoworlds.npz") as data:
        poses = {key: data[key] for key in data.files}
    keys = ("signed_exposure_midpoint_depth_minus_rgb_ms", "nearest_gyro_rad_s", "estimated_rotation_during_sensor_gap_px")
    with TIMING.open(encoding="utf-8", newline="") as f:
        timing = {int(row["rgb_message_id"]): {key: float(row[key]) for key in keys} for row in csv.DictReader(f)}
    assert len(frozen) == 317 and len(archived["excluded_boundaries"]) == 159 and len(source) == 477

    @lru_cache(maxsize=6)
    def load(mid):
        path = R4 / "inputs" / f"{mid}.npz"
        assert db3.sha(path) == manifest[mid]["sha256"]
        with np.load(path) as data:
            return {key: data[key] for key in data.files}, coverage(data["depth"])

    rows = []
    for edge in archived["edges"]:
        i = edge["boundary"]
        a_id, b_id = edge["source_ids"]
        assert [a_id, b_id] == [x["message_id"] for x in source[i:i+2]]
        (a, cov_a), (b, cov_b) = load(a_id), load(b_id)
        shape = a["depth"].shape
        match_stats = {}
        for role in ("fit", "check"):
            xy, uv, za, zb, valid = matched(a, b, role == "check")
            match_stats[role] = dict(rgb_matches=len(xy), rgb_hull=[hull(xy, shape), hull(uv, shape)],
                                     rgb_tiles=[tiles(xy, shape), tiles(uv, shape)],
                                     stable_depth_matches=int(valid.sum()),
                                     stable_depth_match_fraction=float(valid.mean()) if len(valid) else None)
            if role == "check":
                x, y = db3.backproject(xy[valid], za[valid], k), db3.backproject(uv[valid], zb[valid], k)
        assert len(x) == edge["check_matches"]
        arm_stats = {}
        ref_row = frozen[i]
        support = ref_row["same_direct_fit_support"]
        for name, matrices in poses.items():
            transform = np.linalg.solve(matrices[i+1], matrices[i])
            arm = scores(x, y, transform, k)
            reference = ref_row["arms"][name]
            for key in ("median_p90_m", "median_p90_px"):
                if arm[key] is not None:
                    np.testing.assert_allclose(arm[key], reference[key], rtol=1e-8, atol=1e-9)
            arm["state"] = REGION_NAMES[0] if support and reference["endpoint_pass"] else (
                REGION_NAMES[1] if support and len(x) >= 20 else REGION_NAMES[2])
            arm_stats[name] = arm
        control = arm_stats["A0"]
        timing_rows = [timing[mid] for mid in (a_id, b_id)]
        gate, evidence, cause = diagnose(edge, support, control["state"], control, match_stats["fit"],
                                         match_stats["check"], timing_rows, cov_a, cov_b)
        retry = retries.get(i)
        local_score = scores(x, y, np.array(edge["transform"]), k) if "transform" in edge else None
        rows.append(dict(boundary=i, rgb_ids=[a_id, b_id], frames=[f["name"] for f in source[i:i+2]],
                         depth_ids=[manifest[mid]["depth_source_id"] for mid in (a_id, b_id)],
                         seconds=edge["seconds"], state=control["state"], cause=cause,
                         gate_failures=gate, observed_flags=evidence, native_retry=native_reason(retry),
                         original_r4_pass=ref_row["archived_pass"], direct_fit_support=support,
                         original_local_score=local_score,
                         original_fit_inlier_ratio=edge.get("inliers", 0)/edge["fit_matches"] if local_score else None,
                         original_fit_inliers=edge.get("inliers"), original_fit_inlier_hull=edge.get("fit_hull_fraction"),
                         original_information_min_eigenvalue=edge.get("information_eigenvalues", [None])[0],
                         original_information_condition=edge.get("information_condition"),
                         rgbd_matches=match_stats, depth_coverage=[cov_a, cov_b], timing=timing_rows,
                         static_surface_coverage=None, arms=arm_stats))
    totals = {arm: dict(Counter(row["arms"][arm]["state"] for row in rows)) for arm in poses}
    blocks = {}
    for row in rows:
        block = int(row["seconds"][0]//20)*20
        blocks.setdefault(block, Counter())[row["state"]] += 1
    unresolved = [row for row in rows if row["cause"] == "unresolved"]
    db3.write(out / "frame_pair_error_map.json", dict(
        selected_cameras=len(source), eligible=len(rows), states=totals,
        original_r4_passes=sum(row["original_r4_pass"] for row in rows),
        differences={"R4_pass_A0_fails": [r["boundary"] for r in rows if r["original_r4_pass"] and r["state"] != "passed"],
                     "A0_pass_R4_fails": [r["boundary"] for r in rows if not r["original_r4_pass"] and r["state"] == "passed"]},
        unresolved_count=len(unresolved), observed_flags=dict(Counter(flag for row in unresolved for flag in row["observed_flags"])),
        unresolved_without_flags=sum(not row["observed_flags"] for row in unresolved),
        refinement_candidates=[row["boundary"] for row in unresolved if row["original_r4_pass"]
                               and row["state"] == REGION_NAMES[1]],
        twenty_second_blocks=blocks, excluded=archived["excluded_boundaries"], rows=rows))
    write_csv(out, rows)
    trajectory(out, rows, poses["A0"], archived["excluded_boundaries"])
    print("ROUTE", totals, dict(blocks), flush=True)


def write_csv(out, rows):
    flat = []
    for row in rows:
        output = {key: row[key] for key in ("boundary", "state", "cause", "native_retry", "original_r4_pass",
                  "original_fit_inlier_ratio", "original_fit_inliers", "original_fit_inlier_hull")}
        output["gate_failures"] = " | ".join(row["gate_failures"])
        output["observed_flags"] = " | ".join(row["observed_flags"])
        for index, side in enumerate(("source", "target")):
            for key in ("frames", "rgb_ids", "depth_ids", "seconds"):
                output[f"{side}_{key}"] = row[key][index]
            for data in ("depth_coverage", "timing"):
                output.update({f"{side}_{key}": value for key, value in row[data][index].items()})
        for role, values in row["rgbd_matches"].items():
            output.update({f"{role}_{key}": val for key, val in values.items()})
        for arm, summary in row["arms"].items():
            for key, value in summary.items():
                if key in ("median_p90_m", "median_p90_px"):
                    median, p90 = value or (None, None)
                    output.update({f"{arm}_median_{key.split('_')[-1]}": median, f"{arm}_p90_{key.split('_')[-1]}": p90})
                else:
                    output[f"{arm}_{key}"] = value
        flat.append(output)
    with (out / "frame_pair_error_map.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def trajectory(out, rows, poses, excluded):
    # Use the frozen route graph's world convention. Negative Y is camera-up
    # near the start. This chart uses the same X-Z projection as the IMU audit.
    xy = poses[:, [0, 2], 3]
    segments = np.stack((xy[:-1], xy[1:]), axis=1)
    fig, (ax, timeline) = plt.subplots(1, 2, figsize=(13, 6), width_ratios=(1.2, 1), layout="constrained")
    for state, color in {"reserved": "#c5cbd2", **STATES}.items():
        if state == "reserved":
            ids = [row["boundary"] for row in excluded]
            secs = [row["seconds"][0] for row in excluded]
        else:
            ids = [row["boundary"] for row in rows if row["state"] == state]
            secs = [row["seconds"][0] for row in rows if row["state"] == state]
        if ids:
            ax.add_collection(LineCollection(segments[ids], colors=color, linewidths=1.55))
            timeline.scatter(secs, ids, s=7, c=color, label=state.replace("_", " ").capitalize())
    for text, i in (("Start", 0), ("Wall 200", 200), ("Wall 240", 240), ("Display", 362), ("End", 476)):
        ax.plot(*xy[i], "o", markersize=3, color="#15233a")
        ax.annotate(text, xy[i], xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.autoscale()
    ax.set(aspect="equal", xlabel="Archived X (m)", ylabel="Archived Z (m)", title="A0 camera route")
    timeline.set(xlabel="Capture time (s)", ylabel="Selected interval index", title="Direct gate by capture time")
    timeline.legend(loc="lower right", markerscale=1.6)
    fig.savefig(out / "trajectory_failure_map.png", dpi=190)
    plt.close(fig)


def region_plots(out, cal):
    k = np.array(cal["K"])
    report = db3.read(WALL / "measurements.json")
    wall_checks = db3.read(IMU / "wall_same_points.json")
    sources = db3.read(WINDOW / "inputs.json")
    index = [0, 7, 14]
    with np.load(MASK / "target/masks.npz") as data:
        reflections = data["reflection_glare"]
        static = data["static_candidate"]
    with np.load(WINDOW_POSES) as data:
        a0, c = np.linalg.inv(data["A0"]), np.linalg.inv(data["C"])
        assert data["source_ids"][index].tolist() == [sources[i]["source_id"] for i in index]
    prepared = []
    with closing(sqlite3.connect(RECORDING.as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        for i in index:
            row = sources[i]
            mid = row["depth_source_id"]
            stamp = db.execute("SELECT timestamp FROM messages WHERE id=?", (mid,)).fetchone()[0]
            raw, _ = db3.clean.adapter.decode_depth(blob(db, mid, stamp))
            _, _, _, evidence = db3.projection.align_depth(raw, cal["depth_units"], cal["raw_depth"],
                 cal["depth_to_color_R"], cal["depth_to_color_t_m"], k, cal["width"], cal["height"], filtered=False)
            image_path = WINDOW / "inputs" / f"{row['source_id']}.png"
            assert db3.sha(image_path) == row["image_sha256"]
            prepared.append((row, np.asarray(Image.open(image_path).convert("RGB")), evidence["depth_m"]))

    def image_stats(z, mask):
        selected = int(mask.sum())
        if not selected:
            return None
        valid, stable = sensor_masks(z)
        return dict(roi_pixels=selected, valid_fraction=float(valid[mask].mean()),
                    stable_nonedge_fraction=float(stable[mask].mean()),
                    depth_percentiles_m=np.percentile(z[mask & valid], [10, 50, 90]).tolist()
                    if (mask & valid).any() else None)

    fig, axes = plt.subplots(2, 3, figsize=(11, 4.8), layout="constrained")
    display = []
    for col, (idx, (row, rgb, z)) in enumerate(zip(index, prepared, strict=True)):
        glare = reflections[idx].astype(bool)
        opaque = static[idx].astype(bool) & ~glare
        ax = axes[0, col]
        ax.imshow(rgb)
        if glare.any():
            ax.contour(glare, levels=[.5], colors=["#ffbb3b"], linewidths=.85)
        if opaque.any():
            ax.contour(opaque, levels=[.5], colors=["#45dbd3"], linewidths=.85)
        ax.set_title(f"RGB {row['source_id']}")
        artist = axes[1, col].imshow(np.ma.masked_equal(z, 0), vmin=.6, vmax=4)
        display.append(dict(rgb_id=row["source_id"], depth_id=row["depth_source_id"],
                            glare=image_stats(z, glare), annotated_static_candidates=image_stats(z, opaque),
                            other=image_stats(z, ~glare & ~opaque)))
    for ax in axes.flat:
        ax.axis("off")
    fig.colorbar(artist, ax=axes[1], label="Aligned axial depth (m)")
    fig.savefig(out / "display_rgb_and_depth.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained", sharex="row", sharey="row")
    for col, (arm, poses) in enumerate((("A0", a0), ("C", c))):
        for j, (i, (row, _, z)) in enumerate(zip(index, prepared, strict=True)):
            y, x = np.nonzero(z)
            x, y = x[::19], y[::19]
            points = db3.backproject(np.c_[x, y], z[y, x], k)
            xyz = world(points, np.linalg.solve(poses[0], poses[i]))
            for ax in axes[:, col]:
                ax.scatter(xyz[:, 0], xyz[:, 2], s=.8, alpha=.4, color=["#5855a5", "#d47e28", "#00857d"][j],
                           label=str(row["source_id"]))
        axes[0, col].set(title=f"{arm} aligned display clouds", aspect="equal")
        axes[1, col].set(aspect="equal", xlim=(-1.5, 1.4), ylim=(.75, 1.65),
                         xlabel="Reference camera X (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Reference camera Z (m)")
    axes[0, 1].legend(markerscale=4)
    fig.savefig(out / "display_registered_clouds.png", dpi=180)
    plt.close(fig)

    planes = report["frames"]
    with np.load(IMU / "route_v2/full_route_camtoworlds.npz") as data:
        arms = {key: data[key] for key in ("A0", "C")}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained", sharex=True, sharey=True)
    n = np.array(planes[0]["plane_normal_color"])
    tangent = np.cross([0, 1, 0], n)
    tangent /= np.linalg.norm(tangent)
    for col, (arm, poses) in enumerate(arms.items()):
        for frame, color in zip(planes, ["#5855a5", "#d47e28", "#00857d"], strict=True):
            number = frame["id"]
            path = WALL / f"frame_{number:03d}.npz"
            assert db3.sha(path) == wall_checks["inputs"][str(path.relative_to(ROOT))]
            with np.load(path) as data:
                xyz = world(data["wall_xyz_color_camera_m"], np.linalg.solve(poses[200], poses[number]))
            xyz -= planes[0]["plane_center_color_m"]
            axes[col].scatter(xyz[:, None, :] @ tangent, xyz[:, None, :] @ n, s=1, alpha=.30, c=color, label=str(number))
        axes[col].set(title=f"{arm} registered wall patches", xlabel="Along frame 200 wall plane (m)", aspect="equal")
    axes[0].set_ylabel("Normal to frame 200 wall plane (m)")
    axes[1].legend(markerscale=4)
    fig.savefig(out / "wall_registered_clouds.png", dpi=180)
    plt.close(fig)
    db3.write(out / "regions.json", dict(
        display={"boundary": 362, "frames": display, "mask_labels": "Amber glare; teal visually static candidates.",
                 "withheld": db3.read(IMU / "local_v2/target/metrics.json"),
                 "interpretation": "Glare and multi-surface depth are visible. Opaque corner identity and motion or calibration contributions are unresolved."},
        wall={"frames": planes, "matched_overlap": wall_checks,
              "individual_rgb_and_depth": str((WALL / "01_raw_depth_and_rgb_edges.png").relative_to(ROOT)),
              "interpretation": "Single-frame metric wall patches and fixed-cloud overlay isolate a cross-view component. Pose error, calibration contributions and wall shape are not independently separated."}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    out = parser.parse_args().output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("db3_measured_depth.py"), CAL, TIMING,
             db3.INPUT / "capture_manifest.json", R4 / "selected_edges.json", R4 / "native_retries.json",
             R4 / "inputs_manifest.json", IMU / "route_v2/full_route_camtoworlds.npz",
             IMU / "route_evaluation/route_checks.json", WINDOW / "inputs.json", WINDOW_POSES,
             IMU / "wall_same_points.json", WALL / "measurements.json", MASK / "target/masks.npz"]
    db3.write(out / "protocol.json", dict(
        inputs={str(p.relative_to(ROOT)): db3.sha(p) for p in paths},
        primary="Archived trajectory A0, frozen R4 feature rosters and masks. Other IMU arms rescored on identical checks.",
        coverage="Depth fractions use archived filtered R4 rasters. Stable non-edge depth is a support proxy. "
                 "Physically static surface coverage is unavailable route-wide.",
        failure="Usable if at least 20 check matches and original R4 fit hull/information pass. "
                "Motion and support flags are evidence, with causal status unresolved unless independently identified.",
        reprojection="Max bidirectional 2D euclidean error after depth back-projection; "
                     "R4 acceptance is 3D separation and support, independent of this diagnostic number."))
    cv2.setNumThreads(4)
    cal = db3.read(CAL)
    route(out, cal)
    region_plots(out, cal)


if __name__ == "__main__":
    main()
