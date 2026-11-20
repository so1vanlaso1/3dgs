"""Same-wall range, native repeatability and frozen-pose filtering follow-up."""

import argparse
from contextlib import closing
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycolmap
from scipy.spatial import ConvexHull

import db3_wall_diagnostic as wall


# RGB-only panel-corner annotations: top-left, top-right, bottom-right, bottom-left.
# The last three panels are clipped on the left; extrapolation is based on the
# immediately preceding RGB views. Only the visible interior enters the ROI.
CORNERS = {
    52341: ((524, 148), (672, 137), (668, 218), (525, 213)),
    54421: ((468, 79), (648, 60), (644, 154), (470, 158)),
    60535: ((484, 42), (719, 58), (683, 171), (473, 142)),
    61185: ((457, 42), (679, 58), (646, 166), (446, 141)),
    62511: ((374, 25), (557, 25), (549, 125), (374, 119)),
    109767: ((397, 150), (455, 131), (453, 210), (396, 207)),
    110519: ((306, 127), (375, 108), (370, 200), (301, 196)),
    111041: ((255, 109), (334, 87), (331, 194), (254, 190)),
    111327: ((175, 102), (269, 79), (265, 198), (175, 193)),
    111431: ((129, 100), (232, 79), (230, 200), (129, 195)),
    111509: ((83, 98), (196, 75), (193, 202), (83, 198)),
    111587: ((40, 94), (164, 69), (163, 202), (40, 192)),
    111639: ((16, 86), (145, 58), (142, 198), (16, 189)),
    111689: ((-7, 77), (125, 49), (123, 198), (-7, 188)),
    111743: ((-26, 72), (110, 44), (108, 198), (-26, 185)),
    111821: ((-50, 66), (94, 33), (92, 194), (-50, 184)),
}
FIT = (52341, 54421, 62511)
PANEL_COORDS = np.float32([(0, 0), (1, 0), (1, 1), (0, 1)])
PATCH_COORDS = np.float32([(.48, 1.25), (.79, 1.25), (.79, 1.57), (.48, 1.57)])


def conservative_depth(raw, depth_unit):
    # Median of immediately compatible samples. Never fill a missing center or
    # average across a jump. Coverage changes can arise after RGB projection.
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(raw, 1), (3, 3))
    neighbors = windows.astype(np.float32).reshape(*raw.shape, 9)
    compatible = (neighbors > 0) & (abs(neighbors - raw[:, :, None]) <= .04 / depth_unit)
    count = compatible.sum(axis=2)
    ordered = np.sort(np.where(compatible, neighbors, np.inf), axis=2)
    selected = (raw > 0) & (count >= 5)
    result = raw.copy()
    median = np.take_along_axis(ordered, (count // 2)[:, :, None], axis=2)[:, :, 0]
    result[selected] = median[selected]
    np.testing.assert_array_equal(raw > 0, result > 0)
    return result


def region(corners, shape):
    homography = cv2.getPerspectiveTransform(PANEL_COORDS, np.float32(corners))
    polygon = cv2.perspectiveTransform(PATCH_COORDS[None], homography)[0]
    mask = np.zeros(shape, dtype="uint8")
    cv2.fillConvexPoly(mask, np.rint(polygon).astype("int32"), 1)
    return polygon, mask.astype(bool)


def surface(raw, cal, roi):
    y, x = np.nonzero(raw)
    depth = raw[y, x] * cal["depth_units"]
    native = wall.p.deproject(np.column_stack((x, y)), depth, cal["raw_depth"])
    points = native @ np.array(cal["depth_to_color_R"]).T + np.array(cal["depth_to_color_t_m"])
    K = np.array(cal["K"])
    pixels = points[:, :2] / points[:, 2:3] * [K[0, 0], K[1, 1]] + K[:2, 2]
    winner_p, winner_z, evidence = wall.p.zbuffer(pixels, points[:, 2], cal["width"], cal["height"])
    u, v = np.rint(pixels).astype(int).T
    h, w = roi.shape
    in_frame = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    use = in_frame & roi[np.clip(v, 0, h - 1), np.clip(u, 0, w - 1)]
    xyz = points[use]  # retain all unfilled native XYZ samples inside the common RGB ROI
    assert len(xyz) >= 100
    center, normal = wall.plane_fit(xyz)
    uu, vv = np.rint(winner_p).astype(int).T
    winner = roi[vv, uu]
    # Native rows/columns for local repeatability. Z-buffer winners for measuring
    # RGB-space coverage, including coverage of subregions with few valid points.
    observed = evidence["observed"] & roi
    ny, nx = np.nonzero(roi)
    bins = np.stack(((ny - ny.min()) * 4 // (np.ptp(ny) + 1),
                     (nx - nx.min()) * 4 // (np.ptp(nx) + 1)), axis=1)
    labels = bins[:, 0] * 4 + bins[:, 1]
    possible = np.bincount(labels, minlength=16)
    supported = np.bincount(labels[observed[ny, nx]], minlength=16)
    eligible = possible >= 10
    return dict(xyz=xyz, px=(y[use], x[use]), projected_z=evidence["depth_m"],
                stats=dict(point_count=len(xyz), roi_pixels=int(roi.sum()),
                           observed_pixels=int(observed.sum()), projected_valid_fraction=float(observed.sum() / roi.sum()),
                           roi_tiles_with_half_coverage=int((eligible & (supported >= possible / 2)).sum()),
                           roi_tiles=int(eligible.sum()), median_native_z_m=float(np.median(depth[use])),
                           native_z_p10_p90_m=np.percentile(depth[use], [10, 90]).tolist(),
                           median_camera_range_m=float(np.median(np.linalg.norm(native[use], axis=1))),
                           incidence_deg=float(np.degrees(np.arccos(np.clip(normal @ center / np.linalg.norm(center), -1, 1)))),
                           projected_z_median_m=float(np.median(winner_z[winner])),
                           **wall.residual_stats((xyz - center) @ normal)))


def nearest_depth(stamp, rows, stamps):
    j = int(np.searchsorted(stamps, stamp))
    return min(rows[max(0, j - 1):j + 1], key=lambda row: abs(row[1] - stamp))


def temporal(db, rgb_id, rgb_stamp, gray, roi, reference, read_rgb, read_depth,
             rgb_topic, depth_rows, depth_stamps, cal):
    # Gate native observations by RGB feature displacement, without motion
    # compensation. This is an image-motion proxy, not a static-rig certificate.
    mask = cv2.dilate(roi.astype("uint8"), np.ones((111, 111), dtype="uint8"))
    keypoints = cv2.goodFeaturesToTrack(gray, 120, .02, 6, mask=mask)
    scan = db.execute(
        "SELECT id,timestamp FROM messages WHERE topic_id=? AND timestamp BETWEEN ? AND ? ORDER BY timestamp",
        (rgb_topic, rgb_stamp - 180_000_000, rgb_stamp + 180_000_000),
    ).fetchall()
    y, x = reference["px"]
    accepted = []
    measurements = []
    stack = []
    for candidate, stamp in scan:
        depth_id, depth_stamp = nearest_depth(stamp, depth_rows, depth_stamps)
        if abs(depth_stamp - stamp) > 20_000_000 or keypoints is None:
            continue
        image = read_rgb(candidate, gray=True)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(gray, image, keypoints, None)
        back, backward, _ = cv2.calcOpticalFlowPyrLK(image, gray, next_points, None)
        tracked = (status[:, 0] > 0) & (backward[:, 0] > 0) & (np.linalg.norm((back - keypoints)[:, 0], axis=1) <= .5)
        if tracked.sum() < max(12, .70 * len(keypoints)):
            continue
        displacement = np.linalg.norm((next_points - keypoints)[tracked, 0], axis=1)
        p90 = float(np.percentile(displacement, 90))
        measurements.append(dict(rgb_id=candidate, depth_id=depth_id, rgb_stamp_ns=stamp, tracked=int(tracked.sum()),
                                 p90_rgb_motion_px=p90))
        if p90 <= 1.0 and depth_id not in accepted:
            accepted.append(depth_id)
            stack.append(read_depth(depth_id)[y, x])
    result = dict(reference_rgb_id=rgb_id, tracked_candidates=measurements, accepted_depth_ids=accepted,
                  threshold_p90_flow_px=1.0, pair_repeatability=None)
    if len(stack) >= 2:
        series = np.stack(stack)
        common = (series > 0).all(axis=0)
        if common.sum() >= 100:
            depth_m = series[:, common] * cal["depth_units"]
            spread = np.ptp(depth_m, axis=0)
            pixels = np.column_stack((x[common], y[common]))
            spatial_residuals = []
            for z in depth_m:
                xyz = wall.p.deproject(pixels, z, cal["raw_depth"])
                center, normal = wall.plane_fit(xyz)
                spatial_residuals.append((xyz - center) @ normal)
            diff = 1000 * (depth_m[-1] - depth_m[0])
            result["pair_repeatability"] = dict(frame_count=len(stack), common_native_pixels=int(common.sum()),
                                                common_native_fraction=float(common.mean()),
                                                p50_p90_pixel_range_mm=(1000 * np.percentile(spread, [50, 90])).tolist(),
                                                last_minus_first_median_mm=float(np.median(diff)),
                                                last_minus_first_p90_abs_centered_mm=float(np.percentile(abs(diff - np.median(diff)), 90)),
                                                first_last_spatial_residual_pearson=float(np.corrcoef(
                                                    spatial_residuals[0], spatial_residuals[-1])[0, 1]))
    return result


def evaluate_fusion(samples, poses, out):
    # Fit two separate wall planes after transforming exactly the same three
    # source frames with the same saved poses. Give each source equal weight.
    rng = np.random.default_rng(20261120)
    arms = ("raw", "filtered")
    source_pairs = {}
    for mid in FIT:
        # Use exactly the same native pixel IDs in both fitted arms. Points can
        # move just across the RGB mask after filtering, so fit on their common IDs.
        pixel_ids = [np.ravel_multi_index(samples[mid][arm]["px"], (480, 848)) for arm in arms]
        _, ri, fi = np.intersect1d(*pixel_ids, assume_unique=True, return_indices=True)
        source_pairs[mid] = (ri, fi)
    count = min(len(ri) for ri, _ in source_pairs.values())
    assert count >= 100
    selections = {mid: rng.choice(len(source_pairs[mid][0]), count, replace=False) for mid in FIT}
    clouds = {}
    planes = {}
    for j, arm in enumerate(arms):
        sources = [wall.transform(samples[mid][arm]["xyz"][source_pairs[mid][j][selections[mid]]], poses[mid])
                   for mid in FIT]
        clouds[arm] = np.vstack(sources)
        planes[arm] = wall.plane_fit(clouds[arm])

    center, normal = planes["raw"]
    _, _, axes = np.linalg.svd(clouds["raw"] - center, full_matrices=False)
    basis = axes[:2].T
    footprints = {arm: ConvexHull((clouds[arm] - center) @ basis) for arm in arms}
    report = []
    for mid, sample in samples.items():
        if mid in FIT or mid not in poses:
            continue
        xyz = wall.transform(sample["raw"]["xyz"], poses[mid])
        uv = (xyz - center) @ basis
        masks = {arm: (uv @ h.equations[:, :2].T + h.equations[:, 2] <= -.01).all(axis=1)
                 for arm, h in footprints.items()}
        common = masks["raw"] & masks["filtered"]
        distances = {arm: wall.residual_stats((xyz[common] - c) @ n)
                     for arm, (c, n) in planes.items()} if common.sum() >= 50 else None
        report.append(dict(rgb_id=mid, heldout_raw_points=len(xyz), common_points=int(common.sum()),
                           fraction={arm: float(masks[arm].mean()) for arm in arms},
                           common_fraction=float(common.mean()), common_residuals=distances))
    np.savez_compressed(out / "fused_wall_samples.npz", **clouds)
    return dict(fit_rgb_ids=FIT, points_sampled_per_frame=count,
                common_native_fit_points={mid: len(pair[0]) for mid, pair in source_pairs.items()},
                plane_models={arm: [c.tolist(), n.tolist()] for arm, (c, n) in planes.items()},
                heldout=report, third_arm="Skipped: no separately validated refined trajectory covers these fit and evaluation views.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    base = wall.baseline
    cal = base.read(base.DATA / "calibration_clean.json")
    maps, valid = wall.p.remap(cal["raw_color"], np.array(cal["K"]), cal["width"], cal["height"])
    assert valid.all()
    manifest = base.clean.check_source(base.BASE)["frames"]
    records = {r["message_id"]: r for r in manifest}
    model_path = base.BASE / "rgbd/dataset/sparse"
    model = pycolmap.Reconstruction(model_path)
    by_name = {im.name: im for im in model.images.values()}
    poses = {mid: wall.pose_matrix(by_name[records[mid]["name"]], 1.0) for mid in CORNERS if mid in records}
    samples, rows = {}, []
    fig, axes = plt.subplots(4, 4, figsize=(14, 8.5), layout="constrained")
    with closing(base.connect()) as db:
        rgb_topic = db.execute("SELECT topic_id FROM messages WHERE id=?", (FIT[0],)).fetchone()[0]
        depth_topic = db.execute("SELECT topic_id FROM messages WHERE id=?", (records[FIT[0]]["depth_message_id"],)).fetchone()[0]
        depth_rows = db.execute("SELECT id,timestamp FROM messages WHERE topic_id=? ORDER BY timestamp", (depth_topic,)).fetchall()
        depth_stamps = np.array([stamp for _, stamp in depth_rows])

        def read_rgb(mid, gray=False):
            blob = db.execute("SELECT data FROM messages WHERE id=?", (mid,)).fetchone()[0]
            rgb, _ = base.clean.capture.CDR(blob).image()
            if gray:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            return cv2.remap(rgb, maps[:, :, 0], maps[:, :, 1], cv2.INTER_LINEAR)

        def read_depth(mid):
            blob = db.execute("SELECT data FROM messages WHERE id=?", (mid,)).fetchone()[0]
            return base.clean.adapter.decode_depth(blob)[0]

        for (mid, corners), ax in zip(CORNERS.items(), axes.flat):
            stamp = db.execute("SELECT timestamp FROM messages WHERE id=?", (mid,)).fetchone()[0]
            did, depth_stamp = nearest_depth(stamp, depth_rows, depth_stamps)
            if mid in records:
                assert did == records[mid]["depth_message_id"]
            assert abs(depth_stamp - stamp) < 20_000_000
            rgb = read_rgb(mid)
            polygon, roi = region(corners, rgb.shape[:2])
            raw = read_depth(did)
            filtered = conservative_depth(raw, cal["depth_units"])
            samples[mid] = dict(raw=surface(raw, cal, roi), filtered=surface(filtered, cal, roi))
            stats = samples[mid]["raw"]["stats"]
            rows.append(dict(rgb_id=mid, depth_id=did, frame_name=records.get(mid, {}).get("name"),
                             seconds=(stamp - manifest[0]["timestamp_ns"]) / 1e9,
                             rgb_depth_playback_gap_ms=(depth_stamp - stamp) / 1e6,
                             roi_polygon=polygon.tolist(), raw=stats, filtered=samples[mid]["filtered"]["stats"],
                             temporal=temporal(db, mid, stamp, read_rgb(mid, gray=True), roi, samples[mid]["raw"],
                                               read_rgb, read_depth, rgb_topic, depth_rows, depth_stamps, cal)))
            ax.imshow(rgb)
            ax.add_patch(plt.Polygon(polygon, fill=False, ec="yellow", lw=1.4))
            ax.set_title(f"{mid} · camera Z {stats['median_native_z_m']:.2f} m", fontsize=9)
            ax.axis("off")
    fig.savefig(out / "wall_regions.png", dpi=130)
    plt.close(fig)
    result = evaluate_fusion(samples, poses, out)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, layout="constrained")
    range_m = [r["raw"]["median_camera_range_m"] for r in rows]
    angles = [r["raw"]["incidence_deg"] for r in rows]
    colors = plt.Normalize(0, 90)
    for ax, values, label in (
        (axes[0], [r["raw"]["p50_p90_p99_absolute_mm"][1] for r in rows], "Raw wall-plane deviation, p90 (mm)"),
        (axes[1], [100 * r["raw"]["projected_valid_fraction"] for r in rows], "Observed RGB-region pixels (%)"),
    ):
        scatter = ax.scatter(range_m, values, c=angles, norm=colors, cmap="viridis", s=45, edgecolors="#222222", linewidths=.5)
        ax.set(xlabel="Observed camera-to-point range (m)", ylabel=label)
        ax.grid(alpha=.2)
    axes[1].set_ylim(0, 108)
    fig.colorbar(scatter, ax=axes, label="Incidence from fitted depth plane (°)")
    fig.savefig(out / "range_and_coverage.png", dpi=155)
    plt.close(fig)
    base.write(out / "measurements.json", dict(
        source=str(base.DB), saved_pose_path=str(model_path), pose_hash=base.sha(model_path / "images.bin"),
        roi_panel_coordinates=PATCH_COORDS.tolist(), rgb_corners=CORNERS,
        filter="3×3 native median of at least five samples within 40 mm of a valid center; preserves zeros and unsupported centers.",
        range_limit="Observed native Z is not true range. Angle, location and path are confounded with distance.",
        reservation="Non-fit prepared frames are newly reserved for this comparison. The archived pose graph used historical observations.",
        frame_measurements=rows, fusion=result))


if __name__ == "__main__":
    main()
