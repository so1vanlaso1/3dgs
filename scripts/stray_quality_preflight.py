"""Validate a decoded Stray Scanner capture and freeze its training views."""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from stray_scanner_reconstruct import columns, percentiles, read_csv


SIZE = (512, 384)
LAGS = np.arange(-9, 10)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def track_errors(a, b, frame, step, source, xyz, rotations, calibration):
    start = cv2.goodFeaturesToTrack(a, 500, .012, 7)
    if start is None:
        return np.empty((len(LAGS), 0))
    end, status, _ = cv2.calcOpticalFlowPyrLK(a, b, start, None)
    back, backward, _ = cv2.calcOpticalFlowPyrLK(b, a, end, None)
    keep = (status.ravel() > 0) & (backward.ravel() > 0)
    keep &= np.linalg.norm((back - start)[:, 0], axis=1) < .65
    start, end = start[keep, 0], end[keep, 0]
    x = np.clip(np.round((start[:, 0] + .5) * 256 / SIZE[0] - .5).astype(int), 0, 255)
    y = np.clip(np.round((start[:, 1] + .5) * 192 / SIZE[1] - .5).astype(int), 0, 191)
    valid = np.ones(len(start), dtype=bool)
    errors = []
    for lag in LAGS:
        i, j = frame + lag, frame + step + lag
        depth = cv2.imread(str(source / "depth" / f"{i:06}.png"), cv2.IMREAD_UNCHANGED)
        confidence = cv2.imread(str(source / "confidence" / f"{i:06}.png"), cv2.IMREAD_UNCHANGED)
        z = depth[y, x] * .001
        fx, fy, cx, cy = calibration[i]
        points = np.column_stack(((start[:, 0] - cx) * z / fx,
                                  (start[:, 1] - cy) * z / fy, z))
        other = (points @ rotations[i].T + xyz[i] - xyz[j]) @ rotations[j]
        fx, fy, cx, cy = calibration[j]
        with np.errstate(invalid="ignore", divide="ignore"):
            target = other[:, :2] / other[:, 2:] * [fx, fy] + [cx, cy]
        errors.append(np.linalg.norm(target - end, axis=1))
        valid &= (confidence[y, x] == 2) & (z >= .25) & (z <= 5) & (other[:, 2] > 0)
    return np.asarray(errors)[:, valid]


def contact_sheets(run, proxy, frames, times, label):
    for page, batch in enumerate(np.array_split(frames, int(np.ceil(len(frames) / 30))), 1):
        sheet = Image.new("RGB", (1200, 175 * int(np.ceil(len(batch) / 6))), "#172228")
        draw = ImageDraw.Draw(sheet)
        for i, frame in enumerate(batch):
            x, y = i % 6 * 200, i // 6 * 175
            sheet.paste(Image.fromarray(proxy[frame]).resize((200, 150)), (x, y))
            draw.text((x + 4, y + 151), f"{frame:06}   {times[frame]:.1f} s", fill="white")
        sheet.save(run / "dataset_audit" / f"{label}_{page:02}.jpg", quality=91)


def audit(run, source):
    out = run / "dataset_audit"
    video = json.loads((out / "video.json").read_text())["streams"][0]
    rows = read_csv(source / "odometry.csv")
    ids = np.arange(len(rows))
    if not np.array_equal(ids, [int(row["frame"]) for row in rows]):
        raise ValueError("Odometry IDs are not consecutive")
    times = columns(rows, "timestamp").ravel()
    times -= times[0]
    expected = {f"{i:06}.png" for i in ids}
    for name in ("depth", "confidence"):
        if {p.name for p in (source / name).glob("*.png")} != expected:
            raise ValueError(f"{name} frame set differs from odometry")
    proxy_path = out / "proxy_gray.bin"
    if proxy_path.stat().st_size != len(ids) * SIZE[0] * SIZE[1]:
        raise ValueError("Decoded grayscale RGB frame count differs from odometry")
    proxy = np.memmap(proxy_path, mode="r", dtype=np.uint8, shape=(len(ids), SIZE[1], SIZE[0]))
    pts = np.sort(np.array([float(value) for value in subprocess.check_output([
        "ffprobe", "-v", "error", "-ignore_editlist", "1", "-select_streams", "v:0",
        "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(source / "rgb.mp4")
    ], text=True).splitlines() if value]))
    if len(pts) != len(ids) or np.any(np.diff(pts) <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError("Unexpected video or odometry count or clock ordering")
    np.savetxt(out / "video_pts_unedited.csv", pts, fmt="%.6f")
    ticks = np.rint((pts - pts[0]) * 60).astype(int)
    gap_frames = np.flatnonzero(np.diff(ticks) != 1) + 1
    xyz = columns(rows, "x", "y", "z")
    poses = Rotation.from_quat(columns(rows, "qx", "qy", "qz", "qw"))
    rotations = poses.as_matrix()
    k = columns(rows, "fx", "fy", "cx", "cy") * ([SIZE[0] / video["width"], SIZE[1] / video["height"]] * 2)
    translation = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    angle = np.degrees((poses[:-1].inv() * poses[1:]).magnitude())
    extremes = np.unique(np.r_[np.argsort(translation)[-12:] + 1, np.argsort(angle)[-12:] + 1])
    boundaries = [0, int(gap_frames[0]), int(gap_frames[-1]) + 1, len(ids)] if len(gap_frames) else [0, len(ids)]
    bands = [(f"route_{i + 1}", start, end) for i, (start, end) in enumerate(zip(boundaries, boundaries[1:]))]
    bands.extend((f"route_section_{i + 1}", int(start), int(end)) for i, (start, end) in enumerate(
        zip(np.linspace(0, len(ids), 9, dtype=int)[:-1], np.linspace(0, len(ids), 9, dtype=int)[1:])
    ))
    report = {}
    for name, start, end in bands:
        anchors = np.unique(np.linspace(max(start, 10), min(end - 7, len(ids) - 17), 32, dtype=int))
        errors = [track_errors(proxy[i], proxy[i + 6], i, 6, source, xyz, rotations, k) for i in anchors]
        pooled = np.concatenate(errors, axis=1)
        if not pooled.shape[1]:
            raise ValueError(f"No forward-backward LiDAR-backed tracks in {name}")
        report[name] = {
            "ordinals": [start, end], "track_count": pooled.shape[1],
            "median_error_px": {str(lag): float(error) for lag, error in zip(LAGS, np.median(pooled, axis=1))},
        }
    jump_report = []
    for i in extremes:
        residuals = track_errors(proxy[i-1], proxy[i], i-1, 1, source, xyz, rotations, k)[len(LAGS)//2]
        jump_report.append({
            "frame": int(i), "capture_s": float(times[i]), "translation_m": float(translation[i-1]),
            "rotation_deg": float(angle[i-1]), "track_count": len(residuals),
            "reprojection_error_px": percentiles(residuals),
        })
    write_json(out / "alignment.json", {"bands": report, "pose_jumps": jump_report, "rgb_gap_frames": gap_frames.tolist()})
    write_json(out / "preflight.json", {
        "native_rgb_size": [video["width"], video["height"]], "frames": len(ids),
        "decoded_frames": len(proxy), "seconds": float(times[-1]), "translation_mm": percentiles(translation * 1000),
        "rotation_deg": percentiles(angle), "camera_path_m": float(translation.sum()), "rgb_gap_frames": gap_frames.tolist(),
    })
    contact_sheets(run, proxy, np.arange(0, len(ids), 180), times, "route")
    jump_views = np.unique(np.clip(np.r_[extremes - 4, extremes, extremes + 4], 0, len(ids) - 1))
    contact_sheets(run, proxy, jump_views, times, "tracking")
    print(json.dumps({name: (min(band["median_error_px"], key=band["median_error_px"].get),
                            band["median_error_px"]["0"], band["track_count"]) for name, band in report.items()}), flush=True)


def freeze_split(run, source, excluded_file):
    out = run / "dataset_audit"
    split = run / "splits"
    rows = read_csv(source / "odometry.csv")
    times = columns(rows, "timestamp").ravel()
    times -= times[0]
    proxy = np.memmap(out / "proxy_gray.bin", mode="r", dtype=np.uint8, shape=(len(times), SIZE[1], SIZE[0]))
    excluded = np.zeros(len(times), dtype=bool)
    if excluded_file:
        for a, b in json.loads(excluded_file.read_text()):
            excluded[a:b + 1] = True
    # Select sharp frames locally so textured parts of the route cannot dominate less textured surfaces.
    sharpness = np.empty(len(times), dtype=np.float32)
    for i, image in enumerate(proxy):
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
        fine = cv2.Laplacian(image, cv2.CV_32F)
        sharpness[i] = np.mean(np.abs(fine)) / (np.mean(np.abs(gx) + np.abs(gy)) + .25)
    # Disjoint temporal blocks stay out of both TSDF fusion and training.
    phase = times % 6
    withheld = (phase >= 2.65) & (phase <= 3.35)
    fit, heldout = [], []
    for group_ids, groups, output in ((np.flatnonzero(~withheld & ~excluded), times // .152, fit),
                                      (np.flatnonzero((phase >= 2.88) & (phase <= 3.12) & ~excluded), times // 6, heldout)):
        for group in np.split(group_ids, np.flatnonzero(np.diff(groups[group_ids])) + 1):
            if group.size:
                output.append(int(group[np.argmax(sharpness[group])]))
    frozen = {"fit": fit, "heldout": heldout, "withheld": np.flatnonzero(withheld | excluded).tolist()}
    write_json(split / "frame_split.json", frozen)
    write_json(split / "train_frames.json", fit)
    write_json(split / "test_frames.json", heldout)
    np.save(out / "sharpness.npy", sharpness)
    write_json(out / "keyframes.json", {
        "fit": len(fit), "heldout": len(heldout), "withheld": int(withheld.sum()),
        "tracking_rejects": int(excluded.sum()), "fit_sharpness": percentiles(sharpness[fit]),
        "all_sharpness": percentiles(sharpness),
    })
    contact_sheets(run, proxy, np.array(heldout), times, "heldout")
    print(f"Frozen {len(fit)} training keyframes and {len(heldout)} held-out views.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--exclude", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.freeze:
        freeze_split(args.run, args.source, args.exclude)
    else:
        audit(args.run, args.source)
