"""Freeze the rescan split, audit timing and prepare source-image inspection."""

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from stray_scanner_reconstruct import columns, percentiles, read_csv


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def reserve_ranges(ids):
    return np.split(ids, np.where(np.diff(ids) != 1)[0] + 1)


def expression_sum(terms):
    if len(terms) == 1:
        return terms[0]
    half = len(terms) // 2
    return f"({expression_sum(terms[:half])}+{expression_sum(terms[half:])})"


def select_frames(video, expression, target, width, height, *, native=False):
    filters = f"select='{expression}'"
    if not native:
        filters += f",scale={width}:{height}:flags=area"
    filter_file = target.with_suffix(".filter")
    filter_file.write_text(filters, encoding="ascii")
    # The frozen newscan filter fits in a command-line argument, including on
    # FFmpeg builds that predate the slash syntax for loading files.
    filter_args = ["-vf", filters] if len(filters) <= 8192 else ["-/filter:v", str(filter_file)]
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-threads", "6", "-ignore_editlist", "1",
        "-noautorotate", "-i", str(video), "-map", "0:v:0", "-nostdin",
        *filter_args, "-filter_threads", "3", "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "rgb24", str(target),
    ], check=True)


def audit(run, source, width, height, cache):
    out = run / "dataset_audit"
    split_dir = run / "splits"
    split_dir.mkdir(exist_ok=True)
    rows = read_csv(source / "odometry.csv")
    time = columns(rows, "timestamp").ravel()
    time -= time[0]
    ids = np.array([int(row["frame"]) for row in rows])
    pts = np.loadtxt(out / "video_pts_unedited.csv")
    if (not np.array_equal(ids, np.arange(len(rows))) or len(pts) != len(ids)
            or np.any(np.diff(pts) <= 0) or np.any(np.diff(time) <= 0)):
        raise ValueError("RGB timestamp and odometry records do not match")
    scale, offset = np.polyfit(time, pts, 1)
    residual = (pts - (scale * time + offset)) * 1000
    counter_residual_ms = (pts - pts[0] - ids / 60) * 1000
    if np.max(np.abs(counter_residual_ms)) > .01:
        raise ValueError("RGB PTS does not follow the exported 60 Hz frame counter")
    phase = time % 4
    withheld = ids[(phase >= 1.6) & (phase < 2.4)]
    ranges = reserve_ranges(withheld)
    heldout = np.array([block[np.argmin(abs(phase[block] - 2))] for block in ranges])
    fit = ids[(ids % 6 == 0) & ~np.isin(ids, withheld)]
    split = {"fit": fit.tolist(), "withheld": withheld.tolist(), "heldout": heldout.tolist()}
    write_json(split_dir / "frame_split.json", split)
    write_json(split_dir / "train_frames.json", split["fit"])
    write_json(split_dir / "test_frames.json", split["heldout"])

    xyz = columns(rows, "x", "y", "z")
    quat = columns(rows, "qx", "qy", "qz", "qw")
    r = Rotation.from_quat(quat)
    trans = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    angle = np.degrees((r[:-1].inv() * r[1:]).magnitude())
    write_json(out / "preflight.json", {
        "frames": len(ids), "native_rgb_size": [width, height], "seconds": time[-1],
        "rgb_minus_odometry_ms": percentiles(residual), "clock_scale": scale,
        "rgb_frame_counter_residual_ms": percentiles(counter_residual_ms),
        "odometry_steps_over_25ms": int((np.diff(time) > .025).sum()),
        "clock_offset_s": offset, "frame_delta_ms": percentiles(np.diff(time) * 1000),
        "translation_step_mm": percentiles(trans * 1000), "rotation_step_deg": percentiles(angle),
        "largest_translation_frame": int(trans.argmax()) + 1,
        "largest_rotation_frame": int(angle.argmax()) + 1,
        "camera_path_m": float(trans.sum()), "reserved_fraction": len(withheld) / len(ids),
        "fit_count": len(fit), "heldout_count": len(heldout),
        "nearest_fit_time_gap_s": percentiles(np.abs(time[heldout, None] - time[fit]).min(1)),
    })
    print(f"Frozen split: {len(fit)} fit, {len(withheld)} reserved, {len(heldout)} scored.", flush=True)
    if cache:
        destination = out / "native_rgb.bin"
        blocked = expression_sum([f"between(n,{block[0]},{block[-1]})" for block in ranges])
        score = expression_sum([f"eq(n,{frame})" for frame in heldout])
        select_frames(source / "rgb.mp4", f"not(mod(n,6))*not({blocked})+{score}",
                      destination, width, height, native=True)
        write_json(out / "native_rgb_frames.json", sorted(np.r_[fit, heldout].tolist()))
        expected = (len(fit) + len(heldout)) * width * height * 3
        if destination.stat().st_size != expected:
            raise ValueError(f"Expected {expected} bytes for selected native RGB frames")
        return

    sample = np.union1d(heldout, ids[::60])
    target = out / "contact.rgb"
    scored = expression_sum([f"eq(n,{i})" for i in heldout])
    select_frames(source / "rgb.mp4", f"not(mod(n,60))+{scored}", target, 256, 192)
    frames = np.memmap(target, dtype=np.uint8, mode="r", shape=(len(sample), 192, 256, 3))
    for group, indices in (("heldout", heldout), ("route", ids[::60])):
        for page, batch in enumerate(np.array_split(indices, int(np.ceil(len(indices) / 36)))):
            sheet = Image.new("RGB", (6 * 192, int(np.ceil(len(batch) / 6)) * 162), "#172228")
            draw = ImageDraw.Draw(sheet)
            for i, frame in enumerate(batch):
                x, y = i % 6 * 192, i // 6 * 162
                image = Image.fromarray(frames[np.searchsorted(sample, frame)])
                sheet.paste(image.resize((192, 144), Image.Resampling.LANCZOS), (x, y))
                draw.text((x + 4, y + 145), f"{frame:06}   {time[frame]:.1f} s", fill="white")
            sheet.save(out / f"{group}_{page + 1}.jpg", quality=93)
    del frames
    print(f"Source contact sheets saved in {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--cache-native", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    video = json.loads((args.run / "dataset_audit/video.json").read_text(encoding="utf-8-sig"))
    stream = video["streams"][0]
    audit(args.run, args.source, stream["width"], stream["height"], args.cache_native)


if __name__ == "__main__":
    main()
