"""Prepare a RealSense ROS2 SQLite recording for the COLMAP gsplat loader.

The source is opened read-only. Depth remains in the original recording; this
first reconstruction uses RGB and estimates poses with COLMAP.
"""

import argparse
import io
import json
import sqlite3
import struct
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


class CDR:
    def __init__(self, blob):
        if blob[:4] != b"\x00\x01\x00\x00":
            raise ValueError("Expected little-endian CDR")
        self.blob, self.offset = blob, 4

    def uint32(self):
        self.offset = 4 + ((self.offset - 4 + 3) // 4) * 4
        value = struct.unpack_from("<I", self.blob, self.offset)[0]
        self.offset += 4
        return value

    def string(self):
        size = self.uint32()
        value = self.blob[self.offset:self.offset + size - 1].decode("utf-8")
        self.offset += size
        return value

    def image(self):
        sec, ns, frame = self.uint32(), self.uint32(), self.string()
        height, width, encoding = self.uint32(), self.uint32(), self.string()
        self.offset += 1  # is_bigendian, irrelevant for RGB8
        step, size = self.uint32(), self.uint32()
        if encoding != "rgb8" or size != step * height or self.offset + size > len(self.blob):
            raise ValueError("Unsupported or incomplete RGB message")
        array = np.frombuffer(self.blob, np.uint8, count=size, offset=self.offset)
        rgb = array.reshape(height, step)[:, :width * 3].reshape(height, width, 3)
        return rgb, {"header_seconds": sec + ns / 1e9, "frame_id": frame}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def extract(args):
    root = args.output.resolve()
    manifest_path = root / "capture_manifest.json"
    if manifest_path.exists():
        raise RuntimeError("Extraction already exists; use a different output directory")
    images = root / "raw_images"
    images.mkdir(parents=True, exist_ok=True)
    uri = args.recording.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        topics = {name: (tid, typ) for tid, name, typ in db.execute("SELECT id,name,type FROM topics")}
        color_name = next(n for n in topics if n.endswith("/Color_0/image/data"))
        tid = topics[color_name][0]
        calibration = {}
        for name, (topic_id, typ) in topics.items():
            if name.endswith(("/camera_info", "/info", "/Depth_Units/value")) and typ == "std_msgs/msg/String":
                row = db.execute("SELECT data FROM messages WHERE topic_id=? LIMIT 1", (topic_id,)).fetchone()
                if row:
                    calibration[name] = CDR(row[0]).string()
        first = db.execute("SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1", (tid,)).fetchone()[0]
        last = db.execute("SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp DESC LIMIT 1", (tid,)).fetchone()[0]
        save_json(root / "calibration.json", calibration)
        frames, previous_id = [], -1
        targets = np.arange(first / 1e9, last / 1e9 + 0.001, 1 / args.fps)
        for index, target in enumerate(targets):
            candidates = {}
            for offset in [-0.10, 0.0, 0.10]:
                timestamp = max(first, int((target + offset) * 1e9))
                row = db.execute("SELECT id,timestamp,data FROM messages WHERE timestamp>=? AND topic_id=? ORDER BY timestamp LIMIT 1", (timestamp, tid)).fetchone()
                if row and row[0] > previous_id:
                    rgb, header = CDR(row[2]).image()
                    gray = cv2.cvtColor(cv2.resize(rgb, (640, 360)), cv2.COLOR_RGB2GRAY)
                    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    candidates[row[0]] = (sharpness, row[1], rgb, header)
            if not candidates:
                continue
            mid = max(candidates, key=lambda k: candidates[k][0])
            sharpness, timestamp, rgb, header = candidates[mid]
            name = f"frame_{len(frames):06d}.jpg"
            Image.fromarray(rgb).save(images / name, quality=96, subsampling=0)
            frames.append({"name": name, "message_id": mid, "timestamp_ns": timestamp,
                           "time_seconds": timestamp / 1e9, "sharpness_640": sharpness,
                           "width": rgb.shape[1], "height": rgb.shape[0], **header})
            previous_id = mid
            if index % 50 == 0:
                print(f"Extracted {len(frames)}/{len(targets)} selected RGB frames", flush=True)
        stat = args.recording.stat()
        manifest = {"source": str(args.recording.resolve()), "source_bytes": stat.st_size,
                    "source_mtime_ns": stat.st_mtime_ns, "source_open_mode": "read-only",
                    "rgb_topic": color_name, "duration_seconds": (last-first)/1e9,
                    "target_fps": args.fps, "selection": "sharpest of three nearby samples per interval",
                    "depth_used": False, "frames": frames}
        save_json(manifest_path, manifest)
    sheet = Image.new("RGB", (1280, 800), "#222222")
    draw = ImageDraw.Draw(sheet)
    for j, i in enumerate(np.linspace(0, len(frames)-1, 16, dtype=int)):
        f = frames[i]
        with Image.open(images / f["name"]) as im:
            im.thumbnail((320, 180))
            x, y = (j % 4)*320, (j // 4)*200
            sheet.paste(im, (x, y+20))
            draw.text((x+4,y+3), f'{f["time_seconds"]:.1f}s | sharpness {f["sharpness_640"]:.0f}', fill="white")
    sheet.save(root / "capture_preview.jpg", quality=92)
    print(f"EXTRACTION_COMPLETE: {len(frames)} frames in {root}", flush=True)


def reconstruct(args):
    import pycolmap as p
    root = args.output.resolve()
    manifest = json.loads((root / "capture_manifest.json").read_text())
    calibration = json.loads((root / "calibration.json").read_text())
    info = next(v for k,v in calibration.items() if k.endswith("/Color_0/camera_info"))
    values = dict(field.split("=", 1) for field in info.split(";"))
    database = root / "colmap.db"
    reader = p.ImageReaderOptions()
    reader.camera_model = "OPENCV"
    # RealSense reports inverse distortion; do not copy it into a forward model.
    # Seed measured intrinsics and let COLMAP estimate forward distortion.
    reader.camera_params = ",".join([values[k] for k in ["fx","fy","ppx","ppy"]] + ["0"]*4)
    features = p.FeatureExtractionOptions()
    features.num_threads = args.threads
    features.max_image_size = 1280
    features.sift.max_num_features = 4096
    features.sift.first_octave = 0
    print("STAGE: extracting COLMAP features (CPU)", flush=True)
    p.extract_features(database, root/"raw_images", camera_mode=p.CameraMode.SINGLE,
                       reader_options=reader, extraction_options=features, device=p.Device.cpu)
    matching = p.FeatureMatchingOptions()
    matching.num_threads = args.threads
    pairs = p.SequentialPairingOptions()
    pairs.overlap = 12
    pairs.quadratic_overlap = True
    print("STAGE: sequential feature matching (CPU)", flush=True)
    p.match_sequential(database, matching_options=matching, pairing_options=pairs, device=p.Device.cpu)
    options = p.IncrementalPipelineOptions()
    options.num_threads = args.threads
    options.random_seed = 42
    options.max_num_models = 5
    options.min_model_size = 20
    sparse = root / "reconstruction"
    sparse.mkdir(exist_ok=True)
    print("STAGE: estimating camera poses", flush=True)
    models = p.incremental_mapping(database, root/"raw_images", sparse, options=options)
    report = {"selected_frames": len(manifest["frames"]), "depth_used": False, "models": []}
    for model_id, model in models.items():
        report["models"].append({"id": model_id, "registered_images": model.num_reg_images(),
                                 "points3D": model.num_points3D(),
                                 "mean_reprojection_error": model.compute_mean_reprojection_error()})
    save_json(root/"reconstruction_report.json", report)
    if not models:
        raise RuntimeError("COLMAP did not recover a scene; inspect overlap and camera motion")
    best_id = max(models, key=lambda i: models[i].num_reg_images())
    best = models[best_id]
    report["best_model"] = best_id
    report["registered_fraction"] = best.num_reg_images()/len(manifest["frames"])
    report["unregistered_images"] = sorted(set(f["name"] for f in manifest["frames"]) - {im.name for im in best.images.values()})
    registered = sorted(im.name for im in best.images.values())
    dataset = root / "dataset"
    dataset.mkdir(exist_ok=True)
    undistort = p.UndistortCameraOptions()
    undistort.max_image_size = args.train_width
    print(f"STAGE: preparing {best.num_reg_images()} registered images for training", flush=True)
    p.undistort_images(dataset, sparse/str(best_id), root/"raw_images", image_names=registered,
                       undistort_options=undistort, num_threads=args.threads)
    save_json(root/"reconstruction_report.json", report)
    print("RECONSTRUCTION_COMPLETE", json.dumps(report, default=str), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["extract", "reconstruct"])
    parser.add_argument("--recording", type=Path, default=Path("20261029_140739.db3"))
    parser.add_argument("--output", type=Path, default=Path("data/realsense_20261029_140739"))
    parser.add_argument("--fps", type=float, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--train-width", type=int, default=640)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    {"extract": extract, "reconstruct": reconstruct}[args.stage](args)


if __name__ == "__main__":
    main()
