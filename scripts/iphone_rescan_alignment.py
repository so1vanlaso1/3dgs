"""Corroborate ordinal RGB/pose associations with image tracks and LiDAR."""

from pathlib import Path
import json
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from iphone_rescan_audit import write_json
from stray_scanner_reconstruct import columns, read_csv


class NativeFrames:
    def __init__(self, audit, size):
        self.ids = json.loads((audit / "native_rgb_frames.json").read_text())
        self.index = {frame: row for row, frame in enumerate(self.ids)}
        self.shape = (size[1], size[0], 3)
        index = audit / "native_rgb_offsets.npy"
        if index.is_file():
            from zstandard import ZstdDecompressor

            self.offsets = np.load(index)
            self.data = np.memmap(audit / "native_rgb.zst", mode="r", dtype=np.uint8)
            if (self.offsets.shape != (len(self.ids) + 1,) or self.offsets[0] != 0
                    or self.offsets[-1] != self.data.size or np.any(np.diff(self.offsets) <= 0)):
                raise ValueError("Incomplete native RGB cache")
            self.decoder = ZstdDecompressor()
        else:
            self.offsets = None
            self.data = np.memmap(audit / "native_rgb.bin", mode="r", dtype=np.uint8,
                                  shape=(len(self.ids), *self.shape))

    def __getitem__(self, frame):
        row = self.index[frame]
        if self.offsets is None:
            return self.data[row]
        block = self.data[self.offsets[row]:self.offsets[row + 1]]
        return np.frombuffer(self.decoder.decompress(block), dtype=np.uint8).reshape(self.shape)


def rgb_depth(source, frame):
    name = f"{frame:06}.png"
    depth = cv2.imread(str(source / "depth" / name), cv2.IMREAD_UNCHANGED)
    conf = cv2.imread(str(source / "confidence" / name), cv2.IMREAD_UNCHANGED)
    if depth is None or conf is None:
        raise ValueError(f"Missing depth or confidence for {frame}")
    return depth, conf


def check(run, source, examples):
    out = run / "dataset_audit"
    preflight = json.loads((out / "preflight.json").read_text())
    rgb_w, rgb_h = preflight["native_rgb_size"]
    rgb = NativeFrames(out, (rgb_w, rgb_h))
    rows = read_csv(source / "odometry.csv")
    positions = columns(rows, "x", "y", "z")
    rotations = Rotation.from_quat(columns(rows, "qx", "qy", "qz", "qw")).as_matrix()
    image_size = (640, 480)
    scale = [image_size[0] / rgb_w, image_size[1] / rgb_h] * 2
    intrinsics = columns(rows, "fx", "fy", "cx", "cy") * scale
    fit = json.loads((run / "splits/train_frames.json").read_text())
    paired = [i for i in fit if i + 6 in rgb.index and 12 <= i < len(rows) - 18]
    lags = (-12, -6, -2, 0, 2, 6, 12)
    results = []
    for group in np.array_split(paired, 3):
        residuals = {lag: [] for lag in lags}
        checked = []
        for frame in group[np.linspace(0, len(group) - 1, 24, dtype=int)]:
            a, b = [cv2.cvtColor(cv2.resize(rgb[i], image_size, interpolation=cv2.INTER_AREA),
                                 cv2.COLOR_RGB2GRAY) for i in (frame, frame + 6)]
            p = cv2.goodFeaturesToTrack(a, 400, .015, 7)
            if p is None:
                continue
            q, ok1, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None)
            back, ok2, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None)
            consistent = (ok1.ravel() != 0) & (ok2.ravel() != 0)
            consistent &= np.linalg.norm((back - p)[:, 0], axis=1) < .6
            p, q = p[consistent, 0], q[consistent, 0]
            ix = np.clip(np.round(p[:, 0] * 256 / image_size[0]).astype(int), 0, 255)
            iy = np.clip(np.round(p[:, 1] * 192 / image_size[1]).astype(int), 0, 191)
            per_lag = {}
            common = np.ones(len(p), dtype=bool)
            for lag in lags:
                i, j = frame + lag, frame + 6 + lag
                depth, conf = rgb_depth(source, i)
                z = depth[iy, ix] * .001
                fx, fy, cx, cy = intrinsics[i]
                xyz = np.column_stack(((p[:, 0] - cx) * z / fx, (p[:, 1] - cy) * z / fy, z))
                other = (xyz @ rotations[i].T + positions[i] - positions[j]) @ rotations[j]
                fx, fy, cx, cy = intrinsics[j]
                with np.errstate(divide="ignore", invalid="ignore"):
                    projected = (other[:, :2] / other[:, 2:]) * [fx, fy] + [cx, cy]
                per_lag[lag] = np.linalg.norm(projected - q, axis=1)
                common &= (conf[iy, ix] == 2) & (z >= .25) & (z <= 5) & (other[:, 2] > 0)
            if common.sum() < 20:
                continue
            checked.append(int(frame))
            for lag in lags:
                residuals[lag].append(per_lag[lag][common])
        results.append({
            "anchor_frames": checked,
            "track_count": sum(len(x) for x in residuals[0]),
            "median_flow_error_px": {
                str(lag): float(np.median(np.concatenate(values))) for lag, values in residuals.items()
            },
        })
    alignment = out / "rgb_depth_alignment"
    alignment.mkdir(exist_ok=True)
    write_json(alignment / "flow_pose_lag.json", {"rgb_size": image_size, "frame_stride": 6,
                                                 "capture_thirds": results})
    for frame in examples:
        depth, conf = rgb_depth(source, frame)
        valid = (conf == 2) & (depth >= 250) & (depth <= 5000)
        z = depth.astype(np.float32)
        edges = (abs(cv2.Sobel(z, -1, 1, 0)) + abs(cv2.Sobel(z, -1, 0, 1))) > 450
        edges &= cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        rgb_small = cv2.resize(rgb[frame], (512, 384), interpolation=cv2.INTER_AREA)
        depth_rgb = cv2.applyColorMap(np.round(np.clip((depth - 250.) / 3000, 0, 1) * 255)
                                     .astype(np.uint8), cv2.COLORMAP_VIRIDIS)[..., ::-1].copy()
        depth_rgb[~valid] = (25, 29, 33)
        overlay = rgb_small.copy()
        overlay[edges.repeat(2, axis=0).repeat(2, axis=1)] = (14, 242, 241)
        sheet = Image.new("RGB", (1536, 411), "#172228")
        draw = ImageDraw.Draw(sheet)
        for i, (label, panel) in enumerate(zip(
                ("RGB", "Confidence-2 depth", "RGB with valid LiDAR depth edges"),
                (rgb_small, cv2.resize(depth_rgb, (512, 384), interpolation=cv2.INTER_NEAREST), overlay))):
            draw.text((i * 512 + 8, 6), f"{frame:06}   {label}", fill="white")
            sheet.paste(Image.fromarray(panel), (i * 512, 27))
        sheet.save(alignment / f"{frame:06}.png")


if __name__ == "__main__":
    cv2.setNumThreads(2)
    check(Path(sys.argv[1]), Path(sys.argv[2]), [int(frame) for frame in sys.argv[3:]])
