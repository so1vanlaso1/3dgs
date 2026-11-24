"""Compare both painter arms over the frozen RGB and projected-depth views."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchmetrics.functional.image import structural_similarity_index_measure
import trimesh

import stray_painter as painter
from iphone_rescan_alignment import NativeFrames, rgb_depth


ARMS = {"B0": "B0_standard", "B1": "B1_fixed_painter"}
CHECKPOINTS = (3000, 5000, 7000)


def rgb_tensor(image):
    return torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].float() / 255


def gradients(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    return np.stack([cv2.Scharr(gray, -1, dx, dy, scale=1 / 32) for dx, dy in ((1, 0), (0, 1))], axis=-1)


def poster_row(frame, arm, step, source, prediction):
    gt, pred = rgb_tensor(source), rgb_tensor(prediction)
    reference_grad = gradients(source)[1:-1, 1:-1]
    predicted_grad = gradients(prediction)[1:-1, 1:-1]
    magnitude = np.linalg.norm(reference_grad, axis=-1)
    edges = magnitude >= max(.02, np.percentile(magnitude, 75))
    edge_power = float(np.sum(reference_grad[edges] ** 2))
    return {
        "frame": frame, "arm": arm, "update": step,
        "psnr_db": float(-10 * torch.log10((gt - pred).square().mean())),
        "ssim": float(structural_similarity_index_measure(pred, gt, data_range=1.)),
        "source_edge_pixels": int(edges.sum()),
        "edge_gradient_retention": float(np.sum(reference_grad[edges] * predicted_grad[edges]) / edge_power),
    }


def save_panel(path, frames, labels, frame):
    w, h = frames[0].size
    gap = 8
    sheet = Image.new("RGB", (len(frames) * (w + gap) - gap, h + 25), "#18232a")
    draw = ImageDraw.Draw(sheet)
    for i, (label, panel) in enumerate(zip(labels, frames)):
        x = i * (w + gap)
        draw.text((x + 7, 6), f"{frame:06}   {label}", fill="white")
        sheet.paste(panel, (x, 25))
    sheet.save(path)


def depth_picture(z, alpha, size):
    clamped = np.nan_to_num(np.clip((z - .25) / 3.75, 0, 1))
    values = (255 * clamped).astype(np.uint8)
    encoded = cv2.applyColorMap(values, cv2.COLORMAP_VIRIDIS)[..., ::-1].copy()
    encoded[(alpha < painter.ALPHA) | ~np.isfinite(z) | (z <= 0)] = (25, 29, 33)
    return Image.fromarray(encoded).resize(size, Image.Resampling.NEAREST)


def compare(run, source):
    audit = run / "dataset_audit"
    out = run / "comparisons"
    out.mkdir(exist_ok=True)
    rgb = NativeFrames(audit, tuple(json.loads((audit / "preflight.json").read_text())["native_rgb_size"]))
    with np.load(run / "geometry/camera_split.npz") as data:
        split = dict(data)
    frames = split["heldout"].tolist()
    config = json.loads((run / "splits/evaluation_regions.json").read_text())
    poster = {int(k): tuple(v) for k, v in config["poster_native_xyxy"].items()}
    materials = {int(k): v for k, v in config["materials_depth_xyxy"].items()}
    if not set(poster).union(materials).issubset(frames):
        raise ValueError("An evaluation crop contains a non-evaluation frame")
    mesh = trimesh.load(run / "geometry/scene.glb", force="mesh", process=False)
    scene = painter.mesh_scene(mesh)
    refs = painter.references(scene, split, [{"frame": i} for i in frames], source,
                              regions=materials, expected_pixels=None)
    region_rows = []
    for frame, regions in materials.items():
        depth, confidence = rgb_depth(source, frame)
        _, eligible, mesh_z, _ = refs[frame]
        for label, (x0, y0, x1, y1) in regions.items():
            box = np.s_[y0:y1, x0:x1]
            valid = eligible[box]
            hits = valid & np.isfinite(mesh_z[box])
            region_rows.append({
                "frame": frame, "material": label, "pixels": valid.size,
                "confidence_0": int((confidence[box] == 0).sum()),
                "confidence_1": int((confidence[box] == 1).sum()),
                "confidence_2": int((confidence[box] == 2).sum()),
                "eligible": int(valid.sum()), "mesh_hits": int(hits.sum()),
                **painter.error_stats(mesh_z[box][hits] - depth[box][hits] * .001),
            })
    painter.save_rows(out / "material_confidence.csv", region_rows)
    poster_rows, summary = [], {}
    for step in CHECKPOINTS:
        step_out = out / "depth_comparison" / str(step)
        step_out.mkdir(exist_ok=True, parents=True)
        targets = {arm: run / folder / "renders" / str(step) for arm, folder in ARMS.items()}
        maps = {}
        for arm, target in targets.items():
            with np.load(target / "depth.npz") as data:
                if not np.array_equal(data["frames"], frames):
                    raise ValueError(f"{arm} depth maps differ from frozen views")
                maps[arm] = dict(zip(frames, zip(data["z_m"], data["alpha"])))
        summary[step] = {
            "rgb": {arm: json.loads((target / "metrics.json").read_text()) for arm, target in targets.items()},
            "depth": painter.depth_comparison(refs, maps, step_out),
            "poster": {},
        }
        size = (384, 288)
        for frame in frames:
            original = Image.fromarray(rgb[frame])
            rendered = {arm: Image.open(target / arm / f"{frame:06}.png").convert("RGB")
                        for arm, target in targets.items()}
            if frame in poster:
                base = original.resize(rendered["B0"].size, Image.Resampling.BOX)
                sx, sy = base.width / original.width, base.height / original.height
                box = tuple(round(v * s) for v, s in zip(poster[frame], (sx, sy, sx, sy)))
                source_crop = base.crop(box)
                crops = {arm: image.crop(box) for arm, image in rendered.items()}
                for arm, crop in crops.items():
                    poster_rows.append(poster_row(frame, arm, step, np.asarray(source_crop), np.asarray(crop)))
                folder = out / "poster_crops" / str(step)
                folder.mkdir(exist_ok=True, parents=True)
                save_panel(folder / f"{frame:06}.png",
                           [image.transpose(Image.Transpose.ROTATE_270) for image in [source_crop, *crops.values()]],
                           ["Recorded RGB", "B0", "B1"], frame)
            if step == CHECKPOINTS[-1]:
                save_panel(step_out / f"{frame:06}.jpg",
                           [image.resize(size, Image.Resampling.LANCZOS) for image in [original, *rendered.values()]]
                           + [depth_picture(*maps[arm][frame], size) for arm in ARMS],
                           ["Recorded RGB", "B0 RGB", "B1 RGB", "B0 depth", "B1 depth"], frame)
        for arm in ARMS:
            rows = [row for row in poster_rows if row["arm"] == arm and row["update"] == step]
            summary[step]["poster"][arm] = {
                key: float(np.mean([row[key] for row in rows]))
                for key in ("psnr_db", "ssim", "edge_gradient_retention")
            }
    painter.save_rows(out / "poster_frames.csv", poster_rows)
    painter.save_json(out / "summary.json", {
        "checkpoints": summary, "mesh_sensor": painter.mesh_readback(refs),
        "edge_gradient_retention": "Signed Scharr gradient agreement divided by source gradient energy on "
                                   "the top quartile of source gradients at luminance-gradient magnitude >= 0.02.",
        "depth_panels": "Common 0.25–4 m expected Z scale; dark grey where opacity or finite-depth gate fails.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    compare(args.run, args.source)
