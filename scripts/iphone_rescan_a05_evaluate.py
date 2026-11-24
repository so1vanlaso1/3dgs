"""Verify an A05-Abs run and compare its endpoint with a rendered baseline."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from iphone_rescan_a05 import load_scan
from iphone_rescan_compare import poster_row, save_panel
import stray_painter as painter


def readback(out):
    protocol = json.loads((out / "protocol.json").read_text())
    training = json.loads((out / "training.json").read_text())
    arm = protocol["arm"]
    steps = protocol["steps"]
    checkpoints = protocol["checkpoints"]
    strategy = protocol["strategy"]
    assert steps == training["steps"] == checkpoints[-1] and strategy["cap_max"] is None
    for path, digest in protocol["input_sha256"].items():
        assert painter.sha(Path(path)) == digest, path
    for path, digest in protocol["source_sha256"].items():
        assert painter.sha(out / "training_source" / Path(path).name) == digest, path
    growth = json.loads((out / "growth.json").read_text())
    refine_every = strategy["refine_every"]
    first_refine = (strategy["refine_start_iter"] // refine_every + 1) * refine_every
    refinement_end = training.get("refinement_stopped_at", strategy["refine_stop_iter"])
    assert [row["step"] for row in growth] == list(
        range(first_refine, min(steps, refinement_end), refine_every)
    )
    count = protocol["initial_gaussians"]
    counts = {}
    for row in growth:
        assert row["target"] is None and row["before"] == count
        assert row["eligible"] == row["added"] == row["split"] and row["duplicated"] == 0
        assert row["after"] == count + row["added"] - row["pruned"]
        count = row["after"]
    assert count == training["gaussians"]
    loss = np.load(out / "loss.npy")
    assert loss.shape == (steps,) and np.isfinite(loss).all()
    for step in checkpoints:
        counts[step] = next((row["before"] for row in growth if row["step"] >= step), count)
        path = out / "checkpoints" / f"{arm}_{step}.pt"
        assert painter.sha(path) == training["checkpoint_sha256"][path.name]
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert checkpoint["sh_degree"] == min((step - 1) // 1000, 3)
        for key, dims in (("means", (3,)), ("quats", (4,)), ("scales", (3,)),
                          ("opacities", ()), ("sh", (16, 3))):
            values = checkpoint[key]
            assert values.shape == (counts[step], *dims) and torch.isfinite(values).all(), (step, key)
        assert (checkpoint["scales"] > 0).all()
        assert ((checkpoint["opacities"] >= 0) & (checkpoint["opacities"] <= 1)).all()
    return {
        "gaussians_by_checkpoint": counts, "growth_events": len(growth),
        "eligible_parents_split": sum(row["added"] for row in growth),
        "pruned": sum(row["pruned"] for row in growth),
        "peak_gaussians_before_prune": max(
            (row["before"] + row["added"] for row in growth), default=count
        ),
        "training": training,
    }


@torch.inference_mode()
def checkpoint_posters(out, arm, step, views, ks, images, size, boxes):
    path = out / "checkpoints" / f"{arm}_{step}.pt"
    checkpoint = torch.load(path, map_location="cuda", weights_only=True)
    state = tuple(checkpoint[key] for key in ("means", "quats", "scales", "opacities", "sh"))
    rows = []
    for frame, (left, top, right, bottom) in boxes.items():
        frame = int(frame)
        pred, _ = painter.render(state, views[frame], ks[frame], *size, checkpoint["sh_degree"])
        source = images.native[frame][top:bottom, left:right]
        crop = pred[0, top:bottom, left:right].clamp(0, 1).mul(255).round().byte().cpu().numpy()
        rows.append(poster_row(frame, arm, step, source, crop))
    return rows


def compare(run, source, out, baseline=None):
    audit = readback(out)
    protocol = json.loads((out / "protocol.json").read_text())
    arm, steps = protocol["arm"], protocol["steps"]
    if baseline is None:
        archived = run / "B0_standard/renders/7000"
        old_arm, old_steps, old_name, old_label = "B0", 7000, "B0", "B0 120k"
    else:
        readback(baseline)
        old_protocol = json.loads((baseline / "protocol.json").read_text())
        old_arm, old_steps = old_protocol["arm"], old_protocol["steps"]
        archived = baseline / "renders" / str(old_steps)
        old_name, old_label = f"A05_absgrad_{old_steps}", f"A05-Abs {old_steps:,} steps"
    frozen, images, split, mesh = load_scan(run)
    views, ks, depth_ks = painter.camera_arrays(split, frozen)
    frames = split["heldout"].tolist()
    regions = json.loads((run / "splits/evaluation_regions.json").read_text())
    scene = painter.mesh_scene(mesh)
    refs = painter.references(
        scene, split, [{"frame": frame} for frame in frames], source,
        regions={int(frame): boxes for frame, boxes in regions["materials_depth_xyxy"].items()},
        expected_pixels=2_874_584,
    )
    target = out / "renders" / str(steps)
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(out / "checkpoints" / f"{arm}_{steps}.pt", map_location="cuda", weights_only=True)
    state = tuple(checkpoint[key] for key in ("means", "quats", "scales", "opacities", "sh"))
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    print(f"Evaluating the saved checkpoint on {len(frames)} native held-out views.", flush=True)
    metrics, maps, _ = painter.evaluate(
        arm, state, checkpoint["sh_degree"], views, ks, depth_ks, images, refs, lpips, scene, None,
        target, size=frozen, preview_frames=(), png_compression=1,
    )
    metrics["evaluation_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    del state, checkpoint, lpips
    np.savez_compressed(target / "depth.npz", frames=frames,
                        z_m=np.stack([z for z, _ in maps.values()]),
                        alpha=np.stack([a for _, a in maps.values()]))
    with np.load(archived / "depth.npz") as depths:
        assert np.array_equal(depths["frames"], frames)
        old_maps = dict(zip(frames, zip(depths["z_m"], depths["alpha"])))
    depth = painter.depth_comparison(refs, {old_name: old_maps, arm: maps}, target)
    poster_out = target / "poster_crops"
    poster_out.mkdir(exist_ok=True)
    poster = []
    for frame, box in regions["poster_native_xyxy"].items():
        source_crop = Image.fromarray(images.native[int(frame)]).crop(box)
        with Image.open(archived / old_arm / f"{int(frame):06}.png") as old:
            before = old.crop(box)
        with Image.open(target / arm / f"{int(frame):06}.png") as new:
            after = new.crop(box)
        for name, step, prediction in ((old_name, old_steps, before), (arm, steps, after)):
            poster.append(poster_row(int(frame), name, step, np.asarray(source_crop), np.asarray(prediction)))
        save_panel(poster_out / f"{int(frame):06}.png",
                   [image.transpose(Image.Transpose.ROTATE_270) for image in (source_crop, before, after)],
                   ["Recorded RGB", old_label, f"A05-Abs {steps:,} steps"], int(frame))
    painter.save_rows(target / "poster_frames.csv", poster)
    if baseline is not None:
        trajectory = list(poster)
        for step in protocol["checkpoints"]:
            if old_steps <= step < steps:
                trajectory.extend(checkpoint_posters(
                    out, arm, step, views, ks, images, frozen, regions["poster_native_xyxy"]
                ))
        trajectory.sort(key=lambda row: (row["update"], row["arm"], row["frame"]))
        painter.save_rows(target / "poster_trajectory.csv", trajectory)
    with (archived / f"{old_arm}_frames.csv").open(newline="") as file:
        old_rows = list(csv.DictReader(file))
    with (target / f"{arm}_frames.csv").open(newline="") as file:
        new_rows = list(csv.DictReader(file))
    assert [int(row["frame"]) for row in old_rows] == [int(row["frame"]) for row in new_rows] == frames
    assert list(sorted(path.name for path in (target / arm).glob("*.png"))) == [
        f"{frame:06}.png" for frame in sorted(frames)
    ]
    for frame in frames:
        with Image.open(target / arm / f"{frame:06}.png") as image:
            assert image.size == frozen and image.mode == "RGB"
            image.verify()
    comparisons = {
        "psnr_db": sum(float(new["psnr_db"]) > float(old["psnr_db"]) for old, new in zip(old_rows, new_rows)),
        "ssim": sum(float(new["ssim"]) > float(old["ssim"]) for old, new in zip(old_rows, new_rows)),
        "lpips_alex": sum(float(new["lpips_alex"]) < float(old["lpips_alex"]) for old, new in zip(old_rows, new_rows)),
    }
    with np.load(target / "depth.npz") as saved:
        assert saved["frames"].tolist() == frames
        for key in ("z_m", "alpha"):
            assert saved[key].shape == (len(frames), 192, 256) and np.isfinite(saved[key]).all()
    summary = audit | {
        "heldout_views": len(frames),
        "rgb": {old_name: json.loads((archived / "metrics.json").read_text()), arm: metrics},
        f"views_improved_vs_{old_name}": comparisons, "depth": depth,
        "poster": {name: {key: float(np.mean([row[key] for row in poster if row["arm"] == name]))
                         for key in ("psnr_db", "ssim", "edge_gradient_retention")}
                   for name in (old_name, arm)},
    }
    painter.save_json(out / "summary.json", summary)
    painter.save_json(target / "metrics.json", metrics)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    compare(args.scan, args.source, args.output, args.baseline)
