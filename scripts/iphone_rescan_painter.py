"""Run the paired 120k painter experiments and save native held-out renders."""

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import trimesh

import stray_painter as painter
from iphone_rescan_alignment import NativeFrames


CHECKPOINTS = (3000, 5000, 7000)
ARMS = {"B0": "B0_standard", "B1": "B1_fixed_painter"}


class CudaRgb:
    def __init__(self, native, size):
        self.native = native
        self.size = size

    def __getitem__(self, frame):
        image = self.native[frame]
        if image.shape[:2] == self.size[::-1]:
            image = np.array(image)
        else:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        return torch.from_numpy(image).to("cuda")


def run_experiment(run, source, size, smoke=False):
    geometry = run / "geometry"
    audit = run / "dataset_audit"
    preflight = json.loads((audit / "preflight.json").read_text())
    native_size = tuple(preflight["native_rgb_size"])
    rgb = NativeFrames(audit, native_size)
    images = CudaRgb(rgb, size)
    split_path = geometry / "camera_split.npz"
    with np.load(split_path) as file:
        split = dict(file)
    frozen = json.loads((run / "splits/frame_split.json").read_text())
    for key in ("fit", "heldout", "withheld"):
        if not np.array_equal(frozen[key], split[key]):
            raise ValueError(f"Geometry and predeclared {key} IDs differ")
    if rgb.ids != sorted(frozen["fit"] + frozen["heldout"]):
        raise ValueError("Decoded native frame cache differs from the frozen split")
    regions_file = run / "splits/evaluation_regions.json"
    regions = {int(frame): boxes for frame, boxes in
               json.loads(regions_file.read_text())["materials_depth_xyxy"].items()}
    mesh = trimesh.load(geometry / "scene.glb", force="mesh", process=False)
    if len(mesh.faces) == 0 or mesh.visual.kind != "vertex":
        raise ValueError("Geometry lacks colored triangle data")
    views, ks, depth_ks = painter.camera_arrays(split, size)
    fit = split["fit"]
    audits = painter.split_audit(split, original=False)
    steps = 8 if smoke else CHECKPOINTS[-1]
    count, seed = 120_000, 43
    out = run / ("smoke" if smoke else "training")
    out.mkdir(exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sequence = np.concatenate([rng.permutation(fit) for _ in range(math.ceil(steps / len(fit)))])[:steps]
    init = painter.initialize(mesh, count, rng)
    spacing = math.sqrt(mesh.area / count)
    np.savez_compressed(out / "initialization.npz", **init)
    np.save(out / "training_sequence.npy", sequence)
    painter.save_rows(out / "split_audit.csv", audits)
    inputs = [regions_file, split_path, run / "splits/frame_split.json", geometry / "scene.glb",
              geometry / "tsdf_10mm.ply", geometry / "registered_cloud.ply", source / "rgb.mp4",
              source / "odometry.csv", audit / "archive_sha256.json",
              audit / "native_rgb_frames.json", out / "initialization.npz", out / "training_sequence.npy"]
    inputs.extend(source / key / f"{frame:06}.png" for frame in split["heldout"]
                  for key in ("depth", "confidence"))
    source_files = [Path(__file__), Path(painter.__file__), Path(__file__).with_name("iphone_rescan_alignment.py"),
                    Path(__file__).with_name("iphone_rescan_audit.py"),
                    Path(__file__).with_name("stray_scanner_reconstruct.py"),
                    Path(__file__).with_name("iphone_rescan_compare.py")]
    for script in source_files:
        shutil.copy2(script, out / script.name)
    print("Recording input hashes and training protocol.", flush=True)
    painter.save_json(out / "protocol.json", {
        "steps": steps, "gaussians": count, "seed": seed, "rgb_size": list(size),
        "depth_size": [256, 192], "checkpoint_updates": list(CHECKPOINTS) if not smoke else [],
        "training_frames": len(fit), "reserved_frames": len(split["withheld"]), "test_frames": len(audits),
        "rgb_objective": "0.8 L1 + 0.2 (1 - SSIM); RGB only; SH degree 0 to 3 by 1000 updates",
        "B0": "centres, orientation and scales optimized; fixed cameras and Gaussian count",
        "B1": "centres, orientation and normal thickness fixed; tangent scales bounded",
        "alpha_threshold": painter.ALPHA, "spacing_m": spacing,
        "initial_tangent_scale_m": .65 * spacing, "initial_normal_scale_m": min(.002, .15 * .65 * spacing),
        "input_sha256": {str(path): painter.sha(path) for path in inputs},
        "source_sha256": {path.name: painter.sha(path) for path in source_files},
        "appearance_margins_B1_vs_B0": {"max_psnr_drop_db": 1., "max_ssim_drop": .02,
                                       "max_lpips_increase": .03},
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "gpu_details": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version,memory.total",
                                                "--format=csv,noheader"], text=True).strip(),
    })
    scene = painter.mesh_scene(mesh)
    refs = painter.references(scene, split, audits[:3] if smoke else audits, source,
                              regions=regions, expected_pixels=None)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda().eval()
    summaries = {}
    for arm, name in ARMS.items():
        arm_dir = (out if smoke else run) / name
        checkpoints = arm_dir / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        updates = (steps,) if smoke else CHECKPOINTS
        print(f"Training {arm} at {size[0]}×{size[1]}.", flush=True)
        torch.cuda.reset_peak_memory_stats()
        final, elapsed, degree = painter.train(arm, init, spacing, views, ks, images, sequence, checkpoints,
                                               size=size, checkpoint_steps=updates[:-1])
        (checkpoints / f"{arm}.pt").rename(checkpoints / f"{arm}_{steps:04d}.pt")
        training_peak = torch.cuda.max_memory_allocated()
        summaries[arm] = {}
        for step in updates:
            target = arm_dir / "renders" / str(step)
            target.mkdir(parents=True, exist_ok=True)
            if step == steps:
                state, sh_degree = final, degree
            else:
                checkpoint = torch.load(checkpoints / f"{arm}_{step:04d}.pt", map_location="cuda")
                state = tuple(checkpoint[key] for key in ("means", "quats", "scales", "opacities", "sh"))
                if not all(torch.isfinite(value).all() for value in state):
                    raise ValueError(f"Non-finite evaluation state in {arm} at update {step}")
                sh_degree = checkpoint["sh_degree"]
            print(f"Evaluating {arm} at {step} updates ({len(refs)} views).", flush=True)
            torch.cuda.reset_peak_memory_stats()
            metrics, maps, _ = painter.evaluate(
                arm, state, sh_degree, views, ks, depth_ks, images, refs, lpips, scene, init, target,
                size=size, preview_frames=(), png_compression=1)
            np.savez_compressed(target / "depth.npz", frames=list(refs),
                                z_m=np.stack([z for z, _ in maps.values()]),
                                alpha=np.stack([alpha for _, alpha in maps.values()]))
            metrics["training_seconds"] = elapsed
            metrics["training_peak_allocated_mib"] = training_peak / 2**20
            metrics["evaluation_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            painter.save_json(target / "metrics.json", metrics)
            summaries[arm][step] = metrics
        painter.save_json(arm_dir / "metrics.json", summaries[arm])
    painter.save_json(out / "metrics.json", summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--size", type=int, nargs=2, default=(1920, 1440))
    parser.add_argument("--smoke", action="store_true")
    opts = parser.parse_args()
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    run_experiment(opts.run, opts.source, tuple(opts.size), opts.smoke)
