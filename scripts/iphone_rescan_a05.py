"""Run one uncapped A05-Abs training arm on the frozen iPhone rescan."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
from shutil import copy2
import subprocess
import time

import cv2
from fused_ssim import fused_ssim
import numpy as np
import torch
import trimesh

from gsplat import rasterization
from gsplat.strategy import BudgetStrategy
from iphone_rescan_alignment import NativeFrames
from iphone_rescan_painter import CudaRgb
import stray_painter as painter
from stray_quality_bootstrap import paths as surface_tile_paths, seed_tiles


ARM = "A05_absgrad_uncapped"
CHECKPOINTS = (3000, 5000, 7000)
QUALITY_CHECKPOINTS = (3000, 10000, 20000, 32000, 40000)
RATES = {
    "means": .00016, "scales": .005, "opacities": .05,
    "quats": .001, "sh0": .0025, "shN": .0025 / 20,
}


def load_scan(run, *, load_mesh=True):
    audit = run / "dataset_audit"
    size = tuple(json.loads((audit / "preflight.json").read_text())["native_rgb_size"])
    images = CudaRgb(NativeFrames(audit, size), size)
    with np.load(run / "geometry/camera_split.npz") as file:
        split = dict(file)
    frozen = json.loads((run / "splits/frame_split.json").read_text())
    for key in ("fit", "heldout", "withheld"):
        if not np.array_equal(frozen[key], split[key]):
            raise ValueError(f"Frozen {key} IDs differ from the camera archive")
    if images.native.ids != sorted(frozen["fit"] + frozen["heldout"]):
        raise ValueError("Native RGB cache IDs differ from the frozen split")
    mesh = trimesh.load(run / "geometry/scene.glb", force="mesh", process=False) if load_mesh else None
    return size, images, split, mesh


def render_state(params):
    return (
        params["means"], params["quats"], params["scales"].exp(),
        params["opacities"].sigmoid(), torch.cat((params["sh0"], params["shN"]), dim=1),
    )


def load_posed(run):
    rgb = np.load(run / "rgb.npy", mmap_mode="r")
    size = (rgb.shape[2], rgb.shape[1])
    with np.load(run / "cameras.npz") as archive:
        split = dict(archive)
    return size, CudaRgb(rgb, size), split, painter.tensor(split["views"]), painter.tensor(split["ks"])


def train(run, out, count, steps=CHECKPOINTS[-1], *, quality=False, resume=False,
          finish_refinement=False, growth_headroom_mib=768, rgb_long_edge=None, posed_rgb=False):
    strategy = BudgetStrategy(
        cap_max=None, absgrad=True, grow_grad2d=.0008, refine_stop_iter=16000 if quality else 6000,
        reset_every=6000 if quality else 3000,
        edge_weight=.25, split_mode="long_axis", clone_policy="split",
    )
    baseline = run / "training"
    seed = 43 if quality else json.loads((baseline / "protocol.json").read_text())["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if posed_rgb:
        if not quality or rgb_long_edge is not None:
            raise ValueError("Posed RGB requires quality mode and its prepared resolution")
        size, images, split, views, ks = load_posed(run)
        mesh = None
    else:
        size, images, split, mesh = load_scan(run, load_mesh=not quality)
        views, ks, _ = painter.camera_arrays(split, size)
    if rgb_long_edge is not None and rgb_long_edge < max(size):
        resized = tuple(max(1, round(edge * rgb_long_edge / max(size))) for edge in size)
        ks[:, 0] *= resized[0] / size[0]
        ks[:, 1] *= resized[1] / size[1]
        size = resized
        images = CudaRgb(images.native, size)
    fit = split["fit"]
    resume_path = out / "latest_resume.pt"
    saved = torch.load(resume_path, map_location="cpu", weights_only=True) if resume else None
    if resume:
        previous = json.loads((out / "protocol.json").read_text())
        if (previous["steps"] != steps or previous["initial_gaussians"] != count or not previous["quality_mode"]
                or tuple(previous["rgb_size"]) != size
                or previous.get("posed_rgb", False) != posed_rgb
                or previous.get("growth_headroom_mib", 768) != growth_headroom_mib
                or previous["source_sha256"][str(Path(__file__))] != painter.sha(Path(__file__))):
            raise ValueError("Resume settings or source differ from the quality run protocol")
    if quality:
        sequence = np.concatenate([rng.permutation(fit) for _ in range(math.ceil(steps / len(fit)))])[:steps]
    if posed_rgb:
        if not resume:
            with np.load(run / "seeds.npz") as archive:
                init = dict(archive)
            if len(init["points"]) != count:
                raise ValueError("Seed count differs from the requested Gaussian count")
        spacing = tangent = normal = None
    elif quality:
        if resume:
            area = previous["surface_area_m2"]
        else:
            print(f"Sampling {count:,} Gaussians from the fused surface tiles.", flush=True)
            area, init = seed_tiles(run / "geometry", count, rng)
    else:
        archived_sequence = np.load(baseline / "training_sequence.npy")
        epochs = [rng.permutation(fit) for _ in range(math.ceil(len(archived_sequence) / len(fit)))]
        if not np.array_equal(archived_sequence, np.concatenate(epochs)[:len(archived_sequence)]):
            raise ValueError("Archived camera sequence differs from its reproducible protocol")
        # Extend the camera sequence while preserving the original RNG state for mesh sampling.
        camera_rng = deepcopy(rng)
        epochs.extend(camera_rng.permutation(fit) for _ in range(math.ceil(steps / len(fit)) - len(epochs)))
        sequence = np.concatenate(epochs)[:steps]
        init = painter.initialize(mesh, count, rng)
        area = mesh.area
    if not posed_rgb:
        spacing = math.sqrt(area / count)
        tangent = .65 * spacing
        normal = min(.002, .15 * tangent)
    centers = split["positions_m"][fit]
    scene_scale = 1.1 * np.linalg.norm(centers - centers.mean(axis=0), axis=1).max()
    if resume:
        params = torch.nn.ParameterDict({key: tensor.cuda() for key, tensor in saved["params"].items()})
    else:
        if posed_rgb:
            scales = init["scales"]
        else:
            scales = np.broadcast_to([tangent, tangent, normal], (count, 3)).copy()
            if quality:
                scales[:, :2] *= init["relative_spacing"].clip(.55, 1.2)[:, None]
        params = torch.nn.ParameterDict({
            "means": painter.tensor(init["points"]),
            "quats": painter.tensor(init["quats"]),
            "scales": painter.tensor(np.log(scales)),
            "opacities": torch.full((count,), math.log(.3 / .7), device="cuda"),
            "sh0": ((painter.tensor(init["rgb"]) - .5) / .28209479177387814)[:, None],
            "shN": torch.zeros((count, 15, 3), device="cuda"),
        })
    del mesh
    optimizers = {
        name: torch.optim.Adam([params[name]], lr=rate, eps=1e-15, fused=True)
        for name, rate in RATES.items()
    }
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(float(scene_scale))
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(exist_ok=resume or posed_rgb)
    milestones = QUALITY_CHECKPOINTS if quality else (*CHECKPOINTS, 10000, 15000)
    saves = tuple(step for step in milestones if step < steps) + (steps,)
    if not resume:
        np.save(out / "training_sequence.npy", sequence)
        np.savez_compressed(out / "initialization.npz", **init)
        del init
    elif not np.array_equal(sequence, np.load(out / "training_sequence.npy")):
        raise ValueError("Resumed camera schedule differs from the original")
    inputs = [
        run / "geometry/camera_split.npz", run / "geometry/scene.glb",
        run / "splits/frame_split.json",
        run / "dataset_audit/native_rgb_frames.json", run / "dataset_audit/preflight.json",
        out / "initialization.npz", out / "training_sequence.npy",
    ]
    if posed_rgb:
        inputs = [run / name for name in ("rgb.npy", "cameras.npz", "seeds.npz", "inputs.json")]
        inputs.extend((out / "initialization.npz", out / "training_sequence.npy"))
    elif quality:
        if images.native.offsets is None:
            inputs.extend(run / path for path in (
                "dataset_audit/native_rgb.bin", "dataset_audit/archive_sha256.json",
                "dataset_audit/rgb_depth_alignment/flow_pose_lag.json", "splits/evaluation_regions.json",
            ))
        else:
            inputs.extend(run / path for path in (
                "dataset_audit/native_rgb_offsets.npy", "dataset_audit/native_rgb.zst",
                "dataset_audit/alignment.json", "splits/evaluation_protocol.json",
                "geometry/surface_tiles.json", "geometry/heldout_mesh_depth.npz",
            ))
        inputs.extend(path for path in surface_tile_paths(run / "geometry") if path not in inputs)
    else:
        inputs.extend((baseline / "protocol.json", baseline / "training_sequence.npy",
                       run / "splits/evaluation_regions.json"))
    sources = [Path(__file__), Path(painter.__file__), Path("scripts/iphone_rescan_painter.py"),
               Path("scripts/iphone_rescan_alignment.py"), Path("gsplat/strategy/budget.py"),
               Path("gsplat/strategy/default.py"), Path("gsplat/strategy/ops.py"),
               Path("gsplat/rendering.py")]
    if posed_rgb:
        sources.append(Path(__file__).with_name("run_dl3dv_a05.py"))
    elif quality:
        sources.append(Path("scripts/stray_quality_bootstrap.py"))
    snapshot = out / "training_source"
    snapshot.mkdir(exist_ok=True)
    if not resume:
        for source in sources:
            copy2(source, snapshot / source.name)
        painter.save_json(out / "protocol.json", {
            "arm": ARM, "seed": seed, "steps": len(sequence), "initial_gaussians": count,
            "rgb_size": size, "fit_views": len(fit), "heldout_views": len(split["heldout"]),
            "nominal_spacing_m": spacing, "initial_tangent_m": tangent, "initial_normal_m": normal,
            "scene_scale_m": float(scene_scale), "scene_scale_rule": "1.1 × fit camera radius from mean",
            "strategy": asdict(strategy), "packed": True, "checkpoints": saves,
            "optimizer": "fused Adam, eps=1e-15, means LR decays to 1% of initial value",
            "learning_rates": RATES, "objective": "0.8 RGB L1 + 0.2 DSSIM" if quality else
            json.loads((baseline / "protocol.json").read_text())["rgb_objective"],
            "near_m": .05, "far_m": 20., "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version",
                                              "--format=csv,noheader"], text=True).strip(),
            "input_sha256": {str(path): painter.sha(path) for path in inputs},
            "source_sha256": {str(path): painter.sha(path) for path in sources},
        } | ({"quality_mode": True, "posed_rgb": True, "growth_headroom_mib": growth_headroom_mib,
              "sampling": "Triangulated from benchmark input RGB only"} if posed_rgb else
             {"quality_mode": True, "surface_area_m2": area, "growth_headroom_mib": growth_headroom_mib,
              "sampling": "72% mesh surface area, 28% local mesh RGB contrast times area"} if quality else {}))
    losses = torch.empty(len(sequence), device="cuda")
    first_update, prior_elapsed = 0, 0.
    prior_peak = (0, 0)
    tracking_until = strategy.refine_stop_iter
    if resume:
        first_update = saved["step"]
        prior_elapsed = saved["elapsed"]
        prior_peak = saved["peak"]
        tracking_until = saved["tracking_until"]
        if finish_refinement:
            tracking_until = min(first_update, tracking_until)
        strategy_state = {key: value.cuda() if isinstance(value, torch.Tensor) else value
                          for key, value in saved["strategy"].items()}
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(saved["optimizer"][name])
        losses[:first_update] = saved["loss"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        del saved
    print(f"{ARM}: {len(params['means']):,} Gaussians at update {first_update:,}, "
          f"target {len(sequence):,} updates.", flush=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for step, frame in enumerate(sequence[first_update:], start=first_update):
        degree = min(step // 1000, 3)
        optimizers["means"].param_groups[0]["lr"] = RATES["means"] * .01 ** (step / (len(sequence) - 1))
        tracking = step < tracking_until
        predicted, _, info = rasterization(
            *render_state(params), views[frame][None], ks[frame][None], *size,
            sh_degree=degree, packed=True, near_plane=.05, far_plane=20., absgrad=tracking,
        )
        target = images[frame][None].float() / 255
        if tracking:
            strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
            strategy.observe_image(params, strategy_state, step, info, predicted, target)
        l1 = (predicted - target).abs().mean()
        ssim = fused_ssim(predicted.permute(0, 3, 1, 2).contiguous(),
                          target.permute(0, 3, 1, 2).contiguous(), padding="valid")
        loss = .8 * l1 + .2 * (1 - ssim)
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if tracking:
            if (quality and growth_headroom_mib > 0 and step > strategy.refine_start_iter
                    and step % strategy.refine_every == 0):
                free, _ = torch.cuda.mem_get_info()
                allocated = torch.cuda.memory_allocated()
                available = free + torch.cuda.memory_reserved() - allocated
                if available < max(growth_headroom_mib * 2**20, .32 * allocated):
                    tracking_until = step
                    tracking = False
                    print(f"Paused refinement at {step:,}: {available / 2**20:.0f} MiB remaining for growth.", flush=True)
            if tracking:
                strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=True)
        losses[step] = loss.detach()
        if step + 1 in saves:
            with torch.no_grad():
                state = render_state(params)
                if not all(torch.isfinite(value).all() for value in state):
                    raise FloatingPointError(f"Nonfinite checkpoint at update {step + 1}")
                painter.save_checkpoint(state, degree, checkpoints / f"{ARM}_{step + 1}.pt")
                del state
            painter.save_json(out / "growth.json", strategy_state["growth_history"])
        if quality and (step + 1) % 3000 == 0 and step + 1 < steps:
            temporary = resume_path.with_suffix(".tmp")
            torch.save({"step": step + 1, "params": params.state_dict(), "tracking_until": tracking_until,
                        "strategy": strategy_state, "optimizer": {key: value.state_dict() for key, value in optimizers.items()},
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                        "loss": losses[:step + 1].cpu(), "elapsed": time.perf_counter() - start + prior_elapsed,
                        "peak": (max(prior_peak[0], torch.cuda.max_memory_allocated()),
                                 max(prior_peak[1], torch.cuda.max_memory_reserved()))}, temporary)
            temporary.replace(resume_path)
        if (step + 1) % 500 == 0:
            print(f"{step + 1}/{len(sequence)}  loss={loss.item():.5f}  "
                  f"Gaussians={len(params['means']):,}  "
                  f"elapsed={time.perf_counter() - start + prior_elapsed:.1f}s", flush=True)
    torch.cuda.synchronize()
    if not torch.isfinite(losses).all():
        raise FloatingPointError("Training losses contain nonfinite values")
    np.save(out / "loss.npy", losses.cpu().numpy())
    painter.save_json(out / "training.json", {
        "steps": len(sequence), "gaussians": len(params["means"]),
        "training_seconds": time.perf_counter() - start + prior_elapsed,
        "peak_allocated_mib": max(prior_peak[0], torch.cuda.max_memory_allocated()) / 2**20,
        "peak_reserved_mib": max(prior_peak[1], torch.cuda.max_memory_reserved()) / 2**20,
        "checkpoint_sha256": {path.name: painter.sha(path) for path in sorted(checkpoints.glob("*.pt"))},
    } | ({"refinement_stopped_at": min(tracking_until, len(sequence))} if quality else {}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--initial-gaussians", type=int, required=True)
    parser.add_argument("--steps", type=int, default=CHECKPOINTS[-1])
    parser.add_argument("--quality", action="store_true", help="Use the longer Stray Scanner detail schedule")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest quality-run state")
    parser.add_argument("--finish-refinement", action="store_true", help="Resume without further growth or opacity resets")
    parser.add_argument("--growth-headroom-mib", type=int, default=768,
                        help="Minimum CUDA memory reserve for refinement, or 0 to disable the guard")
    parser.add_argument("--rgb-long-edge", type=int,
                        help="Downsample training RGB to this maximum long edge, preserving aspect ratio")
    args = parser.parse_args()
    if args.initial_gaussians < 1:
        parser.error("--initial-gaussians must be positive")
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    if args.growth_headroom_mib < 0:
        parser.error("--growth-headroom-mib cannot be negative")
    if args.rgb_long_edge is not None and args.rgb_long_edge < 16:
        parser.error("--rgb-long-edge must be at least 16")
    if (args.resume and not args.quality) or (args.finish_refinement and not args.resume):
        parser.error("--resume requires --quality, and --finish-refinement requires --resume")
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    train(args.scan, args.output, args.initial_gaussians, args.steps, quality=args.quality,
          resume=args.resume, finish_refinement=args.finish_refinement,
          growth_headroom_mib=args.growth_headroom_mib, rgb_long_edge=args.rgb_long_edge)
