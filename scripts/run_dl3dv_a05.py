"""Run A05-Abs on the published TTB/DVSM DL3DV 64-input benchmark split."""

import argparse
import json
import math
from pathlib import Path
import tempfile
from urllib.request import urlopen

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch
from torch.nn import functional as F

from iphone_rescan_a05 import ARM, load_posed, train
from stray_quality_evaluate import export_checkpoint
import stray_painter as painter


DVSM_REVISION = "826be87c1a7c8ab3c0385b66b0ceaf8aa71f123b"
SPLIT_URL = (f"https://raw.githubusercontent.com/NVlabs/DVSM/{DVSM_REVISION}/"
             "configs/test_views_sampling/dl3dv_fold8_kmeans64.json")
DATASET = "DL3DV/DL3DV-Benchmark"
DATASET_REVISION = "9684e8382278c5e18173c1e72bd246daf2874539"
SIZE = (960, 544)
METRICS = ("psnr_db", "ssim", "lpips_vgg")


def download(cache, scene, files):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    folder = cache / scene / "nerfstudio"
    try:
        for filename in files:
            hf_hub_download(DATASET, f"{scene}/nerfstudio/{filename}", repo_type="dataset",
                            revision=DATASET_REVISION, local_dir=cache)
    except HfHubHTTPError as error:
        if isinstance(error, GatedRepoError) or (
            error.response is not None and error.response.status_code in (401, 403)
        ):
            raise SystemExit(
                f"Accept the dataset terms at https://huggingface.co/datasets/{DATASET} "
                "and run hf auth login (or provide HF_TOKEN), then retry."
            ) from error
        raise
    return folder


def preprocess(folder, names, metadata, destination):
    with Image.open(folder / "images_4" / names[0]) as image:
        width, height = image.size
    xscale, yscale = width / metadata["w"], height / metadata["h"]
    original = np.array([
        [metadata["fl_x"] * xscale, 0, metadata["cx"] * xscale],
        [0, metadata["fl_y"] * yscale, metadata["cy"] * yscale],
        [0, 0, 1],
    ])
    distortion = np.array([metadata.get(key, 0.) for key in ("k1", "k2", "p1", "p2")])
    rectified, _ = cv2.getOptimalNewCameraMatrix(original, distortion, (width, height), 0,
                                                 (width, height))
    ratio = (width / height) / (16 / 9)
    crop_width = round(width * min(1., 1. / ratio))
    crop_height = round(height * min(1., ratio))
    left = round((width - width * min(1., 1. / ratio)) / 2)
    top = round((height - height * min(1., ratio)) / 2)
    rectified_k = rectified.copy()
    rectified_k[0, 2] -= left
    rectified_k[1, 2] -= top
    rectified_k[0] *= SIZE[0] / crop_width
    rectified_k[1] *= SIZE[1] / crop_height

    rgb = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32,
                                   shape=(len(names), SIZE[1], SIZE[0], 3))
    for index, name in enumerate(names):
        with Image.open(folder / "images_4" / name) as image:
            raw = np.asarray(image.convert("RGB"))
        if raw.shape != (height, width, 3):
            raise ValueError(f"Inconsistent RGB dimensions in {name}: {raw.shape}")
        rectified_rgb = cv2.undistort(raw, original, distortion, None, rectified)
        crop = rectified_rgb[top:top + crop_height, left:left + crop_width]
        pixels = torch.from_numpy(crop.copy()).permute(2, 0, 1)[None].float()
        rgb[index] = F.interpolate(pixels, size=SIZE[::-1], mode="bilinear",
                                   align_corners=False, antialias=True)[0].permute(1, 2, 0).numpy()
    rgb.flush()
    return rgb, rectified_k.astype(np.float32)


def triangulate(rgb, views, ks, fit):
    # Match and triangulate only images from the official input split.
    sift = cv2.SIFT_create(nfeatures=2500)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    features = []
    for index in fit:
        image = np.rint(rgb[index]).clip(0, 255).astype(np.uint8)
        features.append(sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None))

    points, colors = [], []
    for i, (key_i, desc_i) in enumerate(features):
        for j in range(i + 1, min(i + 5, len(features))):
            key_j, desc_j = features[j]
            if desc_i is None or desc_j is None or min(len(desc_i), len(desc_j)) < 2:
                continue
            matches = [(a.queryIdx, a.trainIdx) for a, b in matcher.knnMatch(desc_i, desc_j, k=2)
                       if a.distance < .75 * b.distance]
            reverse = {(a.trainIdx, a.queryIdx) for a, b in matcher.knnMatch(desc_j, desc_i, k=2)
                       if a.distance < .75 * b.distance}
            matches = [pair for pair in matches if pair in reverse]
            if len(matches) < 8:
                continue
            # The benchmark and gsplat use half-integer pixel centers.
            uv_i = np.array([key_i[a].pt for a, _ in matches], dtype=np.float64) + .5
            uv_j = np.array([key_j[b].pt for _, b in matches], dtype=np.float64) + .5
            view_i, view_j = views[fit[i]], views[fit[j]]
            proj_i, proj_j = ks[fit[i]] @ view_i[:3], ks[fit[j]] @ view_j[:3]
            homogeneous = cv2.triangulatePoints(proj_i, proj_j, uv_i.T, uv_j.T).T
            depth_i, depth_j = homogeneous @ proj_i.T, homogeneous @ proj_j.T
            with np.errstate(divide="ignore", invalid="ignore"):
                xyz = homogeneous[:, :3] / homogeneous[:, 3:]
                error_i = np.linalg.norm(depth_i[:, :2] / depth_i[:, 2:] - uv_i, axis=1)
                error_j = np.linalg.norm(depth_j[:, :2] / depth_j[:, 2:] - uv_j, axis=1)
                z_i = depth_i[:, 2] / homogeneous[:, 3]
                z_j = depth_j[:, 2] / homogeneous[:, 3]
            center_i = -view_i[:3, :3].T @ view_i[:3, 3]
            center_j = -view_j[:3, :3].T @ view_j[:3, 3]
            ray_i, ray_j = xyz - center_i, xyz - center_j
            with np.errstate(divide="ignore", invalid="ignore"):
                cosine = ((ray_i * ray_j).sum(axis=1) /
                          (np.linalg.norm(ray_i, axis=1) * np.linalg.norm(ray_j, axis=1)))
            keep = ((error_i < 2) & (error_j < 2) & (z_i > .05) & (z_j > .05) &
                    (z_i < 20) & (z_j < 20) & (cosine < math.cos(math.radians(1))))
            xy = np.rint(uv_i[keep] - .5).astype(int)
            xy = np.clip(xy, [0, 0], np.array(SIZE) - 1)
            points.append(xyz[keep])
            colors.append(rgb[fit[i], xy[:, 1], xy[:, 0]] / 255.)

    if not points:
        raise ValueError("No overlapping input frames produced usable feature matches")
    xyz, rgb_values = np.concatenate(points), np.concatenate(colors)
    _, unique = np.unique(np.floor(xyz / .003).astype(np.int64), axis=0, return_index=True)
    if len(unique) < 100:
        raise ValueError(f"Only {len(unique)} input-only points triangulated; need at least 100")
    if len(unique) > 100000:
        unique = np.random.default_rng(43).choice(unique, 100000, replace=False)
    xyz = xyz[unique]
    distances, _ = cKDTree(xyz).query(xyz, k=4)
    scale = np.clip(np.square(distances[:, 1:]).mean(axis=1) ** .5, .002, .15)
    quats = np.zeros((len(xyz), 4), dtype=np.float32)
    quats[:, 0] = 1.
    return dict(points=xyz.astype(np.float32), rgb=rgb_values[unique].astype(np.float32),
                quats=quats, scales=np.repeat(scale[:, None], 3, axis=1).astype(np.float32))


def prepare(scene, split, cache, source_root, output):
    folder = (source_root / scene / "nerfstudio" if source_root else
              download(cache, scene, ["transforms.json"]))
    metadata = json.loads((folder / "transforms.json").read_text())
    fit_ids, target_ids = split["input"], split["heldout"]
    frame_ids = fit_ids + target_ids
    frames = metadata["frames"]
    if (len(fit_ids) != 64 or not target_ids or len(set(frame_ids)) != len(frame_ids)
            or min(frame_ids) < 0 or max(frame_ids) >= len(frames)):
        raise ValueError(f"{scene}: official split does not match the source frame list")
    names = [Path(frames[i]["file_path"]).stem + ".png" for i in frame_ids]
    if len(set(names)) != len(names):
        raise ValueError(f"{scene}: duplicate source image names")
    if not source_root:
        download(cache, scene, [f"images_4/{name}" for name in names])

    fit = np.arange(len(fit_ids))
    heldout = np.arange(len(fit_ids), len(frame_ids))
    c2w = np.array([frames[index]["transform_matrix"] for index in frame_ids], dtype=np.float64)
    if c2w.shape != (len(frame_ids), 4, 4) or not np.isfinite(c2w).all():
        raise ValueError(f"{scene}: invalid poses")
    c2w[:, :3, 1:3] *= -1
    c2w[:, :3, 3] -= c2w[fit, :3, 3].mean(axis=0)
    radius = float(np.max(np.linalg.norm(c2w[fit, :3, 3], axis=1)))
    if radius <= 0:
        raise ValueError(f"{scene}: input cameras have zero translation baseline")
    c2w[:, :3, 3] /= radius
    views = np.linalg.inv(c2w)

    with tempfile.TemporaryDirectory(dir=output.parent) as work:
        prepared = Path(work) / "prepared"
        prepared.mkdir()
        rgb, intrinsics = preprocess(folder, names, metadata, prepared / "rgb.npy")
        ks = np.repeat(intrinsics[None], len(frame_ids), axis=0)
        seeds = triangulate(rgb, views, ks, fit)
        del rgb
        np.savez(prepared / "cameras.npz", fit=fit, heldout=heldout, views=views.astype(np.float32),
                 ks=ks, positions_m=c2w[:, :3, 3].astype(np.float32))
        np.savez_compressed(prepared / "seeds.npz", **seeds)
        painter.save_json(prepared / "inputs.json", {
            "scene": scene, "input": fit_ids, "heldout": target_ids, "image_names": names,
            "split_url": SPLIT_URL, "dataset": DATASET, "dataset_revision": DATASET_REVISION,
            "source_mode": "local override" if source_root else "pinned Hugging Face dataset",
            "rgb_size": SIZE, "seed_count": len(seeds["points"]), "source_camera_radius": radius,
            "initialization": "SIFT triangulation from only the 64 input RGB images",
            "source_transforms_sha256": painter.sha(folder / "transforms.json"),
        })
        prepared.rename(output)
    return len(seeds["points"])


@torch.inference_mode()
def evaluate(scene_output, steps, metric):
    from skimage.metrics import structural_similarity

    model = scene_output / "model"
    size, images, split, views, ks = load_posed(scene_output / "prepared")
    inputs = json.loads((scene_output / "prepared/inputs.json").read_text())
    checkpoint = model / "checkpoints" / f"{ARM}_{steps}.pt"
    saved = torch.load(checkpoint, map_location="cuda", weights_only=True)
    state = tuple(saved[key] for key in ("means", "quats", "scales", "opacities", "sh"))
    rows = []
    for index, frame in zip(split["heldout"], inputs["heldout"], strict=True):
        predicted, _ = painter.render(state, views[index], ks[index], *size, saved["sh_degree"])
        predicted = predicted.clamp(0., 1.)
        target = images[index][None] / 255.
        mse = (target - predicted).square().mean().item()
        rows.append({
            "frame": frame, "image": inputs["image_names"][index],
            "psnr_db": -10 * math.log10(max(mse, 1e-12)),
            "ssim": structural_similarity(target[0].cpu().numpy(), predicted[0].cpu().numpy(),
                                          win_size=11, gaussian_weights=True, channel_axis=2,
                                          data_range=1.),
            "lpips_vgg": metric(predicted.permute(0, 3, 1, 2) * 2 - 1,
                                target.permute(0, 3, 1, 2) * 2 - 1).item(),
        })
    del saved, state
    painter.save_rows(scene_output / "heldout.csv", rows)
    summary = {key: float(np.mean([row[key] for row in rows])) for key in METRICS}
    summary.update(scene=inputs["scene"], heldout_views=len(rows), steps=steps,
                   export=export_checkpoint(checkpoint, scene_output / "scene.ply"))
    temporary = scene_output / "summary.json.tmp"
    painter.save_json(temporary, summary)
    temporary.replace(scene_output / "summary.json")
    return summary


def write_progress(output, scenes, completed):
    ordered = [completed[scene] for scene in scenes if scene in completed]
    mean = {key: float(np.mean([result[key] for result in ordered])) for key in METRICS} if ordered else None
    temporary = output / "summary.json.tmp"
    painter.save_json(temporary, {
        "scenes_requested": len(scenes),
        "scenes_evaluated": len(ordered),
        "complete": len(ordered) == len(scenes),
        "macro_average": mean,
        "scenes": ordered,
    })
    temporary.replace(output / "summary.json")
    return mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scene", action="append", help="Official scene hash; repeat for multiple scenes")
    group.add_argument("--all", action="store_true", help="Run all 140 scenes instead of the first scene")
    parser.add_argument("--cache", type=Path, default=Path("results/dl3dv_source"))
    parser.add_argument("--source-root", type=Path, help="Use local <scene>/nerfstudio source folders")
    parser.add_argument("--steps", type=int, default=48000)
    parser.add_argument("--refine-stop-iter", type=int,
                        help="Stop splitting and pruning at this update (default: train to the end)")
    parser.add_argument("--growth-headroom-mib", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.steps < 2 or args.growth_headroom_mib < 0:
        parser.error("Require at least two steps and nonnegative growth headroom")
    if args.refine_stop_iter is not None and args.refine_stop_iter < 1:
        parser.error("--refine-stop-iter must be positive")
    refine_stop_iter = args.steps if args.refine_stop_iter is None else args.refine_stop_iter
    torch.set_num_threads(min(4, torch.get_num_threads()))
    cv2.setNumThreads(4)
    with urlopen(SPLIT_URL, timeout=60) as source:
        splits = json.load(source)
    scenes = list(splits) if args.all else list(dict.fromkeys(args.scene or [next(iter(splits))]))
    if len(splits) != 140 or any(scene not in splits for scene in scenes):
        parser.error("Require a scene hash from the pinned 140-scene benchmark split")
    metric = None
    if not args.prepare_only:
        from lpips import LPIPS
        import plyfile
        import skimage
        if not torch.cuda.is_available():
            parser.error("CUDA is required for fitting and evaluation")

    args.output.mkdir(parents=True, exist_ok=True)
    config = args.output / "run.json"
    settings = {"splits": SPLIT_URL, "scenes": scenes, "steps": args.steps,
                "growth_headroom_mib": args.growth_headroom_mib,
                "refine_stop_iter": refine_stop_iter,
                "source_root": str(args.source_root.resolve()) if args.source_root else None,
                "dataset_revision": DATASET_REVISION}
    if config.exists():
        previous = json.loads(config.read_text())
        previous.setdefault("refine_stop_iter", 16000)
        if previous != settings:
            parser.error("Existing output uses different scene or training settings; choose a fresh output")
    painter.save_json(config, settings)
    log = args.output / "experiment_log.md"
    if not log.exists():
        log.write_text("# A05-Abs on DL3DV\n\n"
                       "Published 64-input split. Sparse initialization from input RGB only.\n"
                       f"Training updates: {args.steps:,}. Scheduled refinement end: "
                       f"{min(refine_stop_iter, args.steps):,}.\n\n",
                       encoding="utf-8")
    completed = {}
    if not args.prepare_only:
        for scene in scenes:
            result = args.output / scene / "summary.json"
            if not result.exists():
                continue
            summary = json.loads(result.read_text())
            if (summary["scene"] != scene or summary["steps"] != args.steps
                    or summary["heldout_views"] != len(splits[scene]["heldout"])
                    or any(not math.isfinite(summary[key]) for key in METRICS)):
                raise ValueError(f"Saved result does not match the benchmark run: {result}")
            completed[scene] = summary
        mean = write_progress(args.output, scenes, completed)
        print(f"DL3DV progress: {len(completed)}/{len(scenes)} scenes evaluated", flush=True)
    for scene in scenes:
        if scene in completed:
            continue
        output = args.output / scene
        output.mkdir(exist_ok=True)
        prepared = output / "prepared"
        if not prepared.exists():
            print(f"{scene}: preparing benchmark cameras and input-only seed points", flush=True)
            prepare(scene, splits[scene], args.cache, args.source_root, prepared)
        count = json.loads((prepared / "inputs.json").read_text())["seed_count"]
        if args.prepare_only:
            print(f"{scene}: prepared {count:,} initial Gaussians", flush=True)
            continue
        if metric is None:
            metric = LPIPS(net="vgg", version="0.1").eval()
        model = output / "model"
        model.mkdir(exist_ok=True)
        if not (model / "training.json").exists():
            train(prepared, model, count, args.steps, quality=True, posed_rgb=True,
                  resume=(model / "latest_resume.pt").exists(),
                  growth_headroom_mib=args.growth_headroom_mib, refine_stop_iter=refine_stop_iter)
        summary = evaluate(output, args.steps, metric.cuda())
        metric.cpu()
        completed[scene] = summary
        mean = write_progress(args.output, scenes, completed)
        entry = (f"- {scene}: {summary['psnr_db']:.4f} dB, SSIM {summary['ssim']:.5f}, "
                 f"VGG LPIPS {summary['lpips_vgg']:.5f}, {summary['heldout_views']} held-out views.\n")
        with log.open("a", encoding="utf-8") as handle:
            handle.write(entry)
        print(f"[{len(completed)}/{len(scenes)}] {entry}", flush=True)
    if completed:
        print(json.dumps(mean), flush=True)


if __name__ == "__main__":
    main()
