"""Check the higher-resolution room run and write a reproducible report."""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pycolmap
import torch
from PIL import Image


def read_json(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/room_factor2"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/room_hq_gtx1660"))
    parser.add_argument("--reload-dir", type=Path, default=Path("results/room_hq_gtx1660_reload"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("results/room_hq_old_checkpoint"))
    args = parser.parse_args()
    dataset = read_json(args.data_dir / "dataset_manifest.json")
    prepared = pycolmap.Reconstruction(str(args.data_dir / "sparse/0"))
    source = pycolmap.Reconstruction(str(args.data_dir / "source/sparse/0"))
    checks = {}
    checks["source_poses_preserved"] = (
        set(prepared.images) == set(source.images)
        and all(np.array_equal(image.cam_from_world().matrix(), source.images[key].cam_from_world().matrix())
                for key, image in prepared.images.items())
    )
    checks["source_points_and_colors_preserved"] = (
        set(prepared.points3D) == set(source.points3D)
        and all(np.array_equal(point.xyz, source.points3D[key].xyz)
                and np.array_equal(point.color, source.points3D[key].color)
                for key, point in prepared.points3D.items())
    )
    dimensions_ok = True
    for image in prepared.images.values():
        with Image.open(args.data_dir / "images" / image.name) as pixels:
            dimensions_ok &= pixels.size == tuple(dataset["image_dimensions"][str(image.camera_id)])
    checks["all_image_dimensions_match"] = bool(dimensions_ok)
    for key, camera in source.cameras.items():
        camera.rescale(*dataset["image_dimensions"][str(key)])
    checks["intrinsics_rescaled_correctly"] = all(
        np.allclose(camera.params, source.cameras[key].params, rtol=0, atol=1e-10)
        and (camera.width, camera.height) == (source.cameras[key].width, source.cameras[key].height)
        for key, camera in prepared.cameras.items()
    )

    checkpoint_path = sorted((args.result_dir / "ckpts").glob("ckpt_*_rank0.pt"),
                             key=lambda path: int(path.stem.split("_")[1]))[-1]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    step = checkpoint["step"]
    splats = checkpoint["splats"]
    count = len(splats["means"])
    checks["checkpoint_finite"] = all(bool(torch.isfinite(value).all()) for value in splats.values())
    checks["checkpoint_shapes_consistent"] = all(len(value) == count for value in splats.values())
    checks["degree_three_coefficients_saved"] = splats["shN"].shape[1:] == (15, 3)
    ply_path = args.result_dir / "ply" / f"point_cloud_{step}.ply"
    with ply_path.open("rb") as file:
        header = []
        while True:
            line = file.readline()
            if not line:
                raise ValueError("PLY header is incomplete")
            header.append(line.decode("ascii").strip())
            if line.strip() == b"end_header":
                break
        assert "format binary_little_endian 1.0" in header
        vertex_count = int(next(line for line in header if line.startswith("element vertex ")).split()[-1])
        properties = [line for line in header if line.startswith("property ")]
        assert all(line.startswith("property float ") for line in properties)
        values = np.fromfile(file, dtype="<f4")
    checks["ply_finite"] = bool(np.isfinite(values).all())
    checks["ply_vertex_count_matches"] = vertex_count == count and values.size == count * len(properties)

    initial = read_json(args.result_dir / "stats/val_step0000.json")
    final = read_json(args.result_dir / f"stats/val_step{step:04d}.json")
    reload_metrics = read_json(args.reload_dir / f"stats/val_step{step:04d}.json")
    baseline = read_json(args.baseline_dir / "stats/val_step0599.json")
    training = read_json(args.result_dir / f"stats/train_step{step:04d}_rank0.json")
    checks["quality_improved_over_old_model_at_same_resolution"] = (
        final["psnr"] > baseline["psnr"] and final["ssim"] > baseline["ssim"] and final["lpips"] < baseline["lpips"]
    )
    checks["reload_metrics_match"] = all(
        abs(final[key] - reload_metrics[key]) < tolerance
        for key, tolerance in (("psnr", 0.05), ("ssim", 0.002), ("lpips", 0.002))
    )
    config_text = (args.result_dir / "cfg.yml").read_text()
    test_every = int(next(line.split(":", 1)[1] for line in config_text.splitlines() if line.startswith("test_every:")))
    validation_views = len(range(0, dataset["images"], test_every))
    renders = list((args.result_dir / "renders").glob(f"val_step{step}_*.png"))
    checks["all_validation_renders_saved"] = len(renders) == validation_views
    low_resolution = read_json(Path("data/room_small/dataset_manifest.json"))
    width, height = next(iter(dataset["image_dimensions"].values()))
    render_dimensions_ok = True
    for render in renders:
        with Image.open(render) as canvas:
            render_dimensions_ok &= canvas.size == (2 * width, height)
    checks["all_validation_renders_at_full_resolution"] = bool(render_dimensions_ok)
    low_width, low_height = next(iter(low_resolution["image_dimensions"].values()))
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "verified_date": datetime.now(timezone(timedelta(hours=7))).date().isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "dataset": dataset,
        "dataset_size_note": "Same 311 camera views; larger dataset by image resolution and byte size, not view count.",
        "pixels_per_image_ratio": width * height / (low_width * low_height),
        "compressed_size_ratio": dataset["compressed_dataset_bytes"] / low_resolution["compressed_dataset_bytes"],
        "train_views": dataset["images"] - validation_views,
        "validation_views": validation_views,
        "steps": step + 1,
        "old_checkpoint_at_same_high_resolution": baseline,
        "initial": initial,
        "final": final,
        "checkpoint_reload": reload_metrics,
        "training": training,
        "peak_torch_tensor_allocation_mib": training["mem"] * 1024,
        "memory_note": "Trainer peak tensor allocation before final evaluation; excludes driver, CUDA context, desktop, and other apps.",
        "timing_note": "Training ellipse_time includes initial evaluation; excludes dataset/worker startup and final evaluation.",
        "reload_note": "Trainer saves before the last optimizer update, so tiny metric differences are expected.",
        "checks": checks,
        "artifacts": {
            "checkpoint": str(checkpoint_path), "ply": str(ply_path),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "ply_bytes": ply_path.stat().st_size,
            "preview": str(args.result_dir / f"renders/val_step{step}_0000.png"),
        },
    }
    report_path = args.result_dir / "large_dataset_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise SystemExit("Verification failed; see checks in the report")


if __name__ == "__main__":
    main()
