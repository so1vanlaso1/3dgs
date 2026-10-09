"""Compare completed room runs under matching training/evaluation settings."""

import argparse
import json
from pathlib import Path

import torch
import yaml


def read_run(path):
    cfg = yaml.load((path / "cfg.yml").read_text(), Loader=yaml.BaseLoader)
    steps = int(cfg["max_steps"])
    final_step = steps - 1
    stats = path / "stats"
    final = json.loads((stats / f"val_step{final_step:04d}.json").read_text())
    initial = json.loads((stats / "val_step0000.json").read_text())
    train = json.loads((stats / f"train_step{final_step:04d}_rank0.json").read_text())
    checkpoint_path = path / "ckpts" / f"ckpt_{final_step}_rank0.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    splats = checkpoint["splats"]
    count = len(splats["means"])
    checks = {
        "checkpoint_step_matches": checkpoint["step"] == final_step,
        "checkpoint_finite": all(bool(torch.isfinite(t).all()) for t in splats.values()),
        "checkpoint_shapes_match": all(len(t) == count for t in splats.values()),
        "metrics_count_matches": final["num_GS"] == train["num_GS"] == count,
    }
    ply_path = path / "ply" / f"point_cloud_{final_step}.ply"
    ply_count = None
    with ply_path.open("rb") as f:
        for _ in range(256):
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex "):
                ply_count = int(line.split()[-1])
            if line == "end_header":
                break
        else:
            raise ValueError(f"Invalid PLY header: {ply_path}")
    checks["ply_count_matches"] = ply_count == count
    history = train.get("growth_history", [])
    if "gaussian_budget" in train:
        cap = train["gaussian_budget"]
        checks["within_budget"] = count <= cap and all(
            h["before"] + h["added"] <= h["target"] <= cap
            and h["after"] <= cap for h in history)
        checks["growth_history_present"] = bool(history)
        checks["allocation_accounting_matches"] = all(
            h["before"] + h["added"] - h["pruned"] == h["after"] for h in history)
    return dict(path=str(path), steps=steps, initial=initial, final=final,
                training_seconds=train["ellipse_time"],
                peak_tensor_gib=train["mem"], checkpoint_bytes=checkpoint_path.stat().st_size,
                checks=checks, cfg=cfg, growth_history=history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--reload", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = read_run(args.baseline)
    shared_keys = ("data_dir", "data_factor", "test_every", "max_steps", "batch_size",
                   "packed", "sh_degree", "sh_degree_interval", "ssim_lambda", "init_type",
                   "init_opa", "init_scale", "normalize_world_space", "global_scale",
                   "means_lr", "scales_lr", "opacities_lr", "quats_lr", "sh0_lr", "shN_lr",
                   "opacity_reg", "scale_reg", "pose_opt", "app_opt", "post_processing",
                   "random_bkgd", "depth_loss", "antialiased", "with_ut", "with_eval3d")
    strategy_keys = ("grow_grad2d", "grow_scale3d", "refine_start_iter", "refine_stop_iter",
                     "refine_every", "reset_every", "prune_opa", "prune_scale3d", "absgrad",
                     "revised_opacity", "refine_scale2d_stop_iter", "pause_refine_after_reset")
    runs = {}
    for spec in args.run:
        label, path = spec.split("=", 1)
        run = read_run(Path(path))
        run["checks"]["same_training_settings"] = all(
            run["cfg"].get(k) == baseline["cfg"].get(k) for k in shared_keys)
        run["checks"]["same_density_settings_except_allocation"] = all(
            run["cfg"]["strategy"].get(k) == baseline["cfg"]["strategy"].get(k)
            for k in strategy_keys)
        run["checks"]["same_initial_model_metrics"] = all(
            abs(run["initial"][k] - baseline["initial"][k]) < 1e-4
            for k in ("psnr", "ssim", "lpips"))
        run["gaussian_reduction_percent"] = 100 * (
            1 - run["final"]["num_GS"] / baseline["final"]["num_GS"])
        run["quality_delta"] = {k: run["final"][k] - baseline["final"][k]
                                for k in ("psnr", "ssim", "lpips")}
        run["training_time_change_percent"] = 100 * (
            run["training_seconds"] / baseline["training_seconds"] - 1)
        runs[label] = run
    for spec in args.reload:
        label, path = spec.split("=", 1)
        run = runs[label]
        step = run["steps"] - 1
        reloaded = json.loads((Path(path) / "stats" / f"val_step{step:04d}.json").read_text())
        tolerance = {"psnr": 0.02, "ssim": 0.001, "lpips": 0.001}
        run["checkpoint_reload"] = dict(
            path=path, metrics=reloaded, tolerance=tolerance,
            delta={k: reloaded[k] - run["final"][k] for k in tolerance},
        )
        run["checks"]["reload_metrics_match"] = (
            reloaded["num_GS"] == run["final"]["num_GS"]
            and all(abs(reloaded[k] - run["final"][k]) <= v
                    for k, v in tolerance.items())
        )
    for run in [baseline, *runs.values()]:
        run["strategy"] = run.pop("cfg")["strategy"]
    report = dict(
        baseline=baseline, runs=runs,
        all_checks_pass=all(all(r["checks"].values()) for r in [baseline, *runs.values()]),
        limitations=[
            "One run per variant, using the trainer's fixed seed 42; seed variance is unmeasured.",
            "Sampler settings match; exact camera order can diverge because split operations consume RNG.",
            "Trainer timer includes initial validation; excludes dataset/worker startup, final export, final optimizer update and final validation.",
            "Peak tensor allocation excludes CUDA context, driver and other applications.",
            "Checkpoint is saved before the final optimizer update, as in the existing trainer.",
            "Top-four contribution attribution is an approximation to all contributing Gaussians.",
        ],
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Budgeted Gaussian allocation experiment", "",
             f"All variants use {baseline['steps']:,} optimizer updates and the same room dataset/evaluation settings.", "",
             "| Variant | Gaussians | PSNR | SSIM | LPIPS | Train seconds | Peak tensor GiB |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for label, run in [("Baseline", baseline), *runs.items()]:
        m = run["final"]
        lines.append(f"| {label} | {m['num_GS']:,} | {m['psnr']:.4f} | {m['ssim']:.5f} | "
                     f"{m['lpips']:.5f} | {run['training_seconds']:.1f} | {run['peak_tensor_gib']:.3f} |")
    lines += ["", "PSNR and SSIM: higher is better. LPIPS: lower is better.", "",
              "## Limits", "", *[f"- {s}" for s in report["limitations"]], "",
              f"Artifact and configuration checks passed: {report['all_checks_pass']}", ""]
    args.output.with_suffix(".md").write_text("\n".join(lines))
    print("\n".join(lines[:len(runs) + 8]))
    if not report["all_checks_pass"]:
        raise SystemExit("One or more experiment checks failed; inspect the JSON report.")


if __name__ == "__main__":
    main()
