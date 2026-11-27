"""Prepare, train, score, and export a native-resolution newscan reconstruction."""

import argparse
from importlib.util import find_spec
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic


ROOT = Path(__file__).resolve().parent.parent
PROGRESS_PREFIXES = (
    "Extracting ", "Frozen split:", "Auditing ", "Fusing ", "Integrated through frame ",
    "A05-Abs inputs ready", "Sampling ", "A05_absgrad_uncapped:", "Paused refinement",
    "Scoring ", "Verified checkpoint ",
)


def run_stage(title, script, args, log, raw_log, start):
    log.write(f"\n## {title}\n\n")
    command = [sys.executable, "-u", str(ROOT / "scripts" / script), *map(str, args)]
    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace") as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            raw_log.write(line)
            progress = line.strip()
            if progress.startswith(PROGRESS_PREFIXES) or "Gaussians=" in progress:
                log.write(f"- {monotonic() - start:.0f} s | {progress}\n")
    if process.returncode:
        log.write(f"\n**{title} failed** with exit code {process.returncode}. See [run.log](run.log).\n")
        raise SystemExit(process.returncode)


def write_results(log, model):
    summary = json.loads((model / "summary.json").read_text())
    log.write("\n## Final results\n\n")
    log.write(f"Evaluated on {summary['final_views']} final views after checkpoint selection "
              f"on {summary['development_views']} development views. Depth measures agreement with "
              "the capture's LiDAR.\n\n")
    log.write("| Export | Update | Gaussians | PSNR | SSIM | LPIPS | LiDAR within 50 mm |\n"
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    results = [("Geometry and appearance", summary["selected_checkpoint"], summary["final"], summary["export"])]
    visual = summary.get("visual_checkpoint")
    if visual:
        results.append(("Appearance", visual["update"], visual["final"], visual["export"]))
    for label, step, scores, export in results:
        depth = scores["depth"]["within_50mm_fraction_of_eligible"]
        log.write(f"| [{label}](model/{export['filename']}) | {step:,} | {export['gaussians']:,} | "
                  f"{scores['psnr_db']:.2f} dB | {scores['ssim']:.4f} | {scores['lpips_alex']:.4f} | {depth:.1%} |\n")
    log.write("\n[Full metrics](model/summary.json) · [Rendered previews](model/final_comparison_1.jpg) "
              "· [Full run output](run.log)\n")


def run(archive, output, steps, initial_gaussians, prepared, *, growth_headroom_mib=768):
    if find_spec("plyfile") is None:
        raise SystemExit("PLY verification requires plyfile. Install it with python -m pip install plyfile")
    output.mkdir(parents=True, exist_ok=False)
    model = output / "model"
    model.mkdir()
    dataset = prepared if prepared is not None else output / "dataset"
    start = monotonic()
    with (output / "experiment_log.md").open("w", encoding="utf-8", buffering=1) as log, \
            (output / "run.log").open("w", encoding="utf-8", buffering=1) as raw_log:
        log.write("# newscan A05-Abs quality run\n\n")
        log.write(f"Native 1920 × 1440 RGB. Target {steps:,} updates from {initial_gaussians:,} Gaussians. "
                  "The LiDAR mesh initializes Gaussian centers. The RGB loss updates centers freely.\n")
        if growth_headroom_mib:
            log.write(f"The CUDA headroom guard can stop densification early (minimum {growth_headroom_mib:,} MiB).\n")
        else:
            log.write("The CUDA memory headroom guard is disabled.\n")
        print(f"Progress log: {output / 'experiment_log.md'}", flush=True)
        if prepared is None:
            run_stage("Dataset preparation", "prepare_newscan.py", [archive, dataset], log, raw_log, start)
        else:
            log.write(f"\nReusing prepared data at `{prepared}`.\n")
        run_stage("Training", "iphone_rescan_a05.py",
                  [dataset, model, "--quality", "--initial-gaussians", initial_gaussians, "--steps", steps,
                   "--growth-headroom-mib", growth_headroom_mib],
                  log, raw_log, start)
        run_stage("Evaluation and PLY export", "stray_quality_evaluate.py",
                  [dataset, dataset / "raw/e1cfcfffe4", model], log, raw_log, start)
        write_results(log, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to newscan.zip")
    parser.add_argument("output", type=Path, help="New directory for the run")
    parser.add_argument("--steps", type=int, default=48_000)
    parser.add_argument("--initial-gaussians", type=int, default=1_500_000)
    parser.add_argument("--growth-headroom-mib", type=int, default=768,
                        help="Minimum CUDA memory reserve for refinement, or 0 to disable the guard")
    parser.add_argument("--prepared", type=Path, help="Reuse a dataset previously built by prepare_newscan.py")
    args = parser.parse_args()
    if args.steps < 2 or args.initial_gaussians < 1:
        parser.error("Provide at least 2 updates and a positive Gaussian count")
    if args.growth_headroom_mib < 0:
        parser.error("--growth-headroom-mib must be nonnegative")
    run(args.archive.resolve(), args.output.resolve(), args.steps, args.initial_gaussians,
        args.prepared.resolve() if args.prepared else None, growth_headroom_mib=args.growth_headroom_mib)
