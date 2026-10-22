"""Capture campaign provenance or summarize verified baseline/A04 comparisons."""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


SCENES = ("counter", "kitchen", "bonsai")
SOURCES = (
    "gsplat/strategy/budget.py", "gsplat/strategy/default.py", "gsplat/strategy/ops.py",
    "examples/simple_trainer.py", "examples/datasets/colmap.py",
    "scripts/test-small-dataset.ps1", "scripts/download-room-small.py",
    "scripts/run-multiscene-comparison.ps1", "scripts/report-budget-experiment.py",
    "scripts/summarize-multiscene.py",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(root):
    evidence = root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in SOURCES:
        path = Path(name)
        target = evidence / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[name] = digest(path)
    scenes = {}
    for scene in SCENES:
        data = Path(f"data/{scene}_factor2")
        names = sorted(p.name for p in (data / "images").iterdir() if p.is_file())
        manifest = json.loads((data / "dataset_manifest.json").read_text())
        views = [{"name": name, "role": "held_out" if i % 8 == 0 else "train",
                  "sha256": digest(data / "images" / name)} for i, name in enumerate(names)]
        scenes[scene] = {"manifest": manifest, "views": views,
                         "train_count": sum(v["role"] == "train" for v in views),
                         "held_out_count": sum(v["role"] == "held_out" for v in views)}
    record = dict(
        recorded_utc=datetime.now(timezone.utc).isoformat(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256=hashes, scenes=scenes,
        protocol={"steps": 7000, "seed": 42, "image_factor": 2, "data_factor": 1,
                  "methods": ["B00", "A04"], "cap_max": 700000,
                  "gradient_weight": 1, "error_weight": 1, "edge_weight": 0.25,
                  "split_mode": "long_axis", "split_distance": 0.45,
                  "opacity_reduction": 0.6,
                  "selection": "Three additional indoor scenes selected before observing their results; no tuning.",
                  "scope": "One seed per method/scene; equal updates, not equal compute; A04 replaces cloning too."},
    )
    (evidence / "protocol.json").write_text(json.dumps(record, indent=2) + "\n")
    for label, command in (
        ("gpu.txt", ["nvidia-smi"]),
        ("pip_freeze.txt", [str(Path(".venv/Scripts/python.exe")), "-m", "pip", "freeze"]),
    ):
        (evidence / label).write_text(subprocess.check_output(command, text=True))
    print("Provenance and exact view manifests captured.")


def verify_ply(path, count):
    with path.open("rb") as f:
        header = []
        for _ in range(256):
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        else:
            raise ValueError(f"Invalid PLY: {path}")
        assert "format binary_little_endian 1.0" in header
        properties = [s for s in header if s.startswith("property ")]
        assert all(s.startswith("property float ") for s in properties)
        vertices = int(next(s for s in header if s.startswith("element vertex ")).split()[-1])
        values = np.fromfile(f, dtype="<f4")
    return bool(vertices == count and values.size == count * len(properties) and np.isfinite(values).all())


def summarize(root):
    protocol = json.loads((root / "evidence/protocol.json").read_text())
    source_matches = all(digest(Path(p)) == sha for p, sha in protocol["source_sha256"].items())
    records = {}
    for scene in SCENES:
        report = json.loads((root / scene / "comparison.json").read_text())
        assert report["all_checks_pass"], scene
        extra = {}
        width, height = next(iter(protocol["scenes"][scene]["manifest"]["image_dimensions"].values()))
        for method, run in (("B00", report["baseline"]), ("A04", report["runs"]["A04"])):
            path = Path(run["path"])
            images = list((path / "renders").glob("val_step6999_*.png"))
            sizes = []
            for p in images:
                with Image.open(p) as im:
                    sizes.append(im.size == (2 * width, height))
            extra[method] = {
                "finite_ply_and_count": verify_ply(path / "ply/point_cloud_6999.ply", run["final"]["num_GS"]),
                "all_held_out_renders": len(images) == protocol["scenes"][scene]["held_out_count"] and all(sizes),
            }
        assert all(all(c.values()) for c in extra.values()), (scene, extra)
        records[scene] = {"comparison": report, "additional_checks": extra}
    assert source_matches, "Source changed during campaign"
    result = {"all_checks_pass": True, "source_matches_capture": source_matches, "scenes": records}
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# A04 evaluation on three additional scenes", "",
             "Protocol: 7,000 updates, official images_2, one seed (42), matching per-scene initialization and train/held-out split. No parameter tuning.", "",
             "| Scene | Method | Gaussians | PSNR | SSIM | LPIPS | Trainer seconds | Peak tensor GiB |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    deltas = []
    for scene, record in records.items():
        report = record["comparison"]
        for method, run in (("B00", report["baseline"]), ("A04", report["runs"]["A04"])):
            m = run["final"]
            lines.append(f"| {scene} | {method} | {m['num_GS']:,} | {m['psnr']:.4f} | {m['ssim']:.5f} | {m['lpips']:.5f} | {run['training_seconds']:.1f} | {run['peak_tensor_gib']:.3f} |")
        a = report["runs"]["A04"]
        deltas.append((scene, a["gaussian_reduction_percent"], a["quality_delta"]))
    lines += ["", "## A04 changes relative to the same-scene baseline", "",
              "| Scene | Gaussian reduction | PSNR delta | SSIM delta | LPIPS delta |",
              "|---|---:|---:|---:|---:|"]
    for scene, reduction, d in deltas:
        lines.append(f"| {scene} | {reduction:.2f}% | {d['psnr']:+.4f} | {d['ssim']:+.5f} | {d['lpips']:+.5f} |")
    lines += ["", "Positive PSNR/SSIM deltas and negative LPIPS deltas indicate improvement.", "",
              "## Verification", "",
              "All per-scene configuration, checkpoint finiteness/count, PLY finiteness/count, full-resolution held-out render, budget-accounting and A04 checkpoint-reload checks passed. Source hashes match the campaign capture.", "",
              "## Limits", "",
              "- One seed per method/scene; no uncertainty estimate.",
              "- These three scenes are indoor scenes; outdoor generalization is untested.",
              "- Equal optimizer updates do not imply equal computation. Reported trainer time excludes startup, final export and final validation; peak tensor allocation excludes other GPU memory.",
              "- Random splitting consumes RNG differently, so subsequent camera orders can diverge.",
              "- A04 changes split placement, scale, opacity and clone replacement together.",
              "- Previously evaluated Room is historical development evidence; its results are not pooled into the new-scene average.",
              "- Checkpoints precede the final optimizer update; final in-memory validation and checkpoint reload are reported separately in each comparison JSON.", "",
              "## Reproduce", "", "```powershell",
              *[f".\\.venv\\Scripts\\python.exe scripts/download-room-small.py --scene {s} --image-factor 2" for s in SCENES],
              ".\\scripts\\run-multiscene-comparison.ps1",
              ".\\.venv\\Scripts\\python.exe scripts/summarize-multiscene.py", "```", ""]
    (root / "summary.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = [s.title() for s, _, _ in deltas]
    axes[0].bar(names, [r for _, r, _ in deltas], color="#397c91")
    axes[0].set_ylabel("Gaussian count reduction (%)")
    axes[1].bar(names, [d["psnr"] for _, _, d in deltas], color="#397c91")
    axes[1].set_ylabel("PSNR change (dB)")
    axes[1].axhline(0, color="black", linewidth=0.8)
    fig.suptitle("A04 versus baseline: 7,000 updates per run")
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=180)
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/multiscene_a04_20261022"))
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    if args.capture:
        capture(args.root)
    else:
        summarize(args.root)
