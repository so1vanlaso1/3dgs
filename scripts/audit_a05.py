"""Audit completed native A05 runs against their paired archived A04 runs."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from research_campaign import REPO, sha, write


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def png_psnr(target, render):
    error = (target.astype(np.float64) - render.astype(np.float64)) / 255
    return float(-10 * np.log10(np.mean(error * error)))


def flatten_config(config, prefix=""):
    fields = {}
    for key, value in config.items():
        name = prefix + key
        if isinstance(value, dict):
            fields.update(flatten_config(value, name + "."))
        else:
            fields[name] = value
    return fields


def audit(root):
    protocol = read(root / "protocol.json")
    assert not protocol["smoke"]
    for name, expected in protocol["source_hashes"].items():
        assert sha(root / "evidence/source" / name) == expected, name
    results = {}
    last = protocol["steps"] - 1
    for spec in protocol["jobs"]:
        scene, seed = spec["scene"], spec["seed"]
        run = root / spec["id"]
        reference = Path(protocol["control"]) / scene / f"A04_s{seed}"
        record = read(run / "verified.json")
        control = read(reference / "verified.json")
        for key in ("initial_sha256", "camera_sha256"):
            assert record[key] == control[key], (scene, key)
        assert sha(run / "ckpts/final_post_update.pt") == record["final_checkpoint_sha256"]
        config = yaml.load((run / "cfg.yml").read_text(), Loader=yaml.BaseLoader)
        reference_config = yaml.load((reference / "cfg.yml").read_text(), Loader=yaml.BaseLoader)
        current_fields, reference_fields = map(flatten_config, (config, reference_config))
        config_changes = {
            key: [reference_fields[key], current_fields[key]]
            for key in current_fields.keys() & reference_fields.keys()
            if current_fields[key] != reference_fields[key]
        }
        if protocol["variant"] == "absgrad":
            assert config["strategy"]["absgrad"] == "true"
            assert float(config["strategy"]["grow_grad2d"]) == 0.0008
            assert config["strategy"]["fallback_rank"] == "off"
            assert set(config_changes) == {"result_dir", "strategy.absgrad", "strategy.grow_grad2d"}
        ply_count = None
        with (run / f"ply/point_cloud_{last}.ply").open("rb") as stream:
            for line in stream:
                if line.startswith(b"element vertex "):
                    ply_count = int(line.split()[-1])
                if line.strip() == b"end_header":
                    break
        assert ply_count == record["metrics"]["num_GS"]
        filenames = sorted(path.name for path in (reference / "renders").glob(f"val_step{last}_*.png"))
        assert filenames and filenames == sorted(
            path.name for path in (run / "renders").glob(f"val_step{last}_*.png")
        )
        views = []
        for filename in filenames:
            with Image.open(reference / "renders" / filename) as image:
                base = np.asarray(image)
            with Image.open(run / "renders" / filename) as image:
                candidate = np.asarray(image)
            assert candidate.shape == base.shape and base.shape[1] % 2 == 0
            width = base.shape[1] // 2
            assert np.array_equal(base[:, :width], candidate[:, :width]), filename
            base_psnr = png_psnr(base[:, :width], base[:, width:])
            candidate_psnr = png_psnr(candidate[:, :width], candidate[:, width:])
            assert math.isfinite(base_psnr) and math.isfinite(candidate_psnr)
            views.append(dict(file=filename, a04_psnr=base_psnr, candidate_psnr=candidate_psnr,
                              delta_psnr=candidate_psnr - base_psnr))
        initial = read(run / "stats/val_step0000.json")
        initial_control = read(reference / "stats/val_step0000.json")
        assert all(abs(initial[key] - initial_control[key]) < 0.0001
                   for key in ("psnr", "ssim", "lpips"))
        history = record["train"]["growth_history"]
        result = {
            "config_changes": config_changes,
            "new_config_fields": {key: current_fields[key]
                                  for key in current_fields.keys() - reference_fields.keys()},
            "initial_metric_delta": {key: initial[key] - initial_control[key]
                                     for key in ("psnr", "ssim", "lpips")},
            "gaussian_delta_pct": 100 * (record["metrics"]["num_GS"] / control["metrics"]["num_GS"] - 1),
            "peak_tensor_gib": record["train"]["mem"],
            "a04_peak_tensor_gib": control["train"]["mem"],
            "training_seconds": record["train"]["ellipse_time"],
            "a04_training_seconds": control["train"]["ellipse_time"],
            "wall_seconds": record["wall_seconds"],
            "growth_events": len(history),
            "total_splits": sum(row["split"] for row in history),
            "total_pruned": sum(row["pruned"] for row in history),
            "late_eligibility_limited_events": sum(row["eligible"] < row["target"] - row["before"]
                                                    for row in history if row["step"] >= 5000),
            "improved_views": sum(row["delta_psnr"] > 0 for row in views),
            "total_views": len(views),
            "mean_8bit_psnr_delta": float(np.mean([row["delta_psnr"] for row in views])),
            "png_view_results": views,
        }
        original = REPO / f"results/a05_training_20261107/replication/{scene}/A05_s{seed}/verified.json"
        if original.exists():
            old = read(original)
            assert old["initial_sha256"] == record["initial_sha256"]
            assert old["camera_sha256"] == record["camera_sha256"]
            result["delta_vs_residual_a05"] = {
                key: record["metrics"][key] - old["metrics"][key]
                for key in ("psnr", "ssim", "lpips", "num_GS")
            }
        results[scene] = result
        print(scene, "improved PNG views", result["improved_views"], "/", len(views), flush=True)
    write(root / "render_and_protocol_audit.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    audit(parser.parse_args().root.resolve())
