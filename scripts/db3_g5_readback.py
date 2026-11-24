"""Independently verify the saved G5 depth scores and training seals."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

from db3_g4_readback import validate
from db3_g5 import ARMS, SEEDS
from week12_comparison import read, sha, write
from week12_mesh_review import sensor_masks


def check_row(row, residual, covered, stable):
    if residual.size:
        validate(row, residual, covered, stable)
    else:
        assert row["error"] is None
        np.testing.assert_allclose(row["coverage"], covered / stable, rtol=0, atol=1e-12)


def check_depth(folder, report):
    maps = {}
    for name, expected in report["maps_sha256"].items():
        path = folder / f"{Path(name).stem}_maps.npz"
        assert sha(path) == expected
        with np.load(path) as data:
            maps[name] = dict(data)
        arr = maps[name]
        np.testing.assert_array_equal(arr["stable"], sensor_masks(arr["sensor"])[1])
        for arm in ARMS:
            z = arr[f"{arm}_mesh"]
            np.testing.assert_array_equal(arr[f"{arm}_mesh_valid"], np.isfinite(z) & (z > 0))
    checks = 0
    for kind in ("render", "mesh"):
        tables = [(tuple(ARMS), report["all_arm"][kind])]
        tables.extend(((a, b), report["pairwise"][a + b][kind]) for a, b in combinations(ARMS, 2))
        for arms, table in tables:
            residuals = {a: [] for a in arms}
            covered = {a: 0 for a in arms}
            shared, total = 0, 0
            for name, arr in maps.items():
                valid = [arr[f"{a}_{kind}_valid"] for a in arms]
                mask = arr["stable"] & np.logical_and.reduce(valid)
                stable = int(arr["stable"].sum())
                assert table["per_view"][name]["shared_pixels"] == int(mask.sum())
                assert table["per_view"][name]["stable_pixels"] == stable
                total += stable
                shared += int(mask.sum())
                for arm in arms:
                    n = int((arr[f"{arm}_{kind}_valid"] & arr["stable"]).sum())
                    error = arr[f"{arm}_{kind}"][mask] - arr["sensor"][mask]
                    check_row(table["per_view"][name]["arms"][arm], error, n, stable)
                    residuals[arm].append(error)
                    covered[arm] += n
                    checks += 1
            assert table["stable_pixels"] == total and table["shared_pixels"] == shared
            for arm in arms:
                check_row(table["pooled"][arm], np.concatenate(residuals[arm]), covered[arm], total)
                checks += 1
    return checks


def readback(out):
    summary = read(out / "summary.json")
    counts = {}
    for seed in SEEDS:
        label = f"s{seed}"
        folder = out / label / "evaluation"
        path = folder / "scores.json"
        assert sha(path) == summary["seed_scores_sha256"][label]
        report = read(path)
        counts[label] = {split: check_depth(folder, report[split])
                         for split in ("development", "holdout") if split in report}
        for arm in ARMS:
            run = out / label / arm
            stats = read(run / "stats/val_step6999.json")
            rgb = report["development"]["all_arm"]["render"]["pooled"][arm]["rgb"]
            np.testing.assert_allclose([stats["psnr"], stats["ssim"]], [rgb["psnr_db"], rgb["ssim"]],
                                       atol=1e-4, rtol=0)
            assert stats["num_GS"] == report["counts"][arm]
            recovery = report["recovery"][arm]
            assert sha(folder / f"{arm}_scalars.npz") == recovery["scalar_sha256"]
    write(out / "readback.json", dict(depth_rows_by_seed=counts, trainer_rgb_and_counts_match=True,
                                     summary_sha256=sha(out / "summary.json"),
                                     reader_sha256=sha(Path(__file__))))
    print(f"Verified {sum(sum(values.values()) for values in counts.values())} depth/coverage rows "
          "and 20 trainer RGB/count endpoints.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    readback(parser.parse_args().output.resolve())
