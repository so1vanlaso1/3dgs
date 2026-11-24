"""Verify the rescan inputs, endpoint artifacts and fixed-anchor constraints."""

import json
from pathlib import Path
import sys

from PIL import Image
import numpy as np
import torch

from iphone_rescan_compare import ARMS, CHECKPOINTS
from stray_painter import save_json, sha


def verify(run):
    training = run / "training"
    protocol = json.loads((training / "protocol.json").read_text())
    size = tuple(protocol["rgb_size"])
    expected_depth = (protocol["depth_size"][1], protocol["depth_size"][0])
    frozen = json.loads((run / "splits/frame_split.json").read_text())
    fit, heldout, withheld = (set(frozen[key]) for key in ("fit", "heldout", "withheld"))
    sequence = np.load(training / "training_sequence.npy")
    assert fit.isdisjoint(withheld) and heldout.issubset(withheld)
    assert len(sequence) == CHECKPOINTS[-1] and set(sequence) == fit
    hashes = protocol["input_sha256"]
    for path, expected in hashes.items():
        assert sha(Path(path)) == expected, path
    for name, expected in protocol["source_sha256"].items():
        assert sha(training / name) == expected, name
    archive = json.loads((run / "dataset_audit/archive_sha256.json").read_text(encoding="utf-8-sig"))
    assert sha(Path(archive["Path"])) == archive["Hash"].lower()
    with np.load(training / "initialization.npz") as init:
        means, quats = (torch.from_numpy(init[key]) for key in ("points", "quats"))
    observations = {}
    for arm, directory in ARMS.items():
        root = run / directory
        losses = np.load(root / "checkpoints" / f"{arm}_loss.npy")
        assert len(losses) == CHECKPOINTS[-1] and np.isfinite(losses).all()
        observations[arm] = {}
        for step in CHECKPOINTS:
            model = root / "checkpoints" / f"{arm}_{step:04d}.pt"
            data = torch.load(model, map_location="cpu")
            target = root / "renders" / str(step)
            render_dir = target / arm
            assert data["sh_degree"] == min((step - 1) // 1000, 3)
            assert len(data["means"]) == protocol["gaussians"] == 120_000
            assert all(torch.isfinite(data[key]).all()
                       for key in ("means", "quats", "scales", "opacities", "sh"))
            if arm == "B1":
                assert torch.equal(data["means"], means) and torch.equal(data["quats"], quats)
                normal = protocol["initial_normal_scale_m"]
                assert torch.allclose(data["scales"][:, 2], torch.full((len(means),), normal), rtol=1e-6, atol=1e-9)
                scale = protocol["initial_tangent_scale_m"]
                tangent = data["scales"][:, :2]
                assert float(tangent.min()) >= .5 * scale - 1e-8
                assert float(tangent.max()) <= 2.5 * scale + 1e-8
            assert {int(path.stem) for path in render_dir.glob("*.png")} == heldout
            for path in render_dir.glob("*.png"):
                with Image.open(path) as image:
                    assert image.size == size
                    image.verify()
            with np.load(target / "depth.npz") as depth:
                assert depth["frames"].tolist() == frozen["heldout"]
                assert depth["alpha"].shape == depth["z_m"].shape == (len(heldout), *expected_depth)
                assert np.isfinite(depth["alpha"]).all() and np.isfinite(depth["z_m"]).all()
                assert np.all((depth["alpha"] >= 0) & (depth["alpha"] <= 1))
            observations[arm][step] = {
                "checkpoint_sha256": sha(model), "rgb_and_depth_frames": len(heldout),
                "mean_training_loss_to_update": float(losses[:step].mean()),
                "gaussians": len(data["means"]),
            }
    summary = json.loads((run / "comparisons/summary.json").read_text())["checkpoints"]
    for step in map(str, CHECKPOINTS):
        for key in ("sensor_common", "opaque_mesh_common"):
            b0, b1 = (summary[step]["depth"][arm][key] for arm in ARMS)
            assert b0["eligible"] == b1["eligible"] and b0["covered"] == b1["covered"] > 0
    assert len(list((run / "comparisons/depth_comparison/7000").glob("*.jpg"))) == len(heldout)
    save_json(run / "comparisons/readback.json", {
        "input_hashes_verified": len(hashes), "original_archive_hash_verified": True,
        "fit_frames": len(fit), "reserved_frames": len(withheld), "scored_frames": len(heldout),
        "b1_exact_centres_quaternions_and_scale_bounds_verified": True,
        "checkpoints": observations,
    })


if __name__ == "__main__":
    torch.set_num_threads(4)
    verify(Path(sys.argv[1]))
