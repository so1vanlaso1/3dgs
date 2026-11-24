"""Replay the frozen matte B00 protocol with four depth coefficients."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from week12_comparison import MATTE, REPO, read, sha, write


ARCHIVE = REPO / "results/week12_comparison_20261119"
COEFFICIENTS = {"G3-A": 0.01, "G3-B": 0.03, "G3-C": 0.10, "G3-D": 0.30}
# The trainer saves before its optimizer step. Its final_post_update has 7000 updates.
CHECKPOINT_UPDATES = (0, 100, 500, 501, 1000, 2000, 3000, 3001, 4500, 5901, 6000, 6500)
GATES = dict(min_signed_bias_gain_cm=1.0, min_absolute_median_gain_cm=1.0,
             max_absolute_p90_increase_cm=0.0, max_psnr_loss_db=0.5, max_ssim_loss=0.01,
             max_render_coverage_loss_pp=1.0, max_mesh_coverage_loss_pp=1.0,
             require_all_three_view_medians_to_improve=True)
REPLAY_TOLERANCES = dict(max_depth_stat_difference_cm=0.25,
                         max_psnr_difference_db=0.15, max_ssim_difference=0.005,
                         max_coverage_difference_pp=0.5, max_count_relative_difference=0.05)


def training_protocol(out):
    source = ARCHIVE / "matte_training_protocol.json"
    archived = read(source)
    assert archived["seed"] == 42 and archived["updates"] == 7000
    for relative, digest in archived["input_sha256"].items():
        assert sha(REPO / relative) == digest, relative
    commands = {}
    mesh_commands = {}
    for arm, coefficient in COEFFICIENTS.items():
        args = archived["commands"]["B00"].copy()
        args[0] = sys.executable
        args[args.index("--result-dir") + 1] = str(out / arm)
        args[args.index("--depth-lambda") + 1] = str(coefficient)
        index = args.index("--save-steps") + 1
        assert args[index] == "7000"
        args[index:index] = [str(step + 1) for step in CHECKPOINT_UPDATES]
        commands[arm] = args
        mesh_commands[arm] = [
            sys.executable, "-u", "scripts/gaussians_to_mesh.py",
            "--checkpoint", str(out / arm / "ckpts/final_post_update.pt"),
            "--data", str(MATTE / "dataset"), "--output", str(out / "meshes" / arm),
            "--views", "64", "--downsample", "4", "--voxel", "0.01", "--max-resolution", "192",
        ]
    source_paths = [*archived["source_sha256"], "scripts/gaussians_to_mesh.py",
                    "scripts/db3_g3_depth_sweep.py"]
    source_hashes = {path: sha(REPO / path) for path in source_paths}
    drift = {path: dict(archive=digest, current=source_hashes[path])
             for path, digest in archived["source_sha256"].items() if source_hashes[path] != digest}
    assert set(drift) <= {"scripts/week12_comparison.py", "examples/simple_trainer.py"}
    frozen = dict(seed=42, updates=7000, checkpoint_updates=[*CHECKPOINT_UPDATES, 7000],
                  coefficients=COEFFICIENTS, training=commands, extraction=mesh_commands,
                  gates=GATES, replay_tolerances=REPLAY_TOLERANCES,
                  comparison="Pairwise shared stable sensor support against G3-A plus four-arm shared support; "
                  "own stable-depth coverage; full-frame RGB. Depth gates apply to both rendered depth and mesh first hits. "
                  "Coverage and RGB limits apply pooled and per dev view. Finite throughout sampled checkpoints required. "
                  "Eligibility requires quantitative baseline replay; this is a single-seed development screen.",
                  archived_training_protocol_sha256=sha(source), input_sha256=archived["input_sha256"],
                  source_sha256=source_hashes, archived_source_drift=drift)
    path = out / "protocol.json"
    if path.exists():
        assert read(path) == frozen, "Protocol changed since launch"
    else:
        write(path, frozen)
    return frozen


def execute(args, log):
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(args, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)


def train(out, arms):
    out.mkdir(parents=True, exist_ok=True)
    protocol = training_protocol(out)
    archive_run = Path(read(ARCHIVE / "matte_training_protocol.json")["commands"]["B00"][
        protocol["training"]["G3-A"].index("--result-dir") + 1])
    original = read(archive_run / "experiment.json")
    sequence = read(MATTE / "camera_sequence.json")
    assert len(sequence) == 7000 and len(set(sequence)) == 21
    assert sequence == original["camera_sequence"]
    for arm in arms:
        folder = out / arm
        if not (folder / "verified.json").is_file():
            if folder.exists():
                raise FileExistsError(f"Existing unfinished training directory: {folder}")
            print(f"Training {arm}", flush=True)
            start = time.perf_counter()
            execute(protocol["training"][arm], out / f"{arm}_train.log")
            experiment = read(folder / "experiment.json")
            for key in ("seed", "initial_sha256", "camera_sequence_sha256", "completed_updates"):
                assert experiment[key] == original[key], (arm, key)
            assert experiment["finite_parameters"] and experiment["camera_sequence"] == sequence
            assert all((folder / f"ckpts/ckpt_{update}_rank0.pt").is_file()
                       for update in CHECKPOINT_UPDATES)
            write(folder / "verified.json", dict(seconds=time.perf_counter() - start, **experiment))
        mesh = out / "meshes" / arm
        if not (mesh / "report.json").is_file():
            if mesh.exists():
                raise FileExistsError(f"Existing unfinished extraction directory: {mesh}")
            mesh.parent.mkdir(exist_ok=True)
            print(f"Extracting {arm}", flush=True)
            execute(protocol["extraction"][arm], mesh.with_suffix(".log"))
        report = read(mesh / "report.json")
        assert report["mesh_sha256"] == sha(mesh / "raw_mesh.ply")
        assert report["checkpoint_sha256"] == sha(folder / "ckpts/final_post_update.pt")
        print(f"Verified {arm}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--arms", nargs="+", choices=COEFFICIENTS, default=list(COEFFICIENTS))
    arguments = parser.parse_args()
    train(arguments.output.resolve(), arguments.arms)
