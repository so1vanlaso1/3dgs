"""Freeze and run five contemporaneous, four-arm G5 replications."""

import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from db3_g3_depth_sweep import GATES, execute
from db3_g4 import G3
from week12_comparison import REPO, read, sha, write


SEEDS = (43, 44, 45, 46, 47)
ARMS = {"A": (.01, 0., 0.), "B": (.30, 0., 0.),
        "C": (.30, .30, 0.), "D": (.30, .30, .025)}
UPDATES = (3000, 3001, 3101, 4500, 7000)
FINAL = "ckpts/final_post_update.pt"
G4 = REPO / "results/db3_g4_20261121"


def argument(argv, flag, value):
    argv[argv.index(flag) + 1] = str(value)


def candidates():
    g3 = read(G3 / "protocol.json")
    sources = set(g3["source_sha256"]) | {
        "scripts/db3_g5.py", "scripts/db3_g3_evaluate.py", "scripts/db3_g4_evaluate.py",
        "scripts/db3_gaussian_depth_audit.py", "scripts/run-school-completion.ps1",
    }
    for folder in ("gsplat", "examples/datasets"):
        sources.update(str(p.relative_to(REPO)) for p in (REPO / folder).rglob("*")
                       if p.suffix in (".py", ".h", ".hpp", ".cuh", ".cu", ".cpp", ".inl"))
    return g3, sorted(REPO / path for path in sources)


def state(out):
    out.mkdir(parents=True, exist_ok=True)
    g3, sources = candidates()
    hashes = {str(p.relative_to(REPO)): sha(p) for p in sources}
    g4 = read(G4 / "protocol.json")
    for relative, expected in {**g3["input_sha256"], **g4["source_sha256"]}.items():
        if relative in g3["input_sha256"] or relative.startswith(("gsplat", "examples")):
            assert sha(REPO / relative) == expected, relative
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
         "--format=csv,noheader,nounits"], text=True).strip()
    environment = dict(
        python=sys.version, executable=sys.executable, platform=platform.platform(),
        torch=torch.__version__, torch_cuda=torch.version.cuda, gpu=driver,
        packages=sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()),
        settings={var: os.environ[var] for var in
                  ("CUDA_HOME", "TORCH_EXTENSIONS_DIR", "TORCH_CUDA_ARCH_LIST", "NUM_CHANNELS",
                   "OMP_NUM_THREADS", "PYTHONPATH")},
        toolchain={name: sha(Path(path)) for name, path in (
            ("cl", os.path.join(os.environ["VCToolsInstallDir"], "bin/Hostx64/x64/cl.exe")),
            ("nvcc", os.path.join(os.environ["CUDA_HOME"], "bin/nvcc.exe")),
        )},
    )
    training, extraction, sequences, orders = {}, {}, {}, {}
    for index, seed in enumerate(SEEDS):
        label = f"s{seed}"
        camera_path = out / f"{label}_cameras.json"
        rng = np.random.default_rng(seed)
        sequence = np.concatenate([rng.permutation(21) for _ in range(334)])[:7000].tolist()
        if camera_path.exists():
            assert read(camera_path) == sequence
        else:
            write(camera_path, sequence)
        sequences[label] = sha(camera_path)
        arms = list(ARMS)
        orders[label] = arms[index % len(arms):] + arms[:index % len(arms)]
        for arm, (coefficient, guard, robust) in ARMS.items():
            run = f"{label}/{arm}"
            argv = g3["training"]["G3-A"].copy()
            argv[0] = sys.executable
            argument(argv, "--seed", seed)
            argument(argv, "--result-dir", out / run)
            argument(argv, "--camera-sequence-path", camera_path)
            argument(argv, "--depth-lambda", coefficient)
            start, end = argv.index("--save-steps") + 1, argv.index("--save-ply")
            argv[start:end + 3] = [str(step + 1) for step in UPDATES[:-1]] + ["7000"]
            argv += ["--tb-every", "1"]
            if guard:
                argv += ["--sensor-min-pred-depth-m", str(guard),
                         "--sensor-residual-scale-inv-m", str(robust)]
            training[run] = argv
            argv = g3["extraction"]["G3-A"].copy()
            argv[0] = sys.executable
            argument(argv, "--checkpoint", out / run / FINAL)
            argument(argv, "--output", out / run / "mesh")
            extraction[run] = argv
    protocol = dict(
        seeds=SEEDS, arms=ARMS, checkpoint_updates=UPDATES, arm_order=orders,
        camera_sha256=sequences, input_sha256=g3["input_sha256"], source_sha256=hashes,
        g4_protocol_sha256=sha(G4 / "protocol.json"), environment=environment,
        training=training, extraction=extraction, original_gates=GATES,
        extra_per_view_p90_limit_cm=0.,
        endpoint_rule="D passes against A on every seed under original G3 gates plus no "
                      "per-view p90 increases. Score B vs A, C vs B, D vs B and D vs C separately.",
        recovery_rule="All objective scalars and saved parameters finite. At 4500 updates "
                      "pooled render coverage within 1 pp of 3000 and at least 99%. Measure "
                      "loss peaks and coverage at 3001, 3101, 4500 and final.",
        generalization_rule="Require separate evaluation on previously unused observations with "
                            "validated frozen poses and depth absent from construction. "
                            "Physical accuracy requires an independent reference.",
        archived_replay_interchangeability=False,
    )
    path = out / "protocol.json"
    if path.exists():
        import json

        assert read(path) == json.loads(json.dumps(protocol)), "Frozen G5 protocol changed"
    else:
        write(path, protocol)
        for source in sources:
            snapshot = out / "source" / source.relative_to(REPO)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(source.read_bytes())
    return protocol


def verify_run(run, seed, sequence):
    exp = read(run / "experiment.json")
    assert exp["seed"] == seed and exp["completed_updates"] == 7000
    assert exp["camera_sequence"] == sequence and exp["finite_parameters"]
    checkpoints = [run / "ckpts" / f"ckpt_{step}_rank0.pt" for step in (*UPDATES[:-1], 6999)]
    checkpoints.append(run / FINAL)
    return dict(experiment=exp, checkpoint_sha256={p.name: sha(p) for p in checkpoints})


def run_replications(out, protocol, seeds):
    for seed in seeds:
        label = f"s{seed}"
        camera_path = out / f"{label}_cameras.json"
        assert sha(camera_path) == protocol["camera_sha256"][label]
        sequence = read(camera_path)
        initial = set()
        for arm in protocol["arm_order"][label]:
            key = f"{label}/{arm}"
            run = out / key
            seal = run / "verified.json"
            run.parent.mkdir(exist_ok=True)
            if not seal.exists():
                if not (run / FINAL).exists():
                    if run.exists():
                        raise FileExistsError(f"Preserve partial run before retrying {key}: {run}")
                    print(f"Training {key}", flush=True)
                    start = time.perf_counter()
                    execute(protocol["training"][key], run.with_name(f"{arm}_train.log"))
                    write(run / "elapsed.json", {"training_wall_seconds": time.perf_counter() - start})
                training = verify_run(run, seed, sequence)
                mesh = run / "mesh"
                if not (mesh / "report.json").exists():
                    if mesh.exists():
                        raise FileExistsError(f"Preserve partial mesh before retrying {key}: {mesh}")
                    execute(protocol["extraction"][key], run / "mesh.log")
                report = read(mesh / "report.json")
                assert report["mesh_sha256"] == sha(mesh / "raw_mesh.ply")
                assert report["checkpoint_sha256"] == training["checkpoint_sha256"]["final_post_update.pt"]
                write(seal, dict(**training, mesh_sha256=report["mesh_sha256"]))
                print(f"Verified {key} ({training['experiment']['final_count']:,} Gaussians)", flush=True)
            initial.add(read(seal)["experiment"]["initial_sha256"])
        assert len(initial) == 1, f"Unmatched initializer for {label}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    cli = parser.parse_args()
    output = cli.output.resolve()
    run_replications(output, state(output), cli.seeds)
