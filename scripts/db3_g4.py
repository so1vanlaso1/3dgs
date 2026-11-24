"""Run the G4 matte depth experiment using frozen G3 endpoints as controls."""

import argparse
from pathlib import Path
import sys
import time

from db3_g3_depth_sweep import CHECKPOINT_UPDATES, GATES, REPLAY_TOLERANCES, execute
from week12_comparison import REPO, read, sha, write


G3 = REPO / "results/db3_g3_20261121"
BASE = "G4-A"
STRONG = "G4-B"
REFERENCES = {BASE: "G3-A", STRONG: "G3-D"}
MODES = {"G4-C": 0., "G4-D": .025}


def freeze(out):
    g3 = read(G3 / "protocol.json")
    summary = G3 / "evaluation/summary.json"
    assert sha(summary) == read(G3 / "evaluation/readback.json")["artifacts_sha256"][summary.name]
    scores = read(summary)
    hashes = scores["evaluation_input_sha256"]
    for path, digest in g3["input_sha256"].items():
        assert sha(REPO / path) == digest, path
    for arm in REFERENCES.values():
        path = Path(arm, "ckpts/final_post_update.pt")
        assert sha(G3 / path) == hashes[str(path)]
        assert sha(G3 / "meshes" / arm / "raw_mesh.ply") == scores["mesh_reports"][arm]["mesh_sha256"]
    sources = [*g3["source_sha256"], "gsplat/robust_sensor_depth.py", "scripts/db3_g4.py",
               "scripts/db3_g3_evaluate.py"]
    frozen = dict(name="G4 — Robust, geometry-preserving RGB-D supervision",
                  controls=REFERENCES, min_predicted_depth_m=.30, log1p_scale_inverse_m=MODES,
                  coefficients={BASE:.01, STRONG:.30, "G4-C":.30, "G4-D":.30, "G4-E":.10},
                  normalization="Sum of original positive valid sensor weights; low alpha alone never excludes a ray.",
                  selection=["fewest failed original geometry/RGB/coverage gates versus A",
                             "smallest worst per-view p90 rise versus A across both depth sources",
                             "smallest worst per-view PSNR loss versus A", "G4-C on an exact tie"],
                  gates=GATES, replay_tolerances=REPLAY_TOLERANCES,
                  special_tail="Report every per-view p90 change separately; a worsening tail precludes claiming a worst-case fix.",
                  control_scope="G4-A and G4-B reuse exact frozen G3 checkpoints and meshes. "
                  "Archive replay is a separate diagnostic. Independent reproduction is required for promotion.",
                  g3_protocol_sha256=sha(G3 / "protocol.json"), g3_summary_sha256=sha(summary),
                  source_sha256={p:sha(REPO / p) for p in sources})
    path = out / "protocol.json"
    if path.exists():
        assert frozen == read(path), "G4 protocol or source changed"
    else:
        write(path, frozen)
    return g3


def run(out, arms):
    g3 = freeze(out)
    for arm in arms:
        folder = out / arm
        mesh = out / "meshes" / arm
        reference = read(G3 / REFERENCES[BASE] / "verified.json")
        if arm == "G4-E":
            mode = read(out / "selection.json")["selected"]
        else:
            mode = arm
        coefficient = .10 if arm == "G4-E" else .30
        args = g3["training"]["G3-D"].copy()
        args[0] = sys.executable
        args[args.index("--result-dir")+1] = str(folder)
        args[args.index("--depth-lambda")+1] = str(coefficient)
        args += ["--sensor-min-pred-depth-m", "0.30", "--sensor-residual-scale-inv-m", str(MODES[mode])]
        extraction = g3["extraction"]["G3-D"].copy()
        extraction[0] = sys.executable
        model = folder / "ckpts/final_post_update.pt"
        extraction[extraction.index("--checkpoint")+1] = str(model)
        extraction[extraction.index("--output")+1] = str(mesh)
        if not (folder / "verified.json").exists():
            if folder.exists():
                raise FileExistsError(folder)
            write(out / f"{arm}_commands.json", dict(train=args, mesh=extraction))
            start = time.perf_counter()
            execute(args, out / f"{arm}_train.log")
            exp = read(folder / "experiment.json")
            for key in ("initial_sha256", "camera_sequence", "seed", "completed_updates"):
                assert exp[key] == reference[key], (key, arm)
            assert exp["finite_parameters"]
            paths = [folder / f"ckpts/ckpt_{step}_rank0.pt" for step in CHECKPOINT_UPDATES] + [model]
            assert all(p.is_file() for p in paths)
            write(folder / "verified.json", dict(seconds=time.perf_counter()-start,
                  checkpoint_sha256={p.name:sha(p) for p in paths}, mode=mode))
        if not (mesh / "report.json").exists():
            if mesh.exists():
                raise FileExistsError(mesh)
            mesh.parent.mkdir(parents=True, exist_ok=True)
            execute(extraction, out / f"{arm}_mesh.log")
        report = read(mesh / "report.json")
        assert report["checkpoint_sha256"] == sha(model) and report["mesh_sha256"] == sha(mesh / "raw_mesh.ply")
        print(f"Verified {arm}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("arms", nargs="+", choices=[*MODES, "G4-E"])
    args = parser.parse_args()
    run(args.output.resolve(), args.arms)
