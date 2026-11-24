"""Compare G4 endpoints on pairwise common support and score the archive replays."""

import argparse
from pathlib import Path

import numpy as np
import yaml

from db3_g3_depth_sweep import ARCHIVE
from db3_g3_evaluate import (PAIR_PATH, candidate_checks, compare, load_frames,
                             mesh_maps, model_maps, replay_check)
from db3_g4 import BASE, G3, MODES, REFERENCES, STRONG, freeze
from week12_comparison import read, sha, write


def config(run):
    return yaml.load((run / "cfg.yml").read_text(), Loader=yaml.BaseLoader)


def endpoint(run, mesh, dev, scale, reference):
    path = run / "ckpts/final_post_update.pt"
    report = read(mesh / "report.json")
    assert sha(path) == report["checkpoint_sha256"] and sha(mesh/"raw_mesh.ply") == report["mesh_sha256"]
    for key in ("parser_transform", "scene_scale", "tsdf_backend", "resolution", "box_origin", "box_length",
                "voxel_length", "sdf_trunc", "alpha_threshold"):
        assert report[key] == reference[key], key
    for key in ("train_index", "image_id"):
        assert [v[key] for v in report["training_views"]] == [v[key] for v in reference["training_views"]]
    rendered, state = model_maps(path, dev, scale, 3, None)
    assert state["gaussians"] == report["gaussian_count"]
    return dict(render=rendered, mesh=mesh_maps(mesh/"raw_mesh.ply", dev, scale)), state["gaussians"]


def archive(out, dev, scale, pair):
    info = pair["arms"]["B00"]["mesh"]
    original = Path(info["checkpoint"]).parents[1]
    ref, n = endpoint(original, ARCHIVE.parents[1] / info["folder"], dev, scale, info)
    expected = read(original/"experiment.json")
    results = {}
    replicas = {"archive_replay": out/"archive_replay",
                "driver_replay": out/"driver_replay/replication/matte/B00_s42"}
    for arm, run in replicas.items():
        if not (out/"meshes"/arm/"report.json").exists():
            continue
        exp = read(run/"experiment.json")
        for key in ("seed", "completed_updates", "initial_sha256", "camera_sequence", "finite_parameters"):
            assert exp[key] == expected[key]
        orig_cfg, new_cfg = config(original), config(run)
        for field in ("result_dir", "data_dir", "camera_sequence_path", "sensor_depth_dir"):
            orig_cfg.pop(field)
            new_cfg.pop(field)
        assert new_cfg == orig_cfg
        maps, count = endpoint(run, out/"meshes"/arm, dev, scale, info)
        paired = {k:compare(dev, {arm:maps[k], "B00":ref[k]}) for k in ref}
        counts = {arm:count, "B00":n}
        results[arm] = dict(comparison=paired, counts=counts,
                            replay=replay_check(paired, counts, arm, "B00"))
    write(out/"archive_scores.json", results)


def evaluate(out, stage):
    freeze(out)
    pair = read(PAIR_PATH)
    frames, scale, _, _ = load_frames(pair)
    dev = frames["val"]
    if stage == "archive":
        archive(out, dev, scale, pair)
        return
    arms = [*REFERENCES, *MODES] + (["G4-E"] if stage == "final" else [])
    reference = read(G3/"meshes/G3-A/report.json")
    maps, counts = {}, {}
    for arm in arms:
        root, source = (G3, REFERENCES[arm]) if arm in REFERENCES else (out, arm)
        run, mesh = root/source, root/"meshes"/source
        if arm not in REFERENCES:
            cfg, control = config(run), config(G3/"G3-D")
            mode = read(run/"verified.json")["mode"]
            assert float(cfg.pop("sensor_min_pred_depth_m")) == .3
            assert float(cfg.pop("sensor_residual_scale_inv_m")) == MODES[mode]
            assert float(cfg["depth_lambda"]) == (.1 if arm == "G4-E" else .3)
            for field in ("depth_lambda", "result_dir"):
                cfg.pop(field)
                control.pop(field)
            assert cfg == control, arm
        maps[arm], counts[arm] = endpoint(run, mesh, dev, scale, reference)
    def paired(kind, anchor):
        return {arm:compare(dev, {anchor:maps[anchor][kind], arm:maps[arm][kind]})
                for arm in arms if arm != anchor}
    versus_a = {k:paired(k, BASE) for k in ("render", "mesh")}
    versus_b = {k:paired(k, STRONG) for k in ("render", "mesh")}
    # G3 archive replay misses remain recorded; never infer independence from reference reuse.
    checks = candidate_checks(versus_a["render"], versus_a["mesh"], {"passed":False}, BASE)
    risks, ranks = {}, {}
    for arm, check in checks.items():
        risks[arm] = {kind:{name:100*(row["arms"][arm]["error"]["p90_abs_m"] -
                                      row["arms"][BASE]["error"]["p90_abs_m"])
                            for name,row in comparisons[arm]["per_view"].items()}
                      for kind,comparisons in versus_a.items()}
        rgb = versus_a["render"][arm]["per_view"]
        ranks[arm] = [len(check["failures"]), max(v for r in risks[arm].values() for v in r.values()),
                      max(r["arms"][BASE]["rgb"]["psnr_db"]-r["arms"][arm]["rgb"]["psnr_db"]
                          for r in rgb.values())]
    dose_pair = {}
    if stage == "select":
        choice = min(MODES, key=lambda arm:(*ranks[arm], arm))
        write(out/"selection.json", dict(selected=choice, ranking={a:ranks[a] for a in MODES},
                                         gates={a:checks[a] for a in MODES}, p90_change_cm=risks))
    else:
        selected = read(out/"selection.json")["selected"]
        assert min(MODES, key=lambda arm:(*ranks[arm],arm)) == selected
        dose_pair = {k:compare(dev,{a:maps[a][k] for a in (selected,"G4-E")}) for k in ("render","mesh")}
        for f in dev:
            name = f["name"]
            layers = dict(stable=f["stable"], sensor=f["depth"], rgb=f["sample"]["image"].numpy())
            for arm in arms:
                for kind in ("render","mesh"):
                    m = maps[arm][kind][name]
                    layers[f"{arm}_{kind}"] = m["depth"]
                    layers[f"{arm}_{kind}_valid"] = m["valid"]
                layers[f"{arm}_alpha"] = maps[arm]["render"][name]["alpha"]
                layers[f"{arm}_rgb8"] = maps[arm]["render"][name]["color"]
            np.savez_compressed(out/f"{Path(name).stem}_final.npz", **layers)
    all_arm = {k:compare(dev,{a:maps[a][k] for a in arms}) for k in ("render","mesh")}
    write(out/f"{stage}_scores.json", dict(all_arm=all_arm, versus_A=versus_a, versus_B=versus_b,
                                          counts=counts, checks=checks, p90_change_cm=risks, dose_pair=dose_pair,
                                          evaluator_sha256=sha(Path(__file__))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("stage", choices=["select","final","archive"])
    args = parser.parse_args()
    evaluate(args.output.resolve(), args.stage)
