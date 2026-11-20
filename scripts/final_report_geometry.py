"""Vary the matte TSDF grid while keeping checkpoints, camera samples and truncation fixed."""

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import open3d as o3d

from research_campaign import REPO, sha, write
from week12_comparison import ARMS, data_path, read, run_path
from week12_mesh_review import DB3, depth_statistics, matte_sources, sensor_masks
from datasets.colmap import Dataset, Parser


ARCHIVE = REPO / "results/week12_comparison_20261119"
GRIDS = (128, 192, 256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    pair_path = ARCHIVE / "matte_pair.json"
    pair = read(pair_path)
    source_protocol = matte_sources(pair)
    extractor = "scripts/gaussians_to_mesh.py"
    prior = {arm: pair["arms"][arm]["mesh"] for arm in ARMS}
    truncation = prior[ARMS[0]]["sdf_trunc"]
    protocol = {
        "grids": GRIDS, "fitting_views": pair["extraction_views"], "truncation": truncation,
        "pair_sha256": sha(pair_path), "extractor_sha256": sha(REPO / extractor),
        "analysis_sha256": sha(Path(__file__)), "sources": source_protocol,
        "checkpoint_sha256": {arm: prior[arm]["checkpoint_sha256"] for arm in ARMS},
        "support": "Stable aligned sensor depth and finite positive predictions from both "
                   "Gaussian checkpoints and all six mesh variants",
    }
    write(root / "protocol.json", protocol)
    variants = {}
    scenes = {}
    for arm in ARMS:
        ckpt = run_path(ARCHIVE, "matte", arm) / "ckpts/final_post_update.pt"
        assert sha(ckpt) == prior[arm]["checkpoint_sha256"]
        for grid in GRIDS:
            key = f"{arm}_{grid}"
            if grid == 192:
                folder = REPO / prior[arm]["folder"]
            else:
                folder = root / key
                command = [
                    sys.executable, "-u", extractor, "--checkpoint", str(ckpt),
                    "--data", str(data_path("matte")), "--output", str(folder),
                    "--views", "64", "--downsample", "4", "--voxel", "0.005",
                    "--max-resolution", str(grid), "--truncation", str(truncation),
                ]
                write(root / f"{key}_command.json", command)
                with folder.with_suffix(".log").open("w", encoding="utf-8") as log:
                    subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
            record = read(folder / "report.json")
            assert record["resolution"] == grid
            for param in ("checkpoint_sha256", "gaussian_count", "parser_transform", "scene_scale",
                          "tsdf_backend", "box_origin", "box_length", "alpha_threshold",
                          "sdf_trunc", "training_views"):
                assert record[param] == prior[arm][param], (key, param)
            mesh_path = folder / "raw_mesh.ply"
            assert sha(mesh_path) == record["mesh_sha256"]
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            _, counts, areas = map(np.asarray, mesh.cluster_connected_triangles())
            assert len(counts) == record["components"]
            variants[key] = {
                **record, "mesh_path": str(mesh_path), "largest_component_face_fraction": float(counts.max() / counts.sum()),
                "largest_component_area_fraction": float(areas.max() / areas.sum()),
                "watertight": record["boundary_edges"] == 0 and mesh.is_watertight(),
            }
            scene = scenes[key] = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            print("MESH", key, record["triangles"], flush=True)

    from db3_evaluate_pilot import load_splats, render

    loaded = {arm: load_splats(run_path(ARCHIVE, "matte", arm) / "ckpts/final_post_update.pt") for arm in ARMS}
    colmap = Parser(str(data_path("matte")), factor=1, normalize=True, test_every=8)
    dataset = Dataset(colmap, split="val")
    scale = float(np.linalg.norm(colmap.transform[:3, 0]))
    errors = {**{key: [] for key in scenes}, **{f"{arm}_gaussian": [] for arm in ARMS}}
    joint_total = stable_total = 0
    coverages = dict.fromkeys(errors, 0)
    frames = []
    for index, image_name in enumerate(pair["split"]["val"]):
        sample = dataset[index]
        assert image_name == colmap.image_names[dataset.indices[index]]
        k, c2w = sample["K"].numpy(), sample["camtoworld"].numpy()
        with np.load(DB3 / "depth" / f"{image_name}.npz") as source:
            measured = source["depth_m"]
        height, width = measured.shape
        _, stable = sensor_masks(measured)
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(k, np.linalg.inv(c2w), width, height)
        depths = {key: scene.cast_rays(rays)["t_hit"].numpy() / scale for key, scene in scenes.items()}
        support = {key: stable & np.isfinite(z) & (z > 0) for key, z in depths.items()}
        for arm, splats in loaded.items():
            _, depth, alpha = render(splats, c2w, k, width, height)
            key = f"{arm}_gaussian"
            depths[key] = depth / scale
            support[key] = stable & (alpha > 0.7) & np.isfinite(depth) & (depth > 0)
        joint = np.logical_and.reduce(list(support.values()))
        assert joint.any(), image_name
        if index == 0:
            diagnostic(root, image_name, depths, measured, joint)
        stable_total += int(stable.sum())
        joint_total += int(joint.sum())
        frame = {"name": image_name, "stable": int(stable.sum()), "joint": int(joint.sum()), "variants": {}}
        for key, depth in depths.items():
            coverage = int(support[key].sum())
            coverages[key] += coverage
            error = depth[joint] - measured[joint]
            errors[key].append(error)
            frame["variants"][key] = {"covered": coverage, **depth_statistics(error)}
        frames.append(frame)
    summary = {key: {"coverage": coverages[key] / stable_total, **depth_statistics(np.concatenate(values))}
               for key, values in errors.items()}
    write(root / "summary.json", {"joint_pixels": joint_total, "stable_pixels": stable_total,
                                 "variants": variants, "metrics": summary, "frames": frames})
    print("VERIFIED", joint_total, "common sensor pixels", flush=True)


def diagnostic(root, image_name, depths, measured, joint):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(9.25, 3.08), layout="constrained")
    colormap = plt.get_cmap("RdBu_r").with_extremes(bad="#e0e4e9")
    columns = ("gaussian", *GRIDS)
    for row, (arm, label) in enumerate(zip(ARMS, ("B00", "A05-Abs"), strict=True)):
        for col, variant in enumerate(columns):
            axis = axes[row, col]
            error = np.where(joint, 100 * (depths[f"{arm}_{variant}"] - measured), np.nan)
            panel = axis.imshow(error, vmin=-20, vmax=20, cmap=colormap, interpolation="nearest")
            axis.set(xticks=[], yticks=[])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title("Gaussian ED" if variant == "gaussian" else f"TSDF grid {variant}", fontsize=10)
            if col == 0:
                axis.set_ylabel(label, fontsize=10)
    bar = fig.colorbar(panel, ax=axes, shrink=.83, pad=.02, extend="both")
    bar.set_label("Predicted minus sensor depth (cm)", fontsize=9)
    bar.ax.tick_params(labelsize=8)
    fig.savefig(root / "heldout_grid_discrepancy.png", dpi=220)
    plt.close(fig)
    write(root / "heldout_grid_discrepancy.json", {"image": image_name, "clip_cm": [-20, 20]})


if __name__ == "__main__":
    main()
