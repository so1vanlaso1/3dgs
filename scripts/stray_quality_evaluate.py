"""Evaluate A05-Abs quality checkpoints on distinct development and final views."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from gsplat import export_splats
from gsplat.exporter import load_ply_to_splats
from iphone_rescan_a05 import load_scan
from iphone_rescan_a05_evaluate import readback
from iphone_rescan_alignment import rgb_depth
from iphone_rescan_compare import poster_row, save_panel
import stray_painter as painter


def preview(out, renders, arm, images, frames, stem="final_comparison"):
    sample = [frames[i] for i in np.linspace(0, len(frames) - 1, 12).astype(int)]
    width, height = 270, 360
    for page, batch in enumerate((sample[:6], sample[6:]), 1):
        sheet = Image.new("RGB", (width * 6, (height + 26) * 2), "#192028")
        draw = ImageDraw.Draw(sheet)
        for i, frame in enumerate(batch):
            with Image.open(renders / arm / f"{frame:06}.png") as output:
                for row, (caption, image) in enumerate((("Source", Image.fromarray(images.native[frame])), ("A05-Abs", output))):
                    y = row * (height + 26)
                    rotated = image.transpose(Image.Transpose.ROTATE_270)
                    sheet.paste(rotated.resize((width, height), Image.Resampling.LANCZOS), (i * width, y))
                    draw.text((i * width + 7, y + height + 5), f"{caption}   {frame:06d}", fill="white")
        sheet.save(out / f"{stem}_{page}.jpg", quality=94)


def export_checkpoint(checkpoint_path, destination):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    epsilon = torch.finfo(saved["opacities"].dtype).eps
    values = {"means": saved["means"], "scales": saved["scales"].log(), "quats": saved["quats"],
              "opacities": saved["opacities"].clamp(torch.finfo(torch.float32).tiny, 1 - epsilon).logit(),
              "sh0": saved["sh"][:, :1], "shN": saved["sh"][:, 1:]}
    export_splats(**values, save_to=str(destination))
    exported = load_ply_to_splats(destination)
    for key, value in values.items():
        if not torch.equal(exported[key], value):
            raise ValueError(f"PLY round trip changed {key}")
    return {"filename": destination.name, "sha256": painter.sha(destination), "gaussians": len(saved["means"]),
            "bytes": destination.stat().st_size, "sh_degree": saved["sh_degree"]}


def score_surfaces(output, renders, arm, selected, images, references, regions):
    output.mkdir(exist_ok=True)
    with np.load(renders / "depth.npz") as file:
        depth = dict(zip(file["frames"], zip(file["z_m"], file["alpha"])))
    rows = []
    for index, region in enumerate(regions, 1):
        frame, name, box = region["frame"], region["surface"], region["box"]
        source = Image.fromarray(images.native[frame]).transpose(Image.Transpose.ROTATE_270).crop(box)
        with Image.open(renders / arm / f"{frame:06}.png") as whole:
            prediction = whole.transpose(Image.Transpose.ROTATE_270).crop(box)
        path = output / f"{index:02}_{frame:06}.png"
        save_panel(path, (source, prediction), (f"Recorded | {name}", "A05-Abs"), frame)
        row = poster_row(frame, arm, selected, np.asarray(source), np.asarray(prediction))
        x0, y0, x1, y1 = np.rint(np.array(box) / (1440 / 192)).astype(int)
        region_box = np.s_[y0:y1, x0:x1]
        z, a = depth[frame]
        z, a = (np.rot90(array, k=-1)[region_box] for array in (z, a))
        sensor, eligible, mesh_z, _ = references[frame]
        sensor, eligible, mesh_z = (np.rot90(array, k=-1)[region_box] for array in (sensor, eligible, mesh_z))
        hit = eligible & (a >= painter.ALPHA) & (z > 0) & np.isfinite(z)
        mesh_hit = eligible & np.isfinite(mesh_z)
        rows.append(row | {"surface": name, "native_crop": path.name, "depth_pixels": eligible.size,
                           "eligible_lidar": int(eligible.sum()), "covered_lidar": int(hit.sum()),
                           "within_50mm": int((np.abs((z - sensor)[hit]) <= .05).sum()),
                           "mesh_abs_median_mm": painter.error_stats(mesh_z[mesh_hit] - sensor[mesh_hit])["abs_median_mm"]}
                    | painter.error_stats((z - sensor)[hit]))
    painter.save_rows(output.with_suffix(".csv"), rows)


@torch.inference_mode()
def score(run, source, out, reuse_development=False):
    previous = json.loads((out / "summary.json").read_text()) if reuse_development else None
    verified = previous["readback"] if previous else readback(out)
    protocol = json.loads((out / "protocol.json").read_text())
    evaluation = json.loads((run / "splits/evaluation_protocol.json").read_text())
    size, images, split, _ = load_scan(run, load_mesh=False)
    view, k, depth_k = painter.camera_arrays(split, size)
    arm = protocol["arm"]
    development, final = evaluation["development"], evaluation["final"]
    assert sorted(development + final) == split["heldout"].tolist() and not set(development) & set(final)
    refs = {}
    with np.load(run / "geometry/heldout_mesh_depth.npz") as file:
        assert np.array_equal(file["frames"], split["heldout"])
        for frame, mesh_z in zip(file["frames"], file["depth_m"]):
            depth, confidence = rgb_depth(source, int(frame))
            eligible = (confidence == 2) & (depth >= 250) & (depth <= 5000)
            refs[int(frame)] = depth.astype(np.float32) * .001, eligible, mesh_z, np.zeros_like(eligible)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda().eval()

    def evaluate(step, frames, name):
        target = out / "evaluation" / name / str(step)
        target.mkdir(parents=True, exist_ok=True)
        print(f"Scoring {name} checkpoint {step:,} on {len(frames)} views.", flush=True)
        saved = torch.load(out / "checkpoints" / f"{arm}_{step}.pt", map_location="cuda", weights_only=True)
        state = tuple(saved[key] for key in ("means", "quats", "scales", "opacities", "sh"))
        subset = {frame: refs[frame] for frame in frames}
        metrics, maps, _ = painter.evaluate(arm, state, saved["sh_degree"], view, k, depth_k, images,
                                              subset, lpips, None, None, target, size=size,
                                              preview_frames=(), png_compression=1)
        np.savez_compressed(target / "depth.npz", frames=frames, z_m=np.stack([z for z, _ in maps.values()]),
                            alpha=np.stack([a for _, a in maps.values()]))
        metrics["depth"] = painter.depth_comparison(subset, {arm: maps}, target)[arm]["sensor_own"]
        metrics["mesh_reference"] = painter.mesh_readback(subset)
        for frame in frames:
            with Image.open(target / arm / f"{frame:06}.png") as image:
                assert image.size == size and image.mode == "RGB"
                image.verify()
        assert all(np.isfinite(z).all() and np.isfinite(a).all() for z, a in maps.values())
        painter.save_json(target / "metrics.json", metrics)
        return metrics

    scores = ({int(step): metrics for step, metrics in previous["development"].items()} if previous else
              {step: evaluate(step, development, "development") for step in evaluation["candidate_updates"]})
    best_depth = max(score["depth"]["within_50mm_fraction_of_eligible"] for score in scores.values())
    usable = [step for step, score in scores.items()
              if score["depth"]["within_50mm_fraction_of_eligible"] >= best_depth - .03]
    selected = min(usable, key=lambda step: (scores[step]["lpips_alex"], -scores[step]["psnr_db"]))
    if previous:
        assert selected == previous["selected_checkpoint"]
        result, export = previous["final"], previous["export"]
    else:
        result = evaluate(selected, final, "final")
        renders = out / "evaluation" / "final" / str(selected)
        preview(out, renders, arm, images, final)
        score_surfaces(out / "surface_details", renders, arm, selected, images, refs, evaluation["final_regions_upright_xyxy"])
        export = export_checkpoint(out / "checkpoints" / f"{arm}_{selected}.pt", out / "scene.ply")
    summary = {"selected_checkpoint": selected, "development_views": len(development), "final_views": len(final),
               "development": scores, "final": result, "readback": verified, "export": export}
    visual = min(scores, key=lambda step: (scores[step]["lpips_alex"], -scores[step]["psnr_db"]))
    if visual != selected:
        visual_result = evaluate(visual, final, "final")
        renders = out / "evaluation" / "final" / str(visual)
        preview(out, renders, arm, images, final, "final_visual_comparison")
        score_surfaces(out / "surface_details_visual", renders, arm, visual, images, refs, evaluation["final_regions_upright_xyxy"])
        visual_export = export_checkpoint(out / "checkpoints" / f"{arm}_{visual}.pt", out / "scene_visual.ply")
        summary["visual_checkpoint"] = {"update": visual, "final": visual_result, "export": visual_export}
    painter.save_json(out / "summary.json", summary)
    print(f"Verified checkpoint {selected:,} and all saved final renders.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reuse-development", action="store_true", help="Reuse a completed checkpoint evaluation")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    score(args.run, args.source, args.output, args.reuse_development)
