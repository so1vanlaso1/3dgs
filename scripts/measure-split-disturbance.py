"""Compare immediate split disturbance on a frozen model, without recovery updates.

This diagnostic uses four held-out views and the same largest eligible parents
for both operators. It is not a measurement of disturbance during training.
"""

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from datasets.colmap import Dataset, Parser
from gsplat.rendering import rasterization
from gsplat.strategy.ops import split


@torch.no_grad()
def render(params, frame):
    image = frame["image"]
    height, width = image.shape[:2]
    rgb, _, _ = rasterization(
        means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(),
        colors=torch.cat([params["sh0"], params["shN"]], dim=1), sh_degree=3,
        viewmats=torch.linalg.inv(frame["camtoworld"].cuda())[None],
        Ks=frame["K"].cuda()[None], width=width, height=height,
        packed=True, near_plane=.01, far_plane=1e10)
    return rgb[0].clamp(0, 1)


def psnr(mse):
    return float(-10 * mse.clamp_min(1e-12).log10())


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--checkpoint", type=Path, required=True)
    cli.add_argument("--data-dir", default="data/room_factor2")
    cli.add_argument("--parents", type=int, default=10000)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.parents < 1:
        cli.error("--parents must be positive")
    parser = Parser(args.data_dir, factor=1, normalize=True, test_every=8, load_exposure=False)
    dataset = Dataset(parser, split="val")
    indices = sorted(set([0, len(dataset) // 3, 2 * len(dataset) // 3, len(dataset) - 1]))
    frames = [dataset[i] for i in indices]
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["splats"]
    scales = source["scales"].exp().max(-1).values
    eligible = (scales > .01 * parser.scene_scale * 1.1) & (source["opacities"].sigmoid() >= .005)
    candidates = eligible.nonzero().flatten()
    count = min(args.parents, len(candidates))
    if not count:
        raise ValueError("No eligible large parents in checkpoint")
    selected = candidates[scales[candidates].topk(count).indices]
    mask = torch.zeros(len(scales), dtype=torch.bool, device="cuda")
    mask[selected] = True
    params = torch.nn.ParameterDict({k: v.cuda() for k, v in source.items()})
    originals = [render(params, frame).cpu() for frame in frames]
    targets = [frame["image"].float() / 255 for frame in frames]
    initial_psnr = [psnr((rgb - gt).square().mean()) for rgb, gt in zip(originals, targets)]
    del params
    gc.collect()
    variants = []
    for mode, seed in [("random", s) for s in (42, 43, 44)] + [("long_axis", None)]:
        torch.manual_seed(42 if seed is None else seed)
        params = torch.nn.ParameterDict({k: v.cuda() for k, v in source.items()})
        opts = {k: torch.optim.Adam([v]) for k, v in params.items()}
        split(params, opts, {}, mask, mode=mode)
        views = []
        for index, frame, original, gt, before in zip(indices, frames, originals, targets, initial_psnr):
            rgb = render(params, frame).cpu()
            after = psnr((rgb - gt).square().mean())
            views.append(dict(val_index=index, before_psnr=before, after_psnr=after,
                              psnr_change=after-before,
                              rendered_rgb_mse=float((rgb-original).square().mean()),
                              rendered_rgb_l1=float((rgb-original).abs().mean())))
        variants.append(dict(mode=mode, seed=seed, views=views,
                             mean_psnr_change=sum(v["psnr_change"] for v in views)/len(views),
                             mean_rendered_rgb_mse=sum(v["rendered_rgb_mse"] for v in views)/len(views)))
        del params, opts
        gc.collect()
        torch.cuda.empty_cache()
    report = dict(checkpoint=str(args.checkpoint), parent_count=count,
                  parent_ids=selected.tolist(),
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  original_gaussians=len(scales), after_gaussians=len(scales)+count,
                  val_indices=indices, variants=variants,
                  selection="Largest scales among large (scale > 0.01*scene_scale), opacity >= 0.005 parents; identical mask per operator",
                  limitations=["Frozen final checkpoint and four held-out views, not actual training split events",
                               "No recovery optimizer updates; recovery speed is unmeasured",
                               "Random placement uses three seeds; LAS is deterministic",
                               "Large parents only: this isolates split geometry, not the removal of cloning"])
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"parent_count": count, "variants": [{k:v for k,v in r.items() if k!="views"} for r in variants]}, indent=2))


if __name__ == "__main__":
    main()
