"""Audit training coverage, gradients and the frozen 3,001-update outlier."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from db3_g3_evaluate import (CHECKPOINT_UPDATES, GRADIENT_PARAMS, PAIR_PATH, compare,
                             load_frames, model_maps)
from db3_gaussian_depth_audit import load_splats, render
from db3_g4 import G3, MODES, freeze
from gsplat.losses import confidence_depth_l1_loss, ssim_loss
from gsplat.robust_sensor_depth import protected_sensor_depth
from week12_comparison import read, sha, write


def probe(model, frame, scale, extent, degree, coefficient, mode, outlier_xy=None):
    h, w = frame["depth"].shape
    pixels, alpha, _ = render(model, frame["pose"], frame["K"], w, h, scale, sh_degree=degree)
    sample = frame["sample"]
    xy = sample["points"].cuda()
    grid = (xy / xy.new_tensor([w-1,h-1]) * 2 - 1)[None,:,None]
    def sample_pixels(image):
        return F.grid_sample(image.permute(0,3,1,2), grid, align_corners=True).reshape(1,-1)
    pred = sample_pixels(pixels[...,3:])
    opacity = sample_pixels(alpha)
    target, conf = (sample[k][None].cuda() for k in ("depths","depth_confidence"))
    if mode in MODES:
        loss, support, active, terms = protected_sensor_depth(
            pred,target,conf,units_per_meter=scale,scene_scale=extent,min_depth_m=.3,
            residual_scale_inv_m=MODES[mode])
    else:
        support = (conf > 0) & torch.isfinite(conf) & torch.isfinite(target) & (target > 0)
        weight = conf if mode == "legacy" else torch.where(opacity.detach() > .7, conf, 0.)
        active = support & (weight > 0) & torch.isfinite(pred) & (pred > 0)
        loss = confidence_depth_l1_loss(pred,target,weight,scene_scale=extent)
        inv = torch.where(pred > 0, pred.clamp_min(1e-8).reciprocal(), 0.)
        terms = torch.where(weight > 0,(inv-target.reciprocal()).abs()*weight,0.)
    weighted = loss*coefficient
    color = pixels[...,:3]
    rgb = sample["image"][None].cuda().float()/255
    rgb_loss = torch.lerp(F.l1_loss(color,rgb),ssim_loss(color.permute(0,3,1,2),rgb.permute(0,3,1,2)),.2)
    params = [model[k] for k in GRADIENT_PARAMS]
    rgb_grads = torch.autograd.grad(rgb_loss,params,retain_graph=True)
    depth_grads = torch.autograd.grad(weighted,[*params,pred])
    ratios = {}
    for key, a, b in zip(GRADIENT_PARAMS,rgb_grads,depth_grads[:-1],strict=True):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        ratios[key] = float(b.norm()/a.norm()) if a.norm() else None
    assert torch.isfinite(depth_grads[-1]).all() and torch.isfinite(weighted)
    row = dict(name=frame["name"], evidence=int(support.sum()),active=int(active.sum()),
               alpha_above_07=int((support & (opacity > .7)).sum()), weighted_depth=float(weighted.detach()),
               largest_loss_share=float((terms.max()/terms.sum().clamp_min(1e-20)).detach()),
               depth_to_rgb_gradient=ratios)
    if outlier_xy:
        index = (xy == xy.new_tensor(outlier_xy)).all(dim=1).nonzero().flatten()
        assert index.numel() == 1
        i = int(index)
        row["outlier"] = dict(xy=outlier_xy,pred_m=float(pred[0,i].detach()/scale),
            alpha=float(opacity[0,i].detach()), target_m=float(target[0,i]/scale),
            active=bool(active[0,i]), share=float((terms[0,i]/terms.sum().clamp_min(1e-20)).detach()),
            weighted_dloss_dz_m=float(depth_grads[-1][0,i]*scale))
    return row


def history(run):
    events = EventAccumulator(str(run/"tb"),size_guidance={"scalars":0})
    events.Reload()
    arrays = {}
    for key in ("active_samples","evidence_samples","depth_loss","largest_loss_share"):
        values = events.Scalars("sensor/"+key)
        assert [v.step for v in values] == list(range(7000))
        arrays[key] = np.array([v.value for v in values])
        assert np.isfinite(arrays[key]).all()
    active, evidence = arrays["active_samples"],arrays["evidence_samples"]
    assert (active <= evidence).all() and (evidence > 0).all()
    windows = {}
    for i,j in ((0,7000),(3000,3001),(3001,3002),(3001,3011),(3011,3101),(3101,3501),(3501,4501)):
        portion = slice(i,j)
        windows[f"{i}:{j}"] = dict(active=int(active[portion].sum()),evidence=int(evidence[portion].sum()),
            min_frame_fraction=float((active/evidence)[portion].min()),
            largest_depth_loss=float(arrays["depth_loss"][portion].max()),
            peak_loss_step=i+int(arrays["depth_loss"][portion].argmax()),
            largest_loss_share=float(arrays["largest_loss_share"][portion].max()))
    return arrays, windows


def audit(out):
    freeze(out)
    frames,scale,_,_ = load_frames(read(PAIR_PATH))
    extent = read(G3/"meshes/G3-A/report.json")["scene_scale"]
    frozen = read(G3/"reset_outlier.json")
    step = frozen["replayed_step"]
    expected = read(G3/"evaluation/summary.json")["evaluation_input_sha256"]
    relative = Path(frozen["arm"],f"ckpts/ckpt_{step}_rank0.pt")
    assert sha(G3/relative) == expected[str(relative)]
    model = load_splats(G3/relative)
    for k in GRADIENT_PARAMS:
        model[k].requires_grad_()
    frame = next(f for f in frames["train"] if f["name"] == frozen["view"])
    xy = frozen["by_completed_updates"][str(step)]["xy"]
    comparison = {mode:probe(model,frame,scale,extent,3,.30,mode,xy)
                  for mode in ("legacy","opacity_only",*MODES)}
    assert np.isclose(comparison["legacy"]["weighted_depth"]/3,
                      frozen["by_completed_updates"][str(step)]["weighted_depth_loss"],rtol=1e-4)
    write(out/"same_state_reset.json",comparison)
    del model
    for arm in (*MODES,"G4-E"):
        run = out/arm
        verified = read(run/"verified.json")
        seq = read(run/"experiment.json")["camera_sequence"]
        coefficient = .10 if arm == "G4-E" else .30
        arrays, windows = history(run)
        np.savez_compressed(out/f"{arm}_training_samples.npz",**arrays,camera_indices=seq)
        result = {}
        for step in (*CHECKPOINT_UPDATES,7000):
            name = "final_post_update.pt" if step == 7000 else f"ckpt_{step}_rank0.pt"
            path = run/"ckpts"/name
            assert sha(path) == verified["checkpoint_sha256"][name]
            model = load_splats(path)  # Checks every saved parameter tensor.
            for k in GRADIENT_PARAMS:
                model[k].requires_grad_()
            degree = min(step//1000,3)
            rows = [probe(model,f,scale,extent,degree,coefficient,verified["mode"]) for f in frames["train"]]
            record = dict(fitting=rows)
            del model
            if step < 7000:
                actual = rows[seq[step]]["weighted_depth"]
                recorded = coefficient * arrays["depth_loss"][step]
                assert np.isclose(actual,recorded,rtol=2e-4,atol=1e-7), (arm,step,actual,recorded)
            if step in (3000,3001,4500,7000):
                maps,_ = model_maps(path,frames["val"],scale,degree,None)
                record["development"] = compare(frames["val"],{arm:maps})
                del maps
            result[str(step)] = record
        write(out/f"{arm}_recovery.json",dict(snapshots=result,windows=windows,probe_sha256=sha(Path(__file__))))
        print(f"Verified {arm} training telemetry and checkpoint gradients",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    audit(parser.parse_args().output.resolve())
