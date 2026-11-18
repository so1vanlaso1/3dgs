"""Back-project a subject pixel and render its location from another capture pose."""
import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from capture_path import CapturePath
from datasets.colmap import Parser
from nerfview import CameraState
from gsplat.rendering import rasterization


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pixel", type=int, nargs=2, default=[493, 330])
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    parser = Parser(args.dataset, normalize=True)
    path = CapturePath(parser)
    a, b = path.sample(107.51), path.sample(228.11)
    splats = torch.load(args.checkpoint, map_location="cuda", weights_only=False)["splats"]
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    width, height = 1280, 720
    render_stats = {}

    @torch.inference_mode()
    def render(label, position, rotation, sh_degree=3, active_mask=None):
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, position
        state = CameraState(fov=a[2], aspect=width/height, c2w=pose)
        mask = slice(None) if active_mask is None else active_mask
        rgbd, alpha, _ = rasterization(splats["means"][mask], splats["quats"][mask],
            splats["scales"][mask].exp(), splats["opacities"][mask].sigmoid(), colors[mask],
            viewmats=torch.tensor(np.linalg.inv(pose)[None], device="cuda", dtype=torch.float32),
            Ks=torch.tensor(state.get_K((width,height))[None], device="cuda", dtype=torch.float32),
            width=width, height=height, near_plane=0.01, far_plane=100.,
            sh_degree=sh_degree, packed=False, render_mode="RGB+ED")
        rgbd = rgbd[0].cpu().numpy()
        render_stats[label] = dict(center_expected_depth=float(rgbd[height//2,width//2,3]),
            center_alpha=float(alpha[0,height//2,width//2,0]))
        iio.imwrite(out / f"{label}.png", np.uint8(np.clip(rgbd[..., :3],0,1)*255))
        return rgbd[..., 3], state.get_K((width,height)), alpha[0, ..., 0].cpu().numpy()

    depth, K, alpha = render("a_107", a[0], a[1])
    u, v = args.pixel
    z = float(np.median(depth[v-3:v+4, u-3:u+4]))
    target = a[0] + a[1] @ (np.linalg.inv(K) @ [u+0.5, v+0.5, 1.] * z)
    np.save(out / "a_expected_depth.npy", depth)

    def look_rotation(position):
        forward = target-position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, -a[1][:,1])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        return np.stack([right, down, forward], axis=1)

    render("b_228_aimed_at_subject", b[0], look_rotation(b[0]))
    render("a_107_sh0", a[0], a[1], sh_degree=0)
    render("b_228_aimed_at_subject_sh0", b[0], look_rotation(b[0]), sh_degree=0)
    subject_region = (splats["means"]-torch.tensor(target,device="cuda",dtype=torch.float32)).norm(dim=-1) < 0.10
    np.save(out / "subject_region_gaussian_ids.npy", torch.where(subject_region)[0].cpu().numpy())
    render("a_subject_region_only", a[0], a[1], active_mask=subject_region)
    render("b_subject_region_only", b[0], look_rotation(b[0]), active_mask=subject_region)
    samples = []
    for i, fraction in enumerate([0.1,0.25,0.5,0.75]):
        position = (1-fraction)*a[0]+fraction*b[0]
        render(f"toward_b_{i}_{fraction:g}", position, look_rotation(position))
        samples.append(dict(fraction=fraction,position=position.tolist()))
    report = dict(checkpoint=args.checkpoint, pixel=args.pixel,
        note="Target is a median expected-depth estimate, not a verified person Gaussian ID or ground-truth surface.",
        target=target.tolist(), depth=z, alpha_at_pixel=float(alpha[v,u]),
        camera_a_position=a[0].tolist(), camera_b_position=b[0].tolist(),
        source_units_per_normalized_unit=float(1/np.linalg.norm(parser.transform[:3,0])),
        target_distance_from_a=float(np.linalg.norm(target-a[0])),
        target_distance_from_b=float(np.linalg.norm(target-b[0])), samples=samples,
        render_stats=render_stats, subject_region_radius=0.10,
        subject_region_gaussian_count=int(subject_region.sum()))
    (out/"subject_trace.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
