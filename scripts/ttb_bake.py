"""Run the full DVSM-TTB checkpoint with bounded PyTorch attention."""

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import zipfile
from importlib.machinery import ModuleSpec
from pathlib import Path
from time import perf_counter
from types import ModuleType

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import numpy as np
import torch
import av
from huggingface_hub import hf_hub_download
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

DVSM_MODEL = "DVSM-TTB-p8-DL3DV-960p"
DVSM_CHECKPOINT = (
    "nvidia/DVSM", "327fc06b8e19c3462c8b0110c810f6700101be6b",
    "674c5a3236f1b7d6be56f2a144f8faf73c75a2a21c4df6e46e1dacfe0faa8add",
)
PI3_CHECKPOINT = (
    "yyfz233/Pi3", "ae722e7039287d0c8fde9f11f197f804f44b510c",
    "33580e4702ac671558aedeab1148fd08118f7ce45bdbeb99f3e3cf340062875d",
)


def checkpoint(cache, spec, subfolder=""):
    repo, revision, expected_hash = spec
    name = subfolder or repo.split("/")[-1]
    destination = cache / "checkpoints" / ("" if subfolder else name)
    directory = destination / subfolder
    weights = directory / "model.safetensors"
    if not (directory / "config.json").is_file() or not weights.is_file():
        print(f"[setup] Downloading {repo}/{subfolder or 'model.safetensors'} at {revision}")
        for filename in ("config.json", weights.name):
            hf_hub_download(repo, filename, subfolder=subfolder or None, revision=revision, local_dir=destination)
    digest = hashlib.sha256()
    with weights.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_hash:
        raise ValueError(f"Checkpoint SHA-256 mismatch. Remove the invalid file and retry: {weights}")
    return directory


def bounded_attention(q, k, v):
    params = torch.backends.cuda.SDPAParams(q, k, v, None, 0.0, False, False)
    if (torch.backends.cuda.can_use_flash_attention(params)
            or torch.backends.cuda.can_use_efficient_attention(params)):
        return F.scaled_dot_product_attention(q, k, v)

    # Keep the math backend's per-call FP32 scores bounded when flash is unavailable.
    heads = q.shape[1]
    query_count = max(1, min(512, 8_388_608 // (heads * k.shape[-2])))
    result = torch.empty_like(q)
    for view in range(q.shape[0]):
        batch = slice(view, view + 1)
        for start in range(0, q.shape[-2], query_count):
            queries = slice(start, start + query_count)
            result[batch, :, queries] = F.scaled_dot_product_attention(q[batch, :, queries], k[batch], v[batch])
    return result


def install_attention():
    flash = ModuleType("flash_attn")
    flash.__spec__ = ModuleSpec("flash_attn", loader=None)

    def flash_attention(q, k, v):
        return bounded_attention(*(tensor.transpose(1, 2) for tensor in (q, k, v))).transpose(1, 2)

    flash.flash_attn_func = flash_attention
    sys.modules["flash_attn"] = flash


def sample_video(video, directory, max_views):
    """Sample by decoded frame index, including videos with unreliable seek metadata."""
    directory.mkdir()
    with av.open(str(video)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {video}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        total = stream.frames or sum(1 for _ in container.decode(stream))

    def decode(total):
        selected = set(np.linspace(0, total - 1, min(total, max_views)).round().astype(int))
        paths = []
        observed = 0
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for index, frame in enumerate(container.decode(stream)):
                observed = index + 1
                if index in selected:
                    path = directory / f"frame_{index:08d}.png"
                    frame.to_image().save(path)
                    paths.append(path)
        return paths, observed

    images, observed = decode(total)
    if observed != total:
        print(f"[video] Stream reports {total} frames and decoded {observed}. Resampling.")
        for path in images:
            path.unlink()
        images, _ = decode(observed)
    print(f"[video] Decoded {observed} frames and selected {len(images)}.")
    return images


def estimate_scene(video, directory, max_views, long_edge, pose_size, pi3_path):
    from pi3.models.layers import attention
    from pi3.models.pi3 import Pi3
    from viser_app.pi3_estimator import pi3_to_scene_data, run_pi3
    from viser_app.recon_scene import derive_target_hw

    images = sample_video(video, directory, max_views)
    if len(images) < 2:
        raise ValueError(f"Decoded {len(images)} frame(s), but at least two views are required.")
    height, width = derive_target_hw(images[0], long_edge, 8)
    print(f"[pose] Estimating {len(images)} cameras. Bake resolution {width}x{height}.")
    pi3 = Pi3.from_pretrained(pi3_path).cuda().eval()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16

    def forward(images):
        with torch.autocast("cuda", dtype=dtype, cache_enabled=False):
            return pi3(images)

    # Pi3 can request flash-only attention for BF16. Enable the portable kernels
    # for its unmasked self-attention calls as well.
    original_attention = attention.scaled_dot_product_attention

    def pose_attention(q, k, v, *args, **kwargs):
        if args or kwargs:
            return original_attention(q, k, v, *args, **kwargs)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            return bounded_attention(q, k, v)

    attention.scaled_dot_product_attention = pose_attention
    estimate = run_pi3(forward, images, image_size=pose_size)
    if not np.isfinite(estimate["c2w"]).all() or not np.isfinite(estimate["intr"]).all():
        raise FloatingPointError("Pi3 returned non-finite camera estimates.")
    scene = pi3_to_scene_data(estimate, height, width)
    scene["scene_name"] = video.stem
    return scene


def check_fast_weights(renderer):
    for layer in renderer.kv_lst:
        if not isinstance(layer, dict) or not all(
            key in layer and torch.isfinite(layer[key]).all().item()
            for key in ("w0", "w1", "w2")
        ):
            raise FloatingPointError("A full TTB layer did not produce finite fast weights.")
    if not renderer.kv_lst:
        raise RuntimeError("The checkpoint produced no TTB layers.")


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "cache", "capture", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--member")
    parser.add_argument("--views", type=int)
    parser.add_argument("--long-edge", type=int)
    parser.add_argument("--pose-size", type=int)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA. Check nvidia-smi and the selected CUDA wheel.")
    device = torch.cuda.get_device_properties(0)
    free, _ = torch.cuda.mem_get_info()
    # Full cross-view attention scales quadratically with the combined token count.
    if free >= 20 * 2**30:
        defaults = 32, 960, 518
    elif free >= 10 * 2**30:
        defaults = 16, 720, 336
    else:
        defaults = 8, 384, 224
    views, long_edge, pose_size = (override or default for override, default in
                                  zip((args.views, args.long_edge, args.pose_size), defaults))
    print(f"[CUDA] {device.name}, {free / 2**30:.2f} GiB free, PyTorch {torch.__version__}")
    with (args.out / "experiment_log.md").open("a", encoding="utf-8") as log:
        log.write(
            f"- GPU: {device.name}, {free / 2**30:.2f} GiB free at launch, PyTorch {torch.__version__}\n"
            f"- Requested bake: up to {views} views, {long_edge}-pixel long edge, {pose_size}-pixel Pi3 input\n"
            f"- Full TTB checkpoint: `{DVSM_MODEL}` at `{DVSM_CHECKPOINT[1]}`\n"
            f"- Pi3 checkpoint revision: `{PI3_CHECKPOINT[1]}`\n"
        )
    if free < 4 * 2**30:
        raise RuntimeError(
            f"Only {free / 2**30:.2f} GiB of CUDA memory is free. "
            "Pi3's FP32 parameters alone use about 3.6 GiB. Free GPU memory and retry."
        )

    sys.path.insert(0, str(args.source))
    install_attention()
    from viser_app.recon_scene import reconstruct, save_artifact
    from ttb_fp32 import install as install_fp32

    install_fp32()
    dvsm = checkpoint(args.cache, DVSM_CHECKPOINT, DVSM_MODEL)
    pi3 = checkpoint(args.cache, PI3_CHECKPOINT)
    started = perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory(prefix="ttb-input-", dir=args.out) as temporary:
        directory = Path(temporary)
        video = args.capture
        if args.member:
            video = directory / "rgb.mp4"
            with zipfile.ZipFile(args.capture) as archive, archive.open(args.member) as member, video.open("wb") as dest:
                shutil.copyfileobj(member, dest, length=1024 * 1024)
        scene = estimate_scene(video, directory / "frames", views, long_edge, pose_size, pi3)
    torch.cuda.empty_cache()
    pose_seconds = perf_counter() - started
    started = perf_counter()
    renderer, batch, shape, c2w_r, c2w_t, intr = reconstruct(dvsm, "model.safetensors", scene, [])
    check_fast_weights(renderer)
    partial = args.out / "scene.recon.pt.partial"
    save_artifact(renderer, batch, shape, c2w_r, c2w_t, intr, dvsm, "model.safetensors", partial)
    partial.replace(args.out / "scene.recon.pt")
    with (args.out / "experiment_log.md").open("a", encoding="utf-8") as log:
        log.write(
            f"- Completed: {shape[1]} views at {shape[-1]} × {shape[-2]}, {len(renderer.kv_lst)} finite TTB layers\n"
            f"- Decode and pose estimation: {pose_seconds:.2f} s\n"
            f"- Bake, finite-weight check and artifact write: {perf_counter() - started:.2f} s\n"
            f"- Peak PyTorch CUDA allocation: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB\n"
            "- Artifact: `scene.recon.pt`\n"
        )


if __name__ == "__main__":
    main()
