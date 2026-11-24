"""Serve one or two exhibit checkpoints with an interactive camera."""

import argparse
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import viser
from nerfview import Viewer

from gsplat import rasterization


LABELS = ("B0  Free geometry", "B1  Fixed surface")
DEFAULT_PRESETS = ((99, "Display case"), (954, "Cabinet front"),
                   (1800, "Wire grid"), (2907, "Bright opening"))


class PainterViewer(Viewer):
    def _populate_rendering_tab(self):
        # Fixed rendering controls for the public preview. No server-side exports.
        self.render_tab_state.viewer_res = 1440
        self._rendering_folder.remove()


def load_model(path):
    checkpoint = torch.load(path, map_location="cuda", weights_only=True)
    parameters = tuple(checkpoint[name] for name in ("means", "quats", "scales", "opacities", "sh"))
    print(f"Loaded {path.name} with {len(parameters[0]):,} Gaussians", flush=True)
    return parameters, checkpoint["sh_degree"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("split", type=Path)
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--checkpoints", nargs="+", type=Path, metavar="CHECKPOINT",
                        default=(Path("B0.pt"), Path("B1.pt")), help="One or two checkpoint paths relative to run")
    parser.add_argument("--labels", nargs="+", help="Labels in checkpoint order")
    parser.add_argument("--preset", action="append", nargs=2, metavar=("FRAME", "LABEL"))
    parser.add_argument("--title", default="Stray Scanner exhibit")
    parser.add_argument("--portrait-capture", action="store_true", help="Use the wider sensor axis as the upright vertical field of view")
    args = parser.parse_args()
    if len(args.checkpoints) not in (1, 2):
        parser.error("Supply one or two checkpoints")
    labels = args.labels or (LABELS if len(args.checkpoints) == 2 else (args.checkpoints[0].stem,))
    if len(labels) != len(args.checkpoints):
        parser.error("Supply one label per checkpoint")
    models = [load_model(args.run / name) for name in args.checkpoints]
    with np.load(args.split) as split:
        positions = split["positions_m"]
        rotations = split["camera_to_world_rotation"]
        intrinsics = split["intrinsics_depth_px"]

    presets = [(int(frame), label) for frame, label in args.preset] if args.preset else DEFAULT_PRESETS
    if any(not 0 <= frame < len(positions) for frame, _ in presets):
        parser.error("Preset frame indices must refer to captured camera poses")

    def capture_camera(frame):
        position = positions[frame]
        half_extent, axis = (128, 0) if args.portrait_capture else (96, 1)
        return position, position + rotations[frame, :, 2], 2 * math.atan(half_extent / intrinsics[frame, axis])

    def render_model(model, camera, width, height):
        parameters, degree = model
        view = torch.tensor(np.linalg.inv(camera.c2w)[None], device="cuda", dtype=torch.float32)
        k = torch.tensor(camera.get_K((width, height))[None], device="cuda", dtype=torch.float32)
        rgb, _, _ = rasterization(*parameters, view, k, width, height,
                                  near_plane=.05, far_plane=20., packed=False, sh_degree=degree)
        return (rgb[0].clamp(0, 1) * 255).round().byte().cpu().numpy()

    @torch.inference_mode()
    def render(camera, state):
        width, height = state.viewer_width, state.viewer_height
        if len(models) == 1:
            return render_model(models[0], camera, width, height)
        side_by_side = width / max(height, 1) >= 1.25
        pane_width = max(1, width // 2 if side_by_side else width)
        pane_height = max(1, height if side_by_side else height // 2)
        output = Image.new("RGB", (width, height), "#121619")
        draw = ImageDraw.Draw(output)
        for index, (model, label) in enumerate(zip(models, labels)):
            pixels = render_model(model, camera, pane_width, pane_height)
            x, y = (index * pane_width, 0) if side_by_side else (0, index * pane_height)
            output.paste(Image.fromarray(pixels), (x, y))
            draw.rectangle((x, y, x + pane_width - 1, y + pane_height - 1), outline="#333c46")
            draw.rectangle((x + 8, y + 8, x + 20 + draw.textlength(label), y + 27), fill="#121619")
            draw.text((x + 14, y + 11), label, fill="white")
        return np.asarray(output).copy()

    server = viser.ViserServer(host="127.0.0.1", port=args.port, label=args.title)
    server.scene.set_up_direction("+y")
    position, target, fov = capture_camera(presets[0][0])
    server.initial_camera.position = position
    server.initial_camera.look_at = target
    server.initial_camera.up = (0., 1., 0.)
    server.initial_camera.fov = fov
    viewer = PainterViewer(server, render)
    server.gui.set_panel_label(labels[0] if len(models) == 1 else "Exhibit Gaussian comparison")
    with server.gui.add_folder("Captured viewpoints"):
        for frame, label in presets:
            button = server.gui.add_button(label)

            def jump(event, frame=frame):
                if event.client is None:
                    return
                position, target, fov = capture_camera(frame)
                with event.client.atomic():
                    event.client.camera.position = position
                    event.client.camera.look_at = target
                    event.client.camera.up_direction = (0., 1., 0.)
                    event.client.camera.fov = fov

            button.on_click(jump)
    print(f"LOCAL_VIEWER=http://127.0.0.1:{server.get_port()}", flush=True)
    if args.share:
        print(f"PUBLIC_VIEWER={server.request_share_url()}", flush=True)
    try:
        while viewer.state == "rendering":
            time.sleep(1)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
