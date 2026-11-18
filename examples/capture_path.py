"""Replay reconstructed capture cameras in the same normalized space as training."""

import json
import math
import re
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class CapturePath:
    def __init__(self, parser, manifest=None, fps=15.0):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("capture_fps must be positive and finite")
        names = list(parser.image_names)
        if len(names) < 2:
            raise ValueError("Capture replay needs at least two registered cameras")
        candidates = [Path(manifest)] if manifest else [
            p / "capture_manifest.json"
            for p in [Path(parser.data_dir), *list(Path(parser.data_dir).parents)[:3]]
        ]
        stamps = None
        self.timing_source = f"Estimated timing at {fps:g} FPS (no matching manifest)"
        for candidate in candidates:
            if not candidate.is_file():
                if manifest:
                    raise FileNotFoundError(candidate)
                continue
            frames = json.loads(candidate.read_text(encoding="utf-8"))["frames"]
            lookup = {f["name"]: int(f["timestamp_ns"]) for f in frames}
            if all(name in lookup for name in names):
                stamps = np.array([lookup[name] for name in names], dtype=np.int64)
                self.timing_source = f"Recorded timestamps: {candidate.name}"
                break
            if manifest:
                raise ValueError("Capture manifest is missing registered image names")
        if stamps is not None:
            order = np.argsort(stamps)
            self.times = (stamps[order] - stamps[order[0]]).astype(float) / 1e9
        else:
            def natural_key(i):
                return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", names[i])]

            order = np.array(sorted(range(len(names)), key=natural_key))
            # Preserve skipped frame intervals for frame_000123-style captures.
            numbers = [re.search(r"(\d+)(?=\.[^.]+$)", names[i]) for i in order]
            indices = np.array([int(m[1]) for m in numbers]) if all(numbers) else np.arange(len(names))
            self.times = (indices - indices[0]) / fps
        if not np.all(np.isfinite(self.times)) or not np.all(np.diff(self.times) > 0):
            raise ValueError("Capture timestamps/frame indices must be strictly increasing")
        self.names = [names[i] for i in order]
        self.positions = np.asarray(parser.camtoworlds)[order, :3, 3]
        rotations = Rotation.from_matrix(np.asarray(parser.camtoworlds)[order, :3, :3])
        self.slerp = Slerp(self.times, rotations)
        self.fovs = np.array([
            2 * math.atan(parser.imsize_dict[parser.camera_ids[i]][1]
                          / (2 * parser.Ks_dict[parser.camera_ids[i]][1, 1]))
            for i in order
        ])
        self.duration = float(self.times[-1])

    def sample(self, seconds):
        seconds = float(np.clip(seconds, 0, self.duration))
        i = min(int(np.searchsorted(self.times, seconds, side="right")) - 1, len(self.times) - 2)
        a = (seconds - self.times[i]) / (self.times[i + 1] - self.times[i])
        position = (1 - a) * self.positions[i] + a * self.positions[i + 1]
        rotation = self.slerp(seconds).as_matrix()
        fov = float((1 - a) * self.fovs[i] + a * self.fovs[i + 1])
        return position, rotation, fov, self.names[i if a < 0.5 else i + 1]


class CapturePlayback:
    """Independent replay controls for each browser; paused cameras remain free."""

    def __init__(self, server, path):
        self.path = path
        self.sessions = {}
        self.lock = threading.RLock()
        server.on_client_connect(self.connect)

        @server.on_client_disconnect
        def disconnect(client):
            with self.lock:
                self.sessions.pop(client.client_id, None)

    def connect(self, client):
        path = self.path
        with client.gui.add_folder("Capture replay", order=-10):
            client.gui.add_markdown(
                f"Follow **{len(path.names):,} recovered camera poses** over "
                f"**{path.duration:.1f} seconds**.\n\n{path.timing_source}. "
                "Pause to look around freely. Selecting a time moves the camera smoothly."
            )
            play = client.gui.add_button("Play capture")
            pause = client.gui.add_button("Pause / explore")
            restart = client.gui.add_button("Return to start")
            timeline = client.gui.add_slider("Capture time (s)", min=0.0,
                max=path.duration, step=0.01, initial_value=0.0)
            speed = client.gui.add_slider("Speed", min=0.25, max=2.0,
                step=0.25, initial_value=1.0)
            loop = client.gui.add_checkbox("Loop replay", initial_value=False)
            show = client.gui.add_checkbox("Show capture path", initial_value=False)
            status = client.gui.add_markdown("Paused. Press Play capture to begin.")
        line = client.scene.add_line_segments("/capture_replay/path",
            points=np.stack([path.positions[:-1], path.positions[1:]], axis=1),
            colors=(255, 180, 35), thickness=2.0, thickness_units="screen", visible=False)
        state = dict(client=client, playing=False, seconds=0.0, last=time.monotonic(),
                     timeline=timeline, speed=speed, loop=loop, status=status, updating=False,
                     transition=None)

        @play.on_click
        def _(_event):
            with self.lock:
                if state["seconds"] >= path.duration:
                    state["seconds"] = 0.0
                state["transition"] = None
                state["playing"] = True
                state["last"] = time.monotonic()
                self.apply(state)

        @pause.on_click
        def _(_event):
            with self.lock:
                state["playing"] = False
                state["transition"] = None
                status.content = f"Paused at {state['seconds']:.2f}s. Explore freely."

        @restart.on_click
        def _(_event):
            with self.lock:
                self.seek(state, 0.0)

        @timeline.on_update
        def _(_event):
            with self.lock:
                if state["updating"]:
                    return
                self.seek(state, float(timeline.value))

        @show.on_update
        def _(_event):
            line.visible = show.value

        with self.lock:
            self.sessions[client.client_id] = state

    def seek(self, state, seconds):
        camera = state["client"].camera
        wxyz = np.asarray(camera.wxyz, dtype=float)
        start_rotation = Rotation.from_quat(wxyz[[1, 2, 3, 0]])
        position, rotation, fov, name = self.path.sample(seconds)
        state.update(playing=False, seconds=seconds, last=time.monotonic())
        state["transition"] = dict(start=state["last"], position=np.asarray(camera.position).copy(),
            fov=float(camera.fov), target_position=position, target_fov=fov, name=name,
            slerp=Slerp([0, 1], Rotation.from_quat([start_rotation.as_quat(), Rotation.from_matrix(rotation).as_quat()])))
        state["status"].content = f"Moving to {seconds:.2f}s · {name}"

    def set_camera(self, state, position, rotation, fov):
        client = state["client"]
        # Set exact OpenCV orientation, including roll, using the orbit-camera API.
        with client.atomic():
            client.camera.position = position
            client.camera.look_at = position + rotation[:, 2]
            client.camera.up_direction = -rotation[:, 1]
            client.camera.fov = fov

    def apply(self, state):
        position, rotation, fov, name = self.path.sample(state["seconds"])
        self.set_camera(state, position, rotation, fov)
        state["updating"] = True
        try:
            state["timeline"].value = state["seconds"]
        finally:
            state["updating"] = False
        state["status"].content = (
            f"{'Playing' if state['playing'] else 'Paused'}: "
            f"{state['seconds']:.2f} / {self.path.duration:.2f}s · {name}"
        )

    def tick(self):
        now = time.monotonic()
        with self.lock:
            for state in self.sessions.values():
                elapsed = now - state["last"]
                state["last"] = now
                transition = state["transition"]
                if transition is not None:
                    fraction = min(max((now - transition["start"]) / 3.0, 0.0), 1.0)
                    smooth = fraction * fraction * (3 - 2 * fraction)
                    self.set_camera(state,
                        (1-smooth)*transition["position"] + smooth*transition["target_position"],
                        transition["slerp"](smooth).as_matrix(),
                        (1-smooth)*transition["fov"] + smooth*transition["target_fov"])
                    if fraction >= 1:
                        state["transition"] = None
                        self.apply(state)
                    continue
                if not state["playing"]:
                    continue
                state["seconds"] += elapsed * state["speed"].value
                if state["seconds"] >= self.path.duration:
                    if state["loop"].value:
                        state["seconds"] %= self.path.duration
                    else:
                        state["seconds"] = self.path.duration
                        state["playing"] = False
                self.apply(state)
