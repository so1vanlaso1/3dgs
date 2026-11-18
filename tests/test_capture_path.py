"""Capture timing, pose interpolation, and replay lifecycle checks (no GPU)."""

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from capture_path import CapturePath, CapturePlayback


def parser(tmp_path):
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = 4
    poses[1, :3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    return SimpleNamespace(data_dir=str(tmp_path),
        image_names=["frame_000000.jpg", "frame_000060.jpg"], camtoworlds=poses,
        camera_ids=[1, 1], imsize_dict={1: (1280, 720)},
        Ks_dict={1: np.array([[600., 0, 640], [0, 600, 360], [0, 0, 1]])})


def manifest(tmp_path, stamps):
    target = tmp_path / "capture_manifest.json"
    target.write_text(json.dumps({"frames": [
        {"name": name, "timestamp_ns": stamp}
        for name, stamp in zip(["frame_000000.jpg", "frame_000060.jpg"], stamps)
    ]}))
    return target


def test_recorded_timing_overrides_fps_and_interpolates_rotation(tmp_path):
    manifest(tmp_path, [1700000000000000000, 1700000002000000000])
    path = CapturePath(parser(tmp_path), fps=15)
    assert path.duration == 2
    pos, rot, fov, _ = path.sample(1)
    np.testing.assert_allclose(pos, [2, 0, 0])
    np.testing.assert_allclose(rot, Rotation.from_euler("z", 45, degrees=True).as_matrix(), atol=1e-12)
    assert fov == pytest.approx(2 * np.arctan(720 / 1200))
    np.testing.assert_allclose(path.sample(-1)[0], [0, 0, 0])
    np.testing.assert_allclose(path.sample(20)[0], [4, 0, 0])


def test_timestamp_order_reorders_poses(tmp_path):
    manifest(tmp_path, [2000000000, 1000000000])
    path = CapturePath(parser(tmp_path))
    assert path.names[0] == "frame_000060.jpg"
    np.testing.assert_allclose(path.sample(0)[0], [4, 0, 0])


def test_estimated_timing_preserves_missing_frame_intervals(tmp_path):
    path = CapturePath(parser(tmp_path), fps=15)
    assert path.duration == 4
    assert "Estimated" in path.timing_source


def test_duplicate_timestamps_rejected(tmp_path):
    manifest(tmp_path, [1, 1])
    with pytest.raises(ValueError, match="strictly increasing"):
        CapturePath(parser(tmp_path))


def test_explicit_incomplete_manifest_rejected(tmp_path):
    target = tmp_path / "incomplete.json"
    target.write_text('{"frames": []}')
    with pytest.raises(ValueError, match="missing registered"):
        CapturePath(parser(tmp_path), manifest=target)


class Handle:
    def __init__(self, value=None):
        self._value = value
        self.callbacks = []
        self.visible = False
        self.content = ""

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self.fire()

    def on_update(self, callback):
        self.callbacks.append(callback)

    on_click = on_update

    def fire(self):
        for callback in self.callbacks:
            callback(None)


class Client:
    def __init__(self, client_id):
        self.client_id = client_id
        self.gui = self.scene = self
        self.camera = SimpleNamespace(position=np.zeros(3), wxyz=np.array([1., 0, 0, 0]), fov=1.0)
        self.handles = {}

    def atomic(self):
        return nullcontext()

    def add_folder(self, *args, **kwargs):
        return nullcontext()

    def add(self, label, **kwargs):
        handle = Handle(kwargs.get("initial_value"))
        self.handles[label] = handle
        return handle

    add_button = add_checkbox = add_slider = add_markdown = add_line_segments = add


def test_play_pause_seek_loop_and_independent_clients(tmp_path, monkeypatch):
    server = SimpleNamespace(on_client_connect=lambda cb: None, on_client_disconnect=lambda cb: None)
    playback = CapturePlayback(server, CapturePath(parser(tmp_path), fps=15))
    first, second = Client(1), Client(2)
    playback.connect(first)
    playback.connect(second)
    now = [100.0]
    monkeypatch.setattr("capture_path.time.monotonic", lambda: now[0])
    first.handles["Play capture"].fire()
    now[0] += 1
    playback.tick()
    assert playback.sessions[1]["seconds"] == 1
    assert playback.sessions[2]["seconds"] == 0
    np.testing.assert_allclose(first.camera.position, [1, 0, 0])
    first.handles["Pause / explore"].fire()
    now[0] += 1
    playback.tick()
    assert playback.sessions[1]["seconds"] == 1
    first.handles["Capture time (s)"].value = 3.5
    np.testing.assert_allclose(first.camera.position, [1, 0, 0])
    now[0] += 1.5
    playback.tick()
    np.testing.assert_allclose(first.camera.position, [2.25, 0, 0])
    now[0] += 1.5
    playback.tick()
    np.testing.assert_allclose(first.camera.position, [3.5, 0, 0])
    first.handles["Play capture"].fire()
    now[0] += 1
    playback.tick()
    assert not playback.sessions[1]["playing"]
    assert playback.sessions[1]["seconds"] == 4
    first.handles["Loop replay"].value = True
    first.handles["Play capture"].fire()
    now[0] += 5
    playback.tick()
    assert playback.sessions[1]["seconds"] == 1
    first.handles["Return to start"].fire()
    assert playback.sessions[1]["seconds"] == 0
    assert not playback.sessions[1]["playing"]
    now[0] += 3
    playback.tick()
    np.testing.assert_allclose(first.camera.position, [0, 0, 0])
    first.handles["Show capture path"].value = True
    assert first.handles["/capture_replay/path"].visible
