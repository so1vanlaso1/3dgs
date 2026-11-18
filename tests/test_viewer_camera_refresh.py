"""Camera polling recovers missed callbacks independently of capture replay."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from gsplat_viewer import GsplatViewer


def test_missed_camera_events_refresh_only_changed_client():
    states = {i: SimpleNamespace(c2w=np.eye(4), fov=0.8, aspect=1.5) for i in (1, 2)}
    tasks = {1: [], 2: []}
    clients = {i: i for i in states}
    viewer = GsplatViewer.__new__(GsplatViewer)
    viewer.server = SimpleNamespace(get_clients=lambda: clients.copy())
    viewer._renderers = {i: SimpleNamespace(submit=tasks[i].append) for i in states}
    viewer._camera_snapshots = {}
    viewer.get_camera_state = states.__getitem__

    # No camera callbacks or playback ticks are fired in this test.
    viewer.refresh_camera_views()
    viewer.refresh_camera_views()
    assert [len(tasks[i]) for i in (1, 2)] == [1, 1]
    states[1].c2w[0, 3] = 0.25
    viewer.refresh_camera_views()
    assert [len(tasks[i]) for i in (1, 2)] == [2, 1]
    assert tasks[1][-1].action == "move"
    assert tasks[1][-1].camera_state.c2w[0, 3] == 0.25
    states[1].c2w[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    viewer.refresh_camera_views()
    states[1].fov = 1.0
    viewer.refresh_camera_views()
    states[1].aspect = 2.0
    viewer.refresh_camera_views()
    assert len(tasks[1]) == 5
    assert len(tasks[2]) == 1
    del clients[1]
    viewer.refresh_camera_views()
    assert 1 not in viewer._camera_snapshots
