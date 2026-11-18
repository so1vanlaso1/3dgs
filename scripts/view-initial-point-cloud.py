"""View the untrained RGB-D initialization points for a COLMAP dataset."""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import viser

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from datasets.colmap import Parser

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--port', type=int, default=8089)
a = ap.parse_args()
p = Parser(a.data, factor=1, normalize=True, load_exposure=False)
points = np.asarray(p.points, dtype=np.float32)
colors = np.asarray(p.points_rgb, dtype=np.uint8)
if len(points) == 0 or not np.isfinite(points).all():
    raise ValueError('Point cloud is empty or non-finite')
server = viser.ViserServer(host='127.0.0.1', port=a.port, label='Initial point cloud')
server.scene.add_point_cloud('/initial_points', points, colors, point_size=0.004, point_shape='circle', precision='float32')
center = np.median(points, axis=0)
span = float(np.max(np.percentile(points, 95, axis=0) - np.percentile(points, 5, axis=0)))
server.initial_camera.look_at = center
server.initial_camera.position = center + np.array([span, -span, span])
server.initial_camera.up = (0, 0, 1)
print(f'INITIAL_POINTS count={len(points)} center={center.tolist()} span={span}', flush=True)
while True:
    time.sleep(10)
