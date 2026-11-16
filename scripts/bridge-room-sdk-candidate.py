"""Bridge a provisional RGB-D graph using SDK-consistent RGB remapping.

Only the isolated candidate root is permitted. The original source is read-only.
"""
import importlib.util
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('candidate_bridge',ROOT/'scripts/bridge-clean-rgbd.py')
bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
# Projection selection comes from the dataset calibration; no process-local patch.
import json
cal=json.loads((ROOT/'data/room_improve_sdk_rectified_20261103/calibration_clean.json').read_text())
if bridge.clean.projection.calibration_mode(cal) != bridge.clean.projection.SDK:
    raise ValueError('Historical unversioned candidate must be regenerated before bridging')
sys.argv=[str(ROOT/'scripts/bridge-clean-rgbd.py'),'--root',str(ROOT/'data/room_improve_sdk_rectified_20261103')]
bridge.main()
