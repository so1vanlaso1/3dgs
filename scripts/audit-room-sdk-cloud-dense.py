"""Dense-view check for the SDK-remapped candidate cloud and graph."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('dense',ROOT/'scripts/audit-room-full-cloud-dense.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.DATA=ROOT/'data/room_improve_sdk_rectified_20261103'
m.OUT=ROOT/'results/room_improve_sdk_source_cloud_20261103'
m.main()
