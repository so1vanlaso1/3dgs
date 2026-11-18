"""Run the disjoint-source cloud audit on the SDK-remapped candidate graph."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('full_source',ROOT/'scripts/audit-room-full-source-cloud.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.DATA=ROOT/'data/room_improve_sdk_rectified_20261103'
m.OUT=ROOT/'results/room_improve_sdk_source_cloud_20261103'
m.main()
