"""Run identical remapping on nearby frames selected after initial review."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('compare_sdk_remap',ROOT/'scripts/compare-room-sdk-remap.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.DEP=ROOT/'results/room_improve_sdk_candidate_extra_depth_20261103'
m.OUT=ROOT/'results/room_improve_sdk_remap_extra_20261103'
m.GROUPS={'nearby_check':[1,2,25,26]}
m.main()
