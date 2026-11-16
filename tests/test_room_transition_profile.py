"""Focused checks for missing depth and coordinate handling in the audit."""
import importlib.util
from pathlib import Path
import numpy as np

SPEC=importlib.util.spec_from_file_location('room_transition',Path(__file__).resolve().parents[1]/'scripts/audit-room-transition-profile.py')
MODULE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

def example(shift=0,missing=False):
    gray=np.zeros((3,100),np.uint8);gray[:,45:]=255
    row=np.full(100,2.,np.float32);row[40+shift:50+shift]=np.linspace(2,1,10);row[50+shift:]=1
    if missing:row[45+shift]=0
    z=np.tile(row,(3,1))
    return MODULE.profile(z,gray,1,(5,25),(75,95),(25,70))

def test_transition_moves_with_depth_and_missing_is_ambiguous():
    base=example(); shifted=example(5);missing=example(missing=True)
    assert base['status']=='candidate' and base['band_start'] <= base['band_end']
    assert shifted['band_start']-base['band_start']==5 and shifted['band_end']-base['band_end']==5
    assert base['rgb_x']==shifted['rgb_x']
    assert missing['status']=='ambiguous' and missing['holes_in_band']==1

def test_equal_interior_depth_is_rejected():
    gray=np.zeros((3,100),np.uint8);z=np.ones((3,100),np.float32)
    result=MODULE.profile(z,gray,1,(5,25),(75,95),(25,70))
    assert result['status']=='rejected' and 'difference' in result['reason']
