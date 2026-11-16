"""Check inverse-Brown candidate against librealsense when SDK is available."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.cache/room_sdk_audit/install'))
rs=pytest.importorskip('pyrealsense2')
spec=importlib.util.spec_from_file_location('room_clean',ROOT/'scripts/clean-realsense-scene.py')
clean=importlib.util.module_from_spec(spec);spec.loader.exec_module(clean)

def test_sdk_inverse_brown_roundtrip():
    intr=rs.intrinsics();intr.width=1280;intr.height=720;intr.fx=646.688599;intr.fy=646.027161;intr.ppx=643.358276;intr.ppy=362.999176;intr.model=rs.distortion.inverse_brown_conrady
    intr.coeffs=[-.0559454858,.0645855516,-.000252637488,.000818350061,-.021141313]
    coeff=np.array(intr.coeffs);ideal=np.array([[-.7,-.4],[0,0],[.7,.4]])
    raw=clean.brown(ideal,coeff)*[intr.fx,intr.fy]+[intr.ppx,intr.ppy]
    back=np.array([rs.rs2_deproject_pixel_to_point(intr,[float(x),float(y)],1.)[:2] for x,y in raw])
    assert np.max(np.linalg.norm((back-ideal)*[intr.fx,intr.fy],axis=1))<.04
    wrong=clean.invert_brown(ideal,coeff)*[intr.fx,intr.fy]+[intr.ppx,intr.ppy]
    old=np.array([rs.rs2_deproject_pixel_to_point(intr,[float(x),float(y)],1.)[:2] for x,y in wrong])
    assert np.max(np.linalg.norm((old-ideal)*[intr.fx,intr.fy],axis=1))>5
