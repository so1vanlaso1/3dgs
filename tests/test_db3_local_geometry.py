"""Research controls for withheld geometry and source coordinate mapping."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/name)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def test_withheld_reprojection_detects_pose_error_without_refitting():
    m=load('db3_validate_local.py');K=np.array([[400.,0,200],[0,400.,150],[0,0,1.]])
    uv=np.array([[100.,100.],[200.,150.],[300.,200.]])
    a=np.eye(4);b=np.eye(4);b[0,3]=.1;z=np.array([1.,2.,3.])
    target=uv.copy();target[:,0]+=400*.1/z
    sep,repro=m.metrics(a,b,uv,target,z,z,K)
    np.testing.assert_allclose(sep,0,atol=1e-12);np.testing.assert_allclose(repro,0,atol=1e-12)
    bad=b.copy();bad[0,3]+=.05
    sep,repro=m.metrics(a,bad,uv,target,z,z,K)
    np.testing.assert_allclose(sep,.05,atol=1e-12);assert np.median(repro)>6


def test_common_raw_rgb_correspondence_roundtrip():
    m=load('db3_validate_local.py');cal=m.m.read(ROOT/'data/db3_r1_sdk_20261110/calibration_clean.json');K=np.array(cal['K'])
    uv=np.array([[30.,30.],[400.,220.],[790.,420.]])
    old=m.legacy_pixels(uv,K,cal)
    raw1=m.m.projection.project(m.m.backproject(uv,np.ones(len(uv)),K),cal['raw_color'],m.m.projection.SDK)
    raw2=m.m.projection.project(m.m.backproject(old,np.ones(len(uv)),K),cal['raw_color'],m.m.projection.LEGACY)
    np.testing.assert_allclose(raw1,raw2,atol=1e-4)


def test_left_se3_information_jacobian_matches_perturbation():
    m=load('db3_transition_revisit.py');p=np.array([.2,-.3,2.]);J=np.c_[-m.skew(p),np.eye(3)]
    from scipy.spatial.transform import Rotation
    for j in range(6):
        delta=np.eye(6)[j]*1e-7
        numeric=(Rotation.from_rotvec(delta[:3]).apply(p)+delta[3:]-p)/1e-7
        np.testing.assert_allclose(numeric,J[:,j],atol=2e-7)
