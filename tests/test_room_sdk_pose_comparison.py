"""Focused numerical convention checks for candidate/baseline comparison."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
def module(name):
    s=importlib.util.spec_from_file_location(name.replace('-','_'),ROOT/'scripts'/name);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def test_rigid_alignment_recovers_transform():
    m=module('compare-room-sdk-poses.py');x=np.random.default_rng(33).normal(size=(20,3));r=Rotation.from_euler('xyz',[.2,-.3,.1]).as_matrix();t=np.array([1.,2.,3.]);R,T=m.fit(x,x@r.T+t)
    assert np.allclose(R,r,atol=1e-12) and np.allclose(T,t,atol=1e-12)
def test_candidate_to_old_remap_reaches_same_raw_ray():
    m=module('compare-room-sdk-paired-epipolar.py');K=np.array([[428.,0,419.],[0,428.,233.],[0,0,1.]]);c=np.array([-.0559454858,.0645855516,-.000252637488,.000818350061,-.021141313]);uv=np.array([[30.,30.],[400.,200.],[800.,430.]])
    old=m.to_old(uv,K,c);new_ideal=(uv-[K[0,2],K[1,2]])/[K[0,0],K[1,1]];old_ideal=(old-[K[0,2],K[1,2]])/[K[0,0],K[1,1]]
    assert np.allclose(m.clean.invert_brown(old_ideal,c),m.clean.brown(new_ideal,c),atol=1e-7)
