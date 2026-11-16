"""Check the rigid fitting and relative-pose conventions used by the audit."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def load(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/name
    spec=importlib.util.spec_from_file_location('revisit_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_rigid_fit_recovers_known_motion():
    audit=load('audit-courtyard-revisit.py')
    rng=np.random.default_rng(12);points=rng.normal(size=(100,3))
    rotation=Rotation.from_euler('xyz',[18,-27,42],degrees=True).as_matrix()
    translation=np.array([1.5,-2.0,.3]); target=points@rotation.T+translation
    result=audit.fit(points,target)
    np.testing.assert_allclose(result[:3,:3],rotation,atol=1e-12)
    np.testing.assert_allclose(result[:3,3],translation,atol=1e-12)
    assert np.linalg.det(result[:3,:3])>0

def test_revisit_residual_uses_camera_to_world_convention():
    repair=load('repair-courtyard-revisit.py')
    poses=np.repeat(np.eye(4)[None],2,axis=0)
    poses[0,:3,3]=[1,2,3];poses[1,:3,3]=[2,1,4]
    poses[1,:3,:3]=Rotation.from_euler('y',30,degrees=True).as_matrix()
    actual=np.linalg.inv(poses[1])@poses[0]
    result=repair.residual(poses,0,1,actual)
    assert result['translation_m']<1e-12
    assert result['rotation_deg']<1e-10
