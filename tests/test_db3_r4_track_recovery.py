"""Scientific safeguards and synthetic pose-recovery checks."""
import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('track_recovery',Path(__file__).resolve().parents[1]/'scripts/db3_r4_track_recovery.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def test_mixed_role_component_is_discarded_entirely():
    f=[dict(check=np.array([x])) for x in [False,True,True]]
    tracks,rejected=r.components([((0,0),(1,0)),((1,0),(2,0))],f)
    assert tracks=={'fit':[],'check':[]}
    assert rejected['mixed_fit_check_component']==1


def test_ambiguous_track_cannot_have_two_features_in_one_camera():
    f=[dict(check=np.ones(2,dtype=bool)) for _ in range(3)]
    tracks,rejected=r.components([((0,0),(1,0)),((1,0),(2,0)),((2,0),(0,1))],f)
    assert not tracks['check']
    assert rejected['ambiguous_multiple_features_in_one_camera']==1


def test_indirect_track_adds_endpoint_support_without_endpoint_descriptor_edge():
    f=[dict(check=np.array([True]),valid=np.array([True])) for _ in range(3)]
    tracks,_=r.components([((0,0),(1,0)),((1,0),(2,0))],f)
    rows=r.fixed_checks(f,tracks['check'],[(0,2)])
    assert rows[0]['refs']==[[0,0,0]]


def test_missing_spatial_support_cannot_pass():
    row=dict(matches=100,hull_fraction=[.001,.001],all_in_front=True,
             separation_m=[0,0],symmetric_reprojection_px=[0,0])
    assert not r.support_pass([row])
    assert not r.support_pass([])


def test_joint_refinement_recovers_known_cameras_and_ignores_check_tracks():
    rng=np.random.default_rng(123)
    K=np.array([[500.,0,417],[0,500,231],[0,0,1]])
    points=rng.uniform([-1,-.65,2],[1,.65,4],(50,3))
    poses=np.repeat(np.eye(4)[None],5,axis=0)
    poses[:,0,3]=np.linspace(0,.12,5)
    features=[]
    for T in poses:
        q=r.transform_points(T,points)
        uv=q[:,:2]/q[:,2:]*[500,500]+[417,231]
        features.append(dict(uv=uv,z=q[:,2],valid=np.ones(50,bool),sigma=np.full(50,.02),
                             xyz=q,check=np.r_[np.zeros(40,bool),np.ones(10,bool)],depth=np.zeros((463,834))))
    tracks=[[(c,i) for c in range(5)] for i in range(40)]
    initial=poses.copy()
    initial[1:,:3,3]+=rng.normal(0,.01,(4,3))
    refined,report=r.optimize(features,tracks,initial,K)
    assert report['success'] and report['finite'] and report['all_fit_points_in_front']
    assert np.max(np.abs(refined-poses))<.002
    assert all(i<40 for tr in report['fitting_tracks'] for _,i in tr)
    checks=r.fixed_checks(features,[[(c,i) for c in range(5)] for i in range(40,50)],[(0,4)])
    assert r.evaluate(features,checks,refined,K)[0]['separation_m'][1]<.002


def test_check_observation_is_never_silently_used_for_fit():
    f=[dict(valid=np.ones(30,bool),check=np.ones(30,bool),xyz=np.ones((30,3))) for _ in range(3)]
    import pytest
    with pytest.raises(AssertionError):
        r.optimize(f,[[(c,i) for c in range(3)] for i in range(30)],np.repeat(np.eye(4)[None],3,axis=0),np.eye(3))
