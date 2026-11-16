import importlib.util
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

spec=importlib.util.spec_from_file_location('moving',Path(__file__).resolve().parents[1]/'scripts/db3_moving_holdout.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def synthetic():
    rng=np.random.default_rng(23);K=np.array([[500.,0,417],[0,500,231],[0,0,1]])
    points=rng.uniform([-1,-.6,2],[1,.6,4],(40,3));poses=np.repeat(np.eye(4)[None],4,axis=0)
    poses[:,:3,:3]=Rotation.from_rotvec(np.c_[np.zeros(4),np.arange(4)*.01,np.zeros(4)]).as_matrix()
    poses[:,0,3]=np.arange(4)*.03;features=[]
    for T in poses:
        q=p.r.transform_points(T,points);uv=q[:,:2]/q[:,2:]*[500,500]+[417,231]
        features.append(dict(uv=uv,xyz=q,z=q[:,2],valid=np.ones(40,bool),sigma=np.full(40,.02),check=np.zeros(40,bool)))
    tracks=[[(c,i) for c in range(4)] for i in range(40)]
    initial=poses.copy();initial[1:,:3,3]+=rng.normal(0,.005,(3,3))
    return features,tracks,initial,K,poses


def test_analytic_jacobian_matches_numerical_rotations_and_points():
    features,tracks,initial,K,_=synthetic()
    for mode in ('pixel_depth','metric'):
        x,calc,_,_=p.fit_problem(features,tracks,initial,K,mode)
        jac=calc(x,True).toarray()
        for j in [0,1,2,3,8,12,17,18,29,len(x)-1]:
            a=x.copy();b=x.copy();a[j]+=1e-6;b[j]-=1e-6
            numeric=(calc(a)-calc(b))/2e-6
            assert np.allclose(jac[:,j],numeric,rtol=1e-4,atol=2e-5)


def test_both_objectives_recover_known_metric_cameras():
    features,tracks,initial,K,truth=synthetic()
    for mode in ('pixel_depth','metric'):
        poses,report=p.refine(features,tracks,initial,K,mode)
        assert report['success'] and report['all_fit_points_in_front']
        assert np.max(np.abs(poses-truth))<.001


def test_new_holdout_propagates_through_entire_feature_component():
    features=[dict(uv=np.array([[20.,20],[100,100],[200,200]]),check=np.array([False,False,True])) for _ in range(3)]
    links=[((0,0),(1,1)),((1,1),(2,0)),((0,2),(1,2))]
    reserved=[np.empty((0,2)),np.array([[101.,100]]),np.empty((0,2))]
    blocked=p.propagate_exclusions(features,links,reserved)
    assert blocked[0][0] and blocked[1][1] and blocked[2][0]
    assert all(b[2] for b in blocked)
    assert not blocked[0][1]


def test_moving_checks_survive_tile_crossing():
    rng=np.random.default_rng(12)
    im=cv2.GaussianBlur(rng.integers(0,256,(280,560),dtype='u1'),(5,5),1)
    images=[cv2.cvtColor(cv2.warpAffine(im,np.array([[1,0,c*5],[0,1,0]],float),(560,280)),cv2.COLOR_GRAY2RGB) for c in range(8)]
    uv,reserved,report=p.follow_checks(images)
    assert uv.shape[1]>=20 and report['crosses_original_tile_boundaries']>0
    assert np.max(np.linalg.norm(uv[-1]-uv[0]-[35,0],axis=1))<.4
    assert len(reserved)==len(images)


def test_blank_images_do_not_create_tracks():
    uv,reserved,report=p.follow_checks([np.zeros((100,200,3),'u1')]*3)
    assert uv.shape==(3,0,2) and report['survivors']==0
