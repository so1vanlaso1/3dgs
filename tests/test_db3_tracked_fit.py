import importlib.util
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree

spec=importlib.util.spec_from_file_location('fit',Path(__file__).resolve().parents[1]/'scripts/db3_tracked_fit.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def test_tracked_initializer_recovers_known_translation_and_excludes_checks():
    rng=np.random.default_rng(6);h,w=280,560
    im=cv2.GaussianBlur(rng.integers(0,256,(h,w),dtype='u1'),(5,5),1)
    images=[cv2.cvtColor(cv2.warpAffine(im,np.array([[1,0,c],[0,1,0]],float),(w,h)),cv2.COLOR_GRAY2RGB) for c in range(7)]
    reserved=[np.array([[100.+c,150.]]) for c in range(7)]
    fs,tracks,_=p.fitting_tracks(images,[np.full((h,w),2.,'f4')]*7,reserved)
    for c,f in enumerate(fs):
        assert not p.m.check_tile(f['uv']).any()
        assert (cKDTree(reserved[c]).query(f['uv'])[0]>12).all()
    K=np.array([[500.,0,280],[0,500,140],[0,0,1]])
    p.r.observations(fs,K);poses,_=p.initializer(fs,tracks)
    assert poses is not None
    assert abs(poses[-1,0,3]-.024)<.001
    assert np.max(np.abs(poses[-1,:3,:3]-np.eye(3)))<.001


def test_blank_fitting_images_stop_without_fabricated_initializer():
    fs,tracks,_=p.fitting_tracks([np.zeros((100,200,3),'u1')]*4,[np.ones((100,200),'f4')]*4,[np.empty((0,2))]*4)
    p.r.observations(fs,np.eye(3))
    poses,rows=p.initializer(fs,tracks)
    assert poses is None and not tracks and rows[0]['matches']==0
