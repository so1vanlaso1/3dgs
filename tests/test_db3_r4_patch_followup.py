import importlib.util
from pathlib import Path
import cv2
import numpy as np

spec=importlib.util.spec_from_file_location('patch_followup',Path(__file__).resolve().parents[1]/'scripts/db3_r4_patch_followup.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def test_patch_tracks_recover_known_translation_without_fitting_tiles():
    rng=np.random.default_rng(2026)
    gray=cv2.GaussianBlur(rng.integers(0,256,(240,420),dtype='u1'),(5,5),1)
    images=[]
    for i in range(4):
        im=cv2.warpAffine(gray,np.array([[1,0,i],[0,1,0]],dtype=float),(420,240))
        images.append(cv2.cvtColor(im,cv2.COLOR_GRAY2RGB))
    tracks,report=p.patch_tracks(images)
    assert report['surviving_tracks']>=20
    assert np.max(np.linalg.norm(tracks[-1]-tracks[0]-[3,0],axis=1))<.3
    assert all(p.m.check_tile(xy).all() for xy in tracks)


def test_blank_images_supply_no_fabricated_checks():
    tracks,report=p.patch_tracks([np.zeros((100,200,3),'u1')]*3)
    assert tracks.shape==(3,0,2)
    assert report['surviving_tracks']==0
