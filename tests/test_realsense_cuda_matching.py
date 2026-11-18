"""Descriptor decision tests on CPU; CUDA execution is verified on the recording."""
import importlib.util
from pathlib import Path
import numpy as np
import torch


def load():
    s=importlib.util.spec_from_file_location('matcher',Path(__file__).parents[1]/'scripts/match-clean-rgbd-cuda.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_ratio_rejects_ambiguous_and_distance_rejects_unrelated():
    a=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[-1.,0.,0.]])
    b=torch.tensor([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.],[1.,0.,0.]])
    np.testing.assert_array_equal(load().match_descriptors(a,b),[[1,0],[2,1]])


def test_cross_check_prevents_multiple_sources_using_one_target():
    a=torch.tensor([[1.,0.],[1.,0.],[0.,1.]])
    b=torch.eye(2)
    np.testing.assert_array_equal(load().match_descriptors(a,b),[[0,0],[2,1]])
    assert load().match_descriptors(a,b[:1]).shape==(0,2)
