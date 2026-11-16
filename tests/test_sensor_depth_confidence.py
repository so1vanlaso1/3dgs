"""D5 controls: normalization, invalid/empty observations and metric unit scaling."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from gsplat.losses import confidence_depth_l1_loss, depth_l1_loss, sensor_coverage_loss


def test_weights_normalize_and_preserve_legacy_all_ones():
    p=torch.tensor([[1.,2.,4.]],requires_grad=True);g=torch.tensor([[2.,2.,2.]])
    assert torch.allclose(confidence_depth_l1_loss(p,g,torch.ones_like(g)),depth_l1_loss(p,g))
    w=torch.tensor([[.1,0.,.3]])
    assert confidence_depth_l1_loss(p,g,w).item()==pytest.approx((.1*.5+.3*.25)/.4)
    assert torch.allclose(confidence_depth_l1_loss(p,g,w),confidence_depth_l1_loss(p,g,w*.1))


def test_unweighted_zero_render_depth_has_finite_backward():
    p=torch.tensor([0.,-1.,float('nan'),1.,2.],requires_grad=True)
    gt=torch.tensor([1.,1.,1.,0.,2.])
    loss=depth_l1_loss(p,gt);assert torch.isfinite(loss)
    loss.backward();assert torch.isfinite(p.grad).all()
    assert p.grad[:3].eq(0).all()


def test_supported_sensor_coverage_ignores_unknown_and_empty():
    a=torch.tensor([0.,1.,float('nan')],requires_grad=True)
    loss=sensor_coverage_loss(a,torch.tensor([1.,1.,0.]),torch.tensor([.2,.8,1.]))
    assert loss.item()==pytest.approx(.2);loss.backward();assert torch.isfinite(a.grad).all()
    for alpha,depth,weights in [(torch.empty(0),torch.empty(0),torch.empty(0)),
                               (torch.zeros(2),torch.ones(2),torch.zeros(2))]:
        alpha.requires_grad_();value=sensor_coverage_loss(alpha,depth,weights)
        assert value.item()==0;value.backward();assert torch.isfinite(alpha.grad).all()


def test_invalid_zero_weight_and_empty_have_finite_gradients():
    for p,g,w in [(torch.tensor([1.,2.,float('nan')]),torch.tensor([float('nan'),0.,2.]),torch.tensor([1.,1.,0.])),
                  (torch.empty(0),torch.empty(0),torch.empty(0)),
                  (torch.tensor([0.,-1.]),torch.tensor([1.,1.]),torch.zeros(2))]:
        p.requires_grad_();loss=confidence_depth_l1_loss(p,g,w);assert loss.item()==0
        loss.backward();assert torch.isfinite(p.grad).all()


def test_normalized_world_units_cancel():
    p=torch.tensor([1.,2.]);g=torch.tensor([1.2,2.2]);w=torch.tensor([.4,.8])
    assert torch.allclose(confidence_depth_l1_loss(p,g,w),confidence_depth_l1_loss(p*3,g*3,w,scene_scale=3))


def test_sensor_loader_transports_confidence_and_handles_empty(tmp_path):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
    from datasets.colmap import Dataset
    from types import SimpleNamespace
    parser=SimpleNamespace(data_dir=str(tmp_path),factor=1,params_dict={1:np.empty(0)},
        image_names=['a.png','b.png'],test_every=8,camera_ids=[1,1],camera_indices=[0,0],
        Ks_dict={1:np.eye(3)},camtoworlds=np.array([np.eye(4)]*2),mask_dict={1:None},
        exposure_values=[None,None],transform=np.diag([3.,3.,3.,1.]))
    (tmp_path/'experiment_split.json').write_text('{"train":["a.png","b.png"],"val":[]}')
    np.savez(tmp_path/'a.png.npz',points=np.array([[1.,1.],[2.,2.]]),depths=np.array([1.,2.]),confidence=np.array([.2,0.]))
    np.savez(tmp_path/'b.png.npz',points=np.array([[1.,1.]]),depths=np.array([0.]),confidence=np.ones(1))
    d=Dataset(parser,split='train',load_depths=True,sensor_depth_dir=str(tmp_path));d.images=[torch.zeros(4,4,3,dtype=torch.uint8)]*2
    assert torch.allclose(d[0]['depths'],torch.tensor([3.,6.]))
    assert torch.allclose(d[0]['depth_confidence'],torch.tensor([.2,0.]))
    assert d[1]['depths'].numel()==0
