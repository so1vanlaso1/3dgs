"""D5 controls: normalization, invalid/empty observations and metric unit scaling."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from gsplat.losses import confidence_depth_l1_loss, depth_l1_loss, sensor_coverage_loss
from gsplat.robust_sensor_depth import protected_sensor_depth


@pytest.mark.parametrize("residual_scale", [0., .025])
def test_g4_invalid_predictions_have_zero_gradient_and_keep_sensor_denominator(residual_scale):
    pred = torch.tensor([1.3151e-5, float("nan"), float("inf"), -.2, 1., 2.], requires_grad=True)
    gt = torch.tensor([1.83, 2., 3., 2., 2., 0.])
    w = torch.tensor([.1, .1, .1, .1, .6, float("nan")])
    loss, evidence, active, _ = protected_sensor_depth(
        pred, gt, w, units_per_meter=1., scene_scale=2., min_depth_m=.3,
        residual_scale_inv_m=residual_scale)
    assert evidence.sum() == 5 and active.sum() == 1
    r = .5 if not residual_scale else residual_scale * np.log1p(.5 / residual_scale)
    assert loss.item() == pytest.approx(1.2 * r)
    loss.backward()
    assert torch.isfinite(pred.grad).all() and pred.grad[~active].eq(0).all()
    assert pred.grad[4].item() == pytest.approx(-1.2 / (1 + .5 / residual_scale) if residual_scale else -1.2)


@pytest.mark.parametrize("residual_scale", [0., .025])
def test_g4_unit_conversion_and_empty_frame(residual_scale):
    for p in (torch.tensor([.29, .30, 2.]), torch.empty(0)):
        def measure(units):
            z = (p * units).requires_grad_()
            loss, _, active, _ = protected_sensor_depth(
                z, torch.ones_like(p)*units, torch.ones_like(p), units_per_meter=units,
                scene_scale=units, min_depth_m=.3, residual_scale_inv_m=residual_scale)
            loss.backward()
            assert torch.isfinite(z.grad).all() and torch.isfinite(loss)
            return loss, active
        a, b = measure(1.), measure(2.)
        torch.testing.assert_close(a, b)
        assert a[1].tolist() == ([False, True, True] if len(p) else [])


def test_g4_c_agrees_with_legacy_when_all_predictions_are_supported():
    z = torch.tensor([.31, 1.83, 2.1], requires_grad=True)
    target = torch.tensor([.42, 1.83, 2.])
    confidence = torch.tensor([.14, .3, .2])
    legacy = confidence_depth_l1_loss(z, target, confidence, scene_scale=2.08)
    guarded, _, active, _ = protected_sensor_depth(
        z, target, confidence, scene_scale=2.08, units_per_meter=.988, min_depth_m=.30)
    assert active.all()
    torch.testing.assert_close(legacy, guarded)
    torch.testing.assert_close(torch.autograd.grad(legacy, z), torch.autograd.grad(guarded, z))


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
