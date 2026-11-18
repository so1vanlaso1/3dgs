import torch
from gsplat.surface import surface_depth_loss


def test_planar_and_bump_gradient():
    x = torch.arange(20.)[None, None, :, None].expand(1, 20, 20, 1)
    d = (.01 * x).exp().clone().requires_grad_()
    rgb, alpha = torch.zeros(1, 20, 20, 3), torch.ones_like(d)
    assert surface_depth_loss(d, rgb, alpha, stride=1) < 1e-6
    bumped = d.detach().clone()
    bumped[:, 10, 10] *= 2
    bumped.requires_grad_()
    loss = surface_depth_loss(bumped, rgb, alpha, stride=1)
    assert loss > .001
    loss.backward()
    assert torch.isfinite(bumped.grad).all() and bumped.grad.abs().sum() > 0


def test_empty_support_is_zero():
    d = torch.ones(1, 20, 20, 1, requires_grad=True)
    loss = surface_depth_loss(d, torch.zeros(1, 20, 20, 3), torch.zeros_like(d))
    loss.backward()
    assert loss == 0 and torch.isfinite(d.grad).all()


def test_nonfinite_depth_is_excluded():
    d = torch.ones(1,20,20,1)
    d[:,8,8] = float('nan')
    d.requires_grad_()
    loss = surface_depth_loss(d,torch.zeros(1,20,20,3),torch.ones_like(d),stride=1)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(d.grad).all()
