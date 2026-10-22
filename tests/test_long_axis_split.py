"""Geometry, optimizer continuity and rendering checks for Long-Axis Split."""

import math

import pytest
import torch

from gsplat.strategy import BudgetStrategy
from gsplat.strategy.ops import split
from gsplat.utils import normalized_quat_to_rotmat


def fixture(device="cpu"):
    params = torch.nn.ParameterDict({
        "means": torch.tensor([[0., 0., 3.], [0.1, 0., 3.2], [0., 0.1, 3.]], device=device),
        "scales": torch.tensor([[.3, .1, .05], [.05, .3, .1], [.1, .05, .3]], device=device).log(),
        "quats": torch.tensor([[2., 0., 0., 0.], [math.sqrt(.5), 0., 0., math.sqrt(.5)],
                               [math.sqrt(.5), 0., math.sqrt(.5), 0.]], device=device),
        "opacities": torch.tensor([.5, .2, .8], device=device).logit(),
        "colors": torch.tensor([[.3, .5, .7], [.4, .6, .2], [.6, .4, .2]], device=device),
    })
    opts = {k: torch.optim.Adam([p]) for k, p in params.items()}
    return params, opts


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
def test_rotated_axis_extent_opacity_and_no_rng(device):
    params, opts = fixture(device)
    original = {k: p.detach().clone() for k, p in params.items()}
    rng = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    split(params, opts, {}, torch.ones(3, dtype=torch.bool, device=device), mode="long_axis")
    after_rng = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    assert torch.equal(rng, after_rng)
    centers = params["means"].reshape(2, 3, 3)
    torch.testing.assert_close(centers.mean(0), original["means"])
    rotations = normalized_quat_to_rotmat(torch.nn.functional.normalize(original["quats"], dim=-1))
    offsets = centers[0] - original["means"]
    local = torch.einsum("nji,nj->ni", rotations, offsets)
    torch.testing.assert_close(local, torch.eye(3, device=device) * .405)
    scales = params["scales"].exp().reshape(2, 3, 3)
    # Check that child outer endpoints meet the parent's 3-sigma endpoints.
    torch.testing.assert_close(local.diag() + 3 * scales[0].diag(),
                               3 * original["scales"].exp().diag())
    minor_mask = ~torch.eye(3, dtype=torch.bool, device=device)
    torch.testing.assert_close(scales[0][minor_mask], original["scales"].exp()[minor_mask] * math.sqrt(1 - .45**2))
    torch.testing.assert_close(params["opacities"].sigmoid(), original["opacities"].sigmoid().repeat(2) * .6)
    for k in ("quats", "colors"):
        torch.testing.assert_close(params[k], original[k].repeat(2, 1))


def test_optimizer_state_scene_hooks_and_empty_selection():
    params, opts = fixture()
    sum(p.square().sum() for p in params.values()).backward()
    for opt in opts.values():
        opt.step()
        opt.zero_grad()
    old = {k: opt.state[params[k]]["exp_avg"].clone() for k, opt in opts.items()}
    state = {"counts": torch.tensor([1., 2., 3.]), "history": []}

    class SceneHook:
        def on_split(self, sel, rest):
            self.sel, self.rest = sel, rest

    scene = SceneHook()
    split(params, opts, state, torch.tensor([False, True, False]), scene=scene, mode="long_axis")
    assert scene.sel.tolist() == [1] and scene.rest.tolist() == [0, 2]
    assert state["counts"].tolist() == [1., 3., 2., 2.]
    for k, opt in opts.items():
        assert opt.param_groups[0]["params"][0] is params[k]
        torch.testing.assert_close(opt.state[params[k]]["exp_avg"][:2], old[k][[0, 2]])
        assert not opt.state[params[k]]["exp_avg"][2:].any()
    before = dict(params)
    split(params, opts, state, torch.zeros(4, dtype=torch.bool), mode="long_axis")
    assert all(params[k] is before[k] for k in params)
    # A subsequent optimizer step must reach all replacement parameters.
    sum(p.square().sum() for p in params.values()).backward()
    for opt in opts.values():
        opt.step()
    assert all(torch.isfinite(p).all() for p in params.values())


def test_budget_splits_small_parents_and_resets_evidence():
    params, opts = fixture()
    params["scales"].data.fill_(-8)
    strategy = BudgetStrategy(cap_max=5, refine_start_iter=0, refine_stop_iter=2,
                              refine_every=1, split_mode="long_axis")
    state = strategy.initialize_state()
    strategy._ensure_state(params, state)
    state.update(grad2d=torch.ones(3), count=torch.ones(3))
    assert strategy._grow_gs(params, opts, state, 1) == (0, 2)
    assert len(params["means"]) == 5
    assert state["growth_history"][-1]["duplicated"] == 0
    assert not state["score_count"].any()


@pytest.mark.parametrize("kwargs", [{"split_distance": float("nan")}, {"split_distance": .6},
                                    {"opacity_reduction": 1.}, {"revised_opacity": True}])
def test_invalid_settings_fail_before_mutation(kwargs):
    with pytest.raises(ValueError):
        BudgetStrategy(split_mode="long_axis", **kwargs)
    params, opts = fixture()
    original = dict(params)
    with pytest.raises(ValueError):
        split(params, opts, {}, torch.ones(3, dtype=torch.bool), mode="long_axis", **kwargs)
    assert all(params[k] is original[k] for k in params)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_split_render_backward():
    from gsplat.rendering import rasterization

    params, opts = fixture("cuda")
    split(params, opts, {}, torch.ones(3, dtype=torch.bool, device="cuda"), mode="long_axis")
    rgb, alpha, _ = rasterization(
        means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(), colors=params["colors"],
        viewmats=torch.eye(4, device="cuda")[None],
        Ks=torch.tensor([[[40., 0., 16.], [0., 40., 16.], [0., 0., 1.]]], device="cuda"),
        width=32, height=32, packed=True)
    assert torch.isfinite(rgb).all() and alpha.max() > 0
    rgb.square().mean().backward()
    for p in params.values():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    for opt in opts.values():
        opt.step()
