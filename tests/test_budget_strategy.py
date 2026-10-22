import pytest
import torch

from gsplat.strategy import BudgetStrategy


def test_budget_schedule_uses_actual_refinement_events():
    strategy = BudgetStrategy(cap_max=100, refine_start_iter=500,
                              refine_stop_iter=1000, refine_every=100)
    assert strategy.target_count(20, 500) == 20
    assert strategy.target_count(20, 600) == 55
    assert strategy.target_count(20, 900) == 100
    assert strategy.target_count(20, 999) == 100


def test_paused_events_and_empty_schedule():
    strategy = BudgetStrategy(cap_max=100, refine_start_iter=0,
                              refine_stop_iter=600, refine_every=100,
                              reset_every=300, pause_refine_after_reset=200)
    assert strategy.target_count(20, 100) == 20
    assert strategy.target_count(20, 200) == 80
    assert strategy.target_count(20, 500) == 100
    strategy.refine_stop_iter = 100
    assert strategy.target_count(20, 200) == 20


def test_error_attribution_ignores_padding_and_counts_views():
    state = {k: torch.zeros(3) for k in ("error_mass", "edge_mass", "score_count")}
    ids = torch.tensor([[[[0, 1], [0, -1]]], [[[0, 2], [2, 1]]]])
    weights = torch.tensor([[[[0.8, 0.2], [0.5, 0.0]]], [[[0.5, 0.5], [0.4, 0.0]]]])
    errors = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    BudgetStrategy._accumulate(state, ids, weights, errors)
    torch.testing.assert_close(state["error_mass"], torch.tensor([3.3, 0.2, 3.1]))
    torch.testing.assert_close(state["score_count"], torch.tensor([2.0, 1.0, 1.0]))


def make_params(n, device="cpu"):
    params = torch.nn.ParameterDict({
        "means": torch.zeros(n, 3, device=device),
        "scales": torch.full((n, 3), -6.0, device=device),
        "quats": torch.tensor([[1., 0., 0., 0.]], device=device).repeat(n, 1),
        "opacities": torch.zeros(n, device=device),
        "colors": torch.zeros(n, 3, device=device),
    })
    optimizers = {k: torch.optim.Adam([v]) for k, v in params.items()}
    return params, optimizers


def test_error_selects_useful_parent_and_keeps_optimizer_state():
    strategy = BudgetStrategy(cap_max=4, refine_start_iter=0, refine_stop_iter=2,
                              refine_every=1, gradient_weight=0, error_weight=1)
    params, optimizers = make_params(3)
    params["means"].data[:, 0] = torch.tensor([1., 2., 3.])
    sum(p.sum() for p in params.values()).backward()
    for opt in optimizers.values():
        opt.step()
        opt.zero_grad()
    expected = params["means"][1].detach().clone()
    state = strategy.initialize_state()
    strategy._ensure_state(params, state)
    state.update(grad2d=torch.ones(3), count=torch.ones(3))
    state["error_mass"][:] = torch.tensor([1., 10., 2.])
    state["score_count"].fill_(5)
    assert strategy._grow_gs(params, optimizers, state, 1) == (1, 0)
    torch.testing.assert_close(params["means"][-1], expected)
    assert len(params["means"]) == 4
    for k, opt in optimizers.items():
        assert opt.param_groups[0]["params"][0] is params[k]
        assert opt.state[params[k]]["exp_avg"].shape == params[k].shape
    assert not state["error_mass"].any()
    assert not state["score_count"].any()
    assert strategy._grow_gs(params, optimizers, state, 1) == (0, 0)


def test_rejects_initial_count_over_budget_and_invalid_settings():
    params, _ = make_params(3)
    strategy = BudgetStrategy(cap_max=2)
    with pytest.raises(ValueError, match="Initial count"):
        strategy._ensure_state(params, strategy.initialize_state())
    with pytest.raises(ValueError):
        BudgetStrategy(score_every=0)
    with pytest.raises(ValueError):
        BudgetStrategy(error_weight=float("nan"))


def test_split_adds_one_net_gaussian_and_can_underfill():
    strategy = BudgetStrategy(cap_max=7, refine_start_iter=0, refine_stop_iter=2,
                              refine_every=1, error_weight=0)
    params, optimizers = make_params(3)
    params["scales"].data.fill_(-2)
    state = strategy.initialize_state()
    strategy._ensure_state(params, state)
    state.update(grad2d=torch.tensor([1., 2., 0.]), count=torch.ones(3))
    # Optimizer state is created by a real step before structural updates.
    sum(p.sum() for p in params.values()).backward()
    for opt in optimizers.values():
        opt.step()
        opt.zero_grad()
    assert strategy._grow_gs(params, optimizers, state, 1) == (0, 2)
    assert len(params["means"]) == 5  # Only two parents qualify; do not force 7.
    strategy._prune_gs(params, optimizers, state, 1)
    assert state["growth_history"][-1]["after"] == 5
    for key in ("grad2d", "count", "error_mass", "edge_mass", "score_count"):
        assert state[key].shape == (5,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_and_dense_error_attribution_agree():
    from gsplat.rendering import rasterization

    params, _ = make_params(2, "cuda")
    params["means"].data[:] = torch.tensor([[0., 0., 3.], [0.1, 0., 3.2]], device="cuda")
    params["scales"].data.fill_(-2)
    params["colors"].data.fill_(0.5)
    strategy = BudgetStrategy(cap_max=3, refine_start_iter=0, score_every=1)
    scores = []
    for packed in (False, True):
        colors, _, info = rasterization(
            means=params["means"], quats=params["quats"],
            scales=params["scales"].exp(), opacities=params["opacities"].sigmoid(),
            colors=params["colors"], viewmats=torch.eye(4, device="cuda")[None],
            Ks=torch.tensor([[[30., 0., 8.], [0., 30., 8.], [0., 0., 1.]]], device="cuda"),
            width=16, height=16, packed=packed)
        state = strategy.initialize_state()
        strategy.observe_image(params, state, 0, info, colors, torch.zeros_like(colors))
        scores.append(state["error_mass"])
        assert state["error_mass"].sum() > 0
    torch.testing.assert_close(scores[0], scores[1], atol=1e-5, rtol=1e-5)
    state = strategy.initialize_state()
    strategy.edge_weight = 1
    strategy.observe_image(params, state, 0, info, colors, torch.zeros_like(colors),
                           masks=torch.zeros(colors.shape[:-1], device="cuda", dtype=torch.bool))
    assert state["error_mass"].sum() == 0
    assert state["score_count"].sum() == 0
    assert state["edge_mass"].sum() == 0
