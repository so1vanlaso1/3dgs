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


@pytest.mark.parametrize("mode,policy,expected", [
    ("random", "clone", (1, 1)), ("long_axis", "clone", (1, 1)),
    ("random", "split", (0, 2)), ("long_axis", "split", (0, 2)),
    ("random", "method_default", (1, 1)),
    ("long_axis", "method_default", (0, 2)),
])
def test_clone_policy_is_independent_of_split_operator(mode, policy, expected):
    strategy = BudgetStrategy(cap_max=4, refine_start_iter=0, refine_stop_iter=2,
        refine_every=1, error_weight=0, split_mode=mode, clone_policy=policy)
    params, opts = make_params(2)
    params['scales'].data[1].fill_(-2)
    sum(p.square().sum() for p in params.values()).backward()
    for opt in opts.values():
        opt.step()
        opt.zero_grad()
    state = strategy.initialize_state()
    strategy._ensure_state(params, state)
    state.update(grad2d=torch.ones(2), count=torch.ones(2))
    assert strategy._grow_gs(params, opts, state, 1) == expected
    assert len(params['means']) == 4


def test_independent_sampler_ignores_structural_random_draws():
    def sequence(draws):
        generator = torch.Generator().manual_seed(104771)
        loader = torch.utils.data.DataLoader(list(range(23)), batch_size=1,
                                            shuffle=True, generator=generator)
        result = []
        for _ in range(3):
            for item in loader:
                result.extend(item.tolist())
                torch.randn(draws)
        return result
    assert sequence(0) == sequence(1000)


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


@pytest.mark.parametrize("rank,extra_parent", [("gradient", 1), ("residual", 2)])
def test_fallback_uses_vacant_slots_after_ordinary_growth(rank, extra_parent):
    strategy = BudgetStrategy(
        cap_max=7, refine_start_iter=0, refine_stop_iter=2, refine_every=1,
        split_mode="long_axis", fallback_rank=rank, fallback_start_iter=1,
        fallback_max=1,
    )
    params, optimizers = make_params(5)
    params["means"].data[:, 1] = torch.arange(5)
    params["opacities"].data[4] = -10
    state = strategy.initialize_state()
    strategy._ensure_state(params, state)
    state.update(
        grad2d=torch.tensor([0.0003, 0.00018, 0.0001, 0.00019, 0.00019]),
        count=torch.ones(5),
    )
    state["fallback_views"][:] = torch.tensor([3, 3, 3, 2, 3])
    state["fallback_error"][:] = torch.tensor([1., 2., 9., 40., 40.])
    assert strategy._grow_gs(params, optimizers, state, 1) == (0, 2)
    assert sorted(params["means"][-4:, 1].tolist()) == [0, 0, extra_parent, extra_parent]
    assert state["growth_history"][-1]["fallback_added"] == 1
    assert len(params["means"]) == strategy.cap_max
    assert not state["fallback_views"].any()
    assert not state["fallback_error"].any()


def test_fallback_counts_only_new_cameras_without_changing_base_observations(monkeypatch):
    monkeypatch.setattr(
        "gsplat.strategy.budget.rasterize_top_contributing_gaussian_ids",
        lambda **kwargs: (torch.tensor([[[[0]]]]), torch.ones(1, 1, 1, 1)),
    )
    params, _ = make_params(1)
    strategy = BudgetStrategy(
        fallback_rank="residual", fallback_start_iter=1, score_every=1,
        refine_start_iter=0,
    )
    state = strategy.initialize_state()
    info = dict.fromkeys(("conics", "opacities", "means2d"), torch.empty(0))
    info.update(
        isect_offsets=None, flatten_ids=None, width=1, height=1,
        tile_size=16, gaussian_ids=None,
    )
    rgb = torch.ones(1, 1, 1, 3)
    for step, camera in enumerate((5, 5, 9), start=1):
        strategy.observe_image(
            params, state, step, info, rgb, torch.zeros_like(rgb), camera_ids=[camera]
        )
    assert state["score_count"].item() == 3
    assert state["error_mass"].item() == 3
    assert state["fallback_views"].item() == 2
    assert state["fallback_error"].item() == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("packed", [True, False])
def test_absgrad_recovers_cancelled_pixel_gradients_in_budget_strategy(packed):
    from gsplat.rendering import rasterization

    params, optimizers = make_params(1, "cuda")
    params["means"].data[0, 2] = 3
    params["scales"].data.fill_(-2)
    params["colors"].data.fill_(0.5)
    strategy = BudgetStrategy(absgrad=True, grow_grad2d=0.0008)
    state = strategy.initialize_state()
    colors, _, info = rasterization(
        means=params["means"], quats=params["quats"],
        scales=params["scales"].exp(), opacities=params["opacities"].sigmoid(),
        colors=params["colors"], viewmats=torch.eye(4, device="cuda")[None],
        Ks=torch.tensor([[[30., 0., 8.], [0., 30., 8.], [0., 0., 1.]]], device="cuda"),
        width=16, height=16, packed=packed, absgrad=True,
    )
    strategy.step_pre_backward(params, optimizers, state, 0, info)
    # Symmetric pixels cancel the signed positional gradient at the image center.
    colors.mean().backward()
    signed = info["means2d"].grad.norm().item() * 8
    strategy._update_state(params, state, info, packed=packed)
    absolute = (state["grad2d"] / state["count"]).item()
    assert signed < strategy.grow_grad2d < absolute


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
