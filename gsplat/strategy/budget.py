"""Budgeted densification using statistics from existing training views.

Inspired by Taming 3DGS; this is an experimental adaptation, not a reproduction.
The contributor query retains only the strongest K contributors per pixel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Literal

import torch
import torch.nn.functional as F

from gsplat.cuda._wrapper import rasterize_top_contributing_gaussian_ids

from .default import DefaultStrategy
from .ops import duplicate, split


@dataclass
class BudgetStrategy(DefaultStrategy):
    """Limit growth and rank eligible parents by gradient and attributed error.

    Call :meth:`observe_image` before backward/optimizer updates, with the
    projections and images from the same render. DefaultStrategy's callbacks
    retain their existing semantics. The cap bounds live Gaussian count;
    structural updates still temporarily allocate old and replacement tensors.
    """

    cap_max: int = 700_000
    score_every: int = 10
    top_contributors: int = 4
    gradient_weight: float = 1.0
    error_weight: float = 1.0
    edge_weight: float = 0.0
    split_mode: Literal["random", "long_axis"] = "random"
    split_distance: float = 0.45
    opacity_reduction: float = 0.6
    # Separate child generation from the treatment of normally cloned parents.
    clone_policy: Literal["method_default", "clone", "split"] = "method_default"
    # Optional second route to growth when the ordinary candidate pool underfills.
    fallback_rank: Literal["off", "gradient", "residual"] = "off"
    fallback_start_iter: int = 5000
    fallback_max: int = 512
    fallback_min_views: int = 3

    def __post_init__(self):
        if self.split_mode not in ("random", "long_axis"):
            raise ValueError(f"Unknown split mode: {self.split_mode}")
        if self.clone_policy not in ("method_default", "clone", "split"):
            raise ValueError(f"Unknown clone policy: {self.clone_policy}")
        if self.fallback_rank not in ("off", "gradient", "residual"):
            raise ValueError(f"Unknown fallback ranking: {self.fallback_rank}")
        if self.fallback_start_iter < 0 or self.fallback_max < 1 or self.fallback_min_views < 1:
            raise ValueError("Fallback start must be nonnegative and limits positive")
        if not 0 < self.split_distance <= 0.5 or not 0 < self.opacity_reduction < 1:
            raise ValueError("Require 0 < split_distance <= 0.5 and 0 < opacity_reduction < 1")
        if self.split_mode == "long_axis" and self.revised_opacity:
            raise ValueError("Long-Axis Split uses its own opacity reduction")
        if self.cap_max < 1 or self.score_every < 1:
            raise ValueError("cap_max and score_every must be positive")
        if not 1 <= self.top_contributors <= 8:
            raise ValueError("top_contributors must be between 1 and 8")
        weights = (self.gradient_weight, self.error_weight, self.edge_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights) or not any(weights):
            raise ValueError("Score weights must be finite, nonnegative, and not all zero")
        if self.refine_every < 1 or self.reset_every < 1:
            raise ValueError("Refinement and opacity-reset intervals must be positive")

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        state = super().initialize_state(scene_scale)
        state.update(
            initial_count=None,
            error_mass=None,
            edge_mass=None,
            score_count=None,
            growth_history=[],
        )
        return state

    def _ensure_state(self, params, state):
        if state["initial_count"] is None:
            n = len(params["means"])
            if n > self.cap_max:
                raise ValueError(
                    f"Initial count {n} exceeds Gaussian budget {self.cap_max}"
                )
            state["initial_count"] = n
            for key in ("error_mass", "edge_mass", "score_count"):
                state[key] = torch.zeros(n, device=params["means"].device)
            if self.fallback_rank != "off":
                state["fallback_error"] = torch.zeros_like(state["error_mass"])
                state["fallback_views"] = torch.zeros_like(state["score_count"])
                state["fallback_seen"] = set()

    def _update_state(self, params, state, info, packed=False):
        self._ensure_state(params, state)
        super()._update_state(params, state, info, packed=packed)

    @staticmethod
    def _accumulate(state, ids, weights, pixel_error, pixel_edges=None, first_views=()):
        """Accumulate global Gaussian IDs, ignoring padded/zero-weight entries."""
        weights = weights.detach()
        valid = (ids >= 0) & torch.isfinite(weights) & (weights > 0)
        gs_ids = ids[valid].long()
        contributions = weights[valid]
        errors = pixel_error.detach().unsqueeze(-1).expand_as(weights)[valid]
        state["error_mass"].index_add_(0, gs_ids, contributions * errors)
        if pixel_edges is not None:
            edges = pixel_edges.detach().unsqueeze(-1).expand_as(weights)[valid]
            state["edge_mass"].index_add_(0, gs_ids, contributions * edges)
        # Count contributing views, rather than projected visibility or pixel hits.
        observed_ids = ids.masked_fill(~valid, -1)
        for view, view_ids in enumerate(observed_ids):
            # The public image observer calls this with [C,H,W,K].
            unique_ids = view_ids[view_ids >= 0].unique().long()
            state["score_count"][unique_ids] += 1
            if first_views and first_views[view]:
                keep = valid[view]
                state["fallback_views"][unique_ids] += 1
                view_errors = pixel_error[view].detach().unsqueeze(-1).expand_as(weights[view])
                state["fallback_error"].index_add_(
                    0, view_ids[keep].long(), weights[view][keep] * view_errors[keep]
                )

    @torch.no_grad()
    def observe_image(self, params, state, step, info, colors, pixels, masks=None, *, camera_ids=None):
        """Attribute detached pixel errors using the existing render's projections."""
        self._ensure_state(params, state)
        # Accumulate the window leading up to the first enabled refinement.
        track_fallback = (
            self.fallback_rank != "off"
            and step > self.fallback_start_iter - self.refine_every
        )
        if (
            step < self.refine_start_iter
            or step >= self.refine_stop_iter
            or step % self.score_every
            or (self.error_weight == 0 and self.edge_weight == 0 and not track_fallback)
        ):
            return
        if colors.ndim != 4 or colors.shape != pixels.shape:
            raise ValueError("Expected matching [C,H,W,3] training images")
        first_views = []
        if track_fallback:
            if camera_ids is None or len(camera_ids) != len(colors):
                raise ValueError("Fallback scoring requires one training camera ID per image")
            seen = state["fallback_seen"]
            for camera_id in camera_ids:
                first_views.append(camera_id not in seen)
                seen.add(camera_id)
        ids, weights = rasterize_top_contributing_gaussian_ids(
            means2d=info["means2d"].detach(),
            conics=info["conics"].detach(),
            opacities=info["opacities"].detach(),
            tile_offsets=info["isect_offsets"],
            flatten_ids=info["flatten_ids"],
            image_width=info["width"],
            image_height=info["height"],
            tile_size=info["tile_size"],
            num_depth_samples=self.top_contributors,
        )
        valid = ids >= 0
        if info["gaussian_ids"] is not None:
            # Packed contributor IDs index projected entries, not global splats.
            ids = ids.long()
            ids[valid] = info["gaussian_ids"][ids[valid]].long()
        errors = (colors.detach() - pixels.detach()).abs().mean(-1)
        edges = None
        if self.edge_weight:
            gray = pixels.detach().mean(-1).unsqueeze(1)
            kernel = gray.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])[None, None]
            edges = F.conv2d(
                F.pad(gray, (1, 1, 1, 1), mode="replicate"), kernel
            ).abs()[:, 0]
        if masks is not None:
            ids = ids.masked_fill(~masks.bool().unsqueeze(-1), -1)
        self._accumulate(state, ids, weights, errors, edges, first_views)

    def _fallback_candidates(self, state, grads, eligible, opaque):
        views = state["fallback_views"]
        residual = state["fallback_error"] / views.clamp_min(1)
        supported = (
            (views >= self.fallback_min_views) & torch.isfinite(residual) & (residual > 0)
        )
        candidates = (
            ~eligible & opaque & torch.isfinite(grads)
            & (grads <= self.grow_grad2d) & supported
        ).nonzero().flatten()
        return candidates, residual

    def target_count(self, initial_count: int, step: int) -> int:
        # Match DefaultStrategy's strict start/stop and reset-pause conditions.
        events = [
            s
            for s in range(
                (self.refine_start_iter // self.refine_every + 1) * self.refine_every,
                self.refine_stop_iter,
                self.refine_every,
            )
            if s % self.reset_every >= self.pause_refine_after_reset
        ]
        if not events:
            return initial_count
        completed = sum(s <= step for s in events)
        u = completed / len(events)
        return min(
            self.cap_max,
            math.floor(
                initial_count + (self.cap_max - initial_count) * (2 * u - u * u)
            ),
        )

    @staticmethod
    def _normalize(values, eligible):
        values = torch.nan_to_num(
            values, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0)
        positive = values[eligible & (values > 0)]
        if positive.numel() == 0:
            return torch.zeros_like(values)
        return (values / positive.median().clamp_min(1e-12)).clamp_max(20)

    @torch.no_grad()
    def _grow_gs(self, params, optimizers, state, step, scene=None):
        n = len(params["means"])
        target = self.target_count(state["initial_count"], step)
        grads = state["grad2d"] / state["count"].clamp_min(1)
        small = (
            params["scales"].exp().max(-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        eligible = grads > self.grow_grad2d
        if step < self.refine_scale2d_stop_iter:
            eligible |= state["radii"] > self.grow_scale2d
        opaque = params["opacities"].sigmoid().flatten() >= self.prune_opa
        eligible &= opaque
        candidates = eligible.nonzero().flatten()
        additions = min(max(0, target - n), candidates.numel())
        chosen = torch.zeros(n, dtype=torch.bool, device=grads.device)
        if additions:
            score = self.gradient_weight * self._normalize(grads, eligible)
            observations = state["score_count"]
            support = (observations / 3).clamp_max(1).sqrt()
            if self.error_weight:
                attributed_error = state["error_mass"] / observations.clamp_min(1)
                score += (
                    self.error_weight
                    * support
                    * self._normalize(attributed_error, eligible)
                )
            if self.edge_weight:
                attributed_edges = state["edge_mass"] / observations.clamp_min(1)
                score += (
                    self.edge_weight
                    * support
                    * self._normalize(attributed_edges, eligible)
                )
            selected = candidates[
                score[candidates].topk(additions, sorted=False).indices
            ]
            chosen[selected] = True
        fallback_stats = {}
        if self.fallback_rank != "off" and step >= self.fallback_start_iter:
            fallback, residual = self._fallback_candidates(state, grads, eligible, opaque)
            extra = min(self.fallback_max, max(0, target - n - additions), len(fallback))
            if extra:
                score = residual if self.fallback_rank == "residual" else grads
                chosen[fallback[score[fallback].topk(extra, sorted=False).indices]] = True
            fallback_stats = dict(
                fallback_candidates=len(fallback),
                fallback_added=extra,
                fallback_distinct_cameras=len(state["fallback_seen"]),
            )
            additions += extra
        replace_clones = self.clone_policy == "split" or (
            self.clone_policy == "method_default" and self.split_mode == "long_axis"
        )
        is_dupli = torch.zeros_like(chosen) if replace_clones else chosen & small
        is_split = chosen.clone() if replace_clones else chosen & ~small
        if step < self.refine_scale2d_stop_iter:
            is_split |= chosen & (state["radii"] > self.grow_scale2d)
            is_dupli &= ~is_split
        n_dupli, n_split = int(is_dupli.sum()), int(is_split.sum())
        if n_dupli:
            duplicate(params, optimizers, state, is_dupli, scene=scene)
        if n_split:
            is_split = torch.cat(
                [is_split, torch.zeros(n_dupli, dtype=torch.bool, device=grads.device)]
            )
            split(
                params, optimizers, state, is_split,
                revised_opacity=self.revised_opacity, scene=scene,
                mode=self.split_mode, split_distance=self.split_distance,
                opacity_reduction=self.opacity_reduction,
            )
        state["growth_history"].append(
            dict(step=step, before=n, target=target,
                 eligible=candidates.numel(), added=additions,
                 duplicated=n_dupli, split=n_split, split_mode=self.split_mode,
                 clone_policy=self.clone_policy, **fallback_stats)
        )
        # New children must not inherit evidence collected for their parent.
        for key in ("error_mass", "edge_mass", "score_count"):
            state[key].zero_()
        if self.fallback_rank != "off":
            state["fallback_error"].zero_()
            state["fallback_views"].zero_()
            state["fallback_seen"].clear()
        return n_dupli, n_split

    def _prune_gs(self, params, optimizers, state, step, scene=None):
        pruned = super()._prune_gs(params, optimizers, state, step, scene=scene)
        state["growth_history"][-1].update(pruned=pruned, after=len(params["means"]))
        return pruned
