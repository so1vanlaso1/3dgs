"""Confidence-weighted inverse-depth supervision on range-supported predictions."""

import torch
from torch import Tensor


def protected_sensor_depth(
    pred: Tensor, target: Tensor, confidence: Tensor, *, units_per_meter: float,
    scene_scale: float, min_depth_m: float, residual_scale_inv_m: float = 0.0,
):
    """Return the training loss, evidence mask, active mask and per-sample loss.

    Input Z is in normalized scene units. The optional log1p penalty preserves
    the L1 slope near zero and attenuates large inverse-meter discrepancies.
    Keep the original confidence denominator when a prediction is rejected.
    """
    if pred.shape != target.shape or pred.shape != confidence.shape:
        raise ValueError("Sensor depth and confidence arrays must match")
    if not (units_per_meter > 0 and min_depth_m > 0 and residual_scale_inv_m >= 0):
        raise ValueError("Invalid sensor scale or depth limit")
    evidence = torch.isfinite(target) & (target > 0) & torch.isfinite(confidence) & (confidence > 0)
    active = evidence & torch.isfinite(pred) & (pred >= min_depth_m * units_per_meter)
    # Replace invalid operands before the reciprocal, including in backward.
    inv_pred = torch.where(active, pred, 1.0).reciprocal()
    inv_target = torch.where(active, target, 1.0).reciprocal()
    error = (inv_pred - inv_target).abs()
    if residual_scale_inv_m:
        scale = residual_scale_inv_m / units_per_meter
        error = scale * torch.log1p(error / scale)
    weights = torch.where(evidence, confidence, 0.0)
    terms = error * weights
    loss = terms.sum() / weights.sum().clamp_min(1e-8) * scene_scale
    return loss, evidence, active, terms
