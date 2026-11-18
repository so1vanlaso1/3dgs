"""Conservative image-edge-aware depth regularization (experimental)."""
import torch


def surface_depth_loss(depth, rgb, alpha, mask=None, stride=4):
    """Penalize log-depth curvature on opaque, same-color neighborhoods.

    Affine log-depth planes have zero penalty. Alpha and RGB weights are
    detached to prevent opacity/color from changing merely to evade the loss.
    This is a smoothness prior, not a guarantee of correct geometry.
    """
    d = depth[:, ::stride, ::stride, 0]
    c = rgb[:, ::stride, ::stride]
    valid = (alpha[:, ::stride, ::stride, 0].detach() > .7) & (d.detach() > 1e-5)
    valid &= torch.isfinite(d.detach())
    if mask is not None:
        valid &= mask[:, ::stride, ::stride]
    safe = torch.where(torch.isfinite(d) & (d > 0), d, torch.ones_like(d))
    z = safe.clamp_min(1e-5).log()
    terms = []
    for axis in (1, 2):
        a, b, e = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
        a[axis], b[axis], e[axis] = slice(None, -2), slice(1, -1), slice(2, None)
        a, b, e = tuple(a), tuple(b), tuple(e)
        keep = valid[a] & valid[b] & valid[e]
        edge = (c[a] - c[b]).abs().mean(-1) + (c[e] - c[b]).abs().mean(-1)
        weight = torch.exp(-10 * edge.detach()) * keep
        curvature = z[a] - 2 * z[b] + z[e]
        terms.append((curvature.abs() * weight).sum() / weight.sum().clamp_min(1))
    return sum(terms) / 2
