"""FP32, token-batched implementation of the released DVSM-TTB MSE bake."""
import torch
from torch.nn import functional as F

TOKEN_BATCH = 2048


def newton_schulz(grad, steps):
    x = grad
    transposed = x.size(1) > x.size(2)
    if transposed:
        x = x.transpose(1, 2)
    x = x / (x.norm(dim=(1, 2), keepdim=True) + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.transpose(1, 2)
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.transpose(1, 2) if transposed else x


def accumulate_mse_gradients(w0, w1, w2, k, v, lr0, lr1, lr2):
    g0, g1, g2 = (torch.zeros_like(w) for w in (w0, w1, w2))
    for start in range(0, k.shape[1], TOKEN_BATCH):
        tokens = slice(start, start + TOKEN_BATCH)
        key = k[:, tokens].float()
        gate = key @ w0
        hidden_before_mul = key @ w2
        activated = F.silu(gate)
        hidden = activated * hidden_before_mul
        error = hidden @ w1 - v[:, tokens].float()
        dhidden = error @ w1.transpose(-1, -2)
        sig = torch.sigmoid(gate)
        dgate = (dhidden * hidden_before_mul) * sig * (1 + gate * (1 - sig))
        g0.baddbmm_((key * lr0[:, tokens]).transpose(-1, -2), dgate, alpha=-1)
        g1.baddbmm_((hidden * lr1[:, tokens]).transpose(-1, -2), error, alpha=-1)
        g2.baddbmm_((key * lr2[:, tokens]).transpose(-1, -2), dhidden * activated, alpha=-1)
    return g0, g1, g2


def install():
    from dvsm.models import ttb

    original = ttb.fast_weight_swish_glu_weight_norm_mini_batch_apply

    def fast_weights(w0, w1, w2, q, k, v, lr0, lr1, lr2, ttb_ua_order,
                     muon_update_steps=0, use_mse=False, per_step_norm=False,
                     inner_grad_checkpoint=False):
        if not any(op.update for op in ttb_ua_order):
            return original(w0, w1, w2, q, k, v, lr0, lr1, lr2, ttb_ua_order,
                            muon_update_steps, use_mse, per_step_norm, inner_grad_checkpoint)
        assert use_mse and not per_step_norm and not torch.is_grad_enabled()
        assert all(op.start == 0 and op.end == q.shape[1] and op.update for op in ttb_ua_order)
        assert ttb_ua_order[-1].apply and not any(op.apply for op in ttb_ua_order[:-1])

        with torch.autocast("cuda", enabled=False):
            for _ in ttb_ua_order:
                gradients = accumulate_mse_gradients(w0, w1, w2, k, v, lr0, lr1, lr2)
                updates = (
                    newton_schulz(g, muon_update_steps) if muon_update_steps else g / k.shape[1]
                    for g in gradients
                )
                w0, w1, w2 = (weight + update for weight, update in zip((w0, w1, w2), updates))
            output = torch.empty_like(q)
            for start in range(0, q.shape[1], TOKEN_BATCH):
                tokens = slice(start, start + TOKEN_BATCH)
                query = q[:, tokens].float()
                output[:, tokens] = (F.silu(query @ w0, inplace=True) * (query @ w2)) @ w1
            if not all(torch.isfinite(w).all().item() for w in (w0, w1, w2, output)):
                raise FloatingPointError("Non-finite TTB bake values")
        return output, w0, w1, w2

    ttb.fast_weight_swish_glu_weight_norm_mini_batch_apply = fast_weights
