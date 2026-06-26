"""
counter_state_rms.py — finite-state counter synapse WITH per-row RMS adaptive
scaling (doc section 4). Subclasses CompactCounterLinear and only overrides the
tile update: a per-row second moment v_i (O(out_features), negligible memory)
normalises the gradient before it drives the counter, the cheap analogue of
Adam's variance term. This tests whether the +16.6% counter-vs-Adam gap is mostly
"no adaptive LR" rather than something fundamental to finite-state weights.
"""
from __future__ import annotations
import math
import torch

from counter_state_fused_v2 import (
    CompactCounterLinear, encode_state, stochastic_round, ternary_gradient_unbiased,
)


class RMSCounterLinear(CompactCounterLinear):
    def __init__(self, *args, rms_beta: float = 0.9, rms_eps: float = 1e-3,
                 use_rms: bool = True, **kw) -> None:
        super().__init__(*args, **kw)
        self.rms_beta = float(rms_beta)
        self.rms_eps = float(rms_eps)
        self.use_rms = bool(use_rms)
        # per-output-row second moment: O(out_features), not O(out*in)
        self.register_buffer("v", torch.zeros((self.out_features, 1), dtype=torch.float32))

    @torch.no_grad()
    def _update_tile(self, lo, hi, grad_w, t_i, c_i, s_i) -> None:
        if self.use_rms:
            g_sq = grad_w.pow(2).mean(dim=1, keepdim=True)
            self.v[lo:hi].mul_(self.rms_beta).add_(g_sq, alpha=1.0 - self.rms_beta)
            denom = self.v[lo:hi].sqrt().clamp_min(self.rms_eps)
            grad_eff = grad_w / denom
        else:
            grad_eff = grad_w

        if self.local_grad_clip > 0:
            row_norm = grad_eff.norm(dim=1, keepdim=True).clamp_min(1e-30)
            grad_eff = grad_eff * (self.local_grad_clip / row_norm).clamp_max(1.0)

        # scale is still learned from the RAW gradient (its statistics are not normalised)
        grad_s = (grad_w * t_i.float()).sum(dim=1, keepdim=True) / math.sqrt(self.in_features)
        s_new = (s_i - self.lr_scale * grad_s).clamp_(1e-5, 10.0)

        update_signal = (
            grad_eff if self.pulse_mode == "direct" else ternary_gradient_unbiased(grad_eff)
        )
        c_rebased = c_i.float() * (s_i / s_new)
        ticks = (-self.lr * update_signal) * (self.C / s_new)
        cc = stochastic_round(c_rebased + ticks)

        carry = torch.trunc(cc / self.C)
        remainder = cc - carry * self.C
        proposed_t = t_i.float() + carry
        new_t = proposed_t.clamp_(-1, 1)
        blocked = proposed_t != new_t
        remainder = torch.where(
            blocked, torch.sign(cc) * (self.C - 1), remainder
        ).clamp_(-(self.C - 1), self.C - 1)

        self.state[lo:hi].copy_(encode_state(new_t, remainder, self.C))
        self.scale[lo:hi].copy_(s_new)
        self.update_events.add_(int((cc != c_i).sum().item()))
        self.weight_flips.add_(int((new_t != t_i).sum().item()))
