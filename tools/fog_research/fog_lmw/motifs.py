import math
import torch
from torch import nn

from .core import RMSNorm


class LowRankCompareSelectAggregate(nn.Module):
    """
    Explicit semantic motif macrograph:

      project  : q -> Q, context -> K,V
      compare  : S = Q K^T / sqrt(r)
      select   : A = softmax(S)
      aggregate: O = A V

    compare_rank is intentionally independent from d_model.
    """

    def __init__(self, d_model: int, compare_rank: int, out_dim: int | None = None):
        super().__init__()
        out_dim = out_dim or d_model
        self.compare_rank = compare_rank
        self.q_proj = nn.Linear(d_model, compare_rank, bias=False)
        self.k_proj = nn.Linear(d_model, compare_rank, bias=False)
        self.v_proj = nn.Linear(d_model, out_dim, bias=False)
        self.o_proj = nn.Linear(out_dim, d_model, bias=False)

    def forward(
        self,
        query: torch.Tensor,     # [B,Q,D]
        context: torch.Tensor,   # [B,S,D]
        context_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(query)
        k = self.k_proj(context)
        v = self.v_proj(context)
        score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.compare_rank)
        if context_mask is not None:
            if context_mask.ndim != 2 or context_mask.shape != context.shape[:2]:
                raise ValueError("context_mask must have shape [batch, context_length]")
            if not torch.all(context_mask.any(dim=-1)):
                raise ValueError("each context_mask row must select at least one token")
            score = score.masked_fill(~context_mask[:, None, :], float("-inf"))
        weight = torch.softmax(score.float(), dim=-1).to(v.dtype)
        out = torch.matmul(weight, v)
        return self.o_proj(out), weight


class AddressedPayloadBinding(nn.Module):
    """Compare learned addresses while copying payload coordinates unchanged.

    Keys and values play different roles.  ``address_context`` may be a rich
    contextual representation used only for matching; ``payload_context`` is
    the exact code that must survive selection (for example a tied token/value
    embedding).  The same soft attention weights select the payload, but no
    learned V/O projection rotates it before the protected latent slot.
    """

    def __init__(self, d_model: int, compare_rank: int):
        super().__init__()
        self.compare_rank = compare_rank
        self.q_proj = nn.Linear(d_model, compare_rank, bias=False)
        self.k_proj = nn.Linear(d_model, compare_rank, bias=False)
        # Cosine comparison makes an identical query/key code the unique
        # optimum independently of embedding norm.  A fairly sharp fixed
        # scale keeps selection differentiable while avoiding an imprecise
        # average of several payloads.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0)))

    def forward(
        self,
        query: torch.Tensor,
        address_context: torch.Tensor,
        payload_context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if address_context.shape != payload_context.shape:
            raise ValueError("address and payload contexts must have identical shape")
        q = torch.nn.functional.normalize(self.q_proj(query).float(), dim=-1)
        k = torch.nn.functional.normalize(
            self.k_proj(address_context).float(), dim=-1
        )
        score_scale = self.logit_scale.exp().clamp(max=200.0)
        score = score_scale * torch.matmul(q, k.transpose(-1, -2))
        if context_mask is not None:
            if context_mask.shape != address_context.shape[:2]:
                raise ValueError("context_mask must match context sequence shape")
            if not torch.all(context_mask.any(dim=-1)):
                raise ValueError("each context row must select at least one address")
            score = score.masked_fill(~context_mask[:, None, :], float("-inf"))
        weight = torch.softmax(score.float(), dim=-1).to(payload_context.dtype)
        return torch.matmul(weight, payload_context), weight


class ReasoningModeRouter(nn.Module):
    """
    route/control motif. The mode vectors are generic continuous reasoning
    directions. A multilingual teacher can later be used to initialize or
    regularize them, but the forward path never needs to emit language tokens.
    """

    def __init__(self, d_model: int, n_modes: int):
        super().__init__()
        self.n_modes = n_modes
        if n_modes > 0:
            self.mode = nn.Parameter(torch.randn(n_modes, d_model) * 0.02)
            self.router = nn.Linear(2 * d_model, n_modes)

    def forward(
        self, queries: torch.Tensor, pooled_context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.n_modes <= 0:
            return queries, None
        b, k, _ = queries.shape
        ctx = pooled_context[:, None, :].expand(b, k, -1)
        probs = torch.softmax(self.router(torch.cat([queries, ctx], dim=-1)), dim=-1)
        mode_mix = torch.einsum("bkl,ld->bkd", probs, self.mode)
        return queries + mode_mix, probs


class ExpandCompressFFN(nn.Module):
    """expand/compress motif inside a slot state."""

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))
