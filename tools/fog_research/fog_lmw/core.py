import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep normalization statistics stable under BF16/FP16 autocast.
        scale = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * scale * self.weight.float()).to(x.dtype)


class CausalBlock(nn.Module):
    """Small decoder-only block used only to make the skeleton runnable."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.n1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.n2 = RMSNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        s = x.size(1)
        # True = forbidden in PyTorch boolean attn_mask.
        causal = torch.triu(
            torch.ones(s, s, dtype=torch.bool, device=x.device), diagonal=1
        )
        h = self.n1(x)
        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != x.shape[:2]:
                raise ValueError("attention_mask must have shape [batch, sequence]")
            key_padding_mask = ~attention_mask.bool()
        a, _ = self.attn(
            h,
            h,
            h,
            attn_mask=causal,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.n2(x)))
        if attention_mask is not None:
            # Padded queries can be fully masked (especially with left padding).
            # Zeroing them prevents NaNs/non-content states from reaching pooling.
            x = x.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
        return x


class TinyDecoderBackbone(nn.Module):
    """
    Reference backbone. In a real experiment this can be replaced by a Qwen/Llama
    decoder while retaining the planner/memory API.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pos = nn.Embedding(max_seq_len, d_model)
        # 0 = lexical token state, 1 = latent-memory state
        self.kind = nn.Embedding(2, d_model)
        self.blocks = nn.ModuleList(
            [CausalBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.out_norm = RMSNorm(d_model)

    def forward_embeds(
        self,
        x: torch.Tensor,
        kind_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        if s > self.max_seq_len:
            raise ValueError(f"sequence {s} > max_seq_len={self.max_seq_len}")
        if kind_ids is None:
            kind_ids = torch.zeros(b, s, dtype=torch.long, device=x.device)
        if attention_mask is None:
            attention_mask = torch.ones(b, s, dtype=torch.bool, device=x.device)
        else:
            attention_mask = attention_mask.to(device=x.device, dtype=torch.bool)
            if attention_mask.shape != (b, s):
                raise ValueError("attention_mask must have shape [batch, sequence]")
        if not torch.all(attention_mask.any(dim=-1)):
            raise ValueError("each sequence must contain at least one unmasked token")
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(dim=-1).sub(1).clamp_min(0)
        if position_ids.shape != (b, s):
            raise ValueError("position_ids must have shape [batch, sequence]")
        if int(position_ids.max()) >= self.max_seq_len:
            raise ValueError("position_ids exceed max_seq_len")
        h = x + self.pos(position_ids) + self.kind(kind_ids)
        h = h.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            h = block(h, attention_mask=attention_mask)
        h = self.out_norm(h)
        return h.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
