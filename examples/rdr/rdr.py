#!/usr/bin/env python3
"""
RDR — Recurrent-Depth Reasoner: runnable reference skeleton (toy scale, CPU).

This proves the WIRING of the spec (RDR_architecture_spec.md), not scale:
  - Sequence substrate: linear (sub-quadratic) attention, mostly, + a full-attn recall layer
  - Recurrent reasoning core: dense block reused with token-level adaptive depth (MoR router)
  - AttnRes: attention over depth-steps for selective aggregation (zero-init -> averaging)
  - Depth-wise LoRA: one adapter per recursion step (per-step specialization, CBC hook)
  - MoE prelude/coda (breadth) + MTP head
  - RLVR hook: verifier reward shapes (a) router depth and (b) per-step usage
Ablation flags map directly to spec section 6: use_recurrence / use_attnres / use_rlvr.

For SCALE: swap these toy modules for the open repos
  MoR (arXiv:2507.10524), Kimi Linear + Attention-Residuals (Moonshot), Titans (Google).
The verifier reward (verifier_reward) is the hook for pattern_generator.py.

CEILING NOTE: this sharpens a disposition latent in the base; it does not install
frontier capability. Goal = test whether loop x reward x AttnRes is real at small scale.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass

# ----------------------------------------------------------------------------- #
@dataclass
class Cfg:
    vocab: int = 256
    d_model: int = 64
    n_heads: int = 4
    n_sub_layers: int = 3      # sequence substrate depth
    full_attn_every: int = 3   # 1 full-attn "recall" layer every k linear layers
    n_recur: int = 4           # max recursion steps N_r
    lora_rank: int = 4
    n_experts: int = 4         # MoE experts in prelude/coda
    ff_mult: int = 2
    # ablations
    use_recurrence: bool = True
    use_attnres: bool = True
    use_rlvr: bool = True

# --------------------------- sub-quadratic attention -------------------------- #
def feature_map(x):                      # phi(x) = elu(x)+1  (positive)
    return F.elu(x) + 1.0

def linear_attention(q, k, v):
    """Causal linear attention, O(N) formulation. q,k,v: (B,H,N,d)."""
    q, k = feature_map(q), feature_map(k)
    kv = k.unsqueeze(-1) * v.unsqueeze(-2)          # (B,H,N,d,dv)
    kv_cum = kv.cumsum(dim=2)                        # prefix sums over time
    z_cum = k.cumsum(dim=2)                          # (B,H,N,d)
    num = (q.unsqueeze(-1) * kv_cum).sum(dim=-2)     # (B,H,N,dv)
    den = (q * z_cum).sum(dim=-1, keepdim=True) + 1e-6
    return num / den

def full_attention(q, k, v):
    """Causal softmax attention (recall layer). q,k,v: (B,H,N,d)."""
    B, H, N, d = q.shape
    s = (q @ k.transpose(-1, -2)) / math.sqrt(d)
    mask = torch.triu(torch.ones(N, N, device=q.device), diagonal=1).bool()
    s = s.masked_fill(mask, float('-inf'))
    return s.softmax(-1) @ v

class Attention(nn.Module):
    def __init__(self, cfg, full=False):
        super().__init__()
        self.h, self.dh, self.full = cfg.n_heads, cfg.d_model // cfg.n_heads, full
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.o = nn.Linear(cfg.d_model, cfg.d_model)
    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = [t.view(B, N, self.h, self.dh).transpose(1, 2) for t in (q, k, v)]
        a = full_attention(q, k, v) if self.full else linear_attention(q, k, v)
        return self.o(a.transpose(1, 2).reshape(B, N, D))

# ------------------------------ FFN + LoRA + MoE ------------------------------ #
class FFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.d_model * cfg.ff_mult
        self.w1 = nn.Linear(cfg.d_model, h)
        self.w2 = nn.Linear(h, cfg.d_model)
    def forward(self, x, lora=None):
        h = self.w1(x)
        if lora is not None:
            h = h + lora(x)            # depth-wise LoRA injection
        return self.w2(F.gelu(h))

class LoRA(nn.Module):
    def __init__(self, d_in, d_out, r):
        super().__init__()
        self.a = nn.Linear(d_in, r, bias=False)
        self.b = nn.Linear(r, d_out, bias=False)
        nn.init.zeros_(self.b.weight)   # zero-init -> no-op at start
    def forward(self, x): return self.b(self.a(x))

class MoEFFN(nn.Module):
    """Top-1 routed MoE FFN (breadth). Used only in prelude/coda, never in the loop."""
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.n_experts)
        self.experts = nn.ModuleList([FFN(cfg) for _ in range(cfg.n_experts)])
    def forward(self, x):
        w = self.gate(x).softmax(-1)            # (B,N,E)
        idx = w.argmax(-1)                       # top-1
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            m = (idx == e).unsqueeze(-1)
            if m.any():
                out = out + m * expert(x) * w[..., e:e+1]
        return out

class Block(nn.Module):
    """Pre-norm attn + FFN; FFN accepts an optional per-step LoRA."""
    def __init__(self, cfg, full=False, moe=False):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg, full=full)
        self.ff = MoEFFN(cfg) if moe else FFN(cfg)
        self.moe = moe
    def forward(self, x, lora=None):
        x = x + self.attn(self.n1(x))
        x = x + (self.ff(self.n2(x)) if self.moe else self.ff(self.n2(x), lora))
        return x

# --------------------------------- AttnRes ----------------------------------- #
class AttnRes(nn.Module):
    """Attention over depth-steps. Per-step learned pseudo-query (zero-init ->
    uniform averaging at start). Lets step t selectively retrieve prior steps."""
    def __init__(self, cfg):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(cfg.n_recur, cfg.d_model))   # zero-init
    def forward(self, stack, t):
        # stack: (B,N,S,D) prior step outputs;  returns (B,N,D) aggregate for step t
        scores = (stack * self.q[t]).sum(-1) / math.sqrt(stack.shape[-1])  # (B,N,S)
        w = scores.softmax(-1)
        return (w.unsqueeze(-1) * stack).sum(2)

# ------------------------------- MoR router ---------------------------------- #
class MoRRouter(nn.Module):
    """Token-level adaptive recursion depth: per token, a distribution over 1..N_r."""
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.n_recur)
        self.n_recur = cfg.n_recur
    def forward(self, e, sample=True):
        logits = self.fc(e)                        # (B,N,N_r)
        dist = torch.distributions.Categorical(logits=logits)
        depth_idx = dist.sample() if sample else logits.argmax(-1)   # 0..N_r-1
        return depth_idx + 1, dist.log_prob(depth_idx)               # depth in 1..N_r

# -------------------------- recurrent reasoning core ------------------------- #
class RecurrentCore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block = Block(cfg, full=False, moe=False)               # DENSE, shared
        self.loras = nn.ModuleList([LoRA(cfg.d_model, cfg.d_model * cfg.ff_mult,
                                         cfg.lora_rank) for _ in range(cfg.n_recur)])
        self.router = MoRRouter(cfg)
        self.attnres = AttnRes(cfg)
    def forward(self, x, e, sample=True):
        cfg = self.cfg
        N_r = cfg.n_recur if cfg.use_recurrence else 1
        depths, logp = self.router(e, sample=sample)
        if not cfg.use_recurrence:
            depths = torch.ones_like(depths)
        current = e
        stack = e.unsqueeze(2)                                       # (B,N,1,D)
        for t in range(1, N_r + 1):
            inp = self.attnres(stack, t - 1) if cfg.use_attnres else current
            out = self.block(inp, lora=self.loras[t - 1])
            active = (depths >= t).float().unsqueeze(-1)             # token-level halting
            current = active * out + (1 - active) * current
            stack = torch.cat([stack, current.unsqueeze(2)], dim=2)
        return current, depths, logp

# ------------------------------- full model ---------------------------------- #
class RDR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.prelude = Block(cfg, full=False, moe=True)             # MoE breadth
        sub = []
        for i in range(cfg.n_sub_layers):
            sub.append(Block(cfg, full=((i + 1) % cfg.full_attn_every == 0)))
        self.substrate = nn.ModuleList(sub)                         # hybrid linear+full
        self.core = RecurrentCore(cfg)
        self.coda = Block(cfg, full=False, moe=True)               # MoE breadth
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab)
        self.mtp_head = nn.Linear(cfg.d_model, cfg.vocab)          # multi-token pred
    def forward(self, ids, sample=True):
        x = self.embed(ids)
        x = self.prelude(x)
        e = x                                                      # injection signal
        for lyr in self.substrate:
            x = lyr(x)
        x, depths, logp = self.core(x, e, sample=sample)
        x = self.coda(x)
        x = self.norm(x)
        return self.lm_head(x), self.mtp_head(x), depths, logp

# --------------------------- verifier / RLVR hook ---------------------------- #
def verifier_reward(ids, logits):
    """HOOK for pattern_generator.py's executable verifier.
    Real use: decode the model's answer, run the verifier (executable check vs a
    trusted high-precision reference), return reward in [0,1] per example
    (1 iff answer correct AND claimed error bound honest & tight; else 0).
    Stub here: a smooth proxy (next-token accuracy) so the skeleton runs end-to-end."""
    pred = logits[:, :-1].argmax(-1)
    tgt = ids[:, 1:]
    return (pred == tgt).float().mean(dim=1)                        # (B,) in [0,1]

def step(model, ids, opt, ponder_coef=0.02):
    cfg = model.cfg
    logits, mtp, depths, logp = model(ids, sample=cfg.use_rlvr)
    # supervised next-token (answer) loss
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab), ids[:, 1:].reshape(-1))
    loss = ce
    metrics = {"ce": ce.item(), "mean_depth": depths.float().mean().item()}
    if cfg.use_rlvr:
        reward = verifier_reward(ids, logits)                      # (B,)  <-- verifier
        baseline = reward.mean()
        # REINFORCE: reward shapes router depth choices
        reinforce = -((reward - baseline).detach().unsqueeze(1) * logp).mean()
        ponder = ponder_coef * depths.float().mean()               # discourage over-looping
        loss = ce + reinforce + ponder
        metrics.update(reward=reward.mean().item(), reinforce=reinforce.item())
    opt.zero_grad(); loss.backward(); opt.step()
    metrics["loss"] = loss.item()
    return metrics

# --------------------------------- smoke test -------------------------------- #
def smoke():
    torch.manual_seed(0)
    cfg = Cfg()
    m = RDR(cfg)
    ids = torch.randint(0, cfg.vocab, (2, 16))
    logits, mtp, depths, logp = m(ids)
    print(f"forward ok: logits {tuple(logits.shape)}, mtp {tuple(mtp.shape)}, "
          f"depths {tuple(depths.shape)}  range[{depths.min().item()},{depths.max().item()}]")
    assert logits.shape == (2, 16, cfg.vocab)
    print("per-token depth varies:", depths[0].tolist())
    params = sum(p.numel() for p in m.parameters())
    print(f"params: {params/1e3:.1f}K")

    print("\n-- train 30 steps (RLVR on) --")
    opt = torch.optim.Adam(m.parameters(), 1e-3)   # swap for Muon at scale
    for i in range(30):
        mt = step(m, torch.randint(0, cfg.vocab, (4, 16)), opt)
        if i % 10 == 0 or i == 29:
            print(f"  step {i:2d}: loss {mt['loss']:.3f} ce {mt['ce']:.3f} "
                  f"reward {mt.get('reward',0):.3f} mean_depth {mt['mean_depth']:.2f}")

    print("\n-- ablation grid (each must run) --")
    for rec in (True, False):
        for ar in (True, False):
            for rl in (True, False):
                c = Cfg(use_recurrence=rec, use_attnres=ar, use_rlvr=rl)
                mm = RDR(c); oo = torch.optim.Adam(mm.parameters(), 1e-3)
                mt = step(mm, torch.randint(0, c.vocab, (2, 16)), oo)
                print(f"  recur={int(rec)} attnres={int(ar)} rlvr={int(rl)} "
                      f"-> loss {mt['loss']:.3f} depth {mt['mean_depth']:.2f}  OK")
    print("\nALL OK.")

if __name__ == "__main__":
    smoke()
