#!/usr/bin/env python3
"""
anchor_rdr.py — Span-Anchor Subquadratic Attention variant of RDR, + a long-context
retrieval task to actually test it.

Implements the ANCHOR formalization (toy-scale, faithful where it matters):
  * Section 3  span LANDMARK : each width-B block -> 1 attention-pooled KV slot.
  * Section 2  anchor SCORE  : prior content + runtime surprise, z-scored, pooled
                               per block -> g_b (bias on span selection).
  * Section 4  span-anchor ATTENTION : query attends to (local window W_t) UNION
                               (top-k landmark blocks before the window). ONE exact
                               softmax over the sparse union (not a kernel approx).
  * Section 6  learnable SELECTION : Gumbel-TopK ranking (exploration) over the
                               exact logits used for weighting.
  * Section 5  depth-persistent LANDMARKS (toggle): each layer's landmark is its
                               pooled value plus a learned residual from the SAME
                               block's landmark one layer below (anti-decay AttnRes
                               over the depth axis). Threaded through the substrate.
Omitted on purpose: the counterfactual Omega(s) supervision (eq. 1/11) — too costly
and not needed for this mechanism test; selection is learned through the LM loss.

WHY a new task: with T~40 final-value tasks there is nothing for span selection to
select. So we add KEYVAL retrieval: many "key:val" pairs then "GET key =", answer is
that key's value. Difficulty = number of pairs = context length = needle distance.
This is exactly the long-range regime the anchor mechanism is built for. We compare
substrate attention in {linear, full, anchor} on it.

Runs on CPU (small) or a Kaggle GPU (bump CONFIG). `python anchor_rdr.py --help`.
"""
import math, json, time, random, argparse
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS = 256, 257, 258
VOCAB = 259


@dataclass
class Cfg:
    vocab: int = VOCAB
    d_model: int = 64
    n_heads: int = 4
    n_sub_layers: int = 4
    full_attn_every: int = 99          # (only used by 'full'/'linear' hybrid)
    n_recur: int = 4
    lora_rank: int = 4
    n_experts: int = 4
    ff_mult: int = 2
    use_recurrence: bool = True
    use_attnres: bool = True
    use_rlvr: bool = True
    # --- substrate attention selection ---
    attn_kind: str = "anchor"          # 'linear' | 'full' | 'anchor'
    sparse_attn_window: int = 8        # w : local window
    sparse_attn_topk: int = 4          # k : landmark blocks selected per query
    sparse_attn_block: int = 6         # B : span / block width
    landmark_depth_residual: bool = False
    learnable_selection: bool = True   # Gumbel-TopK vs deterministic top-k


# ----------------------------- basic attentions ----------------------------- #
def feature_map(x):
    return F.elu(x) + 1.0

def linear_attention(q, k, v):
    q, k = feature_map(q), feature_map(k)
    kv = (k.unsqueeze(-1) * v.unsqueeze(-2)).cumsum(dim=2)
    z = k.cumsum(dim=2)
    num = (q.unsqueeze(-1) * kv).sum(dim=-2)
    den = (q * z).sum(dim=-1, keepdim=True) + 1e-6
    return num / den

def full_attention(q, k, v):
    B, H, N, d = q.shape
    s = (q @ k.transpose(-1, -2)) / math.sqrt(d)
    mask = torch.triu(torch.ones(N, N, device=q.device), diagonal=1).bool()
    return s.masked_fill(mask, float('-inf')).softmax(-1) @ v


# --------------------------- span-anchor attention -------------------------- #
class AnchorAttention(nn.Module):
    """Sections 2-6. Causal. Returns (out, landmarks) so the substrate can thread
    depth-persistent landmarks (Section 5)."""
    def __init__(self, cfg):
        super().__init__()
        self.h, self.dh = cfg.n_heads, cfg.d_model // cfg.n_heads
        self.w, self.k, self.B = cfg.sparse_attn_window, cfg.sparse_attn_topk, cfg.sparse_attn_block
        self.learn_sel = cfg.learnable_selection
        self.depth_res = cfg.landmark_depth_residual
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.o = nn.Linear(cfg.d_model, cfg.d_model)
        # anchor score (Section 2)
        self.prior = nn.Linear(cfg.d_model, 1)
        self.wp = nn.Parameter(torch.tensor(1.0)); self.wr = nn.Parameter(torch.tensor(1.0))
        # landmark pool query (Section 3) + landmark k/v projections
        self.pool_q = nn.Linear(cfg.d_model, 1)
        self.kL = nn.Linear(cfg.d_model, cfg.d_model)
        self.vL = nn.Linear(cfg.d_model, cfg.d_model)
        # depth-persistent residual (Section 5)
        if self.depth_res:
            self.res = nn.Linear(cfg.d_model, cfg.d_model)
            self.ln = nn.LayerNorm(cfg.d_model)

    def _heads(self, t):
        B, N, _ = t.shape
        return t.view(B, N, self.h, self.dh).transpose(1, 2)        # (B,H,N,dh)

    def forward(self, x, prev_landmarks=None):
        B, N, D = x.shape
        dev = x.device
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = self._heads(q), self._heads(k), self._heads(v)

        # ---- anchor score a_t, z-scored (Section 2) ----
        prior = self.prior(x).squeeze(-1)                           # (B,N)
        surprise = (x[:, 1:] - x[:, :-1]).norm(dim=-1) / math.sqrt(D)
        surprise = F.pad(surprise, (1, 0))                          # a_0 surprise = 0
        a = self.wp * prior + self.wr * surprise
        a = (a - a.mean(1, keepdim=True)) / (a.std(1, keepdim=True) + 1e-6)

        # ---- blocks / landmarks (Section 3) ----
        nb = math.ceil(N / self.B)
        pad = nb * self.B - N
        idx = torch.arange(N, device=dev)
        blk_of = (idx // self.B)                                    # (N,) block id per token
        block_end = torch.clamp((torch.arange(nb, device=dev) + 1) * self.B - 1, max=N - 1)  # (nb,)
        # attention-pool weights within each block
        pw = self.pool_q(x).squeeze(-1)                             # (B,N) pooling logits
        # build (B, nb, N) pool weights masked to block membership
        memb = (blk_of.view(1, nb, 1) == blk_of.view(1, 1, N)) if False else None
        # vectorised pooling: scatter softmax per block
        onehot = F.one_hot(blk_of, nb).float()                     # (N,nb)
        pw_exp = pw.exp().unsqueeze(-1) * onehot.unsqueeze(0)      # (B,N,nb)
        denom = pw_exp.sum(1, keepdim=True) + 1e-9                  # (B,1,nb)
        alpha = pw_exp / denom                                      # (B,N,nb) weights per block
        m = torch.einsum('bnc,bnd->bcd', alpha, x)                 # (B,nb,D) landmark vectors

        # g_b : pooled anchor evidence per block (Section 2->3)
        g = torch.einsum('bn,nc->bc', a, onehot) / (onehot.sum(0) + 1e-9)  # (B,nb)

        # depth-persistent landmark (Section 5)
        if self.depth_res and prev_landmarks is not None:
            m = self.ln(m + self.res(prev_landmarks))
        landmarks_out = m

        kL = self._heads(self.kL(m))                               # (B,H,nb,dh)
        vL = self._heads(self.vL(m))

        # ---- masks ----
        ar = torch.arange(N, device=dev)
        local = (ar.view(N, 1) >= ar.view(1, N)) & (ar.view(1, N) > ar.view(N, 1) - self.w)  # (N,N)
        # landmark visible iff its block ends strictly before the local window start
        win_start = (ar - self.w + 1).clamp(min=0)                 # (N,)
        vis = block_end.view(1, nb) < win_start.view(N, 1)         # (N,nb)

        # ---- logits ----
        scale = 1.0 / math.sqrt(self.dh)
        loc_logits = (q @ k.transpose(-1, -2)) * scale             # (B,H,N,N)
        loc_logits = loc_logits.masked_fill(~local.view(1, 1, N, N), float('-inf'))
        lm_logits = (q @ kL.transpose(-1, -2)) * scale + g.view(B, 1, 1, nb)  # (B,H,N,nb)
        lm_logits = lm_logits.masked_fill(~vis.view(1, 1, N, nb), float('-inf'))

        # ---- top-k landmark selection (Section 6) ----
        rank = lm_logits
        if self.learn_sel and self.training:
            u = torch.rand_like(rank).clamp_(1e-9, 1.0)
            rank = rank + (-torch.log(-torch.log(u)))              # Gumbel noise on ranking
        kk = min(self.k, nb)
        topk_idx = rank.topk(kk, dim=-1).indices                   # (B,H,N,kk)
        keep = torch.zeros_like(lm_logits, dtype=torch.bool).scatter_(-1, topk_idx, True)
        lm_logits = lm_logits.masked_fill(~keep, float('-inf'))    # exact softmax over sparse union

        # ---- joint exact softmax over union(local, selected landmarks) ----
        joint = torch.cat([loc_logits, lm_logits], dim=-1)         # (B,H,N,N+nb)
        # guard rows that are all -inf (shouldn't happen: self is always local)
        joint = torch.nan_to_num(joint, neginf=-1e9)
        w = joint.softmax(-1)
        vv = torch.cat([v, vL], dim=2)                             # (B,H,N+nb,dh)
        out = w @ vv                                               # (B,H,N,dh)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.o(out), landmarks_out


# ------------------------------ FFN / LoRA / MoE ---------------------------- #
class FFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.d_model * cfg.ff_mult
        self.w1 = nn.Linear(cfg.d_model, h); self.w2 = nn.Linear(h, cfg.d_model)
    def forward(self, x, lora=None):
        h = self.w1(x)
        if lora is not None: h = h + lora(x)
        return self.w2(F.gelu(h))

class LoRA(nn.Module):
    def __init__(self, di, do, r):
        super().__init__()
        self.a = nn.Linear(di, r, bias=False); self.b = nn.Linear(r, do, bias=False)
        nn.init.zeros_(self.b.weight)
    def forward(self, x): return self.b(self.a(x))

class MoEFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.n_experts)
        self.experts = nn.ModuleList([FFN(cfg) for _ in range(cfg.n_experts)])
    def forward(self, x):
        w = self.gate(x).softmax(-1); idx = w.argmax(-1); out = torch.zeros_like(x)
        for e, ex in enumerate(self.experts):
            mk = (idx == e).unsqueeze(-1)
            if mk.any(): out = out + mk * ex(x) * w[..., e:e+1]
        return out

class PlainAttention(nn.Module):
    """linear or full causal attention (for the baselines)."""
    def __init__(self, cfg, full):
        super().__init__()
        self.h, self.dh, self.full = cfg.n_heads, cfg.d_model // cfg.n_heads, full
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model); self.o = nn.Linear(cfg.d_model, cfg.d_model)
    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = [t.view(B, N, self.h, self.dh).transpose(1, 2) for t in (q, k, v)]
        a = full_attention(q, k, v) if self.full else linear_attention(q, k, v)
        return self.o(a.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    """Pre-norm attn + FFN. Substrate blocks may use anchor attention and thread
    landmarks; core/prelude/coda use plain attention."""
    def __init__(self, cfg, attn_kind="linear", moe=False):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.kind = attn_kind
        if attn_kind == "anchor":
            self.attn = AnchorAttention(cfg)
        else:
            self.attn = PlainAttention(cfg, full=(attn_kind == "full"))
        self.ff = MoEFFN(cfg) if moe else FFN(cfg)
        self.moe = moe
    def forward(self, x, lora=None, prev_landmarks=None):
        landmarks = None
        if self.kind == "anchor":
            a, landmarks = self.attn(self.n1(x), prev_landmarks)
            x = x + a
        else:
            x = x + self.attn(self.n1(x))
        x = x + (self.ff(self.n2(x)) if self.moe else self.ff(self.n2(x), lora))
        return (x, landmarks) if self.kind == "anchor" else x


class AttnRes(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(cfg.n_recur, cfg.d_model))
    def forward(self, stack, t):
        s = (stack * self.q[t]).sum(-1) / math.sqrt(stack.shape[-1])
        return (s.softmax(-1).unsqueeze(-1) * stack).sum(2)

class MoRRouter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.n_recur)
    def forward(self, e, sample=True):
        logits = self.fc(e); dist = torch.distributions.Categorical(logits=logits)
        di = dist.sample() if sample else logits.argmax(-1)
        return di + 1, dist.log_prob(di)

class RecurrentCore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block = Block(cfg, attn_kind="linear", moe=False)
        self.loras = nn.ModuleList([LoRA(cfg.d_model, cfg.d_model * cfg.ff_mult, cfg.lora_rank)
                                    for _ in range(cfg.n_recur)])
        self.router = MoRRouter(cfg); self.attnres = AttnRes(cfg)
    def forward(self, x, e, sample=True):
        cfg = self.cfg
        N_r = cfg.n_recur if cfg.use_recurrence else 1
        depths, logp = self.router(e, sample=sample)
        if not cfg.use_recurrence: depths = torch.ones_like(depths)
        current = e; stack = e.unsqueeze(2)
        for t in range(1, N_r + 1):
            inp = self.attnres(stack, t - 1) if cfg.use_attnres else current
            out = self.block(inp, lora=self.loras[t - 1])
            active = (depths >= t).float().unsqueeze(-1)
            current = active * out + (1 - active) * current
            stack = torch.cat([stack, current.unsqueeze(2)], dim=2)
        return current, depths, logp


class RDR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.prelude = Block(cfg, attn_kind="linear", moe=True)
        self.substrate = nn.ModuleList([Block(cfg, attn_kind=cfg.attn_kind)
                                        for _ in range(cfg.n_sub_layers)])
        self.core = RecurrentCore(cfg)
        self.coda = Block(cfg, attn_kind="linear", moe=True)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab)
        self.mtp_head = nn.Linear(cfg.d_model, cfg.vocab)
    def forward(self, ids, sample=True):
        x = self.prelude(self.embed(ids)); e = x
        prev_lm = None
        for lyr in self.substrate:
            if lyr.kind == "anchor":
                x, prev_lm = lyr(x, prev_landmarks=prev_lm)
            else:
                x = lyr(x)
        x, depths, logp = self.core(x, e, sample=sample)
        x = self.norm(self.coda(x))
        return self.lm_head(x), self.mtp_head(x), depths, logp


# ============================ KEYVAL retrieval task ========================= #
def gen_keyval(rng, n_pairs):
    keys = rng.sample("abcdefghijklmnopqrstuvwxyz", n_pairs)
    vals = [rng.randint(0, 9) for _ in range(n_pairs)]
    qi = rng.randrange(n_pairs)
    pairs = " ".join(f"{k}:{v}" for k, v in zip(keys, vals))
    return f"KV {pairs} GET {keys[qi]} =", str(vals[qi])

def verify(family, target, output):
    return output.strip() == target.strip()


# ================================= trainer ================================= #
def encode(s): return [ord(c) for c in s]
def make_example(row):
    inp, tgt = encode(row["input"]), encode(row["target"])
    ids = [BOS] + inp + tgt + [EOS]
    ans = [0] * (1 + len(inp)) + [1] * (len(tgt) + 1)
    return ids, ans

def build_dataset(n_per_cell, d_train, d_max, val_frac, seed):
    rng = random.Random(seed); rows, seen = [], set()
    for diff in range(2, d_max + 1):                 # n_pairs from 2..d_max
        made, tries = 0, 0
        while made < n_per_cell and tries < n_per_cell * 50:
            tries += 1
            inp, tgt = gen_keyval(rng, diff)
            if inp in seen: continue
            seen.add(inp)
            split = "test_extrap" if diff > d_train else ("val_interp" if rng.random() < val_frac else "train")
            rows.append({"family": "keyval", "difficulty": diff, "split": split,
                         "input": inp, "target": tgt}); made += 1
    rng.shuffle(rows)
    by = defaultdict(list)
    for r in rows:
        r["_ids"], r["_ans"] = make_example(r); by[r["split"]].append(r)
    return by

def collate(batch, device):
    L = max(len(r["_ids"]) for r in batch); B = len(batch)
    ids = torch.full((B, L), PAD, dtype=torch.long); ans = torch.zeros((B, L), dtype=torch.bool)
    real = torch.zeros((B, L), dtype=torch.bool)
    for i, r in enumerate(batch):
        n = len(r["_ids"])
        ids[i, :n] = torch.tensor(r["_ids"]); ans[i, :n] = torch.tensor(r["_ans"], dtype=torch.bool)
        real[i, :n] = True
    diffs = torch.tensor([r["difficulty"] for r in batch], dtype=torch.long)
    return ids.to(device), ans.to(device), real.to(device), diffs.to(device)

def answer_ce(logits, ids, ans):
    lo, tg, mk = logits[:, :-1], ids[:, 1:], ans[:, 1:]
    ce = F.cross_entropy(lo.reshape(-1, logits.shape[-1]), tg.reshape(-1), reduction="none").reshape(tg.shape) * mk
    return ce.sum() / mk.sum().clamp_min(1)

def exact_reward(logits, ids, ans):
    pred, tg, mk = logits[:, :-1].argmax(-1), ids[:, 1:], ans[:, 1:]
    return ((pred == tg) | (~mk)).all(dim=1).float()

def group_baseline(values, groups):
    adv = values.clone()
    for gg in groups.unique():
        mm = groups == gg; adv[mm] = values[mm] - values[mm].mean()
    return adv

def train_step(model, batch, opt, ponder):
    cfg = model.cfg; ids, ans, real, diffs = batch
    logits, _m, depths, logp = model(ids, sample=cfg.use_rlvr)
    ce = answer_ce(logits, ids, ans); reward = exact_reward(logits, ids, ans)
    md_ex = (depths.float() * real).sum(1) / real.sum(1).clamp_min(1); loss = ce
    if cfg.use_rlvr:
        ret = reward - ponder * (md_ex / cfg.n_recur); adv = group_baseline(ret, diffs)
        lp = (logp * real).sum(1) / real.sum(1).clamp_min(1)
        loss = ce + -(adv.detach() * lp).mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return dict(ce=ce.item(), reward=reward.mean().item(), depth=md_ex.mean().item())

@torch.no_grad()
def generate(model, prompt, device, max_new=4):
    ids = list(prompt); out = []
    for _ in range(max_new):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _m, _d, _l = model(x, sample=False)
        nxt = logits[0, -1].argmax().item()
        if nxt >= 128: break
        out.append(chr(nxt)); ids.append(nxt)
    return "".join(out)

@torch.no_grad()
def evaluate(model, rows, device, eval_n):
    model.eval(); rows = rows[:eval_n]; ok = 0; dt, do = defaultdict(int), defaultdict(int)
    for r in rows:
        a = generate(model, [BOS] + encode(r["input"]), device)
        c = verify(r["family"], r["target"], a); ok += c
        dt[r["difficulty"]] += 1; do[r["difficulty"]] += c
    model.train()
    return dict(n=len(rows), acc=ok / max(len(rows), 1),
                diff_acc={d: do[d] / dt[d] for d in sorted(dt)})


def train_one(cfg, by, device, steps, batch, lr, ponder, eval_n, log_every, tag, seed):
    torch.manual_seed(seed); random.seed(seed)
    model = RDR(cfg).to(device); npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr); tr = by["train"]; t0 = time.time()
    print(f"\n[{tag}] params {npar/1e3:.0f}K  attn={cfg.attn_kind} "
          f"w={cfg.sparse_attn_window} k={cfg.sparse_attn_topk} B={cfg.sparse_attn_block} "
          f"depthres={cfg.landmark_depth_residual}", flush=True)
    for s in range(1, steps + 1):
        m = train_step(model, collate(random.sample(tr, min(batch, len(tr))), device), opt, ponder)
        if s % log_every == 0 or s == 1:
            print(f"[{tag}] step {s:5d} | ce {m['ce']:.3f} reward {m['reward']:.3f} "
                  f"depth {m['depth']:.2f} | {time.time()-t0:.0f}s", flush=True)
    res = {}
    for sp in ("val_interp", "test_extrap"):
        ev = evaluate(model, by[sp], device, eval_n); res[sp] = ev
        print(f"[{tag}] [{sp}] acc={ev['acc']:.3f}  by_diff " +
              " ".join(f"d{d}={a:.2f}" for d, a in ev['diff_acc'].items()), flush=True)
    return {"params": npar, "results": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--ponder", type=float, default=0.0)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_pairs_train", type=int, default=10)   # d_train
    ap.add_argument("--n_pairs_max", type=int, default=16)     # d_max (extrap)
    ap.add_argument("--n_per_cell", type=int, default=1500)
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kinds", default="linear,full,anchor")
    ap.add_argument("--depth_res", action="store_true",
                    help="also run anchor with landmark depth-residual (Section 5)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    by = build_dataset(args.n_per_cell, args.n_pairs_train, args.n_pairs_max, 0.15, args.seed)
    maxlen = max(len(r["_ids"]) for v in by.values() for r in v)
    print(f"device {device} | data " + ", ".join(f"{k}={len(v)}" for k, v in by.items())
          + f" | max_seq_len {maxlen}", flush=True)

    base = dict(vocab=VOCAB, d_model=args.d_model, n_heads=4, n_sub_layers=4,
                n_recur=4, n_experts=4, ff_mult=2)
    out = {}
    for kind in args.kinds.split(","):
        cfg = Cfg(**base, attn_kind=kind, sparse_attn_window=8, sparse_attn_topk=4,
                  sparse_attn_block=6)
        out[kind] = train_one(cfg, by, device, args.steps, args.batch, args.lr,
                              args.ponder, args.eval_n, args.log_every, kind, args.seed)
    if args.depth_res:
        cfg = Cfg(**base, attn_kind="anchor", sparse_attn_window=8, sparse_attn_topk=4,
                  sparse_attn_block=6, landmark_depth_residual=True)
        out["anchor+dres"] = train_one(cfg, by, device, args.steps, args.batch, args.lr,
                                       args.ponder, args.eval_n, args.log_every, "anchor+dres", args.seed)

    print("\n=== SUMMARY (KEYVAL retrieval acc: interp / extrap) ===", flush=True)
    for tag, o in out.items():
        vi = o["results"]["val_interp"]["acc"]; te = o["results"]["test_extrap"]["acc"]
        print(f"  {tag:12s} -> interp {vi:.3f} / extrap {te:.3f}")
    json.dump(out, open("anchor_rdr_results.json", "w"), indent=2)
    print("\nwrote anchor_rdr_results.json")


if __name__ == "__main__":
    main()
