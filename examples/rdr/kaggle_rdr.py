#!/usr/bin/env python3
"""
kaggle_rdr.py — self-contained moderate-scale RDR training run for a Kaggle GPU.

Everything (model + dataset + exact verifier + trainer + eval) is in this one
file, so you can paste it into a single Kaggle notebook cell (or add it as a
script kernel), switch the accelerator to GPU (T4), and hit Run. No internet, no
external data needed.

It trains TWO models at the SAME moderate scale and compares them:
    * full RDR        (recurrence + AttnRes + RLVR)         <- "our mechanism"
    * depth-1 baseline (no recurrence loop)                 <- the control
so we can see whether, at this larger scale, the recurrent-depth mechanism finally
earns its keep on interpolation and on the harder extrapolation split.

Scale (T4-friendly): d_model 256, 6 substrate layers, n_recur 8, 8 experts.
Dataset: 3 exact verifiable families, difficulty == reasoning hops, d_train 5,
extrapolation difficulties 6..8. Final-value targets (short), exact-match verifier.

If you OOM, lower BATCH or D_MODEL at the top of CONFIG.
"""
import math, json, time, random, argparse
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================== CONFIG (edit here) ============================= #
D_MODEL   = 256
N_HEADS   = 8
N_SUB     = 6
N_RECUR   = 8          # >= D_MAX so depth-per-hop is representable
N_EXPERTS = 8
FF_MULT   = 4
LORA_RANK = 8

BATCH     = 128
STEPS     = 12000
LR        = 1.5e-3
PONDER    = 0.05

N_PER_CELL = 2000
D_TRAIN    = 5
D_MAX      = 8         # extrapolation difficulties: 6,7,8
VAL_FRAC   = 0.15

EVAL_N    = 400
MAX_NEW   = 8
LOG_EVERY = 500
SEED      = 0

PAD, BOS, EOS = 256, 257, 258
VOCAB = 259


@dataclass
class Cfg:
    vocab: int = VOCAB
    d_model: int = D_MODEL
    n_heads: int = N_HEADS
    n_sub_layers: int = N_SUB
    full_attn_every: int = 3
    n_recur: int = N_RECUR
    lora_rank: int = LORA_RANK
    n_experts: int = N_EXPERTS
    ff_mult: int = FF_MULT
    use_recurrence: bool = True
    use_attnres: bool = True
    use_rlvr: bool = True


# ============================== MODEL (rdr.py) ============================== #
def feature_map(x):
    return F.elu(x) + 1.0

def linear_attention(q, k, v):
    q, k = feature_map(q), feature_map(k)
    kv = k.unsqueeze(-1) * v.unsqueeze(-2)
    kv_cum = kv.cumsum(dim=2)
    z_cum = k.cumsum(dim=2)
    num = (q.unsqueeze(-1) * kv_cum).sum(dim=-2)
    den = (q * z_cum).sum(dim=-1, keepdim=True) + 1e-6
    return num / den

def full_attention(q, k, v):
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

class FFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.d_model * cfg.ff_mult
        self.w1 = nn.Linear(cfg.d_model, h)
        self.w2 = nn.Linear(h, cfg.d_model)
    def forward(self, x, lora=None):
        h = self.w1(x)
        if lora is not None:
            h = h + lora(x)
        return self.w2(F.gelu(h))

class LoRA(nn.Module):
    def __init__(self, d_in, d_out, r):
        super().__init__()
        self.a = nn.Linear(d_in, r, bias=False)
        self.b = nn.Linear(r, d_out, bias=False)
        nn.init.zeros_(self.b.weight)
    def forward(self, x): return self.b(self.a(x))

class MoEFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.n_experts)
        self.experts = nn.ModuleList([FFN(cfg) for _ in range(cfg.n_experts)])
    def forward(self, x):
        w = self.gate(x).softmax(-1)
        idx = w.argmax(-1)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            m = (idx == e).unsqueeze(-1)
            if m.any():
                out = out + m * expert(x) * w[..., e:e+1]
        return out

class Block(nn.Module):
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

class AttnRes(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(cfg.n_recur, cfg.d_model))
    def forward(self, stack, t):
        scores = (stack * self.q[t]).sum(-1) / math.sqrt(stack.shape[-1])
        w = scores.softmax(-1)
        return (w.unsqueeze(-1) * stack).sum(2)

class MoRRouter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.n_recur)
        self.n_recur = cfg.n_recur
    def forward(self, e, sample=True):
        logits = self.fc(e)
        dist = torch.distributions.Categorical(logits=logits)
        depth_idx = dist.sample() if sample else logits.argmax(-1)
        return depth_idx + 1, dist.log_prob(depth_idx)

class RecurrentCore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block = Block(cfg, full=False, moe=False)
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
        stack = e.unsqueeze(2)
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
        self.prelude = Block(cfg, full=False, moe=True)
        sub = []
        for i in range(cfg.n_sub_layers):
            sub.append(Block(cfg, full=((i + 1) % cfg.full_attn_every == 0)))
        self.substrate = nn.ModuleList(sub)
        self.core = RecurrentCore(cfg)
        self.coda = Block(cfg, full=False, moe=True)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab)
        self.mtp_head = nn.Linear(cfg.d_model, cfg.vocab)
    def forward(self, ids, sample=True):
        x = self.embed(ids)
        x = self.prelude(x)
        e = x
        for lyr in self.substrate:
            x = lyr(x)
        x, depths, logp = self.core(x, e, sample=sample)
        x = self.coda(x)
        x = self.norm(x)
        return self.lm_head(x), self.mtp_head(x), depths, logp


# ====================== DATASET (build_dataset.py) ========================== #
def gen_modchain(rng, diff):
    m = rng.choice([7, 11, 13, 17, 19, 23])
    a = rng.randint(0, m - 1); c = rng.randint(1, m - 1); d = rng.randint(0, m - 1)
    x = a
    for _ in range(diff):
        x = (c * x + d) % m
    return f"MODCHAIN a={a} c={c} d={d} m={m} k={diff} =", str(x)

def gen_ptrchase(rng, diff):
    n = rng.randint(6, 10)
    lst = [rng.randint(0, n - 1) for _ in range(n)]
    start = rng.randint(0, n - 1)
    idx = start
    for _ in range(diff):
        idx = lst[idx]
    return f"PTRCHASE list={lst} start={start} k={diff} =", str(idx)

def _build_expr(rng, depth):
    if depth == 0:
        v = rng.randint(1, 4); return str(v), v
    ls, lv = _build_expr(rng, depth - 1)
    rv = rng.randint(1, 4); op = rng.choice(["+", "-", "*"])
    val = {"+": lv + rv, "-": lv - rv, "*": lv * rv}[op]
    return f"({ls}{op}{rv})", val

def gen_eval(rng, diff):
    s, v = _build_expr(rng, diff)
    return f"EVAL {s} =", str(v)

FAMILIES = {"modchain": gen_modchain, "ptrchase": gen_ptrchase, "eval": gen_eval}

def verify(family, target, output):
    return output.strip() == target.strip()

def build_dataset(seed):
    rng = random.Random(seed)
    rows, seen = [], set()
    for fam, gen in FAMILIES.items():
        for diff in range(1, D_MAX + 1):
            made, tries = 0, 0
            while made < N_PER_CELL and tries < N_PER_CELL * 50:
                tries += 1
                inp, tgt = gen(rng, diff)
                if inp in seen:
                    continue
                seen.add(inp)
                if diff > D_TRAIN:
                    split = "test_extrap"
                else:
                    split = "val_interp" if rng.random() < VAL_FRAC else "train"
                rows.append({"family": fam, "difficulty": diff, "split": split,
                             "input": inp, "target": tgt})
                made += 1
    rng.shuffle(rows)
    by = defaultdict(list)
    for r in rows:
        r["_ids"], r["_ans"] = make_example(r)
        by[r["split"]].append(r)
    return by


# ============================ TRAINER (train.py) ============================ #
def encode(s):
    return [ord(c) for c in s]

def make_example(row):
    inp = encode(row["input"]); tgt = encode(row["target"])
    ids = [BOS] + inp + tgt + [EOS]
    ans = [0] * (1 + len(inp)) + [1] * (len(tgt) + 1)
    return ids, ans

def collate(batch, device):
    L = max(len(r["_ids"]) for r in batch); B = len(batch)
    ids = torch.full((B, L), PAD, dtype=torch.long)
    ans = torch.zeros((B, L), dtype=torch.bool)
    real = torch.zeros((B, L), dtype=torch.bool)
    for i, r in enumerate(batch):
        n = len(r["_ids"])
        ids[i, :n] = torch.tensor(r["_ids"])
        ans[i, :n] = torch.tensor(r["_ans"], dtype=torch.bool)
        real[i, :n] = True
    diffs = torch.tensor([r["difficulty"] for r in batch], dtype=torch.long)
    return ids.to(device), ans.to(device), real.to(device), diffs.to(device)

def answer_ce(logits, ids, ans):
    logit = logits[:, :-1]; target = ids[:, 1:]; mask = ans[:, 1:]
    ce = F.cross_entropy(logit.reshape(-1, logits.shape[-1]),
                         target.reshape(-1), reduction="none").reshape(target.shape) * mask
    return ce.sum() / mask.sum().clamp_min(1)

def exact_reward(logits, ids, ans):
    pred = logits[:, :-1].argmax(-1); target = ids[:, 1:]; mask = ans[:, 1:]
    correct_tok = (pred == target) | (~mask)
    return correct_tok.all(dim=1).float()

def group_baseline(values, groups):
    adv = values.clone()
    for g in groups.unique():
        m = groups == g
        adv[m] = values[m] - values[m].mean()
    return adv

def train_step(model, batch, opt, ponder_coef):
    cfg = model.cfg
    ids, ans, real, diffs = batch
    logits, _m, depths, logp = model(ids, sample=cfg.use_rlvr)
    ce = answer_ce(logits, ids, ans)
    reward = exact_reward(logits, ids, ans)
    md_ex = (depths.float() * real).sum(1) / real.sum(1).clamp_min(1)
    md = md_ex.mean()
    loss = ce
    if cfg.use_rlvr:
        ret = reward - ponder_coef * (md_ex / cfg.n_recur)
        adv = group_baseline(ret, diffs)
        lp = (logp * real).sum(1) / real.sum(1).clamp_min(1)
        loss = ce + -(adv.detach() * lp).mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return dict(loss=loss.item(), ce=ce.item(), reward=reward.mean().item(),
                mean_depth=md.item())

@torch.no_grad()
def generate(model, prompt_ids, device):
    ids = list(prompt_ids); P = len(ids); out = []; mean_depth = None
    for _ in range(MAX_NEW):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _m, depths, _lp = model(x, sample=False)
        if mean_depth is None:
            mean_depth = depths[0, :P].float().mean().item()
        nxt = logits[0, -1].argmax().item()
        if nxt >= 128:
            break
        out.append(chr(nxt)); ids.append(nxt)
    return "".join(out), mean_depth

@torch.no_grad()
def evaluate(model, rows, device):
    model.eval()
    rows = rows[:EVAL_N]
    correct = 0
    fam_t, fam_o = defaultdict(int), defaultdict(int)
    dif_t, dif_o = defaultdict(int), defaultdict(int)
    dif_d = defaultdict(list)
    for r in rows:
        ans, md = generate(model, [BOS] + encode(r["input"]), device)
        ok = verify(r["family"], r["target"], ans)
        correct += ok
        fam_t[r["family"]] += 1; fam_o[r["family"]] += ok
        dif_t[r["difficulty"]] += 1; dif_o[r["difficulty"]] += ok
        dif_d[r["difficulty"]].append(md)
    model.train()
    return dict(n=len(rows), acc=correct / max(len(rows), 1),
                fam_acc={f: fam_o[f] / fam_t[f] for f in fam_t},
                diff_acc={d: dif_o[d] / dif_t[d] for d in sorted(dif_t)},
                diff_depth={d: sum(v) / len(v) for d, v in sorted(dif_d.items())})

def train_model(cfg, by, device, tag):
    torch.manual_seed(SEED); random.seed(SEED)
    model = RDR(cfg).to(device)
    n = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    train_rows = by["train"]
    print(f"\n[{tag}] params {n/1e6:.2f}M  device={device}  "
          f"(recur={cfg.use_recurrence} attnres={cfg.use_attnres} rlvr={cfg.use_rlvr})",
          flush=True)
    t0 = time.time()
    for s in range(1, STEPS + 1):
        b = collate(random.sample(train_rows, min(BATCH, len(train_rows))), device)
        m = train_step(model, b, opt, PONDER)
        if s % LOG_EVERY == 0 or s == 1:
            print(f"[{tag}] step {s:6d} | ce {m['ce']:.3f} reward {m['reward']:.3f} "
                  f"depth {m['mean_depth']:.2f} | {time.time()-t0:.0f}s", flush=True)
    res = {}
    for split in ("val_interp", "test_extrap"):
        ev = evaluate(model, by[split], device)
        res[split] = ev
        print(f"[{tag}] [{split}] n={ev['n']} acc={ev['acc']:.3f}", flush=True)
        print(f"[{tag}]     family: " + " ".join(f"{f}={a:.2f}" for f, a in ev['fam_acc'].items()))
        print(f"[{tag}]     diff  : " + " ".join(f"d{d}={a:.2f}" for d, a in ev['diff_acc'].items()))
        print(f"[{tag}]     depth : " + " ".join(f"d{d}={z:.2f}" for d, z in ev['diff_depth'].items()))
    return {"params": n, "results": res}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    if device == "cpu":
        print("WARNING: no GPU detected — enable the GPU accelerator on Kaggle.", flush=True)
    by = build_dataset(SEED)
    print("data: " + ", ".join(f"{k}={len(v)}" for k, v in by.items()), flush=True)

    out = {}
    out["full_rdr"] = train_model(Cfg(), by, device, "full_rdr")
    out["depth1"]   = train_model(Cfg(use_recurrence=False, use_attnres=False), by, device, "depth1")

    print("\n=== SUMMARY (acc: val_interp / test_extrap) ===", flush=True)
    for tag, o in out.items():
        vi = o["results"]["val_interp"]["acc"]; te = o["results"]["test_extrap"]["acc"]
        print(f"  {tag:10s} -> interp {vi:.3f} / extrap {te:.3f}")
    json.dump(out, open("kaggle_rdr_results.json", "w"), indent=2)
    print("\nwrote kaggle_rdr_results.json")


if __name__ == "__main__":
    main()
