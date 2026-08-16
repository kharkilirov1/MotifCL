"""
parity_gate.py — Ladder step B: finite-state synapse vs BF16/FP32 AdamW baseline
on a REAL char-level LM task (tinyshakespeare), same arch/data/seed/steps/compute.

Question it answers: does the 45/63-state counter synapse track a normal AdamW
linear on a non-toy next-token task, or does it diverge? This is the gate the
research doc says must pass before any kernel work.
"""
from __future__ import annotations
import argparse, math, os, time, urllib.request
import torch, torch.nn as nn, torch.nn.functional as F

from counter_state_fused_v2 import CompactCounterLinear, fmt_bytes
from counter_state_rms import RMSCounterLinear

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def _synthetic_corpus(n_chars: int = 200_000) -> str:
    """Offline fallback when tinyshakespeare can't be fetched (e.g. no network).

    A deterministic, mildly structured char stream: a small vocabulary with
    word-like runs and spacing, so a char-LM has real next-token signal to fit and
    the finite-state-vs-AdamW comparison still means something. It is NOT shakespeare
    and the absolute losses differ; for a publishable parity number use the real
    corpus via --data-path. The gate prints which source it used."""
    import random
    rng = random.Random(1234)
    letters = "abcdefghijklmnopqrstuvwxyz"
    words = ["".join(rng.choice(letters) for _ in range(rng.randint(2, 8)))
             for _ in range(96)]
    out = []
    while sum(len(w) + 1 for w in out) < n_chars:
        out.append(rng.choice(words))
        r = rng.random()
        out.append("\n" if r < 0.08 else (", " if r < 0.2 else (". " if r < 0.32 else " ")))
    return "".join(out)


def load_data(device, data_path: str | None = None):
    # Resolution order: explicit --data-path, then a cached/co-located file, then a
    # network download, then a deterministic synthetic fallback (offline-safe).
    candidates = []
    if data_path:
        candidates.append(data_path)
    candidates.append(os.path.join(os.path.dirname(__file__), "tinyshakespeare.txt"))

    text = None
    source = None
    for path in candidates:
        if os.path.exists(path):
            text = open(path, encoding="utf-8").read()
            source = path
            break

    if text is None:
        if data_path:
            raise FileNotFoundError(f"--data-path not found: {data_path}")
        cache = candidates[-1]
        try:
            urllib.request.urlretrieve(DATA_URL, cache)
            text = open(cache, encoding="utf-8").read()
            source = cache + " (downloaded)"
        except Exception as exc:  # offline / blocked: fall back, don't crash the gate
            text = _synthetic_corpus()
            source = f"SYNTHETIC fallback (download failed: {exc})"

    print(f"corpus source: {source}")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n].to(device), data[n:].to(device), len(chars)


def get_batch(data, block, batch, device, g):
    ix = torch.randint(0, len(data) - block - 1, (batch,), generator=g)
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


class TernaryQATLinear(nn.Module):
    """BitNet-b1.58-style ternary QAT: FP32 master weight, per-row absmean ternary
    in forward, straight-through gradient to the master. Adam updates the master.
    Same ternary inference as finite-state, but with a full FP32 optimizer — so the
    finite-state vs this gap isolates the cost of the 6-bit counter optimizer."""

    def __init__(self, fin, fout, init_gain=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(fout, fin))
        std = init_gain * math.sqrt(1.0 / fin)
        nn.init.normal_(self.weight, mean=0.0, std=std)

    def forward(self, x):
        w = self.weight
        scale = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-5)  # per-row, like finite-state
        wq = (w / scale).round().clamp_(-1, 1) * scale
        wq = w + (wq - w).detach()  # straight-through estimator
        return F.linear(x, wq)


def make_lin(kind, fin, fout, init_gain, C, lr, lr_scale, tile_rows, pulse_mode):
    if kind == "fs":
        return CompactCounterLinear(fin, fout, C=C, lr=lr, lr_scale=lr_scale,
                                    init_gain=init_gain, tile_rows=tile_rows,
                                    pulse_mode=pulse_mode)
    if kind == "fs_rms":
        return RMSCounterLinear(fin, fout, C=C, lr=lr, lr_scale=lr_scale,
                                init_gain=init_gain, tile_rows=tile_rows,
                                pulse_mode=pulse_mode, use_rms=True)
    if kind == "qat":
        return TernaryQATLinear(fin, fout, init_gain)
    lin = nn.Linear(fin, fout, bias=False)
    # match the finite-state init variance: Var(w)=gain^2/fan_in
    std = init_gain * math.sqrt(1.0 / fin)
    nn.init.normal_(lin.weight, mean=0.0, std=std)
    return lin


class Block(nn.Module):
    def __init__(self, kind, d, n_head, n_layer, **kw):
        super().__init__()
        rg = 1.0 / math.sqrt(2.0 * n_layer)
        self.ln1 = nn.LayerNorm(d)
        self.q = make_lin(kind, d, d, 1.0, **kw)
        self.k = make_lin(kind, d, d, 1.0, **kw)
        self.v = make_lin(kind, d, d, 1.0, **kw)
        self.proj = make_lin(kind, d, d, rg, **kw)
        self.ln2 = nn.LayerNorm(d)
        self.fc = make_lin(kind, d, 4 * d, 1.0, **kw)
        self.fc2 = make_lin(kind, 4 * d, d, rg, **kw)
        self.n_head = n_head

    def forward(self, x):
        b, t, c = x.shape
        h = self.ln1(x)
        q = self.q(h).view(b, t, self.n_head, -1).transpose(1, 2)
        k = self.k(h).view(b, t, self.n_head, -1).transpose(1, 2)
        v = self.v(h).view(b, t, self.n_head, -1).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.proj(a)
        x = x + self.fc2(F.gelu(self.fc(self.ln2(x))))
        return x


class GPT(nn.Module):
    def __init__(self, kind, vocab, block_size, n_layer, n_head, d, **kw):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(kind, d, n_head, n_layer, **kw) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight  # tie
        self.kind = kind

    def forward(self, idx, targets=None):
        _, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def counter_layers(self):
        return [m for m in self.modules() if isinstance(m, CompactCounterLinear)]


def is_counter_child(model):
    counter_ids = set()
    for m in model.counter_layers():
        for p in m.buffers():
            counter_ids.add(id(p))
    return counter_ids


@torch.no_grad()
def eval_loss(model, data, block, batch, device, g, iters=20):
    model.eval()
    tot = 0.0
    for _ in range(iters):
        x, y = get_batch(data, block, batch, device, g)
        _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / iters


def mem_report(model):
    fs = model.counter_layers()
    counter_w = sum(m.state.numel() for m in fs)
    # non-counter trainable params (embedding, norm, head)
    fp = 0
    seen = set()
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        fp += p.numel()
    if fs:
        # finite-state: packed 6-bit weights + fp params (no per-weight Adam)
        bytes_w = counter_w * 6 / 8
        bytes_opt = 0  # row scales O(d) negligible; counted in buffers
        total = bytes_w + fp * 4 + fp * 8  # fp params fp32 + AdamW(m,v) for the few fp params
        return counter_w, fp, total
    else:
        # baseline: fp32 weights + AdamW (m,v) => ~12 bytes/param
        total = fp * 4 + fp * 8
        return 0, fp, total


def train(kind, args, vocab, train_data, val_data, device):
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(1234)  # SAME data stream for both runs
    fs_lr = args.fs_lr_rms if kind == "fs_rms" else args.fs_lr
    kw = dict(C=args.C, lr=fs_lr, lr_scale=args.fs_lr_scale,
              tile_rows=args.tile_rows, pulse_mode="direct")
    model = GPT(kind, vocab, args.block, args.n_layer, args.n_head, args.d, **kw).to(device)

    if kind in ("fs", "fs_rms"):
        counter_ids = is_counter_child(model)
        fp_params = [p for p in model.parameters() if id(p) not in counter_ids]
        opt = torch.optim.AdamW(fp_params, lr=args.fp_lr)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.base_lr)

    cw, fp, mem = mem_report(model)
    curve = []
    t0 = time.time()
    for step in range(args.steps + 1):
        x, y = get_batch(train_data, args.block, args.batch, device, g)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()  # fs: counter layers update here
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            vl = eval_loss(model, val_data, args.block, args.batch, device,
                           torch.Generator().manual_seed(99))
            curve.append((step, loss.item(), vl))
            print(f"[{kind:8s}] step={step:4d} train={loss.item():.4f} val={vl:.4f}")
    dt = time.time() - t0
    return dict(kind=kind, curve=curve, counter_w=cw, fp=fp, mem=mem, sec=dt,
                final_val=curve[-1][2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--C", type=int, default=8)
    ap.add_argument("--fs-lr", type=float, default=0.04)
    ap.add_argument("--fs-lr-rms", type=float, default=3e-3)
    ap.add_argument("--fs-lr-scale", type=float, default=2e-4)
    ap.add_argument("--fp-lr", type=float, default=2e-3)
    ap.add_argument("--base-lr", type=float, default=3e-3)
    ap.add_argument("--tile-rows", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--kinds", type=str, default="baseline,qat,fs,fs_rms")
    ap.add_argument("--data-path", type=str, default=None,
                    help="path to a char corpus (e.g. tinyshakespeare input.txt). "
                         "If omitted: cached file, then download, then synthetic fallback.")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    train_data, val_data, vocab = load_data(device, args.data_path)
    print(f"data: vocab={vocab} train_tokens={len(train_data)} val_tokens={len(val_data)}")
    print(f"arch: L={args.n_layer} d={args.d} head={args.n_head} block={args.block} "
          f"batch={args.batch} steps={args.steps} | fs C={args.C}")
    print("=" * 70)

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    res = {}
    for k in kinds:
        res[k] = train(k, args, vocab, train_data, val_data, device)
        print("-" * 70)
    print("=" * 70)

    label = {"baseline": "FP32", "qat": "ternary-QAT", "fs": "finite-state",
             "fs_rms": "finite-state+RMS"}
    ref = res.get("baseline") or res.get("qat") or next(iter(res.values()))
    rv = ref["final_val"]
    print("FINAL val loss:")
    for k in kinds:
        v = res[k]["final_val"]
        g = v - rv
        print(f"  {label.get(k, k):18s} {v:.4f}   gap {g:+.4f} ({g/rv*100:+.1f}%)")
    if "qat" in res:
        qv = res["qat"]["final_val"]
        for k in ("fs", "fs_rms"):
            if k in res:
                iso = res[k]["final_val"] - qv
                print(f"  ISOLATION {label[k]} - ternary-QAT = {iso:+.4f} ({iso/qv*100:+.1f}%)  "
                      f"= pure counter-optimizer cost")
        if "fs" in res and "fs_rms" in res:
            base_iso = res["fs"]["final_val"] - qv
            closed = base_iso - (res["fs_rms"]["final_val"] - qv)
            if abs(base_iso) > 1e-9:
                print(f"  >>> row-RMS closes {closed:+.4f} = {closed/base_iso*100:.0f}% of counter gap <<<")
    cw = next((res[k]["counter_w"] for k in res if res[k]["counter_w"] > 0), 0)
    print(f"counter weights: {cw/1e6:.3f}M | fp params: {ref['fp']/1e6:.3f}M")
    print("param+opt mem (training): " +
          "  ".join(f"{label.get(k, k)}~{fmt_bytes(res[k]['mem'])}" for k in kinds))
    print("wall: " + "  ".join(f"{label.get(k, k)}={res[k]['sec']:.0f}s" for k in kinds))


if __name__ == "__main__":
    main()
