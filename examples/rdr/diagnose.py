#!/usr/bin/env python3
"""
diagnose.py — quantify the TRAIN vs INFERENCE gap for one trained RDR model.

Motivated by the question: "this is all training — does the model behave
differently at inference?" Two concrete gaps exist in this setup:

  (1) TEACHER FORCING vs AUTOREGRESSIVE.
      Training CE *and* the RLVR reward are teacher-forced (each answer token is
      scored given the TRUE previous tokens). Inference is autoregressive (the
      model feeds back its OWN tokens, so errors compound). We measure both,
      per difficulty, to size the gap.

  (2) DEPTH: SAMPLED (train) vs ARGMAX (inference).
      With RLVR the router SAMPLES depth during training but takes the ARGMAX
      (mode) at inference. The headline "flat depth" was the argmax. Here we log
      the full sampled-depth distribution (mean +/- std over many samples) per
      difficulty, alongside the argmax, to check whether difficulty-dependence
      is hidden by collapsing to the mode.

Trains one full RDR model in-process (no checkpoint needed) and reports a table.
"""
import argparse, time, random
from collections import defaultdict
import torch

from rdr import Cfg, RDR
from build_dataset import verify
import train as T   # reuse load/collate/exact_reward/generate/train_step/encode/specials


@torch.no_grad()
def teacher_forced_acc_by_diff(model, rows):
    """Exact-match accuracy under TEACHER FORCING, grouped by difficulty.
    Uses argmax-at-each-position given the TRUE prefix (train.exact_reward)."""
    model.eval()
    ok, tot = defaultdict(float), defaultdict(int)
    by_diff = defaultdict(list)
    for r in rows:
        by_diff[r["difficulty"]].append(r)
    for d, rs in by_diff.items():
        for i in range(0, len(rs), 64):
            batch = T.collate(rs[i:i + 64])
            ids, ans, real, _diffs = batch
            logits, _m, _dp, _lp = model(ids, sample=False)
            per_ex = T.exact_reward(logits, ids, ans)        # (B,) 1 iff all answer toks ok
            ok[d] += per_ex.sum().item(); tot[d] += len(rs[i:i + 64])
    model.train()
    return {d: ok[d] / tot[d] for d in sorted(tot)}


@torch.no_grad()
def autoregressive_acc_by_diff(model, rows):
    """Exact-match accuracy under real AUTOREGRESSIVE generation + verify()."""
    model.eval()
    ok, tot = defaultdict(float), defaultdict(int)
    for r in rows:
        prompt = [T.BOS] + T.encode(r["input"])
        ans, _md = T.generate(model, prompt)
        ok[r["difficulty"]] += verify(r["family"], r["target"], ans)
        tot[r["difficulty"]] += 1
    model.train()
    return {d: ok[d] / tot[d] for d in sorted(tot)}


@torch.no_grad()
def depth_stats_by_diff(model, rows, k_samples=32):
    """Per difficulty: argmax depth (inference) vs sampled-depth mean/std (train-time
    policy). Depth is measured on the PROMPT tokens, as in eval. K samples per prompt
    are drawn in one batched forward (sampling is independent across the batch)."""
    model.eval()
    argmax_d, samp_mean, samp_all = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        prompt = torch.tensor([[T.BOS] + T.encode(r["input"])], dtype=torch.long)
        P = prompt.shape[1]
        # argmax (inference) depth
        _l, _m, da, _lp = model(prompt, sample=False)
        argmax_d[r["difficulty"]].append(da[0, :P].float().mean().item())
        # sampled (training-policy) depth: K independent samples in one forward
        xb = prompt.repeat(k_samples, 1)
        _l, _m, ds, _lp = model(xb, sample=True)
        per_sample = ds[:, :P].float().mean(dim=1)           # (K,) mean depth over prompt
        samp_mean[r["difficulty"]].append(per_sample.mean().item())
        samp_all[r["difficulty"]].extend(per_sample.tolist())
    model.train()
    out = {}
    for d in sorted(argmax_d):
        t = torch.tensor(samp_all[d])
        out[d] = dict(argmax=sum(argmax_d[d]) / len(argmax_d[d]),
                      samp_mean=t.mean().item(), samp_std=t.std().item())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="rdr_dataset.jsonl")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--ponder", type=float, default=0.05)
    ap.add_argument("--max_new", type=int, default=12)
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--k_samples", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--n_recur", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    T.MAX_NEW = args.max_new

    torch.manual_seed(args.seed); random.seed(args.seed)
    by_split = T.load(args.data)
    train_rows = by_split["train"]
    cfg = Cfg(vocab=T.VOCAB, d_model=args.d_model, n_recur=args.n_recur)
    model = RDR(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"training full RDR ({sum(p.numel() for p in model.parameters())/1e3:.0f}K) "
          f"for {args.steps} steps on {args.data} ...")
    t0 = time.time()
    for s in range(1, args.steps + 1):
        b = T.collate(random.sample(train_rows, min(args.batch, len(train_rows))))
        m = T.train_step(model, b, opt, args.ponder)
        if s % 500 == 0 or s == 1:
            print(f"  step {s:5d} | ce {m['ce']:.3f} reward {m['reward']:.3f} "
                  f"depth {m['mean_depth']:.2f} | {time.time()-t0:.0f}s")

    for split in ("val_interp", "test_extrap"):
        rows = by_split[split][:args.eval_n]
        tf = teacher_forced_acc_by_diff(model, rows)
        ar = autoregressive_acc_by_diff(model, rows)
        dp = depth_stats_by_diff(model, rows, k_samples=args.k_samples)
        print(f"\n=== {split}  (n={len(rows)}) ===")
        print(f"{'diff':>4} | {'TF acc':>7} {'AR acc':>7} {'gap':>6} | "
              f"{'argmax d':>8} {'sampled d (mean±std)':>22}")
        print("-" * 64)
        for d in sorted(tf):
            g = tf[d] - ar[d]
            s = dp[d]
            print(f"{d:>4} | {tf[d]:>7.3f} {ar[d]:>7.3f} {g:>6.3f} | "
                  f"{s['argmax']:>8.2f} {s['samp_mean']:>10.2f} ± {s['samp_std']:<8.2f}")
        tf_o = sum(tf.values()) / len(tf); ar_o = sum(ar.values()) / len(ar)
        print(f"  overall TF {tf_o:.3f}  AR {ar_o:.3f}  gap {tf_o-ar_o:.3f}")


if __name__ == "__main__":
    main()
