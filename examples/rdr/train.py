#!/usr/bin/env python3
"""
train.py — train the RDR skeleton on the bounded verifiable-task dataset.

This wires together the two reference pieces:
  * rdr.py            -> the Recurrent-Depth Reasoner model
  * build_dataset.py  -> 3 verifiable families where difficulty == reasoning depth

What this script proves (the mechanism study), end to end:
  1. The model can be trained char-level on the tasks (supervised next-token CE
     on the ANSWER region only).
  2. The RLVR loop uses the EXACT verifier (build_dataset.verify) as reward — not
     the next-token-accuracy stub in rdr.py — to shape the MoR router's per-token
     recursion depth via REINFORCE.
  3. The central hypothesis is *measured*: does the router allocate MORE depth to
     HARDER difficulties, and does this transfer to the extrapolation split
     (difficulties strictly harder than anything seen in training)?

Tokenisation: char/byte level. Specials live above the byte range so they never
collide with real characters:  PAD=256, BOS=257, EOS=258  -> vocab=259.

Usage:
  python build_dataset.py --out rdr_dataset.jsonl
  python train.py --data rdr_dataset.jsonl --steps 4000
"""
import argparse, json, math, time, random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdr import Cfg, RDR
from build_dataset import verify

PAD, BOS, EOS = 256, 257, 258
VOCAB = 259
MAX_NEW = 8            # max answer tokens to generate at eval


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
def encode(s):
    return [ord(c) for c in s]          # all task chars are ASCII (<128)

def make_example(row):
    """Return (ids, ans_mask) where ans_mask[i]=1 iff token i is part of the
    answer region (target chars + EOS) — i.e. a token we want the model to PRODUCE."""
    inp = encode(row["input"])
    tgt = encode(row["target"])
    ids = [BOS] + inp + tgt + [EOS]
    ans = [0] * (1 + len(inp)) + [1] * (len(tgt) + 1)    # target + EOS are "answer"
    assert len(ids) == len(ans)
    return ids, ans

def load(path):
    rows = [json.loads(l) for l in open(path)]
    by_split = defaultdict(list)
    for r in rows:
        r["_ids"], r["_ans"] = make_example(r)
        by_split[r["split"]].append(r)
    return by_split

def collate(batch):
    L = max(len(r["_ids"]) for r in batch)
    B = len(batch)
    ids = torch.full((B, L), PAD, dtype=torch.long)
    ans = torch.zeros((B, L), dtype=torch.bool)
    real = torch.zeros((B, L), dtype=torch.bool)        # non-pad tokens
    for i, r in enumerate(batch):
        n = len(r["_ids"])
        ids[i, :n] = torch.tensor(r["_ids"])
        ans[i, :n] = torch.tensor(r["_ans"], dtype=torch.bool)
        real[i, :n] = True
    return ids, ans, real


# --------------------------------------------------------------------------- #
#  Losses / reward
# --------------------------------------------------------------------------- #
def answer_ce(logits, ids, ans):
    """Next-token CE restricted to answer positions. Position i predicts ids[i+1];
    we supervise it iff ids[i+1] is an answer token."""
    logit = logits[:, :-1]                       # predict next
    target = ids[:, 1:]
    mask = ans[:, 1:]                            # the predicted token is an answer token
    ce = F.cross_entropy(logit.reshape(-1, logits.shape[-1]),
                         target.reshape(-1), reduction="none")
    ce = ce.reshape(target.shape) * mask
    return ce.sum() / mask.sum().clamp_min(1)

def exact_reward(logits, ids, ans):
    """EXACT sequence-level verifier reward under teacher forcing: reward=1 iff the
    model's greedy prediction matches the FULL answer (every answer token), else 0.
    This is the discrete, executable-verifier-style signal (build_dataset.verify is
    string-exact; teacher-forced argmax over the answer span is its tensor analogue)."""
    pred = logits[:, :-1].argmax(-1)
    target = ids[:, 1:]
    mask = ans[:, 1:]
    correct_tok = (pred == target) | (~mask)     # ignore non-answer positions
    per_ex = correct_tok.all(dim=1).float()      # all answer tokens correct -> 1.0
    return per_ex                                # (B,)


# --------------------------------------------------------------------------- #
#  Train step
# --------------------------------------------------------------------------- #
def train_step(model, batch, opt, ponder_coef):
    cfg = model.cfg
    ids, ans, real = batch
    logits, _mtp, depths, logp = model(ids, sample=cfg.use_rlvr)

    ce = answer_ce(logits, ids, ans)
    loss = ce
    reward = exact_reward(logits, ids, ans)

    metrics = {"ce": ce.item(), "reward": reward.mean().item()}
    # mean depth over REAL tokens only
    md = (depths.float() * real).sum() / real.sum().clamp_min(1)
    metrics["mean_depth"] = md.item()

    if cfg.use_rlvr:
        baseline = reward.mean()
        # REINFORCE: exact verifier reward shapes router depth choices.
        # logp is per-token; weight by real-token mask, average per example.
        lp = (logp * real).sum(1) / real.sum(1).clamp_min(1)        # (B,)
        reinforce = -((reward - baseline).detach() * lp).mean()
        ponder = ponder_coef * md
        loss = ce + reinforce + ponder
        metrics["reinforce"] = reinforce.item()

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    metrics["loss"] = loss.item()
    return metrics


# --------------------------------------------------------------------------- #
#  Evaluation — REAL autoregressive generation + exact verify()
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate(model, prompt_ids):
    """Greedy autoregressive decode from a single prompt (list of ids ending at '=').
    Returns (decoded_answer_str, mean_depth_on_prompt)."""
    ids = list(prompt_ids)
    prompt_len = len(ids)
    out_chars = []
    mean_depth = None
    for _ in range(MAX_NEW):
        x = torch.tensor([ids], dtype=torch.long)
        logits, _m, depths, _lp = model(x, sample=False)
        if mean_depth is None:                       # depth measured on the prompt
            mean_depth = depths[0, :prompt_len].float().mean().item()
        nxt = logits[0, -1].argmax().item()
        if nxt == EOS:
            break
        if nxt < 128:
            out_chars.append(chr(nxt))
        ids.append(nxt)
    return "".join(out_chars), mean_depth

@torch.no_grad()
def evaluate(model, rows, max_n=None):
    """Returns dict with overall accuracy, per-family acc, per-difficulty acc and
    mean router depth. Uses exact verify() on autoregressively generated answers."""
    model.eval()
    if max_n:
        rows = rows[:max_n]
    n = len(rows)
    correct = 0
    fam_tot, fam_ok = defaultdict(int), defaultdict(int)
    diff_tot, diff_ok = defaultdict(int), defaultdict(int)
    diff_depth = defaultdict(list)
    for r in rows:
        prompt = [BOS] + encode(r["input"])
        ans, md = generate(model, prompt)
        ok = verify(r["family"], r["target"], ans)
        correct += ok
        fam_tot[r["family"]] += 1; fam_ok[r["family"]] += ok
        diff_tot[r["difficulty"]] += 1; diff_ok[r["difficulty"]] += ok
        diff_depth[r["difficulty"]].append(md)
    model.train()
    return {
        "n": n,
        "acc": correct / max(n, 1),
        "fam_acc": {f: fam_ok[f] / fam_tot[f] for f in fam_tot},
        "diff_acc": {d: diff_ok[d] / diff_tot[d] for d in sorted(diff_tot)},
        "diff_depth": {d: sum(v) / len(v) for d, v in sorted(diff_depth.items())},
    }


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def run(args, cfg, log_prefix=""):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    by_split = load(args.data)
    train_rows = by_split["train"]
    print(f"{log_prefix}data: " + ", ".join(f"{k}={len(v)}" for k, v in by_split.items()))

    model = RDR(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"{log_prefix}model params: {n_params/1e3:.1f}K  "
          f"(d_model={cfg.d_model}, n_recur={cfg.n_recur}, "
          f"recur={cfg.use_recurrence} attnres={cfg.use_attnres} rlvr={cfg.use_rlvr})")

    t0 = time.time()
    for stepi in range(1, args.steps + 1):
        batch = collate(random.sample(train_rows, min(args.batch, len(train_rows))))
        m = train_step(model, batch, opt, args.ponder)
        if stepi % args.log_every == 0 or stepi == 1:
            dt = time.time() - t0
            print(f"{log_prefix}step {stepi:5d} | loss {m['loss']:.3f} ce {m['ce']:.3f} "
                  f"reward {m['reward']:.3f} depth {m['mean_depth']:.2f} "
                  f"| {dt:.0f}s")

    # ---- final evaluation ----
    print(f"\n{log_prefix}=== EVAL (autoregressive generation + exact verify) ===")
    results = {}
    for split in ("val_interp", "test_extrap"):
        if split not in by_split:
            continue
        ev = evaluate(model, by_split[split], max_n=args.eval_n)
        results[split] = ev
        print(f"{log_prefix}[{split}] n={ev['n']} acc={ev['acc']:.3f}")
        print(f"{log_prefix}    by family : " +
              " ".join(f"{f}={a:.2f}" for f, a in ev["fam_acc"].items()))
        print(f"{log_prefix}    by diff   : " +
              " ".join(f"d{d}={a:.2f}" for d, a in ev["diff_acc"].items()))
        print(f"{log_prefix}    depth/diff: " +
              " ".join(f"d{d}={z:.2f}" for d, z in ev["diff_depth"].items()))
    return model, results, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="rdr_dataset.jsonl")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--ponder", type=float, default=0.01)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--eval_n", type=int, default=400, help="cap eval rows per split")
    ap.add_argument("--seed", type=int, default=0)
    # model size (defaults sized up a bit from rdr.Cfg so n_recur >= d_max)
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--n_recur", type=int, default=6)
    ap.add_argument("--ablate", action="store_true",
                    help="run the recurrence/attnres/rlvr ablation grid instead")
    ap.add_argument("--out", default="rdr_results.json")
    args = ap.parse_args()

    base = dict(vocab=VOCAB, d_model=args.d_model, n_recur=args.n_recur)

    if not args.ablate:
        cfg = Cfg(**base)
        _, results, n_params = run(args, cfg)
        json.dump({"config": base, "params": n_params, "results": results},
                  open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")
        return

    # ---- ablation grid: the spec's section-6 comparison ----
    grid = []
    combos = [
        dict(use_recurrence=True,  use_attnres=True,  use_rlvr=True),   # full RDR
        dict(use_recurrence=True,  use_attnres=True,  use_rlvr=False),  # no reward
        dict(use_recurrence=True,  use_attnres=False, use_rlvr=True),   # no AttnRes
        dict(use_recurrence=False, use_attnres=False, use_rlvr=True),   # no loop (depth=1)
    ]
    for combo in combos:
        cfg = Cfg(**base, **combo)
        tag = (f"[rec{int(combo['use_recurrence'])}"
               f"_ar{int(combo['use_attnres'])}"
               f"_rl{int(combo['use_rlvr'])}] ")
        print("\n" + "=" * 70 + f"\n{tag}\n" + "=" * 70)
        _, results, n_params = run(args, cfg, log_prefix=tag)
        grid.append({"combo": combo, "params": n_params, "results": results})
    json.dump({"config": base, "grid": grid}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    print("\n=== ABLATION SUMMARY (acc: val_interp / test_extrap) ===")
    for g in grid:
        c = g["combo"]
        vi = g["results"].get("val_interp", {}).get("acc", 0)
        te = g["results"].get("test_extrap", {}).get("acc", 0)
        print(f"  rec={int(c['use_recurrence'])} ar={int(c['use_attnres'])} "
              f"rl={int(c['use_rlvr'])}  ->  interp {vi:.3f} / extrap {te:.3f}")


if __name__ == "__main__":
    main()
