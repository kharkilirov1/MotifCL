#!/usr/bin/env python3
"""
build_traj.py — FORCED-UNROLLING variant of the verifiable-task dataset.

Motivation (see RESULTS.md): with final-value-only targets the model can solve
every difficulty with a single parallel shortcut, so recurrence depth never has
to scale with task hops. Here the target is the FULL TRAJECTORY of intermediate
states — one state per hop — so the iterative computation is on the critical path
and cannot be skipped: to emit state t you must have produced state t-1 and
applied exactly one more step.

Same families, same splits, same JSONL schema {family,difficulty,split,input,
target} as build_dataset.py, and the SAME exact verifier (string match) — only the
target changes from "final value" to "space-separated trajectory".

Difficulty == number of hops == length of the trajectory == number of sequential
steps required. This is the load-bearing change.
"""
import json, argparse, random
from collections import Counter
from build_dataset import verify   # exact string-match verifier (family-agnostic)


def gen_modchain(rng, diff):
    """x <- (c*x + d) mod m, k=diff times. Target = the k successive states."""
    m = rng.choice([7, 11, 13, 17, 19, 23])
    a = rng.randint(0, m - 1); c = rng.randint(1, m - 1); d = rng.randint(0, m - 1)
    k = diff
    x, traj = a, []
    for _ in range(k):
        x = (c * x + d) % m
        traj.append(x)
    return f"MODCHAIN a={a} c={c} d={d} m={m} k={k} =", " ".join(map(str, traj))

def gen_ptrchase(rng, diff):
    """idx <- list[idx], k=diff times. Target = the k successive indices visited."""
    n = rng.randint(6, 10)
    lst = [rng.randint(0, n - 1) for _ in range(n)]
    start = rng.randint(0, n - 1); k = diff
    idx, traj = start, []
    for _ in range(k):
        idx = lst[idx]
        traj.append(idx)
    return f"PTRCHASE list={lst} start={start} k={k} =", " ".join(map(str, traj))

def _build_expr_traj(rng, depth):
    """Left-deep nested expression of EXACT nesting depth == depth.
    Returns (string, [partial values from innermost literal outward])."""
    if depth == 0:
        v = rng.randint(1, 4)
        return str(v), [v]
    ls, vals = _build_expr_traj(rng, depth - 1)
    lv = vals[-1]
    rv = rng.randint(1, 4)
    op = rng.choice(["+", "-", "*"])
    val = {"+": lv + rv, "-": lv - rv, "*": lv * rv}[op]
    return f"({ls}{op}{rv})", vals + [val]

def gen_eval(rng, diff):
    """Nested arithmetic. Target = the innermost-to-outermost partial values."""
    s, vals = _build_expr_traj(rng, diff)
    return f"EVAL {s} =", " ".join(map(str, vals))

FAMILIES = {"modchain": gen_modchain, "ptrchase": gen_ptrchase, "eval": gen_eval}


def build(n_per_cell, d_train, d_max, val_frac, seed):
    rng = random.Random(seed)
    rows, seen = [], set()
    for fam, gen in FAMILIES.items():
        for diff in range(1, d_max + 1):
            made, tries = 0, 0
            while made < n_per_cell and tries < n_per_cell * 50:
                tries += 1
                inp, tgt = gen(rng, diff)
                if inp in seen:
                    continue
                seen.add(inp)
                if diff > d_train:
                    split = "test_extrap"
                else:
                    split = "val_interp" if rng.random() < val_frac else "train"
                rows.append({"family": fam, "difficulty": diff,
                             "split": split, "input": inp, "target": tgt})
                made += 1
    rng.shuffle(rows)
    return rows

def summarize(rows):
    print("split sizes:", dict(Counter(r["split"] for r in rows)))
    maxlen = max(len(r["target"]) for r in rows)
    print(f"max target length (chars): {maxlen}")
    print("\nsample rows:")
    for r in rows[:8]:
        print(f"  ({r['split']:11s} {r['family']:9s} d{r['difficulty']}) "
              f"{r['input']} -> {r['target']}")

def selftest():
    rng = random.Random(0)
    print("SELFTEST: trajectory targets + exact verifier.\n")
    for fam, gen in FAMILIES.items():
        inp, tgt = gen(rng, 3)
        print(f"  [{fam}] {inp} -> {tgt}")
        assert verify(fam, tgt, tgt) is True
        assert verify(fam, tgt, tgt + " 0") is False
    print("\n  verifier OK (accepts exact trajectory, rejects wrong).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_cell", type=int, default=400)
    ap.add_argument("--d_train", type=int, default=4)
    ap.add_argument("--d_max", type=int, default=6)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rdr_traj.jsonl")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    rows = build(args.n_per_cell, args.d_train, args.d_max, args.val_frac, args.seed)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}\n")
    summarize(rows)

if __name__ == "__main__":
    main()
