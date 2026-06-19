#!/usr/bin/env python3
"""
build_dataset.py — bounded verifiable-task dataset for the RDR mechanism study.

Design (matches the spec + the "what to do" plan):
  * 3 DISCRETE families with EXACT ground truth (the verifier is exact, no tol).
  * In every family, DIFFICULTY == REASONING DEPTH (number of hops / nesting).
    This is load-bearing: the central hypothesis (router allocates more recursion
    depth to harder tasks) is only testable if difficulty == required depth.
  * Splits:
      - train          : difficulty 1..d_train, seen instances
      - val_interp     : difficulty 1..d_train, UNSEEN instances (interpolation)
      - test_extrap    : difficulty d_train+1..d_max (HARDER than anything in train)
                         -> systematic-generalisation test (the key scientific split)
  * Output: JSONL rows {family, difficulty, split, input, target}.
    `input`/`target` are plain strings -> byte/char-level tokenisation, no mismatch.
  * verify(family, target, output) is the exact reward signal used by the RDR trainer
    (this is the small-scale analogue of pattern_generator's executable verifier).

Scope: calibrated to a SMALL from-scratch model. Difficulty = hops, not RH-hard math.
Goal = test the mechanism, not solve frontier problems.
"""
import json, argparse, random
from collections import Counter

# ----------------------------------------------------------------------------- #
#  FAMILIES — each returns (input_str, target_str); the builder computes the
#  reference by actually executing the task (trusted ground truth).
# ----------------------------------------------------------------------------- #
def gen_modchain(rng, diff):
    """Iterated modular update, k=diff hops:  x <- (c*x + d) mod m, k times."""
    m = rng.choice([7, 11, 13, 17, 19, 23])
    a = rng.randint(0, m - 1); c = rng.randint(1, m - 1); d = rng.randint(0, m - 1)
    k = diff
    x = a
    for _ in range(k):
        x = (c * x + d) % m
    return f"MODCHAIN a={a} c={c} d={d} m={m} k={k} =", str(x)

def gen_ptrchase(rng, diff):
    """Pointer chasing, k=diff hops:  idx <- list[idx], k times."""
    n = rng.randint(6, 10)
    lst = [rng.randint(0, n - 1) for _ in range(n)]
    start = rng.randint(0, n - 1); k = diff
    idx = start
    for _ in range(k):
        idx = lst[idx]
    return f"PTRCHASE list={lst} start={start} k={k} =", str(idx)

def _build_expr(rng, depth):
    """Left-deep nested expression of EXACT nesting depth == depth. Returns (string, value)."""
    if depth == 0:
        v = rng.randint(1, 4)
        return str(v), v
    ls, lv = _build_expr(rng, depth - 1)
    rv = rng.randint(1, 4)
    op = rng.choice(["+", "-", "*"])
    val = {"+": lv + rv, "-": lv - rv, "*": lv * rv}[op]
    return f"({ls}{op}{rv})", val

def gen_eval(rng, diff):
    """Nested arithmetic expression, nesting depth = diff (recursion-depth test)."""
    s, v = _build_expr(rng, diff)
    return f"EVAL {s} =", str(v)

FAMILIES = {"modchain": gen_modchain, "ptrchase": gen_ptrchase, "eval": gen_eval}

# ----------------------------------------------------------------------------- #
#  EXACT verifier — the reward signal (1 iff model output matches ground truth).
# ----------------------------------------------------------------------------- #
def verify(family, target, output):
    return output.strip() == target.strip()

# ----------------------------------------------------------------------------- #
#  Builder
# ----------------------------------------------------------------------------- #
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
                    split = "test_extrap"                  # harder than train
                else:
                    split = "val_interp" if rng.random() < val_frac else "train"
                rows.append({"family": fam, "difficulty": diff,
                             "split": split, "input": inp, "target": tgt})
                made += 1
    rng.shuffle(rows)
    return rows

def summarize(rows):
    by_split = Counter(r["split"] for r in rows)
    print("split sizes:", dict(by_split))
    print("\nfamily x difficulty x split counts:")
    cells = Counter((r["family"], r["difficulty"], r["split"]) for r in rows)
    for fam in FAMILIES:
        diffs = sorted({d for (f, d, s) in cells if f == fam})
        print(f"  {fam}:")
        for d in diffs:
            parts = {s: cells[(fam, d, s)] for s in ("train", "val_interp", "test_extrap")
                     if cells[(fam, d, s)]}
            print(f"    diff={d}: {parts}")

def selftest():
    rng = random.Random(0)
    print("SELFTEST: generate one per family, verify correct passes & wrong fails.\n")
    for fam, gen in FAMILIES.items():
        inp, tgt = gen(rng, 3)
        print(f"  [{fam}] {inp} {tgt}")
        assert verify(fam, tgt, tgt) is True, "correct must pass"
        assert verify(fam, tgt, str(int(tgt) + 1 if tgt.lstrip('-').isdigit() else tgt + 'x')) is False
    print("\n  verifier OK (accepts truth, rejects wrong).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_cell", type=int, default=400, help="instances per family x difficulty")
    ap.add_argument("--d_train", type=int, default=4, help="max difficulty seen in train/val")
    ap.add_argument("--d_max", type=int, default=6, help="max difficulty (>d_train -> extrap test)")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rdr_dataset.jsonl")
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
    print("\nsample rows:")
    for r in rows[:6]:
        print(f"  ({r['split']:11s} d{r['difficulty']}) {r['input']} -> {r['target']}")

if __name__ == "__main__":
    main()
