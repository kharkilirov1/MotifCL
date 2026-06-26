#!/usr/bin/env python3
"""Ablate finite-state synapse bit budgets on a discrete teacher-recovery task.

This is intentionally small and deterministic enough to run on CPU.  It answers a
very specific question: how much of the finite-state residual budget can we remove
before learning a ternary matrix becomes unreliable?

The state family is t in {-1,0,+1}, c in {-(C-1),...,C-1}; total states = 3(2C-1).
Logical bits = ceil(log2(states)).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "python"))
from counter_state_fused_v2 import CompactCounterLinear, decode_state  # noqa: E402


def entropy_bits(x: torch.Tensor, max_states: int) -> float:
    h = torch.bincount(x.flatten().to(torch.int64).cpu(), minlength=max_states).float()
    p = h / h.sum().clamp_min(1)
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum().item())


def run_one(C: int, mode: str, seed: int, *, n: int, samples: int, steps: int, lr: float) -> dict:
    torch.manual_seed(seed)
    teacher = torch.randint(-1, 2, (n, n), dtype=torch.int16)
    scale = 0.25
    x = torch.randn(samples, n, requires_grad=True)
    y = x.detach() @ (scale * teacher.float()).t()
    layer = CompactCounterLinear(n, n, C=C, lr=lr, lr_scale=0.0, tile_rows=min(32, n), pulse_mode=mode)
    layer.scale.fill_(scale)
    hit = None
    for step in range(steps):
        x.grad = None
        pred = layer(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            t, _ = decode_state(layer.state, C)
            acc = (t == teacher).float().mean().item()
        if hit is None and acc == 1.0:
            hit = step + 1
            # Keep a few extra updates out of the result?  No, exact recovery has happened.
            break
    with torch.no_grad():
        layer.update_enabled = False
        pred = layer(x.detach())
        final_loss = ((pred - y) ** 2).mean().item()
        t, _ = decode_state(layer.state, C)
        final_acc = (t == teacher).float().mean().item()
        h = entropy_bits(layer.state, 3 * (2 * C - 1))
    states = 3 * (2 * C - 1)
    return {
        "C": C,
        "states": states,
        "logical_bits": math.ceil(math.log2(states)),
        "mode": mode,
        "seed": seed,
        "hit_step": hit,
        "final_acc": final_acc,
        "final_mse": final_loss,
        "state_entropy_bits": h,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--C-list", type=str, default="3,5,8,11",
                    help="Comma list. C=3 -> 15 states/4 bits, C=5 -> 27/5 bits, C=11 -> 63/6 bits")
    ap.add_argument("--modes", type=str, default="direct,ternary")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    torch.set_num_threads(2)

    Cs = [int(x) for x in args.C_list.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    rows: list[dict] = []
    for C in Cs:
        for mode in modes:
            trials = [run_one(C, mode, seed, n=args.n, samples=args.samples, steps=args.steps, lr=args.lr)
                      for seed in range(args.seeds)]
            hit_steps = [r["hit_step"] if r["hit_step"] is not None else 999999 for r in trials]
            row = {
                "C": C,
                "states": trials[0]["states"],
                "logical_bits": trials[0]["logical_bits"],
                "mode": mode,
                "median_hit_step": statistics.median(hit_steps),
                "mean_final_acc": statistics.mean(r["final_acc"] for r in trials),
                "mean_final_mse": statistics.mean(r["final_mse"] for r in trials),
                "mean_state_entropy_bits": statistics.mean(r["state_entropy_bits"] for r in trials),
                "trials": trials,
            }
            rows.append(row)

    print("| C | states | logical bits | mode | median hit | mean acc | mean MSE | state entropy |")
    print("|---:|---:|---:|---|---:|---:|---:|---:|")
    for r in rows:
        hit = r["median_hit_step"]
        hit_s = "miss" if hit >= 999999 else str(int(hit))
        print(f"| {r['C']} | {r['states']} | {r['logical_bits']} | {r['mode']} | {hit_s} | "
              f"{r['mean_final_acc']:.4f} | {r['mean_final_mse']:.6g} | {r['mean_state_entropy_bits']:.3f} |")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
