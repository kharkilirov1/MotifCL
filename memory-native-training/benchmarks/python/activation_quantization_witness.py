#!/usr/bin/env python3
"""Witness for saving activation memory in finite-state training.

A counter layer update needs x and grad_out to form grad_w = grad_out^T x.  If a
reversible block is not yet available, we can store a stochastic low-bit version
Q(x) instead of the full activation.  Since E[Q(x)|x]=x, the weight-gradient used
by the finite-state update remains unbiased:

    E[grad_out^T Q(x) | x, grad_out] = grad_out^T x.

This script measures the price of that variance on a small ternary teacher task.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def encode_state(t: torch.Tensor, c: torch.Tensor, C: int) -> torch.Tensor:
    lv = 2 * C - 1
    return ((t.to(torch.int16) + 1) * lv + (c.to(torch.int16) + (C - 1))).to(torch.uint8)


def decode_state(state: torch.Tensor, C: int) -> tuple[torch.Tensor, torch.Tensor]:
    lv = 2 * C - 1
    z = state.to(torch.int16)
    return torch.div(z, lv, rounding_mode="floor") - 1, torch.remainder(z, lv) - (C - 1)


def stochastic_round(x: torch.Tensor) -> torch.Tensor:
    fl = torch.floor(x)
    return fl + (torch.rand_like(x) < (x - fl)).to(x.dtype)


def quantize_per_token_symmetric(x: torch.Tensor, bits: int | None) -> torch.Tensor:
    if bits is None or bits >= 16:
        return x
    if bits < 2:
        raise ValueError("activation bits must be >=2 or fp16/fp32 sentinel")
    qmax = (1 << (bits - 1)) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    q = stochastic_round(x / scale).clamp(-qmax, qmax)
    return q * scale


@torch.no_grad()
def counter_update(state: torch.Tensor, scale: float, grad_w: torch.Tensor, *, C: int, lr: float) -> None:
    t, c = decode_state(state, C)
    val = c.float() - lr * grad_w * (C / scale)
    ccv = stochastic_round(val)
    carry = torch.trunc(ccv / C)
    rem = ccv - carry * C
    nt = t.float() + carry
    clipped = nt.clamp(-1, 1)
    saturated = clipped != nt
    rem = torch.where(saturated, torch.sign(ccv).clamp(-1, 1) * (C - 1), rem)
    ri = rem.round().to(torch.int16).clamp(-(C - 1), C - 1)
    state.copy_(encode_state(clipped.to(torch.int16), ri, C))


def run(bits: int | None, seed: int, *, n: int, samples: int, steps: int, C: int, lr: float) -> dict:
    torch.manual_seed(seed)
    teacher = torch.randint(-1, 2, (n, n), dtype=torch.int16)
    scale = 0.25
    x = torch.randn(samples, n)
    y = x @ (scale * teacher.float()).t()
    t0 = torch.randint(-1, 2, (n, n), dtype=torch.int16)
    c0 = torch.zeros_like(t0)
    state = encode_state(t0, c0, C)
    hit = None
    for step in range(steps):
        t, _ = decode_state(state, C)
        pred = x @ (scale * t.float()).t()
        grad_out = (2.0 / pred.numel()) * (pred - y)
        x_saved = quantize_per_token_symmetric(x, bits)
        grad_w = grad_out.t() @ x_saved
        counter_update(state, scale, grad_w, C=C, lr=lr)
        t, _ = decode_state(state, C)
        acc = (t == teacher).float().mean().item()
        if hit is None and acc == 1.0:
            hit = step + 1
            break
    t, _ = decode_state(state, C)
    pred = x @ (scale * t.float()).t()
    mse = ((pred - y) ** 2).mean().item()
    acc = (t == teacher).float().mean().item()
    label = "fp" if bits is None or bits >= 16 else f"int{bits}"
    # For long hidden sizes, per-token scale overhead is tiny.  For this witness n=small, include it explicitly.
    if bits is None or bits >= 16:
        activation_bits_per_element = 16.0
    else:
        activation_bits_per_element = float(bits) + 16.0 / n
    return {
        "activation": label,
        "seed": seed,
        "bits": bits if bits is not None else 16,
        "effective_activation_bits_per_element_with_fp16_token_scale": activation_bits_per_element,
        "hit_step": hit,
        "final_acc": acc,
        "final_mse": mse,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=str, default="fp,8,4,3")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--C", type=int, default=11)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    torch.set_num_threads(2)
    bit_list: list[int | None] = []
    for item in args.bits.split(","):
        item = item.strip().lower()
        bit_list.append(None if item in {"fp", "fp16", "float"} else int(item))

    rows = []
    for bits in bit_list:
        trials = [run(bits, seed, n=args.n, samples=args.samples, steps=args.steps, C=args.C, lr=args.lr)
                  for seed in range(args.seeds)]
        hits = [t["hit_step"] if t["hit_step"] is not None else 999999 for t in trials]
        rows.append({
            "activation": trials[0]["activation"],
            "effective_bits_per_element": trials[0]["effective_activation_bits_per_element_with_fp16_token_scale"],
            "median_hit_step": sorted(hits)[len(hits) // 2],
            "mean_final_acc": sum(t["final_acc"] for t in trials) / len(trials),
            "mean_final_mse": sum(t["final_mse"] for t in trials) / len(trials),
            "trials": trials,
        })

    print("| saved activation | effective bits/elem | median hit | mean acc | mean MSE |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        hit = "miss" if r["median_hit_step"] >= 999999 else str(int(r["median_hit_step"]))
        print(f"| {r['activation']} | {r['effective_bits_per_element']:.2f} | {hit} | "
              f"{r['mean_final_acc']:.4f} | {r['mean_final_mse']:.6g} |")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
