#!/usr/bin/env python3
"""Memory budget calculator for finite-state / memory-native transformer training.

The goal is not to predict allocator-level peak exactly.  It is a transparent
accounting model that separates the four memory pools that matter:
  1) persistent weights/state,
  2) materialized gradients,
  3) optimizer state/master weights,
  4) saved activations.

Example:
  python tools/memory_budget_calculator.py --layers 24 --d-model 2048 --seq 1024 --batch 1 --state-bits 6
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict

GIB = 1024 ** 3


@dataclass
class TransformerShape:
    layers: int
    d_model: int
    vocab: int
    linear_param_factor: float
    tied_lm_head: bool

    @property
    def linear_params(self) -> int:
        return int(round(self.linear_param_factor * self.layers * self.d_model * self.d_model))

    @property
    def embedding_params(self) -> int:
        return self.vocab * self.d_model

    @property
    def total_dense_params(self) -> int:
        return self.linear_params + self.embedding_params * (1 if self.tied_lm_head else 2)

    @property
    def linear_row_count(self) -> int:
        # For the common 12 L d^2 GPT approximation: rows per layer ~= 12d.
        # This is only used for row-wise scale/RMS overhead, so exactness is not critical.
        return int(round(self.linear_param_factor * self.layers * self.d_model))


@dataclass
class MemoryBudget:
    name: str
    persistent_weights_gib: float
    gradients_gib: float
    optimizer_state_gib: float
    saved_activations_gib: float
    misc_gib: float

    @property
    def total_gib(self) -> float:
        return (self.persistent_weights_gib + self.gradients_gib + self.optimizer_state_gib +
                self.saved_activations_gib + self.misc_gib)

    def row(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "persistent_weights_GiB": round(self.persistent_weights_gib, 4),
            "gradients_GiB": round(self.gradients_gib, 4),
            "optimizer_state_GiB": round(self.optimizer_state_gib, 4),
            "saved_activations_GiB": round(self.saved_activations_gib, 4),
            "misc_GiB": round(self.misc_gib, 4),
            "total_GiB": round(self.total_gib, 4),
        }


def gib(nbytes: float) -> float:
    return nbytes / GIB


def activation_bytes(layers: int, batch: int, seq: int, d_model: int, bytes_per_elem: float,
                     multiplier: float, policy: str, anchor_every: int) -> float:
    token_state = batch * seq * d_model * bytes_per_elem * multiplier
    if policy == "store_all":
        return layers * token_state
    if policy == "selective_half":
        return 0.5 * layers * token_state
    if policy == "reversible":
        anchors = 1 + (layers // max(anchor_every, 1))
        return anchors * token_state
    if policy == "none_lower_bound":
        return 0.0
    raise ValueError(f"unknown activation policy {policy!r}")


def make_budgets(args: argparse.Namespace) -> tuple[TransformerShape, list[MemoryBudget]]:
    sh = TransformerShape(args.layers, args.d_model, args.vocab, args.linear_param_factor, args.tied_lm_head)
    n = sh.total_dense_params
    n_lin = sh.linear_params
    n_emb = sh.embedding_params * (1 if sh.tied_lm_head else 2)
    rows = sh.linear_row_count

    # Baseline: common mixed-precision AdamW accounting.
    # bf16 params + bf16 grads + fp32 master + fp32 m + fp32 v = 2+2+4+4+4 = 16 bytes/param.
    base_acts = activation_bytes(args.layers, args.batch, args.seq, args.d_model, args.act_bytes,
                                 args.activation_multiplier, args.activation_policy_baseline,
                                 args.anchor_every)
    baseline = MemoryBudget(
        name="BF16 AdamW-style",
        persistent_weights_gib=gib(2 * n),
        gradients_gib=gib(2 * n),
        optimizer_state_gib=gib(12 * n),
        saved_activations_gib=gib(base_acts),
        misc_gib=args.misc_gib,
    )

    # Counter state for linear weights; embeddings kept at configurable precision.
    counter_state = (args.state_bits / 8.0) * n_lin
    row_state = args.row_state_bytes * rows
    emb = args.embedding_bits / 8.0 * n_emb
    counter_acts = activation_bytes(args.layers, args.batch, args.seq, args.d_model, args.counter_act_bits / 8.0,
                                    args.activation_multiplier, args.activation_policy_counter,
                                    args.anchor_every)
    counter = MemoryBudget(
        name=f"finite-state {args.state_bits:g}b + {args.counter_act_bits:g}b acts",
        persistent_weights_gib=gib(counter_state + row_state + emb),
        gradients_gib=0.0 if args.fused_gradients else gib(args.grad_bytes * n_lin),
        optimizer_state_gib=0.0,
        saved_activations_gib=gib(counter_acts),
        misc_gib=args.misc_gib,
    )

    # Lower-bound-ish: visible ternary entropy plus activation policy, no residual optimizer.
    visible_ternary_bits = 1.5849625007211563
    lower_counter = MemoryBudget(
        name="visible ternary entropy lower-bound",
        persistent_weights_gib=gib((visible_ternary_bits / 8.0) * n_lin + emb),
        gradients_gib=0.0,
        optimizer_state_gib=0.0,
        saved_activations_gib=gib(counter_acts),
        misc_gib=args.misc_gib,
    )
    return sh, [baseline, counter, lower_counter]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--d-model", type=int, default=2048)
    ap.add_argument("--vocab", type=int, default=50257)
    ap.add_argument("--linear-param-factor", type=float, default=12.0,
                    help="linear params ~= factor * L * d^2; GPT-style default is 12")
    ap.add_argument("--tied-lm-head", action="store_true", default=True)
    ap.add_argument("--untied-lm-head", dest="tied_lm_head", action="store_false")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--act-bytes", type=float, default=2.0, help="baseline activation bytes/element")
    ap.add_argument("--activation-multiplier", type=float, default=6.0,
                    help="rough saved tensors per token per layer")
    ap.add_argument("--activation-policy-baseline", choices=["store_all", "selective_half", "reversible", "none_lower_bound"],
                    default="store_all")
    ap.add_argument("--activation-policy-counter", choices=["store_all", "selective_half", "reversible", "none_lower_bound"],
                    default="reversible")
    ap.add_argument("--anchor-every", type=int, default=8)
    ap.add_argument("--state-bits", type=float, default=6.0)
    ap.add_argument("--embedding-bits", type=float, default=16.0)
    ap.add_argument("--counter-act-bits", type=float, default=4.0)
    ap.add_argument("--row-state-bytes", type=float, default=8.0,
                    help="per output row scale+RMS; 8=fp32+fp32, 4=fp16/log16+fp16/log16")
    ap.add_argument("--fused-gradients", action="store_true", default=True)
    ap.add_argument("--unfused-gradients", dest="fused_gradients", action="store_false")
    ap.add_argument("--grad-bytes", type=float, default=2.0)
    ap.add_argument("--misc-gib", type=float, default=1.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    sh, budgets = make_budgets(args)
    payload = {
        "shape": asdict(sh),
        "linear_params": sh.linear_params,
        "embedding_params_counted": sh.embedding_params * (1 if sh.tied_lm_head else 2),
        "linear_row_count_estimate": sh.linear_row_count,
        "budgets": [b.row() for b in budgets],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"shape: L={sh.layers}, d={sh.d_model}, vocab={sh.vocab}, dense params~{sh.total_dense_params:,}")
        print(f"linear params~{sh.linear_params:,}, counted embedding params~{payload['embedding_params_counted']:,}")
        print()
        headers = ["name", "persistent", "grad", "optim", "acts", "misc", "total"]
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join(["---"] * len(headers)) + "|")
        for b in budgets:
            r = b.row()
            print(f"| {r['name']} | {r['persistent_weights_GiB']:.3f} | {r['gradients_GiB']:.3f} | "
                  f"{r['optimizer_state_GiB']:.3f} | {r['saved_activations_GiB']:.3f} | "
                  f"{r['misc_GiB']:.3f} | {r['total_GiB']:.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
