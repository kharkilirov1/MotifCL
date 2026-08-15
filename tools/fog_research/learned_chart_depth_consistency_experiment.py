#!/usr/bin/env python3
"""EXP-011: stabilize a learned cyclic chart with multi-depth consistency.

Builds on EXP-010.  Identity phases are learned from the successor law, but the
same +1 transition is required to be consistent after terminal depths 1, 2 and
3.  No arbitrary binary (a,b) addition pairs are exposed during training.

The hypothesis is that multi-depth consistency constrains tiny geometric errors
that are harmless for one-step top-1 accuracy but catastrophic under recurrent
feedback.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from learned_cyclic_chart_experiment import (
    LearnedChartConfig,
    LearnedPhaseChart,
    evaluate_chart,
)

EXPERIMENT_NAME = "exp_011_learned_chart_depth_consistency"


def train_consistent(
    cfg: LearnedChartConfig,
    *,
    seed: int,
    depths: tuple[int, ...],
    steps: int,
    lr: float,
) -> tuple[LearnedPhaseChart, list[dict]]:
    torch.manual_seed(seed)
    model = LearnedPhaseChart(cfg, seed)
    optimizer = torch.optim.Adam([model.phase], lr=lr)
    source = torch.arange(cfg.modulus)
    trace = []
    for step in range(steps):
        code = model.codebook()
        one = code[torch.ones_like(source)]
        losses = []
        for depth in depths:
            state = code[source]
            for _ in range(depth):
                state = model.multiply_codes(state, one)
            target = code[(source + depth) % cfg.modulus]
            losses.append(
                (1.0 - F.cosine_similarity(state, target, dim=-1)).mean()
            )
        consistency_loss = torch.stack(losses).mean()
        separation_loss = F.cross_entropy(
            cfg.head_scale * code @ code.T, torch.arange(cfg.modulus)
        )
        loss = consistency_loss + cfg.separation_weight * separation_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 10) == 0 or step + 1 == steps:
            trace.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "consistency_loss": float(consistency_loss.detach()),
                    "separation_loss": float(separation_loss.detach()),
                }
            )
    return model.eval(), trace


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/research/exp_011/metrics.json"))
    p.add_argument("--train-depths", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--eval-examples", type=int, default=512)
    p.add_argument("--eval-depths", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    cfg = LearnedChartConfig()
    rows = []
    for seed in args.seeds:
        model, trace = train_consistent(
            cfg,
            seed=seed,
            depths=tuple(args.train_depths),
            steps=args.steps,
            lr=args.lr,
        )
        rows.append(
            {
                "seed": seed,
                "trace": trace,
                "evaluation": evaluate_chart(
                    model,
                    arm="closed_cycle",
                    sequence_examples=args.eval_examples,
                    sequence_depths=tuple(args.eval_depths),
                    seed=seed,
                ),
            }
        )
    payload = {
        "experiment": EXPERIMENT_NAME,
        "config": asdict(cfg),
        "train_depths": args.train_depths,
        "training_examples_are_successor_sequences_only": True,
        "arbitrary_binary_pairs_seen_in_training": False,
        "rows": rows,
    }
    write_json(args.output, payload)
    print(json.dumps({
        "runs": len(rows),
        "heldout_binary_accuracy": sum(r["evaluation"]["heldout_binary_pair_accuracy_b_ne_1"] for r in rows) / len(rows),
        "depth64_accuracy": sum(next(x for x in r["evaluation"]["recurrence"] if x["depth"] == 64)["final_accuracy"] for r in rows) / len(rows),
        "minimum_depth64_accuracy": min(next(x for x in r["evaluation"]["recurrence"] if x["depth"] == 64)["final_accuracy"] for r in rows),
        "root_residual_mean": sum(r["evaluation"]["mean_pth_root_residual"] for r in rows) / len(rows),
    }, indent=2))


if __name__ == "__main__":
    main()
