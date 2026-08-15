#!/usr/bin/env python3
"""EXP-012: jointly learn latent chart and local bilinear operator.

EXP-010/011 learned the chart while complex multiplication was fixed.  Here the
per-plane bilinear operator itself is also trainable.

Training labels still contain only repeated successor programs (+1) at depths
1,2,3.  No arbitrary (a,b)->a+b target pairs are shown.

Two arms:
- successor_only: only generator-depth consistency + code separation;
- algebraic: additionally unlabeled identity, commutativity and associativity
  consistency on random states.

The experiment asks whether algebraic self-consistency can turn a +1 specialist
into a general binary operation without direct binary-pair supervision.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

EXPERIMENT_NAME = "exp_012_joint_chart_operator"
Arm = Literal["successor_only", "algebraic"]


@dataclass(frozen=True)
class JointConfig:
    modulus: int = 31
    harmonics: int = 4
    separation_weight: float = 0.05
    algebraic_weight: float = 0.5
    head_scale: float = 20.0

    @property
    def d_model(self) -> int:
        return 2 * self.harmonics


class JointLatentAlgebra(nn.Module):
    def __init__(self, cfg: JointConfig, seed: int):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        self.phase = nn.Parameter(
            torch.rand(cfg.modulus, cfg.harmonics, generator=g) * 2 * math.pi - math.pi
        )
        self.operator = nn.Parameter(
            torch.randn(cfg.harmonics, 4, 2, generator=g) * 0.3
        )

    def codebook(self) -> torch.Tensor:
        phase = self.phase - self.phase[:1]
        return F.normalize(
            torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1), dim=-1
        )

    def op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h = self.cfg.harmonics
        ca, sa = a[..., :h], a[..., h:]
        cb, sb = b[..., :h], b[..., h:]
        feat = torch.stack((ca * cb, ca * sb, sa * cb, sa * sb), dim=-1)
        out = torch.einsum("...hf,hfo->...ho", feat, self.operator)
        return F.normalize(torch.cat((out[..., 0], out[..., 1]), dim=-1), dim=-1)

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.cfg.head_scale * F.normalize(z, dim=-1) @ self.codebook().T


def train_one(
    cfg: JointConfig,
    *,
    arm: Arm,
    seed: int,
    steps: int,
    lr: float,
    algebraic_batch: int,
) -> tuple[JointLatentAlgebra, list[dict]]:
    torch.manual_seed(seed)
    model = JointLatentAlgebra(cfg, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ids = torch.arange(cfg.modulus)
    generator = torch.Generator().manual_seed(seed + 991)
    trace = []

    for step in range(steps):
        code = model.codebook()
        one = code[torch.ones_like(ids)]
        depth_losses = []
        for depth in (1, 2, 3):
            state = code[ids]
            for _ in range(depth):
                state = model.op(state, one)
            target = code[(ids + depth) % cfg.modulus]
            depth_losses.append(
                (1.0 - F.cosine_similarity(state, target, dim=-1)).mean()
            )
        successor_loss = torch.stack(depth_losses).mean()
        separation_loss = F.cross_entropy(
            cfg.head_scale * code @ code.T, ids
        )
        loss = successor_loss + cfg.separation_weight * separation_loss
        identity_loss = torch.tensor(0.0)
        comm_loss = torch.tensor(0.0)
        assoc_loss = torch.tensor(0.0)

        if arm == "algebraic":
            zero = code[torch.zeros_like(ids)]
            identity_loss = (
                2.0
                - F.cosine_similarity(model.op(code, zero), code, dim=-1)
                - F.cosine_similarity(model.op(zero, code), code, dim=-1)
            ).mean()
            a = torch.randint(cfg.modulus, (algebraic_batch,), generator=generator)
            b = torch.randint(cfg.modulus, (algebraic_batch,), generator=generator)
            c = torch.randint(cfg.modulus, (algebraic_batch,), generator=generator)
            ab = model.op(code[a], code[b])
            ba = model.op(code[b], code[a])
            comm_loss = (
                1.0 - F.cosine_similarity(ab, ba, dim=-1)
            ).mean()
            left = model.op(ab, code[c])
            right = model.op(code[a], model.op(code[b], code[c]))
            assoc_loss = (
                1.0 - F.cosine_similarity(left, right, dim=-1)
            ).mean()
            loss = loss + cfg.algebraic_weight * (
                identity_loss + comm_loss + assoc_loss
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 0 or (step + 1) % max(1, steps // 10) == 0 or step + 1 == steps:
            trace.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "successor_loss": float(successor_loss.detach()),
                    "separation_loss": float(separation_loss.detach()),
                    "identity_loss": float(identity_loss.detach()),
                    "commutativity_loss": float(comm_loss.detach()),
                    "associativity_loss": float(assoc_loss.detach()),
                }
            )
    return model.eval(), trace


@torch.inference_mode()
def evaluate(
    model: JointLatentAlgebra,
    *,
    seed: int,
    recurrent_depth: int,
    recurrent_examples: int,
) -> dict:
    cfg = model.cfg
    code = model.codebook()
    ids = torch.arange(cfg.modulus)
    one = code[torch.ones_like(ids)]
    successor = model.logits(model.op(code, one)).argmax(-1)
    successor_acc = float(successor.eq((ids + 1) % cfg.modulus).float().mean())

    a = ids.repeat_interleave(cfg.modulus)
    b = ids.repeat(cfg.modulus)
    target = (a + b) % cfg.modulus
    pair_pred = model.logits(model.op(code[a], code[b])).argmax(-1)
    pair_acc = float(pair_pred.eq(target).float().mean())

    g = torch.Generator().manual_seed(seed + 50000)
    start = torch.randint(cfg.modulus, (recurrent_examples,), generator=g)
    operands = torch.randint(
        cfg.modulus, (recurrent_examples, recurrent_depth), generator=g
    )
    state = code[start]
    oracle = start.clone()
    hops = []
    for t in range(recurrent_depth):
        state = model.op(state, code[operands[:, t]])
        oracle = (oracle + operands[:, t]) % cfg.modulus
        hops.append(
            float(model.logits(state).argmax(-1).eq(oracle).float().mean())
        )

    # Algebraic diagnostics on the discrete codebook (no target binary labels
    # are needed for these consistency errors).
    gen = torch.Generator().manual_seed(seed + 70000)
    aa = torch.randint(cfg.modulus, (1024,), generator=gen)
    bb = torch.randint(cfg.modulus, (1024,), generator=gen)
    cc = torch.randint(cfg.modulus, (1024,), generator=gen)
    ab = model.op(code[aa], code[bb])
    ba = model.op(code[bb], code[aa])
    comm = float((1 - F.cosine_similarity(ab, ba, dim=-1)).mean())
    assoc = float(
        (
            1
            - F.cosine_similarity(
                model.op(ab, code[cc]),
                model.op(code[aa], model.op(code[bb], code[cc])),
                dim=-1,
            )
        ).mean()
    )
    return {
        "successor_accuracy": successor_acc,
        "all_binary_addition_accuracy": pair_acc,
        "recurrent_depth": recurrent_depth,
        "recurrent_final_accuracy": hops[-1],
        "recurrent_minimum_hop_accuracy": min(hops),
        "commutativity_error": comm,
        "associativity_error": assoc,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/research/exp_012/metrics.json"))
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--algebraic-batch", type=int, default=256)
    p.add_argument("--recurrent-depth", type=int, default=16)
    p.add_argument("--recurrent-examples", type=int, default=512)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    cfg = JointConfig()
    rows = []
    for arm in ("successor_only", "algebraic"):
        for seed in args.seeds:
            model, trace = train_one(
                cfg,
                arm=arm,  # type: ignore[arg-type]
                seed=seed,
                steps=args.steps,
                lr=args.lr,
                algebraic_batch=args.algebraic_batch,
            )
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "trace": trace,
                    "evaluation": evaluate(
                        model,
                        seed=seed,
                        recurrent_depth=args.recurrent_depth,
                        recurrent_examples=args.recurrent_examples,
                    ),
                }
            )
    payload = {
        "experiment": EXPERIMENT_NAME,
        "config": asdict(cfg),
        "binary_pair_targets_seen_in_training": False,
        "successor_depths_seen": [1, 2, 3],
        "rows": rows,
    }
    write_json(args.output, payload)
    summary = {}
    for arm in ("successor_only", "algebraic"):
        rr = [r["evaluation"] for r in rows if r["arm"] == arm]
        summary[arm] = {
            "successor_accuracy_mean": sum(x["successor_accuracy"] for x in rr) / len(rr),
            "binary_accuracy_by_seed": [x["all_binary_addition_accuracy"] for x in rr],
            "recurrent_final_by_seed": [x["recurrent_final_accuracy"] for x in rr],
            "commutativity_error_by_seed": [x["commutativity_error"] for x in rr],
            "associativity_error_by_seed": [x["associativity_error"] for x in rr],
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
