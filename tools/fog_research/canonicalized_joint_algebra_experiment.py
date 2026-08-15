#!/usr/bin/env python3
"""EXP-013: joint latent algebra with in-transition soft canonicalization.

EXP-012 found an off-manifold failure mode: a jointly learned chart/operator can
be nearly associative and commutative in continuous space while arbitrary
operator outputs live far from every canonical identity code.  Penalizing
closure after the fact did not repair the semantics.

This experiment changes the transition itself:

    raw bilinear operator -> soft codebook attractor -> next latent register

The attractor is fully continuous (no argmax, token decode, or hard snap).  Its
scale is fixed before evaluation.  Training still sees only repeated successor
(+1) targets at depths 1,2,3 plus unlabeled identity/commutativity/associativity
constraints.  No arbitrary (a,b)->a+b targets are used for optimization.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

EXPERIMENT_NAME = "exp_013_canonicalized_joint_algebra"


@dataclass(frozen=True)
class Config:
    modulus: int = 31
    harmonics: int = 4
    separation_weight: float = 0.05
    algebraic_weight: float = 0.5
    head_scale: float = 20.0
    canonicalizer_scale: float = 12.0

    @property
    def d_model(self) -> int:
        return 2 * self.harmonics


class CanonicalizedLatentAlgebra(nn.Module):
    def __init__(self, cfg: Config, seed: int):
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

    def raw_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h = self.cfg.harmonics
        ca, sa = a[..., :h], a[..., h:]
        cb, sb = b[..., :h], b[..., h:]
        feat = torch.stack((ca * cb, ca * sb, sa * cb, sa * sb), dim=-1)
        out = torch.einsum("...hf,hfo->...ho", feat, self.operator)
        return F.normalize(torch.cat((out[..., 0], out[..., 1]), dim=-1), dim=-1)

    def canonicalize(
        self, z: torch.Tensor, *, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        code = self.codebook()
        similarity = F.normalize(z, dim=-1) @ code.T
        weight = torch.softmax(self.cfg.canonicalizer_scale * similarity, dim=-1)
        projected = F.normalize(weight @ code, dim=-1)
        if return_weights:
            return projected, weight
        return projected

    def op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.canonicalize(self.raw_op(a, b))  # type: ignore[return-value]

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.cfg.head_scale * F.normalize(z, dim=-1) @ self.codebook().T


def train_one(
    cfg: Config,
    *,
    seed: int,
    steps: int,
    lr: float,
    algebraic_batch: int,
) -> tuple[CanonicalizedLatentAlgebra, list[dict]]:
    torch.manual_seed(seed)
    model = CanonicalizedLatentAlgebra(cfg, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ids = torch.arange(cfg.modulus)
    generator = torch.Generator().manual_seed(seed + 991)
    trace: list[dict] = []

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
        separation_loss = F.cross_entropy(cfg.head_scale * code @ code.T, ids)

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
        comm_loss = (1.0 - F.cosine_similarity(ab, ba, dim=-1)).mean()
        left = model.op(ab, code[c])
        right = model.op(code[a], model.op(code[b], code[c]))
        assoc_loss = (1.0 - F.cosine_similarity(left, right, dim=-1)).mean()

        loss = (
            successor_loss
            + cfg.separation_weight * separation_loss
            + cfg.algebraic_weight * (identity_loss + comm_loss + assoc_loss)
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
    model: CanonicalizedLatentAlgebra,
    *,
    seed: int,
    recurrent_depth: int,
    recurrent_examples: int,
) -> dict:
    cfg = model.cfg
    code = model.codebook()
    ids = torch.arange(cfg.modulus)
    one = code[torch.ones_like(ids)]
    successor_acc = float(
        model.logits(model.op(code, one))
        .argmax(-1)
        .eq((ids + 1) % cfg.modulus)
        .float()
        .mean()
    )

    a = ids.repeat_interleave(cfg.modulus)
    b = ids.repeat(cfg.modulus)
    target = (a + b) % cfg.modulus
    raw = model.raw_op(code[a], code[b])
    canonical, weight = model.canonicalize(raw, return_weights=True)
    pair_acc = float(model.logits(canonical).argmax(-1).eq(target).float().mean())
    raw_nearest = (raw @ code.T).max(-1).values
    canonical_nearest = (canonical @ code.T).max(-1).values
    true_cos = F.cosine_similarity(canonical, code[target], dim=-1)
    entropy = -(weight * weight.clamp_min(1e-9).log()).sum(-1)

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
        hops.append(float(model.logits(state).argmax(-1).eq(oracle).float().mean()))

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
        "raw_nearest_code_cosine_mean": float(raw_nearest.mean()),
        "canonical_nearest_code_cosine_mean": float(canonical_nearest.mean()),
        "canonical_true_target_cosine_mean": float(true_cos.mean()),
        "canonicalizer_top_mass_mean": float(weight.max(-1).values.mean()),
        "canonicalizer_entropy_mean": float(entropy.mean()),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/research/exp_013/metrics.json"))
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12])
    p.add_argument("--algebraic-batch", type=int, default=256)
    p.add_argument("--recurrent-depth", type=int, default=32)
    p.add_argument("--recurrent-examples", type=int, default=512)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    cfg = Config()
    rows = []
    for seed in args.seeds:
        model, trace = train_one(
            cfg,
            seed=seed,
            steps=args.steps,
            lr=args.lr,
            algebraic_batch=args.algebraic_batch,
        )
        rows.append(
            {
                "seed": seed,
                "trace": trace,
                "metrics": evaluate(
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
        "training": {
            "steps": args.steps,
            "lr": args.lr,
            "algebraic_batch": args.algebraic_batch,
            "supervised_terminal_depths": [1, 2, 3],
            "arbitrary_binary_pair_targets_used": False,
        },
        "rows": rows,
    }
    write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "rows": [{"seed": r["seed"], **r["metrics"]} for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
