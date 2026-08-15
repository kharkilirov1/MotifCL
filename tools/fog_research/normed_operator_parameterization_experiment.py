#!/usr/bin/env python3
"""EXP-014: algebraic laws by construction vs by penalty.

EXP-012/013 show that a flexible bilinear operator can satisfy terminal successor
and even near-zero continuous associativity/commutativity while mapping arbitrary
pairs off the canonical identity manifold.  This experiment tests a narrower
operator class whose critical geometry is architectural rather than penalized.

The structured arm operates independently in H learned 2D planes.  In each
plane it performs complex multiplication in a learned orthogonal coordinate
frame.  Therefore, for unit-plane inputs, the operation is commutative,
associative, norm preserving, and closed on the product of circles by
construction.  Only the coordinate-frame angle and identity phases are learned.

Training supervision remains only repeated successor (+1) facts at terminal
depths 1,2,3.  Arbitrary binary addition pairs are never labeled during
optimization.
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

EXPERIMENT_NAME = "exp_014_normed_operator_parameterization"
Arm = Literal["structured_normed", "flexible_penalty"]


@dataclass(frozen=True)
class Config:
    modulus: int = 31
    harmonics: int = 4
    head_scale: float = 20.0
    separation_weight: float = 0.05
    algebraic_weight: float = 0.5

    @property
    def d_model(self) -> int:
        return 2 * self.harmonics


def _codes_from_phase(phase: torch.Tensor) -> torch.Tensor:
    phase = phase - phase[:1]
    return F.normalize(torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1), dim=-1)


def _split_planes(z: torch.Tensor, h: int) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = z[..., :h], z[..., h:]
    norm = torch.sqrt(x.square() + y.square()).clamp_min(1e-8)
    return x / norm, y / norm


class StructuredNormedAlgebra(nn.Module):
    def __init__(self, cfg: Config, seed: int):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        self.phase = nn.Parameter(
            torch.rand(cfg.modulus, cfg.harmonics, generator=g) * 2 * math.pi - math.pi
        )
        # Coordinate-frame angle per plane.  This is the only operator parameter.
        self.frame_angle = nn.Parameter(
            torch.rand(cfg.harmonics, generator=g) * 2 * math.pi - math.pi
        )

    def codebook(self) -> torch.Tensor:
        return _codes_from_phase(self.phase)

    def op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h = self.cfg.harmonics
        ar, ai = _split_planes(a, h)
        br, bi = _split_planes(b, h)

        # R(phi) maps the learned latent plane into the multiplication frame.
        c, s = torch.cos(self.frame_angle), torch.sin(self.frame_angle)
        a_r = c * ar - s * ai
        a_i = s * ar + c * ai
        b_r = c * br - s * bi
        b_i = s * br + c * bi
        p_r = a_r * b_r - a_i * b_i
        p_i = a_r * b_i + a_i * b_r
        # Map the product back with R(-phi).
        out_r = c * p_r + s * p_i
        out_i = -s * p_r + c * p_i
        return F.normalize(torch.cat((out_r, out_i), dim=-1), dim=-1)

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.cfg.head_scale * F.normalize(z, dim=-1) @ self.codebook().T


class FlexiblePenaltyAlgebra(nn.Module):
    def __init__(self, cfg: Config, seed: int):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        self.phase = nn.Parameter(
            torch.rand(cfg.modulus, cfg.harmonics, generator=g) * 2 * math.pi - math.pi
        )
        self.operator = nn.Parameter(torch.randn(cfg.harmonics, 4, 2, generator=g) * 0.3)

    def codebook(self) -> torch.Tensor:
        return _codes_from_phase(self.phase)

    def op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h = self.cfg.harmonics
        ca, sa = a[..., :h], a[..., h:]
        cb, sb = b[..., :h], b[..., h:]
        feat = torch.stack((ca * cb, ca * sb, sa * cb, sa * sb), dim=-1)
        out = torch.einsum("...hf,hfo->...ho", feat, self.operator)
        return F.normalize(torch.cat((out[..., 0], out[..., 1]), dim=-1), dim=-1)

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.cfg.head_scale * F.normalize(z, dim=-1) @ self.codebook().T


Model = StructuredNormedAlgebra | FlexiblePenaltyAlgebra


def train_one(
    cfg: Config,
    *,
    arm: Arm,
    seed: int,
    steps: int,
    lr: float,
    algebraic_batch: int,
) -> tuple[Model, list[dict]]:
    torch.manual_seed(seed)
    model: Model
    if arm == "structured_normed":
        model = StructuredNormedAlgebra(cfg, seed)
    else:
        model = FlexiblePenaltyAlgebra(cfg, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ids = torch.arange(cfg.modulus)
    generator = torch.Generator().manual_seed(seed + 1901)
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
            depth_losses.append((1 - F.cosine_similarity(state, target, dim=-1)).mean())
        successor_loss = torch.stack(depth_losses).mean()
        separation_loss = F.cross_entropy(cfg.head_scale * code @ code.T, ids)
        loss = successor_loss + cfg.separation_weight * separation_loss
        identity_loss = torch.tensor(0.0)
        comm_loss = torch.tensor(0.0)
        assoc_loss = torch.tensor(0.0)

        # The flexible arm receives the strongest EXP-012 penalty baseline.
        # The structured arm does not need these laws as losses: they hold by
        # its operator class (up to numerical normalization).
        if arm == "flexible_penalty":
            zero = code[torch.zeros_like(ids)]
            identity_loss = (
                2
                - F.cosine_similarity(model.op(code, zero), code, dim=-1)
                - F.cosine_similarity(model.op(zero, code), code, dim=-1)
            ).mean()
            a = torch.randint(cfg.modulus, (algebraic_batch,), generator=generator)
            b = torch.randint(cfg.modulus, (algebraic_batch,), generator=generator)
            c = torch.randint(cfg.modulus, (algebraic_batch,), generator=generator)
            ab = model.op(code[a], code[b])
            comm_loss = (1 - F.cosine_similarity(ab, model.op(code[b], code[a]), dim=-1)).mean()
            assoc_loss = (
                1
                - F.cosine_similarity(
                    model.op(ab, code[c]),
                    model.op(code[a], model.op(code[b], code[c])),
                    dim=-1,
                )
            ).mean()
            loss = loss + cfg.algebraic_weight * (identity_loss + comm_loss + assoc_loss)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 10) == 0 or step + 1 == steps:
            trace.append({
                "step": step + 1,
                "loss": float(loss.detach()),
                "successor_loss": float(successor_loss.detach()),
                "separation_loss": float(separation_loss.detach()),
                "identity_loss": float(identity_loss.detach()),
                "commutativity_loss": float(comm_loss.detach()),
                "associativity_loss": float(assoc_loss.detach()),
            })
    return model.eval(), trace


@torch.inference_mode()
def evaluate(model: Model, *, seed: int, recurrent_depth: int, recurrent_examples: int) -> dict:
    cfg = model.cfg
    code = model.codebook()
    ids = torch.arange(cfg.modulus)
    one = code[torch.ones_like(ids)]
    successor = model.logits(model.op(code, one)).argmax(-1)
    successor_acc = float(successor.eq((ids + 1) % cfg.modulus).float().mean())

    a = ids.repeat_interleave(cfg.modulus)
    b = ids.repeat(cfg.modulus)
    target = (a + b) % cfg.modulus
    out = model.op(code[a], code[b])
    pred = model.logits(out).argmax(-1)
    pair_acc = float(pred.eq(target).float().mean())
    nearest = (out @ code.T).max(-1).values
    true_cos = F.cosine_similarity(out, code[target], dim=-1)

    g = torch.Generator().manual_seed(seed + 50000)
    start = torch.randint(cfg.modulus, (recurrent_examples,), generator=g)
    operands = torch.randint(cfg.modulus, (recurrent_examples, recurrent_depth), generator=g)
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
    comm = float((1 - F.cosine_similarity(ab, model.op(code[bb], code[aa]), dim=-1)).mean())
    assoc = float((1 - F.cosine_similarity(
        model.op(ab, code[cc]), model.op(code[aa], model.op(code[bb], code[cc])), dim=-1
    )).mean())
    return {
        "successor_accuracy": successor_acc,
        "all_binary_addition_accuracy": pair_acc,
        "recurrent_final_accuracy": hops[-1],
        "recurrent_minimum_hop_accuracy": min(hops),
        "recurrent_depth": recurrent_depth,
        "nearest_code_cosine_mean": float(nearest.mean()),
        "true_target_cosine_mean": float(true_cos.mean()),
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
    p.add_argument("--output", type=Path, default=Path("artifacts/research/exp_014/metrics.json"))
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--seeds", type=int, nargs="+", default=[20, 21, 22])
    p.add_argument("--arms", nargs="+", choices=["structured_normed", "flexible_penalty"], default=["structured_normed", "flexible_penalty"])
    p.add_argument("--algebraic-batch", type=int, default=256)
    p.add_argument("--recurrent-depth", type=int, default=64)
    p.add_argument("--recurrent-examples", type=int, default=512)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    cfg = Config()
    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            model, trace = train_one(cfg, arm=arm, seed=seed, steps=args.steps, lr=args.lr, algebraic_batch=args.algebraic_batch)
            rows.append({"arm": arm, "seed": seed, "trace": trace, "metrics": evaluate(model, seed=seed, recurrent_depth=args.recurrent_depth, recurrent_examples=args.recurrent_examples)})
    payload = {
        "experiment": EXPERIMENT_NAME,
        "config": asdict(cfg),
        "training": {"steps": args.steps, "lr": args.lr, "supervised_terminal_depths": [1,2,3], "arbitrary_binary_pair_targets_used": False},
        "rows": rows,
    }
    write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "rows": [{"arm": r["arm"], "seed": r["seed"], **r["metrics"]} for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
