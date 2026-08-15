#!/usr/bin/env python3
"""EXP-003: terminal-only supervision can hide periodic latent coordinate gauges.

This is a controlled generated-value computation gate over Z_p.  A canonical
Fourier register represents a value x.  Each instruction adds an operand b by an
exact rotation, producing a *new* latent value E(x+b) that need not appear in the
prompt.  After every transition, a shared trainable gauge rotation G_phi is
applied to the register.

Because the semantic add rotations commute with the gauge rotation, a depth-d
terminal observation constrains only G_phi**d.  Therefore supervision at a
single depth d cannot identify the per-step latent coordinate system: any d-th
root of identity is an exact solution.  Terminal depths whose gcd is one remove
this exact periodic ambiguity in this gauge family, although optimization may
still have non-zero-loss local minima.

No intermediate latent is decoded or supervised during training.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

EXPERIMENT_NAME = "exp_003_generated_value_gauge"


@dataclass(frozen=True)
class GaugeConfig:
    modulus: int = 31
    harmonics: int = 15
    head_scale: float = 30.0

    def validate(self) -> None:
        if self.modulus < 5 or self.modulus % 2 == 0:
            raise ValueError("modulus must be an odd integer >= 5")
        if not (1 <= self.harmonics <= (self.modulus - 1) // 2):
            raise ValueError("harmonics must be in [1, (modulus-1)//2]")


class FourierLatentRegister(nn.Module):
    """Canonical Z_p register plus one shared latent gauge angle."""

    def __init__(self, cfg: GaugeConfig, initial_phi: float):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.phi = nn.Parameter(torch.tensor(float(initial_phi)))
        h = torch.arange(1, cfg.harmonics + 1, dtype=torch.float32)
        x = torch.arange(cfg.modulus, dtype=torch.float32)
        angle = 2.0 * math.pi * x[:, None] * h[None, :] / cfg.modulus
        code = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
        code = code.reshape(cfg.modulus, 2 * cfg.harmonics)
        code = F.normalize(code, dim=-1)
        self.register_buffer("codebook", code)
        self.register_buffer("harmonic_ids", h)

    @property
    def d_model(self) -> int:
        return 2 * self.cfg.harmonics

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return self.codebook[value]

    def _rotate_pairs(self, z: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        pairs = z.reshape(*z.shape[:-1], self.cfg.harmonics, 2)
        while angle.ndim < pairs.ndim - 1:
            angle = angle.unsqueeze(-1)
        c = torch.cos(angle)
        s = torch.sin(angle)
        x, y = pairs[..., 0], pairs[..., 1]
        out = torch.stack((c * x - s * y, s * x + c * y), dim=-1)
        return out.reshape(*z.shape[:-1], self.d_model)

    def semantic_add(self, z: torch.Tensor, operand: torch.Tensor) -> torch.Tensor:
        # Each Fourier harmonic rotates by h * 2pi*b/p for x -> x+b.
        angle = (
            2.0
            * math.pi
            * operand.float()[..., None]
            * self.harmonic_ids[None, :]
            / self.cfg.modulus
        )
        return self._rotate_pairs(z, angle)

    def apply_gauge(self, z: torch.Tensor) -> torch.Tensor:
        # Same coordinate-chart rotation in every Fourier plane.
        angle = self.phi.expand(*z.shape[:-1], self.cfg.harmonics)
        return self._rotate_pairs(z, angle)

    def step(self, z: torch.Tensor, operand: torch.Tensor) -> torch.Tensor:
        return self.apply_gauge(self.semantic_add(z, operand))

    def run(
        self,
        start: torch.Tensor,
        operands: torch.Tensor,
        *,
        return_history: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if operands.ndim != 2 or start.ndim != 1 or operands.size(0) != start.size(0):
            raise ValueError("start must be [B], operands must be [B, depth]")
        z = self.encode(start)
        history: list[torch.Tensor] = []
        for t in range(operands.size(1)):
            z = self.step(z, operands[:, t])
            if return_history:
                history.append(z)
        return z, history

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.cfg.head_scale * F.normalize(z.float(), dim=-1) @ self.codebook.float().T


def terminal_target(start: torch.Tensor, operands: torch.Tensor, modulus: int) -> torch.Tensor:
    return (start + operands.sum(dim=1)) % modulus


def exact_gauge_order_residual(phi: float, order: int) -> float:
    # Distance of exp(i*order*phi) from identity in the complex plane.
    return abs(complex(math.cos(order * phi), math.sin(order * phi)) - 1.0)


def gcd_depths(depths: Iterable[int]) -> int:
    g = 0
    for depth in depths:
        g = math.gcd(g, int(depth))
    return g


def sample_batch(cfg: GaugeConfig, *, depth: int, batch_size: int, generator: torch.Generator):
    start = torch.randint(cfg.modulus, (batch_size,), generator=generator)
    operands = torch.randint(cfg.modulus, (batch_size, depth), generator=generator)
    target = terminal_target(start, operands, cfg.modulus)
    return start, operands, target


def train_arm(
    *,
    cfg: GaugeConfig,
    depths: tuple[int, ...],
    initial_phi: float,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
) -> tuple[FourierLatentRegister, list[dict]]:
    torch.manual_seed(seed)
    model = FourierLatentRegister(cfg, initial_phi=initial_phi)
    optimizer = torch.optim.Adam([model.phi], lr=lr)
    generator = torch.Generator().manual_seed(seed + 99173)
    trace = []
    for step in range(steps):
        depth = depths[step % len(depths)]
        start, operands, target = sample_batch(
            cfg, depth=depth, batch_size=batch_size, generator=generator
        )
        z, _ = model.run(start, operands)
        logits = model.logits(z)
        target_code = model.encode(target)
        # Terminal-only *latent* supervision.  This deliberately asks whether
        # the final register itself is canonical, not merely whether a broad
        # classifier still gives the right top-1 token.  Intermediate hops
        # receive no loss.
        loss = (1.0 - F.cosine_similarity(z.float(), target_code.float(), dim=-1)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 10) == 0 or step + 1 == steps:
            trace.append(
                {
                    "step": step + 1,
                    "depth": depth,
                    "loss": float(loss.detach()),
                    "phi": float(model.phi.detach()),
                    "batch_accuracy": float(logits.argmax(-1).eq(target).float().mean()),
                    "terminal_latent_cosine": float(
                        F.cosine_similarity(z.float(), target_code.float(), dim=-1).mean().detach()
                    ),
                }
            )
    return model.eval(), trace


@torch.inference_mode()
def evaluate_depth(
    model: FourierLatentRegister,
    *,
    depth: int,
    examples: int,
    seed: int,
) -> dict:
    cfg = model.cfg
    generator = torch.Generator().manual_seed(seed + 104729 * depth)
    start, operands, target = sample_batch(
        cfg, depth=depth, batch_size=examples, generator=generator
    )
    z, history = model.run(start, operands, return_history=True)
    pred = model.logits(z).argmax(-1)
    oracle = start.clone()
    hop_rows = []
    for hop, state in enumerate(history, 1):
        oracle = (oracle + operands[:, hop - 1]) % cfg.modulus
        oracle_code = model.encode(oracle)
        hop_pred = model.logits(state).argmax(-1)
        hop_rows.append(
            {
                "hop": hop,
                "canonical_accuracy": float(hop_pred.eq(oracle).float().mean()),
                "canonical_cosine": float(
                    F.cosine_similarity(state.float(), oracle_code.float(), dim=-1).mean()
                ),
            }
        )
    return {
        "depth": depth,
        "accuracy": float(pred.eq(target).float().mean()),
        "terminal_cosine": hop_rows[-1]["canonical_cosine"],
        "hops": hop_rows,
    }


def theoretical_exact_roots(depths: tuple[int, ...]) -> dict:
    g = gcd_depths(depths)
    roots = [2.0 * math.pi * k / g for k in range(g)] if g else [0.0]
    return {
        "gcd": g,
        "exact_gauge_roots_mod_2pi": roots,
        "noncanonical_exact_roots_exist": g > 1,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def run_protocol(args: argparse.Namespace) -> dict:
    cfg = GaugeConfig(args.modulus, args.harmonics, args.head_scale)
    arms = {
        "depth2_only": ((2,), args.initial_phi),
        "depth3_only": ((3,), args.initial_phi),
        "coprime_2_3": ((2, 3), args.coprime_initial_phi),
        # Same exact-identifiability constraints, but deliberately initialized
        # in the noncanonical basin.  This distinguishes removal of exact
        # gauge degeneracy from the separate optimization-landscape problem.
        "coprime_2_3_far_init": ((2, 3), args.initial_phi),
    }
    result = {
        "experiment": EXPERIMENT_NAME,
        "config": asdict(cfg),
        "note": "Intermediate states receive no loss. Terminal loss aligns only the final latent register to the canonical target. Exact gauge ambiguity is controlled by gcd(terminal depths).",
        "arms": {},
    }
    for name, (depths, init_phi) in arms.items():
        seed_rows = []
        for seed in args.seeds:
            model, trace = train_arm(
                cfg=cfg,
                depths=depths,
                initial_phi=init_phi,
                seed=seed,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
            )
            phi = float(model.phi.detach())
            evals = [
                evaluate_depth(model, depth=d, examples=args.eval_examples, seed=seed + 800000)
                for d in range(1, args.eval_max_depth + 1)
            ]
            seed_rows.append(
                {
                    "seed": seed,
                    "initial_phi": init_phi,
                    "final_phi": phi,
                    "trace": trace,
                    "gauge_residuals": {
                        str(d): exact_gauge_order_residual(phi, d)
                        for d in range(1, args.eval_max_depth + 1)
                    },
                    "evaluation": evals,
                }
            )
        result["arms"][name] = {
            "terminal_depths": depths,
            "theory": theoretical_exact_roots(depths),
            "seeds": seed_rows,
        }
    write_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/research/exp_003/metrics.json"))
    p.add_argument("--modulus", type=int, default=31)
    p.add_argument("--harmonics", type=int, default=15)
    p.add_argument("--head-scale", type=float, default=30.0)
    p.add_argument("--initial-phi", type=float, default=2.7,
                   help="Initialization deliberately near a noncanonical root for single-depth arms.")
    p.add_argument("--coprime-initial-phi", type=float, default=0.9,
                   help="Initialization inside the canonical basin for the coprime-depth optimization arm.")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--eval-examples", type=int, default=4096)
    p.add_argument("--eval-max-depth", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--threads", type=int, default=4)
    return p


def main() -> None:
    args = parser().parse_args()
    torch.set_num_threads(args.threads)
    result = run_protocol(args)
    summary = {}
    for arm, payload in result["arms"].items():
        summary[arm] = [
            {
                "seed": row["seed"],
                "final_phi": row["final_phi"],
                "accuracy_1_8": [x["accuracy"] for x in row["evaluation"][:8]],
            }
            for row in payload["seeds"]
        ]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
