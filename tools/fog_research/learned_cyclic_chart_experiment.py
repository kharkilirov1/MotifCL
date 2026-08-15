#!/usr/bin/env python3
"""EXP-010: learn an operator-compatible cyclic latent chart from successor laws.

Unlike EXP-004/005, the identity codebook is not hand-designed Fourier geometry.
Each of p identities owns H learnable phase coordinates.  The only hard
operator bias is a shared per-plane complex product, so if the learned chart
becomes a representation of Z_p then arbitrary addition should emerge from a
small set of local transition constraints.

Training data contains only successor facts (x, 1) -> x+1.  The closed-cycle arm
includes the single closure relation (p-1,1)->0; the open-chain control omits
that edge.  All other binary addition pairs are held out from training.
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

EXPERIMENT_NAME = "exp_010_learned_cyclic_chart"
Arm = Literal["closed_cycle", "open_chain", "closed_cycle_root"]


@dataclass(frozen=True)
class LearnedChartConfig:
    modulus: int = 31
    harmonics: int = 6
    separation_weight: float = 0.05
    head_scale: float = 20.0

    @property
    def d_model(self) -> int:
        return 2 * self.harmonics


class LearnedPhaseChart(nn.Module):
    def __init__(self, cfg: LearnedChartConfig, seed: int):
        super().__init__()
        self.cfg = cfg
        generator = torch.Generator().manual_seed(seed)
        phase = torch.rand(
            cfg.modulus, cfg.harmonics, generator=generator
        ) * (2 * math.pi) - math.pi
        self.phase = nn.Parameter(phase)

    def gauged_phase(self) -> torch.Tensor:
        # Identity 0 defines the global chart origin but its raw parameter need
        # not be frozen; subtracting it removes the unidentifiable global phase.
        return self.phase - self.phase[:1]

    def codebook(self) -> torch.Tensor:
        phase = self.gauged_phase()
        code = torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)
        return F.normalize(code, dim=-1)

    @staticmethod
    def multiply_codes(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h = a.size(-1) // 2
        aa = a.reshape(*a.shape[:-1], 2, h)
        bb = b.reshape(*b.shape[:-1], 2, h)
        ca, sa = aa[..., 0, :], aa[..., 1, :]
        cb, sb = bb[..., 0, :], bb[..., 1, :]
        out = torch.cat((ca * cb - sa * sb, sa * cb + ca * sb), dim=-1)
        return F.normalize(out, dim=-1)

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.cfg.head_scale * F.normalize(z, dim=-1) @ self.codebook().T


def train_chart(
    cfg: LearnedChartConfig,
    *,
    arm: Arm,
    seed: int,
    steps: int,
    lr: float,
) -> tuple[LearnedPhaseChart, list[dict]]:
    torch.manual_seed(seed)
    model = LearnedPhaseChart(cfg, seed=seed)
    optimizer = torch.optim.Adam([model.phase], lr=lr)
    has_closure = arm != "open_chain"
    root_weight = 1.0 if arm == "closed_cycle_root" else 0.0
    sources = torch.arange(cfg.modulus if has_closure else cfg.modulus - 1)
    targets = (sources + 1) % cfg.modulus
    trace = []
    for step in range(steps):
        code = model.codebook()
        predicted = model.multiply_codes(code[sources], code[torch.ones_like(sources)])
        target_code = code[targets]
        transition_loss = (
            1.0 - F.cosine_similarity(predicted, target_code, dim=-1)
        ).mean()
        logits = cfg.head_scale * code @ code.T
        separation_loss = F.cross_entropy(logits, torch.arange(cfg.modulus))
        step_phase = model.gauged_phase()[1]
        root_loss = (1.0 - torch.cos(cfg.modulus * step_phase)).mean()
        loss = (
            transition_loss
            + cfg.separation_weight * separation_loss
            + root_weight * root_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 10) == 0 or step + 1 == steps:
            trace.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "transition_loss": float(transition_loss.detach()),
                    "separation_loss": float(separation_loss.detach()),
                    "root_loss": float(root_loss.detach()),
                    "root_weight": root_weight,
                }
            )
    return model.eval(), trace


@torch.inference_mode()
def evaluate_chart(
    model: LearnedPhaseChart,
    *,
    arm: Arm,
    sequence_examples: int,
    sequence_depths: tuple[int, ...],
    seed: int,
) -> dict:
    cfg = model.cfg
    code = model.codebook()
    identities = torch.arange(cfg.modulus)
    self_acc = float((cfg.head_scale * code @ code.T).argmax(-1).eq(identities).float().mean())

    a = identities.repeat_interleave(cfg.modulus)
    b = identities.repeat(cfg.modulus)
    target = (a + b) % cfg.modulus
    z = model.multiply_codes(code[a], code[b])
    pred = model.logits(z).argmax(-1)
    all_pair_acc = float(pred.eq(target).float().mean())
    heldout_mask = b.ne(1)
    heldout_pair_acc = float(pred[heldout_mask].eq(target[heldout_mask]).float().mean())

    successor_sources = torch.arange(cfg.modulus)
    successor_target = (successor_sources + 1) % cfg.modulus
    successor_pred = model.logits(
        model.multiply_codes(code[successor_sources], code[torch.ones_like(successor_sources)])
    ).argmax(-1)
    successor_acc = float(successor_pred.eq(successor_target).float().mean())
    closure_acc = float(successor_pred[-1].eq(torch.tensor(0)).float())

    phase = model.gauged_phase()
    step_phase = phase[1]
    root_residual = torch.sqrt(
        (torch.cos(cfg.modulus * step_phase) - 1.0).square()
        + torch.sin(cfg.modulus * step_phase).square()
    ).mean()

    generator = torch.Generator().manual_seed(seed + 100000)
    recurrence = []
    for depth in sequence_depths:
        start = torch.randint(cfg.modulus, (sequence_examples,), generator=generator)
        operands = torch.randint(
            cfg.modulus, (sequence_examples, depth), generator=generator
        )
        state = code[start]
        oracle = start.clone()
        hop_acc = []
        for t in range(depth):
            state = model.multiply_codes(state, code[operands[:, t]])
            oracle = (oracle + operands[:, t]) % cfg.modulus
            hop_acc.append(float(model.logits(state).argmax(-1).eq(oracle).float().mean()))
        recurrence.append(
            {
                "depth": depth,
                "final_accuracy": hop_acc[-1],
                "minimum_hop_accuracy": min(hop_acc),
            }
        )

    return {
        "arm": arm,
        "self_decode_accuracy": self_acc,
        "successor_accuracy_all_edges": successor_acc,
        "closure_edge_accuracy": closure_acc,
        "all_binary_addition_accuracy": all_pair_acc,
        "heldout_binary_pair_accuracy_b_ne_1": heldout_pair_acc,
        "mean_pth_root_residual": float(root_residual),
        "recurrence": recurrence,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/research/exp_010/metrics.json"))
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--sequence-examples", type=int, default=2048)
    p.add_argument("--sequence-depths", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--arms", nargs="+",
        choices=("closed_cycle", "open_chain", "closed_cycle_root"),
        default=["closed_cycle", "open_chain", "closed_cycle_root"],
    )
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    cfg = LearnedChartConfig()
    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            model, trace = train_chart(
                cfg, arm=arm, seed=seed, steps=args.steps, lr=args.lr  # type: ignore[arg-type]
            )
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "trace": trace,
                    "evaluation": evaluate_chart(
                        model,
                        arm=arm,  # type: ignore[arg-type]
                        sequence_examples=args.sequence_examples,
                        sequence_depths=tuple(args.sequence_depths),
                        seed=seed,
                    ),
                }
            )
    payload = {
        "experiment": EXPERIMENT_NAME,
        "config": asdict(cfg),
        "training_facts_closed_cycle": cfg.modulus,
        "full_binary_table_size": cfg.modulus ** 2,
        "rows": rows,
    }
    write_json(args.output, payload)
    summary = {}
    for arm in args.arms:
        rr = [r["evaluation"] for r in rows if r["arm"] == arm]
        summary[arm] = {
            "runs": len(rr),
            "successor_accuracy": sum(x["successor_accuracy_all_edges"] for x in rr) / len(rr),
            "heldout_binary_accuracy": sum(x["heldout_binary_pair_accuracy_b_ne_1"] for x in rr) / len(rr),
            "root_residual": sum(x["mean_pth_root_residual"] for x in rr) / len(rr),
            "depth32_accuracy": sum(
                next(y for y in x["recurrence"] if y["depth"] == 32)["final_accuracy"]
                for x in rr
            ) / len(rr),
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
