#!/usr/bin/env python3
"""EXP-016: independent identity codes vs a generator-orbit latent chart.

A failure in EXP-015 had 100% successor top-1 accuracy but non-constant phase
increments across identities; it learned a near-table of states rather than one
shared cyclic transition law.

This experiment compares:
- free_codebook: one learnable phase vector per identity;
- generator_orbit: E(x) is generated as x times one learned phase increment.

Both use the same normed law-by-construction binary operator and only repeated
successor supervision at depths 1,2,3.  No arbitrary binary pair labels are
used.  The orbit parameterization makes translation sharing architectural.
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

EXPERIMENT_NAME = "exp_016_generator_orbit_chart"
Arm = Literal["free_codebook", "generator_orbit", "generator_orbit_closure"]


@dataclass(frozen=True)
class Config:
    order: int = 30
    harmonics: int = 4
    head_scale: float = 20.0
    separation_weight: float = 0.05


class CyclicChart(nn.Module):
    def __init__(self, cfg: Config, seed: int, arm: Arm):
        super().__init__()
        self.cfg = cfg
        self.arm = arm
        g = torch.Generator().manual_seed(seed)
        if arm == "free_codebook":
            self.phase = nn.Parameter(
                torch.rand(cfg.order, cfg.harmonics, generator=g) * 2 * math.pi - math.pi
            )
            self.register_parameter("generator_phase", None)
        else:
            self.generator_phase = nn.Parameter(
                torch.rand(cfg.harmonics, generator=g) * 2 * math.pi - math.pi
            )
            self.register_parameter("phase", None)
        self.frame_angle = nn.Parameter(
            torch.rand(cfg.harmonics, generator=g) * 2 * math.pi - math.pi
        )

    def phases(self) -> torch.Tensor:
        if self.arm == "free_codebook":
            assert self.phase is not None
            return self.phase - self.phase[:1]
        assert self.generator_phase is not None
        x = torch.arange(self.cfg.order, device=self.generator_phase.device, dtype=self.generator_phase.dtype)
        return x[:, None] * self.generator_phase[None, :]

    def codebook(self) -> torch.Tensor:
        phase = self.phases()
        return F.normalize(torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1), dim=-1)

    def op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h = self.cfg.harmonics
        ar, ai = a[..., :h], a[..., h:]
        br, bi = b[..., :h], b[..., h:]
        an = torch.sqrt(ar.square() + ai.square()).clamp_min(1e-8)
        bn = torch.sqrt(br.square() + bi.square()).clamp_min(1e-8)
        ar, ai, br, bi = ar / an, ai / an, br / bn, bi / bn
        c, s = torch.cos(self.frame_angle), torch.sin(self.frame_angle)
        a_r, a_i = c * ar - s * ai, s * ar + c * ai
        b_r, b_i = c * br - s * bi, s * br + c * bi
        p_r = a_r * b_r - a_i * b_i
        p_i = a_r * b_i + a_i * b_r
        out_r, out_i = c * p_r + s * p_i, -s * p_r + c * p_i
        return F.normalize(torch.cat((out_r, out_i), dim=-1), dim=-1)

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.cfg.head_scale * F.normalize(z, dim=-1) @ self.codebook().T


def train_one(cfg: Config, *, seed: int, arm: Arm, steps: int, lr: float) -> CyclicChart:
    torch.manual_seed(seed)
    model = CyclicChart(cfg, seed, arm)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ids = torch.arange(cfg.order)
    for _ in range(steps):
        code = model.codebook()
        one = code[1].expand(cfg.order, -1)
        losses = []
        for depth in (1, 2, 3):
            state = code
            for _ in range(depth):
                state = model.op(state, one)
            target = code[(ids + depth) % cfg.order]
            losses.append((1 - F.cosine_similarity(state, target, dim=-1)).mean())
        sep = F.cross_entropy(cfg.head_scale * code @ code.T, ids)
        loss = torch.stack(losses).mean() + cfg.separation_weight * sep
        if arm == "generator_orbit_closure":
            assert model.generator_phase is not None
            # Because every identity is T^x E(0), this single constraint closes
            # the whole learned orbit rather than regularizing one unrelated scalar.
            closure_loss = (1.0 - torch.cos(cfg.order * model.generator_phase)).mean()
            zero = code[:1]
            identity_loss = (1.0 - F.cosine_similarity(model.op(zero, zero), zero, dim=-1)).mean()
            loss = loss + closure_loss + identity_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.eval()


@torch.inference_mode()
def evaluate(model: CyclicChart, *, seed: int, depth: int, examples: int) -> dict:
    cfg = model.cfg
    code = model.codebook()
    ids = torch.arange(cfg.order)
    one = code[1].expand(cfg.order, -1)
    succ_out = model.op(code, one)
    succ_target = (ids + 1) % cfg.order
    succ_acc = float(model.logits(succ_out).argmax(-1).eq(succ_target).float().mean())
    succ_cos = float(F.cosine_similarity(succ_out, code[succ_target], dim=-1).mean())

    a = ids.repeat_interleave(cfg.order)
    b = ids.repeat(cfg.order)
    target = (a + b) % cfg.order
    out = model.op(code[a], code[b])
    pair_acc = float(model.logits(out).argmax(-1).eq(target).float().mean())
    pair_cos = float(F.cosine_similarity(out, code[target], dim=-1).mean())

    phase = model.phases()
    increment = torch.atan2(
        torch.sin(phase.roll(-1, 0) - phase),
        torch.cos(phase.roll(-1, 0) - phase),
    )
    concentration = torch.abs(torch.exp(1j * increment).mean(0))

    g = torch.Generator().manual_seed(seed + 90000)
    start = torch.randint(cfg.order, (examples,), generator=g)
    operands = torch.randint(cfg.order, (examples, depth), generator=g)
    state = code[start]
    oracle = start.clone()
    checkpoints = {1,2,4,8,16,32,64,depth}
    trajectory = {}
    for t in range(depth):
        state = model.op(state, code[operands[:,t]])
        oracle = (oracle + operands[:,t]) % cfg.order
        d=t+1
        if d in checkpoints:
            trajectory[str(d)] = {
                "accuracy": float(model.logits(state).argmax(-1).eq(oracle).float().mean()),
                "target_cosine_mean": float(F.cosine_similarity(state, code[oracle], dim=-1).mean()),
            }
    return {
        "successor_accuracy": succ_acc,
        "successor_cosine_mean": succ_cos,
        "all_binary_accuracy": pair_acc,
        "all_binary_target_cosine_mean": pair_cos,
        "phase_increment_concentration_min": float(concentration.min()),
        "phase_increment_concentration_mean": float(concentration.mean()),
        "trajectory": trajectory,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('artifacts/research/exp_016/metrics.json'))
    p.add_argument('--seeds',type=int,nargs='+',default=[40,41,42,43,44])
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--lr',type=float,default=.02)
    p.add_argument('--depth',type=int,default=64)
    p.add_argument('--examples',type=int,default=1024)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args(); torch.set_num_threads(args.threads); cfg=Config(); rows=[]
    for arm in ('free_codebook','generator_orbit','generator_orbit_closure'):
        for seed in args.seeds:
            model=train_one(cfg,seed=seed,arm=arm,steps=args.steps,lr=args.lr)
            rows.append({'arm':arm,'seed':seed,'metrics':evaluate(model,seed=seed,depth=args.depth,examples=args.examples)})
    payload={'experiment':EXPERIMENT_NAME,'config':asdict(cfg),'training':{'steps':args.steps,'lr':args.lr,'supervised_terminal_depths':[1,2,3],'arbitrary_binary_pair_targets_used':False},'rows':rows}
    write_json(args.output,payload); print(json.dumps(payload,indent=2))

if __name__=='__main__': main()
