#!/usr/bin/env python3
"""Mechanistic diagnostic for recurrent latent transition stability.

For a learned transition T and canonical code E, estimate:

  closure defect epsilon = angle(T(E(a), E(b)), E(a+b))
  local perturbation gain lambda ~= ||T(E(a)+d,b)-T(E(a),b)|| / ||d||

If a uniform bound e_{t+1} <= lambda e_t + epsilon holds, then lambda<1
implies bounded recurrent error epsilon/(1-lambda), whereas lambda~1 permits
approximately linear drift.  The empirical diagnostic does not prove a global
bound; it is intended as an early warning before long-horizon evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from normed_operator_parameterization_experiment import Config, train_one

EXPERIMENT_NAME = "diagnostic_recurrent_transition_stability"


@torch.inference_mode()
def diagnose(model, *, seed: int, perturb_eps: float, samples: int, max_depth: int) -> dict:
    cfg = model.cfg
    code = model.codebook()
    ids = torch.arange(cfg.modulus)
    a = ids.repeat_interleave(cfg.modulus)
    b = ids.repeat(cfg.modulus)
    target = (a + b) % cfg.modulus
    out = model.op(code[a], code[b])
    cos = F.cosine_similarity(out, code[target], dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos)

    g = torch.Generator().manual_seed(seed + 88000)
    aa = torch.randint(cfg.modulus, (samples,), generator=g)
    bb = torch.randint(cfg.modulus, (samples,), generator=g)
    x = code[aa]
    base = model.op(x, code[bb])
    noise = torch.randn(x.shape, generator=g)
    noise = noise - (noise * x).sum(-1, keepdim=True) * x
    noise = F.normalize(noise, dim=-1)
    perturbed = F.normalize(x + perturb_eps * noise, dim=-1)
    changed = model.op(perturbed, code[bb])
    gain = (changed - base).norm(dim=-1) / (perturbed - x).norm(dim=-1).clamp_min(1e-9)

    # Long-horizon trajectory on the same frozen transition.
    n = samples
    start = torch.randint(cfg.modulus, (n,), generator=g)
    operands = torch.randint(cfg.modulus, (n, max_depth), generator=g)
    state = code[start]
    oracle = start.clone()
    checkpoints = {1, 2, 4, 8, 12, 16, 24, 32, 48, max_depth}
    trajectory = {}
    for t in range(max_depth):
        state = model.op(state, code[operands[:, t]])
        oracle = (oracle + operands[:, t]) % cfg.modulus
        depth = t + 1
        if depth in checkpoints:
            pred = model.logits(state).argmax(-1)
            trajectory[str(depth)] = {
                "accuracy": float(pred.eq(oracle).float().mean()),
                "target_cosine_mean": float(F.cosine_similarity(state, code[oracle], dim=-1).mean()),
            }

    return {
        "closure_angle_mean_rad": float(angle.mean()),
        "closure_angle_p95_rad": float(torch.quantile(angle, 0.95)),
        "closure_angle_max_rad": float(angle.max()),
        "local_gain_mean": float(gain.mean()),
        "local_gain_p95": float(torch.quantile(gain, 0.95)),
        "local_gain_max": float(gain.max()),
        "perturb_eps": perturb_eps,
        "trajectory": trajectory,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--samples", type=int, default=4096)
    p.add_argument("--perturb-eps", type=float, default=0.01)
    p.add_argument("--max-depth", type=int, default=64)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", type=Path, default=Path("artifacts/research/stability_diagnostic/seed22.json"))
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    cfg = Config()
    rows = []
    for arm in ("structured_normed", "flexible_penalty"):
        model, _ = train_one(cfg, arm=arm, seed=args.seed, steps=args.steps, lr=0.02, algebraic_batch=256)
        rows.append({"arm": arm, "metrics": diagnose(model, seed=args.seed, perturb_eps=args.perturb_eps, samples=args.samples, max_depth=args.max_depth)})
    payload = {"experiment": EXPERIMENT_NAME, "seed": args.seed, "rows": rows}
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
