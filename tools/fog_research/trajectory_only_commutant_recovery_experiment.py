#!/usr/bin/env python3
"""EXP-033: recover repeated latent operator structure from trajectory pairs only.

The structural compiler no longer receives action matrices directly.  For each
hidden operator it observes only continuous state transition pairs

    z -> f(z)

with noisy output states.  No discrete identity labels, canonical codebook, or
multiplicity metadata are used for system identification or compilation.

A ridge linear system-identification stage estimates one dense action per
operator from random continuous probes.  The approximate-commutant compiler
from EXP-032 then tests whether those estimates contain a shared repeated-irrep
law and, if accepted, aligns and averages the repeated blocks.

The codebook is used only by the held-out evaluator after compilation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from approximate_commutant_denoising_experiment import (
    compile_approximate_repeated_irrep,
    execute,
)
from joint_commutant_block_compiler_experiment import make_problem

EXPERIMENT_NAME = "exp_033_trajectory_only_commutant_recovery"


def collect_trajectory_pairs(W: torch.Tensor, samples: int, noise: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    d = W.size(0)
    g = torch.Generator().manual_seed(seed)
    X = F.normalize(torch.randn(samples, d, generator=g, dtype=torch.float64), dim=-1)
    Y = X @ W.T
    if noise > 0:
        N = F.normalize(torch.randn(Y.shape, generator=g, dtype=torch.float64), dim=-1)
        Y = F.normalize(Y + noise * N, dim=-1)
    return X, Y


def ridge_estimate(X: torch.Tensor, Y: torch.Tensor, ridge: float = 1e-4) -> torch.Tensor:
    d = X.size(1)
    eye = torch.eye(d, dtype=X.dtype)
    # Row convention Y ~= X W^T.
    WT = torch.linalg.solve(X.T @ X + ridge * eye, X.T @ Y)
    return WT.T


def identify_actions(problem: dict, samples: int, noise: float, seed: int, ridge: float) -> list[torch.Tensor]:
    out = []
    for j, key in enumerate(["A", "B"]):
        X, Y = collect_trajectory_pairs(problem[key], samples, noise, seed + 100 * j)
        out.append(ridge_estimate(X, Y, ridge=ridge))
    return out


def _jsonable(comp: dict) -> dict:
    hidden = {"G", "shared_blocks", "shared_polar_blocks", "reconstructed_shared", "reconstructed_shared_polar"}
    return {k: v for k, v in comp.items() if k not in hidden}


def run_one(multiplicity: int, seed: int, sample_factor: float, noise: float, ridge: float, gap_threshold: float, depth: int, examples: int) -> dict:
    problem = make_problem(multiplicity, seed)
    d = 2 * multiplicity
    samples = max(1, int(round(sample_factor * d)))
    estimated = identify_actions(problem, samples, noise, seed + 330_000, ridge)
    comp = compile_approximate_repeated_irrep(estimated, seed, gap_threshold=gap_threshold)
    raw = execute(problem, estimated, seed + 1, depth, examples)
    row = {
        "seed": seed,
        "multiplicity": multiplicity,
        "dimension": d,
        "samples_per_operator": samples,
        "sample_factor_times_dimension": sample_factor,
        "trajectory_output_noise": noise,
        "raw_identified_execution": raw,
        "compiler": _jsonable(comp),
    }
    if comp.get("accepted"):
        row["compiled_shared_execution"] = execute(problem, comp["reconstructed_shared"], seed + 1, depth, examples)
        row["compiled_shared_polar_execution"] = execute(problem, comp["reconstructed_shared_polar"], seed + 1, depth, examples)
    else:
        row["compiled_shared_execution"] = None
        row["compiled_shared_polar_execution"] = None
    return row


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/research/exp_033/metrics.json"))
    p.add_argument("--multiplicities", type=int, nargs="+", default=[3, 4])
    p.add_argument("--seeds", type=int, nargs="+", default=[130, 131, 132])
    p.add_argument("--sample-factors", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    p.add_argument("--noise", type=float, nargs="+", default=[0.05, 0.10])
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--gap-threshold", type=float, default=5.0)
    p.add_argument("--program-depth", type=int, default=256)
    p.add_argument("--examples", type=int, default=1024)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    rows = []
    for seed in args.seeds:
        for m in args.multiplicities:
            for factor in args.sample_factors:
                for noise in args.noise:
                    row = run_one(m, seed, factor, noise, args.ridge, args.gap_threshold, args.program_depth, args.examples)
                    rows.append(row)
                    c = row["compiler"]
                    ce = row["compiled_shared_execution"]
                    print(
                        f"seed={seed} m={m} n={row['samples_per_operator']} noise={noise:.2f} "
                        f"gap={c.get('best',{}).get('gap_ratio')} accepted={c.get('accepted')} "
                        f"raw={row['raw_identified_execution']['accuracy']:.4f} "
                        f"compiled={None if ce is None else round(ce['accuracy'],4)}"
                    )

    payload = {
        "experiment": EXPERIMENT_NAME,
        "protocol": {
            "compiler_receives_true_action_matrices": False,
            "compiler_receives_identity_codebook": False,
            "compiler_receives_discrete_state_labels": False,
            "compiler_receives_multiplicity": False,
            "system_identification": "ridge linear regression on continuous random state-transition pairs",
            "program_depth": args.program_depth,
            "examples": args.examples,
            "gap_threshold": args.gap_threshold,
        },
        "rows": rows,
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
