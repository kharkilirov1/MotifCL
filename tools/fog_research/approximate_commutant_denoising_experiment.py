#!/usr/bin/env python3
"""EXP-032: approximate-commutant discovery and repeated-block denoising.

EXP-031 used exact shared irreducible copies.  Real learned actions will only
approximately share a commutant.  This experiment perturbs the hidden dense
operators, projects the perturbation to the nearest orthogonal matrix, and asks
whether a semantics-free compiler can still discover multiplicity from the
*singular-value gap* of the joint commutator operator.

The compiler never receives the true multiplicity.  Candidate repeated-irrep
models correspond to square approximate-commutant dimensions m^2 compatible
with the total latent width.  A candidate is accepted only when the singular
value gap after that subspace exceeds a fixed threshold.

After discovery, approximate invariant copies are aligned by intertwiners and
the corresponding action blocks are averaged.  We report two compiled arms:

  shared_average : repeated-block structure only;
  shared_polar   : same, then nearest-orthogonal projection of the shared block.

The latter is a generic near-isometry prior, not an S3-specific law.  Generic
random orthogonal operator pairs are negative controls and should not exhibit a
square low-singular-value cluster.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from joint_commutant_block_compiler_experiment import (
    _block_slices,
    _commutator_constraint,
    _intertwiner,
    _orthogonal,
    _separator_from_commutant,
    make_problem,
)

EXPERIMENT_NAME = "exp_032_approximate_commutant_denoising"


def polar(W: torch.Tensor) -> torch.Tensor:
    u, _s, vh = torch.linalg.svd(W)
    return u @ vh


def perturb_orthogonal(W: torch.Tensor, relative_noise: float, seed: int) -> torch.Tensor:
    if relative_noise == 0.0:
        return W.clone()
    g = torch.Generator().manual_seed(seed)
    N = torch.randn(W.shape, generator=g, dtype=W.dtype)
    N = N / N.norm().clamp_min(1e-30) * W.norm() * relative_noise
    return polar(W + N)


def infer_approximate_commutant(actions: list[torch.Tensor], gap_threshold: float = 5.0) -> dict:
    d = actions[0].size(0)
    C = _commutator_constraint(actions)
    _u, s_desc, vh = torch.linalg.svd(C, full_matrices=True)
    s_asc = s_desc.flip(0)
    candidates = []
    for m in range(2, d + 1):
        if d % m != 0:
            continue
        block_dim = d // m
        if block_dim <= 1:
            continue
        k = m * m
        if k >= s_asc.numel():
            continue
        low = float(s_asc[k - 1])
        high = float(s_asc[k])
        gap = high / max(low, 1e-12)
        candidates.append(
            {
                "multiplicity": m,
                "block_dim": block_dim,
                "approx_commutant_dim": k,
                "low_cluster_max": low,
                "high_cluster_min": high,
                "gap_ratio": gap,
            }
        )
    if not candidates:
        return {"accepted": False, "reason": "no compatible repeated-irrep candidate", "candidates": []}
    best = max(candidates, key=lambda x: x["gap_ratio"])
    accepted = best["gap_ratio"] >= gap_threshold
    out = {
        "accepted": accepted,
        "gap_threshold": gap_threshold,
        "best": best,
        "candidates": candidates,
        "smallest_singular_values": [float(x) for x in s_asc[: min(24, s_asc.numel())]],
    }
    if accepted:
        k = best["approx_commutant_dim"]
        out["basis"] = [v.reshape(d, d) for v in vh[-k:]]
    return out


def compile_approximate_repeated_irrep(actions: list[torch.Tensor], seed: int, gap_threshold: float = 5.0) -> dict:
    inferred = infer_approximate_commutant(actions, gap_threshold=gap_threshold)
    base = {k: v for k, v in inferred.items() if k != "basis"}
    if not inferred["accepted"]:
        return base
    best = inferred["best"]
    m = best["multiplicity"]
    b = best["block_dim"]
    basis = inferred["basis"]
    sep = _separator_from_commutant(basis, b, m, seed)
    U = sep["U"]
    block_actions = [U.T @ W @ U for W in actions]
    slices = _block_slices(m, b)
    refs = [W[slices[0], slices[0]] for W in block_actions]
    Qs = [torch.eye(b, dtype=torch.float64)]
    intertwiner_residuals = [0.0]
    for j in range(1, m):
        cur = [W[slices[j], slices[j]] for W in block_actions]
        it = _intertwiner(refs, cur)
        Qs.append(it["Q"])
        intertwiner_residuals.append(it["residual"])
    S = torch.block_diag(*Qs)
    G = U @ S
    aligned = [G.T @ W @ G for W in actions]

    shared = []
    shared_polar = []
    block_spread = []
    for W in aligned:
        blocks = torch.stack([W[sl, sl] for sl in slices], dim=0)
        avg = blocks.mean(dim=0)
        shared.append(avg)
        shared_polar.append(polar(avg))
        spread = ((blocks - avg).square().sum(dim=(-2, -1)).sqrt() / avg.norm().clamp_min(1e-30)).mean()
        block_spread.append(float(spread))

    rec_shared = [G @ torch.block_diag(*([R] * m)) @ G.T for R in shared]
    rec_polar = [G @ torch.block_diag(*([R] * m)) @ G.T for R in shared_polar]
    noisy_fit_shared = [float((R - W).norm() / W.norm().clamp_min(1e-30)) for R, W in zip(rec_shared, actions)]
    noisy_fit_polar = [float((R - W).norm() / W.norm().clamp_min(1e-30)) for R, W in zip(rec_polar, actions)]
    base.update(
        {
            "separator_within_eigen_spread": sep["within_spread"],
            "separator_min_between_gap": sep["min_between_gap"],
            "intertwiner_residuals": intertwiner_residuals,
            "aligned_block_relative_spread": block_spread,
            "noisy_operator_fit_shared": noisy_fit_shared,
            "noisy_operator_fit_shared_polar": noisy_fit_polar,
            "G": G,
            "shared_blocks": shared,
            "shared_polar_blocks": shared_polar,
            "reconstructed_shared": rec_shared,
            "reconstructed_shared_polar": rec_polar,
        }
    )
    return base


def _decode(codes: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return (F.normalize(z, dim=-1) @ F.normalize(codes, dim=-1).T).argmax(-1)


@torch.inference_mode()
def execute(problem: dict, actions: list[torch.Tensor], seed: int, depth: int, examples: int) -> dict:
    codes = problem["codes"]
    g = torch.Generator().manual_seed(seed + 321_000)
    idx = torch.randint(codes.size(0), (examples,), generator=g)
    z = codes[idx]
    target = idx.clone()
    for _ in range(depth):
        kind = torch.randint(2, (examples,), generator=g)
        za = F.normalize(z @ actions[0].T, dim=-1)
        zb = F.normalize(z @ actions[1].T, dim=-1)
        z = torch.where(kind[:, None].eq(0), za, zb)
        channel = torch.div(target, 3, rounding_mode="floor")
        symbol = target % 3
        symbol = torch.where(kind.eq(0), (symbol + 1) % 3, (-symbol) % 3)
        target = channel * 3 + symbol
    pred = _decode(codes, z)
    return {
        "accuracy": float(pred.eq(target).float().mean()),
        "target_cosine": float(F.cosine_similarity(z, codes[target], dim=-1).mean()),
    }


def _jsonable(comp: dict) -> dict:
    hidden = {"G", "shared_blocks", "shared_polar_blocks", "reconstructed_shared", "reconstructed_shared_polar"}
    return {k: v for k, v in comp.items() if k not in hidden}


def run_noise(problem: dict, seed: int, noise: float, depth: int, examples: int, gap_threshold: float) -> dict:
    raw = [
        perturb_orthogonal(problem["A"], noise, seed + 1),
        perturb_orthogonal(problem["B"], noise, seed + 2),
    ]
    comp = compile_approximate_repeated_irrep(raw, seed, gap_threshold=gap_threshold)
    row = {
        "relative_noise": noise,
        "raw_execution": execute(problem, raw, seed, depth, examples),
        "compiler": _jsonable(comp),
    }
    if comp.get("accepted"):
        row["shared_average_execution"] = execute(problem, comp["reconstructed_shared"], seed, depth, examples)
        row["shared_polar_execution"] = execute(problem, comp["reconstructed_shared_polar"], seed, depth, examples)
    else:
        row["shared_average_execution"] = None
        row["shared_polar_execution"] = None
    return row


def random_pair_control(d: int, seed: int, gap_threshold: float) -> dict:
    actions = [_orthogonal(seed + 1, d), _orthogonal(seed + 2, d)]
    comp = infer_approximate_commutant(actions, gap_threshold=gap_threshold)
    return {
        "seed": seed,
        "dimension": d,
        "accepted": comp["accepted"],
        "best": comp.get("best"),
        "candidates": comp.get("candidates"),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/research/exp_032/metrics.json"))
    p.add_argument("--multiplicities", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--seeds", type=int, nargs="+", default=[120, 121, 122])
    p.add_argument("--noise", type=float, nargs="+", default=[0.03, 0.05, 0.10, 0.15])
    p.add_argument("--gap-threshold", type=float, default=5.0)
    p.add_argument("--program-depth", type=int, default=256)
    p.add_argument("--examples", type=int, default=1024)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    rows = []
    controls = []
    for seed in args.seeds:
        for m in args.multiplicities:
            problem = make_problem(m, seed)
            for noise in args.noise:
                row = run_noise(problem, seed + int(noise * 10_000), noise, args.program_depth, args.examples, args.gap_threshold)
                row.update({"seed": seed, "multiplicity": m, "dimension": 2 * m})
                rows.append(row)
                c = row["compiler"]
                sa = row["shared_average_execution"]
                sp = row["shared_polar_execution"]
                print(
                    f"seed={seed} m={m} noise={noise:.3f} gap={c.get('best',{}).get('gap_ratio')} "
                    f"accepted={c.get('accepted')} raw={row['raw_execution']['accuracy']:.4f} "
                    f"shared={None if sa is None else round(sa['accuracy'],4)} "
                    f"polar={None if sp is None else round(sp['accuracy'],4)}"
                )
            controls.append(random_pair_control(2 * m, seed + 9000 + m, args.gap_threshold))

    payload = {
        "experiment": EXPERIMENT_NAME,
        "protocol": {
            "multiplicity_given_to_compiler": False,
            "candidate_rule": "square approximate-commutant dimension m^2 compatible with total width",
            "acceptance_gap_threshold": args.gap_threshold,
            "noise_model": "relative dense Gaussian perturbation followed by nearest orthogonal projection",
            "shared_average_uses_group_specific_law": False,
            "shared_polar_prior": "generic near-isometry only",
            "program_depth": args.program_depth,
            "examples": args.examples,
        },
        "rows": rows,
        "generic_random_pair_controls": controls,
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
