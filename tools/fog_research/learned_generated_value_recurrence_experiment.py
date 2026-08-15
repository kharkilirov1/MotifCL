#!/usr/bin/env python3
"""EXP-005: learned generated latent values reused recurrently on OOD transitions.

The one-step ALU is fit *only* on train operand pairs from EXP-004.  Evaluation
builds chains where every transition (current_value, operand) belongs to the
held-out pair split.  The model must therefore:

  z_t, E(b_t) -> z_{t+1}

and feed its own continuous z_{t+1} back into the same learned operator without
snapping/decoding it to a discrete identity between steps.

This directly tests the M4 boundary: create a new latent value absent from the
input pair table and reuse that generated latent as the next operand/state.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np

from operator_compatible_geometry_experiment import (
    EXPERIMENT_NAME as PARENT_EXPERIMENT,
    GeometryConfig,
    all_pairs,
    fit_ridge,
    full_features,
    local_features,
    make_fourier_codebook,
    make_random_codebook,
    pair_split,
)

EXPERIMENT_NAME = "exp_005_learned_generated_value_recurrence"
FeatureMode = Literal["local", "full"]


def vector_local_features(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] % 2:
        raise ValueError("a and b must be [B, 2H]")
    h = a.shape[1] // 2
    aa = a.reshape(-1, h, 2)
    bb = b.reshape(-1, h, 2)
    ca, sa = aa[..., 0], aa[..., 1]
    cb, sb = bb[..., 0], bb[..., 1]
    return np.stack((ca * cb, ca * sb, sa * cb, sa * sb), axis=-1).reshape(
        len(a), 4 * h
    )


def vector_full_features(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a[:, :, None] * b[:, None, :]).reshape(len(a), -1)


def vector_features(a: np.ndarray, b: np.ndarray, mode: FeatureMode) -> np.ndarray:
    return vector_local_features(a, b) if mode == "local" else vector_full_features(a, b)


def fit_operator(
    cfg: GeometryConfig,
    codebook: np.ndarray,
    *,
    split_seed: int,
    mode: FeatureMode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    pairs, target = all_pairs(cfg)
    train, test, digest = pair_split(cfg, pairs, split_seed)
    x = local_features(codebook, pairs[train]) if mode == "local" else full_features(codebook, pairs[train])
    w = fit_ridge(x, codebook[target[train]], cfg.ridge)
    return w, pairs, test, digest


def heldout_operand_choices(
    cfg: GeometryConfig, pairs: np.ndarray, heldout_indices: np.ndarray
) -> list[np.ndarray]:
    choices = []
    held = pairs[heldout_indices]
    for current in range(cfg.modulus):
        operands = held[held[:, 0] == current, 1]
        if len(operands) < 2:
            raise AssertionError("each current state needs multiple held-out operands")
        choices.append(operands)
    return choices


def make_heldout_chains(
    cfg: GeometryConfig,
    pairs: np.ndarray,
    heldout_indices: np.ndarray,
    *,
    depth: int,
    examples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    choices = heldout_operand_choices(cfg, pairs, heldout_indices)
    start = rng.integers(0, cfg.modulus, size=examples, dtype=np.int64)
    operands = np.empty((examples, depth), dtype=np.int64)
    target = start.copy()
    for t in range(depth):
        for i, current in enumerate(target.tolist()):
            pool = choices[current]
            operands[i, t] = pool[rng.integers(0, len(pool))]
        target = (target + operands[:, t]) % cfg.modulus
    return start, operands, target


def run_chain(
    cfg: GeometryConfig,
    codebook: np.ndarray,
    w: np.ndarray,
    *,
    mode: FeatureMode,
    start: np.ndarray,
    operands: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    z = codebook[start].copy()
    oracle = start.copy()
    history = []
    for t in range(operands.shape[1]):
        b = codebook[operands[:, t]]
        z = vector_features(z, b, mode) @ w
        z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
        oracle = (oracle + operands[:, t]) % cfg.modulus
        logits = z @ codebook.T
        pred = logits.argmax(axis=1)
        cosine = np.sum(z * codebook[oracle], axis=1)
        history.append(
            {
                "hop": t + 1,
                "accuracy": float(np.mean(pred == oracle)),
                "target_cosine_mean": float(np.mean(cosine)),
                "target_cosine_min": float(np.min(cosine)),
            }
        )
    return z, history


def evaluate_arm(
    cfg: GeometryConfig,
    *,
    geometry: Literal["fourier", "random"],
    mode: FeatureMode,
    split_seed: int,
    code_seed: int,
    depths: tuple[int, ...],
    examples: int,
) -> dict:
    codebook = make_fourier_codebook(cfg) if geometry == "fourier" else make_random_codebook(cfg, code_seed)
    w, pairs, heldout, split_digest = fit_operator(
        cfg, codebook, split_seed=split_seed, mode=mode
    )
    evals = []
    for depth in depths:
        start, operands, target = make_heldout_chains(
            cfg,
            pairs,
            heldout,
            depth=depth,
            examples=examples,
            seed=split_seed * 100000 + code_seed * 1000 + depth,
        )
        z, history = run_chain(
            cfg,
            codebook,
            w,
            mode=mode,
            start=start,
            operands=operands,
        )
        pred = (z @ codebook.T).argmax(axis=1)
        evals.append(
            {
                "depth": depth,
                "accuracy": float(np.mean(pred == target)),
                "hops": history,
            }
        )
    return {
        "geometry": geometry,
        "feature_mode": mode,
        "split_seed": split_seed,
        "code_seed": code_seed,
        "split_sha256": split_digest,
        "all_runtime_transitions_are_heldout_pairs": True,
        "evaluation": evals,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/research/exp_005/metrics.json"))
    p.add_argument("--split-seeds", type=int, nargs="+", default=[101, 202, 303])
    p.add_argument("--code-seeds", type=int, nargs="+", default=[11, 22, 33])
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--examples", type=int, default=2048)
    args = p.parse_args()
    cfg = GeometryConfig()
    depths = tuple(args.depths)
    arms = []
    for split_seed in args.split_seeds:
        arms.append(
            evaluate_arm(
                cfg,
                geometry="fourier",
                mode="local",
                split_seed=split_seed,
                code_seed=0,
                depths=depths,
                examples=args.examples,
            )
        )
        arms.append(
            evaluate_arm(
                cfg,
                geometry="fourier",
                mode="full",
                split_seed=split_seed,
                code_seed=0,
                depths=depths,
                examples=args.examples,
            )
        )
        for code_seed in args.code_seeds:
            arms.append(
                evaluate_arm(
                    cfg,
                    geometry="random",
                    mode="full",
                    split_seed=split_seed,
                    code_seed=code_seed,
                    depths=depths,
                    examples=args.examples,
                )
            )
    payload = {
        "experiment": EXPERIMENT_NAME,
        "parent_experiment": PARENT_EXPERIMENT,
        "config": asdict(cfg),
        "depths": depths,
        "examples_per_depth": args.examples,
        "arms": arms,
    }
    write_json(args.output, payload)
    for geometry, mode in (("fourier", "local"), ("fourier", "full"), ("random", "full")):
        rows = [a for a in arms if a["geometry"] == geometry and a["feature_mode"] == mode]
        print(f"{geometry}_{mode}")
        for depth in depths:
            vals = [next(e for e in r["evaluation"] if e["depth"] == depth)["accuracy"] for r in rows]
            print(depth, float(np.mean(vals)), float(np.min(vals)), float(np.max(vals)))


if __name__ == "__main__":
    main()
