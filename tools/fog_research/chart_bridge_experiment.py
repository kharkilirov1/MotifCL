#!/usr/bin/env python3
"""EXP-009: can a cheap linear bridge generalize between operator-specific charts?

We use the additive and multiplicative charts from EXP-008 over non-zero F_31
identities.  A linear map is fit on a hash-selected subset of identities and is
then evaluated on identities never used to fit the bridge.

The bridge has enough capacity to interpolate the seen identities exactly.  The
question is whether the relationship between charts is simple enough to extend
to unseen identities without identity lookup / decoding.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from multichart_operator_specificity_experiment import (
    MultiChartConfig,
    additive_codebook,
    multiplicative_codebook,
)

EXPERIMENT_NAME = "exp_009_chart_bridge"


def identity_split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    mask = []
    digest = hashlib.sha256()
    for i in range(n):
        raw = f"exp009|{seed}|{i}".encode()
        is_train = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % 10 < 7
        mask.append(is_train)
        digest.update(raw + (b"T" if is_train else b"E"))
    mask = np.asarray(mask, dtype=bool)
    train, test = np.flatnonzero(mask), np.flatnonzero(~mask)
    if len(train) < 2 or len(test) < 2:
        raise AssertionError("split too small")
    return train, test, digest.hexdigest()


def fit_linear(x: np.ndarray, y: np.ndarray, indices: np.ndarray, ridge: float) -> np.ndarray:
    xx = x[indices]
    yy = y[indices]
    return xx.T @ np.linalg.solve(xx @ xx.T + ridge * np.eye(len(indices)), yy)


def evaluate(x: np.ndarray, y: np.ndarray, indices: np.ndarray, w: np.ndarray) -> dict:
    z = x[indices] @ w
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    logits = z @ y.T
    pred = logits.argmax(axis=1)
    cosine = np.sum(z * y[indices], axis=1)
    return {
        "count": int(len(indices)),
        "identity_accuracy": float(np.mean(pred == indices)),
        "target_cosine_mean": float(np.mean(cosine)),
        "target_cosine_min": float(np.min(cosine)),
    }


def run_direction(
    source: np.ndarray,
    target: np.ndarray,
    *,
    seed: int,
    ridge: float,
    direction: str,
) -> dict:
    train, test, digest = identity_split(len(source), seed)
    w = fit_linear(source, target, train, ridge)
    # Expressivity ceiling: with all finite identities available a dense linear
    # map can interpolate the codebook relation (up to rank limitations).
    wall = fit_linear(source, target, np.arange(len(source)), ridge)
    return {
        "direction": direction,
        "split_seed": seed,
        "split_sha256": digest,
        "train": evaluate(source, target, train, w),
        "heldout_identities": evaluate(source, target, test, w),
        "all_identity_fit": evaluate(source, target, np.arange(len(source)), wall),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/research/exp_009/metrics.json"))
    p.add_argument("--split-seeds", type=int, nargs="+", default=[101, 202, 303, 404, 505, 606, 707])
    p.add_argument("--ridge", type=float, default=1e-8)
    args = p.parse_args()
    cfg = MultiChartConfig()
    add = additive_codebook(cfg)
    mul, primitive = multiplicative_codebook(cfg)
    arms = []
    for seed in args.split_seeds:
        arms.append(run_direction(add, mul, seed=seed, ridge=args.ridge, direction="add_to_mul"))
        arms.append(run_direction(mul, add, seed=seed, ridge=args.ridge, direction="mul_to_add"))
    payload = {
        "experiment": EXPERIMENT_NAME,
        "config": asdict(cfg),
        "primitive_root": primitive,
        "chance_identity_accuracy": 1.0 / (cfg.prime - 1),
        "arms": arms,
    }
    write_json(args.output, payload)
    for direction in ("add_to_mul", "mul_to_add"):
        rows = [r for r in arms if r["direction"] == direction]
        print(direction, json.dumps({
            "train_accuracy_mean": float(np.mean([r["train"]["identity_accuracy"] for r in rows])),
            "heldout_accuracy_mean": float(np.mean([r["heldout_identities"]["identity_accuracy"] for r in rows])),
            "heldout_accuracy_min": float(np.min([r["heldout_identities"]["identity_accuracy"] for r in rows])),
            "heldout_accuracy_max": float(np.max([r["heldout_identities"]["identity_accuracy"] for r in rows])),
            "heldout_cosine_mean": float(np.mean([r["heldout_identities"]["target_cosine_mean"] for r in rows])),
            "all_identity_fit_accuracy": float(np.mean([r["all_identity_fit"]["identity_accuracy"] for r in rows])),
        }, indent=2))


if __name__ == "__main__":
    main()
