#!/usr/bin/env python3
"""EXP-004: operator-compatible latent geometry vs memorization capacity.

Task: predict the latent code of (a+b) mod p from latent codes of a and b.
Train/test are disjoint in operand *pairs* while every identity can occur in both.
No intermediate text or lookup table is supplied.

We compare two code geometries:
  1. Fourier group codes, where addition has a local harmonic bilinear law.
  2. Frozen random codes with the same dimensionality.

And two bilinear feature classes:
  local: 4 products per paired 2D plane (4H features, strong structural bias)
  full:  full Kronecker product (D^2 features, enough capacity to interpolate)

The linear readout is solved in closed form by ridge regression.  This makes the
experiment deterministic, fast, and separates representation/inductive bias
from optimizer behavior.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np

EXPERIMENT_NAME = "exp_004_operator_compatible_geometry"
Geometry = Literal["fourier", "random"]
FeatureMode = Literal["local", "full"]


@dataclass(frozen=True)
class GeometryConfig:
    modulus: int = 31
    harmonics: int = 15
    train_fraction_tenths: int = 7
    ridge: float = 1e-8

    @property
    def d_model(self) -> int:
        return 2 * self.harmonics

    def validate(self) -> None:
        if self.modulus < 5 or self.modulus % 2 == 0:
            raise ValueError("modulus must be an odd integer >= 5")
        if not (1 <= self.harmonics <= (self.modulus - 1) // 2):
            raise ValueError("invalid harmonic count")
        if not (1 <= self.train_fraction_tenths <= 9):
            raise ValueError("train_fraction_tenths must be 1..9")
        if self.ridge <= 0:
            raise ValueError("ridge must be positive")


def make_fourier_codebook(cfg: GeometryConfig) -> np.ndarray:
    x = np.arange(cfg.modulus, dtype=np.float64)[:, None]
    h = np.arange(1, cfg.harmonics + 1, dtype=np.float64)[None, :]
    angle = 2.0 * np.pi * x * h / cfg.modulus
    code = np.stack((np.cos(angle), np.sin(angle)), axis=-1).reshape(
        cfg.modulus, cfg.d_model
    )
    return code / np.linalg.norm(code, axis=1, keepdims=True)


def make_random_codebook(cfg: GeometryConfig, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    code = rng.normal(size=(cfg.modulus, cfg.d_model))
    return code / np.linalg.norm(code, axis=1, keepdims=True)


def all_pairs(cfg: GeometryConfig) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.array(
        [(a, b) for a in range(cfg.modulus) for b in range(cfg.modulus)],
        dtype=np.int64,
    )
    target = (pairs[:, 0] + pairs[:, 1]) % cfg.modulus
    return pairs, target


def pair_split(
    cfg: GeometryConfig, pairs: np.ndarray, split_seed: int
) -> tuple[np.ndarray, np.ndarray, str]:
    mask = []
    digest = hashlib.sha256()
    for a, b in pairs.tolist():
        raw = f"exp004|{split_seed}|{a}|{b}".encode()
        h = hashlib.sha256(raw).digest()
        bucket = int.from_bytes(h[:8], "big") % 10
        is_train = bucket < cfg.train_fraction_tenths
        mask.append(is_train)
        digest.update(raw + (b"T" if is_train else b"E"))
    mask = np.asarray(mask, dtype=bool)
    train = np.flatnonzero(mask)
    test = np.flatnonzero(~mask)
    # Every scalar identity must appear in train and test as both left and right
    # operands; otherwise pair generalization could be confounded with unseen IDs.
    for col in (0, 1):
        if set(pairs[train, col].tolist()) != set(range(cfg.modulus)):
            raise AssertionError("train split must cover every operand identity")
        if set(pairs[test, col].tolist()) != set(range(cfg.modulus)):
            raise AssertionError("test split must cover every operand identity")
    return train, test, digest.hexdigest()


def local_features(codebook: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    h = codebook.shape[1] // 2
    a = codebook[pairs[:, 0]].reshape(-1, h, 2)
    b = codebook[pairs[:, 1]].reshape(-1, h, 2)
    ca, sa = a[..., 0], a[..., 1]
    cb, sb = b[..., 0], b[..., 1]
    return np.stack((ca * cb, ca * sb, sa * cb, sa * sb), axis=-1).reshape(
        len(pairs), 4 * h
    )


def full_features(codebook: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    a = codebook[pairs[:, 0]]
    b = codebook[pairs[:, 1]]
    return (a[:, :, None] * b[:, None, :]).reshape(len(pairs), -1)


def features(codebook: np.ndarray, pairs: np.ndarray, mode: FeatureMode) -> np.ndarray:
    if mode == "local":
        return local_features(codebook, pairs)
    if mode == "full":
        return full_features(codebook, pairs)
    raise ValueError(mode)


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    # Primal is faster for local features; dual avoids a 900x900 inversion for
    # the overparameterized full Kronecker arm.
    n, f = x.shape
    if f <= n:
        return np.linalg.solve(x.T @ x + ridge * np.eye(f), x.T @ y)
    k = x @ x.T + ridge * np.eye(n)
    alpha = np.linalg.solve(k, y)
    return x.T @ alpha


def evaluate(
    codebook: np.ndarray,
    pairs: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    w: np.ndarray,
    mode: FeatureMode,
) -> dict:
    z = features(codebook, pairs[indices], mode) @ w
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / np.maximum(norms, 1e-12)
    target_code = codebook[target[indices]]
    logits = z @ codebook.T
    pred = logits.argmax(axis=1)
    cosine = np.sum(z * target_code, axis=1)
    margin_sorted = np.partition(logits, -2, axis=1)[:, -2:]
    margin = margin_sorted[:, 1] - margin_sorted[:, 0]
    return {
        "count": int(len(indices)),
        "accuracy": float(np.mean(pred == target[indices])),
        "target_cosine_mean": float(np.mean(cosine)),
        "target_cosine_min": float(np.min(cosine)),
        "top1_margin_mean": float(np.mean(margin)),
    }


def run_one(
    cfg: GeometryConfig,
    *,
    geometry: Geometry,
    mode: FeatureMode,
    split_seed: int,
    code_seed: int,
) -> dict:
    pairs, target = all_pairs(cfg)
    train, test, split_digest = pair_split(cfg, pairs, split_seed)
    codebook = (
        make_fourier_codebook(cfg)
        if geometry == "fourier"
        else make_random_codebook(cfg, code_seed)
    )
    x_train = features(codebook, pairs[train], mode)
    y_train = codebook[target[train]]
    w = fit_ridge(x_train, y_train, cfg.ridge)
    return {
        "geometry": geometry,
        "feature_mode": mode,
        "feature_dim": int(x_train.shape[1]),
        "readout_parameters": int(w.size),
        "split_seed": split_seed,
        "code_seed": code_seed,
        "split_sha256": split_digest,
        "train": evaluate(codebook, pairs, target, train, w, mode),
        "heldout_pairs": evaluate(codebook, pairs, target, test, w, mode),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/research/exp_004/metrics.json"))
    p.add_argument("--modulus", type=int, default=31)
    p.add_argument("--harmonics", type=int, default=15)
    p.add_argument("--ridge", type=float, default=1e-8)
    p.add_argument("--split-seeds", type=int, nargs="+", default=[101, 202, 303])
    p.add_argument("--code-seeds", type=int, nargs="+", default=[11, 22, 33])
    args = p.parse_args()
    cfg = GeometryConfig(args.modulus, args.harmonics, 7, args.ridge)
    cfg.validate()
    arms = []
    for split_seed in args.split_seeds:
        # Fourier is deterministic; code_seed is metadata only.
        for geometry, mode, code_seed in (
            ("fourier", "local", 0),
            ("fourier", "full", 0),
        ):
            arms.append(
                run_one(
                    cfg,
                    geometry=geometry,  # type: ignore[arg-type]
                    mode=mode,  # type: ignore[arg-type]
                    split_seed=split_seed,
                    code_seed=code_seed,
                )
            )
        for code_seed in args.code_seeds:
            for mode in ("local", "full"):
                arms.append(
                    run_one(
                        cfg,
                        geometry="random",
                        mode=mode,  # type: ignore[arg-type]
                        split_seed=split_seed,
                        code_seed=code_seed,
                    )
                )
    payload = {
        "experiment": EXPERIMENT_NAME,
        "config": asdict(cfg),
        "chance_accuracy": 1.0 / cfg.modulus,
        "arms": arms,
    }
    write_json(args.output, payload)
    summary = {}
    for geometry, mode in (("fourier", "local"), ("fourier", "full"), ("random", "local"), ("random", "full")):
        rows = [r for r in arms if r["geometry"] == geometry and r["feature_mode"] == mode]
        summary[f"{geometry}_{mode}"] = {
            "runs": len(rows),
            "train_accuracy_mean": float(np.mean([r["train"]["accuracy"] for r in rows])),
            "heldout_accuracy_mean": float(np.mean([r["heldout_pairs"]["accuracy"] for r in rows])),
            "heldout_accuracy_min": float(np.min([r["heldout_pairs"]["accuracy"] for r in rows])),
            "heldout_accuracy_max": float(np.max([r["heldout_pairs"]["accuracy"] for r in rows])),
            "heldout_cosine_mean": float(np.mean([r["heldout_pairs"]["target_cosine_mean"] for r in rows])),
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
