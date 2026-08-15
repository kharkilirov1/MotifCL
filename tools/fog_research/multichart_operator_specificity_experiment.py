#!/usr/bin/env python3
"""EXP-008: operator specificity of latent coordinate charts.

Over the non-zero elements of F_31 we compare two 30D charts:

- additive Fourier chart: characters of x in (F_31,+);
- multiplicative log-Fourier chart: characters of log_g(x) in F_31^*.

The same local 60-feature bilinear operator class is fit for two tasks:
addition (excluding the zero-output pairs so the non-zero domain stays closed
for evaluation) and multiplication.

Prediction: each chart makes its native group operation exactly local while the
other operation remains non-local.  Full Kronecker controls test whether brute
bilinear capacity can memorize the wrong chart without recovering the true law.
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

EXPERIMENT_NAME = "exp_008_multichart_operator_specificity"
Chart = Literal["additive", "multiplicative"]
Operation = Literal["add", "mul"]
FeatureMode = Literal["local", "full"]


@dataclass(frozen=True)
class MultiChartConfig:
    prime: int = 31
    harmonics: int = 15
    ridge_local: float = 1e-8
    ridge_full: float = 1e-6

    @property
    def d_model(self) -> int:
        return 2 * self.harmonics


def primitive_root(p: int) -> int:
    target = set(range(1, p))
    for g in range(2, p):
        if {pow(g, k, p) for k in range(p - 1)} == target:
            return g
    raise ValueError("no primitive root found")


def additive_codebook(cfg: MultiChartConfig) -> np.ndarray:
    values = np.arange(1, cfg.prime, dtype=np.float64)[:, None]
    h = np.arange(1, cfg.harmonics + 1, dtype=np.float64)[None, :]
    angle = 2 * np.pi * values * h / cfg.prime
    code = np.stack((np.cos(angle), np.sin(angle)), axis=-1).reshape(
        cfg.prime - 1, cfg.d_model
    )
    return code / np.linalg.norm(code, axis=1, keepdims=True)


def multiplicative_codebook(cfg: MultiChartConfig) -> tuple[np.ndarray, int]:
    g = primitive_root(cfg.prime)
    log = {pow(g, k, cfg.prime): k for k in range(cfg.prime - 1)}
    values = np.arange(1, cfg.prime, dtype=np.int64)
    exponent = np.array([log[int(x)] for x in values], dtype=np.float64)[:, None]
    h = np.arange(1, cfg.harmonics + 1, dtype=np.float64)[None, :]
    angle = 2 * np.pi * exponent * h / (cfg.prime - 1)
    code = np.stack((np.cos(angle), np.sin(angle)), axis=-1).reshape(
        cfg.prime - 1, cfg.d_model
    )
    return code / np.linalg.norm(code, axis=1, keepdims=True), g


def task_pairs(cfg: MultiChartConfig, operation: Operation) -> tuple[np.ndarray, np.ndarray]:
    values = np.arange(1, cfg.prime, dtype=np.int64)
    pairs = []
    target = []
    for ia, a in enumerate(values.tolist()):
        for ib, b in enumerate(values.tolist()):
            y = (a + b) % cfg.prime if operation == "add" else (a * b) % cfg.prime
            if y == 0:
                # Multiplicative chart has no zero identity; for the additive
                # cross-chart task we lock evaluation to the non-zero codomain.
                continue
            pairs.append((ia, ib))
            target.append(y - 1)
    return np.asarray(pairs, dtype=np.int64), np.asarray(target, dtype=np.int64)


def pair_split(pairs: np.ndarray, split_seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    mask = []
    digest = hashlib.sha256()
    for a, b in pairs.tolist():
        raw = f"exp008|{split_seed}|{a}|{b}".encode()
        is_train = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % 10 < 7
        mask.append(is_train)
        digest.update(raw + (b"T" if is_train else b"E"))
    mask = np.asarray(mask, dtype=bool)
    return np.flatnonzero(mask), np.flatnonzero(~mask), digest.hexdigest()


def local_features(code: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    h = code.shape[1] // 2
    a = code[pairs[:, 0]].reshape(-1, h, 2)
    b = code[pairs[:, 1]].reshape(-1, h, 2)
    ca, sa = a[..., 0], a[..., 1]
    cb, sb = b[..., 0], b[..., 1]
    return np.stack((ca * cb, ca * sb, sa * cb, sa * sb), axis=-1).reshape(
        len(pairs), 4 * h
    )


def full_features(code: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    a = code[pairs[:, 0]]
    b = code[pairs[:, 1]]
    return (a[:, :, None] * b[:, None, :]).reshape(len(pairs), -1)


def feat(code: np.ndarray, pairs: np.ndarray, mode: FeatureMode) -> np.ndarray:
    return local_features(code, pairs) if mode == "local" else full_features(code, pairs)


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    n, f = x.shape
    if f <= n:
        return np.linalg.solve(x.T @ x + ridge * np.eye(f), x.T @ y)
    return x.T @ np.linalg.solve(x @ x.T + ridge * np.eye(n), y)


def evaluate(
    code: np.ndarray,
    pairs: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    w: np.ndarray,
    mode: FeatureMode,
) -> dict:
    z = feat(code, pairs[indices], mode) @ w
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    pred = (z @ code.T).argmax(axis=1)
    cosine = np.sum(z * code[target[indices]], axis=1)
    return {
        "count": int(len(indices)),
        "accuracy": float(np.mean(pred == target[indices])),
        "target_cosine_mean": float(np.mean(cosine)),
    }


def run_arm(
    cfg: MultiChartConfig,
    *,
    chart: Chart,
    operation: Operation,
    mode: FeatureMode,
    split_seed: int,
) -> dict:
    if chart == "additive":
        code = additive_codebook(cfg)
        generator = None
    else:
        code, generator = multiplicative_codebook(cfg)
    pairs, target = task_pairs(cfg, operation)
    train, test, digest = pair_split(pairs, split_seed)
    x = feat(code, pairs[train], mode)
    ridge = cfg.ridge_local if mode == "local" else cfg.ridge_full
    w = fit_ridge(x, code[target[train]], ridge)
    return {
        "chart": chart,
        "operation": operation,
        "feature_mode": mode,
        "split_seed": split_seed,
        "primitive_root": generator,
        "feature_dim": int(x.shape[1]),
        "split_sha256": digest,
        "train": evaluate(code, pairs, target, train, w, mode),
        "heldout_pairs": evaluate(code, pairs, target, test, w, mode),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/research/exp_008/metrics.json"))
    p.add_argument("--split-seeds", type=int, nargs="+", default=[101, 202, 303])
    args = p.parse_args()
    cfg = MultiChartConfig()
    arms = []
    for split_seed in args.split_seeds:
        for chart in ("additive", "multiplicative"):
            for operation in ("add", "mul"):
                for mode in ("local", "full"):
                    arms.append(
                        run_arm(
                            cfg,
                            chart=chart,  # type: ignore[arg-type]
                            operation=operation,  # type: ignore[arg-type]
                            mode=mode,  # type: ignore[arg-type]
                            split_seed=split_seed,
                        )
                    )
    payload = {
        "experiment": EXPERIMENT_NAME,
        "config": asdict(cfg),
        "domain": "nonzero elements of F_31; zero-output addition pairs excluded",
        "arms": arms,
    }
    write_json(args.output, payload)
    summary = {}
    for chart in ("additive", "multiplicative"):
        for operation in ("add", "mul"):
            for mode in ("local", "full"):
                rows = [
                    r for r in arms
                    if r["chart"] == chart and r["operation"] == operation and r["feature_mode"] == mode
                ]
                summary[f"{chart}_{operation}_{mode}"] = {
                    "train_accuracy": float(np.mean([r["train"]["accuracy"] for r in rows])),
                    "heldout_accuracy": float(np.mean([r["heldout_pairs"]["accuracy"] for r in rows])),
                    "heldout_cosine": float(np.mean([r["heldout_pairs"]["target_cosine_mean"] for r in rows])),
                }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
