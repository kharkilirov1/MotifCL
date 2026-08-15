#!/usr/bin/env python3
"""EXP-006: latent program counter with unique HALT and OOD program lengths.

The PC successor is learned only from transitions 0->1, 1->2, 2->3, 3->4.
The machine then executes instruction memories containing exactly one HALT.
Positions after HALT contain real distractor operations, never extra HALTs.

No external program index is used during execution: the current continuous PC
register addresses instruction memory by cosine compare/select.  The Python loop
is only a maximum safety cap.

This is a control-plane micro-gate, not terminal-only training: successor fitting
is deliberately supervised so we can isolate whether the learned transition law
generalizes beyond the finite prefix it observed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np

EXPERIMENT_NAME = "exp_006_latent_program_counter"
Geometry = Literal["fourier", "random"]
Operator = Literal["local", "full"]
HALT = 4
N_OPS = 5


@dataclass(frozen=True)
class PCConfig:
    pc_modulus: int = 13
    pc_harmonics: int = 6
    value_modulus: int = 13
    train_last_source: int = 3
    ridge: float = 1e-8

    @property
    def pc_dim(self) -> int:
        return 2 * self.pc_harmonics


def fourier_codebook(n: int, harmonics: int) -> np.ndarray:
    x = np.arange(n, dtype=np.float64)[:, None]
    h = np.arange(1, harmonics + 1, dtype=np.float64)[None, :]
    angle = 2 * np.pi * x * h / n
    code = np.stack((np.cos(angle), np.sin(angle)), axis=-1).reshape(n, 2 * harmonics)
    return code / np.linalg.norm(code, axis=1, keepdims=True)


def random_codebook(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    code = rng.normal(size=(n, d))
    return code / np.linalg.norm(code, axis=1, keepdims=True)


def fit_local_successor(code: np.ndarray, last_source: int, ridge: float) -> np.ndarray:
    h = code.shape[1] // 2
    w = np.zeros((2 * h, 2 * h), dtype=np.float64)
    src = code[np.arange(last_source + 1)].reshape(-1, h, 2)
    dst = code[np.arange(1, last_source + 2)].reshape(-1, h, 2)
    for k in range(h):
        x = src[:, k, :]
        y = dst[:, k, :]
        block = np.linalg.solve(x.T @ x + ridge * np.eye(2), x.T @ y)
        w[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] = block
    return w


def fit_full_successor(code: np.ndarray, last_source: int, ridge: float) -> np.ndarray:
    x = code[np.arange(last_source + 1)]
    y = code[np.arange(1, last_source + 2)]
    # Minimum-norm dual ridge solution; with D >> four examples this can
    # interpolate the observed prefix without defining a shared successor law.
    return x.T @ np.linalg.solve(x @ x.T + ridge * np.eye(len(x)), y)


def fit_successor(code: np.ndarray, cfg: PCConfig, op: Operator) -> np.ndarray:
    return (
        fit_local_successor(code, cfg.train_last_source, cfg.ridge)
        if op == "local"
        else fit_full_successor(code, cfg.train_last_source, cfg.ridge)
    )


def successor_metrics(code: np.ndarray, w: np.ndarray, cfg: PCConfig) -> dict:
    z = code @ w
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    pred = (z @ code.T).argmax(axis=1)
    expected = (np.arange(cfg.pc_modulus) + 1) % cfg.pc_modulus
    seen = np.arange(cfg.train_last_source + 1)
    unseen = np.arange(cfg.train_last_source + 1, cfg.pc_modulus - 1)
    return {
        "seen_transition_accuracy": float(np.mean(pred[seen] == expected[seen])),
        "unseen_transition_accuracy": float(np.mean(pred[unseen] == expected[unseen])),
        "predicted_successor": pred.tolist(),
    }


def value_transition_table(modulus: int) -> np.ndarray:
    x = np.arange(modulus)
    table = np.empty((4, modulus), dtype=np.int64)
    table[0] = (x + 1) % modulus
    table[1] = (x + 3) % modulus
    table[2] = (2 * x) % modulus
    table[3] = (-x) % modulus
    return table


def make_programs(
    cfg: PCConfig,
    *,
    min_length: int,
    max_length: int,
    examples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    length = rng.integers(min_length, max_length + 1, size=examples, dtype=np.int64)
    program = rng.integers(0, 4, size=(examples, cfg.pc_modulus), dtype=np.int64)
    program[np.arange(examples), length] = HALT
    # There is exactly one HALT because all other cells were sampled from 0..3.
    assert np.all(np.sum(program == HALT, axis=1) == 1)
    start = rng.integers(0, cfg.value_modulus, size=examples, dtype=np.int64)
    table = value_transition_table(cfg.value_modulus)
    target = start.copy()
    for t in range(max_length):
        active = t < length
        if np.any(active):
            target[active] = table[program[active, t], target[active]]
    return length, program, start, target


def value_matrices(cfg: PCConfig) -> np.ndarray:
    table = value_transition_table(cfg.value_modulus)
    mats = np.zeros((4, cfg.value_modulus, cfg.value_modulus), dtype=np.float64)
    for op in range(4):
        mats[op, np.arange(cfg.value_modulus), table[op]] = 1.0
    return mats


def execute(
    cfg: PCConfig,
    pc_code: np.ndarray,
    successor: np.ndarray,
    length: np.ndarray,
    program: np.ndarray,
    start: np.ndarray,
) -> dict:
    b = len(start)
    pc = np.repeat(pc_code[0][None, :], b, axis=0)
    value = np.eye(cfg.value_modulus, dtype=np.float64)[start]
    mats = value_matrices(cfg)
    halted = np.zeros(b, dtype=bool)
    halt_step = np.full(b, -1, dtype=np.int64)
    halt_position = np.full(b, -1, dtype=np.int64)
    trajectory_correct = []

    for step in range(cfg.pc_modulus + 2):
        pos = (pc @ pc_code.T).argmax(axis=1)
        expected_pos = np.minimum(step, length)
        trajectory_correct.append(float(np.mean(pos == expected_pos)))
        opcode = program[np.arange(b), pos]
        new_halt = (~halted) & (opcode == HALT)
        halt_step[new_halt] = step
        halt_position[new_halt] = pos[new_halt]
        halted |= new_halt
        active = ~halted
        if not np.any(active):
            break
        # Latent value register: one-hot distribution transformed by a fixed
        # permutation matrix selected by the fetched instruction.
        for op in range(4):
            mask = active & (opcode == op)
            if np.any(mask):
                value[mask] = value[mask] @ mats[op]
        pc[active] = pc[active] @ successor
        pc[active] /= np.maximum(np.linalg.norm(pc[active], axis=1, keepdims=True), 1e-12)

    return {
        "value_prediction": value.argmax(axis=1),
        "halted": halted,
        "halt_step": halt_step,
        "halt_position": halt_position,
        "trajectory_pc_accuracy": trajectory_correct,
    }


def evaluate_programs(
    cfg: PCConfig,
    pc_code: np.ndarray,
    successor: np.ndarray,
    *,
    min_length: int,
    max_length: int,
    examples: int,
    seed: int,
) -> dict:
    length, program, start, target = make_programs(
        cfg,
        min_length=min_length,
        max_length=max_length,
        examples=examples,
        seed=seed,
    )
    out = execute(cfg, pc_code, successor, length, program, start)
    return {
        "length_range": [min_length, max_length],
        "examples": examples,
        "value_accuracy": float(np.mean(out["value_prediction"] == target)),
        "halt_rate": float(np.mean(out["halted"])),
        "halt_step_accuracy": float(np.mean(out["halt_step"] == length)),
        "halt_position_accuracy": float(np.mean(out["halt_position"] == length)),
        "trajectory_pc_accuracy": out["trajectory_pc_accuracy"],
    }


def run_arm(
    cfg: PCConfig,
    *,
    geometry: Geometry,
    operator: Operator,
    code_seed: int,
    examples: int,
    eval_seed: int,
) -> dict:
    code = (
        fourier_codebook(cfg.pc_modulus, cfg.pc_harmonics)
        if geometry == "fourier"
        else random_codebook(cfg.pc_modulus, cfg.pc_dim, code_seed)
    )
    w = fit_successor(code, cfg, operator)
    return {
        "geometry": geometry,
        "operator": operator,
        "code_seed": code_seed,
        "successor": successor_metrics(code, w, cfg),
        "id_programs": evaluate_programs(
            cfg, code, w, min_length=1, max_length=4, examples=examples, seed=eval_seed
        ),
        "ood_programs": evaluate_programs(
            cfg, code, w, min_length=5, max_length=10, examples=examples, seed=eval_seed + 1
        ),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/research/exp_006/metrics.json"))
    p.add_argument("--examples", type=int, default=4096)
    p.add_argument("--code-seeds", type=int, nargs="+", default=[11, 22, 33])
    args = p.parse_args()
    cfg = PCConfig()
    arms = []
    arms.append(run_arm(cfg, geometry="fourier", operator="local", code_seed=0, examples=args.examples, eval_seed=8001))
    arms.append(run_arm(cfg, geometry="fourier", operator="full", code_seed=0, examples=args.examples, eval_seed=8001))
    for seed in args.code_seeds:
        arms.append(run_arm(cfg, geometry="random", operator="full", code_seed=seed, examples=args.examples, eval_seed=8001 + seed))
    payload = {"experiment": EXPERIMENT_NAME, "config": asdict(cfg), "unique_halt": True, "post_halt_distractors": True, "arms": arms}
    write_json(args.output, payload)
    summary = {}
    for geometry, operator in (("fourier", "local"), ("fourier", "full"), ("random", "full")):
        rows = [r for r in arms if r["geometry"] == geometry and r["operator"] == operator]
        summary[f"{geometry}_{operator}"] = {
            "runs": len(rows),
            "seen_successor": float(np.mean([r["successor"]["seen_transition_accuracy"] for r in rows])),
            "unseen_successor": float(np.mean([r["successor"]["unseen_transition_accuracy"] for r in rows])),
            "id_value": float(np.mean([r["id_programs"]["value_accuracy"] for r in rows])),
            "id_halt": float(np.mean([r["id_programs"]["halt_step_accuracy"] for r in rows])),
            "ood_value": float(np.mean([r["ood_programs"]["value_accuracy"] for r in rows])),
            "ood_halt": float(np.mean([r["ood_programs"]["halt_step_accuracy"] for r in rows])),
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
