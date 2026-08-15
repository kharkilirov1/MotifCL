#!/usr/bin/env python3
"""EXP-007: integrated latent register machine on OOD programs.

Combines two independently learned micro-laws:

1. data plane: the one-step modular-add ALU from EXP-004/005;
2. control plane: the latent PC successor from EXP-006.

Programs contain ADD(operand) instructions and exactly one HALT.  The program
length is not passed separately to the machine.  The current continuous PC
register addresses instruction memory.  The current continuous value register
and fetched operand code are passed through the learned ALU.  Neither register
is snapped/decoded between transitions.

OOD evaluation uses program lengths 5..10 even though the PC successor was fit
only on source positions 0..3.  In addition, every *correct-path* arithmetic
transition (current_value, operand) is selected from the ALU's held-out pair
split, so the data plane is OOD at every executed semantic hop as well.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np

from learned_generated_value_recurrence_experiment import (
    fit_operator,
    heldout_operand_choices,
    vector_features,
)
from latent_program_counter_experiment import (
    HALT,
    PCConfig,
    fit_successor,
    fourier_codebook as pc_fourier_codebook,
)
from operator_compatible_geometry_experiment import (
    GeometryConfig,
    make_fourier_codebook,
    make_random_codebook,
)

EXPERIMENT_NAME = "exp_007_integrated_latent_machine"
AluMode = Literal["local", "full"]
PcMode = Literal["local", "full"]
Intervention = Literal["normal", "shuffle_value_after_2", "shift_pc_after_2"]


def build_programs(
    value_cfg: GeometryConfig,
    pc_cfg: PCConfig,
    pairs: np.ndarray,
    heldout_indices: np.ndarray,
    *,
    min_length: int,
    max_length: int,
    examples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    choices = heldout_operand_choices(value_cfg, pairs, heldout_indices)
    length = rng.integers(min_length, max_length + 1, size=examples, dtype=np.int64)
    operand_memory = rng.integers(
        0, value_cfg.modulus, size=(examples, pc_cfg.pc_modulus), dtype=np.int64
    )
    halt_memory = np.zeros((examples, pc_cfg.pc_modulus), dtype=bool)
    halt_memory[np.arange(examples), length] = True
    start = rng.integers(0, value_cfg.modulus, size=examples, dtype=np.int64)
    target = start.copy()

    # Correct-path operands are chosen adaptively so every semantic transition
    # is guaranteed held out from ALU fitting.
    for t in range(max_length):
        active = t < length
        indices = np.flatnonzero(active)
        for i in indices.tolist():
            pool = choices[int(target[i])]
            operand_memory[i, t] = int(pool[rng.integers(0, len(pool))])
        target[active] = (target[active] + operand_memory[active, t]) % value_cfg.modulus

    assert np.all(halt_memory.sum(axis=1) == 1)
    return length, operand_memory, halt_memory, start, target


def execute(
    value_cfg: GeometryConfig,
    pc_cfg: PCConfig,
    *,
    value_code: np.ndarray,
    alu_w: np.ndarray,
    alu_mode: AluMode,
    pc_code: np.ndarray,
    pc_w: np.ndarray,
    length: np.ndarray,
    operand_memory: np.ndarray,
    halt_memory: np.ndarray,
    start: np.ndarray,
    intervention: Intervention = "normal",
) -> dict:
    b = len(start)
    value = value_code[start].copy()
    pc = np.repeat(pc_code[0][None, :], b, axis=0)
    halted = np.zeros(b, dtype=bool)
    halt_step = np.full(b, -1, dtype=np.int64)
    halt_position = np.full(b, -1, dtype=np.int64)
    value_hop_accuracy = []
    pc_hop_accuracy = []
    oracle_value = start.copy()

    for step in range(pc_cfg.pc_modulus + 2):
        pc_logits = pc @ pc_code.T
        pos = pc_logits.argmax(axis=1)
        expected_pos = np.minimum(step, length)
        pc_hop_accuracy.append(float(np.mean(pos == expected_pos)))

        is_halt = halt_memory[np.arange(b), pos]
        new_halt = (~halted) & is_halt
        halt_step[new_halt] = step
        halt_position[new_halt] = pos[new_halt]
        halted |= new_halt
        active = ~halted
        if not np.any(active):
            break

        operand_id = operand_memory[np.arange(b), pos]
        operand_code = value_code[operand_id]
        value_next = vector_features(value, operand_code, alu_mode) @ alu_w
        value_next /= np.maximum(
            np.linalg.norm(value_next, axis=1, keepdims=True), 1e-12
        )
        value[active] = value_next[active]

        # Oracle advances only along the intended sequential program path.  It
        # is diagnostic only and never fed back to the machine.
        if step < operand_memory.shape[1]:
            intended_active = step < length
            oracle_value[intended_active] = (
                oracle_value[intended_active]
                + operand_memory[intended_active, step]
            ) % value_cfg.modulus
            value_pred = (value @ value_code.T).argmax(axis=1)
            value_hop_accuracy.append(
                float(np.mean(value_pred[intended_active] == oracle_value[intended_active]))
                if np.any(intended_active)
                else 1.0
            )
        else:
            # Only failing controls can survive beyond the finite instruction
            # memory. There is no intended semantic transition to diagnose.
            value_hop_accuracy.append(1.0)

        pc_next = pc @ pc_w
        pc_next /= np.maximum(np.linalg.norm(pc_next, axis=1, keepdims=True), 1e-12)
        pc[active] = pc_next[active]

        if step + 1 == 2:
            if intervention == "shuffle_value_after_2":
                value[active] = np.roll(value[active], 1, axis=0)
            elif intervention == "shift_pc_after_2":
                # Distribution-preserving wrong control state: move the latent
                # PC one extra canonical position forward.
                wrong = np.minimum(step + 2, pc_cfg.pc_modulus - 1)
                pc[active] = pc_code[wrong]

    return {
        "value_prediction": (value @ value_code.T).argmax(axis=1),
        "halted": halted,
        "halt_step": halt_step,
        "halt_position": halt_position,
        "value_hop_accuracy": value_hop_accuracy,
        "pc_hop_accuracy": pc_hop_accuracy,
    }


def evaluate_arm(
    *,
    value_cfg: GeometryConfig,
    pc_cfg: PCConfig,
    alu_mode: AluMode,
    pc_mode: PcMode,
    value_geometry: Literal["fourier", "random"],
    split_seed: int,
    value_code_seed: int,
    min_length: int,
    max_length: int,
    examples: int,
    seed: int,
    intervention: Intervention = "normal",
) -> dict:
    value_code = (
        make_fourier_codebook(value_cfg)
        if value_geometry == "fourier"
        else make_random_codebook(value_cfg, value_code_seed)
    )
    alu_w, pairs, heldout, split_digest = fit_operator(
        value_cfg, value_code, split_seed=split_seed, mode=alu_mode
    )
    pc_code = pc_fourier_codebook(pc_cfg.pc_modulus, pc_cfg.pc_harmonics)
    pc_w = fit_successor(pc_code, pc_cfg, pc_mode)
    length, operand_memory, halt_memory, start, target = build_programs(
        value_cfg,
        pc_cfg,
        pairs,
        heldout,
        min_length=min_length,
        max_length=max_length,
        examples=examples,
        seed=seed,
    )
    out = execute(
        value_cfg,
        pc_cfg,
        value_code=value_code,
        alu_w=alu_w,
        alu_mode=alu_mode,
        pc_code=pc_code,
        pc_w=pc_w,
        length=length,
        operand_memory=operand_memory,
        halt_memory=halt_memory,
        start=start,
        intervention=intervention,
    )
    return {
        "value_geometry": value_geometry,
        "alu_mode": alu_mode,
        "pc_mode": pc_mode,
        "intervention": intervention,
        "length_range": [min_length, max_length],
        "examples": examples,
        "split_seed": split_seed,
        "value_code_seed": value_code_seed,
        "alu_pair_split_sha256": split_digest,
        "all_correct_path_alu_transitions_heldout": True,
        "value_accuracy": float(np.mean(out["value_prediction"] == target)),
        "halt_rate": float(np.mean(out["halted"])),
        "halt_step_accuracy": float(np.mean(out["halt_step"] == length)),
        "halt_position_accuracy": float(np.mean(out["halt_position"] == length)),
        "value_hop_accuracy": out["value_hop_accuracy"],
        "pc_hop_accuracy": out["pc_hop_accuracy"],
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/research/exp_007/metrics.json"))
    p.add_argument("--examples", type=int, default=4096)
    p.add_argument("--split-seeds", type=int, nargs="+", default=[101, 202, 303])
    args = p.parse_args()
    value_cfg = GeometryConfig()
    pc_cfg = PCConfig()
    arms = []
    for split_seed in args.split_seeds:
        base = dict(
            value_cfg=value_cfg,
            pc_cfg=pc_cfg,
            split_seed=split_seed,
            value_code_seed=11,
            min_length=5,
            max_length=10,
            examples=args.examples,
            seed=990000 + split_seed,
        )
        arms.append(evaluate_arm(**base, value_geometry="fourier", alu_mode="local", pc_mode="local"))
        arms.append(evaluate_arm(**base, value_geometry="fourier", alu_mode="full", pc_mode="local"))
        arms.append(evaluate_arm(**base, value_geometry="fourier", alu_mode="local", pc_mode="full"))
        arms.append(evaluate_arm(**base, value_geometry="random", alu_mode="full", pc_mode="local"))
        arms.append(evaluate_arm(**base, value_geometry="fourier", alu_mode="local", pc_mode="local", intervention="shuffle_value_after_2"))
        arms.append(evaluate_arm(**base, value_geometry="fourier", alu_mode="local", pc_mode="local", intervention="shift_pc_after_2"))
    payload = {
        "experiment": EXPERIMENT_NAME,
        "value_config": asdict(value_cfg),
        "pc_config": asdict(pc_cfg),
        "unique_halt": True,
        "post_halt_distractors": True,
        "arms": arms,
    }
    write_json(args.output, payload)
    keys = [
        ("fourier", "local", "local", "normal"),
        ("fourier", "full", "local", "normal"),
        ("fourier", "local", "full", "normal"),
        ("random", "full", "local", "normal"),
        ("fourier", "local", "local", "shuffle_value_after_2"),
        ("fourier", "local", "local", "shift_pc_after_2"),
    ]
    summary = {}
    for key in keys:
        rows = [r for r in arms if (r["value_geometry"], r["alu_mode"], r["pc_mode"], r["intervention"]) == key]
        summary["_".join(key)] = {
            "runs": len(rows),
            "value_accuracy": float(np.mean([r["value_accuracy"] for r in rows])),
            "halt_accuracy": float(np.mean([r["halt_step_accuracy"] for r in rows])),
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
