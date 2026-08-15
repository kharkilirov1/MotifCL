"""Small executable hooks for the paper's discrete resource-law viewpoint."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class MotifBudget:
    compare_rank: int
    latent_slots: int
    reasoning_steps: int
    memory_slots: int
    expansion_width: int


def compare_frobenius_error(singular_values: torch.Tensor, rank: int) -> torch.Tensor:
    """Exact best rank-r Frobenius error: sqrt(sum_{j>r} sigma_j^2)."""
    if rank >= singular_values.numel():
        return singular_values.new_zeros(())
    return singular_values[rank:].pow(2).sum().sqrt()


def allocate_equal_cost_discrete(
    gain_curves: Sequence[Sequence[float]],
    sensitivity: Sequence[float],
    total_units: int,
) -> list[int]:
    """
    Prefix-constrained greedy allocator for nonincreasing equal-cost gains.
    Implements the discrete exchange idea: each next unit competes by
    sensitivity * next marginal gain.
    """
    if len(gain_curves) != len(sensitivity):
        raise ValueError("gain_curves and sensitivity must have equal length")
    k = [0] * len(gain_curves)
    for _ in range(total_units):
        best_i, best_score = None, float("-inf")
        for i, curve in enumerate(gain_curves):
            j = k[i]
            if j < len(curve):
                score = sensitivity[i] * curve[j]
                if score > best_score:
                    best_i, best_score = i, score
        if best_i is None:
            break
        k[best_i] += 1
    return k
