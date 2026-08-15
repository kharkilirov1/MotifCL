from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Iterable

import torch

Tensor = torch.Tensor
Transition = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class JVPStats:
    mean: float
    p95: float
    maximum: float
    minimum: float
    probes: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class RelationEvidence:
    state_relative_rms: float
    jvp_relative_rms: float
    probes: int
    states: int

    def accepted(self, *, state_threshold: float, jvp_threshold: float) -> bool:
        return (
            self.state_relative_rms <= state_threshold
            and self.jvp_relative_rms <= jvp_threshold
        )

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _generator_for(state: Tensor, seed: int) -> torch.Generator:
    # CPU generators also work for CPU production probes. CUDA callers should
    # supply probes explicitly until device-specific deterministic generators
    # are needed by the training pipeline.
    if state.device.type != "cpu":
        raise ValueError("automatic structural probes currently require CPU tensors")
    return torch.Generator(device="cpu").manual_seed(seed)


def random_unit_probe_like(state: Tensor, *, generator: torch.Generator) -> Tensor:
    probe = torch.randn(
        state.shape,
        generator=generator,
        device=state.device,
        dtype=state.dtype,
    )
    return probe / probe.norm().clamp_min(torch.finfo(probe.dtype).eps)



def _jvp(transition: Transition, state: Tensor, vector: Tensor) -> tuple[Tensor, Tensor]:
    """JVP using the differentiable math-SDPA backend when attention appears.

    PyTorch's CPU flash-SDPA kernel currently lacks forward-AD (and the double
    backward needed by ``autograd.functional.jvp``). Structural probes force
    the math backend locally; normal model execution remains unchanged.
    """
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        return torch.func.jvp(transition, (state,), (vector,))

def jvp_gain_stats(
    transition: Transition,
    state: Tensor,
    *,
    probes: int = 8,
    seed: int = 0,
) -> JVPStats:
    """Estimate local perturbation gain without materializing a Jacobian.

    The statistic is ``||J v|| / ||v||`` for random unit probes. It is not the
    spectral norm; it is a cheap production diagnostic for whether perturbation
    directions are typically contracted or amplified by a recurrent step.
    """
    if probes < 1:
        raise ValueError("probes must be >= 1")
    generator = _generator_for(state, seed)
    gains = []
    for _ in range(probes):
        v = random_unit_probe_like(state, generator=generator)
        _, jv = _jvp(transition, state, v)
        gains.append(float((jv.norm() / v.norm().clamp_min(1e-30)).detach()))
    values = torch.tensor(gains, dtype=torch.float64)
    return JVPStats(
        mean=float(values.mean()),
        p95=float(torch.quantile(values, 0.95)),
        maximum=float(values.max()),
        minimum=float(values.min()),
        probes=probes,
    )


def relation_evidence(
    lhs: Transition,
    rhs: Transition,
    states: Iterable[Tensor],
    *,
    probes: int = 8,
    seed: int = 0,
) -> RelationEvidence:
    """Randomized state/JVP evidence that two black-box transitions agree."""
    states = list(states)
    if not states:
        raise ValueError("at least one state is required")
    if probes < 1:
        raise ValueError("probes must be >= 1")
    state_num = 0.0
    state_den = 0.0
    tangent_num = 0.0
    tangent_den = 0.0
    total_probes = 0
    for si, state in enumerate(states):
        yl = lhs(state)
        yr = rhs(state)
        state_num += float((yl - yr).pow(2).sum())
        state_den += float(yr.pow(2).sum())
        generator = _generator_for(state, seed + si * 104729)
        for _ in range(probes):
            v = random_unit_probe_like(state, generator=generator)
            _, jl = _jvp(lhs, state, v)
            _, jr = _jvp(rhs, state, v)
            tangent_num += float((jl - jr).pow(2).sum())
            tangent_den += float(jr.pow(2).sum())
            total_probes += 1
    return RelationEvidence(
        state_relative_rms=math.sqrt(state_num / max(state_den, 1e-30)),
        jvp_relative_rms=math.sqrt(tangent_num / max(tangent_den, 1e-30)),
        probes=total_probes,
        states=len(states),
    )
