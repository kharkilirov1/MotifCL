from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

Tensor = torch.Tensor


@dataclass
class Config:
    dim: int = 128
    prime: int = 31
    scale_generator: int = 3
    nonlinear_scale: float = 0.30
    perturbation: float = 0.0
    probes: int = 8
    states: int = 3
    seed: int = 0
    max_order: int = 35


def _orthogonal_matrix(dim: int, generator: torch.Generator) -> Tensor:
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=generator, dtype=torch.float64))
    sign = torch.sign(torch.diag(r))
    sign[sign == 0] = 1
    return q * sign


def _zero_sum_basis(prime: int) -> Tensor:
    # Nullspace of the all-ones row. SVD gives a deterministic orthonormal basis.
    one = torch.ones(1, prime, dtype=torch.float64)
    _, _, vh = torch.linalg.svd(one, full_matrices=True)
    return vh[1:].T.contiguous()  # [p, p-1]


def _perm_matrix(mapping: list[int]) -> Tensor:
    n = len(mapping)
    out = torch.zeros(n, n, dtype=torch.float64)
    for src, dst in enumerate(mapping):
        out[dst, src] = 1.0
    return out


def affine_group_actions(prime: int = 31, scale_generator: int = 3) -> tuple[Tensor, Tensor]:
    basis = _zero_sum_basis(prime)
    add_perm = _perm_matrix([(x + 1) % prime for x in range(prime)])
    mul_perm = _perm_matrix([(scale_generator * x) % prime for x in range(prime)])
    add = basis.T @ add_perm @ basis
    mul = basis.T @ mul_perm @ basis
    return add, mul


class NonlinearConjugateSystem:
    """High-dimensional nonlinear black box hiding a finite operator algebra.

    Observed coordinates are w=tanh(s*x), while x is globally mixed by an
    unknown orthogonal gauge. The compiler only calls ``A``/``M`` and JVPs; it
    never receives the hidden matrices or semantic basis.
    """

    def __init__(self, cfg: Config):
        if cfg.dim < cfg.prime - 1:
            raise ValueError("dim must be at least prime-1")
        self.cfg = cfg
        g = torch.Generator().manual_seed(cfg.seed)
        add30, mul30 = affine_group_actions(cfg.prime, cfg.scale_generator)
        intrinsic = cfg.prime - 1
        add = torch.eye(cfg.dim, dtype=torch.float64)
        mul = torch.eye(cfg.dim, dtype=torch.float64)
        add[:intrinsic, :intrinsic] = add30
        mul[:intrinsic, :intrinsic] = mul30
        gauge = _orthogonal_matrix(cfg.dim, g)
        self._add_hidden = gauge @ add @ gauge.T
        self._mul_hidden = gauge @ mul @ gauge.T
        # Fixed generic low-rank perturbations emulate imperfect learned maps.
        rank = min(8, cfg.dim)
        self._noise_u_a = torch.randn(cfg.dim, rank, generator=g, dtype=torch.float64) / math.sqrt(cfg.dim)
        self._noise_v_a = torch.randn(rank, cfg.dim, generator=g, dtype=torch.float64) / math.sqrt(cfg.dim)
        self._noise_u_m = torch.randn(cfg.dim, rank, generator=g, dtype=torch.float64) / math.sqrt(cfg.dim)
        self._noise_v_m = torch.randn(rank, cfg.dim, generator=g, dtype=torch.float64) / math.sqrt(cfg.dim)

    def _phi(self, x: Tensor) -> Tensor:
        return torch.tanh(self.cfg.nonlinear_scale * x)

    def _phi_inv(self, w: Tensor) -> Tensor:
        eps = torch.finfo(w.dtype).eps * 16
        return torch.atanh(w.clamp(-1 + eps, 1 - eps)) / self.cfg.nonlinear_scale

    def _map(self, w: Tensor, action: Tensor, noise_u: Tensor, noise_v: Tensor) -> Tensor:
        hidden = self._phi_inv(w)
        exact = self._phi(action @ hidden)
        if self.cfg.perturbation == 0:
            return exact
        perturb = noise_u @ torch.tanh(noise_v @ w)
        return (exact + self.cfg.perturbation * perturb).clamp(-0.995, 0.995)

    def A(self, w: Tensor) -> Tensor:
        return self._map(w, self._add_hidden, self._noise_u_a, self._noise_v_a)

    def M(self, w: Tensor) -> Tensor:
        return self._map(w, self._mul_hidden, self._noise_u_m, self._noise_v_m)

    def states(self, count: int) -> list[Tensor]:
        g = torch.Generator().manual_seed(self.cfg.seed + 10_000)
        out = []
        for _ in range(count):
            hidden = torch.randn(self.cfg.dim, generator=g, dtype=torch.float64)
            hidden = hidden / hidden.norm() * (0.7 * math.sqrt(self.cfg.dim))
            out.append(self._phi(hidden))
        return out


def compose_power(fn: Callable[[Tensor], Tensor], power: int) -> Callable[[Tensor], Tensor]:
    if power < 0:
        raise ValueError("power must be non-negative")

    def run(x: Tensor) -> Tensor:
        y = x
        for _ in range(power):
            y = fn(y)
        return y

    return run


def compose(*fns: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    def run(x: Tensor) -> Tensor:
        y = x
        for fn in reversed(fns):
            y = fn(y)
        return y
    return run


def random_unit_vectors(dim: int, count: int, seed: int) -> list[Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(count):
        v = torch.randn(dim, generator=g, dtype=torch.float64)
        out.append(v / v.norm().clamp_min(1e-12))
    return out


def sketched_equivalence(
    lhs: Callable[[Tensor], Tensor],
    rhs: Callable[[Tensor], Tensor],
    states: list[Tensor],
    *,
    probes: int,
    seed: int,
) -> dict[str, float]:
    state_num = 0.0
    state_den = 0.0
    tangent_num = 0.0
    tangent_den = 0.0
    total = 0
    for si, x in enumerate(states):
        yl = lhs(x)
        yr = rhs(x)
        state_num += float((yl - yr).pow(2).sum())
        state_den += float(yr.pow(2).sum())
        for pi, v in enumerate(random_unit_vectors(x.numel(), probes, seed + 1009 * si)):
            _, jl = torch.func.jvp(lhs, (x,), (v,))
            _, jr = torch.func.jvp(rhs, (x,), (v,))
            tangent_num += float((jl - jr).pow(2).sum())
            tangent_den += float(jr.pow(2).sum())
            total += 1
    return {
        "state_relative_rms": math.sqrt(state_num / max(state_den, 1e-30)),
        "jvp_relative_rms": math.sqrt(tangent_num / max(tangent_den, 1e-30)),
        "jvp_samples": total,
    }


def state_equivalence(
    lhs: Callable[[Tensor], Tensor], rhs: Callable[[Tensor], Tensor], states: list[Tensor]
) -> float:
    num = 0.0
    den = 0.0
    for x in states:
        yl = lhs(x)
        yr = rhs(x)
        num += float((yl - yr).pow(2).sum())
        den += float(yr.pow(2).sum())
    return math.sqrt(num / max(den, 1e-30))


def discover_order(
    fn: Callable[[Tensor], Tensor],
    states: list[Tensor],
    *,
    max_order: int,
    probes: int,
    seed: int,
) -> tuple[int, dict[str, float], list[dict[str, float]]]:
    # Search cheaply in state space, then use JVPs only to verify the few best
    # candidate laws. This is the scalable path for high-dimensional systems.
    identity = lambda x: x
    curve = []
    for order in range(2, max_order + 1):
        residual = state_equivalence(compose_power(fn, order), identity, states)
        curve.append({"order": order, "state_relative_rms": residual})
    finalists = sorted(curve, key=lambda row: row["state_relative_rms"])[:3]
    verified = []
    for row in finalists:
        order = int(row["order"])
        score = sketched_equivalence(
            compose_power(fn, order), identity, states,
            probes=probes, seed=seed + order * 17,
        )
        verified.append({"order": order, **score})
    best = min(verified, key=lambda row: row["state_relative_rms"] + row["jvp_relative_rms"])
    return int(best["order"]), {k: v for k, v in best.items() if k != "order"}, curve


def discover_conjugacy_exponent(
    A: Callable[[Tensor], Tensor],
    M: Callable[[Tensor], Tensor],
    order_m: int,
    states: list[Tensor],
    *,
    max_exponent: int,
    probes: int,
    seed: int,
) -> tuple[int, dict[str, float], list[dict[str, float]]]:
    m_inv = compose_power(M, order_m - 1)
    lhs = compose(M, A, m_inv)  # M(A(M^-1(x)))
    curve = []
    for exponent in range(1, max_exponent + 1):
        rhs = compose_power(A, exponent)
        residual = state_equivalence(lhs, rhs, states)
        curve.append({"exponent": exponent, "state_relative_rms": residual})
    finalists = sorted(curve, key=lambda row: row["state_relative_rms"])[:3]
    verified = []
    for row in finalists:
        exponent = int(row["exponent"])
        rhs = compose_power(A, exponent)
        score = sketched_equivalence(lhs, rhs, states, probes=probes, seed=seed + exponent * 31)
        verified.append({"exponent": exponent, **score})
    best = min(verified, key=lambda row: row["state_relative_rms"] + row["jvp_relative_rms"])
    return int(best["exponent"]), {k: v for k, v in best.items() if k != "exponent"}, curve


def full_jacobian_relative_residual(
    lhs: Callable[[Tensor], Tensor], rhs: Callable[[Tensor], Tensor], x: Tensor
) -> float:
    jl = torch.autograd.functional.jacobian(lhs, x, vectorize=True)
    jr = torch.autograd.functional.jacobian(rhs, x, vectorize=True)
    return float((jl - jr).norm() / jr.norm().clamp_min(1e-30))


def run(cfg: Config) -> dict:
    system = NonlinearConjugateSystem(cfg)
    states = system.states(cfg.states)
    order_a, score_a, curve_a = discover_order(
        system.A, states, max_order=cfg.max_order, probes=cfg.probes, seed=cfg.seed + 2000
    )
    order_m, score_m, curve_m = discover_order(
        system.M, states, max_order=cfg.max_order, probes=cfg.probes, seed=cfg.seed + 3000
    )
    exponent, score_conj, curve_conj = discover_conjugacy_exponent(
        system.A, system.M, order_m, states,
        max_exponent=cfg.prime - 1, probes=cfg.probes, seed=cfg.seed + 4000,
    )
    result = {
        "config": asdict(cfg),
        "discovered": {
            "order_A": order_a,
            "order_M": order_m,
            "conjugacy_exponent": exponent,
        },
        "best_scores": {"A_order": score_a, "M_order": score_m, "conjugacy": score_conj},
        "curves": {"A_order": curve_a, "M_order": curve_m, "conjugacy": curve_conj},
    }
    if cfg.dim <= 40 and cfg.perturbation == 0:
        m_inv = compose_power(system.M, order_m - 1)
        lhs = compose(system.M, system.A, m_inv)
        rhs = compose_power(system.A, exponent)
        exact = full_jacobian_relative_residual(lhs, rhs, states[0])
        sketch = sketched_equivalence(lhs, rhs, [states[0]], probes=max(128, cfg.probes), seed=cfg.seed + 9999)
        result["full_jacobian_validation"] = {
            "full_relative_frobenius": exact,
            "jvp_relative_rms_128": sketch["jvp_relative_rms"],
        }
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--perturbation", type=float, default=0.0)
    p.add_argument("--probes", type=int, default=8)
    p.add_argument("--states", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    cfg = Config(dim=args.dim, perturbation=args.perturbation, probes=args.probes, states=args.states, seed=args.seed)
    result = run(cfg)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["discovered"], indent=2))
    print(json.dumps(result["best_scores"], indent=2))


if __name__ == "__main__":
    main()
