#!/usr/bin/env python3
"""EXP-035: black-box structural compilation of a genuinely nonlinear latent map.

Observed hidden coordinates are a nonlinear warp of a simple semantic plane:

    w = phi(z) = z + alpha z^2.

Two black-box recurrent operators act semantically by a 7-fold rotation and a
reflection, then return through the nonlinear warp.  In observed coordinates
both maps are nonlinear and no single global linear matrix fits the orbit well.

The compiler is *not* given semantic z, the warp inverse, hidden charts, local
Jacobians, or the context graph.  Starting from one hidden state it:

1. actively explores the black-box operator orbit and deduplicates hidden states;
2. infers the operator transition graph by applying each black-box map to every
   discovered state;
3. estimates local Jacobians by finite perturbation probes;
4. keeps only each Jacobian's orthogonal/polar action on tangent direction;
5. gauge-synchronizes the local Jacobian field into shared O(2) operators;
6. discovers finite closure from cycle structure and measured loop holonomy;
7. projects only laws whose holonomy residual passes a fixed threshold.

Evaluation tracks tangent directions through depth-256 programs.  This is not a
claim about arbitrary nonlinear networks: the warp is 2D conformal and the
active orbit is finite.  It is the first gate where structural evidence is
recovered from black-box nonlinear dynamics rather than supplied matrices.
"""
from __future__ import annotations

import argparse
import cmath
import json
import math
import os
from math import gcd
from pathlib import Path

import torch
from torch.nn import functional as F

from state_conditioned_jacobian_gauge_experiment import o2, polar, rotation, synchronize

EXPERIMENT_NAME = "exp_035_nonlinear_blackbox_structural_compiler"


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


class NonlinearDihedralBlackBox:
    def __init__(self, order: int = 7, alpha: float = 0.55, reflection_phase: float = 0.71):
        self.order = int(order)
        self.alpha = float(alpha)
        self.omega = cmath.exp(2j * math.pi / self.order)
        self.reflection = cmath.exp(1j * reflection_phase)

    def warp(self, z: complex) -> complex:
        return z + self.alpha * z * z

    def unwarp(self, w: complex) -> complex:
        # Principal branch is the local inverse around the small-radius orbit.
        return (-1.0 + cmath.sqrt(1.0 + 4.0 * self.alpha * w)) / (2.0 * self.alpha)

    @staticmethod
    def to_tensor(z: complex) -> torch.Tensor:
        return torch.tensor([z.real, z.imag], dtype=torch.float64)

    @staticmethod
    def to_complex(x: torch.Tensor) -> complex:
        return complex(float(x[0]), float(x[1]))

    def apply(self, x: torch.Tensor, operator_index: int) -> torch.Tensor:
        z = self.unwarp(self.to_complex(x))
        if operator_index == 0:
            z2 = self.omega * z
        elif operator_index == 1:
            z2 = self.reflection * z.conjugate()
        else:
            raise ValueError("operator_index must be 0 or 1")
        return self.to_tensor(self.warp(z2))

    def start_state(self) -> torch.Tensor:
        z0 = 0.35 * cmath.exp(1j * 0.173)
        return self.to_tensor(self.warp(z0))


def discover_orbit(box: NonlinearDihedralBlackBox, tolerance: float = 1e-8, max_nodes: int = 64) -> torch.Tensor:
    queue = [box.start_state()]
    states: list[torch.Tensor] = []
    while queue:
        x = queue.pop(0)
        if any(float((x - y).norm()) <= tolerance for y in states):
            continue
        states.append(x)
        if len(states) > max_nodes:
            raise RuntimeError("black-box orbit exceeded max_nodes")
        queue.extend([box.apply(x, 0), box.apply(x, 1)])
    return torch.stack(states)


def infer_edge_graph(box: NonlinearDihedralBlackBox, states: torch.Tensor, match_tolerance: float = 1e-7) -> dict[str, torch.Tensor]:
    edges = {}
    for op_idx, name in enumerate(["op0", "op1"]):
        dest = []
        for x in states:
            y = box.apply(x, op_idx)
            dist = (states - y).square().sum(dim=-1).sqrt()
            j = int(dist.argmin())
            if float(dist[j]) > match_tolerance:
                raise RuntimeError("black-box transition does not close on discovered orbit")
            dest.append(j)
        edges[name] = torch.tensor(dest, dtype=torch.long)
    return edges


def permutation_cycles(edge: torch.Tensor) -> list[list[int]]:
    seen: set[int] = set()
    cycles = []
    for i in range(edge.numel()):
        if i in seen:
            continue
        cur = []
        x = i
        while x not in cur:
            cur.append(x)
            seen.add(x)
            x = int(edge[x])
        if x != i:
            # For this gate operators must act as permutations on the orbit.
            raise RuntimeError("edge graph is not a permutation cycle decomposition")
        cycles.append(cur)
    return cycles


def finite_difference_jacobian(box: NonlinearDihedralBlackBox, x: torch.Tensor, operator_index: int, eps: float = 1e-5) -> torch.Tensor:
    cols = []
    for j in range(2):
        delta = torch.zeros(2, dtype=torch.float64)
        delta[j] = eps
        cols.append((box.apply(x + delta, operator_index) - box.apply(x - delta, operator_index)) / (2.0 * eps))
    return torch.stack(cols, dim=1)


def estimate_local_polar_actions(
    box: NonlinearDihedralBlackBox,
    states: torch.Tensor,
    relative_noise: float,
    seed: int,
    eps: float = 1e-5,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    exact = {}
    observed = {}
    for op_idx, name in enumerate(["op0", "op1"]):
        exact_rows = []
        observed_rows = []
        for x in states:
            P = polar(finite_difference_jacobian(box, x, op_idx, eps=eps))
            exact_rows.append(P)
            if relative_noise > 0:
                N = torch.randn(P.shape, generator=g, dtype=P.dtype)
                N = N / N.norm().clamp_min(1e-30) * P.norm() * relative_noise
                P = polar(P + N)
            observed_rows.append(P)
        exact[name] = torch.stack(exact_rows)
        observed[name] = torch.stack(observed_rows)
    return observed, exact


def global_linear_fit_error(box: NonlinearDihedralBlackBox, states: torch.Tensor) -> dict[str, float]:
    out = {}
    for op_idx, name in enumerate(["op0", "op1"]):
        Y = torch.stack([box.apply(x, op_idx) for x in states])
        WT = torch.linalg.lstsq(states, Y).solution
        pred = states @ WT
        out[name] = float((pred - Y).norm() / Y.norm().clamp_min(1e-30))
    return out


def cycle_holonomy_rows(local: torch.Tensor, cycles: list[list[int]]) -> dict:
    residuals = []
    I = torch.eye(2, dtype=torch.float64)
    for cyc in cycles:
        P = I.clone()
        for x in cyc:
            P = local[x] @ P
        residuals.append(float((P - I).norm() / I.norm()))
    return {
        "cycle_lengths": [len(c) for c in cycles],
        "identity_residuals": residuals,
        "max_identity_residual": max(residuals) if residuals else float("inf"),
    }


def project_graph_laws(sync: dict, observed: dict[str, torch.Tensor], edges: dict[str, torch.Tensor], threshold: float) -> dict:
    H = sync["gauges"]
    shared = {k: v.clone() for k, v in sync["shared"].items()}
    evidence = {}
    accepted = {}
    for name, edge in edges.items():
        cycles = permutation_cycles(edge)
        ev = cycle_holonomy_rows(observed[name], cycles)
        order = 1
        for length in ev["cycle_lengths"]:
            order = _lcm(order, length)
        ev["graph_order"] = order
        sign = sync["det_sign"][name]
        ok = ev["max_identity_residual"] <= threshold
        if ok and sign > 0:
            angle = float(sync["operator_angles"][name])
            step = 2 * math.pi / order
            k = int(round(angle / step))
            shared[name] = rotation(torch.tensor(k * step, dtype=torch.float64))
        # det=-1 O(2) actions are already exact reflections and square to I.
        evidence[name] = ev
        accepted[name] = ok
    reconstructed = {
        name: torch.stack([H[int(edges[name][x])] @ shared[name] @ H[x].T for x in range(H.size(0))])
        for name in shared
    }
    return {"shared": shared, "reconstructed": reconstructed, "evidence": evidence, "accepted": accepted, "threshold": threshold}


@torch.inference_mode()
def tangent_execution(
    exact: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    edges: dict[str, torch.Tensor],
    seed: int,
    depth: int,
    examples: int,
) -> dict:
    nodes = next(iter(edges.values())).numel()
    g = torch.Generator().manual_seed(seed + 350_000)
    ctx = torch.randint(nodes, (examples,), generator=g)
    ref = F.normalize(torch.randn(examples, 2, generator=g, dtype=torch.float64), dim=-1)
    pred = ref.clone()
    for _ in range(depth):
        kind = torch.randint(2, (examples,), generator=g)
        ref_next = torch.empty_like(ref)
        pred_next = torch.empty_like(pred)
        new_ctx = torch.empty_like(ctx)
        for j, name in enumerate(["op0", "op1"]):
            sel = torch.where(kind.eq(j))[0]
            if sel.numel() == 0:
                continue
            c = ctx[sel]
            ref_next[sel] = torch.einsum("nij,nj->ni", exact[name][c], ref[sel])
            pred_next[sel] = torch.einsum("nij,nj->ni", candidate[name][c], pred[sel])
            new_ctx[sel] = edges[name][c]
        ref = F.normalize(ref_next, dim=-1)
        pred = F.normalize(pred_next, dim=-1)
        ctx = new_ctx
    cosine = F.cosine_similarity(pred, ref, dim=-1)
    return {
        "depth": depth,
        "mean_tangent_cosine": float(cosine.mean()),
        "fraction_cosine_gt_0_99": float(cosine.gt(0.99).float().mean()),
        "fraction_cosine_gt_0_999": float(cosine.gt(0.999).float().mean()),
    }


def run_one(seed: int, noise: float, depth: int, examples: int, holonomy_threshold: float = 0.40) -> dict:
    box = NonlinearDihedralBlackBox()
    states = discover_orbit(box)
    edges = infer_edge_graph(box, states)
    observed, exact = estimate_local_polar_actions(box, states, noise, seed + 1)
    sync = synchronize(observed, edges, seed, steps=900, restarts=4)
    law = project_graph_laws(sync, observed, edges, holonomy_threshold)
    cycles = {name: [len(c) for c in permutation_cycles(edge)] for name, edge in edges.items()}
    return {
        "seed": seed,
        "jacobian_noise": noise,
        "discovered_orbit_size": int(states.size(0)),
        "discovered_cycle_lengths": cycles,
        "global_linear_fit_relative_error": global_linear_fit_error(box, states),
        "synchronization_loss": sync["loss"],
        "raw_tangent_execution": tangent_execution(exact, observed, edges, seed, depth, examples),
        "synchronized_tangent_execution": tangent_execution(exact, sync["reconstructed"], edges, seed, depth, examples),
        "compiled_tangent_execution": tangent_execution(exact, law["reconstructed"], edges, seed, depth, examples),
        "law_evidence": law["evidence"],
        "law_accepted": law["accepted"],
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/research/exp_035/metrics.json"))
    p.add_argument("--seeds", type=int, nargs="+", default=[150, 151, 152])
    p.add_argument("--noise", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    p.add_argument("--program-depth", type=int, default=256)
    p.add_argument("--examples", type=int, default=1024)
    p.add_argument("--holonomy-threshold", type=float, default=0.40)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    rows = []
    for seed in args.seeds:
        for noise in args.noise:
            row = run_one(seed, noise, args.program_depth, args.examples, args.holonomy_threshold)
            rows.append(row)
            print(
                f"seed={seed} noise={noise:.2f} orbit={row['discovered_orbit_size']} "
                f"raw={row['raw_tangent_execution']['fraction_cosine_gt_0_99']:.4f} "
                f"sync={row['synchronized_tangent_execution']['fraction_cosine_gt_0_99']:.4f} "
                f"compiled={row['compiled_tangent_execution']['fraction_cosine_gt_0_99']:.4f} "
                f"accepted={row['law_accepted']}"
            )

    payload = {
        "experiment": EXPERIMENT_NAME,
        "protocol": {
            "semantic_coordinates_given_to_compiler": False,
            "nonlinear_warp_given_to_compiler": False,
            "context_graph_given_to_compiler": False,
            "jacobians_given_to_compiler": False,
            "orbit_discovery": "active black-box BFS from one hidden state",
            "edge_discovery": "apply black-box operators and nearest-match within discovered orbit",
            "jacobian_estimation": "central finite differences in observed hidden coordinates",
            "compiled_object": "polar/tangent action field",
            "holonomy_threshold": args.holonomy_threshold,
            "program_depth": args.program_depth,
            "examples": args.examples,
            "claim_boundary": "finite 2D conformal nonlinear orbit; tangent dynamics only",
        },
        "rows": rows,
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
