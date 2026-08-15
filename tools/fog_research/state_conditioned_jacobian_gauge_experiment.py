#!/usr/bin/env python3
"""EXP-034: compile state-conditioned local Jacobians into one shared grammar.

There is no global action matrix in the observed latent coordinates.  Each
context x has its own local orthogonal gauge H_x and the same logical operator g
is observed through a state-conditioned Jacobian

    J[g,x] = H[g*x] R[g] H[x]^T.

The compiler receives only the context transition graph and noisy local
Jacobians.  It does not receive the hidden gauges, canonical shared operators,
or semantic state codebook.

A small gauge-synchronization optimization factorizes the local Jacobians into
per-context gauges plus shared O(2) operators.  The determinant component of
each O(2) operator is inferred from the observed Jacobians.  No finite group
law is imposed during this factorization.

For the A transition graph, the compiler additionally observes that the graph
forms one closed cycle.  If the measured cycle holonomy is near identity, the
recovered shared A is projected to the nearest root compatible with that cycle
length.  This uses a structural closure visible in the local transition graph,
not the hidden semantic labels.

We compare long-horizon continuous-state tracking for:
  raw_local          noisy local Jacobians;
  synchronized       factorized shared grammar without finite-order snap;
  synchronized_law   same grammar after cycle-closure projection.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch.nn import functional as F

EXPERIMENT_NAME = "exp_034_state_conditioned_jacobian_gauge"


def rotation(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def o2(theta: torch.Tensor, determinant_sign: int) -> torch.Tensor:
    if determinant_sign > 0:
        return rotation(theta)
    c, s = torch.cos(theta), torch.sin(theta)
    # General 2D reflection; theta is twice the reflection-axis angle.
    return torch.stack([torch.stack([c, s]), torch.stack([s, -c])])


def polar(W: torch.Tensor) -> torch.Tensor:
    u, _s, vh = torch.linalg.svd(W)
    return u @ vh


def context_edges(contexts: int) -> dict[str, torch.Tensor]:
    x = torch.arange(contexts)
    return {
        "A": (x + 1) % contexts,
        "B": torch.remainder(-x, contexts),
    }


def make_system(contexts: int, seed: int, noise: float) -> dict:
    g = torch.Generator().manual_seed(seed)
    gauge_angles = torch.rand(contexts, generator=g, dtype=torch.float64) * 2 * torch.pi
    H = torch.stack([rotation(t) for t in gauge_angles])
    true_R = {
        "A": rotation(torch.tensor(2 * math.pi / contexts, dtype=torch.float64)),
        "B": o2(torch.tensor(0.73, dtype=torch.float64), -1),
    }
    edges = context_edges(contexts)
    exact = {}
    observed = {}
    for name, R in true_R.items():
        rows = []
        noisy_rows = []
        for x in range(contexts):
            y = int(edges[name][x])
            J = H[y] @ R @ H[x].T
            rows.append(J)
            if noise > 0:
                N = torch.randn(J.shape, generator=g, dtype=torch.float64)
                N = N / N.norm().clamp_min(1e-30) * J.norm() * noise
                noisy_rows.append(polar(J + N))
            else:
                noisy_rows.append(J.clone())
        exact[name] = torch.stack(rows)
        observed[name] = torch.stack(noisy_rows)
    return {
        "contexts": contexts,
        "edges": edges,
        "hidden_gauges": H,
        "true_shared": true_R,
        "exact_local": exact,
        "observed_local": observed,
    }


def cycle_holonomy(local_A: torch.Tensor, edges_A: torch.Tensor) -> dict:
    contexts = local_A.size(0)
    seen = []
    x = 0
    P = torch.eye(2, dtype=torch.float64)
    for _ in range(contexts + 1):
        if x in seen:
            break
        seen.append(x)
        P = local_A[x] @ P
        x = int(edges_A[x])
    closed = x == 0 and len(seen) == contexts
    residual = float((P - torch.eye(2, dtype=torch.float64)).norm() / math.sqrt(2.0))
    return {"closed_single_cycle": closed, "cycle_length": len(seen), "identity_residual": residual}


def synchronize(
    local: dict[str, torch.Tensor],
    edges: dict[str, torch.Tensor],
    seed: int,
    steps: int = 900,
    lr: float = 0.03,
    restarts: int = 4,
) -> dict:
    contexts = next(iter(local.values())).size(0)
    det_sign = {
        name: (1 if float(torch.det(J).mean()) >= 0 else -1)
        for name, J in local.items()
    }
    best = None
    for restart in range(restarts):
        torch.manual_seed(seed + 340_000 + restart)
        gauge_tail = torch.nn.Parameter(torch.randn(contexts - 1, dtype=torch.float64))
        op_angles = {
            name: torch.nn.Parameter(torch.randn((), dtype=torch.float64))
            for name in local
        }
        opt = torch.optim.Adam([gauge_tail, *op_angles.values()], lr=lr)
        for _step in range(steps):
            angles = torch.cat([torch.zeros(1, dtype=torch.float64), gauge_tail])
            H = torch.stack([rotation(t) for t in angles])
            loss = torch.tensor(0.0, dtype=torch.float64)
            for name, Jrows in local.items():
                R = o2(op_angles[name], det_sign[name])
                dest = edges[name]
                pred = torch.stack([H[int(dest[x])] @ R @ H[x].T for x in range(contexts)])
                loss = loss + (pred - Jrows).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        value = float(loss.detach())
        if best is None or value < best["loss"]:
            best = {
                "loss": value,
                "gauge_angles": torch.cat([torch.zeros(1, dtype=torch.float64), gauge_tail.detach().clone()]),
                "operator_angles": {name: p.detach().clone() for name, p in op_angles.items()},
                "det_sign": det_sign,
            }
    assert best is not None
    H = torch.stack([rotation(t) for t in best["gauge_angles"]])
    R = {name: o2(best["operator_angles"][name], best["det_sign"][name]) for name in local}
    reconstructed = {
        name: torch.stack([H[int(edges[name][x])] @ R[name] @ H[x].T for x in range(contexts)])
        for name in local
    }
    best.update({"gauges": H, "shared": R, "reconstructed": reconstructed})
    return best


def project_cycle_law(sync: dict, local: dict[str, torch.Tensor], edges: dict[str, torch.Tensor], holonomy_threshold: float = 0.40) -> dict:
    H = sync["gauges"]
    shared = {k: v.clone() for k, v in sync["shared"].items()}
    holo = cycle_holonomy(local["A"], edges["A"])
    projected = False
    if holo["closed_single_cycle"] and holo["identity_residual"] <= holonomy_threshold:
        order = holo["cycle_length"]
        angle = float(sync["operator_angles"]["A"])
        step = 2 * math.pi / order
        k = int(round(angle / step))
        shared["A"] = rotation(torch.tensor(k * step, dtype=torch.float64))
        projected = True
    reconstructed = {
        name: torch.stack([H[int(edges[name][x])] @ shared[name] @ H[x].T for x in range(H.size(0))])
        for name in shared
    }
    return {
        "shared": shared,
        "reconstructed": reconstructed,
        "holonomy": holo,
        "projected": projected,
        "holonomy_threshold": holonomy_threshold,
    }


@torch.inference_mode()
def execute_continuous(system: dict, local_actions: dict[str, torch.Tensor], seed: int, depth: int, examples: int) -> dict:
    contexts = system["contexts"]
    edges = system["edges"]
    exact = system["exact_local"]
    Htrue = system["hidden_gauges"]
    g = torch.Generator().manual_seed(seed + 341_000)
    ctx = torch.randint(contexts, (examples,), generator=g)
    canonical = F.normalize(torch.randn(examples, 2, generator=g, dtype=torch.float64), dim=-1)
    z = torch.einsum("nij,nj->ni", Htrue[ctx], canonical)
    target = z.clone()
    target_ctx = ctx.clone()
    for _ in range(depth):
        kind = torch.randint(2, (examples,), generator=g)
        nxt = torch.empty_like(z)
        target_nxt = torch.empty_like(target)
        new_ctx = torch.empty_like(ctx)
        for j, name in enumerate(["A", "B"]):
            sel = torch.where(kind.eq(j))[0]
            if sel.numel() == 0:
                continue
            c = ctx[sel]
            tc = target_ctx[sel]
            nxt[sel] = torch.einsum("nij,nj->ni", local_actions[name][c], z[sel])
            target_nxt[sel] = torch.einsum("nij,nj->ni", exact[name][tc], target[sel])
            new_ctx[sel] = edges[name][c]
        z = F.normalize(nxt, dim=-1)
        target = F.normalize(target_nxt, dim=-1)
        ctx = new_ctx
        target_ctx = new_ctx
    cosine = F.cosine_similarity(z, target, dim=-1)
    return {
        "depth": depth,
        "mean_target_cosine": float(cosine.mean()),
        "fraction_cosine_gt_0_99": float(cosine.gt(0.99).float().mean()),
        "fraction_cosine_gt_0_999": float(cosine.gt(0.999).float().mean()),
    }


def shared_law_metrics(shared: dict[str, torch.Tensor], contexts: int) -> dict:
    A, B = shared["A"], shared["B"]
    I = torch.eye(2, dtype=torch.float64)
    return {
        "A_cycle_residual": float((torch.linalg.matrix_power(A, contexts) - I).norm() / I.norm()),
        "B_square_residual": float((B @ B - I).norm() / I.norm()),
        "dihedral_conjugacy_residual": float((B @ A @ B - A.T).norm() / A.norm()),
    }


def global_matrix_mismatch(local: dict[str, torch.Tensor]) -> dict:
    out = {}
    for name, J in local.items():
        mean = J.mean(dim=0)
        out[name] = float((J - mean).square().sum(dim=(-2, -1)).mean().sqrt() / J.square().sum(dim=(-2, -1)).mean().sqrt())
    return out


def run_one(contexts: int, seed: int, noise: float, depth: int, examples: int) -> dict:
    system = make_system(contexts, seed, noise)
    sync = synchronize(system["observed_local"], system["edges"], seed)
    law = project_cycle_law(sync, system["observed_local"], system["edges"])
    return {
        "seed": seed,
        "contexts": contexts,
        "noise": noise,
        "global_matrix_relative_mismatch": global_matrix_mismatch(system["observed_local"]),
        "synchronization_loss": sync["loss"],
        "raw_local_execution": execute_continuous(system, system["observed_local"], seed, depth, examples),
        "synchronized_execution": execute_continuous(system, sync["reconstructed"], seed, depth, examples),
        "synchronized_law_execution": execute_continuous(system, law["reconstructed"], seed, depth, examples),
        "synchronized_shared_laws": shared_law_metrics(sync["shared"], contexts),
        "projected_shared_laws": shared_law_metrics(law["shared"], contexts),
        "cycle_holonomy": law["holonomy"],
        "cycle_law_projection_applied": law["projected"],
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/research/exp_034/metrics.json"))
    p.add_argument("--contexts", type=int, default=7)
    p.add_argument("--seeds", type=int, nargs="+", default=[140, 141, 142])
    p.add_argument("--noise", type=float, nargs="+", default=[0.03, 0.05, 0.10, 0.15])
    p.add_argument("--program-depth", type=int, default=256)
    p.add_argument("--examples", type=int, default=1024)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    rows = []
    for seed in args.seeds:
        for noise in args.noise:
            row = run_one(args.contexts, seed, noise, args.program_depth, args.examples)
            rows.append(row)
            print(
                f"seed={seed} noise={noise:.2f} raw={row['raw_local_execution']['fraction_cosine_gt_0_99']:.4f} "
                f"sync={row['synchronized_execution']['fraction_cosine_gt_0_99']:.4f} "
                f"law={row['synchronized_law_execution']['fraction_cosine_gt_0_99']:.4f} "
                f"hol={row['cycle_holonomy']['identity_residual']:.4f} projected={row['cycle_law_projection_applied']}"
            )

    payload = {
        "experiment": EXPERIMENT_NAME,
        "protocol": {
            "global_action_matrix_exists_in_observed_gauge": False,
            "compiler_input": "context transition graph plus noisy local Jacobians",
            "hidden_gauges_given": False,
            "shared_operators_given": False,
            "semantic_codebook_given": False,
            "gauge_family": "2D local rotations",
            "shared_action_family": "O(2), determinant component inferred from observed Jacobians",
            "finite_order_used_during_synchronization": False,
            "cycle_projection": "only when observed A-edge graph is one closed cycle with near-identity measured holonomy",
            "program_depth": args.program_depth,
            "examples": args.examples,
        },
        "rows": rows,
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
