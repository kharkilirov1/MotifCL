#!/usr/bin/env python3
"""EXP-031: recover repeated joint irreducible blocks through the commutant.

The simple-spectrum compiler from EXP-023/030 is deliberately made impossible.
We build a repeated real 2D irreducible representation of S3,

    rho = rho_std (+) ... (+) rho_std   (multiplicity m),

then hide it behind a random orthogonal gauge.  Every element of the generated
operator algebra has each eigenvalue repeated at least ``m`` times, so no word
in the learned operators can supply a simple spectral anchor.

The compiler therefore uses the *joint commutant*:

    X A = A X,   X B = B X.

For an absolutely irreducible real block repeated m times, the commutant is
M_m(R), hence has dimension m^2.  A generic symmetric commutant element splits
the multiplicity space into m invariant copies.  The copies are then aligned by
solving an intertwiner equation, leaving one shared 2x2 operator grammar that
can be reused across all copies.

A broken-sharing control perturbs one copy to an inequivalent 2D dihedral block.
Its commutant dimension no longer equals m^2, so the compiler must reject the
claim that all copies implement one shared irreducible module.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch.nn import functional as F

EXPERIMENT_NAME = "exp_031_joint_commutant_block_compiler"


def _orthogonal(seed: int, d: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    signs = torch.sign(torch.diagonal(r)).clamp(min=-1.0, max=1.0)
    signs = torch.where(signs.eq(0), torch.ones_like(signs), signs)
    return q @ torch.diag(signs)


def s3_standard(angle_delta: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a 2D dihedral block and three canonical simplex states.

    At ``angle_delta=0`` this is the standard 2D irreducible representation of
    S3: A is a 120 degree rotation and B is a reflection, with
    A^3=B^2=I and B A B = A^{-1}.

    A non-zero delta intentionally makes an inequivalent D_n-like control block;
    it remains orthogonal but no longer shares the same finite S3 law.
    """
    theta = 2.0 * math.pi / 3.0 + angle_delta
    c, s = math.cos(theta), math.sin(theta)
    A = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
    B = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64)
    # Three unit vectors 120 degrees apart.  Under A they cycle; B swaps two.
    states = torch.stack(
        [
            torch.tensor([math.cos(2 * math.pi * k / 3), math.sin(2 * math.pi * k / 3)], dtype=torch.float64)
            for k in range(3)
        ],
        dim=0,
    )
    return A, B, states


def make_problem(multiplicity: int, seed: int, broken: bool = False, broken_delta: float = 0.11) -> dict:
    base_A, base_B, base_states = s3_standard()
    blocks_A = []
    blocks_B = []
    for j in range(multiplicity):
        if broken and j == multiplicity - 1:
            Aj, Bj, _ = s3_standard(angle_delta=broken_delta)
        else:
            Aj, Bj = base_A, base_B
        blocks_A.append(Aj)
        blocks_B.append(Bj)
    A0 = torch.block_diag(*blocks_A)
    B0 = torch.block_diag(*blocks_B)
    d = A0.size(0)
    Q = _orthogonal(seed + 310_000, d)
    A = Q @ A0 @ Q.T
    B = Q @ B0 @ Q.T

    # Full-rank semantic codebook: (multiplicity channel, S3 symbol).
    codes = []
    labels = []
    for channel in range(multiplicity):
        for symbol in range(3):
            z = torch.zeros(d, dtype=torch.float64)
            z[2 * channel : 2 * channel + 2] = base_states[symbol]
            codes.append(Q @ z)
            labels.append((channel, symbol))
    codes = F.normalize(torch.stack(codes), dim=-1)
    return {
        "A": A,
        "B": B,
        "Q": Q,
        "codes": codes,
        "labels": labels,
        "multiplicity": multiplicity,
        "block_dim": 2,
        "broken": broken,
    }


def _commutator_constraint(actions: list[torch.Tensor]) -> torch.Tensor:
    d = actions[0].size(0)
    cols = []
    for k in range(d * d):
        X = torch.zeros(d * d, dtype=torch.float64)
        X[k] = 1.0
        X = X.reshape(d, d)
        cols.append(torch.cat([(X @ A - A @ X).reshape(-1) for A in actions]))
    return torch.stack(cols, dim=1)


def commutant_basis(actions: list[torch.Tensor], relative_tol: float = 1e-9) -> dict:
    C = _commutator_constraint(actions)
    _u, s, vh = torch.linalg.svd(C, full_matrices=True)
    scale = float(s.max().item()) if s.numel() else 1.0
    tol = relative_tol * max(scale, 1.0)
    nullity = int(s.lt(tol).sum().item())
    # C has d^2 columns and SVD returns d^2 right singular vectors.
    basis = [v.reshape(actions[0].shape) for v in vh[-nullity:]] if nullity else []
    return {
        "constraint": C,
        "singular_values": s,
        "tol": tol,
        "dimension": nullity,
        "basis": basis,
        "smallest_singular_values": [float(x) for x in s[-min(12, s.numel()) :]],
    }


def _infer_repeated_irrep(d: int, commutant_dim: int) -> tuple[bool, int | None, int | None]:
    m = int(round(math.sqrt(commutant_dim)))
    if m < 1 or m * m != commutant_dim or d % m != 0:
        return False, None, None
    block_dim = d // m
    return block_dim > 1, m, block_dim


def _separator_from_commutant(basis: list[torch.Tensor], block_dim: int, multiplicity: int, seed: int) -> dict:
    sym = []
    for X in basis:
        H = 0.5 * (X + X.T)
        if H.norm() > 1e-10:
            sym.append(H / H.norm())
    if not sym:
        raise RuntimeError("commutant has no usable symmetric element")
    g = torch.Generator().manual_seed(seed + 311_000)
    best = None
    for attempt in range(64):
        coeff = torch.randn(len(sym), generator=g, dtype=torch.float64)
        H = sum(c * X for c, X in zip(coeff, sym))
        H = 0.5 * (H + H.T)
        vals, vecs = torch.linalg.eigh(H)
        order = torch.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
        groups = [vals[j * block_dim : (j + 1) * block_dim] for j in range(multiplicity)]
        within = max(float((q.max() - q.min()).abs()) for q in groups)
        centers = torch.tensor([float(q.mean()) for q in groups], dtype=torch.float64)
        gaps = torch.diff(centers).abs()
        min_gap = float(gaps.min()) if gaps.numel() else float("inf")
        score = min_gap / max(within, 1e-14)
        row = {"H": H, "eigvals": vals, "U": vecs, "within_spread": within, "min_between_gap": min_gap, "score": score, "attempt": attempt}
        if best is None or row["score"] > best["score"]:
            best = row
    assert best is not None
    return best


def _block_slices(multiplicity: int, block_dim: int) -> list[slice]:
    return [slice(j * block_dim, (j + 1) * block_dim) for j in range(multiplicity)]


def _offblock_fraction(W: torch.Tensor, multiplicity: int, block_dim: int) -> float:
    keep = torch.zeros_like(W)
    for sl in _block_slices(multiplicity, block_dim):
        keep[sl, sl] = W[sl, sl]
    total = W.square().sum().clamp_min(1e-30)
    return float(((W - keep).square().sum() / total).item())


def _intertwiner(ref_actions: list[torch.Tensor], cur_actions: list[torch.Tensor]) -> dict:
    b = ref_actions[0].size(0)
    cols = []
    for k in range(b * b):
        X = torch.zeros(b * b, dtype=torch.float64)
        X[k] = 1.0
        X = X.reshape(b, b)
        # cur * X = X * ref  => X maps reference coordinates into current block.
        cols.append(torch.cat([(C @ X - X @ R).reshape(-1) for R, C in zip(ref_actions, cur_actions)]))
    M = torch.stack(cols, dim=1)
    _u, s, vh = torch.linalg.svd(M)
    X = vh[-1].reshape(b, b)
    u, _sv, vh2 = torch.linalg.svd(X)
    Q = u @ vh2
    residual = torch.stack([(C @ Q - Q @ R).norm() / C.norm().clamp_min(1e-30) for R, C in zip(ref_actions, cur_actions)]).max()
    return {"Q": Q, "residual": float(residual), "smallest_singular": float(s[-1])}


def compile_repeated_irrep(actions: list[torch.Tensor], seed: int, relative_tol: float = 1e-9) -> dict:
    d = actions[0].size(0)
    comm = commutant_basis(actions, relative_tol=relative_tol)
    accepted, m, b = _infer_repeated_irrep(d, comm["dimension"])
    base = {
        "accepted": False,
        "dimension": d,
        "commutant_dimension": comm["dimension"],
        "commutant_smallest_singular_values": comm["smallest_singular_values"],
        "inferred_multiplicity": m,
        "inferred_block_dim": b,
    }
    if not accepted or m is None or b is None:
        base["reason"] = "commutant dimension is not a nontrivial square multiplicity compatible with total width"
        return base

    sep = _separator_from_commutant(comm["basis"], b, m, seed)
    U = sep["U"]
    block_actions = [U.T @ W @ U for W in actions]
    pre_off = [_offblock_fraction(W, m, b) for W in block_actions]
    slices = _block_slices(m, b)
    refs = [W[slices[0], slices[0]] for W in block_actions]
    Qs = [torch.eye(b, dtype=torch.float64)]
    inter_res = [0.0]
    for j in range(1, m):
        cur = [W[slices[j], slices[j]] for W in block_actions]
        it = _intertwiner(refs, cur)
        Qs.append(it["Q"])
        inter_res.append(it["residual"])
    S = torch.block_diag(*Qs)
    G = U @ S
    aligned = [G.T @ W @ G for W in actions]
    post_off = [_offblock_fraction(W, m, b) for W in aligned]
    ref_blocks = [W[slices[0], slices[0]].clone() for W in aligned]
    max_block_difference = 0.0
    for W, R in zip(aligned, ref_blocks):
        for sl in slices[1:]:
            diff = (W[sl, sl] - R).norm() / R.norm().clamp_min(1e-30)
            max_block_difference = max(max_block_difference, float(diff))

    reconstructed = [G @ torch.block_diag(*([R] * m)) @ G.T for R in ref_blocks]
    reconstruction_errors = [float((R - W).norm() / W.norm().clamp_min(1e-30)) for R, W in zip(reconstructed, actions)]
    base.update(
        {
            "accepted": True,
            "separator_within_eigen_spread": sep["within_spread"],
            "separator_min_between_gap": sep["min_between_gap"],
            "separator_score": sep["score"],
            "pre_alignment_offblock_fraction": pre_off,
            "post_alignment_offblock_fraction": post_off,
            "intertwiner_residuals": inter_res,
            "max_aligned_block_relative_difference": max_block_difference,
            "reconstruction_relative_errors": reconstruction_errors,
            "runtime_operator_entries_dense": len(actions) * d * d,
            "runtime_operator_entries_compiled": len(actions) * b * b,
            "operator_storage_compression_factor": (d / b) ** 2,
            "G": G,
            "ref_blocks": ref_blocks,
            "reconstructed": reconstructed,
        }
    )
    return base


def _distinct_eigenvalue_count(W: torch.Tensor, tol: float = 1e-6) -> int:
    eig = torch.linalg.eigvals(W.to(torch.complex128))
    remaining = list(range(eig.numel()))
    groups = 0
    while remaining:
        i = remaining.pop(0)
        z = eig[i]
        nxt = []
        for j in remaining:
            if abs(complex(eig[j].item()) - complex(z.item())) > tol:
                nxt.append(j)
        remaining = nxt
        groups += 1
    return groups


def max_word_spectral_distinct(actions: list[torch.Tensor], max_depth: int = 8) -> dict:
    words = [(torch.eye(actions[0].size(0), dtype=torch.float64), "I")]
    best = {"distinct": 1, "word": "I"}
    frontier = words
    for _depth in range(1, max_depth + 1):
        new = []
        for W, name in frontier:
            for j, A in enumerate(actions):
                Z = A @ W
                nm = name + ("A" if j == 0 else "B")
                distinct = _distinct_eigenvalue_count(Z)
                if distinct > best["distinct"]:
                    best = {"distinct": distinct, "word": nm}
                new.append((Z, nm))
        # For S3 the group closes quickly; deduplicate matrices to keep search tiny.
        uniq = []
        for Z, nm in new:
            if not any((Z - U).abs().max() < 1e-8 for U, _ in uniq):
                uniq.append((Z, nm))
        frontier = uniq
    return best


def _decode(codes: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return (F.normalize(z, dim=-1) @ F.normalize(codes, dim=-1).T).argmax(-1)


@torch.inference_mode()
def evaluate_execution(problem: dict, compiled: dict, seed: int, depth: int, examples: int) -> dict:
    if not compiled["accepted"]:
        return {"compiled_program_accuracy": None}
    codes = problem["codes"]
    A, B = compiled["reconstructed"]
    g = torch.Generator().manual_seed(seed + 312_000)
    idx = torch.randint(codes.size(0), (examples,), generator=g)
    z = codes[idx]
    target = idx.clone()
    m = problem["multiplicity"]
    for _ in range(depth):
        kind = torch.randint(2, (examples,), generator=g)
        za = F.normalize(z @ A.T, dim=-1)
        zb = F.normalize(z @ B.T, dim=-1)
        z = torch.where(kind[:, None].eq(0), za, zb)
        # label order: channel*3 + symbol. A cycles +1; B maps symbol s -> -s mod 3.
        channel = torch.div(target, 3, rounding_mode="floor")
        symbol = target % 3
        symbol = torch.where(kind.eq(0), (symbol + 1) % 3, (-symbol) % 3)
        target = channel * 3 + symbol
    pred = _decode(codes, z)
    return {
        "compiled_program_depth": depth,
        "compiled_program_accuracy": float(pred.eq(target).float().mean()),
        "compiled_target_cosine": float(F.cosine_similarity(z, codes[target], dim=-1).mean()),
    }


def _jsonable_compiled(compiled: dict) -> dict:
    skip = {"G", "ref_blocks", "reconstructed"}
    return {k: v for k, v in compiled.items() if k not in skip}


def run_one(multiplicity: int, seed: int, broken: bool, depth: int, examples: int) -> dict:
    problem = make_problem(multiplicity, seed, broken=broken)
    actions = [problem["A"], problem["B"]]
    spec = max_word_spectral_distinct(actions)
    compiled = compile_repeated_irrep(actions, seed)
    execution = evaluate_execution(problem, compiled, seed, depth, examples)
    return {
        "seed": seed,
        "multiplicity": multiplicity,
        "dimension": 2 * multiplicity,
        "broken_sharing_control": broken,
        "max_word_distinct_eigenvalues": spec,
        "compiler": _jsonable_compiled(compiled),
        "execution": execution,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/research/exp_031/metrics.json"))
    p.add_argument("--multiplicities", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--seeds", type=int, nargs="+", default=[110, 111, 112])
    p.add_argument("--program-depth", type=int, default=256)
    p.add_argument("--examples", type=int, default=1024)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    rows = []
    controls = []
    for seed in args.seeds:
        for m in args.multiplicities:
            row = run_one(m, seed, False, args.program_depth, args.examples)
            rows.append(row)
            c = row["compiler"]
            print(
                f"seed={seed} m={m} d={2*m} simple_best={row['max_word_distinct_eigenvalues']['distinct']} "
                f"comm={c['commutant_dimension']} accepted={c['accepted']} "
                f"block_err={c.get('max_aligned_block_relative_difference')} "
                f"R{args.program_depth}={row['execution']['compiled_program_accuracy']}"
            )
        # One broken-sharing control per seed at m=3.
        ctl = run_one(3, seed + 10_000, True, args.program_depth, args.examples)
        controls.append(ctl)
        print(
            f"CONTROL seed={seed} comm={ctl['compiler']['commutant_dimension']} "
            f"accepted={ctl['compiler']['accepted']}"
        )

    payload = {
        "experiment": EXPERIMENT_NAME,
        "protocol": {
            "base_irrep": "real 2D standard irrep of S3",
            "simple_spectrum_anchor_available": False,
            "compiler_input": "dense action matrices only; no semantic state labels",
            "compiler_method": "joint commutant -> symmetric separator -> invariant copies -> intertwiner alignment",
            "broken_control": "last repeated block replaced by inequivalent rotation angle",
            "program_depth": args.program_depth,
            "examples": args.examples,
        },
        "rows": rows,
        "broken_controls": controls,
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
