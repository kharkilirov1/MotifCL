#!/usr/bin/env python3
"""EXP-023: recover sparse operator motifs from a learned dense invariant representation.

EXP-022 learns a 30D latent representation and two *dense* linear generator
matrices A (x -> x+1) and M (x -> 3x) from transition data.  The learned gauge
is arbitrary, so a dense matrix in that gauge is not evidence that the operator
itself is intrinsically dense.

This experiment performs a semantics-free gauge recovery:

1. diagonalize the learned A action;
2. express M in the same learned spectral gauge;
3. measure whether M becomes monomial (one non-zero transition per eigenspace);
4. hard-prune A to its diagonal and M to one entry per input spectral mode;
5. transform the pruned actions back to the original real gauge and test long
   mixed affine programs.

No hand-written Fourier codebook or field-state basis is supplied to the gauge
recovery.  Dimension 29 is a matched insufficient-representation control.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from isotropic_affine_representation_experiment import train_one
from learned_affine_representation_experiment import Config, P, G, log_table

EXPERIMENT_NAME = "exp_023_spectral_gauge_motif_discovery"


def act_matrix(z: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.normalize(z @ weight.T, dim=-1)


def apply_power_matrix(z: torch.Tensor, weight: torch.Tensor, power: int) -> torch.Tensor:
    for _ in range(power):
        z = act_matrix(z, weight)
    return z


def spectral_sparsify(add_weight: torch.Tensor, mul_weight: torch.Tensor) -> dict:
    """Find A's eigen-gauge and sparsify A/M there.

    Returns complex spectral matrices and real reconstructed matrices.  The
    transformation is inferred only from A itself.
    """
    A = add_weight.detach().double()
    M = mul_weight.detach().double()
    eigvals, V = torch.linalg.eig(A)
    V_inv = torch.linalg.inv(V)
    A_c = A.to(torch.complex128)
    M_c = M.to(torch.complex128)
    A_spec = V_inv @ A_c @ V
    M_spec = V_inv @ M_c @ V

    A_sparse = torch.diag(torch.diagonal(A_spec))
    energy = M_spec.abs().square()
    # Keep one output eigenspace for each input eigenspace.  In the exact
    # affine representation this mapping is a permutation.
    output_for_input = energy.argmax(dim=0)
    M_sparse = torch.zeros_like(M_spec)
    cols = torch.arange(M_spec.size(1), device=M_spec.device)
    M_sparse[output_for_input, cols] = M_spec[output_for_input, cols]

    A_rec_c = V @ A_sparse @ V_inv
    M_rec_c = V @ M_sparse @ V_inv
    # Exact 30D runs have imaginary residuals at numerical noise after
    # reconstruction because the original actions are real.
    A_rec = A_rec_c.real.to(add_weight.dtype)
    M_rec = M_rec_c.real.to(mul_weight.dtype)

    row_energy = energy.sum(dim=1).clamp_min(1e-30)
    col_energy = energy.sum(dim=0).clamp_min(1e-30)
    row_top_fraction = energy.max(dim=1).values / row_energy
    col_top_fraction = energy.max(dim=0).values / col_energy
    row_effective_support = row_energy.square() / energy.square().sum(dim=1).clamp_min(1e-30)

    total_A = A_spec.abs().square().sum().clamp_min(1e-30)
    diag_A = torch.diagonal(A_spec).abs().square().sum()
    total_M = energy.sum().clamp_min(1e-30)
    kept_M = energy[output_for_input, cols].sum()

    return {
        "eigvals": eigvals,
        "V": V,
        "A_spec": A_spec,
        "M_spec": M_spec,
        "A_sparse": A_sparse,
        "M_sparse": M_sparse,
        "A_rec": A_rec,
        "M_rec": M_rec,
        "metrics": {
            "eigenvector_condition_number": float(torch.linalg.cond(V).real),
            "A_diagonal_energy_fraction": float((diag_A / total_A).real),
            "M_row_top1_energy_fraction_mean": float(row_top_fraction.mean().real),
            "M_row_top1_energy_fraction_min": float(row_top_fraction.min().real),
            "M_col_top1_energy_fraction_mean": float(col_top_fraction.mean().real),
            "M_col_top1_energy_fraction_min": float(col_top_fraction.min().real),
            "M_row_effective_support_mean": float(row_effective_support.mean().real),
            "M_kept_energy_fraction": float((kept_M / total_M).real),
            "A_sparse_support": int(A_sparse.ne(0).sum().item()),
            "M_sparse_support": int(M_sparse.ne(0).sum().item()),
            "spectral_matrix_entries_per_operator": int(A_spec.numel()),
            "A_reconstruction_relative_error": float(
                (A_rec_c - A_c).abs().norm() / A_c.abs().norm().clamp_min(1e-30)
            ),
            "M_reconstruction_relative_error": float(
                (M_rec_c - M_c).abs().norm() / M_c.abs().norm().clamp_min(1e-30)
            ),
            "A_reconstruction_imaginary_fraction": float(
                A_rec_c.imag.norm() / A_rec_c.abs().norm().clamp_min(1e-30)
            ),
            "M_reconstruction_imaginary_fraction": float(
                M_rec_c.imag.norm() / M_rec_c.abs().norm().clamp_min(1e-30)
            ),
        },
    }


def root_labels(eigvals: torch.Tensor) -> torch.Tensor:
    """Nearest 31st-root labels for analysis only (never used for sparsifying)."""
    angle = torch.remainder(torch.angle(eigvals), 2 * torch.pi)
    return torch.remainder(torch.round(angle * P / (2 * torch.pi)).to(torch.long), P)


@torch.inference_mode()
def relation_mapping_accuracy(eigvals: torch.Tensor, M_spec: torch.Tensor) -> float:
    """Check the semidirect eigenspace permutation after motif discovery.

    From M A = A^3 M, an A-eigenmode with root label k must be mapped by M to
    the mode k/3 mod 31.  This is diagnostic only; the expected mapping is not
    used to discover the sparse support.
    """
    if eigvals.numel() != P - 1:
        return float("nan")
    labels = root_labels(eigvals)
    if len(set(labels.tolist())) != P - 1 or 0 in labels.tolist():
        return 0.0
    out_index = M_spec.abs().argmax(dim=0)
    observed = labels[out_index]
    inv_g = pow(G, -1, P)
    expected = torch.remainder(labels * inv_g, P)
    return float(observed.eq(expected).float().mean())


@torch.inference_mode()
def evaluate_reconstructed(model, A_weight: torch.Tensor, M_weight: torch.Tensor, seed: int, depth: int, examples: int) -> dict:
    code = model.codebook()
    ids = torch.arange(P)
    exp_to, _ = log_table()

    A1 = act_matrix(code, A_weight)
    M1 = act_matrix(code, M_weight)
    add_acc = float(model.logits(A1).argmax(-1).eq((ids + 1) % P).float().mean())
    mul_acc = float(model.logits(M1).argmax(-1).eq((ids * G) % P).float().mean())

    g = torch.Generator().manual_seed(seed + 823000)
    value = torch.randint(P, (examples,), generator=g)
    state = code[value]
    for _ in range(depth):
        kind = torch.randint(2, (examples,), generator=g)
        add_b = torch.randint(P, (examples,), generator=g)
        mul_e = torch.randint(P - 1, (examples,), generator=g)
        nxt = torch.empty_like(state)
        for b in range(P):
            sel = torch.where(kind.eq(0) & add_b.eq(b))[0]
            if sel.numel():
                nxt[sel] = apply_power_matrix(state[sel], A_weight, b)
                value[sel] = (value[sel] + b) % P
        for e in range(P - 1):
            sel = torch.where(kind.eq(1) & mul_e.eq(e))[0]
            if sel.numel():
                nxt[sel] = apply_power_matrix(state[sel], M_weight, e)
                value[sel] = (value[sel] * exp_to[e]) % P
        state = nxt

    return {
        "generator_add_accuracy": add_acc,
        "generator_mul_accuracy": mul_acc,
        "mixed_program_depth": depth,
        "mixed_program_accuracy": float(model.logits(state).argmax(-1).eq(value).float().mean()),
        "mixed_program_target_cosine": float(F.cosine_similarity(state, code[value], dim=-1).mean()),
    }


def run_one(d: int, seed: int, train_steps: int, lr: float, depth: int, examples: int) -> dict:
    model, trace = train_one(d, seed, train_steps, lr, Config())
    recovered = spectral_sparsify(model.add1.weight, model.mul3.weight)
    metrics = recovered["metrics"]
    metrics["semidirect_eigenspace_mapping_accuracy"] = relation_mapping_accuracy(
        recovered["eigvals"], recovered["M_spec"]
    )
    metrics["sparse_reconstructed_execution"] = evaluate_reconstructed(
        model, recovered["A_rec"], recovered["M_rec"], seed, depth, examples
    )
    return {"seed": seed, "dimension": d, "train_trace": trace, "metrics": metrics}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/research/exp_023/metrics.json"))
    p.add_argument("--dimensions", type=int, nargs="+", default=[29, 30])
    p.add_argument("--seeds", type=int, nargs="+", default=[73, 74, 75])
    p.add_argument("--train-steps", type=int, default=1200)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--program-depth", type=int, default=64)
    p.add_argument("--examples", type=int, default=1024)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    rows = []
    for seed in args.seeds:
        for d in args.dimensions:
            row = run_one(d, seed, args.train_steps, args.lr, args.program_depth, args.examples)
            rows.append(row)
            m = row["metrics"]
            e = m["sparse_reconstructed_execution"]
            print(
                f"seed={seed} d={d} A_diag={m['A_diagonal_energy_fraction']:.6f} "
                f"M_top1={m['M_row_top1_energy_fraction_mean']:.6f} "
                f"M_eff={m['M_row_effective_support_mean']:.3f} "
                f"map={m['semidirect_eigenspace_mapping_accuracy']:.3f} "
                f"sparse_R{args.program_depth}={e['mixed_program_accuracy']:.6f}"
            )

    payload = {
        "experiment": EXPERIMENT_NAME,
        "protocol": {
            "training_semantics": "identical to EXP-022: only x->x+1 and x->3x plus isotropic code geometry",
            "gauge_discovery": "eigendecomposition of learned A only",
            "sparsification": "A diagonal; M one spectral entry per input eigenspace",
            "hand_written_fourier_basis": False,
            "program_depth": args.program_depth,
            "examples": args.examples,
        },
        "rows": rows,
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
