#!/usr/bin/env python3
"""EXP-024: spectral motif projection as a recurrent denoiser.

Start from an exact d=30 learned affine representation (EXP-022), corrupt the
learned dense generator matrices with isotropic Gaussian weight noise, and
compare three execution paths:

1. noisy dense matrices;
2. support-only spectral projection discovered from A (EXP-023);
3. support + family-law projection: A eigenvalues are snapped to the nearest
   31st root of unity and the retained monomial M coefficients are normalized
   to unit magnitude.

The closure-aware projection uses the finite cyclic order of A and the
norm-preserving operator-family contract, but no hand-written Fourier basis and
no state transition table beyond the already learned matrices.
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path
import torch

from isotropic_affine_representation_experiment import train_one
from learned_affine_representation_experiment import Config, P
from spectral_gauge_motif_discovery_experiment import spectral_sparsify, evaluate_reconstructed

EXPERIMENT_NAME = "exp_024_motif_projection_denoising"


def closure_project(add_weight: torch.Tensor, mul_weight: torch.Tensor):
    A = add_weight.detach().double(); M = mul_weight.detach().double()
    eigvals, V = torch.linalg.eig(A); V_inv = torch.linalg.inv(V)
    A_spec = V_inv @ A.to(torch.complex128) @ V
    M_spec = V_inv @ M.to(torch.complex128) @ V

    # Finite-order cyclic family law: eigenvalues of A must be P-th roots.
    angles = torch.remainder(torch.angle(torch.diagonal(A_spec)), 2 * torch.pi)
    root_index = torch.round(angles * P / (2 * torch.pi))
    roots = torch.exp(1j * 2 * torch.pi * root_index / P)
    A_proj = torch.diag(roots)

    # Discovered monomial support, then unit-norm family law.
    energy = M_spec.abs().square(); out_idx = energy.argmax(dim=0)
    cols = torch.arange(M_spec.size(1)); coeff = M_spec[out_idx, cols]
    coeff = coeff / coeff.abs().clamp_min(1e-12)
    M_proj = torch.zeros_like(M_spec); M_proj[out_idx, cols] = coeff

    A_rec = (V @ A_proj @ V_inv).real.to(add_weight.dtype)
    M_rec = (V @ M_proj @ V_inv).real.to(mul_weight.dtype)
    return A_rec, M_rec


def noisy_copy(weight: torch.Tensor, sigma: float, generator: torch.Generator) -> torch.Tensor:
    d = weight.size(0)
    # Orthogonal dxd weights have O(1/sqrt(d)) entries; this makes sigma an
    # approximate relative Frobenius perturbation scale.
    return weight.detach() + sigma * torch.randn(weight.shape, generator=generator) / (d ** 0.5)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('artifacts/research/exp_024/metrics.json'))
    p.add_argument('--seeds',type=int,nargs='+',default=[73,74,75])
    p.add_argument('--noise',type=float,nargs='+',default=[0.0,0.03,0.05,0.10])
    p.add_argument('--train-steps',type=int,default=1200)
    p.add_argument('--program-depth',type=int,default=64)
    p.add_argument('--examples',type=int,default=128)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args(); torch.set_num_threads(args.threads)

    rows=[]
    for seed in args.seeds:
        model,_=train_one(30,seed,args.train_steps,0.02,Config())
        A=model.add1.weight.detach(); M=model.mul3.weight.detach()
        for sigma in args.noise:
            g=torch.Generator().manual_seed(seed*100003 + int(round(sigma*1_000_000)) + 17)
            An=noisy_copy(A,sigma,g); Mn=noisy_copy(M,sigma,g)
            dense=evaluate_reconstructed(model,An,Mn,seed,args.program_depth,args.examples)
            support=spectral_sparsify(An,Mn)
            sparse=evaluate_reconstructed(model,support['A_rec'],support['M_rec'],seed,args.program_depth,args.examples)
            Ac,Mc=closure_project(An,Mn)
            closure=evaluate_reconstructed(model,Ac,Mc,seed,args.program_depth,args.examples)
            row={
                'seed':seed,'noise_sigma':sigma,
                'dense':dense,'support_projection':sparse,'closure_projection':closure,
                'discovery':support['metrics'],
            }
            rows.append(row)
            print(seed,sigma,dense['mixed_program_accuracy'],sparse['mixed_program_accuracy'],closure['mixed_program_accuracy'])

    payload={
        'experiment':EXPERIMENT_NAME,
        'protocol':{
            'base_model':'EXP-022 d=30 isotropic learned affine representation',
            'noise':'Gaussian weight noise scaled by 1/sqrt(d)',
            'support_projection':'A spectral diagonal + discovered monomial M support',
            'closure_projection':'support projection + A^31=I root snapping + unit-magnitude M coefficients',
            'hand_written_fourier_basis':False,
            'program_depth':args.program_depth,'examples':args.examples,
        },'rows':rows,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True); tmp=args.output.with_suffix('.json.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,args.output)

if __name__=='__main__': main()
