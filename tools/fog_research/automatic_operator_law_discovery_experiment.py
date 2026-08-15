#!/usr/bin/env python3
"""EXP-025: automatically infer finite operator laws from learned dense actions.

Given only the learned dense generator matrices from EXP-022, infer:

- whether A has a finite order n (search n <= max_order);
- whether M conjugates A to a power A^r (search r < n);
- the sparse spectral support discovered from A's eigengauge.

No field modulus, multiplier, Fourier basis, or semantic operation label is used
by the discovery procedure.  If the finite-order and conjugacy residuals pass a
fixed threshold, compile the pair to a closure-projected sparse grammar and test
long recurrent execution.  d=29 is the insufficient-representation control.
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path
import torch

from isotropic_affine_representation_experiment import train_one
from learned_affine_representation_experiment import Config
from spectral_gauge_motif_discovery_experiment import spectral_sparsify, evaluate_reconstructed

EXPERIMENT_NAME='exp_025_automatic_operator_law_discovery'


def rel_error(x:torch.Tensor,y:torch.Tensor)->float:
    return float((x-y).norm()/y.norm().clamp_min(1e-12))


def discover_order(A:torch.Tensor,max_order:int):
    d=A.size(0); I=torch.eye(d,dtype=A.dtype,device=A.device); cur=I; rows=[]
    for n in range(1,max_order+1):
        cur=cur@A; rows.append({'order':n,'residual':rel_error(cur,I)})
    best=min(rows,key=lambda x:x['residual'])
    return best, sorted(rows,key=lambda x:x['residual'])[:5]


def discover_conjugacy(A:torch.Tensor,M:torch.Tensor,order:int):
    d=A.size(0); I=torch.eye(d,dtype=A.dtype,device=A.device)
    conj=M@A@torch.linalg.inv(M); cur=I; rows=[]
    for r in range(order):
        Ar=I if r==0 else cur@A
        if r>0: cur=Ar
        rows.append({'exponent':r,'residual':rel_error(conj,Ar)})
    return min(rows,key=lambda x:x['residual']), sorted(rows,key=lambda x:x['residual'])[:5]


def compile_discovered(Aw:torch.Tensor,Mw:torch.Tensor,order:int):
    A=Aw.detach().double(); M=Mw.detach().double(); eigvals,V=torch.linalg.eig(A); Vi=torch.linalg.inv(V)
    As=Vi@A.to(torch.complex128)@V; Ms=Vi@M.to(torch.complex128)@V
    angles=torch.remainder(torch.angle(torch.diagonal(As)),2*torch.pi)
    roots=torch.exp(1j*2*torch.pi*torch.round(angles*order/(2*torch.pi))/order)
    Asp=torch.diag(roots)
    energy=Ms.abs().square(); out_idx=energy.argmax(dim=0); cols=torch.arange(Ms.size(1)); coeff=Ms[out_idx,cols]; coeff=coeff/coeff.abs().clamp_min(1e-12)
    Msp=torch.zeros_like(Ms); Msp[out_idx,cols]=coeff
    return (V@Asp@Vi).real.to(Aw.dtype),(V@Msp@Vi).real.to(Mw.dtype)


def run_one(d:int,seed:int,steps:int,max_order:int,threshold:float,depth:int,examples:int):
    model,_=train_one(d,seed,steps,.02,Config()); A=model.add1.weight.detach().double(); M=model.mul3.weight.detach().double()
    order,order_top=discover_order(A,max_order); conj,conj_top=discover_conjugacy(A,M,order['order'])
    support=spectral_sparsify(model.add1.weight,model.mul3.weight)
    accepted=order['residual']<threshold and conj['residual']<threshold
    compiled=None
    if accepted:
        Ac,Mc=compile_discovered(model.add1.weight,model.mul3.weight,order['order'])
        compiled=evaluate_reconstructed(model,Ac,Mc,seed,depth,examples)
    return {
        'seed':seed,'dimension':d,'accepted_operator_grammar':accepted,
        'discovered_order':order,'top_order_candidates':order_top,
        'discovered_conjugacy':conj,'top_conjugacy_candidates':conj_top,
        'spectral_support':support['metrics'],'compiled_execution':compiled,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=Path('artifacts/research/exp_025/metrics.json')); p.add_argument('--dimensions',type=int,nargs='+',default=[29,30]); p.add_argument('--seeds',type=int,nargs='+',default=[73,74,75]); p.add_argument('--train-steps',type=int,default=1200); p.add_argument('--max-order',type=int,default=64); p.add_argument('--accept-residual',type=float,default=.01); p.add_argument('--program-depth',type=int,default=64); p.add_argument('--examples',type=int,default=256); p.add_argument('--threads',type=int,default=4); args=p.parse_args(); torch.set_num_threads(args.threads)
    rows=[]
    for seed in args.seeds:
        for d in args.dimensions:
            r=run_one(d,seed,args.train_steps,args.max_order,args.accept_residual,args.program_depth,args.examples); rows.append(r); c=r['compiled_execution']; print(seed,d,r['accepted_operator_grammar'],r['discovered_order'],r['discovered_conjugacy'],None if c is None else c['mixed_program_accuracy'])
    payload={'experiment':EXPERIMENT_NAME,'protocol':{'discovery_inputs':['dense A matrix','dense M matrix'],'semantic_metadata_used':False,'max_order':args.max_order,'accept_residual':args.accept_residual,'compiled_program_depth':args.program_depth},'rows':rows}; args.output.parent.mkdir(parents=True,exist_ok=True); tmp=args.output.with_suffix('.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,args.output)
if __name__=='__main__': main()
