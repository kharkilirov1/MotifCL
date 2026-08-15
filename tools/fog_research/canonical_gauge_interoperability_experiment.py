#!/usr/bin/env python3
"""EXP-029: canonical gauge recovery enables zero-shot latent module interoperability.

Independently train several d=30 EXP-022 representations.  Each run is free to
choose an arbitrary internal orthogonal gauge.  Recover a canonical gauge from
its learned A spectrum only:

1. sort A eigenmodes by their nearest finite-order root label;
2. fix each eigenvector phase using the shared identity state E(0) as anchor.

Then compare independently trained canonical codebooks and derive a bridge
between any two runs entirely from their recovered gauges.  No bridge network
is trained and no paired non-anchor identity examples are used.

Finally, execute part of a random affine program in one model, transfer the
continuous latent state through the structural bridge, and continue execution
in another model.
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path
import torch
from torch.nn import functional as F

from isotropic_affine_representation_experiment import train_one
from learned_affine_representation_experiment import Config, P, log_table

EXPERIMENT_NAME='exp_029_canonical_gauge_interoperability'


def canonicalize(model):
    A=model.add1.weight.detach().double(); M=model.mul3.weight.detach().double(); C=model.codebook().detach().double()
    eigvals,V=torch.linalg.eig(A); angle=torch.remainder(torch.angle(eigvals),2*torch.pi); labels=torch.remainder(torch.round(angle*P/(2*torch.pi)).long(),P)
    order=torch.argsort(labels); labels=labels[order]; V=V[:,order]
    Vi=torch.linalg.inv(V); Y=(Vi@C.T.to(torch.complex128)).T
    # One shared anchor is enough to fix the remaining per-eigenvector U(1) gauge.
    phase=Y[0]/Y[0].abs().clamp_min(1e-12); V=V@torch.diag(phase); Vi=torch.linalg.inv(V); Y=(Vi@C.T.to(torch.complex128)).T
    Acan=Vi@A.to(torch.complex128)@V; Mcan=Vi@M.to(torch.complex128)@V
    return {'V':V,'Vi':Vi,'labels':labels,'code':Y,'A':Acan,'M':Mcan}


def bridge_state(z:torch.Tensor,source,target):
    # column convention: z_target = V_target V_source^{-1} z_source
    out=(target['V']@source['Vi']@z.detach().double().T.to(torch.complex128)).T
    imag=float(out.imag.norm()/out.abs().norm().clamp_min(1e-30))
    return F.normalize(out.real.to(z.dtype),dim=-1),imag


def act(model,z,which): return model.act(z,which)


@torch.inference_mode()
def cross_program(source_model,target_model,source_can,target_can,seed:int,examples:int,pre_depth:int,post_depth:int):
    code=source_model.codebook(); g=torch.Generator().manual_seed(seed+1929000); value=torch.randint(P,(examples,),generator=g); z=code[value]; exp_to,_=log_table()
    def segment(model,z,value,steps):
        for _ in range(steps):
            kind=torch.randint(2,(examples,),generator=g); nxt=torch.empty_like(z)
            sel=torch.where(kind.eq(0))[0]
            if sel.numel(): nxt[sel]=model.act(z[sel],'A'); value[sel]=(value[sel]+1)%P
            sel=torch.where(kind.eq(1))[0]
            if sel.numel(): nxt[sel]=model.act(z[sel],'M'); value[sel]=(value[sel]*3)%P
            z=nxt
        return z,value
    z,value=segment(source_model,z,value,pre_depth); z,imag=bridge_state(z,source_can,target_can); z,value=segment(target_model,z,value,post_depth)
    pred=target_model.logits(z).argmax(-1); return {'accuracy':float(pred.eq(value).float().mean()),'target_cosine':float(F.cosine_similarity(z,target_model.codebook()[value],dim=-1).mean()),'bridge_imaginary_fraction':imag}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=Path('artifacts/research/exp_029/metrics.json')); p.add_argument('--seeds',type=int,nargs='+',default=[73,74,75]); p.add_argument('--train-steps',type=int,default=1200); p.add_argument('--examples',type=int,default=512); p.add_argument('--pre-depth',type=int,default=32); p.add_argument('--post-depth',type=int,default=32); p.add_argument('--threads',type=int,default=4); args=p.parse_args(); torch.set_num_threads(args.threads)
    models={}; cans={}
    for seed in args.seeds:
        m,_=train_one(30,seed,args.train_steps,.02,Config()); models[seed]=m; cans[seed]=canonicalize(m)
    pairs=[]
    for i,s in enumerate(args.seeds):
        for t in args.seeds[i+1:]:
            Ys=cans[s]['code']; Yt=cans[t]['code']; Ys=Ys/Ys.abs().square().sum(1,keepdim=True).sqrt(); Yt=Yt/Yt.abs().square().sum(1,keepdim=True).sqrt(); sim=(Ys.conj()*Yt).sum(1).real
            bridged,imag=bridge_state(models[s].codebook(),cans[s],cans[t]); transfer_acc=float(models[t].logits(bridged).argmax(-1).eq(torch.arange(P)).float().mean())
            prog=cross_program(models[s],models[t],cans[s],cans[t],s*100+t,args.examples,args.pre_depth,args.post_depth)
            row={'source_seed':s,'target_seed':t,'canonical_code_cosine_mean':float(sim.mean()),'canonical_code_cosine_min':float(sim.min()),'direct_state_transfer_accuracy':transfer_acc,'direct_bridge_imaginary_fraction':imag,'cross_model_program':prog}; pairs.append(row); print(s,t,row['canonical_code_cosine_mean'],transfer_acc,prog['accuracy'])
    payload={'experiment':EXPERIMENT_NAME,'protocol':{'bridge_training':False,'paired_nonanchor_identities_used_for_bridge':False,'phase_anchor':'shared E(0) only','pre_depth':args.pre_depth,'post_depth':args.post_depth},'pairs':pairs}; args.output.parent.mkdir(parents=True,exist_ok=True); tmp=args.output.with_suffix('.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,args.output)
if __name__=='__main__': main()
