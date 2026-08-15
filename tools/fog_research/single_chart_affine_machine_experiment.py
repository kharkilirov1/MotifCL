#!/usr/bin/env python3
"""EXP-019: mixed ADD/MUL programs in one additive-character latent chart.

The full additive Fourier/character basis over F_p supports two different sparse
operator motifs without changing representation:

  chi_k(x+y) = chi_k(x) chi_k(y)      -> local complex product
  chi_k(x*y) = chi_{k*y}(x)           -> operand-conditioned frequency permutation

Thus non-commuting affine operations need not force a chart switch if the
operator grammar contains the right action motif.  MUL addresses the operand
identity continuously by cosine soft selection over the same canonical codebook;
there is no hard decode/snap between recurrent steps.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path

import torch
from torch.nn import functional as F

EXPERIMENT_NAME='exp_019_single_chart_affine_machine'


@dataclass(frozen=True)
class Config:
    prime:int=31
    address_scale:float=20.0

    @property
    def harmonics(self): return self.prime
    @property
    def d_model(self): return 2*self.prime


def codebook(cfg:Config)->torch.Tensor:
    x=torch.arange(cfg.prime,dtype=torch.float32)[:,None]
    k=torch.arange(cfg.prime,dtype=torch.float32)[None,:]
    phase=2*math.pi*x*k/cfg.prime
    return F.normalize(torch.cat((torch.cos(phase),torch.sin(phase)),dim=-1),dim=-1)


def _planes(z:torch.Tensor,h:int):
    r,i=z[...,:h],z[...,h:]; n=torch.sqrt(r.square()+i.square()).clamp_min(1e-9); return r/n,i/n


def add_op(state:torch.Tensor,operand:torch.Tensor,cfg:Config)->torch.Tensor:
    ar,ai=_planes(state,cfg.harmonics); br,bi=_planes(operand,cfg.harmonics)
    rr=ar*br-ai*bi; ii=ar*bi+ai*br
    return F.normalize(torch.cat((rr,ii),dim=-1),dim=-1)


def operand_weights(operand:torch.Tensor,code:torch.Tensor,cfg:Config)->torch.Tensor:
    return torch.softmax(cfg.address_scale*(F.normalize(operand,dim=-1)@code.T),dim=-1)


def mul_op(state:torch.Tensor,operand:torch.Tensor,cfg:Config,code:torch.Tensor,*,return_weights=False):
    # Candidate y acts as the sparse frequency permutation k -> k*y (mod p).
    b=state.shape[0]; h=cfg.harmonics
    sr,si=_planes(state,h)
    k=torch.arange(h,device=state.device)
    candidates=[]
    for y in range(cfg.prime):
        idx=(k*y)%cfg.prime
        candidates.append(torch.cat((sr[:,idx],si[:,idx]),dim=-1))
    cand=torch.stack(candidates,dim=1) # [B,y,D]
    w=operand_weights(operand,code,cfg)
    out=F.normalize((w[...,None]*cand).sum(dim=1),dim=-1)
    return (out,w) if return_weights else out


def decode(state:torch.Tensor,code:torch.Tensor)->torch.Tensor:
    return (F.normalize(state,dim=-1)@code.T).argmax(-1)


@torch.inference_mode()
def one_step_metrics(cfg:Config):
    code=codebook(cfg); ids=torch.arange(cfg.prime); a=ids.repeat_interleave(cfg.prime); b=ids.repeat(cfg.prime)
    add=add_op(code[a],code[b],cfg); add_t=(a+b)%cfg.prime
    mul,w=mul_op(code[a],code[b],cfg,code,return_weights=True); mul_t=(a*b)%cfg.prime
    # Wrong motif: local product in this chart is addition, not multiplication.
    wrong=add_op(code[a],code[b],cfg)
    return {
        'add_accuracy':float(decode(add,code).eq(add_t).float().mean()),
        'mul_accuracy':float(decode(mul,code).eq(mul_t).float().mean()),
        'wrong_local_mul_accuracy':float(decode(wrong,code).eq(mul_t).float().mean()),
        'mul_address_top_mass_mean':float(w.max(-1).values.mean()),
        'add_target_cosine_mean':float(F.cosine_similarity(add,code[add_t],dim=-1).mean()),
        'mul_target_cosine_mean':float(F.cosine_similarity(mul,code[mul_t],dim=-1).mean()),
    }


@torch.inference_mode()
def mixed_programs(cfg:Config,*,seed:int,depth:int,examples:int,wrong_mul:bool=False):
    code=codebook(cfg); g=torch.Generator().manual_seed(seed+depth*701)
    value=torch.randint(cfg.prime,(examples,),generator=g); state=code[value]
    op_kind=torch.randint(2,(examples,depth),generator=g) # 0 add,1 mul
    operand=torch.randint(cfg.prime,(examples,depth),generator=g)
    checkpoints={1,2,4,8,16,32,64,128,256,depth}; trajectory={}
    for t in range(depth):
        opv=operand[:,t]; addmask=op_kind[:,t].eq(0); mulmask=~addmask
        nxt=torch.empty_like(state)
        if addmask.any():
            nxt[addmask]=add_op(state[addmask],code[opv[addmask]],cfg)
            value[addmask]=(value[addmask]+opv[addmask])%cfg.prime
        if mulmask.any():
            if wrong_mul:
                nxt[mulmask]=add_op(state[mulmask],code[opv[mulmask]],cfg)
            else:
                nxt[mulmask]=mul_op(state[mulmask],code[opv[mulmask]],cfg,code)
            value[mulmask]=(value[mulmask]*opv[mulmask])%cfg.prime
        state=nxt; d=t+1
        if d in checkpoints:
            trajectory[str(d)]={
                'accuracy':float(decode(state,code).eq(value).float().mean()),
                'target_cosine_mean':float(F.cosine_similarity(state,code[value],dim=-1).mean()),
            }
    return trajectory


def write_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+f'.tmp.{os.getpid()}'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=Path('artifacts/research/exp_019/metrics.json')); p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]); p.add_argument('--depth',type=int,default=256); p.add_argument('--examples',type=int,default=4096); p.add_argument('--address-scale',type=float,default=20.0); p.add_argument('--threads',type=int,default=4); args=p.parse_args(); torch.set_num_threads(args.threads); cfg=Config(address_scale=args.address_scale)
    rows=[]
    for seed in args.seeds:
        rows.append({'seed':seed,'mixed':mixed_programs(cfg,seed=seed,depth=args.depth,examples=args.examples),'wrong_local_mul':mixed_programs(cfg,seed=seed,depth=min(args.depth,32),examples=args.examples,wrong_mul=True)})
    payload={'experiment':EXPERIMENT_NAME,'config':asdict(cfg),'one_step':one_step_metrics(cfg),'rows':rows,'hard_decode_between_hops':False,'chart_switches':0}; write_json(args.output,payload); print(json.dumps(payload,indent=2))

if __name__=='__main__': main()
