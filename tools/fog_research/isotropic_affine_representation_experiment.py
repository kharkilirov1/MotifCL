#!/usr/bin/env python3
"""EXP-022: learn the affine invariant representation using isotropic geometry.

EXP-021 showed that generator transition top-1 can be perfect in small latent
dimensions while the generated affine grammar fails.  In d=30, the learned
codebook also collapsed to substantially lower effective rank.

This experiment adds no new semantic transition labels.  It only requires the
31 unit identity codes to be centered and isotropic in their available latent
space.  At d=30 this matches the geometry of the non-trivial zero-sum component
of the natural 31-point permutation representation.

Train only x->x+1 and x->3x; evaluate all generator powers and mixed affine
programs.  Dimensions below 30 are matched controls.
"""
from __future__ import annotations

import argparse, json, os
from dataclasses import asdict
from pathlib import Path
import torch
from torch.nn import functional as F

from learned_affine_representation_experiment import Config, LearnedAffineRep, evaluate

P=31; G=3; EXPERIMENT_NAME='exp_022_isotropic_affine_representation'


def train_one(d:int,seed:int,steps:int,lr:float,cfg:Config):
    torch.manual_seed(seed); m=LearnedAffineRep(d,seed,cfg); opt=torch.optim.Adam(m.parameters(),lr=lr); ids=torch.arange(P); eye=torch.eye(d); trace=[]
    for step in range(steps):
        c=m.codebook(); A=m.act(c,'A'); M=m.act(c,'M')
        la=(1-F.cosine_similarity(A,c[(ids+1)%P],dim=-1)).mean()
        lm=(1-F.cosine_similarity(M,c[(ids*G)%P],dim=-1)).mean()
        sep=F.cross_entropy(cfg.head_scale*c@c.T,ids)
        ort=((m.add1.weight.T@m.add1.weight-eye).square().mean()+
             (m.mul3.weight.T@m.mul3.weight-eye).square().mean())
        mean=c.mean(0); centered=c-mean; cov=centered.T@centered/P
        iso=d*(cov-eye/d).square().sum()+mean.square().sum()
        loss=la+lm+cfg.separation_weight*sep+cfg.orthogonality_weight*ort+iso
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step==0 or (step+1)%max(1,steps//5)==0 or step+1==steps:
            trace.append({'step':step+1,'loss':float(loss.detach()),'add_loss':float(la.detach()),'mul_loss':float(lm.detach()),'isotropy_loss':float(iso.detach())})
    return m.eval(),trace


def write_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+f'.tmp.{os.getpid()}'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=Path('artifacts/research/exp_022/metrics.json')); p.add_argument('--dimensions',type=int,nargs='+',default=[16,24,30]); p.add_argument('--seeds',type=int,nargs='+',default=[70,71,72]); p.add_argument('--steps',type=int,default=1200); p.add_argument('--lr',type=float,default=.02); p.add_argument('--program-depth',type=int,default=16); p.add_argument('--examples',type=int,default=1024); p.add_argument('--threads',type=int,default=4); args=p.parse_args(); torch.set_num_threads(args.threads); cfg=Config(); rows=[]
    for seed in args.seeds:
        for d in args.dimensions:
            m,trace=train_one(d,seed,args.steps,args.lr,cfg); met=evaluate(m,seed,args.program_depth,args.examples); rows.append({'seed':seed,'dimension':d,'trace':trace,'metrics':met}); print(seed,d,met['generator_add_accuracy'],met['generator_mul_accuracy'],met['semidirect_cosine'],met['mixed_program_accuracy'],met['code_effective_rank'])
    payload={'experiment':EXPERIMENT_NAME,'config':asdict(cfg),'training':{'semantic_transition_labels':['x->x+1','x->3x'],'arbitrary_operand_transition_labels':False,'geometric_constraint':'centered isotropic code covariance','steps':args.steps,'lr':args.lr},'rows':rows}; write_json(args.output,payload)
if __name__=='__main__': main()
