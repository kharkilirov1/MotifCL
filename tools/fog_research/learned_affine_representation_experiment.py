#!/usr/bin/env python3
"""EXP-021: learn an invariant affine-group representation from two generators.

No Fourier codebook is supplied.  The model learns:
- one latent code E(x) for each x in F_31;
- one shared linear action A for x -> x+1;
- one shared linear action M for x -> 3x.

Training sees only those two generator transition tables plus code separation and
weak orthogonality.  It never receives arbitrary x+b or x*y transition labels.
After training, powers of A and M generate a much larger affine operator grammar.

A dimension sweep tests whether generator top-1 fit is enough, or whether a
wider invariant subspace is needed for the semidirect relation

    M A = A^3 M

and long mixed-program execution.
"""
from __future__ import annotations

import argparse, json, os
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

P=31; G=3; EXPERIMENT_NAME='exp_021_learned_affine_representation'


def log_table():
    exp_to=[pow(G,e,P) for e in range(P-1)]; value_to=[-1]*P
    for e,v in enumerate(exp_to): value_to[v]=e
    return exp_to,value_to


@dataclass(frozen=True)
class Config:
    head_scale:float=20.0
    separation_weight:float=0.05
    orthogonality_weight:float=0.1


class LearnedAffineRep(nn.Module):
    def __init__(self,d:int,seed:int,cfg:Config):
        super().__init__(); self.d=d; self.cfg=cfg; torch.manual_seed(seed)
        self.embedding=nn.Parameter(torch.randn(P,d))
        self.add1=nn.Linear(d,d,bias=False); self.mul3=nn.Linear(d,d,bias=False)
        nn.init.orthogonal_(self.add1.weight); nn.init.orthogonal_(self.mul3.weight)
    def codebook(self): return F.normalize(self.embedding,dim=-1)
    def act(self,z,which:str):
        layer=self.add1 if which=='A' else self.mul3
        return F.normalize(layer(z),dim=-1)
    def logits(self,z): return self.cfg.head_scale*F.normalize(z,dim=-1)@self.codebook().T


def train_one(d:int,seed:int,steps:int,lr:float,cfg:Config):
    m=LearnedAffineRep(d,seed,cfg); opt=torch.optim.Adam(m.parameters(),lr=lr); ids=torch.arange(P); eye=torch.eye(d); trace=[]
    for step in range(steps):
        c=m.codebook(); A=m.act(c,'A'); M=m.act(c,'M')
        add_loss=(1-F.cosine_similarity(A,c[(ids+1)%P],dim=-1)).mean()
        mul_loss=(1-F.cosine_similarity(M,c[(ids*G)%P],dim=-1)).mean()
        sep=F.cross_entropy(cfg.head_scale*c@c.T,ids)
        ort=((m.add1.weight.T@m.add1.weight-eye).square().mean()+
             (m.mul3.weight.T@m.mul3.weight-eye).square().mean())
        loss=add_loss+mul_loss+cfg.separation_weight*sep+cfg.orthogonality_weight*ort
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step==0 or (step+1)%max(1,steps//5)==0 or step+1==steps:
            trace.append({'step':step+1,'loss':float(loss.detach()),'add_loss':float(add_loss.detach()),'mul_loss':float(mul_loss.detach()),'separation_loss':float(sep.detach()),'orthogonality_loss':float(ort.detach())})
    return m.eval(),trace


@torch.inference_mode()
def apply_power(m,z,which,n):
    for _ in range(n): z=m.act(z,which)
    return z


@torch.inference_mode()
def evaluate(m,seed:int,program_depth:int,examples:int):
    c=m.codebook(); ids=torch.arange(P); exp_to,value_to=log_table()
    a=m.act(c,'A'); mm=m.act(c,'M')
    add_acc=float(m.logits(a).argmax(-1).eq((ids+1)%P).float().mean())
    mul_acc=float(m.logits(mm).argmax(-1).eq((ids*G)%P).float().mean())
    add_cos=float(F.cosine_similarity(a,c[(ids+1)%P],dim=-1).mean())
    mul_cos=float(F.cosine_similarity(mm,c[(ids*G)%P],dim=-1).mean())

    # Semidirect relation M A = A^3 M on the learned representation.
    left=m.act(m.act(c,'A'),'M')
    right=m.act(c,'M')
    right=apply_power(m,right,'A',3)
    semidirect=float(F.cosine_similarity(left,right,dim=-1).mean())

    add_power_acc=[]
    for b in range(P):
        z=apply_power(m,c,'A',b); target=(ids+b)%P
        add_power_acc.append(float(m.logits(z).argmax(-1).eq(target).float().mean()))
    mul_power_acc=[]
    for e in range(P-1):
        z=apply_power(m,c,'M',e); y=exp_to[e]; target=(ids*y)%P
        mul_power_acc.append(float(m.logits(z).argmax(-1).eq(target).float().mean()))

    g=torch.Generator().manual_seed(seed+123000)
    value=torch.randint(P,(examples,),generator=g); state=c[value]
    for _ in range(program_depth):
        kind=torch.randint(2,(examples,),generator=g)
        # High-level ADD_b and MUL_y are executed only by powers of the two
        # learned generators; no direct arbitrary-operand action was trained.
        add_b=torch.randint(P,(examples,),generator=g)
        mul_e=torch.randint(P-1,(examples,),generator=g)
        nxt=torch.empty_like(state)
        for b in range(P):
            sel=torch.where(kind.eq(0)&add_b.eq(b))[0]
            if sel.numel():
                nxt[sel]=apply_power(m,state[sel],'A',b); value[sel]=(value[sel]+b)%P
        for e in range(P-1):
            sel=torch.where(kind.eq(1)&mul_e.eq(e))[0]
            if sel.numel():
                nxt[sel]=apply_power(m,state[sel],'M',e); value[sel]=(value[sel]*exp_to[e])%P
        state=nxt
    program_acc=float(m.logits(state).argmax(-1).eq(value).float().mean())
    program_cos=float(F.cosine_similarity(state,c[value],dim=-1).mean())
    sv=torch.linalg.svdvals(c)
    effective_rank=float((sv.sum().square()/sv.square().sum()).item())
    return {
        'generator_add_accuracy':add_acc,'generator_mul_accuracy':mul_acc,
        'generator_add_cosine':add_cos,'generator_mul_cosine':mul_cos,
        'semidirect_cosine':semidirect,
        'all_add_power_accuracy_mean':sum(add_power_acc)/len(add_power_acc),
        'all_add_power_accuracy_min':min(add_power_acc),
        'all_mul_power_accuracy_mean':sum(mul_power_acc)/len(mul_power_acc),
        'all_mul_power_accuracy_min':min(mul_power_acc),
        'mixed_program_depth':program_depth,'mixed_program_accuracy':program_acc,
        'mixed_program_target_cosine':program_cos,'code_effective_rank':effective_rank,
    }


def write_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+f'.tmp.{os.getpid()}'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=Path('artifacts/research/exp_021/metrics.json')); p.add_argument('--dimensions',type=int,nargs='+',default=[2,4,8,16,24,30,31]); p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]); p.add_argument('--steps',type=int,default=1000); p.add_argument('--lr',type=float,default=.02); p.add_argument('--program-depth',type=int,default=8); p.add_argument('--examples',type=int,default=512); p.add_argument('--threads',type=int,default=4); args=p.parse_args(); torch.set_num_threads(args.threads); cfg=Config(); rows=[]
    for seed in args.seeds:
        for d in args.dimensions:
            m,trace=train_one(d,seed,args.steps,args.lr,cfg); rows.append({'seed':seed,'dimension':d,'trace':trace,'metrics':evaluate(m,seed,args.program_depth,args.examples)}); print(seed,d,rows[-1]['metrics']['generator_add_accuracy'],rows[-1]['metrics']['generator_mul_accuracy'],rows[-1]['metrics']['semidirect_cosine'],rows[-1]['metrics']['mixed_program_accuracy'])
    payload={'experiment':EXPERIMENT_NAME,'config':asdict(cfg),'training':{'only_generator_transition_labels':['x->x+1','x->3x'],'arbitrary_operand_transition_labels':False,'steps':args.steps,'lr':args.lr},'rows':rows}; write_json(args.output,payload)
if __name__=='__main__': main()
