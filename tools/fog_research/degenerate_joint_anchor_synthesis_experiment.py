#!/usr/bin/env python3
"""EXP-030: synthesize a simple-spectrum anchor from degenerate learned operators.

Train only two scaling actions over F_31:

  B : x -> 27x = 3^3 x   (order 10)
  C : x -> 25x = 3^10 x (order 3)

Neither action has a simple spectrum on the 30D zero-sum representation.  The
structural compiler is not told the underlying multiplier or primitive
generator.  It searches compositions B^a C^b, accepts finite-order candidates,
and selects one with maximal discovered order.  Because the two exponents 3
and 10 generate Z_30, the joint grammar contains a synthesized order-30 action
with simple spectrum even though no such operator was trained directly.

The compiler then expresses B and C as powers of the synthesized primitive and
tests recurrent programs using only that new primitive action.
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from learned_affine_representation_experiment import Config, P
from automatic_operator_law_discovery_experiment import discover_order
from operator_grammar_compression_experiment import search_power, matrix_power, act_weight

EXPERIMENT_NAME='exp_030_degenerate_joint_anchor_synthesis'
MULT={'B':pow(3,3,P),'C':pow(3,10,P)}


class DegenerateRep(nn.Module):
    def __init__(self,d:int,seed:int,cfg:Config):
        super().__init__(); self.d=d; self.cfg=cfg; torch.manual_seed(seed)
        self.embedding=nn.Parameter(torch.randn(P,d))
        self.actions=nn.ModuleDict({k:nn.Linear(d,d,bias=False) for k in MULT})
        for layer in self.actions.values(): nn.init.orthogonal_(layer.weight)
    def codebook(self): return F.normalize(self.embedding,dim=-1)
    def act(self,z,k): return F.normalize(self.actions[k](z),dim=-1)
    def logits(self,z): return self.cfg.head_scale*F.normalize(z,dim=-1)@self.codebook().T


def train_one(d:int,seed:int,steps:int,lr:float):
    cfg=Config(); m=DegenerateRep(d,seed,cfg); opt=torch.optim.Adam(m.parameters(),lr=lr); ids=torch.arange(P); eye=torch.eye(d); trace=[]
    for step in range(steps):
        c=m.codebook(); trans=0.0; ort=0.0
        for name,y in MULT.items():
            trans=trans+(1-F.cosine_similarity(m.act(c,name),c[(ids*y)%P],dim=-1)).mean(); W=m.actions[name].weight; ort=ort+(W.T@W-eye).square().mean()
        sep=F.cross_entropy(cfg.head_scale*c@c.T,ids); mean=c.mean(0); centered=c-mean; cov=centered.T@centered/P; iso=d*(cov-eye/d).square().sum()+mean.square().sum(); loss=trans+cfg.separation_weight*sep+cfg.orthogonality_weight*ort+iso
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step==0 or (step+1)%max(1,steps//5)==0 or step+1==steps: trace.append({'step':step+1,'loss':float(loss.detach()),'transition_loss':float(torch.as_tensor(trans).detach()),'isotropy_loss':float(iso.detach())})
    return m.eval(),trace


def distinct_root_count(W:torch.Tensor,order:int):
    eig=torch.linalg.eigvals(W.detach().double()); ang=torch.remainder(torch.angle(eig),2*torch.pi); labels=torch.remainder(torch.round(ang*order/(2*torch.pi)).long(),order); return len(set(labels.tolist()))


def synthesize(model,max_order:int,threshold:float):
    B=model.actions['B'].weight.detach().double(); C=model.actions['C'].weight.detach().double(); ob,_=discover_order(B,max_order); oc,_=discover_order(C,max_order)
    candidates=[]
    for a in range(ob['order']):
        for b in range(oc['order']):
            if a==0 and b==0: continue
            T=matrix_power(B,a)@matrix_power(C,b); order,_=discover_order(T,max_order); accepted=order['residual']<threshold
            distinct=distinct_root_count(T,order['order']) if accepted else 0
            candidates.append({'a':a,'b':b,'order':order['order'],'order_residual':order['residual'],'distinct_eigen_roots':distinct,'accepted':accepted})
    good=[x for x in candidates if x['accepted']]
    best=max(good,key=lambda x:(x['distinct_eigen_roots'],x['order'],-(x['a']+x['b']))) if good else None
    if best is None: return {'accepted':False,'B_order':ob,'C_order':oc,'best':None}
    T=matrix_power(B,best['a'])@matrix_power(C,best['b']); rel={}
    for name,W in [('B',B),('C',C)]: rel[name]=search_power(T,W,best['order'])[0]
    accepted=all(v['residual']<threshold for v in rel.values())
    return {'accepted':accepted,'B_order':ob,'C_order':oc,'B_distinct_roots':distinct_root_count(B,ob['order']) if ob['residual']<threshold else 0,'C_distinct_roots':distinct_root_count(C,oc['order']) if oc['residual']<threshold else 0,'best':best,'power_relations':rel,'T':T}


@torch.inference_mode()
def evaluate(model,comp,seed:int,depth:int,examples:int):
    c=model.codebook(); ids=torch.arange(P); gen={name:float(model.logits(model.act(c,name)).argmax(-1).eq((ids*y)%P).float().mean()) for name,y in MULT.items()}
    if not comp['accepted']: return {'generator_accuracy':gen,'compiled_program_accuracy':None}
    T=comp['T'].to(c.dtype); compiled={name:matrix_power(T,comp['power_relations'][name]['exponent']).to(c.dtype) for name in MULT}
    g=torch.Generator().manual_seed(seed+3030000); value=torch.randint(P,(examples,),generator=g); z=c[value]
    for _ in range(depth):
        kind=torch.randint(2,(examples,),generator=g); nxt=torch.empty_like(z)
        for j,name in enumerate(['B','C']):
            sel=torch.where(kind.eq(j))[0]
            if sel.numel(): nxt[sel]=act_weight(z[sel],compiled[name]); value[sel]=(value[sel]*MULT[name])%P
        z=nxt
    return {'generator_accuracy':gen,'compiled_program_depth':depth,'compiled_program_accuracy':float(model.logits(z).argmax(-1).eq(value).float().mean()),'compiled_target_cosine':float(F.cosine_similarity(z,c[value],dim=-1).mean())}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=Path('artifacts/research/exp_030/metrics.json')); p.add_argument('--dimensions',type=int,nargs='+',default=[30]); p.add_argument('--seeds',type=int,nargs='+',default=[100,101,102]); p.add_argument('--steps',type=int,default=1400); p.add_argument('--lr',type=float,default=.02); p.add_argument('--max-order',type=int,default=64); p.add_argument('--accept-residual',type=float,default=.01); p.add_argument('--program-depth',type=int,default=64); p.add_argument('--examples',type=int,default=256); p.add_argument('--threads',type=int,default=4); args=p.parse_args(); torch.set_num_threads(args.threads); rows=[]
    for seed in args.seeds:
        for d in args.dimensions:
            m,tr=train_one(d,seed,args.steps,args.lr); comp=synthesize(m,args.max_order,args.accept_residual); ev=evaluate(m,comp,seed,args.program_depth,args.examples); comp_json={k:v for k,v in comp.items() if k!='T'}; rows.append({'seed':seed,'dimension':d,'trace':tr,'compiler':comp_json,'evaluation':ev}); print(seed,d,comp_json,ev['compiled_program_accuracy'])
    payload={'experiment':EXPERIMENT_NAME,'protocol':{'trained_actions':MULT,'primitive_action_given':False,'compiler_semantic_metadata_used':False,'program_depth':args.program_depth},'rows':rows}; args.output.parent.mkdir(parents=True,exist_ok=True); tmp=args.output.with_suffix('.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,args.output)
if __name__=='__main__': main()
