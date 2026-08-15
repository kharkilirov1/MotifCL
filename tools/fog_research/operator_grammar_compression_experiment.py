#!/usr/bin/env python3
"""EXP-026: infer a minimal generator set from several learned dense operators.

Learn one isotropic 30D identity representation and four dense actions from
transition tables only:

  A  : x -> x+1
  S3 : x -> 3x
  S5 : x -> 5x
  S7 : x -> 7x

No relations between the actions are given during training.  After training an
automatic compiler:

1. estimates each action's finite order;
2. chooses the highest-order scaling action as a primitive generator;
3. searches whether every other scaling action is a power of that primitive;
4. removes redundant dense scaling matrices;
5. executes random programs using only A and the primitive scaling generator.

Dimension 29 is the insufficient-representation control.
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from learned_affine_representation_experiment import Config, P
from automatic_operator_law_discovery_experiment import discover_order, rel_error

EXPERIMENT_NAME='exp_026_operator_grammar_compression'
SCALARS={'S3':3,'S5':5,'S7':7}


class MultiActionRep(nn.Module):
    def __init__(self,d:int,seed:int,cfg:Config):
        super().__init__(); self.d=d; self.cfg=cfg; torch.manual_seed(seed)
        self.embedding=nn.Parameter(torch.randn(P,d))
        self.actions=nn.ModuleDict({name:nn.Linear(d,d,bias=False) for name in ['A',*SCALARS]})
        for layer in self.actions.values(): nn.init.orthogonal_(layer.weight)
    def codebook(self): return F.normalize(self.embedding,dim=-1)
    def act(self,z,name): return F.normalize(self.actions[name](z),dim=-1)
    def logits(self,z): return self.cfg.head_scale*F.normalize(z,dim=-1)@self.codebook().T


def train_one(d:int,seed:int,steps:int,lr:float,cfg:Config):
    torch.manual_seed(seed); m=MultiActionRep(d,seed,cfg); opt=torch.optim.Adam(m.parameters(),lr=lr); ids=torch.arange(P); eye=torch.eye(d); trace=[]
    targets={'A':(ids+1)%P, **{name:(ids*y)%P for name,y in SCALARS.items()}}
    for step in range(steps):
        c=m.codebook(); trans=0.0; ort=0.0
        for name,target in targets.items():
            trans=trans+(1-F.cosine_similarity(m.act(c,name),c[target],dim=-1)).mean()
            W=m.actions[name].weight; ort=ort+(W.T@W-eye).square().mean()
        sep=F.cross_entropy(cfg.head_scale*c@c.T,ids)
        mean=c.mean(0); centered=c-mean; cov=centered.T@centered/P
        iso=d*(cov-eye/d).square().sum()+mean.square().sum()
        loss=trans+cfg.separation_weight*sep+cfg.orthogonality_weight*ort+iso
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step==0 or (step+1)%max(1,steps//5)==0 or step+1==steps:
            trace.append({'step':step+1,'loss':float(loss.detach()),'transition_loss':float(torch.as_tensor(trans).detach()),'isotropy_loss':float(iso.detach())})
    return m.eval(),trace


def matrix_power(W:torch.Tensor,n:int):
    return torch.linalg.matrix_power(W,n)


def search_power(base:torch.Tensor,target:torch.Tensor,order:int):
    rows=[]
    for e in range(order): rows.append({'exponent':e,'residual':rel_error(matrix_power(base,e),target)})
    return min(rows,key=lambda x:x['residual']),sorted(rows,key=lambda x:x['residual'])[:5]


def compile_grammar(model,max_order:int,threshold:float):
    order_info={}
    for name in SCALARS:
        best,top=discover_order(model.actions[name].weight.detach().double(),max_order)
        order_info[name]={'best':best,'top':top}
    primitive=max(SCALARS,key=lambda name:order_info[name]['best']['order'])
    base=model.actions[primitive].weight.detach().double(); base_order=order_info[primitive]['best']['order']
    relations={}
    accepted=order_info[primitive]['best']['residual']<threshold
    for name in SCALARS:
        best,top=search_power(base,model.actions[name].weight.detach().double(),base_order)
        relations[name]={'best':best,'top':top}
        accepted=accepted and best['residual']<threshold
    return {'accepted':accepted,'primitive':primitive,'primitive_order':base_order,'orders':order_info,'power_relations':relations}


def act_weight(z,W): return F.normalize(z@W.T,dim=-1)


@torch.inference_mode()
def evaluate(model,compiler,seed:int,depth:int,examples:int):
    c=model.codebook(); ids=torch.arange(P)
    generator_acc={}
    target={'A':(ids+1)%P,**{name:(ids*y)%P for name,y in SCALARS.items()}}
    for name,t in target.items(): generator_acc[name]=float(model.logits(model.act(c,name)).argmax(-1).eq(t).float().mean())
    if not compiler['accepted']:
        return {'generator_accuracy':generator_acc,'compiled_mixed_program_accuracy':None}
    base_name=compiler['primitive']; base=model.actions[base_name].weight.detach()
    # Compile every scaling action to a power of the primitive matrix.
    compiled={name:matrix_power(base,compiler['power_relations'][name]['best']['exponent']).to(base.dtype) for name in SCALARS}
    A=model.actions['A'].weight.detach()
    g=torch.Generator().manual_seed(seed+926000); value=torch.randint(P,(examples,),generator=g); z=c[value]
    names=list(SCALARS)
    for _ in range(depth):
        kind=torch.randint(4,(examples,),generator=g); nxt=torch.empty_like(z)
        sel=torch.where(kind.eq(0))[0]
        if sel.numel(): nxt[sel]=act_weight(z[sel],A); value[sel]=(value[sel]+1)%P
        for j,name in enumerate(names,1):
            sel=torch.where(kind.eq(j))[0]
            if sel.numel(): nxt[sel]=act_weight(z[sel],compiled[name]); value[sel]=(value[sel]*SCALARS[name])%P
        z=nxt
    return {'generator_accuracy':generator_acc,'compiled_mixed_program_depth':depth,'compiled_mixed_program_accuracy':float(model.logits(z).argmax(-1).eq(value).float().mean()),'compiled_target_cosine':float(F.cosine_similarity(z,c[value],dim=-1).mean())}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=Path('artifacts/research/exp_026/metrics.json')); p.add_argument('--dimensions',type=int,nargs='+',default=[29,30]); p.add_argument('--seeds',type=int,nargs='+',default=[80,81,82]); p.add_argument('--steps',type=int,default=1600); p.add_argument('--lr',type=float,default=.02); p.add_argument('--max-order',type=int,default=64); p.add_argument('--accept-residual',type=float,default=.01); p.add_argument('--program-depth',type=int,default=64); p.add_argument('--examples',type=int,default=256); p.add_argument('--threads',type=int,default=4); args=p.parse_args(); torch.set_num_threads(args.threads); rows=[]
    for seed in args.seeds:
        for d in args.dimensions:
            m,tr=train_one(d,seed,args.steps,args.lr,Config()); comp=compile_grammar(m,args.max_order,args.accept_residual); ev=evaluate(m,comp,seed,args.program_depth,args.examples); r={'seed':seed,'dimension':d,'trace':tr,'compiler':comp,'evaluation':ev}; rows.append(r); print(seed,d,comp['accepted'],comp['primitive'],{k:v['best'] for k,v in comp['power_relations'].items()},ev['compiled_mixed_program_accuracy'])
    payload={'experiment':EXPERIMENT_NAME,'protocol':{'training_labels':['x->x+1','x->3x','x->5x','x->7x'],'operator_relations_given_during_training':False,'compiler_semantic_metadata_used':False,'accept_residual':args.accept_residual},'rows':rows}; args.output.parent.mkdir(parents=True,exist_ok=True); tmp=args.output.with_suffix('.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,args.output)
if __name__=='__main__': main()
