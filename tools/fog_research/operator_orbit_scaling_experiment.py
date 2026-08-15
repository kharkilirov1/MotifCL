#!/usr/bin/env python3
"""EXP-020: representation width tracks the orbit of the allowed operator set.

In an additive-character representation, scaling x -> y*x sends frequency
k -> k*y.  If allowed multipliers form a subgroup H of F_p^*, choosing the
frequency set S=H makes S exactly invariant and needs |H| complex harmonics.
Mixed ADD plus MUL-by-H programs can therefore stay in one chart of width 2|H|.

For the full multiplicative group, the action is transitive on nonzero
frequencies, so the nontrivial invariant orbit has p-1 harmonics.  This is a
controlled scaling law for this representation/operator family.
"""
from __future__ import annotations

import argparse, json, math, os
from pathlib import Path
import torch
from torch.nn import functional as F

P=31; G=3; EXPERIMENT_NAME='exp_020_operator_orbit_scaling'
DIVISORS=(1,2,3,5,6,10,15,30)


def subgroup(size:int):
    step=(P-1)//size
    return sorted({pow(G,step*j,P) for j in range(size)})


def codes(freqs:list[int]):
    x=torch.arange(P,dtype=torch.float32)[:,None]; k=torch.tensor(freqs,dtype=torch.float32)[None,:]
    ph=2*math.pi*x*k/P
    return F.normalize(torch.cat((torch.cos(ph),torch.sin(ph)),dim=-1),dim=-1)


def planes(z,h):
    r,i=z[...,:h],z[...,h:]; n=torch.sqrt(r*r+i*i).clamp_min(1e-9); return r/n,i/n


def add(state,operand,h):
    ar,ai=planes(state,h); br,bi=planes(operand,h); return F.normalize(torch.cat((ar*br-ai*bi,ar*bi+ai*br),-1),dim=-1)


def mul(state,y:int,freqs:list[int]):
    h=len(freqs); r,i=planes(state,h); pos={k:j for j,k in enumerate(freqs)}
    idx=[pos[(k*y)%P] for k in freqs]
    return F.normalize(torch.cat((r[:,idx],i[:,idx]),-1),dim=-1)


def decode(z,code): return (F.normalize(z,dim=-1)@code.T).argmax(-1)


@torch.inference_mode()
def run(size:int,seed:int,depth:int,examples:int):
    H=subgroup(size); freq=H; code=codes(freq); h=len(freq); g=torch.Generator().manual_seed(seed+size*1000)
    value=torch.randint(P,(examples,),generator=g); state=code[value]
    mul_choices=torch.tensor(H)
    for _ in range(depth):
        kind=torch.randint(2,(examples,),generator=g); operand=torch.randint(P,(examples,),generator=g)
        nxt=torch.empty_like(state); am=kind.eq(0); mm=~am
        if am.any():
            nxt[am]=add(state[am],code[operand[am]],h); value[am]=(value[am]+operand[am])%P
        if mm.any():
            choice=mul_choices[torch.randint(len(H),(int(mm.sum()),),generator=g)]
            # Apply batches grouped by multiplier to keep the structural action explicit.
            rows=torch.where(mm)[0]
            for y in H:
                sel=rows[choice.eq(y)]
                if sel.numel():
                    nxt[sel]=mul(state[sel],y,freq); value[sel]=(value[sel]*y)%P
        state=nxt
    invariant=all(((k*y)%P) in set(freq) for k in freq for y in H)
    outside=[y for y in range(1,P) if y not in set(H)]
    outside_closure=None
    if outside:
        y=outside[0]; outside_closure=sum(((k*y)%P) in set(freq) for k in freq)/len(freq)
    return {'subgroup_size':size,'real_width':2*size,'frequencies':freq,'invariant_under_allowed_multipliers':invariant,'example_outside_multiplier_closure_fraction':outside_closure,'mixed_depth':depth,'mixed_accuracy':float(decode(state,code).eq(value).float().mean()),'target_cosine_mean':float(F.cosine_similarity(state,code[value],dim=-1).mean())}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=Path('artifacts/research/exp_020/metrics.json')); p.add_argument('--depth',type=int,default=128); p.add_argument('--examples',type=int,default=4096); p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]); args=p.parse_args(); rows=[{'seed':s,'sizes':[run(q,s,args.depth,args.examples) for q in DIVISORS]} for s in args.seeds]; payload={'experiment':EXPERIMENT_NAME,'prime':P,'primitive_root':G,'rows':rows}; args.output.parent.mkdir(parents=True,exist_ok=True); tmp=args.output.with_suffix('.json.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,args.output); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
