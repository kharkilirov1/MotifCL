#!/usr/bin/env python3
"""EXP-027: transition-sample threshold in the learned 30D register.

Use the EXP-022 d=30 isotropic representation learner, but supervise the two
shared generator actions only on k of the 31 identity states.  All identities
still participate in code separation and isotropic geometry.

The experiment asks when a generic shared linear action is determined strongly
enough to extrapolate the held-out transition rows and the generated affine
grammar.  The key boundary is k=29 versus k=30.
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path
import torch
from torch.nn import functional as F

from learned_affine_representation_experiment import Config, LearnedAffineRep, P, G, evaluate

EXPERIMENT_NAME='exp_027_partial_transition_sample_complexity'


def train_one(k:int,seed:int,steps:int,lr:float):
    d=30; cfg=Config(); torch.manual_seed(seed); m=LearnedAffineRep(d,seed,cfg); opt=torch.optim.Adam(m.parameters(),lr=lr); ids=torch.arange(P); eye=torch.eye(d)
    g=torch.Generator().manual_seed(seed+12027); perm=torch.randperm(P,generator=g); observed=perm[:k]; heldout=perm[k:]; trace=[]
    for step in range(steps):
        c=m.codebook(); A=m.act(c[observed],'A'); M=m.act(c[observed],'M')
        la=(1-F.cosine_similarity(A,c[(observed+1)%P],dim=-1)).mean(); lm=(1-F.cosine_similarity(M,c[(observed*G)%P],dim=-1)).mean()
        sep=F.cross_entropy(cfg.head_scale*c@c.T,ids)
        ort=((m.add1.weight.T@m.add1.weight-eye).square().mean()+(m.mul3.weight.T@m.mul3.weight-eye).square().mean())
        mean=c.mean(0); centered=c-mean; cov=centered.T@centered/P; iso=d*(cov-eye/d).square().sum()+mean.square().sum()
        loss=la+lm+cfg.separation_weight*sep+cfg.orthogonality_weight*ort+iso
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step==0 or (step+1)%max(1,steps//5)==0 or step+1==steps: trace.append({'step':step+1,'loss':float(loss.detach()),'add_loss':float(la.detach()),'mul_loss':float(lm.detach()),'isotropy_loss':float(iso.detach())})
    return m.eval(),observed,heldout,trace


@torch.inference_mode()
def eval_split(m,observed,heldout,seed,depth,examples):
    c=m.codebook(); ids=torch.arange(P); pa=m.logits(m.act(c,'A')).argmax(-1); pm=m.logits(m.act(c,'M')).argmax(-1)
    def acc(pred,target,idx): return None if idx.numel()==0 else float(pred[idx].eq(target[idx]).float().mean())
    base=evaluate(m,seed,depth,examples)
    return {
        'observed_add_accuracy':acc(pa,(ids+1)%P,observed), 'observed_mul_accuracy':acc(pm,(ids*G)%P,observed),
        'heldout_add_accuracy':acc(pa,(ids+1)%P,heldout), 'heldout_mul_accuracy':acc(pm,(ids*G)%P,heldout),
        'heldout_count':int(heldout.numel()), 'semidirect_cosine':base['semidirect_cosine'],
        'mixed_program_accuracy':base['mixed_program_accuracy'],'code_effective_rank':base['code_effective_rank'],
    }


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=Path('artifacts/research/exp_027/metrics.json')); p.add_argument('--coverage',type=int,nargs='+',default=[24,29,30]); p.add_argument('--seeds',type=int,nargs='+',default=[90,91,92]); p.add_argument('--steps',type=int,default=1500); p.add_argument('--lr',type=float,default=.02); p.add_argument('--program-depth',type=int,default=16); p.add_argument('--examples',type=int,default=256); p.add_argument('--threads',type=int,default=4); args=p.parse_args(); torch.set_num_threads(args.threads); rows=[]
    for seed in args.seeds:
        for k in args.coverage:
            m,obs,hold,tr=train_one(k,seed,args.steps,args.lr); met=eval_split(m,obs,hold,seed,args.program_depth,args.examples); rows.append({'seed':seed,'observed_states':k,'observed_ids':obs.tolist(),'heldout_ids':hold.tolist(),'trace':tr,'metrics':met}); print(seed,k,met['heldout_add_accuracy'],met['heldout_mul_accuracy'],met['semidirect_cosine'],met['mixed_program_accuracy'])
    payload={'experiment':EXPERIMENT_NAME,'protocol':{'latent_dimension':30,'total_states':P,'semantic_labels':'only A/M transitions on observed identity subset','all_states_still_in_geometry_losses':True,'program_depth':args.program_depth},'rows':rows}; args.output.parent.mkdir(parents=True,exist_ok=True); tmp=args.output.with_suffix('.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,args.output)
if __name__=='__main__': main()
