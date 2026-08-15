#!/usr/bin/env python3
"""EXP-017: robust finite operator grammar with orbit-closure modules.

EXP-015 showed that latent consistency can infer ADD vs MUL from demonstrations,
but a free-codebook MUL module sometimes learned a geometrically imperfect
executor.  EXP-016 identified a robust representation: generate all identities
as one learned orbit E(x)=T^xE(0) and impose the matching closure T^n=I.

This experiment plugs those robust learned operator modules back into the finite
grammar router.  Evaluation provides no operation label.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from generator_orbit_chart_experiment import Config as OrbitConfig, train_one
from operator_grammar_induction_experiment import evaluate_episodes

EXPERIMENT_NAME = "exp_017_robust_operator_grammar"


@torch.inference_mode()
def full_group_accuracy(model) -> float:
    code=model.codebook(); n=model.cfg.order; ids=torch.arange(n)
    a=ids.repeat_interleave(n); b=ids.repeat(n); target=(a+b)%n
    return float(model.logits(model.op(code[a],code[b])).argmax(-1).eq(target).float().mean())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('artifacts/research/exp_017/metrics.json'))
    p.add_argument('--seeds',type=int,nargs='+',default=[50,51,52])
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--episodes',type=int,default=3000)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args(); torch.set_num_threads(args.threads); rows=[]
    for seed in args.seeds:
        add=train_one(OrbitConfig(order=31),seed=seed,arm='generator_orbit_closure',steps=args.steps,lr=.02)
        mul=train_one(OrbitConfig(order=30),seed=seed+1000,arm='generator_orbit_closure',steps=args.steps,lr=.02)
        rows.append({
            'seed':seed,
            'add_full_group_accuracy':full_group_accuracy(add),
            'mul_exponent_group_accuracy':full_group_accuracy(mul),
            'episode_metrics':[evaluate_episodes(add,mul,seed=seed,demonstrations=k,episodes=args.episodes) for k in (1,2,3)],
        })
    payload={'experiment':EXPERIMENT_NAME,'operator_modules':'learned generator-orbit + matching closure; no arbitrary binary pair labels','explicit_operation_label_at_evaluation':False,'rows':rows}
    write_json(args.output,payload); print(json.dumps(payload,indent=2))

if __name__=='__main__': main()
