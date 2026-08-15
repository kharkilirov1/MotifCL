#!/usr/bin/env python3
"""EXP-018: infer an operator from demos, then execute it recurrently.

A finite grammar contains robust learned ADD and MUL cyclic operator modules.
Two demonstrations provide no explicit operation label; latent consistency
selects the operator.  The selected module then executes a query chain by
feeding its own continuous output back as the next value register.  No hard
snap/decode occurs between hops.

Programs are homogeneous (all ADD or all MUL).  Mixed-operator chart switching
is deliberately left for a separate experiment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from generator_orbit_chart_experiment import Config as OrbitConfig, train_one
from operator_grammar_induction_experiment import (
    PRIME, discrete_log_table, candidate_demo_score,
    sample_nonzero_add_pair, sample_mul_pair,
)

EXPERIMENT_NAME='exp_018_inferred_operator_recurrent_execution'


def sample_nonambiguous_demos(name: str, k: int, g: torch.Generator):
    sampler=sample_nonzero_add_pair if name=='add' else sample_mul_pair
    while True:
        demos=[sampler(g) for _ in range(k)]
        ambiguous=all(((a+b)%PRIME==c) and ((a*b)%PRIME==c) for a,b,c in demos)
        if not ambiguous: return demos


@torch.inference_mode()
def execute_chain(name, model, start: int, operands: list[int], exp_to_value, value_to_exp):
    code=model.codebook()
    if name=='add':
        state=code[start:start+1]
        oracle=start
        for operand in operands:
            state=model.op(state,code[operand:operand+1])
            oracle=(oracle+operand)%PRIME
        pred=int(model.logits(state).argmax(-1)[0])
        return pred,oracle
    state=code[value_to_exp[start]:value_to_exp[start]+1]
    oracle=start
    for operand in operands:
        e=value_to_exp[operand]
        state=model.op(state,code[e:e+1])
        oracle=(oracle*operand)%PRIME
    pe=int(model.logits(state).argmax(-1)[0])
    return exp_to_value[pe],oracle


@torch.inference_mode()
def evaluate(add,mul,*,seed:int,depth:int,episodes:int,demos_k:int=2):
    exp_to_value,value_to_exp=discrete_log_table(); g=torch.Generator().manual_seed(seed+depth*1009+17000)
    route_ok=answer_ok=0; margins=[]
    for _ in range(episodes):
        true='add' if int(torch.randint(2,(1,),generator=g))==0 else 'mul'
        demos=sample_nonambiguous_demos(true,demos_k,g)
        sa=candidate_demo_score('add',add,demos,value_to_exp); sm=candidate_demo_score('mul',mul,demos,value_to_exp)
        pred_name='add' if sa>sm else 'mul'; route_ok+=int(pred_name==true); margins.append(abs(sa-sm))
        start=int(torch.randint(1,PRIME,(1,),generator=g))
        operands=[int(torch.randint(1,PRIME,(1,),generator=g)) for _ in range(depth)]
        pred,oracle=execute_chain(pred_name,add if pred_name=='add' else mul,start,operands,exp_to_value,value_to_exp)
        # Oracle above follows the selected law; answer correctness must follow true law.
        true_oracle=start
        for operand in operands:
            true_oracle=((true_oracle+operand)%PRIME) if true=='add' else ((true_oracle*operand)%PRIME)
        answer_ok+=int(pred==true_oracle)
    return {'depth':depth,'episodes':episodes,'route_accuracy':route_ok/episodes,'answer_accuracy':answer_ok/episodes,'mean_route_margin':sum(margins)/len(margins)}


def write_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+f'.tmp.{os.getpid()}'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=Path('artifacts/research/exp_018/metrics.json')); p.add_argument('--seeds',type=int,nargs='+',default=[60,61,62]); p.add_argument('--steps',type=int,default=600); p.add_argument('--episodes',type=int,default=1000); p.add_argument('--threads',type=int,default=4); args=p.parse_args(); torch.set_num_threads(args.threads)
    rows=[]
    for seed in args.seeds:
        add=train_one(OrbitConfig(order=31),seed=seed,arm='generator_orbit_closure',steps=args.steps,lr=.02)
        mul=train_one(OrbitConfig(order=30),seed=seed+1000,arm='generator_orbit_closure',steps=args.steps,lr=.02)
        rows.append({'seed':seed,'depths':[evaluate(add,mul,seed=seed,depth=d,episodes=args.episodes) for d in (1,2,4,8,16,32,64)]})
    payload={'experiment':EXPERIMENT_NAME,'explicit_operation_label':False,'demonstrations':2,'hard_snap_between_hops':False,'mixed_operator_programs':False,'rows':rows}; write_json(args.output,payload); print(json.dumps(payload,indent=2))

if __name__=='__main__': main()
