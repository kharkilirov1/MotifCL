#!/usr/bin/env python3
"""EXP-028: structural completion reduces the transition-sample requirement.

For the d=30 isotropic register of EXP-027, train A/M on only k observed source
states, then apply the automatically discovered spectral-motif compiler.

Linear-algebra prediction for an orthogonal d-dimensional action:
- k=d-2 leaves an O(2) continuous ambiguity on the unseen complement;
- k=d-1 leaves only O(1)={+1,-1}, a discrete orientation ambiguity;
- k=d removes the ambiguity.

The experiment tests whether the sparse finite-operator motif can resolve the
last d-1 orientation ambiguity while failing, as it should, when a continuous
O(2) ambiguity remains.
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path
import torch

from partial_transition_sample_complexity_experiment import train_one, eval_split
from spectral_gauge_motif_discovery_experiment import spectral_sparsify, evaluate_reconstructed
from motif_projection_denoising_experiment import closure_project

EXPERIMENT_NAME='exp_028_structural_completion_sample_threshold'


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=Path('artifacts/research/exp_028/metrics.json')); p.add_argument('--coverage',type=int,nargs='+',default=[28,29,30]); p.add_argument('--seeds',type=int,nargs='+',default=[90,91,92]); p.add_argument('--steps',type=int,default=1500); p.add_argument('--program-depth',type=int,default=16); p.add_argument('--examples',type=int,default=128); p.add_argument('--threads',type=int,default=4); args=p.parse_args(); torch.set_num_threads(args.threads); rows=[]
    for seed in args.seeds:
        for k in args.coverage:
            m,obs,hold,tr=train_one(k,seed,args.steps,.02); A=m.add1.weight.detach(); M=m.mul3.weight.detach(); base=eval_split(m,obs,hold,seed,args.program_depth,args.examples)
            support=spectral_sparsify(A,M); support_exec=evaluate_reconstructed(m,support['A_rec'],support['M_rec'],seed,args.program_depth,args.examples)
            Ac,Mc=closure_project(A,M); closure_exec=evaluate_reconstructed(m,Ac,Mc,seed,args.program_depth,args.examples)
            row={'seed':seed,'observed_states':k,'unobserved_complement_dimension':30-k,'heldout_ids':hold.tolist(),'determinants':{'A':float(torch.linalg.det(A.double())),'M':float(torch.linalg.det(M.double()))},'base':base,'motif_discovery':support['metrics'],'support_projection':support_exec,'closure_projection':closure_exec,'trace':tr}; rows.append(row)
            print(seed,k,'base',base['mixed_program_accuracy'],'support',support_exec['mixed_program_accuracy'],'closure',closure_exec['mixed_program_accuracy'],'detM',row['determinants']['M'])
    payload={'experiment':EXPERIMENT_NAME,'protocol':{'latent_dimension':30,'operator_class':'approximately orthogonal shared linear actions','prediction':'ambiguity group O(30-k)','program_depth':args.program_depth},'rows':rows}; args.output.parent.mkdir(parents=True,exist_ok=True); tmp=args.output.with_suffix('.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,args.output)
if __name__=='__main__': main()
