# IDEA-004 — Adaptive Halting From Latent Dynamics

Status: IDEA
Depends on: stable recurrent composition
Evidence: E0 hypothesis

## Goal

Use computation only while the latent machine is still changing meaningfully.

## Start without RL

Candidate observables:

- `||z_t - z_{t-1}||`;
- address entropy;
- top1-top2 address/logit margin;
- control-state phase;
- repeated-state/cycle detection.

A halting rule can first be deterministic/calibrated. Only introduce a learned controller if fixed dynamics cannot distinguish completion from ongoing computation.

## Important caveat

Not all correct computations converge to a fixed point. Counters, graph walks and iterative algorithms can keep changing forever. Therefore halting may need a control register/termination predicate rather than only `z_t ~= z_{t-1}`.
