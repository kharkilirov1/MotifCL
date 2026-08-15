# EXP-035 — nonlinear black-box structural compiler

Status: **PASSED controlled**  
Date: 2026-08-14

## Question

Can the compiler recover recurrent operator structure from a genuinely nonlinear
black-box latent map when semantic coordinates, context graph and Jacobians are
all hidden?

## Hidden system

Observed coordinates are a nonlinear complex-plane warp

\[
w=\phi(z)=z+\alpha z^2.
\]

Two semantic dihedral actions are applied in `z` and mapped back through
`phi`.  On the discovered orbit, neither black-box operator is well fit by one
observed global linear matrix (relative least-squares errors about 0.161 and
0.129).

## Compiler input

Only:

- one starting hidden state;
- black-box callable operators.

The compiler then:

1. actively explores and deduplicates the hidden orbit;
2. applies operators to discovered states to infer the edge graph;
3. estimates local Jacobians by central finite differences;
4. extracts polar/tangent actions;
5. gauge-synchronizes the local field into shared O(2) actions;
6. infers graph cycle lengths and measures local-Jacobian holonomy;
7. applies finite-order projection only when the fixed holonomy gate passes.

No semantic `z`, warp inverse, hidden charts, true shared actions, precomputed
Jacobians, or context labels are provided to compilation.

## Main protocol

- seeds 150,151,152;
- Jacobian perturbation 1%,3%,5%,10%;
- depth 256;
- 1,024 tangent trajectories per run;
- fixed holonomy threshold 0.40.

Artifact: `artifacts/research/exp_035/metrics.json`.

A separate high-noise exploratory arm is stored at
`artifacts/research/exp_035/high_noise_seed150.json` and is not used to tune the
main sweep.
