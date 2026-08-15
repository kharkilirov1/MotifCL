# EXP-034 — state-conditioned local Jacobian gauge synchronization

Status: **PASSED controlled with abstention boundary**  
Date: 2026-08-14

## Question

Can a common latent operator grammar be recovered when there is **no single
global action matrix in observed coordinates**?

## Construction

Seven latent contexts each use their own hidden local 2D rotation gauge `H_x`.
The same logical operator is observed through a state-dependent local Jacobian

\[
J_{g,x}=H_{gx}R_gH_x^{-1}.
\]

The compiler receives only:

- the context transition graph;
- noisy local Jacobians.

It does not receive hidden gauges, shared canonical operators, or semantic
state codes.

A gauge-synchronization fit solves for local `H_x` plus shared O(2) actions.
No finite-order/group relation is imposed during this fit.  Separately, if the
observed A-edge graph is a single closed cycle and its measured holonomy is
near identity, the recovered A action is projected to the nearest root
compatible with that observed cycle length.

Fixed holonomy acceptance threshold: `0.40` relative identity residual.

Sweep:

- seeds 140,141,142;
- Jacobian noise 3%,5%,10%,15%;
- continuous-state depth 256.

Artifact: `artifacts/research/exp_034/metrics.json`.
