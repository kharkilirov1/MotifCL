# IDEA-010 — online nonlinear structural compiler

Status: ACTIVE FRONTIER

## Hypothesis

A production FOG model can learn with a flexible neural recurrent transition,
while a sidecar compiler periodically probes hidden trajectories and discovers
stable reusable operator structure from local neighborhoods.

The compiler need not decode semantic identities.  Candidate evidence can come
from:

- JVP/VJP sketches around recurrent states;
- approximate commutants of local linearizations;
- repeated joint invariant blocks;
- closed-loop holonomy;
- low perturbation gain / closure defect;
- repeated transition motifs across examples.

Accepted motifs can be projected into a cheaper/stabler compiled path.  States
with insufficient evidence remain on the neural fallback path.

## Next falsifiable gate

Move EXP-035 from a finite discovered orbit to a **continuous cloud of hidden
states**.  Infer local neighborhoods without a known discrete context count,
use low-rank Jacobian-vector probes rather than full 2D Jacobians, and test
whether the same operator family is recovered across neighborhoods.
