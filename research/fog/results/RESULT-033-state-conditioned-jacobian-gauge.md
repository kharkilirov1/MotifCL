# RESULT-033 — shared grammar can be compiled from a state-conditioned Jacobian field

EXP-034 removes the assumption that one global action matrix exists in the
observed latent gauge.

Mean fraction of depth-256 trajectories with final cosine > 0.99:

| Jacobian noise | raw local | gauge synchronized | synchronized + accepted cycle law |
|---:|---:|---:|---:|
| 3% | 35.8% | 72.9% | **100%** |
| 5% | 21.1% | 59.0% | **100%** |
| 10% | 11.4% | 43.8% | **100%** |
| 15% | 7.6% | 33.6% | 69.9% |

The 15% aggregate contains an intentional abstention boundary:

- 2/3 seeds still had sufficiently small measured cycle holonomy and were
  projected to **100%**;
- seed 142 had holonomy residual `0.5355 > 0.40`, so the compiler **did not**
  apply the finite-order projection and remained near the synchronized baseline.

At noise <=10%, all 9 runs passed the pre-registered holonomy gate and all 9
returned to 100% depth-256 tracking.

## Interpretation

Structural compilation can operate on a **field of local Jacobians** rather
than one global matrix.  Gauge synchronization recovers common operators, and
closed-loop holonomy provides observable evidence for when a stronger recurrent
law is safe to apply.

The important design behavior is conservative: insufficient structural evidence
causes **abstention**, not forced canonicalization.

This remains a controlled piecewise/local-linear gate.  Contexts and their edge
graph are known, local gauges are 2D orthogonal, and Jacobians are directly
observed.  Arbitrary nonlinear hidden dynamics remain open.
