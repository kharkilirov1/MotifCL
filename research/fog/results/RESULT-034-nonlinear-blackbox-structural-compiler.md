# RESULT-034 — black-box nonlinear dynamics can expose a compilable local operator grammar

EXP-035 passed all 12 main arms.

From a single hidden starting state, active exploration consistently discovered
14 orbit states.  Without semantic labels, the inferred operator graphs had
cycle signatures `[7,7]` and seven 2-cycles.

Mean fraction of depth-256 tangent trajectories with final cosine >0.99:

| Jacobian noise | raw finite-difference field | gauge synchronized | structurally compiled |
|---:|---:|---:|---:|
| 1% | 67.3% | 100.0% | **100.0%** |
| 3% | 22.7% | 94.9% | **100.0%** |
| 5% | 13.9% | 76.3% | **100.0%** |
| 10% | 7.1% | 44.3% | **100.0%** |

All operator-cycle laws passed the pre-registered holonomy gate in the main
1--10% sweep.

## Exploratory failure boundary

On seed 150 only, additional untuned high-noise probes showed:

- 15% noise: compiled >0.99 fraction **97.4%**;
- 20%: **87.7%**;
- 30%: the 7-cycle holonomy law is rejected and compiled execution falls back
  to the synchronized path;
- 40%: same abstention behavior.

So the method does not stay magically exact under arbitrary corruption.  The
structural evidence weakens before the compiler's strongest law is applied.

## Interpretation

This is the first current FOG gate where the structural compiler starts from a
**nonlinear black-box map**, discovers its own finite hidden context graph, and
estimates local Jacobians by active perturbation.

The compiled object is tangent-direction dynamics, not the full nonlinear state
map.  The orbit is finite and 2D/conformal, so this remains a controlled bridge
to production rather than evidence for arbitrary neural hidden dynamics.
