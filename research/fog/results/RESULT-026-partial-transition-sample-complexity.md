# RESULT-026 — Transition fit has a spanning-set boundary

At k=24 all observed transitions reach 100%, while held-out A/M transitions and
mixed programs remain poor.

At k=30 (one source identity held out), all three seeds recover the missing A/M
rows and the complete mixed grammar at 100%.

At k=29, the result bifurcates:

- seeds 90/91: complete grammar 100%;
- seed 92: held-out ADD 100%, held-out MUL 0%, mixed ~1.6%.

This is not well described as a smooth data scaling curve.  EXP-028 identifies
it as the discrete orientation ambiguity left by an almost-complete orthogonal
operator.

Evidence: `artifacts/research/exp_027/metrics.json`.
