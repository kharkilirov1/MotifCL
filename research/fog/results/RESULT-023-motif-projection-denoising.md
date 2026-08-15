# RESULT-023 — Structural projection repairs noisy recurrent dynamics

Across seeds 73/74/75 at program depth 32:

| relative noise | noisy dense | support projection | support + closure |
|---:|---:|---:|---:|
| 0.00 | 100% | 100% | 100% |
| 0.03 | 95.83% mean | 99.48% mean | **100%** |
| 0.05 | 67.71% mean | 92.19% mean | **100%** |
| 0.10 | 3.65% mean | 31.25% mean | **100%** |

Interpretation: discovered sparse structure is useful not only for explanation
or compression.  Projection onto the legal operator family acts as a recurrent
denoiser and can restore long-horizon execution after the dense matrices have
become unusable.

Boundary: the closure arm knows the finite-order/norm-preserving family
contract.  It does not infer that contract in this experiment; EXP-025 attacks
that next.

Evidence: `artifacts/research/exp_024/metrics.json`.
