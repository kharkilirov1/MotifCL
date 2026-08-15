# RESULT-007 — Different operator families prefer different charts

Mean over three pair splits:

| chart | operation | local train | local held-out | full held-out |
|---|---|---:|---:|---:|
| additive | ADD | **100%** | **100%** | 46.89% |
| additive | MUL | 32.61% | **0%** | 56.42% |
| multiplicative | ADD | 17.99% | **0%** | 40.47% |
| multiplicative | MUL | **100%** | **100%** | 63.51% |

The diagonal is exact: each group-Fourier chart makes its native group law a
small local bilinear operation.  The cross-operator local arms fail completely
on held-out pairs.

Again, full Kronecker capacity can fit every train pair but does not recover an
exact transferable law.

Interpretation:

> a single universal latent coordinate system is not automatically the right
> abstraction.  FOG may need an atlas of operator-compatible charts or typed
> registers, with explicit rules for chart transition.

Evidence: `artifacts/research/exp_008/metrics.json`.
