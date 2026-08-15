# RESULT-005 — Shared PC law extrapolates; prefix table does not

| PC geometry/operator | seen successor | unseen successor | ID value/HALT | OOD value | OOD HALT |
|---|---:|---:|---:|---:|---:|
| Fourier + local shared | **100%** | **100%** | **100% / 100%** | **100%** | **100%** |
| Fourier + full | **100%** | 12.5% | **100% / 100%** | 6.45% | **0%** |
| random + full (3 seeds) | **100%** | 4.17% mean | **100% / 100%** | 8.11% mean | **0%** |

The full controllers have sufficient capacity to interpolate the observed PC
prefix exactly.  Their failure begins precisely where a new source position
requires a transition law that was never directly stored.

The local Fourier controller instead recovers a translation-shared `PC+1` law,
so the same transition works at positions never seen during fitting.

Interpretation:

> recurrent reuse of parameters is not by itself algorithmic generalization;
> the parameterization must expose a shared transition law rather than a finite
> state-transition table.

Evidence: `artifacts/research/exp_006/metrics.json`.
