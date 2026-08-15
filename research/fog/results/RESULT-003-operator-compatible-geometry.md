# RESULT-003 — Structure generalizes; capacity memorizes

Across three pair splits and three random-code seeds where applicable:

| geometry | operator | train acc | held-out pair acc |
|---|---|---:|---:|
| Fourier | local 60-feature | **100%** | **100%** |
| Fourier | full 900-feature | **100%** | **56.67% mean** |
| random | local 60-feature | 29.77% mean | 6.91% mean |
| random | full 900-feature | **100%** | **2.85% mean** |

Chance is `1/31 = 3.23%`.

The random/full arm is the decisive capacity control: it has enough bilinear
features to interpolate every training pair exactly, yet its held-out accuracy
is at chance.  It learned a finite table, not the addition law.

The Fourier/full result is also important.  Good representation geometry alone
is not sufficient: an excessively unconstrained operator class can interpolate
training data using a non-compositional solution.  The local harmonic operator
recovers the algebraic law exactly and reaches cosine 1.0 on all held-out pairs.

Interpretation:

> reusable latent computation depends on the *joint inductive bias* of the
> representation geometry and the operator family.

Evidence: `artifacts/research/exp_004/metrics.json`.
