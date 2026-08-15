# EXP-008 — Operator-specific latent charts

Status: **PASSED**  
Date: 2026-08-14

## Question

Is one compact latent coordinate chart naturally simple for qualitatively
different operator families?

## Charts

Over non-zero `F_31` identities:

- additive Fourier chart: characters of `x` under addition mod 31;
- multiplicative log-Fourier chart: characters of `log_g(x)` in the cyclic
  multiplicative group `F_31*`, using primitive root `g=3`.

Both are 30-dimensional.

## Operators

Fit the same local 60-feature bilinear class for:

- addition (zero-output pairs excluded so the multiplicative chart has a valid
  codomain);
- multiplication.

Full 900-feature Kronecker controls measure brute interpolation capacity.
