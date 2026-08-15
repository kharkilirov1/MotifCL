# EXP-025 — Automatic operator-law discovery

Status: **PASSED**  
Date: 2026-08-14

Input to the compiler: only the learned dense `A` and `M` matrices.

The compiler searches:

- a finite order `n <= 64` minimizing `||A^n-I||`;
- a conjugacy exponent `r<n` minimizing `||M A M^{-1}-A^r||`;
- sparse spectral support from `A`'s learned eigengauge.

It receives no field modulus, multiplier value, Fourier basis or semantic
operation label.  A law is accepted only if both residuals are below `0.01`.

Controls: d=29 versus d=30 on seeds 73/74/75.
