# EXP-024 — Motif projection as recurrent denoising

Status: **PASSED controlled noise gate**  
Date: 2026-08-14

Corrupt the exact d=30 learned actions with Gaussian weight noise, then compare:

1. noisy dense execution;
2. spectral support projection from EXP-023;
3. support + operator-family closure projection.

The closure projection uses only the finite-order/norm-preserving family law:
`A` eigenvalues are snapped to nearest 31st roots and retained monomial `M`
coefficients are normalized to unit magnitude.  No Fourier basis is provided.
