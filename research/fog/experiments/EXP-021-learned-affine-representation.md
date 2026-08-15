# EXP-021 — Learn an affine representation from two generators

Status: **DIAGNOSTIC / DIMENSION-SWEEP**  
Date: 2026-08-14

No Fourier coordinates are supplied.  Learn:

- 31 latent identity codes `E(x)`;
- one shared linear action `A: x -> x+1`;
- one shared linear action `M: x -> 3x`.

Training labels contain only those two generator transition tables, code
separation and weak action orthogonality.  No arbitrary `x+b` or `x*y` actions
are labeled.

After training, powers of A and M are used to generate the larger affine grammar.
Dimensions 2,4,8,16,24,30,31 are compared.
