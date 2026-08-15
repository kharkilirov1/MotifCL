# EXP-022 — Learn the 30D invariant affine representation

Status: **PASSED**  
Date: 2026-08-14

Use the same learned codebook and two linear generator actions as EXP-021.
No new semantic transition labels are added.

Add only a representation-geometry constraint:

- code mean -> 0;
- centered code covariance -> isotropic over the available latent dimensions.

This prevents a high-dimensional codebook from using only a low-rank subspace.

Primary dimensions: 16,24,30 on seeds 70/71/72.
Boundary confirmation: 28,29,30 on new seeds 73/74/75.

Evaluation:

- two trained generator edges;
- semidirect relation `M A = A^3 M`;
- all powers of A and M;
- random mixed affine programs built only from powers of those two learned
  generators.
