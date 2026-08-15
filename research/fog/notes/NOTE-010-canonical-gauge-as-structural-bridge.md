# NOTE-010 — Canonical gauges can replace learned CASTs for the same algebra

Suppose two latent modules implement equivalent representations of the same
operator algebra:

\[
A_2 = B A_1 B^{-1},\qquad M_2 = B M_1 B^{-1}.
\]

If a shared operator has simple, identifiable spectrum, diagonalizing that
operator fixes the basis up to permutation and per-eigenvector phase.  Finite
order fixes the eigenvalue labels; one shared anchor state can fix the remaining
phases.

After this canonicalization both modules expose the same coordinates, so the
bridge is the change-of-gauge map implied by their canonical bases rather than a
new learned semantic function.

EXP-029 verifies this numerically to near machine precision across independent
training seeds and transfers mid-program recurrent states without supervision.

This suggests an architectural distinction:

- **different algebra / genuinely different representation:** learn or design a
  CAST;
- **same algebra / different gauge:** canonicalize, do not waste capacity on a
  learned bridge.
