# RESULT-020 — Perfect generator edges do not imply a learned operator algebra

Seed-0 dimension sweep:

- d=2/4 cannot reliably fit both generators;
- d=8,16,24,30 reach **100% top-1 on both trained generator transitions**;
- nevertheless random affine programs generated from powers of A/M stay near
  chance for d=8..30;
- d=31 reaches semidirect cosine ~0.999995 and ~99.6% mixed-program accuracy.

The learned code effective rank grows with dimension.  At d=30 without an
explicit geometry constraint it is only ~21.7, while d=31 reaches ~29.3.

Adding the semidirect consistency loss alone to d=30 makes the relation nearly
perfect but still does not stabilize all powers: the representation itself has
not occupied the required invariant subspace.

Conclusion: local generator classification and even a nearly-correct algebraic
relation can coexist with an insufficient/recurrently fragile representation.
This motivated EXP-022.

Evidence: `artifacts/research/exp_021/seed0.json`.
