# RESULT-019 — Operator orbit gives a controlled latent-width scaling law

For every tested subgroup size `q = 1,2,3,5,6,10,15,30` and all three random
program seeds:

- chosen frequency set is invariant under the allowed multiplier set;
- mixed depth-128 accuracy: **100%**;
- target cosine: **1.0**.

For every proper subgroup, the first tested multiplier outside the subgroup has
zero frequency-closure fraction: it moves the chosen orbit into a disjoint
coset.

Thus in this representation family a width `2q` exactly supports a multiplier
operator set whose frequency orbit has size `q`.

This is not a universal lower bound for arbitrary nonlinear networks.  It is an
exact closure/scaling statement for sparse character-basis actions, and it gives
FOG a concrete way to think about representation budget:

> store enough coordinates to remain invariant under the operator grammar you
> want to execute cheaply.

Evidence: `artifacts/research/exp_020/metrics.json`.
