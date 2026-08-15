# RESULT-002 — Terminal-only loss leaves periodic latent gauges

EXP-003 passed and produced an exact mechanistic counterexample to the idea that
correct terminal answers imply reusable canonical intermediates.

## Depth-2-only

All three seeds converged to

`phi = 3.141592... ≈ pi`.

Hence `G^2 ≈ I`, while `G` itself is maximally noncanonical:

- depth 1: 0% accuracy, cosine `-1`;
- depth 2: 100%, cosine `+1`;
- depth 3: 0%, cosine `-1`;
- depth 4: 100%, cosine `+1`;
- the same alternating pattern continues through depth 16.

The order-2 residual is about `1e-6` while the order-1 residual is approximately
`2`.

## Depth-3-only

All three seeds converged to

`phi = 2.094394... ≈ 2pi/3`.

The register follows a three-chart cycle:

- hops `1,2`: 0% canonical accuracy, cosine `-0.5`;
- hop `3`: 100%, cosine `+1`;
- then the pattern repeats.

## Coprime terminal depths `{2,3}`

With initialization inside the canonical basin, all three seeds converged to
`|phi| < 9e-7`.  Accuracy and latent cosine were effectively 100% / 1.0 at every
depth 1–16, including depth 1 which received no direct training loss.

This matches the exact algebra: in this commuting gauge family,

`G^2 = I` and `G^3 = I` imply `G = I`.

## Important negative result: optimization is separate from identifiability

The far-init `{2,3}` arm did **not** reach the canonical solution.  It converged
to a non-zero-loss local basin near `phi ≈ 2.33586`, with irregular depth
accuracy.  Therefore:

> coprime terminal depths remove exact periodic gauge degeneracy, but they do
> not by themselves make the optimization landscape benign.

## Interpretation

A recurrent latent architecture can be perfectly correct at every supervised
terminal depth while its intermediate representation uses a periodic hidden
coordinate chart.  Final-only supervision is therefore insufficient evidence
that a latent state is a reusable semantic register.

Evidence: `artifacts/research/exp_003/metrics.json`.
