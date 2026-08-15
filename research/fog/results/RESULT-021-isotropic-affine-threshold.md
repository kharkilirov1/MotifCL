# RESULT-021 — Learned affine representation has a sharp 30D threshold

No Fourier basis and no arbitrary-operand transition labels are used.

## Seeds 70/71/72

All dimensions 16,24,30 reach 100% top-1 on the two locally supervised generator
edges.  Effective rank is driven to the available width by isotropy.

Yet mixed affine program accuracy is:

- d=16: ~3–4%;
- d=24: ~3%;
- **d=30: 100% on 3/3 seeds**.

At d=30:

- generator cosines: 1;
- semidirect cosine: 1;
- all ADD-generator powers: 100%;
- all MUL-generator powers: 100%;
- mixed depth-16 affine programs: 100%;
- effective rank: 30.

## Boundary seeds 73/74/75

- d=28: mixed ~3.2–3.7%;
- d=29: mixed ~4.7–6.8%;
- **d=30: 100% for all three seeds**.

The experimental threshold exactly matches NOTE-007's 30D invariant-subspace
argument for an exact linear equivariant representation of the full affine
generator action on `F_31`.

This is the first experiment in the repo where a useful shared invariant latent
representation is **learned from generator transition data rather than supplied
as a Fourier codebook**, while the required width is predicted by the operator
algebra.

Evidence: `artifacts/research/exp_022/metrics.json`.
