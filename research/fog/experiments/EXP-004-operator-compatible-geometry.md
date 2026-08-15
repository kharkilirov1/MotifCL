# EXP-004 — Geometry × operator inductive bias

Status: **PASSED (controlled gate)**  
Date: 2026-08-14

## Question

Can a small latent operator learn addition on unseen operand pairs because the
latent geometry exposes the algebra, rather than because the operator has enough
capacity to memorize a table?

## Task

Predict `E((a+b) mod 31)` from `E(a), E(b)`.

Pair split is hash-locked at roughly 70/30.  Every scalar identity appears as a
left and right operand in both train and test; only the pair combinations are
held out.

## Matched axes

Geometries:

- frozen Fourier group code;
- frozen random code of the same dimension.

Operator feature classes:

- `local`: 4 bilinear products per 2D harmonic plane (`60` features);
- `full`: all pairwise products (`900` features).

The readout is fit by deterministic ridge regression, avoiding optimizer
confounds.

## Reproduction

```bash
python operator_compatible_geometry_experiment.py \
  --output artifacts/research/exp_004/metrics.json
```
