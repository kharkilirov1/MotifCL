# NOTE-001 — Why Soft Recurrent Binding Does Not Drift in EXP-001

Status: derived explanation of RESULT-001
Date: 2026-08-14

## Observation

EXP-001 uses N=8 orthogonal canonical state codes and a permutation table. After several recurrent hops, the protected state is not exactly one-hot, yet accuracy remains 100% even at depth 64.

Empirically the final-hop statistics converge to:

- correct address mass: `0.510181...`;
- cosine to the oracle one-hot identity: `0.9400218...`.

This can be predicted from a one-dimensional recurrence.

## Symmetry reduction

Let the current query direction have one distinguished component `a_t` on the correct state and an equal component `b_t` on each of the other `N-1` states.

Because attention weights sum to one:

`b_t = (1 - a_t)/(N - 1)`.

Let

`r_t = sqrt(a_t^2 + (N-1)b_t^2)`.

With cosine address comparison and logit scale `s`, the next correct attention mass is

`a_{t+1} = exp(s a_t/r_t) / [exp(s a_t/r_t) + (N-1) exp(s b_t/r_t)]`.

Every wrong state receives

`b_{t+1} = (1-a_{t+1})/(N-1)`.

A permutation payload merely moves which coordinate is distinguished; it does not change this shape. RMS normalization rescales all coordinates uniformly and therefore also preserves the direction/ratio.

So the full N-dimensional recurrent trajectory collapses to the scalar map

`a_{t+1} = T_s,N(a_t)`.

## Fixed point for the trained EXP-001 model

For

- `N = 8`,
- learned `s = 2.4493381977`,

iteration of the scalar map gives

`a* = 0.5101811241`.

The corresponding wrong-state mass is

`b* = 0.0699741251`.

The cosine of this soft code with the correct canonical state is

`a* / sqrt(a*^2 + 7 b*^2) = 0.9400217957`.

These values match the depth-64 experiment:

- measured correct mass: `0.5101811886`;
- measured cosine: `0.9400219321`.

The local derivative of the scalar map at the fixed point is approximately

`T'(a*) ~= 0.547`,

so the fixed point is contractive (`|T'| < 1`). This explains why the soft-code shape rapidly stabilizes instead of accumulating error linearly with depth.

## Research implication

The important object may not be an exact vector code. It can be an **equivalence basin**: a family of soft vectors whose distinguished identity is preserved under the transition operator.

This suggests two future directions:

1. deliberately design/train contractive semantic attractors rather than forcing exact one-hot-like states;
2. measure the stability spectrum/Jacobian of learned latent transitions as a predictor of depth extrapolation.

The second point may become more informative than final-task accuracy alone.
