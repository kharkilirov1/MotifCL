# IDEA-002 — Attractor / Error-Correcting Latent Dynamics

Status: IDEA
Depends on: EXP-001
Evidence: E0 hypothesis

## Problem

A soft binding hop may return

`z_hat = E(target) + error`.

If `z_hat` becomes the next query, errors can accumulate with depth and eventually turn a discrete trajectory into a diffuse mixture.

## Hypothesis

Useful latent reasoning states should behave as basins of attraction, not fragile exact points. A transition should map a neighborhood of the current identity toward a neighborhood of the next canonical identity:

`T(E(x) + noise) -> E(f(x))`.

## Measurements

For every hop t record:

- cosine to the oracle state code;
- top-1 state identity;
- address mass on the correct row;
- address entropy and top1-top2 margin;
- sensitivity to injected isotropic/adversarial noise.

Plot/record these by depth. Accuracy alone can hide gradual geometry collapse.

## Candidate mechanisms

1. Sharp cosine binding (baseline).
2. Canonicalizer/denoiser after payload selection.
3. Noise training around state codes.
4. Contrastive pull to the target identity manifold and push from alternatives.
5. Nearest-code projection only as an oracle diagnostic, not the desired differentiable mechanism.

## Decision test

If hard nearest-code recurrence works at long depth while soft recurrence collapses, the bottleneck is coordinate stability rather than composition capacity.
