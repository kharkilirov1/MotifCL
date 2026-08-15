# EXP-010 — Learn a cyclic chart from a successor law

Status: **PASSED for binary generalization; recurrence stability incomplete**  
Date: 2026-08-14

## Question

Can operator-compatible identity geometry be learned from a tiny local transition
law instead of installing a Fourier codebook by hand?

## Model

Each of 31 identities has six learnable phase coordinates.  The shared operator
is fixed per-plane complex multiplication.  The codebook therefore belongs to a
broad phase-chart family, but the phases themselves are learned from scratch.

Training exposes only successor facts `(x,1)->x+1`.

Arms:

- `closed_cycle`: includes the single closure edge `30+1->0`;
- `open_chain`: omits that edge.

A weak self-decoding/separation loss prevents unusable identity collisions.
No arbitrary binary addition pair is a training target.
