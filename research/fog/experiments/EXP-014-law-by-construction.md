# EXP-014 — Algebraic laws by construction vs by penalty

Status: **PASSED**  
Date: 2026-08-14

## Question

Does constraining the operator class to a norm-preserving associative local law
make joint chart/operator learning recurrently reliable?

## Structured arm

Each learned 2D latent plane is mapped into a learned orthogonal frame, combined
by complex multiplication, then mapped back.  The operation is commutative,
associative and norm preserving by construction.

Only repeated successor targets at depths 1/2/3 and code separation are trained.
No arbitrary binary-pair targets.

## Matched flexible arm

Same learned codebook and local width, but an unconstrained trainable bilinear
map plus identity/commutativity/associativity penalties.

New seeds: 20/21/22.  Evaluation depth: 64.
