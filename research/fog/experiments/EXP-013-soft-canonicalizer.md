# EXP-013 — Soft canonicalizer inside the transition

Status: **FAILED / INFORMATIVE**  
Date: 2026-08-14

## Hypothesis

EXP-012 bad seeds fail because arbitrary binary outputs lie off the canonical
identity manifold.  Insert a continuous soft codebook attractor into every
transition:

`raw bilinear -> softmax(code similarity) -> weighted code mixture -> next state`.

No hard argmax/snap and no arbitrary binary target labels.

## Fixed protocol

- modulus 31, four learned phase planes;
- flexible learned local bilinear operator;
- successor terminal depths 1/2/3;
- unlabeled identity, commutativity and associativity constraints;
- fixed canonicalizer scale 12;
- unseen init seeds 10/11/12.

## Falsifier

Canonical proximity improves but semantic binary/recurrent accuracy remains
poor.
