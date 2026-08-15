# EXP-012 — Jointly learn chart and local operator

Status: **PARTIAL / OPTIMIZATION-UNSTABLE**  
Date: 2026-08-14

## Question

Can the operator itself be learned jointly with the chart while arbitrary binary
addition pairs remain unlabeled?

## Model

- 31 identities with learned phase charts;
- four harmonic planes (8D state);
- each plane has a trainable `4 -> 2` bilinear map instead of fixed complex
  multiplication.

## Supervised facts

Only repeated successor programs (`+1`) at depths 1,2,3.
No direct `(a,b)->a+b` target labels.

## Arms

1. `successor_only`: generator consistency + code separation.
2. `algebraic`: additionally **unlabeled** self-consistency:
   - identity;
   - commutativity;
   - associativity.

Three seeds, 1000 steps.
