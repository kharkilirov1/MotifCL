# EXP-007 — Integrated latent register machine

Status: **PASSED (controlled integrated gate)**  
Date: 2026-08-14

## Construction

Combine the two independently learned laws:

1. Fourier/local modular-add ALU from EXP-004/005;
2. Fourier/local latent PC successor from EXP-006.

Program memory contains `ADD(operand)` instructions and one unique HALT.
Post-HALT cells are distractors.  The machine receives no external program
length.

The continuous value register is never snapped/decoded between instructions.
The continuous PC register is never replaced by an external integer position.

OOD programs have lengths 5–10.  Every correct-path arithmetic transition is
chosen from the ALU's held-out operand-pair split.

## Causal controls

- replace the ALU with the over-general Fourier/full interpolator;
- replace the PC law with the full prefix-memorizing controller;
- replace value geometry with random/full memorization;
- shuffle value register after hop 2;
- shift PC register after hop 2.
