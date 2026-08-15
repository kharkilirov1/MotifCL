# EXP-006 — Latent program counter + unique HALT

Status: **PASSED (supervised control-plane micro-gate)**  
Date: 2026-08-14

## Question

Can a continuous PC register learn a *shared successor law* from a short prefix
and then address instruction memory / HALT correctly at unseen positions and
program lengths?

## Protocol

The PC successor is fit only on transitions

`0->1, 1->2, 2->3, 3->4`.

Programs contain exactly one HALT at position `L`.  Every position after HALT is
a real non-HALT distractor operation.  At runtime the program length is not
passed to the machine.  The continuous PC state addresses program memory by
cosine compare/select.  The outer loop is only a safety cap.

The value register is a latent one-hot vector updated by frozen permutation ALU
operators so that this experiment isolates the control plane.

This is **not terminal-only training**: successor fitting is directly
supervised on the prefix transitions.

## Arms

- Fourier PC + local block-shared successor;
- Fourier PC + unconstrained full successor;
- random PC + unconstrained full successor.

ID programs have lengths 1–4; OOD programs have lengths 5–10.
