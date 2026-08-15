# EXP-009 — Generalization of chart transitions

Status: **PASSED negative result**  
Date: 2026-08-14

## Question

If operator-specific charts are useful, can a cheap linear map synchronize them
on identities it has never seen?

## Protocol

Fit dense linear bridges

- additive chart -> multiplicative chart;
- multiplicative chart -> additive chart;

on about 70% of the 30 non-zero `F_31` identities.  Evaluate identity recovery
on the held-out identities.  Seven hash-locked identity splits are used.

An additional all-identity fit checks expressivity: if all 30 identities are
shown, the same dense bridge should be able to interpolate the finite mapping.
