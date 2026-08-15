# EXP-019 — Mixed non-commuting operators in one invariant chart

Status: **PASSED CONTROLLED**  
Date: 2026-08-14

## Motivation

EXP-008 showed ADD and MUL require different charts **if both are forced through
the same local per-harmonic operator class**.  This experiment asks whether a
richer operator grammar can keep both in one shared representation.

## Representation

Full additive characters over `F_31`, including the zero frequency.

## Motifs

\[
\chi_k(x+y)=\chi_k(x)\chi_k(y)
\]

so ADD is local complex multiplication.

\[
\chi_k(xy)=\chi_{ky}(x)
\]

so MUL is an operand-conditioned permutation of frequency coordinates.  The
operand identity is selected continuously by cosine soft addressing over the
same canonical codebook.

## Evaluation

Random mixed ADD/MUL instruction sequences through depth 256, continuous state
feedback, no chart switch and no hard decode between hops.
