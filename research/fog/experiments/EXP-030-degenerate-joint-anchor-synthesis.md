# EXP-030 — Synthesize a simple-spectrum anchor from degenerate operators

Status: **PASSED d=30**  
Date: 2026-08-14

Train only two dense scaling actions:

- `B: x -> 27x`, order 10;
- `C: x -> 25x`, order 3.

Neither action has simple spectrum in the 30D representation.  The compiler is
not told that these are powers of a primitive scaling generator.

It searches finite compositions `B^a C^b`, scores accepted candidates by
finite-order residual and number of distinct spectral roots, and uses the best
joint action as a new spectral anchor.  It then tries to express B/C as powers
of the synthesized operator and executes recurrent programs using only the
synthesized primitive.
