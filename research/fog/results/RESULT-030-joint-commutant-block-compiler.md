# RESULT-030 — repeated irreducible structure is recoverable without a simple spectrum

EXP-031 passed all decisive arms.

Across 9 main runs:

- multiplicity 2 -> commutant dimension **4**;
- multiplicity 3 -> commutant dimension **9**;
- multiplicity 4 -> commutant dimension **16**;
- the largest number of distinct eigenvalues found in the searched operator
  words was only **2**, while total widths were 4, 6 and 8;
- aligned repeated-block relative errors were at numerical precision
  (`~1e-15`);
- compiled recurrent execution at depth 256 was **100% on 9/9 runs**.

The broken-sharing controls all produced commutant dimension **5** rather than
9 and were rejected on 3/3 seeds.

## Interpretation

Simple-spectrum canonicalization is not necessary.  Repeated joint irreducible
structure leaves a different invariant signature: the **commutant algebra**.
A generic symmetric element of that algebra can separate multiplicity copies,
after which intertwiners expose one shared operator grammar.

The result is exact/controlled and does not yet establish robustness to learned
approximation; EXP-032 addresses that separately.
