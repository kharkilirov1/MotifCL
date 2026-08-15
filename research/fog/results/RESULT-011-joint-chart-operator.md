# RESULT-011 — Algebraic laws can recover the operation, but not reliably yet

Formal three-seed results:

## Successor-only

All seeds reach 100% on the observed `+1` generator, but arbitrary binary
addition remains poor:

- seed 0: 33.30%;
- seed 1: 6.56%;
- seed 2: 11.34%.

Depth-16 recurrence is 4.69%, 3.13%, 2.34% respectively (chance is 3.23%).

This is a decisive specialization result: a shared local bilinear operator can
perfectly learn one generator without becoming the group operation.

## + algebraic self-consistency

Without adding binary target labels:

- seed 0: **100% binary**, 60.94% depth-16;
- seed 1: 38.61% binary, 3.13% depth-16;
- seed 2: **100% binary**, **97.66% depth-16**.

For the successful binary seeds, measured commutativity/associativity errors are
near numerical zero.  The bad seed also reaches very low algebraic consistency
error yet implements the wrong semantic law, proving that generic group-like
self-consistency plus a generator constraint still admits undesirable local
solutions under the current parameterization/optimization.

Exploratory follow-ups:

- simply training longer did not rescue the bad seed;
- ramping algebraic losses after a successor warm-up did not rescue it;
- adding unlabeled closure/isometry penalties from the start made all tested
  seeds worse.

Interpretation:

> algebraic priors are powerful enough to change the solution class, but the
> current joint chart/operator landscape is not identifiable/optimizable enough
> for reliable use.

This is the current closest controlled analogue of the production problem:
learn both the representation and the reusable transition law rather than
receiving either from us.

Evidence: `artifacts/research/exp_012/metrics.json` and per-seed artifacts.
