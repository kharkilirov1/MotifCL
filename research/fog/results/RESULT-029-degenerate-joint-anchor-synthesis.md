# RESULT-029 — The compiler can synthesize a missing primitive operator

On d=30 seeds 100/101/102:

- learned B has discovered order 10 and 10 distinct roots;
- learned C has discovered order 3 and 3 distinct roots;
- neither alone fixes a simple spectral gauge;
- the compiler selects `T = B C` on all three runs;
- `T` has discovered order **30** and **30 distinct spectral roots**;
- the compiler then discovers

\[
B \approx T^{21},\qquad C \approx T^{10}.
\]

All compiled B/C programs remain **100% through depth 64**.

This is stronger than anchor selection: the useful primitive operator was not a
trained module.  It was created as a composition because that composition made
the joint representation easier to canonicalize.

The d=29 runs are not a clean negative control for this narrower scaling-only
grammar: two seeds fail the finite-order acceptance test, while one seed finds a
29-root order-30 action and executes the B/C grammar at 100%.  This is consistent
with the broader project lesson that the minimum useful width depends on the
actual operator family, not on the state count alone.

Evidence: `artifacts/research/exp_030/metrics.json`.
