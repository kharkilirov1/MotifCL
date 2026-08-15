# RESULT-024 — The compiler recovers the hidden finite operator law

On all three d=30 seeds, from dense matrices alone the compiler discovers:

\[
A^{31}\approx I,
\qquad
M A M^{-1}\approx A^3.
\]

Order residuals range from `1.3e-6` to `1.1e-3`; conjugacy residuals from
`4.0e-7` to `4.5e-4`.  All three pass the fixed `0.01` threshold.

The resulting automatically compiled sparse grammar executes mixed programs at
**100% through depth 64** on 3/3 seeds.

The d=29 controls happen to have their smallest finite-order residual near 31,
but residuals are huge (`~0.74–0.99`) and conjugacy residuals are `~0.50–0.67`;
all are correctly rejected.

Evidence: `artifacts/research/exp_025/metrics.json`.
