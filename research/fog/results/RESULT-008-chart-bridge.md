# RESULT-008 — Linear chart bridges memorize identities but do not extrapolate

Across seven identity splits:

| bridge | train identity acc | held-out identity acc | all-identity fit |
|---|---:|---:|---:|
| additive -> multiplicative | **100%** | **11.64% mean** | **100%** |
| multiplicative -> additive | **100%** | **12.04% mean** | **100%** |

Chance is `1/30 = 3.33%`.

The bridge is expressive enough to map the finite codebooks exactly when every
identity is supplied, and it perfectly interpolates the training subset.  But
the relationship between charts is not captured by a transferable linear law
from partial identity coverage.

Interpretation:

- multi-chart registers are promising;
- **chart switching is not free**;
- simply concatenating several views and inserting a dense linear bridge risks
  another finite identity lookup/memorization mechanism.

Current architectural options:

1. a shared canonical identity bus with operator-specific views;
2. explicit learned chart-transition operators with their own composition
   constraints;
3. redundant multi-view state updated jointly by each operator;
4. search for a less operator-specific universal chart with a measured
   complexity tradeoff.

Evidence: `artifacts/research/exp_009/metrics.json`.
