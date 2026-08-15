# RESULT-014 — Demonstration consistency can select a latent operator

Across seeds 30/31/32, routing is **100% on semantically non-ambiguous episodes**.
With one demonstration, about 3% of sampled episodes are intrinsically
ambiguous; with two or three demonstrations this nearly disappears.

One seed's free-codebook MUL executor learned only ~60.8% of its complete group
law.  The router still selected that executor correctly, separating two failure
classes:

- operator induction/routing;
- operator execution/geometry.

This motivated EXP-016/017 rather than hiding executor instability inside the
router metric.

Evidence: `artifacts/research/exp_015/metrics.json`.
