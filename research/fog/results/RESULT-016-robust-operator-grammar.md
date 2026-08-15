# RESULT-016 — Robust executors make operator induction clean

Both ADD and MUL operator modules reach 100% complete group accuracy in all
three seeds.

For every semantically non-ambiguous episode:

- route accuracy: **100%**;
- answer accuracy: **100%**.

With one demonstration the overall metric is ~98–99% only because ~3.2–3.4% of
examples are mathematically compatible with both candidate operators.  Two
examples reduce ambiguity to ~0.1–0.2%; three effectively eliminate it.

Evidence: `artifacts/research/exp_017/metrics.json`.
