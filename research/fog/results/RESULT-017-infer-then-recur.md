# RESULT-017 — Operator induction composes with long latent execution

Seeds 60/61/62 each evaluate 1,000 non-ambiguous episodes per depth.

At depths 1, 2, 4, 8, 16, 32 and 64:

- operator routing: **100%**;
- final answer: **100%**;
- no hard intermediate state snap.

This establishes a controlled pipeline:

`demonstrations -> infer primitive -> recurrent generated-state execution`.

It does **not** yet establish heterogeneous programs whose primitive changes
mid-execution.

Evidence: `artifacts/research/exp_018/metrics.json`.
