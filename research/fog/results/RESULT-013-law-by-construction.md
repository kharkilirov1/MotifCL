# RESULT-013 — Structural operator laws remove the recurrent failure

Structured normed arm, seeds 20/21/22:

- successor: **100% / 100% / 100%**;
- complete binary addition: **100% / 100% / 100%**;
- depth-64 recurrent accuracy: **100% / 100% / 100%**;
- target cosine is effectively 1.

Matched flexible-penalty arm:

- binary accuracy: 61.81%, 50.57%, **100%**;
- depth-64 accuracy: 3.52%, 2.34%, **0%**.

Seed 22 is the important counterexample: it is perfect on the complete one-step
binary table and has target cosine around 0.996, but repeated use destroys the
state.  Its closure angle is non-zero and its local perturbation gain has a p95
above one; the structured operator has near-zero closure defect and p95 gain
below one.

Conclusion:

> algebraic correctness at one step is weaker than a transition class whose
> geometry is closed and stable by construction.

Evidence: `artifacts/research/exp_014/metrics.json` and
`artifacts/research/stability_diagnostic/seed22.json`.
