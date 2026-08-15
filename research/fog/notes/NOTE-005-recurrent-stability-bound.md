# NOTE-005 — Closure defect and recurrent stability

A one-step latent operator can be semantically correct yet dynamically unsafe.
A useful abstraction is

\[
e_{t+1} \le \lambda e_t + \varepsilon,
\]

where:

- `epsilon` is the one-step **closure defect**: distance between the learned
  transition from a canonical state and the correct reusable state/manifold;
- `lambda` is a local **error gain**: how strongly the next transition amplifies
  a perturbation already present in its input state.

If a uniform bound with `lambda < 1` holds, recurrent error is bounded by a
geometric series.  If `lambda ~= 1`, systematic closure bias can accumulate
roughly linearly.  If `lambda > 1`, some perturbation directions may grow.

This is not claimed as a global theorem for arbitrary neural networks; the repo
contains an empirical local diagnostic.

## EXP-014 seed-22 case study

Flexible penalty operator:

- one-step complete binary accuracy: 100%;
- target cosine: about 0.996;
- closure angle mean: 0.0684 rad;
- local gain mean: 1.0002;
- local gain p95: 1.1391;
- depth 8: 100%;
- depth 16: 75.3%;
- depth 32: 0.024%;
- depth 64: 0%.

Structured normed operator:

- closure angle approximately numerical zero;
- local gain mean: 0.733;
- p95: 0.961;
- depth 64: 100%.

This complements NOTE-001: stable FOG recurrence can arise either because the
transition is nearly exactly closed (`epsilon ~ 0`) or because its dynamics
actively contract error (`lambda < 1`).  The most robust systems can do both.

Evidence: `artifacts/research/stability_diagnostic/seed22.json`.
