# RESULT-010 — Multi-depth consistency removes observed recurrent drift

Across all three seeds:

- held-out binary addition: **100%**;
- depth 2: **100%**;
- depth 4: **100%**;
- depth 8: **100%**;
- depth 16: **100%**;
- depth 32: **100%**;
- depth 64: **100%**.

No arbitrary binary pair was supervised during training; only repeated successor
programs at terminal depths 1/2/3 were constrained.

A notable diagnostic: the simple `p`-root residual is not monotonically tied to
long-depth accuracy (one successful seed has a larger residual than weaker
EXP-010 seeds).  Stability belongs to the **whole learned transition/chart
system**, not one scalar generator metric.

Interpretation:

> multi-depth transition consistency is a practical gauge/stability training
> signal.  It can make a learned chart reusable far beyond the training depth
> without prescribing the final coordinates by hand.

Evidence: `artifacts/research/exp_011/metrics.json`.
