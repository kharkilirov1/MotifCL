# RESULT-009 — A full addition chart emerges from 31 successor constraints

Three closed-cycle seeds, trained only on the successor cycle, reached:

- 100% successor accuracy;
- **100% on every held-out binary pair `(a,b)` with `b!=1`**;
- 100% recurrence through depth 16.

This means the learned identity phases organize themselves into a cyclic group
representation sufficient for arbitrary addition, despite arbitrary binary
pairs never receiving target supervision.

The closure edge is essential.  The open-chain control gets 30/31 successor
edges correct (96.77%) but averages only about **50.1%** held-out binary
addition and collapses under recurrence.

## Important stability failure

One-step / binary exactness was **not** sufficient for deep feedback.
For the three closed-cycle seeds, depth-32 final accuracy was:

- seed 0: 98.63%;
- seed 1: 100%;
- seed 2: **2.64%**.

All three had 100% held-out binary top-1 accuracy.  Tiny geometric inconsistency
that is invisible to one-step decoding can therefore accumulate catastrophically
under recurrence.

An exploratory direct `p`-root regularizer did not fix this and often worsened
optimization.  This motivated EXP-011, which constrains transition composition
at multiple depths instead of a single scalar invariant.

Interpretation:

> operator-compatible geometry can emerge from a sparse transition law, but
> one-step generalization does not certify recurrent numerical/geometric
> stability.

Evidence: `artifacts/research/exp_010/metrics.json` plus exploratory
`root_seed*.json` diagnostics.
