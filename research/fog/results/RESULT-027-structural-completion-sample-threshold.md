# RESULT-027 — Motif structure resolves the final orientation bit

Across seeds 90/91/92:

## k=28

Base mixed accuracy is ~2.3–4.7%.  Spectral support/closure projection remains
~0.8–6.3%.  The continuous O(2) complement ambiguity is not recoverable.

## k=29

Seeds 90/91 already choose the correct completion and remain 100%.
Seed 92 chooses the opposite orientation for M:

- `det(M) ~= +1`;
- base mixed accuracy: **1.56%**;
- spectral support projection: **100%**;
- closure projection: **100%**.

## k=30

All three runs are 100% without repair.

This is a controlled demonstration that structural operator constraints can
reduce the empirical transition-data requirement from d to d-1 when the only
remaining ambiguity is discrete and identifiable from the motif.

Evidence: `artifacts/research/exp_028/metrics.json`.
