# RESULT-022 — Dense learned actions hide a sparse operator grammar

For EXP-022 boundary seeds 73/74/75:

## d=30

After diagonalizing learned `A`:

- `A` diagonal energy fraction: ~1;
- mean top-1 spectral energy of each `M` row: ~1;
- mean effective support of each `M` row: ~1;
- inferred eigenspace permutation obeys the semidirect relation on 100% modes;
- hard-pruned sparse actions retain **100% mixed-program accuracy at depth 64**.

The relative reconstruction error of `M` after one-entry-per-mode pruning is
approximately `3.9e-7`, `5.1e-4`, `7.0e-6` on seeds 73/74/75.

## d=29 control

- mean `M` top-1 spectral energy: ~0.824–0.858;
- effective support: ~1.44–1.51;
- sparse reconstructed depth-64 execution: ~2.3–3.7%.

Interpretation: the dense matrices in EXP-022 are largely a **gauge artifact**.
At the exact representation threshold, an automatically recovered spectral
gauge exposes diagonal/monomial operator motifs without receiving a Fourier
codebook.

Evidence: `artifacts/research/exp_023/metrics.json`.
