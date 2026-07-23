# memory-native PTQ solver port plan (conversion on MotifCL / RX 580)

Goal: run the full conversion (donor -> group-128 ternary + salient counter state)
inside MotifCL, so the RX 580 box does solve + inference + counter finetune without
PyTorch. The reference implementation and all measured gates live in
`memory-native-training` (donor/ptq.py, donor/asym.py); measured deploy config:
`salient_scope=layer, calibration=asym (residual form), strength=0.15, passes=2,
salient 2%` — aggregate warm metric 3.792 -> 2.899 on Qwen2.5-1.5B, and 2.678 after
6000 KD steps (see that repo's results/).

## Already done (this tree)

- `include/motifcl/nn/group_counter_import.hpp` + `src/nn/group_counter_import.cpp`:
  MNCC0001 container loader + host decode of the deploy format
  (W[o, perm[j]] = scale[o, j/group] * t[o, j]; salient overrides on top).
  Witness: `tests/test_group_counter_import.cpp` — BIT-EXACT vs the PyTorch
  `visible_weight()` reference (max|diff| = 0).
- `include/motifcl/solver/ternary_solver.hpp` + `src/solver/ternary_solver.cpp`:
  exact per-row optimal ternary (`optimal_ternary`, parity 3e-8 / 0 code mismatches)
  and the hdiag group-scale refit (parity 6e-8).
  Witness: `tests/test_ternary_solver.cpp` on vectors exported from the reference.

## Port ladder (each rung has a PyTorch-exported parity vector before code)

1. **H collection (device)**: per-target `H += X^T X` over calibration batches.
   MotifCL has no forward hooks — add a `SolverProbe` wrapper module (wraps a
   Linear/QuantizedLinear, accumulates X^T X into an F32 [in, in] device tensor via
   the existing matmul_transpose ops, then forwards). Memory: largest 1.5B layer H is
   8960^2 f32 = 321 MB — fits 8 GB alongside an F16 model (3.1 GB).
2. **GPTQ group sweep (host first)**: Cholesky of damped H (percdamp=0.01 mean-diag),
   `cholesky_inverse`, act-order permutation by diag(H) descending, column sweep with
   error feedback and nearest-ternary against per-(row,group) scales, in-sweep refit.
   Reference: `gptq_group_ternary` (ptq.py) — port to plain C++ (Eigen-free: write the
   small Cholesky/solve by hand, H fits in RAM; fp64 accumulators like the tests).
3. **Exact sym align re-solve**: per-row SPD system A s = b over a row's groups
   (A[k,l] = <T_k H, T_l>) — small (n_groups^2) Cholesky per row.
4. **itf grid + salient split (layer scope)**: straight ports; salient = global top-K
   of |w|*sqrt(diag H).
5. **asym two-tower (residual form!)**: sequential chunks; H_q = X_q^T X_q,
   G = X_q^T X_fp, target w~ = w + (H_q + lambda I)^{-1} (G - H_q) w.
   CRITICAL: use the residual form — the naive (H_q+lambda)^{-1} G w Tikhonov-to-zero
   variant measurably lost 4.3x network-KL in the reference repo (pinned by its
   tests). On 8 GB run the fp tower as Q8 (existing quantized path) or stream
   per-chunk from disk.
6. **Deploy default** once rungs 1-5 pass their parity/warm gates:
   layer scope + asym s=0.15, 2 passes, salient 2%.

## Notes

- The imported format trains today ONLY through the fp tail; native training of the
  group format needs a group-scale variant of the CounterStateLinear update kernel
  (per-(row,group) RMS second moment instead of per-row) — separate design.
- Every rung must land with a `tests/data/*` vector exported by the reference repo
  and a parity test in `tests/` (pattern: test_ternary_solver.cpp).
