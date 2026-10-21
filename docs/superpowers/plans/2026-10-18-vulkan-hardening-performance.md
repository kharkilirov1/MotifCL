# Vulkan Training Hardening and Performance Implementation Plan

> **For agentic workers:** Use bounded independent workers for disjoint files, with one runtime integrator and a fresh final reviewer. Do not commit or push.

**Goal:** Repair the audit findings and improve complete RX 580 training steps across the remaining bottlenecks, without changing the model or training algorithm.

**Architecture:** Keep C++17, FP32, existing Vulkan buffers and cached dispatches. Preserve portable fallbacks. Add independent numerical oracles and executable path/memory checks before enabling optimizations.

**Tech Stack:** C++17, GLSL/SPIR-V, Vulkan 1.1, GCC/Strawberry, Ninja, Python.

**Spec:** The user's 2026-10-18 request and the preceding evidence-backed audit. Scope is correctness, batched scratch memory, cross-entropy/elementwise fusion, CPU overhead, and remaining measured GEMM/decode opportunities.

## Global Constraints

- Keep FP32, model dimensions, data, optimizer parameters, counter stochastic-rounding semantics and training budget unchanged.
- No OpenCL fallback in Vulkan paths; standalone runtime witnesses remain OpenCL-free.
- No destructive Git, commit, push, system configuration changes, or new build directory.
- Work in the clean existing checkout because both existing build trees bind to this source root. Use the current `build/release` tree; the old `build/rx580-release` cache is stale.
- Only the integrator edits `src/runtime/vulkan_backend.cpp`, its public header, CMake, and generated SPIR-V headers. Serialize builds and all GPU runs.
- Generated headers are regenerated from shader sources, never hand-edited.
- Preserve a baseline trainer binary and run identical training commands before/after. Report warmup, repeated runs, skipped tests and actual selected kernels.

## Review Focus

- Non-finite comparator values and optimizer parameters must fail, not silently pass or corrupt state.
- Subgroup32/non-arithmetic devices must not enter wave64-only counter updates.
- Scratch reused within a batch must not alias saved activations or survive incorrectly through capture/replay.
- New CE/fused backward paths must handle tail dimensions, invalid targets and non-unit upstream gradients.
- Autograd changes must preserve branching accumulation, views, nested/reversible backward and backend lifetime checks.

## Measurement acceptance (before final runs)

- Final numerical/state/capture tests must pass after relinking every affected executable; skipped tests are reported separately.
- Compare immutable baseline/current executables with ABBA x2, 120 steps/run, first 20 excluded, identical pinned configuration and harmonic steady throughput. Record both ABBA ratios and the baseline A/A range; do not extrapolate a single last-step timing.
- Profile separately. A winning isolated kernel is not by itself a whole-trainer speedup; reject a slower candidate default.
- Short-run quality acceptance is finite, decreasing training loss, fresh convergence-gate success, independent numerical oracles, and exact counter-transition controls. Record every final validation loss and explicitly avoid claiming bit-identical complete trajectories or long-run training quality.
- Packed-word decode regressed on RX580. Both SwiGLU tile sizes failed to establish a win against byte GEMM. Both paths remain explicit opt-in only; exact parity was not weakened.
- Portable cached CE is the default. Subgroup CE statistics and RMS dW splitting are explicit opt-in only after short-run trajectory/association checks; prior faster-default measurements do not describe the final binaries.

## Tasks

### Task 1: Freeze baseline and repair verification

**Files:** `tests/vulkan_p0_test_utils.hpp`, `tests/test_vulkan_gemm_parity.cpp`, `tests/test_vulkan_gqa_training.cpp`, `tests/test_vulkan_counter_fused.cpp`, `.hermes.md`.

- [x] Preserve baseline trainer/benchmark executables and run repeated 120-step MoE measurements at B8/T256, seed1337; profile separately.
- [x] Add NaN/Inf negative controls, real causal fast-path cases with kernel-profile engagement, and old-vs-new independent counter update comparisons.
- [x] Observe the NaN negative control fail before fixing comparators; use seeded path disabling/breaks where a new parity test covers already-correct code.
- [x] Replace stale build guidance and require relinking tests after runtime changes; distinguish pass/skip/fail.

### Task 2: Counter validation and bounded scratch

**Files:** `src/nn/compact_counter.cpp`, `src/runtime/vulkan_backend.cpp`, `include/motifcl/runtime/vulkan_backend.hpp`, counter/memory tests.

- [x] RED: invalid optimizer arguments, subgroup eligibility and multi-layer batch memory gate.
- [x] Guard the tiled wave64 path and retain the established portable fallback; mirror finite/range validation.
- [x] Reuse a serial command-ordered grad_w scratch across layers without a shared_ptr ownership cycle; avoid unnecessary Tensor/autograd objects.
- [x] GREEN: independent old update parity, two consecutive batch steps, capture/lifetime tests and bounded peak memory.

### Task 3: Cross-entropy fusion

**Files:** `src/ops/loss.cpp`, new `kernels/vulkan/ce_*` sources, runtime helpers, new CE regression test.

- [x] Add CPU-reference forward/backward cases for odd/wide shapes, invalid targets and non-unit upstream gradients, plus a path-engagement check.
- [x] Implement a device-resident forward-statistics/backward path that avoids recomputing row reductions and unnecessary passes; preserve a forced legacy A/B mode.
- [x] Regenerate shaders, verify numerical tolerances and measure CE and full training; keep portable cached CE as the default, with bit-exact portable cached-vs-legacy gradient controls. Whole-MoE throughput remains inconclusive under the final host variation.

### Task 4: CPU and elementwise/graph overhead

**Files:** tensor/autograd hot-path files, appropriate fused op/shader files and trainers, focused regression tests.

- [x] Measure targeted selector/NoGrad allocations with source-derived probes and inspect trainer phase costs; remove only demonstrably unused work, without claiming an independently isolated whole-trainer CPU gain.
- [x] Add branch/view/repeated-backward/capture and fusion parity witnesses; rerun the existing reversible/autograd tests. General nested-engine/ownership refactors were not attempted.
- [x] Evaluate container reuse, gradient accumulation/fusion and remaining elementwise dispatches. Reject unsafe global freelists and shared-gradient mutation; document add+RMSNorm fusion as deferred, not implemented.
- [x] Benchmark GPU variants with explicit switches and immutable baseline; compare CPU probes before/after and combined full training. Reject slower defaults rather than crediting only an isolated winning case.

### Task 5: Remaining measured GPU opportunities

**Files:** counter/GEMM shaders and helpers, shape benchmark and focused parity tests.

- [x] Evaluate grad_w K-splitting and packed-state/SwiGLU epilogue opportunities against the actual path replaced. K-split was rejected by an anchored CPU FP32 semantic probe before a GPU implementation; no GPU K-split timing is claimed.
- [x] Require tail/multiple-workgroup parity and short training-quality checks; do not weaken an exact oracle merely to accept a faster implementation. Optional RMS split has f64/tolerance tests, not a bit-identical trajectory claim.
- [x] Record rejected alternatives and any scope that needs a separate architecture change.

### Task 6: Full verification and report

**Files:** `reports/vulkan-hardening-performance-2026-10-18.md`, reusable benchmark tooling, repository guidance.

- [x] Rebuild all test targets after the final runtime/shader changes; complete CTest: 30 passed, 29 OpenCL skips, 0 failed / 59 registered, 75.05 seconds. OpenCL-free standalone and high-level Vulkan autograd targets executed.
- [x] Run baseline/current training comparisons (MoE ABBA x2, dense ABBA x1, 120 steps, first 20 excluded), separate 20-step profiles, and fresh memory/convergence checks. Dense +2.93% observed; final MoE -0.21% with opposing block deltas is inconclusive, not a claimed speedup.
- [x] Fresh-context read-only source review, with addenda after default changes; repair portability findings, retain numerical checks without timestamps and bound cached CE local size to128.
- [x] Report changed files, measured deltas/noise, rejected changes and unverified device boundaries. Changes remain uncommitted; no push.

## Bounded outcome

The report is `reports/vulkan-hardening-performance-2026-10-18.md`.
Conservative defaults enable cached portable CE and safe CPU/workspace changes.
Packed-word decode, fused SwiGLU, subgroup CE statistics and split RMS remain
explicit experiments, not hidden default speed claims. General Tensor/Node
freelists, add+RMSNorm graph fusion and GPU K-split are not implemented. Long-run
quality, other devices/drivers and unavailable OpenCL remain unverified.
