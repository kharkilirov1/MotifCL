# Vulkan training: hardening and measured performance, 2026-10-18

## Scope and result

The starting checkout was `ad6a00f` on the RX 580 Windows host. This pass fixes
the audit findings and evaluates several remaining bottlenecks, without changing
model dimensions, FP32 storage, counter RNG seeds or optimizer hyperparameters.
Changes are local and uncommitted; no push, deployment or new build directory.

The enabled CE path is faster and scratch memory is bounded. Complete dense
training improved in the final ABBA control; a whole-MoE speedup is **not
established** by the final noisy ABBA. Slower or association-changing experimental
paths are not silently enabled to produce a larger headline number.

### Final complete-training measurements

Same B8/T256, dim288, six layers/heads, seed1337, FP32, untied head and
hyperparameters. Each run is 120 steps; the first 20 are excluded. Throughput is
the harmonic mean of equal-length steady windows, not the last logged step.
Profiling is disabled in these runs. Baseline and candidate are immutable copies;
the runner records executable/log SHA256, full environment and dataset metadata.

| Model / schedule | Baseline tok/s | Final-default tok/s | Observed result |
|---|---:|---:|---|
| MoE, ABBA x2 / eight runs | 7096.82 | 7082.04 | -0.21%; inconclusive under large run variation |
| Dense control, ABBA x1 / four runs | 15703.79 | 16163.32 | **+2.93%** in this bounded control |

- MoE baseline range **5991.88-8467.11**, candidate **5119.73-8759.41** tok/s;
  the two ABBA block deltas were **-12.84% / +18.30%**. Selecting only the
  favorable block would be misleading. This result is retained, not discarded.
- Dense baseline range **15553.34-15857.19**, candidate **16155.54-16171.10**
  tok/s. Both candidate runs exceeded both baseline runs, but four runs are not
  a confidence interval or a guarantee on other workloads.
- A read-only 2.56-second sample during the final MoE study observed Emulator
  +3.078 CPU seconds and two node processes +1.297/+0.609 CPU seconds.
  Background work was active; it was not stopped. This is not a GPU-utilization
  measurement and does not identify the sole cause of timing variation.

Final 120-step validation losses, in run order within each arm:

| Model | Baseline | Final default |
|---|---|---|
| MoE | 4.4995, 4.5936, 4.5661, 4.5544 | 4.5361, 4.5422, 4.5492, 4.5318 |
| Dense | 4.8584, 4.9510 | 4.7629, 4.7518 |

Every final training run was finite and its last logged training loss was below
its first. These small samples establish neither quality superiority nor long-run
quality parity. Earlier larger speedups with RMS splitting or subgroup CE stats
are not transferred to the conservative final-default binaries.

Separate 20-step profiles prove the intended default path engaged:
`ce_fwd_stats_f32` and `ce_bwd_cached_f32`, with the established
`rmsnorm_bwd_w_f32`. Word-load, fused SwiGLU, split-RMS and subgroup CE-stat
kernel names are absent. In the dense profile, CE backward was **5.295 ->
2.852 ms/step** and total kernel time **2550.88 -> 2506.51 ms/20 steps**.
MoE kernel totals were **3681.03 -> 3576.20 ms/20 steps**, but these separate
profile runs do not rescue an unsupported whole-MoE throughput claim.

Evidence: `abba-moe-conservative/report.json`,
`abba-dense-conservative/report.json`, `profile-moe-conservative/report.json`,
`profile-dense-conservative/report.json`, and `conservative-host-load.json`
under the evidence directory below. All runs exited 0; all raw-log hashes and
both final binary copies were rechecked after the runs.

## Correctness and memory repairs

| Finding | Executable evidence and result |
|---|---|
| NaN-blind floating comparators | CPU negative controls failed before the fix; NaN, Inf, size and tolerance controls now reject. GPU parity uses the hardened comparators. |
| Counter test compared the new update to itself | The reference now directly calls the old fused runtime helper. Two consecutive updates check exact packed state, scale and RMS, and require nonzero changes. Auto/forced-legacy and byte/word modes are independently profiled. |
| Causal attention tests did not enter the optimized kernels | Added B2/T67/H4/KV2/D8 and D7, plus B2/T256/H6/KV6/D48 CPU oracles with actual forward/backward kernel-name checks. Offset/general fallbacks remain covered. |
| Wave64-only reductions selected by arithmetic support alone | One production `supports_wave64_arithmetic()` predicate gates the counter update, CE, softmax, RMS and relevant GQA reductions. Six synthetic capability cases cover subgroup32/64/128 and missing arithmetic; actual non-wave64 hardware is not claimed tested. |
| Invalid counter optimizer parameters | Old and tiled helpers reject NaN/Inf, beta outside [0,1] and negative epsilon before dispatch; state bytes must remain unchanged. Constructor C is restricted to [1,11]. |
| Dtype check lost when removing the typed matmul wrapper | I32 buffers with valid F32 bit patterns reproduced the regression. Both activation/gradient I32 cases now reject before submission, including forced legacy mode; valid F32/C11 still updates. |
| One retained dense grad_w allocation per layer in a batch | A command-ordered weak workspace cache replaces per-layer Tensor scratch. Four 512x512 layers: owned scratch delta **4,198,400 -> 1,048,576 bytes**, with exact old-update parity across batch, capture and replay. |
| Slightly growing shapes still allocated every time | Logical workspace capacity now matches the power-of-two allocation bucket. 65 live requests: **67,633,152 -> 1,572,864 bytes**, **65 -> 2 allocations**. This is bounded geometric growth, not a claim that every shape shares one physical allocation. |
| Timing mixed window throughput with last-step phases | Both trainers now average GPU/phases over the same window and reset its accumulators. Source-derived C++ known-answer tests went from three failures to three passes. Missing GPU samples produce an unavailable value, not a partial mean. |
| Profile checks failed on timestamp-less devices | Only the profile subcheck is skipped without timestamps; numerical/state checks remain mandatory. |

Workspace ownership stays weak to avoid a buffer/runtime reference cycle.
Recorded dispatches retain their buffers until the fence or recording release;
normal compute barriers serialize overwrite/consume. Workspace is not saved
activation storage and must not be used concurrently as independent tensors.

## Improvements and conservative defaults

### Cached cross-entropy statistics

Training CE saves each row's maximum and inverse exponential sum on device.
Backward uses a coalesced flat kernel without repeating row reductions. NoGrad
inference and the public uncached backward retain the established path.
`MOTIFCL_VK_CE_CACHE=0` forces the old composition.

Tests include a double CPU oracle, invalid targets, non-unit upstream gradients,
odd/wide dimensions, the 65535-group boundary, deliberately perturbed saved
statistics, temporary lifetime, capture/replay and the real autograd node.
The existing invalid-target behavior is preserved, not silently redefined.

An important full-training check rejected the initial subgroup-stat default,
despite its passing per-op tolerances: the first eight-run MoE study produced
candidate validation losses **4.5518-4.6570**, versus baseline **4.5325-4.5450**.
Its aggregate +13% throughput is therefore not the final result published here;
that study also had substantial host-load drift.

A frozen follow-up prediction and controlled 120-step runs tested a CE
statistics-association alternative. Portable-tree cached statistics returned validation
**4.5346 / 4.5320**; forced legacy CE, retaining the other changes, returned
**4.5464 / 4.5361**. A separately compiled runtime-only probe confirmed exact
portable cached-vs-legacy gradients on five fixtures, including wide/invalid
targets. The permanent test now requires bit-exact portable gradients, including
525x32000 across the bounded-grid stride. Consequently portable statistics are
the default; `MOTIFCL_VK_CE_STATS_SUBGROUP=1` is explicit opt-in only.
This is evidence for the local default decision, not a long-run quality study.

The flat kernel uses local128. This stays within Vulkan Core's guaranteed
workgroup limits; invocations and the X dimension are distinct constraints.
See the primary [Khronos limits table](https://docs.vulkan.org/spec/latest/chapters/limits.html).
Embedded SPIR-V reflection verifies local128 and the matching host grid stride.

### Optional RMSNorm weight-gradient row split

For rows >=256 and cols <=1024, an optional dW path uses balanced row partitions, followed by
a small reduction, instead of a few underoccupied groups scanning every row.
Unset/`MOTIFCL_VK_RMS_DW_SPLIT=0` retains the old helper; `1` enables automatic splitting.
This FP32 reduction changes association and is tolerance-tested, not claimed
bit-identical. The independent counter update exactness requirement is unchanged.

The numerical test covers 120 shape/split combinations, an f64 oracle, partial
layout, legacy comparison, invalid arguments, capture and replay.

The path was not retained as a default merely because its microbenchmark won.
The initial portable-CE dense study had validation 4.7649/4.8030 on baseline and
4.7695/4.8959 with split RMS. A two-run split0 ablation returned 4.8029/4.7108.
That is too small and variable a sample to prove a systematic quality defect;
it also cannot justify silently changing the default training trajectory.
The conservative final default therefore keeps the old dW association. The
faster implementation remains available explicitly for separately validated runs.

### CPU and allocation work

- NoGrad binary ops no longer eagerly construct backward Nodes that are discarded.
- Native Vulkan basic/activation/norm paths bypass the allocating backend selector.
- Counter update scratch calls the raw native matmul helper, without constructing
  an extra Tensor/autograd result or unused scale/denominator scratch.
- Branching/shared upstream, views, repeated backward, NoGrad and capture/replay
  have a focused native Vulkan behavioral test.

The source-derived CPU probe observed selector allocations **2 -> 0/call** and
unused NoGrad node allocations **1 -> 0/call**. Those are targeted probe results,
not a claimed whole-trainer CPU speedup by themselves.

## Isolated GPU measurements

`bench_vulkan_training_hotpaths` is OpenCL-free. Each suite uses ABBA x2,
50 warmup iterations and 200 measured iterations/block, identical device buffers,
and strict actual kernel-name/call-count checks. GPU time is the sum of dispatch
timestamps, not host wall time. Median of four measured blocks per variant:

| Operation | Established path, ms | Candidate, ms | Decision |
|---|---:|---:|---|
| CE forward + backward, 2048x32000 | 7.442614 | **5.093135** | Enabled, -31.6% isolated GPU time |
| RMS dW including row-inverse pass, 2048x288 | 0.298186 | **0.067372** | Opt-in, -77.4% isolated GPU time; association/quality caveat above |
| Counter forward, 2048x288x384, byte vs word reads | 0.283061 | 0.305664 | Word loads opt-in only |
| Counter backward-input, same shape | 0.369378 | 0.463986 | Word loads opt-in only |
| Counter GEMM + SwiGLU, same saved preactivation/output | 0.287042 | 0.291300 | Fusion opt-in only; no convincing win |

Full per-block ranges and actual profiles are in `hotpaths-portable-final.csv`.
Some counter/RMS blocks vary materially with
the device state; no inference about hardware spills, occupancy or bandwidth
counters is made from these timings alone.

## Rejected or deliberately bounded work

- **Packed-word decode:** correct, including unaligned rows and the final group;
  339312 CPU decoding checks and four caught mutants, plus real GPU parity.
  It regressed on this device. `MOTIFCL_VK_COUNTER_WORD_LOAD=1` is experimental;
  unset/0 uses the existing byte path.
- **Counter GEMM + SwiGLU:** implemented and exact-tested through output, dX,
  two real updates and raw-input learning. The first 32x32-hidden tile used
  12544 shared bytes and 32 scalar accumulators/lane; a second 32x16-hidden tile
  uses 8448 shared bytes and 16 accumulators. Neither established a win against
  the actual byte-GEMM baseline, so default remains the old composition.
  `MOTIFCL_VK_COUNTER_SWIGLU=1` explicitly opts in; NoGrad/eval still fall back.
- **Naive grad_w K-split:** the CPU semantic probe reproduced reordered FP32
  sums and persistent RMS differences: splits2/4/8 changed 981/993/1000 of
  1024 dW values and 26/29/29 of 32 v values. Split1 was bit-identical; the
  known-order cancellation anchor also passed. This is not a GPU benchmark,
  and no K-split GPU implementation/performance is claimed. The exact oracle
  was not weakened to accept a different reduction contract.
- **Sparse expert execution:** not enabled because the current router surrogate
  backward consumes all seven expert candidates; skipping them changes the
  algorithm, not just dispatch overhead.
- **Global Tensor/Node freelist, in-place shared-gradient accumulation and
  add+RMSNorm fusion:** not implemented. Alias/ownership and backward contracts
  need separate witnesses; these are not described as completed improvements.

## Final verification

Every affected executable is relinked after each default change. Final commands
and normalized results:

```text
cmake --build build/release --parallel 12
CONSERVATIVE_BUILD_EXIT=0
ctest --test-dir build/release --output-on-failure --timeout 300 -j 1
30 passed, 29 skipped (OpenCL unavailable), 0 failed / 59 registered
100% tests passed, 0 tests failed out of 59
Total Test time (real) = 75.05 sec
CONSERVATIVE_CTEST_EXIT=0
[B stack] first_loss=7.60164 last_loss=0.0642764
```

The 29 skipped tests require unavailable OpenCL. They do not constitute runtime
coverage. New CE, RMS, counter-validation and fused-forward standalone targets
link only the Vulkan runtime; DLL import inspection found no OpenCL dependency.
The high-level autograd targets separately execute on Vulkan tensors.

Python checks: the timing-window test has three passing cases; the benchmark
runner has 18 passing cases, including malformed/incomplete logs, nonfinite
values, immutable outputs, executable drift, environment isolation, ABBA order,
the real `val.bin`/`validation.bin` dataset naming contract, and the original
dense trainer's concatenated GPU-timing/next-step line. Failed preflight/parser
runs remain recorded as failures and were not silently rewritten.

A fresh-context read-only reviewer found two additional portability issues;
both were repaired and re-read. No actionable P1/P2 findings remained in its
review scope. That review does not replace the runtime tests above.

The final Python rerun exited 0 for both files: **3 timing-window tests** and
**18 benchmark-runner tests**. `git diff --check` is the final whitespace gate;
no source/runtime/shader changes were made after the final rebuild and CTest.

## Changed-file map

- `src/runtime/vulkan_backend.cpp` and
  `include/motifcl/runtime/vulkan_backend.hpp`: validation, wave64 eligibility,
  serial workspace and CE/RMS/fused dispatch helpers.
- `src/nn/compact_counter.cpp`, its public header, `src/nn/moe7.cpp`,
  `src/ops/{basic_ops,activation,norm,loss}.cpp`: native integration and lazy CPU
  work without a global ownership/freelist refactor.
- Seven new `kernels/vulkan/*.comp` sources, regenerated
  `src/runtime/vulkan_spirv_{kernels,names}.inc`, and the generator's corrected
  Vulkan-target metadata in `tools/gen_vulkan_spirv.py`.
- `tools/train_{moe15m,ternary15m}.cpp`,
  `tools/benchmark_vulkan_training.py`: consistent timing windows and reproducible
  comparison tooling.
- `tests/CMakeLists.txt`, hardened existing comparator/GEMM/GQA/counter tests,
  six new Vulkan test sources, the hotpath benchmark, and two Python test files.
- `.hermes.md`, the implementation plan, the old report's audit correction and
  this report. `final-source-manifest.json` records exact changed-file hashes.

## Reproduction and evidence

All run artifacts are under `out/maintenance-20261018/` (ignored local output):

- `baseline-bin/`: unchanged original executables; `candidate-conservative-bin/`:
  final default executables. JSON benchmark reports preserve both SHA256 hashes.
- `conservative-build.log`, `conservative-ctest.log`, `conservative-ctest-details.log`:
  final build, pass/skip counts, numerical and memory witnesses.
- `conservative-timing-tests.log`, `conservative-runner-tests.log`,
  `final-source-manifest.json`, `final-artifact-audit.json`: Python results,
  changed-file inventory and independently rechecked run/default-path integrity.
- `counter-memory-red.txt`, `workspace-growth-red.txt`, `counter-dtype-red.txt`,
  counter parameter/comparator RED and GREEN outputs: observed failing controls.
- `hotpaths-portable-final.csv`, `final-shader-contract.txt`:
  actual isolated paths, timings and embedded shader metadata.
- `final-review.md`, `ksplit_rounding_*`, `cpu-hotpath-*`, `timing-*`:
  independent review, rejected arithmetic fork and targeted CPU witnesses.

```powershell
python tools/benchmark_vulkan_training.py BASELINE.exe CANDIDATE.exe `
  --data-dir data/tinystories --output-dir out/NEW_ABBA_DIR --steps 120 --pairs 2
# Profiling is a separate run and does not publish a throughput speedup:
python tools/benchmark_vulkan_training.py BASELINE.exe CANDIDATE.exe `
  --data-dir data/tinystories --output-dir out/NEW_PROFILE_DIR --steps 20 --profile
```

The output directory must be new. The runner pins configuration, clears ambient
training/Vulkan overrides, keeps full logs and rejects incomplete runs.
The dense trainer historically saves its final checkpoint even with SAVE_CKPT=0;
reported steady training windows exclude that final file I/O.

Not established: long-run pretraining quality, another GPU/driver, actual
subgroup32 or timestamp-less device execution, and OpenCL runtime compatibility.
The short convergence/numerical witnesses are not a claim of bit-identical full
training trajectories or a guarantee over a full training budget.
