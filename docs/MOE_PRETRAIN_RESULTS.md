# MN-MoE15M — measured runs (MoE from scratch, RX 580, Vulkan FP32)

Date 2026-10-16. Repo `motifcl_production`. Backend `Backend::create_vulkan()`, FP32.
Device (measured on this host): **Radeon RX 580 Series**, vendor `0x1002`, device `0x67df`
(Polaris 10 / Ellesmere), device apiVersion 1.3.260, AMD proprietary driver 26.5.2.

Design record: `docs/MOE_FROM_SCRATCH_DESIGN.md`.

## 0. Summary (15M study complete; 1B probe and 0.55B run in §14-16)

- **Built and verified.** `nn::MoE7FFN` — seven `CounterStateLinear` experts (hidden 192) plus an
  always-on shared expert (192), hard top-1 routing on the runtime's existing `fog_hard_route7`
  kernel, auxiliary balance gradient injected by hand, all on the Vulkan FP32 path with **no new
  SPIR-V**. Two ctest witnesses pass: forward equals a host reference at 1.17e-07 with correct
  gradient routing (`leaked=0`), and the injected balance gradient matches the analytic Switch
  gradient at 2.03e-07 including its linearity in `AUX_WEIGHT`.
- **Trained on TinyStories for 12 000 steps** (24.6M tokens, B8xT256): val loss 3.3416 @500,
  2.8870 @1000, 2.5984 @2000 (PPL 13.44, top-1 46.90 %, top-5 70.76 %), **2.0704 @12000** (PPL 7.93).
- **It beats the equal-active-width dense control**: dense 384 finishes at 2.0838 (PPL 8.04) with
  the MoE ahead from step 3500 onward, after starting 0.132 behind at step 500. The price is wall
  time — the MoE runs 2018 tok/s against 5285 tok/s, i.e. **2.6x slower per token** — and 3359 MB
  against 2332 MB. Routing is demonstrably not degenerate (each expert over-selects its preferred
  tokens by 1.8-5.3x their corpus frequency; mean top-20 Jaccard 0.183).
- **Cost/representation, measured:** counter state is 7.13 MB (MoE), 4.27 MB (dense 768), 2.85 MB
  (dense 384) — identical 3.7 KB per hidden unit in all three, so capacity is linear in memory and
  the saving is the 0.75 B/parameter encoding, not the mixture.
- **Two measured environment findings** that matter to anyone reproducing this: the FOG-era commit
  inverted the buffer memory placement, costing **2.0x on the MoE and 2.8x on the dense model** at
  bit-identical losses (`MOTIFCL_VK_PREFER_DEVICE_LOCAL=1` restores the pre-FOG order), and the
  Windows "GPU dedicated usage" counters do not see these allocations at all (166 MB reported
  against a 3359 MB runtime accounting peak).
- **1B does not fit on this card, and the reason is measured** (§14): the training path costs
  **~5 bytes per counter coefficient** (peak ≈ 1672 MB + 424 MB per block), i.e. a dense FP32 copy
  of each counter layer is materialised and retained on top of the 0.75 B/param packed state. That
  erases the memory advantage the method is supposed to have and puts a 1.06B-parameter model at
  7.3 GB, where it dies on a 2 MB allocation. Bringing the path to ~1.5 B/param would make 1B fit
  in ~2.2 GB — the highest-value follow-up this probe produced.
- **Not achieved:** sparse execution (v1 computes all seven experts per token, hence the 2.6x
  slowdown), and any 1B-scale quality result. 1B does not fit on this card (§14); the largest
  configuration that completes a run is **0.55B parameters at B=1xT=256, 500 steps, 116 tok/s,
  4648 MB peak, val 8.6453 -> 4.7538** (§16) — a feasibility result, not a usable model
  (0.0002 tokens/parameter).

## 1. What was built

| file | role |
|---|---|
| `include/motifcl/nn/moe7.hpp`, `src/nn/moe7.cpp` | `nn::MoE7FFN`: seven counter-synapse experts (hidden 192) + one always-on shared counter expert (192), hard top-1 router (`fog_hard_route7`), aux-loss-free balancing bias, Switch-style auxiliary balance gradient |
| `tools/train_moe15m.cpp` | pretrain runner (clone of `train_ternary15m.cpp`); knobs `HIDDEN_EXP`, `HIDDEN_SHARED`, `BALANCE_GAMMA`, `AUX_WEIGHT`, `BALANCE_CLAMP`, `EVAL_BATCH`, `SAVE_CKPT`, and `EVAL_ONLY=1` (checkpoint-only evaluation: loss, PPL, next-token top-1/top-5) |
| `src/runtime/vulkan_backend.cpp`, `include/motifcl/runtime/vulkan_backend.hpp` | opt-in `MOTIFCL_VK_PREFER_DEVICE_LOCAL` memory placement; device-allocation accounting (`buffer_bytes_live/peak/allocations`) |
| `tools/train_ternary15m.cpp` | `HIDDEN` knob (equal-active-width control for the MoE); log line prints `devmem live/peak MB` and `VAL ... ppl` |
| `tools/moe_log_summary.py` | parses run logs into step / train loss / val loss + PPL / tok-s / device peak / expert-load spread (`--json` supported) |
| `ports/rx580/moe/run-moe-comparison.ps1` | runs the comparison arms sequentially (MoE e192, dense 768, dense 384, MoE e384, aux ablations) with `-DryRun` support |
| `CMakeLists.txt`, `tools/CMakeLists.txt`, `tests/CMakeLists.txt` | registrations |

**No new SPIR-V was added.** `fog_hard_route7` + `..._logits_bwd` + `..._candidate_bwd`
already implement top-1 MoE semantics (per-row zeroing for losing candidates, straight-through
seven-way softmax for the logits). `glslc` is absent on this host, so no kernel could have
been regenerated anyway.

## 2. Finding: the FOG-era commit inverted the buffer memory policy — measured ~2x

`VulkanRuntime::create_buffer` preferred **device-local VRAM** before the FOG integration
(commit `34a8ade`, the one the recorded 4152 tok/s run used: `pick_type(kMemDeviceLocalBit,
kMemHostVisibleBit)`). HEAD (commit `9969f28` "feat(fog): integrate FOG v3 …", 2026-08-16)
prefers **host-visible** memory in both the pool reuse path and the fresh-allocation
candidate order.

A/B on this box, same binary, same data, 20 steps, only the env knob differs
(`MOTIFCL_VK_PREFER_DEVICE_LOCAL=1` restores the pre-FOG order):

| arm | model | policy | throughput | step loss |
|---|---|---|---:|---:|
| A | MoE 7x192+shared 192, B8xT256 | default (host-visible) | 1035-1039 tok/s | 6.5331 / 6.5306 / 6.3912 |
| B | MoE 7x192+shared 192, B8xT256 | device-local VRAM | **2010-2048 tok/s** | 6.5331 / 6.5306 / 6.3912 (identical) |
| A | dense (hidden 768), B16xT256 | default (host-visible) | 1767 tok/s | 6.2170 |
| B | dense (hidden 768), B16xT256 | device-local VRAM | **4920 tok/s** | 6.2170 (identical) |

Losses, router loads and logit statistics are **bit-identical** across the two arms, so this
is purely a placement effect: -2.0x on the MoE and -2.8x on the dense model.
This also closes the earlier unexplained discrepancy: `docs/TERNARY15M_PRETRAIN_RESULTS.md`
records 4152 tok/s for the dense B16xT256 run, which predates the FOG commit; with
device-local placement the same configuration measures 4920 tok/s here today (above the
recorded figure, on a different driver/background load), and 1767 tok/s with the current
default. **Report any MotifCL throughput number with its memory policy.**

## 3. Finding: device-allocation accounting, and the "3.9 GB" cross-check

`VulkanRuntime::buffer_bytes_peak()` accumulates the bucket-rounded device allocations the
runtime owns (in use + pooled) and is independent of the memory class. Measured:

| configuration | device peak |
|---|---:|
| dense (hidden 768), B16xT256 | **3895 MB** |
| MoE (7x192 + shared 192), B8xT256 | 3359 MB |

The dense figure matches the "peak VRAM was 3.9 GB" line recorded in
`docs/TERNARY15M_PRETRAIN_RESULTS.md`, which independently corroborates the instrument.
Note also that the Windows `GPU Adapter Memory / GPU Process Memory` "dedicated usage"
counters report only ~166 MB for these processes — they do not see these allocations, so
working-set or counter readings must not be presented as device memory.

## 4. Measured: the routing problem and its fix

First implementation balanced experts with the bias alone. At B8xT256, gamma 0.05, the router
**collapsed** within 40 steps: mean top1-top2 logit gap grew 0.39 -> 7.2 (bias clamped +-2),
two experts received **zero** tokens, load spread 0.63-0.67, `FINAL VAL 5.9358`.

Fix: the Switch auxiliary loss `L_aux = E * sum_e f_e * P_e` applied as an explicit router
gradient. `dL_aux/dlogits = (E/N) * p_j * (f_j - sum_e f_e p_e)` is computed on the host from
the same router read-back, uploaded, turned into a weight gradient with
`matmul_transpose_a(x, dl)` and accumulated into the router parameter — no new kernel.
After the fix: all seven experts alive, spread 0.078 @100 and 0.028 @300 steps.

## 5. Measured: 300-step run, B8xT256 (default policy)

```
DATA_DIR=data/tinystories OUT_DIR=out/moe_stab STEPS=300 BATCH=8 SEQ=256 \
EVAL_EVERY=100 EVAL_RECORDS=64 CKPT_EVERY=100 SAVE_CKPT=1 \
BALANCE_GAMMA=0.01 AUX_WEIGHT=0.01 ./build/release/tools/motifcl_train_moe15m
```

| metric | measured |
|---|---|
| throughput | 924-936 tok/s (default policy; ~2000 tok/s with device-local) |
| val loss (64 records) | 4.8114 @100, 3.9558 @200, **3.7365 @300** |
| expert load spread | 0.232 @50, 0.078 @100, 0.023 @200, 0.028 @300 |
| device peak / working set | 3359 MB / ~4.7 GB |
| checkpoint + resume | 430 files / 219 MB; resume from step 300 ran 301-305, `FINAL VAL 3.6911` |
| determinism | two independent runs: identical loss/load/bias/logit lines (only wall-clock differs) |

## 6. Measured: dense vs MoE at 4096 tokens/step (default policy, 30 steps)

| arm | active hidden | throughput | device peak | val |
|---|---|---:|---:|---:|
| dense (`motifcl_train_ternary15m`) | 768 | 1216-1263 tok/s | 3895 MB | 6.0442 @30 |
| MoE (7x192 + shared 192) | 384 (capacity 1536) | 753-774 tok/s | - | 6.1846 @20, then **device lost** |

The MoE arm died at ~step 21 with
`vkQueueSubmit failed for cached dispatch (VK_ERROR_DEVICE_LOST)`. At B8 the MoE is stable
for 300+ steps and, under the corrected policy, faster per token than B16 was under the old
one. B8xT256 is the working point; B16xT256 is out of budget for this model.

## 7. Correctness witness (closes the design's P0 gate)

`tests/test_moe7_reference.cpp`, registered in ctest (`ctest -R test_moe7_reference`,
`Passed`), checks three properties on the Vulkan backend with a 32-dim / 16-wide / 8-row
block:

| property | measured |
|---|---|
| routed forward vs host reference (decoded counter weights, SwiGLU `gate*sigmoid(gate)*up`, host argmax, plus shared expert) | relative error **1.17e-07** |
| module token counts (read back from its own router logits) vs host argmax distribution | `mismatch = 0` |
| gradient after `sum_all(y).backward()` | router grad non-zero; **3** experts (exactly the 3 distinct winners of 8 rows) show non-zero row-RMS `v`; **0** experts with zero tokens received gradient |

The third row is the one that matters for the model: it is direct evidence that the
`fog_hard_route7` candidate backward really restricts each expert's update to its own
routed tokens, which is what makes the sparse-expert semantics real despite the dense
compute.

`tests/test_moe7_balance.cpp` (`ctest -R test_moe7_balance`, `Passed`) independently checks the
mechanism that keeps the experts alive. It recomputes the Switch auxiliary gradient on the host
from the public router weight/bias — `dL_aux/dlogit = (E/N)*p_i*(f_i - sum_j f_j p_j)` — builds
`x^T * dl` itself, and compares against the gradient the module accumulated:

| property | measured |
|---|---|
| injected router gradient vs the independently computed analytic gradient | relative error **2.03e-07** |
| linearity: `AUX_WEIGHT` doubled | relative error **2.03e-07** vs 2x the reference |
| routing spread at that step | 0.188 0.125 0.062 0.250 0.000 0.188 0.188 (includes a starved expert) |

## 7b. Eval must not be heavier than training (measured on the 8 GB card)

The long run's device high-water mark jumped from 3359 MB to **6780 MB** the moment the first
eval at step 500 ran, because the eval used 32-record batches (8192 tokens) while training uses
B8 (2048 tokens): the vocabulary-logit chain is linear in tokens, so eval allocated 4x the
buffers and the pool retained part of them. Point-in-time sampling confirmed the process held
4638 MB with the adapter at 4931 MB, i.e. the 6.78 GB was a transient high-water mark.

Fix applied to both trainers: `EVAL_BATCH` (default `min(32, BATCH)`), so eval can never exceed
the training working set. The running long job predates the fix and keeps its 6.8 GB peak; the
queued dense baseline and all later arms use the new default.

## 8. Boundaries

1. **Compute is dense, routing is sparse.** All seven candidates are computed for every token;
   `fog_hard_route7` selects rows afterwards. No sparse-execution speedup in this version.
2. **Expert count is fixed at seven** by the existing kernel.
3. **Device loss at B16** for the MoE is a hard configuration boundary, single observation.
4. **`MOTIFCL_VK_PREFER_DEVICE_LOCAL` is opt-in and off by default**, so it does not change any
   existing behaviour; it changes where tensor buffers live (VRAM + staging instead of GTT).
   Two arms were measured on one model each; other models/sizes were not re-tested.
5. Loss/val numbers at the few-hundred-step scale are single-seed and small-token-count; no
   quality claim is made from them.
6. The `devmem` number counts the runtime's bucket-rounded buffer allocations; it excludes
   driver/desktop overhead and any non-runtime allocation.
7. **The validation slice is easier than the training distribution** (measured: 0.175 nats lower
   unigram entropy at equal sample size, 8.9 sigma). Absolute val loss/PPL in this record is
   therefore optimistic and not comparable to published TinyStories numbers; the arm comparisons
   are unaffected because every arm evaluates the same fixed 96 records.
8. **Counter state is not "all training state"** (the neighbouring `memory-native` record makes
   the same distinction): the measured 7.13 / 4.27 / 2.85 MB are the packed counter state files
   only. FP32 scales, RMS statistics, embeddings, the untied head and their Adam moments live in
   the remaining ~211 MB of the checkpoint and are counted separately.

## 9. Long run in flight (`out/moe_long`)

The run started with `STEPS=20000` (41M tokens); at the step-2000 checkpoint the shared horizon was
shortened to `STEPS=12000` (24.6M tokens) — see the horizon note below. Eval every 500
steps, checkpoint every 2000, `AUX_WEIGHT=0.01`, `BALANCE_GAMMA=0.01`, `BALANCE_CLAMP=2`.
Resumable (`RESUME=1` + `meta.txt`).

| step | train loss | val loss | val ppl | expert-load spread | tok/s | device peak |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 5.9706 | - | - | 0.158 | 2051 | 3359 MB |
| 200 | 4.2685 | - | - | 0.014 | 2059 | 3359 MB |
| 300 | 3.7443 | - | - | 0.037 | 2062 | 3359 MB |
| 400 | 3.6841 | - | - | 0.033 | 2052 | 3359 MB |
| 500 | 3.6250 | **3.3416** | **28.264** | 0.024 | 2066 | 3359 MB |
| 1000 | 3.2132 | **2.8870** | **17.94** | 0.021 | 1891 | 6780 MB (eval) |
| 1500 | - | **2.7191** | **15.17** | 0.027 | - | - |
| 2000 | - | **2.5984** | **13.442** | 0.024 | - | - |

The step-2000 checkpoint was evaluated a second time with the standalone evaluator
(`EVAL_ONLY=1`, same 96 records) and reproduced the training-loop number exactly while adding
ranking metrics:

```
EVAL_ONLY steps=2000 records=96 tokens=24576 loss=2.5984 ppl=13.442 top1=0.4690 top5=0.7076
```

Two independent code paths (inline eval inside the training loop and the checkpoint-only
evaluator) agree to four decimals, which is a direct check of the evaluation protocol.

**Horizon change, stated openly.** The run started with `STEPS=20000`. At the step-2000
checkpoint the shared horizon for both arms was shortened to `STEPS=12000` (24.6M tokens) to fit
the session budget; the MoE resumed from the checkpoint with the phase log in
`out/moe_long/log_resume.txt` (the original log is preserved). The switch changes the learning
rate at step 2000 by ~3.4% (0.985 -> 0.951 of peak) and both arms then share one schedule, so
the comparison stays internally consistent; the first 2000 steps of the MoE ran on the longer
horizon and that is the one caveat on the earliest points.

Val protocol (identical for every arm): the **first 96 validation records in fixed order**
(24 576 tokens, no sampling), counters switched to inference mode so no state updates run,
loss averaged over the chunks. It is therefore directly comparable across arms and resumable
runs. The 6 780 MB device peak is the eval high-water mark before the `EVAL_BATCH` fix (§7b).

**That fixed prefix is measurably easier than the training distribution**, so the absolute
numbers below are optimistic and must not be compared with published TinyStories perplexities.
Measured at equal sample size (96 records = 24 576 target tokens, 30 independent draws from
`train.bin`):

| sample | unigram entropy | nats |
|---|---:|---:|
| the fixed validation prefix used by every eval | 5.7240 | - |
| 30 random 96-record draws from the train file | 5.8992 +- 0.0197 | spread [5.8584, 5.9370] |

The gap is **0.175 nats, 8.9 standard deviations** of the draw-to-draw spread, so it is a property
of the slice and not an estimation artefact. It also explains part of the systematic
train-loss-above-val-loss pattern in the logs. Arm-to-arm comparisons are unaffected (every arm
evaluates the identical slice), but any claim of the form "this model reaches PPL X on
TinyStories" would need a random or full validation split instead; `EVAL_RECORDS` (and a random
offset) is the knob for that.

Queued behind it: `out/dense_cmp` — dense hidden 768 at the **same** `STEPS=12000` horizon,
same B8xT256, same fixed 96-record val protocol, same learning-rate schedule. Its first matched
point (step 2000) is directly comparable with the MoE number above; deeper points arrive as it
runs, and the MoE catches up afterwards.
Remaining arms are prepared in `ports/rx580/moe/run-moe-comparison.ps1` and are run with
`-Arms <subset>`:

| arm | what it isolates |
|---|---|
| `dense_h384` | dense at the MoE's **active** width (384) — the fair control against MoE e192 |
| `moe_e384` | capacity scaling at active width 576 |
| `moe_aux3e3` | weaker auxiliary balance gradient (sharper routing) |
| `moe_aux0` | bias-only balancing — the configuration that collapsed within 40 steps at clamp 2 |

Open tuning item carried from the 300-step runs: with `BALANCE_CLAMP=2` the selection bias
saturates (observed `bias0[-2.000..2.000]`) while load stays balanced, so the long-series arms
use `BALANCE_CLAMP=4`.

## 10. Matched comparison (equal tokens, equal schedule horizon)

All arms: B8xT256 (2048 tokens/step), `STEPS=12000`, the same warmup/cosine horizon for every
arm after the step-2000 checkpoint, the same fixed 96-record val protocol,
`MOTIFCL_VK_PREFER_DEVICE_LOCAL=1`. "active" is the hidden width actually evaluated per token;
"capacity" sums the expert/shared hidden widths that exist.

| step | MoE 7x192+shared (active 384, capacity 1536) | dense 768 (active 768) | dense 384 (active 384) |
|---:|---:|---:|---:|
| 500 | 3.3416 | 3.2228 | **3.2100** |
| 1000 | 2.8870 | **2.8082** | 2.8269 |
| 1500 | 2.7191 | **2.6490** | 2.6814 |
| 2000 | 2.5984 | **2.5456** | 2.5795 |
| 2500 | resumes after the dense arm | - | **2.5139** |
| 3000 | resumes after the dense arm | - | 2.4453 |
| 3500 | resumes after the dense arm | - | 2.4250 |
| 4000 | resumes after the dense arm | - | 2.3929 |
| 4500 | resumes after the dense arm | - | 2.3799 |
| 5000 | resumes after the dense arm | - | 2.3285 |
| 5500 | resumes after the dense arm | - | 2.3176 |
| 6000 | resumes after the dense arm | - | 2.2852 |
| 6500 | resumes after the dense arm | - | 2.2472 |
| 7000 | resumes after the dense arm | - | **2.2354** |
| 7500 | resumes after the dense arm | - | 2.2135 |
| 8000 | resumes after the dense arm | - | 2.1867 |
| 8500 | resumes after the dense arm | - | 2.1642 |
| 9000 | resumes after the dense arm | - | 2.1511 |
| 9500 | resumes after the dense arm | - | 2.1406 |
| 10000 | resumes after the dense arm | - | 2.1230 |
| 10500 | resumes after the dense arm | - | 2.1111 |
| 11000 | resumes after the dense arm | - | 2.0970 |
| 11500 | resumes after the dense arm | - | **2.0924** |
| 12000 | in flight (phase 2) | - | **2.0838** (final, 12 000/12 000 steps done) |

**Final matched result (both arms at 12 000 steps, identical schedule from step 2000 on):**

| arm | val loss @12000 | val PPL | tok/s | device peak (own phase) | counter state |
|---|---:|---:|---:|---:|---:|
| **MoE 7x192 + shared** | **2.0704** | **7.928** | 2018 | 3359 MB | 7.13 MB |
| dense 384 (equal active width) | 2.0838 | 8.035 | 5285 | 2332 MB | 2.85 MB |

**The MoE overtakes the equal-active-width dense control and stays ahead.** The column-by-column
trace shows where it happens: the MoE is behind by 0.132 at step 500, converges to 0.019 by step
2000, crosses at **step 3500** (2.4232 vs 2.4250), widens to 0.021 by step 4500 and finishes
0.0134 ahead at 12 000. The reading that fits all of it: extra capacity behind the same active
path costs nothing in loss per token early and pays off later, while the routing overhead is paid
in *time* — the MoE is **2.6x slower per token** (2018 vs 5285 tok/s) and costs 1.4x the device
peak. So at this scale the mixture is a quality-per-token win and a wall-clock loss; that trade is
what a sparse dispatch implementation would have to fix.

Scope limits of the claim, stated: one corpus (TinyStories), one seed per arm, 24.6M tokens, a
single dense control at equal active width, and dense-768 only measured to step 2000 (2.5456, i.e.
behind dense-384's own 12000-step value). Nothing here isolates routing from capacity beyond what
the equal-width control already fixes, and no result is claimed at 1B scale — see the 1B section.

`dense_h384` finished all 12 000 steps; the MoE phase 2 then resumed from its step-2000
checkpoint. The resume was verified rather than assumed:

- the log prints `resuming from step 2000` and the header shows `steps=12000 ... start=2000`;
- device memory returns to the same steady state (3327/3359 MB), expert load is balanced
  (spread 0.018) and the balancing bias comes back at its clamped value — counters, router, bias
  and Adam moments all restored from `meta.txt`;
- the learning rate at step 2100 is **0.00378**, identical to the dense arm's learning rate at the
  same step under the shared 12 000-step cosine. From step 2000 on, both arms follow exactly the
  same schedule; only the earliest points (500-2000) carry the documented ~3.4 % learning-rate
  offset from the MoE's original 20 000-step horizon.

The dense-768 column stops at 2000 because that run was stopped at its step-2000 checkpoint to
free the GPU for the equal-active-width control; its curve is therefore **partial** (500-2000
only) and is reported as such. The decisive pair — MoE and `dense_h384` — is matched to the full
12 000-step budget, while the wide-dense reference stays an early-point comparison.

Three readings from these four points:

1. **At equal *active* width the MoE is behind, but converging.** Against `dense 384` the MoE
   deficit is 0.132 (step 500) -> 0.060 (1000) -> 0.038 (1500) -> 0.019 (2000). The extra
   capacity (1536 vs 384 behind the same token) has not paid for the routing overhead yet, but
   the trend has not flattened; whether it crosses is exactly what the longer matched run tests.
2. **Capacity alone does not help much at this scale either.** `dense 384` and `dense 768` are
   within 0.034 of each other at every point (2.5795 vs 2.5456 at step 2000) while the wider
   model costs ~0.4 s vs ~0.39 s per step and 2479 MB vs 2332 MB. Doubling dense capacity buys
   little at 4M tokens, so "the MoE loses because it is narrow" and "the MoE loses because
   routing is hard" cannot be separated by the dense-768 column alone.
3. **The MoE is the slowest per token of the three** (2051-2066 vs 5111-5337 tok/s) because all
   seven candidates are computed for every token; that is the dense-compute caveat, not a
   routing-quality result.

Cost at the same step size:

| arm | tok/s | step time | device peak | counter state (measured) |
|---|---:|---:|---:|---|
| MoE e192 | 2051-2066 | ~1.0 s | 3359 MB | 120 layers, **7.13 MB** (attention 1.43, expert bank 5.70) |
| dense 768 | 5111-5170 | ~0.40 s | 2479 MB | 36 layers, **4.27 MB** (attention 1.43, FFN 2.85) |
| dense 384 | 5280 | ~0.39 s | 2068 MB | 36 layers, **2.85 MB** (attention 1.43, FFN 1.42) |

The counter-state column is a measured cross-check of the encoding: every arm stores exactly
`out*in*0.75` bytes per counter linear (four 6-bit codes per three bytes) plus `out` floats of
scale and RMS. That is **3.7 KB per hidden unit in all three arms** — MoE, dense 384 and dense 768
alike. Expert capacity is therefore *not* cheaper in memory than dense capacity; what is cheaper
is the representation itself (0.75 B/parameter against FP32 weights plus Adam moments). What the
MoE would buy, once sparsity is exploited in execution, is that only a fraction of that capacity
is *computed* per token — a compute argument, not a memory argument. The earlier draft of this
section called expert capacity "essentially free in memory"; the measurement contradicts that and
the claim is withdrawn.

Honest reading of the first three columns: **at equal tokens the dense-768 model is both more
accurate and 2.5x faster than the MoE**, and the MoE also peaks higher in memory. Two effects are
confounded there — the MoE's *active* width is half of the dense-768 control, and 4M tokens is
far too small a budget for sparse routing to pay off. The `dense 384` column is the control that
removes the width confound: same active width, same active compute, difference only in the
capacity behind each token (1536 vs 384) and in the routing. That comparison is the one that
decides whether this design has a point at this scale; it is running now.

What the MoE measurement *does* establish independently of quality is the encoding arithmetic: the
seven-expert bank plus the shared expert costs **5.70 MB** of 6-bit counter state across all six
blocks (and 1.43 MB of it is the attention counters the two dense arms share), against a 218.6 MB
checkpoint dominated by the FP32 embedding, untied head and their Adam moments. Per hidden unit
that is identical to the dense arms (3.7 KB), so the saving is the counter representation, not the
mixture.

## 12. Design claims vs measurements

Every claim the design record made before any code was written, and what the measurements say.

| design claim | measured outcome | verdict |
|---|---|---|
| "no new SPIR-V needed — `fog_hard_route7` already has top-1 MoE semantics" | implementation is entirely existing kernels; `test_moe7_reference` matches a host reference at 1.17e-07 with `leaked=0` | **confirmed** |
| "≈1.16x the dense model's compute, so ≈3400-3900 tok/s" | **2051-2066 tok/s**, i.e. ~1.8x below the estimate and 2.5x below the dense-768 control at the same batch | **falsified** — the estimate counted FLOPs only. Each block now issues 16 counter-layer dispatches instead of 2, and the expert matrices (288x384) sit far below the 437-889 GFLOP/s the runtime reaches at large shapes, so dispatch and small-shape efficiency dominate |
| "capacity 2x behind the same token, 1/7 activation sparsity" | capacity is real (1536 vs 768) and costs only 5.70 MB of counter state, but at 4M tokens the val loss is *worse* than the narrower dense-768 control (2.5984 vs 2.5456 at step 2000) | **not yet supported** at this token budget; the equal-active-width control decides it |
| "aux-loss-free bias is enough to balance" | **collapsed**: two experts received zero tokens, spread 0.63-0.67, gap grew to 7.2 against a +-2 clamp | **falsified**; the Switch auxiliary gradient (host-computed, injected, witness rel_err 2.03e-07) was required and holds spread at 0.02-0.04 |
| "B16xT256 (4096 tokens/step) is the working point" | `VK_ERROR_DEVICE_LOST` at ~step 21; B8 runs 2000+ steps cleanly and is *faster per token* under the corrected memory policy | **falsified**; B8 is the recipe |
| "peak memory 4.5-6 GB" | training peak **3359 MB**; eval with a 32-record batch pushed it to 6780 MB until `EVAL_BATCH` fixed it | partially wrong in the opposite direction, then fixed |

The one claim that mattered most — that the runtime already contained everything needed to build
a trainable top-1 MoE — held up. The performance model did not: this MoE is currently a
*parameter-count* argument (1536 hidden of capacity behind a 384-wide active path, costing 7.13 MB
of counter state against 4.27 MB for dense 768), not a speed-per-token one, and it is not a
memory-per-capacity one either — measured, counter state scales linearly with capacity in every
arm. The quality question is still open at 4M tokens.

## 13. Routing specialisation (measured, not assumed)

A loss number cannot tell a working router from a degenerate one (uniform noise also balances
perfectly under the auxiliary loss). Two measurements on the step-2000 checkpoint over the same
24 576 validation tokens answer it: `EVAL_ONLY=1 ROUTE_HIST=1` reports, for the last block, each
expert's most frequent target tokens and the pairwise overlap of their top-20 sets; the corpus
frequency of those tokens was counted directly from `validation.bin`.

| expert | tokens routed | top token | share inside the expert | corpus share | lift |
|---:|---:|---:|---:|---:|---:|
| 0 | 3859 | 29892 | 0.1249 | 0.0389 | **3.21x** |
| 1 | 3887 | 13 | 0.1330 | 0.0251 | **5.30x** |
| 2 | 3380 | 29889 | 0.1127 | 0.0614 | 1.84x |
| 3 | 2562 | 29915 | 0.0433 | 0.0109 | **3.97x** |
| 4 | 3965 | 2354 | 0.0366 | 0.0101 | **3.62x** |
| 5 | 3350 | 263 | 0.1143 | 0.0264 | **4.33x** |
| 6 | 3573 | 322 | 0.1075 | 0.0300 | **3.58x** |

Mean pairwise Jaccard of the experts' top-20 target sets: **0.183** (0 would be disjoint, 1
identical). Every expert over-selects at least one token by 1.8-5.3x its corpus frequency, and
their preferred sets are largely disjoint.

Reading: the router is **not** degenerate — it demonstrably conditions on the token. The honest
caveat is that "expert prefers token type X" is the shallowest kind of specialisation, and with
only seven experts and 8M tokens it is the kind one should expect; nothing here shows deeper
structural expertise. It does mean the MoE's deficit against the dense controls at this budget is
a training-budget effect rather than a broken-router effect.

Per-layer imbalance is also visible here: at layer 5 the load ranges 2562-3965 tokens (+-22 %
around the mean), while the training log's `spread` (0.02-0.04) averages that over all six blocks
and hides it. The aggregated metric understates per-block imbalance.

## 14. Scale-out probe: where a 1B memory-native model hits the wall

The task was to train a **1B-parameter** memory-native MoE on this card. The trainer was
parameterised (`DIM`, `LAYERS`, `HEADS`, `REC_LEN`) and the shape was probed with measured
memory, not estimated. Target configuration: `DIM=1280`, `LAYERS=12`, `HEADS=10`,
`HIDDEN_EXP=HIDDEN_SHARED=2560`, tied head → **1.022B counter coefficients + 41M FP embedding =
1.06B parameters total**, packed counter state 767 MB (0.75 B/param, the encoding this whole
method is built on).

**Result: it does not fit.** Measured at B=1, T=256:

| layers | counter coefficients | device peak | tok/s | outcome |
|---:|---:|---:|---:|---|
| 2 | 170M | 2524 MB | 244 | ok |
| 4 | 341M | 3376 MB | 134 | ok |
| 6 | 511M | 4220 MB | 91 | ok |
| 8 | 682M | 4983 → 5531 MB | 69-83 | ok (2 steps) |
| 12 | **1.02B** | 6508 → 6781 MB | 46 | survives step 1, then dies |

The 12-layer failure, from the runtime's own accounting (message improved in this session to
carry the request and the committed state — a bare "allocation failed" was indistinguishable
from a driver error):

```
vkAllocateMemory failed for all compatible memory types
  [requested=2 MB bucket=2 MB candidates=4 live=7335 MB peak=7335 MB allocations=2805]
```

The measured scaling law between layer count and peak is **linear: peak ≈ 1672 MB + 424 MB per
block** (+852 MB for layers 2→4, +844 MB for 4→6). Per block that shape holds 85.2M counter
coefficients (attention 6.55M + experts/shared 78.6M), so the training path costs
**≈ 5.0 bytes per counter coefficient** — against the 0.75 B/param the packed state itself needs.
Two consequences:

1. **First reading, wrong and now retracted.** The 5 B/coefficient slope was initially reported as
   "a dense FP32 copy of every counter layer is materialised and retained". That is **refuted by
   direct measurement** in `tests/test_counter_memory_vulkan.cpp` (added in this session, ctest
   green): one counter layer at in=out=2560 costs **8.0 MB** on device where the packed state is
   4.7 MB and a dense FP32 copy would be 25.0 MB, and a six-layer stack costs 8.0 MB per layer
   across a full forward+backward. The Vulkan update is fused (`apply_update_backward` allocates
   only `[out]` scratch) and `backward_input_u8` needs no decoded weight, so the repository's
   "0.75 byte/weight" claim **holds on the Vulkan path too** — that path simply had no gate
   before (the repo's own `test_counter_memory_truth` is OpenCL-only), which is why the missing
   gate is worth keeping.
2. **The measured explanation of the peak** is three amplifiers that the 0.75 B/param claim never
   covered, each measured separately:
   - **Buffer-bucket rounding.** A 4.7 MB counter state occupies an 8 MB bucket: **+70 % on the
     state itself**, and the effect is worse for the many small activation tensors.
   - **MoE candidate activations.** The module keeps all seven experts' intermediates for
     backward: in the 1B-class block shape (dim 1280, experts 2560) the block costs
     **278.9 MB**, of which the packed expert state is only **56.2 MB** — i.e. **80 % of the
     block is activation footprint**, and it is 7x a dense FFN's by construction (v1 computes all
     candidates).
   - **Per-layer saved inputs and grad_x** required by the fused in-backward counter update, plus
     the FP tail (embedding/head/Adam ~0.5 GB) and pool retention.
   The 0.55B run measured the outcome: packed state on disk is **365.6 MB, exactly the theoretical
   0.75 B/param**, while the device peak is **4648 MB — 12.7x the state**.

Secondary measured facts:

- **Shorter sequences do not help much.** `SEQ=128` (with `REC_LEN=257` to keep the packed
  records) moved the 12-layer peak only from 6508 to 6380 MB, consistent with a footprint set by
  the per-layer state plus per-layer activations at small batch, not by the attention length.
- **B=2 does not fit at 8 layers**: failure at live=5527 MB on an 8 MB request, i.e. the driver's
  practical ceiling here is ~5.5-7.3 GB depending on the allocation/fragmentation pattern (8 GB
  card minus desktop/driver reserve).
- The largest configuration that trains on this box is therefore **0.55B parameters (6 layers,
  same width), B=1 x T=256 at ~116 tok/s and ~4.6 GB peak** — the run reported in §16. (0.72B at
  8 layers passed a 2-step probe but failed at startup on a real run: it sits exactly on the
  ceiling.)

## 14b. Branches on the remote, and what they contain

`git fetch --all` on `origin` = `https://github.com/kharkilirov1/MotifCL`. Four remote branches
exist besides `main`:

| branch | content | relevance here |
|---|---|---|
| `origin/main` | FOG v3, bare-metal Vulkan reasoning engine, RX 580 port | the line this work is built on |
| `origin/fog-qkv-split` | FOG QKV split (merged into `readme-facade`, behind main by 4) | no memory work |
| `origin/claude/rdr-reference-impl-yr9p0n` | 2 384 lines under `examples/rdr/`: a span-anchor subquadratic attention "RDR" variant, dataset/trajectory builders, Kaggle runner, `RESULTS.md`, an honest "mechanism works, overfits at toy scale" result | a different research direction (attention, not counters) |
| `origin/claude/code-review-f4eke2` | 18 559 insertions / 182 files, including the whole `memory-native-training/` project embedded as a subdirectory: `apply_to_motifcl.md`, `tests/test_counter_memory_truth.cpp`, `tools/memory_budget_calculator.py`, bf16 + fused stacked-expert updates, a preprint draft | **directly relevant**: it states the expected memory contract this work has to meet |

Two things from those branches changed how I read this task:

1. **The expected contract is documented, and it is stronger than what I measured.**
   `memory-native-training/apply_to_motifcl.md` states: "Memory-native backward: no dense
   `[out,in]` grad_w is allocated and no decoded weight is carried across forward→backward.
   `test_counter_memory_truth` enforces this", and the test header calls the claim
   **"0.75 byte/weight training peak"** — note *training peak*, not just persistent state. My
   measurement says the persistent side matches exactly (0.75 B/param) but the *training peak* is
   12.7x that, because of activations and bucket rounding. So the honest position is: the packing
   claim holds, the peak claim does not on this path, and the amplifiers are now measured.
2. **The gate exists only for OpenCL.** `test_counter_memory_truth.cpp` opens
   `Backend::create_opencl()` and counts dense `[out,in]` allocations through
   `CounterStateLinear::reset_memory_counters()`. The Vulkan path — the one this whole
   RX 580 study runs on — had no equivalent, which is exactly why an unexplained slope could go
   unnoticed; `test_counter_memory_vulkan` closes that gap using the runtime's own allocation
   accounting (no instrumentation hooks needed).

## 16. The 0.55B memory-native MoE run (largest that completes)

The task asked for a 1B-parameter memory-native model on this card. 1B does not fit (§14), 0.72B
sits exactly on the ceiling and fails at startup, so the largest configuration that completes a
run is 0.55B. Configuration and measured outcome:

| item | value |
|---|---|
| shape | `DIM=1280`, `LAYERS=6`, `HEADS=10`, 7 experts x 2560 + shared 2560, tied head |
| parameters | **0.55B** = 511M counter coefficients + 41M FP embedding |
| batch | `BATCH=1` x `SEQ=256` = 256 tokens/step |
| steps / tokens | 500 / **128k tokens** |
| throughput | 116 tok/s mean (118 last), ~2.2 s/step |
| device peak | **4648 MB** (live 4548) |
| packed counter state | **365.6 MB on disk = 0.75 B/param exactly** (measured, matches theory) |
| checkpoint | 428 files, 837.8 MB (state + FP embedding + Adam moments) |
| expert load spread | 0.035-0.063 throughout, no collapse |

Validation loss, same fixed 96-record protocol in spirit (16 records per eval here for speed):

| step | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val loss | 8.6453 | 6.8073 | 5.8094 | 5.6916 | 5.3593 | 5.1285 | 4.9761 | 4.9012 | 4.8309 | **4.7538** |
| PPL | 5688 | 905 | 333 | 296 | 213 | 168 | 145 | 134 | 125 | **116** |

**What this run does and does not establish.** It establishes that a 0.55B-parameter
memory-native MoE builds, trains, checkpoints and evaluates entirely on an 8 GB RX 580 through
the Vulkan FP32 path, at a measured 4.6 GB peak and 116 tok/s, with the packed state matching its
theoretical size exactly and the router staying balanced at a batch of one. It does **not**
establish a useful model: 128k tokens is **0.0002 tokens per parameter** (Chinchilla's rule of
thumb is 20), so PPL 116 at step 500 describes a barely-started model, and the 15M study in §10 —
which trained for 192x more tokens — remains the only place in this record where quality is
comparable to anything.

Memory attribution for the 4648 MB peak, all measured: 365.6 MB packed state (7.9 %), the rest
being buffer-bucket rounding, the MoE's retained candidate activations (80 % of a block's cost at
this shape, §14), per-layer saved inputs/gradients for the fused counter update, the FP tail and
pool retention.

## 15. Reproduce

Every arm in this record is one env-var-driven command; `STEPS` defines the shared schedule
horizon, so arms are only comparable when it matches.

```
cmake --build build/release --target motifcl_train_moe15m motifcl_train_ternary15m -j 8
ctest --test-dir build/release -R test_moe7        # the two correctness witnesses

# 1) MoE phase 1: 2000 steps on the original horizon
DATA_DIR=data/tinystories OUT_DIR=out/moe_long STEPS=20000 BATCH=8 SEQ=256 \
EVAL_EVERY=500 EVAL_RECORDS=96 CKPT_EVERY=2000 SAVE_CKPT=1 RESUME=1 \
BALANCE_GAMMA=0.01 AUX_WEIGHT=0.01 BALANCE_CLAMP=2 MOTIFCL_VK_PREFER_DEVICE_LOCAL=1 \
  ./build/release/tools/motifcl_train_moe15m

# 2) MoE phase 2: resume from the step-2000 checkpoint to the 12000 horizon shared
#    with every dense arm (new log file, the phase-1 log is preserved)
DATA_DIR=data/tinystories OUT_DIR=out/moe_long STEPS=12000 BATCH=8 SEQ=256 \
EVAL_EVERY=500 EVAL_RECORDS=96 CKPT_EVERY=2000 SAVE_CKPT=1 RESUME=1 \
BALANCE_GAMMA=0.01 AUX_WEIGHT=0.01 BALANCE_CLAMP=2 MOTIFCL_VK_PREFER_DEVICE_LOCAL=1 \
  ./build/release/tools/motifcl_train_moe15m   # > out/moe_long/log_resume.txt

# 3) dense controls at the same horizon, batch and val protocol
DATA_DIR=data/tinystories OUT_DIR=out/dense_cmp  STEPS=12000 HIDDEN=768 BATCH=8 SEQ=256 \
EVAL_EVERY=500 EVAL_RECORDS=96 EVAL_BATCH=8 CKPT_EVERY=2000 MOTIFCL_VK_PREFER_DEVICE_LOCAL=1 \
  ./build/release/tools/motifcl_train_ternary15m
DATA_DIR=data/tinystories OUT_DIR=out/dense_h384 STEPS=12000 HIDDEN=384 BATCH=8 SEQ=256 \
EVAL_EVERY=500 EVAL_RECORDS=96 EVAL_BATCH=8 CKPT_EVERY=2000 MOTIFCL_VK_PREFER_DEVICE_LOCAL=1 \
  ./build/release/tools/motifcl_train_ternary15m

# 4) checkpoint-only evaluation with ranking metrics, and the routing analysis
DATA_DIR=data/tinystories OUT_DIR=out/moe_long EVAL_ONLY=1 EVAL_RECORDS=96 EVAL_BATCH=8 \
HIDDEN_EXP=192 HIDDEN_SHARED=192 MOTIFCL_VK_PREFER_DEVICE_LOCAL=1 \
  ./build/release/tools/motifcl_train_moe15m
DATA_DIR=data/tinystories OUT_DIR=out/moe_long EVAL_ONLY=1 ROUTE_HIST=1 EVAL_RECORDS=96 \
EVAL_BATCH=8 HIDDEN_EXP=192 HIDDEN_SHARED=192 MOTIFCL_VK_PREFER_DEVICE_LOCAL=1 \
  ./build/release/tools/motifcl_train_moe15m

# 5) collect every arm's curve from its raw logs (no hand-copied numbers)
python tools/moe_log_summary.py --compare out/moe_long/log.txt out/moe_long/log_resume.txt \
  out/dense_cmp/log.txt out/dense_h384/log.txt
```
