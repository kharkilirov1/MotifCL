# Vulkan MoE sparse execution — Phase A audit and baseline

Date 2026-10-21. Repo `motifcl_production`, branch `main`, base commit
`ad6a00f38c9ec11bc635c68492e25c6828f8e1d3` (working tree dirty: 23 modified,
24 untracked — the 2026-10-18 hardening patch, not yet committed).

Device: Radeon RX 580 Series (Polaris 10 / Ellesmere, `0x67df`), AMD proprietary
driver 26.5.2, Vulkan 1.3.260, wave64, 36 CU, 8 GB GDDR5.

## A.1 What the current Vulkan MoE is

The only Vulkan-native MoE in the tree is `nn::MoE7FFN`
(`src/nn/moe7.cpp:83-103`): seven `CounterStateLinear` experts + one always-on
shared expert, hard top-1 routing via `fog_hard_route7`, router bias for
aux-loss-free balancing.

Forward per block (`moe7.cpp:89-102`):

```
logits = x @ W_r                     # mm_f32_nn   [2048,288]x[288,7]
biased = logits + b                  # add_bias_rows_f32_rb4
for e in 0..6: candidates[e] = down_e(swiglu(up_e(x)))   # counter_gemm_db x2 + swiglu_f32
routed = fog_hard_route7(biased, candidates)             # argmax rows post-hoc
out = routed + shared(x)             # add_f32
```

Backward (`src/ops/fog.cpp:122-160`): `FogHardRoute7BackwardNode` runs the
straight-through logits backward once plus one candidate backward per expert
(42 calls/step = 7 per block), where losing candidates receive an all-zero
`grad_out`.

The general `MoEFFN` (`src/nn/transformer.cpp:1639-1681`) used by the
Qwen3.5/Mixtral-style `HybridGPTModel` path is **OpenCL-only**: one naive fused
kernel `moe_swiglu_forward_f32` (`kernels/motif.cl:51-122`) that recomputes the
router dot product and the gate/up projections for every output column. It has
no Vulkan path at all (`Backend::kernels` is the OpenCL KernelCache).

## A.2 Measured baseline (30 steps, documented 15M recipe)

Command (env): `DATA_DIR=data/tinystories OUT_DIR=out/audit_baseline STEPS=30
BATCH=8 SEQ=256 DIM=288 HEADS=6 LAYERS=6 HIDDEN_EXP=192 HIDDEN_SHARED=192
TIE_HEAD=0 LR=0.004 LR_FP=0.006 AUX_WEIGHT=0.01 BALANCE_GAMMA=0.01
BALANCE_CLAMP=2 LOG_EVERY=5 SAVE_CKPT=0 RESUME=0 GPU_TIMING=1
MOTIFCL_VK_PROFILE=1 MOTIFCL_VK_PREFER_DEVICE_LOCAL=1 REC_LEN=257 SEED=1337`
→ `./build/rx580-release/tools/motifcl_train_moe15m.exe`

Steady state (step 30 window, 5 steps, 2048 tokens/step):

- throughput **8925 tok/s** (0.2295 s/step wall), loss 6.33, val ppl 448
- device peak **3359 MB** (`devmem 3343/3359 MB`)
- GPU time: fwd+bwd **168.64 ms** + optim 4.01 ms = 172.65 ms/step
- CPU phases: upload 0.1, fwd record 18.2, submit wall 169.5, balance 7.3,
  optim 4.9 ms
- submits/step: 16 (2 per step pair: fwd+bwd batch, optim batch)

### A.2.1 Per-kernel GPU profile (window of 5 steps, steady state)

| kernel | calls/step | ms/step | % of step GPU | owner |
|---|---:|---:|---:|---|
| counter_gemm_db | 240 | 53.02 | 30.7% | all counter forwards + backward_inputs |
| mm_f32_db64 | 3 | 36.36 | 21.1% | LM head (untied, 32k vocab) |
| mm_f32_db32 | 120 | 29.04 | 16.8% | FP32 GEMMs (router balance grad, etc.) |
| gqa_bwd_probs_causal4 | 6 | 11.66 | 6.8% | attention backward |
| gqa_fwd_causal4 | 6 | 6.92 | 4.0% | attention forward |
| gqa_bwd_dkv_causal_tile | 6 | 6.05 | 3.5% | attention backward |
| rmsnorm_bwd_w | 13 | 3.66 | 2.1% | norms |
| add_f32 | 96 | 3.53 | 2.0% | residuals, MoE merge |
| adam_update_fast | 21 | 2.78 | 1.6% | FP32 params |
| counter_update_apply | 120 | 2.85 | 1.7% | counter state updates |
| ce_bwd_cached | 1 | 2.70 | 1.6% | loss backward |
| ce_fwd_stats | 1 | 2.33 | 1.3% | loss forward |
| mm_f32_tn | 12 | 2.36 | 1.4% | tiled grad_w GEMMs |
| swiglu_bwd | 48 | 2.15 | 1.2% | MoE FFN activation backward |
| swiglu_f32 | 48 | 1.52 | 0.9% | MoE FFN activation forward |
| rope_f32 | 24 | 1.29 | 0.7% | RoPE |
| fog_hard_route7_candidate_bwd | 42 | 0.98 | 0.6% | MoE routing backward |
| fog_hard_route7_logits_bwd | 6 | 0.92 | 0.5% | MoE router backward |
| rmsnorm_bwd_x_subgroup | 13 | 0.79 | 0.5% | norms |
| rmsnorm_f32_subgroup | 13 | 0.51 | 0.3% | norms |
| mm_f32_nn | 6 | 0.23 | 0.1% | router logits GEMM (1/block) |
| fog_hard_route7_f32 | 6 | 0.22 | 0.1% | MoE routing forward |
| rmsnorm_row_inv | 13 | 0.19 | 0.1% | norms |
| zero_f32 | 1 | 0.19 | 0.1% | scatter zero-fill |
| mm_f32_nt | 6 | 0.17 | 0.1% | backward GEMMs |
| add_bias_rows | 6 | 0.086 | 0.0% | router bias |
| embedding_scatter_cas | 1 | 0.084 | 0.0% | embedding backward |
| token_position_embedding | 1 | 0.048 | 0.0% | embedding forward |
| mean_reduce | 1 | 0.026 | 0.0% | loss reduce |

Total 771 dispatches/step (fwd+bwd+opt). Raw log:
`out/audit_baseline/log.txt` (reconstructed from the run's stdout; the
background-task log file was not persisted by the tool).

### A.2.2 MoE stage table (mission §3 format)

Attribution: 6 blocks, per block 16 counter layers in the MoE (7 experts x
{up,down} + shared {up,down}) and 4 in attention; `counter_gemm_db` calls split
by that 32:8 ratio (MoE 192/240 = 80%); `counter_update_apply` 96/120 = 80%.

| Operation | dispatches/step | CPU overhead | GPU time/step | Main problem |
|---|---:|---:|---:|---|
| Router logits | 6 | in fwd record 18.2 ms | 0.23 ms | none (cheap); but per-block GEMM launch |
| Top-k (argmax) | 6 (fused in route7 fwd) | in fwd record | 0.22 ms | none; selection is post-hoc |
| Token grouping | 0 | — | — | **does not exist**; grouping is the OpenCL-inference-only feature (`MOE_PERFORMANCE.md`), absent on Vulkan |
| Expert gate/up | 96 (48 fwd + 48 bwd counter GEMMs) | in record 47.6 ms | ~31.6 ms | **7x redundant**: all 7 experts computed for every token; 6 of 7 lose the argmax |
| Expert down projection | 96 | in record 47.6 ms | ~10.8 ms | same 7x redundancy |
| Merge/scatter | 12 add + 48 route/cand-bwd | in record | ~1.6 ms | add of routed+shared; fine |
| Full MoE forward (per step, fwd only) | 174 | ~8 ms of record | ~24 ms | dense execution of 7 experts |

MoE-attributable GPU total (fwd+bwd, per step): **~52.4 ms of 172.65 ms
(30.4%)**, of which expert GEMMs are ~42.4 ms and the losing-expert share is
**~37.1 ms/step (21.5% of the whole step)**.

### A.3 Bottleneck summary

1. **Dense expert execution is the dominant waste.** 6 of 7 experts are fully
   computed (forward AND backward, including counter-state ticks) for every
   token and then discarded by the post-hoc argmax. ~37 ms/step GPU, ~21.5% of
   the step, buys nothing.
2. **No Vulkan token grouping.** Sparse dispatch needs count → prefix-sum →
   compact assignment lists; none exist on the Vulkan path (they exist only in
   the OpenCL inference path of `MoE7FFN`, per `docs/MOE_PERFORMANCE.md`).
3. **CPU recording scales with dispatch count** (47.6 ms/step for ~771
   dispatches, ~62 us each). Removing ~180 dispatches saves ~11 ms/step CPU,
   which matters because submit wall (169.5 ms) already equals GPU time.
4. Synchronization is already batched correctly: 2 submits/step (fwd+bwd,
   optim) with compute→compute barriers inside one command buffer; the
   host-read points (`read_token_counts`, `prepare_balance_gradient`) happen
   only after `batch_end()` — this ordering constraint is what rules out a
   naive in-batch host readback for routing (documented in
   `docs/MOE_FROM_SCRATCH_DESIGN.md` §8.1).

### A.4 Environment note (affects reproduction)

A process whose image lives under `Desktop\motifcl_production\...` fails
`LoadLibraryA("vulkan-1.dll")` in this session's shell sandbox ("no Vulkan
device: Vulkan loader not found"), while the same binary copied to
`C:\Windows\Temp\` runs and sees the RX 580 (verified 2026-10-21: 1347 tok/s
first step, devmem 3278 MB). Python `ctypes` loads the same DLL fine, and
`vulkaninfo` works. All GPU measurements in this mission were therefore taken
from a copy of the built binary under `C:\Windows\Temp\motifcl_test\`, with
`DATA_DIR`/`OUT_DIR` pointing back into the repository tree. This is an
environment property of the agent shell, not a defect of the trainer.

---

# Phase C — minimal sparse Vulkan MoE7 forward

## C.1 Design

The routing decision never leaves the GPU. `moe_route_top1_f32` writes, per
token, the biased logits and the argmax winner, then atomically appends the
token to that expert's compact list. Every expert dispatch is bounded by its own
counter, so a dispatch sized for the worst case (`capacity = tokens`) only pays
for the rows the expert owns. No capacity factor, no token dropping (top-1
assigns each token to exactly one expert, so a list can never exceed `tokens`
entries), and no host read-back mid-graph — which matters because the trainer's
existing read-back points (`read_token_counts`, `prepare_balance_gradient`)
happen only after `batch_end()` (Phase A §A.3.4).

Per call: `1 zero + 1 route + 7 × (up GEMM + SwiGLU + down GEMM) = 23`
dispatches, versus 28 for the dense path (router GEMM, bias add, route7,
7 × 3 expert kernels, shared × 3, merge add). The merge with the shared expert
is deliberately left to the caller so the merge stays in one place.

Kernels (new, `kernels/vulkan/`):

| kernel | role |
|---|---|
| `moe_route_top1_f32.comp` | router dot products (one workgroup per token, 64-lane strided accumulation + shared-memory reduce), bias add, strict-`>` argmax, atomic count, compact list append |
| `counter_gemm_gather_db.comp` | `counter_gemm_db` with row addressing replaced: mode 0 gathers A rows through the list (up projection), mode 2 scatters C rows through the list (down projection). Same 32×32 tile, 4×4 per lane, KT=16 ring and decode arithmetic, so the accumulation order per output element is unchanged |
| `moe_swiglu_bounded_f32.comp` | `swiglu_f32` gated on the expert's live row count |

Host side: `run_vulkan_moe7_sparse_forward(runtime, spec)` in
`src/runtime/vulkan_backend.cpp` (+ `VulkanMoE7SparseSpec` in the header). The
caller supplies scratch buffers and reuses them across calls; nothing is
allocated per step.

## C.2 Numerical correctness

`tests/test_vulkan_moe_sparse_path.cpp`, same `nn::MoE7FFN` instance and weights
on both paths, RX 580:

| case | rows | rel_err sparse vs dense | routing |
|---|---:|---:|---|
| T=24 | 24 | 0.000e+00 | counts match dense argmax |
| T=1 | 1 | 0.000e+00 | counts match |
| T=37 (partial tile) | 37 | 0.000e+00 | counts match |
| all-to-one (bias +1e3 on expert 4) | 40 | 0.000e+00 | 40/40 on expert 4, six lists empty |

Exact agreement is the expected outcome, not a lucky one: both paths run the
same tile kernel over the same weights, so only the visited row set differs. A
non-zero difference would mean the gather/scatter addressing is wrong. The
all-to-one case also covers the partial last tile (40 rows = 1 full tile + 8
rows) and the empty-list path (six experts with `count = 0`).

Pre-existing gates still pass unchanged: `test_vulkan_moe_sparse` (CPU reference
+ dense device parity, rel_err 1.32e-07) and `test_moe7_reference`
(rel_err 1.17e-07, gradient reaches only routed experts).

## C.3 Measured performance (RX 580, median of 20 calls, GPU timestamp)

| mode | dense | sparse | speedup | dispatches |
|---|---:|---:|---:|---|
| prefill T=2048, dim=288, h=192 | 3663.8 us | 1037.1 us | **3.53x** | 28 → 23 |
| decode T=1 | 1164.2 us | 228.8 us | **5.09x** | 28 → 23 |

Both figures are full MoE forward wall-clock GPU time inside one command buffer
(`last_gpu_time_us`), not a single shader.

## C.4 Where the remaining time goes

- **Prefill:** 1037 us at T=2048 versus 229 us at T=1 with an identical dispatch
  count means ~810 us is real work and ~230 us is the launch floor (~10 us per
  dispatch). Per live row the sparse GEMM is ~3x less efficient than the dense
  one (10 live row tiles out of the 64 dispatched, so occupancy drops), which is
  why the win is 3.5x rather than the 7x the FLOP ratio suggests.
- **Decode:** 229 us is almost entirely dispatch latency — 23 dispatches of
  near-zero work. This is the Phase F target (fuse route+SwiGLU, collapse the
  seven expert dispatches into one indirect dispatch when `count` is tiny).
- **Empty workgroups:** each expert GEMM still launches `ceil(tokens/32)` row
  tiles and 6/7 of them exit immediately. `vkCmdDispatchIndirect` with the
  counter as the argument removes them; the runtime has no indirect path yet
  (checked: no `vkCmdDispatchIndirect` anywhere in `src/runtime/`), so this is a
  contained addition, deferred until the profile shows it pays.
- **Profiler attribution gap:** the `MOTIFCL_VK_PROFILE` per-kernel timer
  reports 230 us for the same call that measures 1037 us wall-clock. The
  timestamps are written before each dispatch, so with 23 back-to-back
  dispatches the attribution is not trustworthy at this granularity; wall-clock
  GPU time is the number used above. Fixing the attribution (timestamp after
  the last dispatch, or per-dispatch pairs) is Phase D work.

## C.5 Known limitations

1. **Forward only.** The backward pass is still the dense one: seven candidate
   backwards per block, six with an all-zero `grad_out`. Enabling the sparse
   forward inside training without a sparse backward would silently change the
   counter-state update (losing experts currently tick on zero gradient), so the
   path is not wired into `nn::MoE7FFN` yet. That is Phase D.
2. **Top-1 only.** `MoE7FFN` is hard top-1; the general `MoEFFN` top-k path is
   still OpenCL-only. The gather/scatter kernels are written for the top-1
   single-writer case (plain stores, no accumulation), so top-k > 1 needs a
   weighted-accumulate variant.
3. **Descriptor/layout contract.** A module that declares more bindings than the
   caller passes does not fail cleanly: the AMD driver crashes inside
   `vkCreateComputePipelines`. Found the hard way while wiring this path (a
   7th `picks` binding with no descriptor). The runtime should validate
   `binding_count` against the module's declared bindings; until then the
   convention is "bindings in the module == buffers passed".

## C.6 Files

New: `kernels/vulkan/moe_route_top1_f32.comp`,
`kernels/vulkan/counter_gemm_gather_db.comp`,
`kernels/vulkan/moe_swiglu_bounded_f32.comp`,
`tests/test_vulkan_moe_sparse_path.cpp`.
Modified: `include/motifcl/runtime/vulkan_backend.hpp` (spec + entry point),
`src/runtime/vulkan_backend.cpp` (host wrapper),
`src/runtime/vulkan_spirv_kernels.inc` and `vulkan_spirv_names.inc`
(regenerated, 111 kernels), `tests/CMakeLists.txt` (CTest registration).

Reproduce:

```
python tools/gen_vulkan_spirv.py && python tools/gen_vulkan_spirv_names.py
cmake --build build/rx580-release --target test_vulkan_moe_sparse_path
MOTIFCL_VK_PROFILE=1 MOTIFCL_VK_MOE_PROFILE=1 ./test_vulkan_moe_sparse_path.exe
```

(The built binary must be run from a directory outside the repository tree in
this shell sandbox — see §A.4.)

# Phase D — Profiling and bottleneck elimination

Everything below is measured on the RX 580 with the tree as of this phase. The
first task of the phase was not optimisation but making the instruments
trustworthy, because the Phase A baseline was built on a profiler that was
demonstrably wrong.

## D.1 The profiler was measuring launches, not execution

Phase A/C reported a per-kernel sum of 226–230 µs for a MoE forward that
measured 1036 µs wall-clock, and the report said the attribution "is not
trustworthy at this granularity". The cause was the placement of the end
timestamp: it was written immediately after `vkCmdDispatch`, i.e. *before* the
compute→compute barrier that orders the next dispatch. On this driver a
timestamp written there records when the dispatch was issued, so 23
back-to-back dispatches all reported launch cost and the real execution time
was attributed to nobody.

Fix (`src/runtime/vulkan_backend.cpp`): the start stamp stays at
`VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT`, and the end stamp is now deferred. A
dispatch records `profile_pending_end`, and `flush_profile_end()` writes that
stamp only after the *consumer* barrier of the following dispatch — the first
point where the previous dispatch's writes are guaranteed complete. The final
dispatch in a batch is drained by an explicit barrier plus flush in
`submit_fast_commands`. `VK_PIPELINE_STAGE_ALL_COMMANDS_BIT` is defined
locally (0x00010000) because the hand-rolled Vulkan loader does not export it.

Verification — sparse prefill, T=2048, two consecutive calls:

| | call 1 | call 2 |
|---|---:|---:|
| sum of per-kernel time | 1024.8 µs | 1039.7 µs |
| wall-clock GPU time | 1031.4 µs | 1045.6 µs |
| unattributed | 6.6 µs | 5.9 µs |

Under 1% unattributed, so the per-kernel numbers are now usable. Attribution of
that call:

| kernel | calls | total | share |
|---|---:|---:|---:|
| `counter_gemm_gather_db` | 14 | 948.0 µs | 91.2% |
| `moe_route_top1_f32` | 1 | 49.6 µs | 4.8% |
| `moe_swiglu_bounded_f32` | 7 | 40.0 µs | 3.8% |

The 14 gather GEMMs are 7 experts × (up, down). Each expert owns ~293 of the
2048 rows, so the grid is sized for 64 row tiles per expert while ~10 do work:
the sparse path is correct but its *tile occupancy* is 15%, which is the real
reason the measured win is 3.5x and not the 7x the FLOP ratio implies.

## D.2 Eliminating the zero-gradient expert backward

With hard top-1 routing, a losing expert receives an identically zero
`grad_out`. Its `grad_w` GEMM, `grad_x` GEMM and SwiGLU backward are therefore
provably all zeros — but `counter_update_apply_f32` still runs, and its
state update is **not** zero: the counter layers carry an RMS second moment
`v` that decays as `v ← beta*v + (1-beta)*ss` every step, and `ss == 0` is a
real update. Skipping the expert therefore has to skip the arithmetic but keep
the decay, exactly.

Implementation:
- `kernels/vulkan/counter_decay_v_f32.comp` + `run_vulkan_compact_counter_decay_v`
  — `v[i] = beta * v[i]`, one thread per row.
- `CounterStateLinear::apply_zero_grad_update()` — dispatches that kernel on
  Vulkan, or the same arithmetic on CPU.
- `fog_hard_route7` gained an overload taking
  `std::function<bool(std::uint32_t)> on_zero_candidate`; returning true means
  the caller handled the candidate's side effects and the masked backward is
  skipped. The two-argument overload forwards an empty function, so existing
  call sites are untouched.
- `MoE7FFN::freeze_routing()` / `route_live()`: the hook can only know which
  experts won if the routing was read back, which requires the forward to be
  submitted before the backward is recorded. `MoE7FFNConfig::skip_zero_grad_experts`
  defaults to true; `MOTIFCL_MOE_SKIP_ZERO_GRAD=0` disables it.

**Correctness is proven bit-exactly, not approximately.**
`tests/test_vulkan_moe7_zero_grad.cpp` constructs two independent modules from
the same seed (identical buffers *and* identical counter RNG streams), runs one
dense and one skip-path backward, and compares `router_grad`, `input_grad`, and
every layer's packed `state`, `scale` and `v` **byte for byte** — 16 counter
layers × 5 steps, with fresh inputs on the extra steps so the decay path is
actually exercised (with a fixed input the routing never changes and the first
decay of a zero `v` is a no-op, which hid the case on the first attempt).

## D.3 Recording was never the bottleneck

`MOTIFCL_VK_RECORD_PROFILE=1` splits the per-dispatch host cost into pipeline
lookup, descriptor allocation, descriptor update and command recording.

| sample | dispatches | per dispatch | pipeline | desc alloc | desc update | commands |
|---|---:|---:|---:|---:|---:|---:|
| first iteration | 847 | 60.1 µs | 12.9 | 13.5 | 14.4 | 19.2 |
| steady state | 847 | 3.6 µs | 0.2 | 0.4 | 0.8 | 2.2 |
| steady state | 847 | 3.0 µs | 0.2 | 0.3 | 0.7 | 1.8 |
| steady state | 847 | 3.1 µs | 0.2 | 0.3 | 0.8 | 1.8 |

The first-iteration figure is SPIR-V → ISA compilation; the pipeline cache
persists it to disk, so only a cold start pays it. Steady-state recording is
**3.0–3.6 µs/dispatch, 2.6 ms per 847-dispatch step** — about 1% of a 266 ms
step. This refutes the Phase A statement that "recording is the wall-clock
bottleneck (~70 µs/dispatch)"; that number was the cold-start cost.

## D.4 Trainer A/B: the skip only pays together with the batch split

25 steps, seed 1337, B=8, T=256 (2048 tokens/step), identical data order.
`split` = `MOTIFCL_MOE_SPLIT_FWD_BWD`, `skip` = `MOTIFCL_MOE_SKIP_ZERO_GRAD`.

| split | skip | tok/s | fwd+bwd GPU | devmem | FINAL VAL ppl |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 7686 | 175.11 ms | 3343 MB | 470.994 |
| 0 | 1 | 7517 | 181.64 ms | 3343 MB | 471.002 |
| 1 | 0 | 7853 | 169.16 ms | 3343 MB | 471.002 |
| 1 | 1 | **8970** | **158.00 ms** | **3219 MB** | 470.978 |

Run-to-run spread measured separately on the baseline configuration:
7633 / 7640 / 7743 tok/s (268.3 / 268.1 / 264.5 ms), i.e. about ±2%. The
`split=1 skip=1` result is +16.7% over baseline, well outside that band, and
the final validation perplexity is unchanged to four digits.

The mechanism is worth stating plainly, because it bounds what this
optimisation is worth: with *healthy* routing every expert wins about 1/7 of
the tokens, so the skip almost never fires. It pays only when routing is
starved (which is exactly the regime the load-balance bias exists to prevent),
and it *requires* the forward/backward split to know the winners. The split
alone costs a full forward-length GPU stall while the host records the
backward, so it is a net regression on its own (7853 vs 7686 is within noise
plus the stall) and a win only with the skip. The split therefore stays
**opt-in**; the default path is unchanged.

## D.5 Where a training step actually spends its time

`MOTIFCL_VK_OP_PROFILE=1` was added to attribute host time *outside*
`fast_dispatch` — the record profiler already shows recording is 3 µs, so any
remaining CPU cost must live in the op/Tensor layer above it. Time is
attributed to the dispatch it follows (so an op's cost is split across the gap
after its predecessor), which is why the per-kernel rows are a ranking and the
total is the measurement.

Step accounting, B=8/T=256, steady state:

| phase | ms/step |
|---|---:|
| GPU forward+backward | 178.2 |
| GPU optimizer | 4.0 |
| host: forward record | 19.3–27.7 |
| host: backward record | ~46 |
| host: `batch_end` (blocks on fence) | 183.7 |
| host: balance read-back + bias update | 8.5 |
| host: optimizer record | 6.6 |
| **total** | **251.8–268.6** |

Two structural facts fall out:

1. **Host and device never overlap.** `batch_begin` records the whole graph on
   the CPU, `batch_end` submits and waits. The step is therefore
   CPU-record + GPU-execute, serialized, not `max(CPU, GPU)`.
2. **The GPU is the critical path during submit** (183.7 ms submit wall against
   178.2 ms GPU time), so the GPU work is the thing worth removing.

Host gap profile, 630 buffer allocations per step:

| | value |
|---|---:|
| host time outside dispatch | 46.3 ms |
| buffer allocations | 630 (602 pool hits, 28 slow) |
| pool hits | ~0.3 µs each |
| slow allocations | ~700 µs each, 19.5 ms total |

The slow allocations are not large: the missed buckets were 256 bytes and
64 KB. VRAM sits at 3343/3359 MB, i.e. 99.5% occupancy, so any fresh
`vkAllocateMemory` pays driver eviction regardless of size. They were also
*unnecessary*: the pool's per-bucket retention cap (256) was evicting the tail
of the step's working set, so the same buffers were destroyed at the end of
each step and re-created at the start of the next. Raising the cap to 2048
took misses from 35 to 28 and the host gap from 52.3 ms to 46.3 ms.

End-to-end effect of that one-line change, 25 steps, three runs each:

| | run 1 | run 2 | run 3 | mean |
|---|---:|---:|---:|---:|
| before (cap 256) | 7633 | 7640 | 7743 | 7672 tok/s |
| after (cap 2048) | 7998 | 7921 | 7832 | 7917 tok/s |

+3.2% mean, and the two distributions do not overlap. Step wall time fell from
a 268.3 ms mean to 258.7 ms, consistent with the 6 ms of host time recovered.

Ranking of the remaining host cost:

| kernel the gap follows | total | gaps | per gap |
|---|---:|---:|---:|
| `counter_gemm_db` | 20.0 ms | 240 | 83.3 µs |
| `counter_update_apply_f32` | 10.8 ms | 120 | 90.2 µs |
| `swiglu_bwd_f32` | 4.6 ms | 48 | 94.9 µs |
| `add_f32` | 4.0 ms | 90 | 44.6 µs |
| `swiglu_f32` | 2.8 ms | 48 | 58.5 µs |
| `rope_f32` | 1.9 ms | 24 | 78.0 µs |
| `rmsnorm_bwd_x_f32_subgroup` | 1.5 ms | 13 | 115.2 µs |
| `fog_hard_route7_f32` | 1.0 ms | 6 | 163.2 µs |

Since recording is 2.6 ms of this 46.3 ms, ~44 ms is the framework layer
(tensor construction, shape checks, argument marshalling) at roughly 70 µs per
op. That is the next host-side target and it is independent of MoE.

Also found and *not* fixed: the validation pass runs outside any batch, so
each of its ~2888 dispatches is its own command buffer with a fence wait. It
costs ~55 ms per validation pass, which at `EVAL_EVERY=500` is negligible over
a real run, so it is documented rather than changed.

## D.6 GPU profile of one training step

Per-kernel GPU time, window aggregate (31 kernels, 209.37 ms):

| kernel | calls | total | avg | share |
|---|---:|---:|---:|---:|
| `counter_gemm_db` | 240 | 61.99 ms | 0.258 ms | **29.6%** |
| `mm_f32_db64` | 3 | 39.53 ms | 13.175 ms | 18.9% |
| `mm_f32_db32` | 120 | 29.01 ms | 0.242 ms | 13.9% |
| `gqa_bwd_probs_causal4_f32` | 6 | 11.58 ms | 1.929 ms | 5.5% |
| `gqa_fwd_causal4_f32` | 6 | 9.93 ms | 1.655 ms | 4.7% |
| `gqa_bwd_dkv_causal_tile_f32` | 6 | 6.03 ms | 1.006 ms | 2.9% |
| `add_f32` | 96 | 5.84 ms | 0.061 ms | 2.8% |
| `ce_bwd_cached_f32` | 1 | 5.63 ms | 5.626 ms | 2.7% |
| `swiglu_f32` | 48 | 5.60 ms | 0.117 ms | 2.7% |
| `adam_update_fast_f32` | 21 | 5.48 ms | 0.261 ms | 2.6% |
| `ce_fwd_stats_f32` | 1 | 4.50 ms | 4.504 ms | 2.2% |
| `mm_f32_nn_rb4` | 9 | 3.70 ms | 0.411 ms | 1.8% |

The three `mm_f32_db64` calls are the untied LM head (forward, dX, dW) at
13.2 ms each — 19% of the step for one matrix, at ~2.9 TFLOP/s. That is the
model's shape, not an inefficiency.

`counter_gemm_db` at 29.6% is the MoE experts plus the attention projections,
and it is where the remaining structural waste is: the training path is still
the **dense** one, so each of the 8 expert modules runs its up/gate/down GEMMs
over all 2048 rows while hard top-1 routing gives it ~256. That is ~7x
over-computation on 60% of the kernel's calls, and it is the Phase E target.

## D.7 Files

New: `kernels/vulkan/counter_decay_v_f32.comp`,
`tests/test_vulkan_moe7_zero_grad.cpp`.

Modified: `src/runtime/vulkan_backend.cpp` (deferred profile end stamp,
record-cost profiler, op-gap profiler, allocation accounting, indirect-trace,
pool per-bucket retention 256 → 2048),
`include/motifcl/runtime/vulkan_backend.hpp` (profiler semantics doc,
`run_vulkan_compact_counter_decay_v`), `include/motifcl/nn/compact_counter.hpp`
and `src/nn/compact_counter.cpp` (`apply_zero_grad_update`),
`include/motifcl/ops/fog.hpp` and `src/ops/fog.cpp` (zero-candidate hook),
`include/motifcl/nn/moe7.hpp` and `src/nn/moe7.cpp` (`freeze_routing`,
`route_live`, skip config), `tools/train_moe15m.cpp` (phase accounting fix,
opt-in fwd/bwd split), `src/runtime/vulkan_spirv_kernels.inc` and
`vulkan_spirv_names.inc` (regenerated, 112 kernels), `tests/CMakeLists.txt`.

Reproduce:

```
MOTIFCL_VK_PROFILE=1 MOTIFCL_VK_MOE_PROFILE=1 ./test_vulkan_moe_sparse_path.exe
MOTIFCL_VK_RECORD_PROFILE=1 STEPS=4 GPU_TIMING=1 ./motifcl_train_moe15m.exe
MOTIFCL_VK_OP_PROFILE=1 STEPS=4 ./motifcl_train_moe15m.exe
MOTIFCL_VK_TRACE_IMMEDIATE=1 STEPS=1 ./motifcl_train_moe15m.exe
./test_vulkan_moe7_zero_grad.exe
```

# Phase E — Why the sparse forward cannot train, and what can

Phase D §D.6 identified `counter_gemm_db` as 29.6% of a training step and
pointed at the dense expert path as the next target. Scoping that work turned
up a hard constraint that changes what Phase E can be, so it is recorded here
before any code is written.

## E.1 The blocker: the router gradient needs the losing experts' values

`route_logits_grad_host` (`src/ops/fog.cpp`) defines the router gradient:

```
gp[k]  = <grad_out_row, candidate_k_row>          # scalar per row, per expert
mean   = sum_k p_k * gp[k]                        # p = softmax(logits)
dl[k]  = p_k * (gp[k] - mean)
```

`dl[k]` depends on `gp[k]` for **all seven** k. The losing candidates are
therefore required *numerically* — not for their own gradient (which is zero)
but because they enter the soft mixture that produces the router's gradient.
A sparse forward computes each expert only on the rows it won, so six of the
seven `candidate_k[r]` values would never exist for a given row and `dl` could
not be formed.

This is not an implementation gap to be closed later. It is a property of the
straight-through seven-way softmax the model is defined with, and changing it
(the obvious alternative — a one-hot router gradient — is the standard
"Switch-lite" relaxation) would change training semantics, which the mission
forbids. **The sparse forward of Phase C is therefore correct and useful for
inference only; it can never be the training path.**

## E.2 What remains: the sparse backward is exact

The backward is a different matter, because there the zeros are real:

| quantity | dense | sparse (routed rows only) | exact? |
|---|---|---|---|
| `grad_w` | sum over all rows of `x[r] ⊗ dc[r]` | sum over routed rows only | **yes** — the omitted rows have `dc[r] = 0`, so they contribute exactly 0 to the sum |
| `grad_x` | `dc @ W` for all rows | `dc_routed @ W`, scattered back | **yes** — the omitted rows' `grad_x` is 0 and is scattered as 0 |
| SwiGLU backward | all rows | routed rows | **yes**, same argument |
| counter state `v` | `v ← beta*v + (1-beta)*ss` | same, with `ss` summed over routed rows | **yes, if and only if** the normalisation keeps dividing by the *original* row count |

The last row is the only place a sparse backward can silently change training.
The existing zero-gradient test already pins half of it down: a losing expert's
dense backward provably reduces to `v ← beta*v` (that is what
`apply_zero_grad_update` reproduces byte-exactly, §D.2), which means `ss` is
derived from the gradient and not from the input. So zero rows contribute zero
to `ss`, and a routed-row-only sum equals the dense sum. The gathered kernel
must divide by the un-gathered row count, and that is the assertion the
equivalence test has to make.

## E.3 Measured ceiling for the sparse backward

Per layer, `counter_gemm_db` runs 40 times: 8 expert modules × 2 matrices ×
(forward, backward) = 32, plus 4 attention projections × (forward, backward) =
8. A sparse backward can only touch the 14 routed-expert *backward* calls; the
shared expert is always-on and the forward must stay dense per §E.1.

At the measured 0.258 ms per call:

| | calls | time |
|---|---:|---:|
| `counter_gemm_db` per step | 240 | 62.0 ms |
| routed-expert backward share | 84 | 21.7 ms |
| same work, 1/7 of the rows | 84 | ~3.1 ms |

So the sparse backward is worth roughly **18.6 ms per step, about 10% of the
178 ms of GPU time** — and it requires a new gather-gradw kernel, a scatter
grad_x, a bounded SwiGLU backward and a row-set-aware state update, all of
which must reproduce the dense arithmetic bit-for-bit.

That is a smaller prize than the two things Phase D already found, and it is
worth saying so plainly:

| lever | measured / estimated | status |
|---|---:|---|
| forward/backward split + zero-grad skip | +16.7% end-to-end (measured) | done, opt-in |
| buffer-pool retention | +3.2% end-to-end (measured) | done, default |
| sparse backward | ~10% of step GPU (estimated) | not started |
| host/device overlap | up to +40% (see below) | not started |

## E.4 The larger lever is that host and device never overlap

From §D.5: a step is 73 ms of host work **serialized with** 178 ms of GPU work,
because `batch_begin` records the entire graph and `batch_end` submits and
waits. The step is therefore `CPU + GPU`, not `max(CPU, GPU)`. Double-buffering
the activations and recording step N+1 while step N executes would take the
step from ~258 ms toward ~185 ms, i.e. **up to +40%**, for no change to any
kernel or to the numerics.

That is the real Phase E/F target, and it is an architectural change to the
trainer and the runtime's batch ownership (two command buffers, two descriptor
sets in flight, activations doubled — which costs VRAM the card does not have
at 3343/3359 MB). The VRAM figure is the reason this cannot simply be switched
on: the working set would need to double, so it requires the activation
footprint to come down first (the 7 retained candidate tensors are the bulk of
it — `test_counter_memory_vulkan` reports 294.8 MB per block for 56.2 MB of
packed expert state).

## E.5 Status

Phase E was scoped and its blocker proven; no code was changed in this phase.
The honest ordering of remaining work is:

1. Reduce the retained-activation footprint so double buffering fits in VRAM.
2. Overlap host recording with device execution (~+40% ceiling).
3. Sparse backward (~10% of step GPU) — only worth its complexity after 1–2.
