# Vulkan MoE sparse execution — phases A–G

Date 2026-10-21/22. Repo `motifcl_production`, branch `perf/vulkan-moe-sparse`
(created from `main` at `ad6a00f`), base commit
`ad6a00f38c9ec11bc635c68492e25c6828f8e1d3` (working tree was dirty: 23 modified,
24 untracked — the 2026-10-18 hardening patch, not yet committed; it is included
in the phase A–D commit because the two overlap in
`src/runtime/vulkan_backend.cpp` and the regenerated SPIR-V blob).

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

The slow allocations were not large: the missed buckets were 256 bytes and
64 KB. The first reading of this was wrong and is corrected in §E.6 — the
`3343/3359 MB` figure is the run's own live/peak, not device occupancy (the
card is 8 GB, and a larger eval batch elsewhere in the trainer already reached
6.8 GB), so there was never any eviction pressure. What the numbers did show
correctly is that the allocations were *unnecessary*: the pool was failing to
recycle a stable, shape-deterministic working set. Raising the per-bucket
retention cap from 256 to 2048 took misses from 35 to 28 and the host gap from
52.3 ms to 46.3 ms, which was a real gain but not the root cause.

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

From §D.5: at the time of that measurement a step was 73 ms of host work
**serialized with** 178 ms of GPU work, because `batch_begin` records the
entire graph and `batch_end` submits and waits. The step was therefore
`CPU + GPU`, not `max(CPU, GPU)`, and overlapping them looked like a +40%
ceiling.

**That estimate is stale and the honest number is much smaller.** §E.6 removed
34 ms of that host work (the pool misses), so the current step is ~29 ms of
host work against 171.7 ms of GPU time. The overlap ceiling is therefore about
`max(171.7, 29) = 171.7` versus the present 200.5 ms, i.e. **~+16%**, and less
in practice because the balance read-back and the optimizer both consume GPU
results and cannot be recorded ahead of them. The measurement that made the
overlap look like the big prize was measuring an allocation bug.

That is the real Phase E/F target, and it is an architectural change to the
trainer and the runtime's batch ownership: two command buffers, two descriptor
sets in flight, two fences. It does **not** require doubling the activations —
the backward writes its own temporaries and reads the forward's outputs, and a
single queue orders the two submissions — so the VRAM it needs is one extra
command buffer, not a second graph. What it did need was for the pool to stop
leaking allocations, which §E.6 fixed.

## E.6 Fixed: power-of-two allocation classes were the real cost

`buffer_bytes_live()` was reporting 3343 MB while the graph asked for 2195 MB,
so 1148 MB — **34% of device memory — was rounding waste**, because every
allocation was rounded up to the next power of two (a 2.36 MB activation took
a 4 MB buffer; the 268 MB logits took 512 MB).

Replacing the power-of-two class with the next eighth of an octave (at most
12.5% over the request) changed three things at once:

| | power-of-two | eighth-octave |
|---|---:|---:|
| reserved | 3343 MB | **2360 MB** |
| requested | 2195 MB | 2195 MB |
| rounding waste | 1148 MB (34%) | 165 MB (7%) |
| buffer allocations per step | 630 (602 pooled, 28 slow) | 630 (**630 pooled, 0 slow**) |
| allocation time per step | 19.5 ms | **0.13 ms** |
| host time outside dispatch | 46.3 ms | **14.0 ms** |
| host: forward record | 19.3–27.7 ms | 5.0 ms |
| step total | 251.8–268.6 ms | **200.5 ms** |

End-to-end, 25 steps, three runs each:

| | run 1 | run 2 | run 3 | mean |
|---|---:|---:|---:|---:|
| baseline (start of Phase D) | 7633 | 7640 | 7743 | 7672 tok/s |
| pool cap fix (§D.5) | 7998 | 7921 | 7832 | 7917 tok/s |
| size-class fix | **9341** | **9291** | **9284** | **9305 tok/s** |

+21.3% over the Phase D baseline and +17.5% over the pool-cap fix alone, with
no change to any kernel, any numeric, or the dispatch count. The mechanism is
worth stating precisely because the first explanation was wrong: the working
set is shape-deterministic, so the pool should recycle everything. Rounding to
powers of two inflated it enough that peak demand inside a batch exceeded what
the pool held, so the tail was allocated fresh (and freed again) every step —
at ~700 µs per `vkAllocateMemory`, 28 times a step. Removing the rounding made
peak demand fit, and the misses went to zero.

## E.7 Status

Phase E was scoped, its blocker proven, and one bottleneck fixed; the sparse
backward was deliberately not started. The honest ordering of remaining work
is now:

1. ~~Reduce the retained-activation footprint so double buffering fits~~ —
   done, by fixing allocation classes rather than by touching activations
   (§E.6): 2360 MB, 630/630 pool hits, +21.3% end-to-end.
2. Overlap host recording with device execution. The ceiling is now ~+16%,
   not the +40% §E.4 originally projected, because fixing the allocation
   classes removed most of the host work that would have been hidden. The
   remaining work is two command buffers, two fences and two descriptor sets in
   flight, with the trainer submitting the forward before recording the
   backward; it does not require doubling the activations (§E.4).
3. Sparse backward (~10% of step GPU) — only worth its complexity after 2.

The GPU is now decisively the critical path again: 171.7 ms of a 200.5 ms step
is device execution, and the single largest kernel remains `counter_gemm_db` at
29.6% of it. **That is why the ordering above is not the priority order.** The
next thing to do is not (2) or (3); it is Phase F — measuring whether the
dominant kernel is efficient at all, because 29.6% of the step sitting at 14%
of peak is a bigger and cheaper lever than either. Phase F found that it is
not, and why.

# Phase F — Investigation of counter_gemm_db

Scope: measurement only, no kernel changes. The question was whether the
29.6%-of-step kernel is inefficient (and therefore a lever) or simply doing the
work it must do.

## F.1 Reading the driver's real register allocation

`glslc` cannot answer occupancy questions — the ISA compile happens in the
driver. `VK_AMD_shader_info` can, and two things were needed to get it:

1. The extension must be **enabled at device creation** for
   `vkGetDeviceProcAddr` to return `vkGetShaderInfoAMD`; without it the pointer
   is null and the first attempt silently produced nothing.
2. `VK_SHADER_INFO_TYPE_STATISTICS_AMD` is **0**, not 1 (the enum is
   STATISTICS=0, BINARY=1, DISASSEMBLY=2), and the call is a two-step
   size-then-payload protocol.

Both are now in the runtime behind `MOTIFCL_VK_SHADER_INFO=1`, which prints
VGPR/SGPR/LDS/scratch per pipeline plus the resulting waves per SIMD.

## F.2 Occupancy (RX 580, Polaris 10, 4 SIMDs/CU, 36 CU)

| kernel | VGPR | SGPR | LDS | waves/SIMD | limiter |
|---|---:|---:|---:|---:|---|
| `counter_gemm_db` | 103 | 34 | 8448 B | **2** | VGPR |
| `counter_gemm_gather_db` | 103 | 34 | 8448 B | **2** | VGPR |
| `mm_f32_db32` | 99 | 26 | 8192 B | 2 | VGPR |
| `mm_f32_db64` | 73 | 26 | 16384 B | 3 | VGPR (LDS allows 4) |
| `mm_f32_nn_rb4` | 38 | 26 | 4288 B | 6 | VGPR |
| `mm_f32_nt` / `tn` | 24-25 | 28 | 2176 B | 10 | - |
| `mm_f32_nn` | 31 | 24 | 2112 B | 8 | - |
| `gqa_fwd_causal4_f32` | 24 | 27 | 5120 B | 10 | LDS |
| `swiglu_f32`, `swiglu_bwd_f32` | 5 | 18-22 | 0 | 51 | - |
| `rmsnorm_f32_subgroup` | 6 | 22 | 0 | 42 | - |

A wave64 needs 64 lanes of VGPRs from a 256-per-lane file, so
`waves/SIMD = floor(256 / vgprs)`; the cap is 10. `counter_gemm_db` sits at
**2 of 10**, i.e. 20% of the hardware's latency-hiding capacity. The 4x4
per-lane accumulator tile alone is 16 registers, so 103 means the decode chain
is being kept live across the k-loop.

## F.3 Achieved throughput, same arithmetic

| kernel | shape | FLOPs | time | GFLOP/s | % of 6.17 TFLOP/s |
|---|---|---:|---:|---:|---:|
| `counter_gemm_db` (dense, 14 calls) | 7 x [2048,288]x[288,192] | 3.17 G | 3663.8 us | **866** | 14.0% |
| `counter_gemm_gather_db` (sparse, 14 calls) | 7 x [~293,288]x[288,192] | 0.453 G useful | 1037.1 us | **437** | 7.1% |
| `mm_f32_db64` (LM head) | [2048,288]x[288,32000] | 37.75 G | 13.175 ms | **2866** | 46.4% |

Two separate gaps, and they have different causes.

## F.4 The sparse kernel's gap is parallelism, not waste

The sparse GEMM does the *same tile arithmetic* as the dense one and still
runs at half the efficiency per useful FLOP. The reason is the grid size, not
the gather:

- dense: 64 row tiles x 6 column tiles = **384 workgroups** per expert, one
  wave each, against a machine that holds 288 waves concurrently (36 CU x
  4 SIMD x 2 waves). One dispatch nearly fills it.
- sparse: ~10 live row tiles x 6 = **60 workgroups** per expert dispatch. The
  grid is still sized for 64 row tiles, so 324 of 384 workgroups exit
  immediately — but even the 60 real ones occupy only 60 of 288 slots, i.e.
  **21% of the machine**, and no amount of removing empty workgroups changes
  that.

This is the missing explanation for the Phase C anomaly (`gather_db` = 948 us
of a 1040 us prefill dispatch, 1.65x less efficient per FLOP than dense): the
work is real, but it is delivered in seven small batches instead of one big
one.

**Consequence:** the fix is not a better tile, it is **one dispatch for all
seven experts** — a flat job list of (expert, row-tile, col-tile) triples, with
the grid size read on the device via `vkCmdDispatchIndirect`. That turns
7 x 60 = 420 workgroups into a single dispatch that fills the machine, and the
dense number (866 GFLOP/s) is the reference it should approach. Estimated
prefill GEMM: 948 us -> ~500 us.

This also corrects an earlier judgement: the Phase C report deferred indirect
dispatch on the grounds that empty workgroups probably cost little. They do
cost little — but the *small live grid* costs half the throughput, and indirect
dispatch is the only way to fix it, because the grid size depends on a value
the host does not have.

## F.5 The dense kernel's gap is occupancy (hypothesis)

> **Superseded by G.2.** The ISA was read in the next phase and the hypothesis is
> wrong: the gap is instruction issue, not occupancy. The VGPR count (103 vs 99
> for the equivalent FP32 kernel) is a 4-register difference and cannot explain a
> 3x throughput gap. Kept as written so the reasoning trail stays visible.

`counter_gemm_db` at 866 GFLOP/s is 3.3x below `mm_f32_db64` at 2866 GFLOP/s,
despite reading **5.4x fewer weight bytes** (6-bit packed vs F32). Bandwidth is
therefore not the limiter; the byte-load decode path and occupancy are.

The correlation is suggestive and not yet proven: `mm_f32_db64` runs at 3
waves/SIMD with 73 VGPRs, `counter_gemm_db` at 2 waves with 103. Going from 2 to
3 waves is a 50% increase in in-flight work to hide the `state[]` load latency.
Reducing the kernel below 85 VGPRs would give 3 waves, below 64 would give 4
(LDS allows 7.8, so LDS is not binding).

What is *not* yet separated: how much of the 866 vs 2866 gap is occupancy and
how much is the decode instruction stream itself (3 scalar byte loads plus a
`scale[]` load per group of four weights, with a 216-byte stride between
adjacent output columns). Separating them needs a variant kernel, which is the
next step rather than a conclusion.

## F.6 Next steps, in order

1. **Flat job list + `vkCmdDispatchIndirect`** for the sparse expert GEMMs.
   Largest, best-understood win: recovers roughly half of a 948 us prefill
   dispatch, and it is the only fix for the small-grid problem.
2. **VGPR budget for `counter_gemm_db`**: get 103 -> <=85 for 3 waves/SIMD and
   re-measure. Cheap to try, and the measurement says occupancy is the
   difference between this kernel and the F32 one.
3. **State layout**: if (2) does not move the number, transpose the packed
   state so a warp's lanes read contiguous bytes instead of 12 bytes every
   216, and re-measure. This changes the on-disk checkpoint layout, so it is
   deliberately last.

# Phase G — The counter GEMM is issue-bound, and the measured way around it

Phase F could only see VGPR/LDS and guessed at occupancy. This phase reads the
driver's disassembly, settles the question, and measures a remedy.

## G.1 An isolated harness first

Iterating on a kernel through a 25-step trainer run is too slow and too noisy,
so `benchmarks/bench_vulkan_counter_gemm.cpp` issues only the counter
projection, at the trainer's exact shapes (`d_model=288`, `hidden_exp=192`,
`batch=2048`), and reports GPU time from timestamp queries plus the achieved
GFLOP/s. Every number below comes from it unless stated otherwise.

## G.2 The ISA, which is the whole answer

`MOTIFCL_VK_SHADER_ISA=<kernel>|all` dumps the GCN disassembly through
`VK_SHADER_INFO_TYPE_DISASSEMBLY_AMD` (type **2**; the enum is
STATISTICS=0, BINARY=1, DISASSEMBLY=2) and prints an opcode histogram.
`counter_gemm_db` is **1381 instructions**, of which:

| category | count | share |
|---|---:|---:|
| **FMA** (`v_mac_f32`) | **256** | **18.5%** |
| integer address arithmetic (`v_add_u32`, `v_lshlrev_b32`, `v_and_b32`, `v_mul_lo_u32`, …) | 315 | 22.8% |
| predication / branch (`s_and_b64`, `s_and_saveexec_b64`, `s_cbranch_execz`, `v_cmp_*`) | 197 | 14.3% |
| waits (`s_waitcnt`) | 118 | 8.5% |
| register moves (`v_mov_b32`) | 85 | 6.2% |
| LDS reads (`ds_read_b128`, `ds_read_b32`) | 64 | 4.6% |
| decode converts + FMUL (`v_cvt_f32_i32`, `v_mul_f32`) | 64 | 4.6% |
| global loads (all `buffer_load_ubyte` except 12) | 36 | 2.6% |
| LDS writes | 21 | 1.5% |
| other | 225 | 16.3% |

The kernel issues **5.4 instructions per FMA**, and only 18.5% of its
instruction stream is the arithmetic it exists to perform. It is **issue-bound**
on address computation and on the per-element bounds predication
(`if (gc < p.out_features && gk < p.in_features)`) that wraps every decode.

This retires the Phase F hypothesis. `mm_f32_db32` uses 99 VGPRs and this kernel
103 — a 4-register difference cannot produce a 3x throughput gap — and moving
from 2 to 3 waves/SIMD would not fix a 5:1 instruction overhead. Occupancy was
never the lever.

## G.3 Same tile, same grid, same FLOPs

Forcing the FP32 kernel to the same variant removes every confound except the
decode itself:

| kernel | shape | time | GFLOP/s | % of 6.17 TFLOP/s |
|---|---|---:|---:|---:|
| `counter_gemm_db` | 2048×192×288 | 272.5–295.2 µs | 767–831 | 12.4–13.5% |
| `mm_f32_db32` | 2048×192×288 | 159–164 µs | 2.52–2.62 TFLOP/s | 41–42% |
| `counter_gemm_db` | 2048×288×288 | 351.4–387.7 µs | 877–967 | 14.2–15.7% |
| `mm_f32_db32` | 2048×288×288 | 198–203 µs | 2.5–3.0 TFLOP/s | 42–49% |

A consistent **1.5–2.0x**. The dense kernel reads 5.4x more weight bytes
(41 KB vs 221 KB at the largest shape) and is still twice as fast, which
confirms that at these shapes weight traffic is irrelevant and only issue rate
matters.

## G.4 Remedy: decode to dense, then the tuned GEMM

`MOTIFCL_COUNTER_DECODE_DENSE=1` decodes the packed state into a reusable dense
mirror (`counter_decode_weight_f32`, ~15 µs) and runs the ordinary FP32 GEMM on
it. `CounterStateLinear` keeps the mirror as a private buffer and calls
`run_vulkan_f32_matmul*` **directly** on the runtime — routing through the
generic Tensor matmul op instead costs ~19 ms of host time per step, which is
the same size as the GPU win and cancels it exactly (measured; see G.5).

| shape | mode | counter | decode | dense GEMM | decode+GEMM | gain |
|---|---|---:|---:|---:|---:|---:|
| expert_up 2048×288×192 | forward | 349.0 | 15.5 | 161.0 | 177.9 | **1.96x** |
| expert_down 2048×192×288 | forward | 295.2 | 15.5 | 164.2 | 181.8 | **1.62x** |
| expert_up 2048×288×192 | backward | 273.4 | 15.2 | 160.6 | 177.3 | **1.54x** |
| expert_down 2048×192×288 | backward | 272.5 | 15.2 | 159.2 | 174.2 | **1.56x** |
| attn_qkv 2048×288×288 | forward | 351.4 | 20.8 | 202.9 | 225.4 | **1.56x** |
| attn_qkv 2048×288×288 | backward | 387.7 | 20.8 | 197.8 | 221.4 | **1.75x** |
| expert_up 1×288×192 | forward | 91.2 | 14.9 | 10.9 | 25.6 | **3.56x** |
| expert_down 1×192×288 | forward | 64.8 | 14.7 | 10.2 | 24.5 | **2.65x** |

## G.5 End to end: +6.9%, and why not the 13.7% of GPU it removes

25 steps, seed 1337, B=8, T=256, three runs each, all four numbers from the
same binary:

| config | tok/s | fwd+bwd GPU | step total | FINAL VAL ppl |
|---|---:|---:|---:|---:|
| fused (default) | 9146 / 9177 / 9148 | 168.4–168.9 ms | 223.2–223.9 ms | 470.923 / 470.991 / 470.982 |
| decode+dense | 9761 / 9751 / 9850 | 145.0–146.6 ms | 207.9–210.0 ms | 471.001 / 471.003 / 471.010 |

**+6.9%** (mean 9157 → 9787 tok/s), GPU **−13.7%**, step **−6.4%**, validation
perplexity unchanged to within the ~0.03 run-to-run spread already measured.

The step gain is smaller than the GPU gain because a step is `CPU + GPU`
serialized, not `max(CPU, GPU)`. Today's phase split makes that explicit:

| phase | fused | decode+dense |
|---|---:|---:|
| host: forward record | 15.7 ms | 21.4 ms |
| host+device: record incl. submit | 218.9 ms | 204.9 ms |
| of which `batch_end` (submit + fence) | 169.3 ms | 147.2 ms |
| host: balance read-back + bias | 8.5 ms | 8.5 ms |
| host: optimizer | 4.9 ms | 4.9 ms |
| **step total** | **223.9 ms** | **209.8 ms** |

`record − submit` is the pure host recording cost: **49.6 ms fused vs 57.7 ms
decode+dense**. The extra 8 ms is the added decode dispatch per projection, and
it is why a 23 ms GPU win becomes a 14 ms step win. This is the quantitative
form of the Phase E claim that host and device never overlap, and it is the
strongest argument for the double-buffered submit: at these numbers, overlapping
would recover the whole 14 ms difference and more.

## G.6 The decode case

At one row the fixed decode cost dominates the GEMM, so the path is gated to
batches of **≥128 rows** and the mission's decode target is untouched. The
microbenchmark says decode+dense is also faster at batch 1 (25.6 vs 91.2 µs and
24.5 vs 64.8 µs), which is suspicious on its face — a 1-row GEMM taking 91 µs
means the fused kernel's fixed cost, not its arithmetic, is what is being
measured there. That is a lead, not a result: it is a kernel-level number and
has not been confirmed end to end, so the gate stays.

## G.7 A process error worth recording

The first measurement of this phase was a probe that replaced the three global
byte loads in the decode with constants, to see whether the loads were the cost.
It showed 260 → 247 µs, i.e. 5%, so the loads were not the cost. The probe was
then "reverted" — but the revert copied the backup over the file *after* the
backup had itself been overwritten by the already-patched file (a retried
command re-ran the backup step first). The probe stayed live.

Everything measured between that point and its discovery — the isolated
benchmark's baseline column, the ISA dump, and a full trainer A/B — was measured
on the patched kernel. It was caught by `test_moe7_reference` reporting
`rel_err=4.379` where it had been `1.168e-07`: the counter GEMM was returning
constant-derived weights, the MoE forward no longer matched the CPU reference,
and the trainer's baseline column produced perplexity ~11 000 instead of ~471.
`git stash` proved HEAD was clean, `git diff` found the probe, and every number
in G.2–G.6 was re-measured from a `git checkout`-restored kernel with all ten
tests green.

The lesson is not "check the revert"; it is that the *baseline* column silently
became the broken arm and the result looked like a 6.9% win for the other arm.
A correctness test in the A/B loop is what caught it, and it would not have been
caught by the timing.

## G.8 What is left in this kernel

Decode+dense routes around the problem but does not fix it: it recovers 1.5–2.0x
of a 3x gap, and it pays a dispatch and a 221 KB buffer to do so. The ISA says
where the remaining 2x is, and none of it requires changing the algorithm:

1. **Remove the per-element bounds predication** (197 instructions, 14.3%). Every
   decode element tests `gc < out_features && gk < in_features` individually.
   These shapes are all multiples of 32 in both dimensions, so a single uniform
   "tile fully in bounds" test per workgroup would select a predication-free
   fast path. This is the largest single block after addressing.
2. **Hoist the address arithmetic** (315 instructions, 22.8%). The scattered
   `vec4` stores in `store_a`/`store_b` recompute a full address per component;
   the `gc * groups_per_row` and `gm * K` products are quarter-rate
   `v_mul_lo_u32` and are re-issued per tile.
3. **The 118 waits (8.5%)** follow from (1) and (2): they are the dependencies
   the predicated chains create.

A kernel with 2 instructions per FMA instead of 5.4 would reach the FP32 kernel's
42–49% of peak *without* the decode dispatch and *bit-exactly*, which is strictly
better than the path shipped here. That is the next piece of work, and the
benchmark in G.1 makes it a minutes-long loop instead of a trainer run.

# Phase H — Full-tile pipeline: 1.5–2.0x on the counter GEMM, bit-exact

Phase G named the two instruction blocks to remove: per-element bounds predication
(14.3%) and the repeated address arithmetic (22.8%). This phase removes both from
the tiles that do not need them.

## H.1 Why a full-tile path is possible

The trainer's shapes are batch 2048 (a multiple of the 32-row tile) with in/out
features 288/192 (multiples of 16), so all but the last column tile are fully
inside the matrix and no K tile is partial. For those tiles every `gm < batch`,
`gk < K` and `gc < N` test in the load/store helpers is dead, and every byte
offset advances by a constant per K tile. The edge tiles (and any shape that is
not a multiple of the tile, including decode batch 1) keep the guarded pipeline
unchanged.

## H.2 Implementation

Both pipelines live in one SPIR-V module; the choice is a per-workgroup uniform
branch, so there is no extra dispatch and the arithmetic order is untouched:

- activation offsets are `row * K/4 + kq` plus `4` per K tile (no multiply per
  tile); decode byte offsets are `column * groups_per_row * 3` plus `12` (forward)
  or `16 * groups_per_row * 3` (backward) per K tile;
- the forward loop loads `scale[column]` once per workgroup instead of once per K
  tile because it does not depend on K; the backward loop keeps its per-tile
  scale index, which is the tile's k row;
- the two directions get separate loops rather than one loop that branches on
  mode, so the forward loop never carries the backward's scale arithmetic;
- the epilogue writes the full 4x4 lane tile unguarded with a precomputed row
  stride.

`MOTIFCL_COUNTER_GEMM_GUARDED=1` sets bit 8 of the mode word and forces the
guarded pipeline. Both pipelines are in the same binary, so the A/B below is one
executable with one environment variable rather than two builds.

## H.3 Measured (RX 580, bench_vulkan_counter_gemm, 25 reps, 4 ABBA pairs)

| shape | mode | guarded | full-tile | speedup |
|---|---|---:|---:|---:|
| expert_up | forward | 452 us | 294 us | 1.54x |
| expert_down | forward | 273 us | 185 us | 1.48x |
| expert_up | backward_input | 291 us | 178 us | 1.63x |
| expert_down | backward_input | 321 us | 165 us | 1.95x |
| attn_qkv | forward | 344 us | 221 us | 1.56x |
| attn_qkv | backward_input | 389 us | 210 us | 1.85x |
| expert_up | forward (decode, batch 1) | 92 us | 93 us | 1.00x |
| expert_down | forward (decode, batch 1) | 66 us | 66 us | 1.01x |

The decode rows run the guarded pipeline under both arms, which is the control
for measurement drift: they agree to 1%. VGPR went 103 -> 104 and the pipeline
stays at 2 waves/SIMD, so occupancy did not change.

End to end, 25 steps, seed 1337, three ABBA pairs, same executable:

| arm | tok/s | step | GPU fwd+bwd |
|---|---:|---:|---:|
| guarded | 10702-10712 | 191.2-191.4 ms | 171.6-172.0 ms |
| full-tile | 11990-12030 | 170.2-170.8 ms | 151.0-151.4 ms |

+12.1% throughput, and the training trajectory is bit-identical: the six-step
losses and the final validation loss match exactly (10.4277, 10.2058, 8.8452,
7.2389, 6.7947, 6.5197; FINAL VAL 6.4265 in both arms).

## H.4 What the numbers say

The full-tile path is at parity with or ahead of decode+dense on five of the six
training shapes (decode+dense from G.4 measured 173-221 us in the same harness)
without the decode dispatch and the 221 KB dense mirror. The exception is
expert_up forward (K=288, N=192) at 294 us versus 196 us for decode+dense, and
the cause is the launch geometry rather than the instruction count: that shape
issues 64 x 6 = 384 workgroups against a machine that holds 288 waves, so it runs
as 1.3 waves and cannot fill the CUs; expert_down forward issues 64 x 9 = 576 and
reaches 1.48x. This is the same small-grid effect Phase F.4 found in the sparse
kernel. A 32x64 tile or a 2-way k split would fill the machine for that shape,
and both are compatible with the full-tile pipeline.

## H.5 Files and verification

`kernels/vulkan/counter_gemm_db.comp` (full-tile pipeline and the mode-word bit),
`src/runtime/vulkan_backend.cpp` (the wrapper's `MOTIFCL_COUNTER_GEMM_GUARDED`
selection), regenerated `src/runtime/vulkan_spirv_kernels.inc`.

Full CTest at this tree: 37 passed, 30 skipped (OpenCL unavailable), 0 failed of
67 registered. The trainer A/B above is the end-to-end witness; the guarded
pipeline remains the reference for the edge tiles.

# Phase I — What does not help: split-K on the Polaris GEMMs

Phase H left `counter_gemm_db` at 22.5% of a step. The remaining dense GEMMs were
the next question, so this phase attributed them and then tested the most obvious
remedy for the small ones.

## I.1 Where the 120 `mm_f32_db32` calls per step come from

A temporary `MOTIFCL_VK_MM_SHAPE_LOG=1` in `dispatch_vulkan_matmul_db` logged the
shapes of two trainer steps. The 120 db32 dispatches per step are not attention
projections: they are the transposed-A grad_w GEMMs, all with k = batch = 2048:

| shape (m x k x n) | calls/step | grid (tile 32) |
|---|---:|---:|
| 288 x 2048 x 192 | 48 | 9 x 6 = 54 |
| 384 x 2048 x 288 | 48 | 12 x 9 = 108 |
| 288 x 2048 x 288 | 24 | 9 x 9 = 81 |

The LM head contributes three db64 calls per step (fwd 2048x288x32000, dA
2048x32000x288, dB 288x2048x32000). At the recorded 22-30% of peak the three
grad_w shapes are ~30 ms of the ~170 ms step, and they launch 54-108 workgroups
against 288 wave slots, so "small grid" looked like the cause.

## I.2 Split-K was implemented, measured, and does not pay

The double-buffered body grew an explicit k range plus a C4 block offset, a new
`mm_split_reduce_f32` summed the blocks, and `dispatch_vulkan_matmul_db` chose
2-8 splits whenever `workgroups * splits < 288`. It only splits inside a batch:
outside one every split would be its own submit+fence. `MOTIFCL_VK_MATMUL_SPLIT_K`
turned it off or forced a count.

Measured with `bench_vulkan_gemm_shapes` (the benchmark now times each op inside a
one-dispatch batch, so a multi-dispatch path is measured end to end instead of
measuring only its last dispatch):

| shape | split-K off | auto | |
|---|---:|---:|---:|
| expert_gradw 384x2048x288 | 0.248 ms | 0.277 ms (4) | -12% |
| attn_gradw 288x2048x288 | 0.247 ms | 0.271 ms (4) | -10% |
| expert_up 2048x288x384 | 0.171 ms | 0.230 ms (2) | -34% |
| attn_qkv 2048x288x288 | 0.123 ms | 0.114 ms (1) | not split |
| head_dA 2048x32000x288 | 15.109 ms | 15.185 ms (2) | ~0% |
| head_dB 288x2048x32000 | 11.651 ms | 11.475 ms (1) | not split |

The grad_w shapes are transposed-A GEMMs: each workgroup streams the full k of
its 32-column output slice, so they are bandwidth-bound rather than wave-starved.
Splitting k leaves the traffic unchanged (the same bytes are still read once per
output tile) and adds a dispatch plus a reduce. The LM-head dA is bandwidth-bound
for the same reason: it re-reads the 2048x32000 dy once per output tile, and 160
workgroups already saturate the read path. That is what the numbers show, so the
implementation was reverted rather than shipped behind a flag.

Two supporting changes were kept: the GEMM benchmark now times ops inside a batch
(a bare dispatch under-reports any multi-dispatch path), and
`test_vulkan_gemm_parity` also checks every shape through a batch, so batched
execution (barriers, keepalive, scratch reuse) is covered by the parity test.

## I.3 Where that leaves the step

The LM head is the best-tuned block (fwd 56.8% of peak, dB 53%, dA 41%); its dA
is bandwidth-bound by the dy re-read. The worst GEMMs (grad_w, 22-30%) are
bandwidth-bound as well, so neither a bigger tile (too few workgroups: tile 64
measured 0.295 ms vs 0.248) nor split-K helps on this device; the remaining lever
there is a different decomposition with fewer re-reads, e.g. storing activations
transposed so grad_w becomes an n-tile-friendly layout. The other blocks are
`counter_gemm_db` (22.5%), GQA (15.7%) and CE (4.2%, already at the memory
bandwidth floor). Host time outside dispatch is ~19 ms of the ~170 ms step, so
host/device overlap is still the largest structural lever (~+11% ceiling).

# Phase J — The balance phase: a device-side auxiliary router gradient

Phase I left host time outside dispatch as the largest single lever. The trainer's
own phase split put ~8.5 ms of it in `balance`; this phase located that cost and
removed most of it.

## J.1 What the balance phase actually does

The balance phase runs once per MoE block (6 of them): read the routing counts
back, build the auxiliary-loss router gradient, and update the router bias. Three
25-step runs with everything else equal separate the contributors:

| variant | balance ms/step |
|---|---:|
| baseline | 8.5 |
| `AUX_WEIGHT=0` (no balance gradient) | 1.9 |
| `BALANCE_GAMMA=0` (no bias update) | 8.2 |
| both disabled | 1.7 |

So the bias update is ~0.3 ms and the routing read-back ~1.7 ms; ~6.6 ms is
`prepare_balance_gradient`: a host softmax over the [2048,7] logits plus a
`Tensor::from_cpu` upload of the [2048,7] gradient, per block, per step. 57 KB per
block does not sound like 1.1 ms, but the upload is a staging submit+fence and six
blocks are six round trips.

## J.2 The kernel

`moe_balance_grad_f32` computes the same formula on the device, reading the logits
buffer the forward already produced:

    p[n,j]   = softmax(logits[n])[j]
    dl[n,j]  = scale * p[n,j] * (f_j - sum_e f_e p[n,e]),  scale = (E/N)*aux_weight

One lane per row (N/64 workgroups), the seven loads as scalar push constants (the
target driver only indexes push-constant arrays by dynamically uniform values --
the same convention as the bias kernel).
`MoE7FFN::prepare_balance_gradient_on_device` allocates or reuses the [N,7]
gradient tensor and records the kernel in the optimizer batch, so there is no host
round trip and no extra submit. The host path stays as the non-Vulkan fallback and
as the A/B arm via `MOTIFCL_MOE_HOST_BALANCE_GRAD=1`.

## J.3 Measured

25 steps, seed 1337, three ABBA pairs, same executable:

| arm | tok/s | step | balance | GPU fwd+bwd |
|---|---:|---:|---:|---:|
| host gradient | 10302-10394 | 197.0-198.8 ms | 8.5-8.6 ms | 153.2-154.8 ms |
| device gradient | 10569-10597 | 193.3-193.8 ms | 1.9-2.0 ms | 155.2-156.4 ms |

+2.0% throughput with disjoint ranges. The balance phase loses 6.6 ms, the GPU side
gains 2.6 ms (six extra dispatches plus the kernel, now inside the optimizer
batch), so the step wins 3.5 ms net. Six-step losses and the final validation loss
are identical at printed precision (10.4277, 10.2058, 8.8452, 7.2389, 6.7947,
6.5197; FINAL VAL 6.4265 in both arms).

A new standalone witness (`tests/test_vulkan_moe_balance_grad.cpp`, links only the
Vulkan runtime) compares four fixtures against an independent CPU implementation
(max_abs 1.3e-12..3.3e-11) and requires invalid arguments (zero/overflow rows, NaN
scale or loads, aliased buffers, undersized buffers) to fail before dispatch and
without mutating the output. Its kernel-engagement assertion also caught a real
trap: `tools/gen_vulkan_spirv.py` and `tools/gen_vulkan_spirv_names.py` are two
separate generators, and a kernel missing from the name table still executes but
is invisible to the profiler.

## J.4 What is left in the host path

`read_token_counts` still costs ~1.7 ms per step: it downloads the [2048,7] biased
logits for six blocks and computes load/spread/pick statistics on the host. The
device already computes per-expert counts for the sparse path
(`moe_route_top1_f32`), so a counts-only variant plus a small read-back would
remove most of that. Beyond it, the structural lever remains host/device overlap.

# Phase K — Split forward/backward becomes the default path

## K.1 What the switch does

`MOTIFCL_MOE_SPLIT_FWD_BWD=1` submits the forward batch, reads the routing counts
back, and only then records the backward. That freeze is what lets the MoE
backward skip the losing experts: their gradient is identically zero, and
`tests/test_vulkan_moe7_zero_grad.cpp` already proves the dense and the skip
backward byte-identical (`router_grad`, `input_grad`, and every counter layer's
packed state, scale and v).

Phase D measured the combination at +16.7% and still kept it opt-in, for two
stated reasons: the split alone costs a forward-length GPU stall while the host
records the backward, and the skip only fires when routing is starved. Both were
true then. What changed is that the step's other blocks got much cheaper (Phase H
counter GEMM, Phase J device balance gradient) while the skip's saving is a fixed
fan-out of the backward, and the load read-out of the current configuration shows
exactly the starved routing the skip is for (per-step load spread of 0.02..0.25
over the seven experts). Same-executable A/B in that configuration:

| arm | tok/s | step | fwd+bwd GPU |
|---|---:|---:|---:|
| split off (old default) | 10538-10679 | 191.8-194.3 ms | 154.5-156.1 ms |
| split on, GPU balance on | 11199-11290 | 181.4-182.9 ms | 145.2-145.9 ms |

+5.8% with disjoint ranges over three ABBA pairs: the GPU forward+backward loses
~9.6 ms (the skipped losing-expert graphs) and the step loses 10.6 ms. The final
validation loss and every printed step loss are identical in both arms
(10.4277, 10.2058, 8.8452, 7.2389, 6.7947, 6.5197; FINAL VAL 6.4265).

Bit-level checkpoint equality is *not* usable as a witness here: the trainer is
not run-to-run bit-reproducible. Two identical invocations of the same binary,
same data order, same seed, produce different checkpoint bytes (verified with
`AUX_WEIGHT=0` and `BALANCE_GAMMA=0`, so the drift is in the base step and not in
the balance code). The drift is last-bit and leaves the printed losses identical.
The exactness argument for the skip therefore rests on its own witness
(`tests/test_vulkan_moe7_zero_grad.cpp` compares a dense and a skip backward byte
for byte) plus the printed-loss equality above, not on checkpoint hashes.

The GPU balance-bias update becomes the default alongside it: on its own it is
inside the A/A envelope (Phase 18), but with the split it removes the remaining
host bias round trip from the gap between the two batches (+1.9% over split
alone: 11015 -> 11225 tok/s in the same session). Both switches keep their `=0`
env overrides for A/B, and both are exact: the skip is proved byte-identical by
the zero-grad witness, the bias kernel by `test_vulkan_moe_balance_update`.

## K.2 What this means for the remaining levers

The trainer now runs split batches by default, so the gap between the two GPU
phases (host read-back plus host record of the backward) is the natural place for
host/device overlap: the host records the backward while the forward executes,
which is the largest structural item left (~19 ms of host time per step). That is
the next project, not a flag.

# Phase L — Why training was not reproducible, and the deterministic embedding backward

Phase K recorded that two identical trainer invocations produce different
checkpoints. This phase found the cause and removed it.

## L.1 The nondeterminism is the embedding gradient

Bisection by configuration size: with one layer and one step, two identical runs
differ in exactly three files — `emb.mt`, `adam_m_0.mt`, `adam_v_0.mt` — while the
head, the norms and all counter layers' state/scale/v are byte-identical. Param 0
is the embedding matrix, so the nondeterminism enters through the embedding weight
gradient. Setting `MOTIFCL_DISABLE_EMBEDDING_SCATTER=1` (the brute-force scan)
made two runs byte-identical, which named the culprit: on GCN4 / RX 580 there are
no float atomics, so the scatter accumulates with a float compare-and-swap loop
(`embedding_weight_backward_f32_i32_scatter_cas`). CAS retry order depends on
thread scheduling, and a batch of 2048 tokens over a 32000-token vocabulary
contains ~65 duplicate ids whose gradients are then summed in a scheduling-
dependent order. Adam amplifies the last-bit difference into the whole checkpoint
within a few steps.

## L.2 The fix

`embedding_backward_dup_count_f32_i32` + `embedding_backward_dup_sum_f32_i32`
replace the CAS accumulation with two passes whose order is fixed:

1. one workgroup per token; integer atomics decide only *ownership*
   (`atomicAdd(count[row], 1)`), so the result of this pass does not depend on
   ordering. The first occurrence of a row writes that row (single writer), and
   the first duplicate appends the row to a duplicate list.
2. one workgroup per duplicate list entry re-sums that row by walking the token
   ids in ascending order, overwriting the row.

Every output row therefore has one writer per phase and a fixed summation order.
Two earlier designs were rejected on measurement: `atomicMin`-based ownership (the
owner buffer stayed zero on this driver, which the witness caught as rows left at
zero) and a per-row id-scan kernel (correct and deterministic, but 23 ms per step
because the id loads are latency-bound).

`tests/test_vulkan_embedding_backward_det.cpp` covers it: four fixtures (unique
ids, duplicates, all-ids-equal, the 32000x288x2048 trainer shape), each compared
to an independent CPU implementation (max_abs 0.0 for all four), each run twice
with byte-identical output, plus invalid-argument controls.

## L.3 Cost and reproducibility

Same-executable A/B, 25 steps (the tok/s column is noisy in this window because of
host load; the GPU number is the stable one):

| arm | tok/s | fwd+bwd GPU |
|---|---:|---:|
| scatter (`MOTIFCL_EMBEDDING_NONDETERMINISTIC=1`) | 9262-10336 | 144.5-144.6 ms |
| deterministic (new default) | 9538-10399 | 144.7-145.6 ms |

The deterministic path costs ~0.4 ms of GPU per step (~0.3%), and two identical
6-step full-configuration runs then produce **430 byte-identical checkpoint files**
with the same FINAL VAL (6.4265). It is the default; the scatter stays available
via `MOTIFCL_EMBEDDING_NONDETERMINISTIC=1` for a pure-speed run. Checkpoint hashes
are now a usable exactness witness for A/B, which the Phase K measurement did not
have.

# Phase M — A tiny grid hid 2.5% of the step: RMSNorm grad_w

The fresh profile after Phase L put `rmsnorm_bwd_w_f32` at 2.5% of the step: 13
dispatches per step (two norms per block plus the final norm), 0.285 ms each,
reading 4.7 MB. That is ~16 GB/s, nowhere near the card's bandwidth, and the reason
is the launch shape: the kernel maps one workgroup per 64 columns, so a 288-wide
norm launches **5 workgroups** and each of them walks all 2048 rows.

The split form (`rmsnorm_bwd_w_partial_f32` + `rmsnorm_bwd_w_reduce_f32`, already in
the tree behind `MOTIFCL_VK_RMS_DW_SPLIT`) keeps the same reduction but spreads it
over `splits x ceil(cols/64)` workgroups; for the trainer that is 64 x 5 = 320
workgroups instead of 5. It was opt-in because it adds one dispatch (the partial
pass), so this phase measured it end to end.

25 steps, three ABBA pairs, same executable:

| arm | tok/s | fwd+bwd GPU |
|---|---:|---:|
| non-split (`MOTIFCL_VK_RMS_DW_SPLIT=0`) | 11219-11304 | 145.2-146.1 ms |
| split (new default) | 11420-11432 | 142.8-143.0 ms |

+1.5% throughput with disjoint ranges (GPU loses 2.6 ms of the 3.7 ms the kernel
was spending). The split path is now the default on training-shaped problems
(rows 256..65535, cols <= 1024); `MOTIFCL_VK_RMS_DW_SPLIT=0` restores the old
launch and a number pins the split count.

Verification: two identical 6-step full runs with the new default are still 430
byte-identical checkpoint files with FINAL VAL 6.4265, and the full CTest suite
covers the split parity (`test_vulkan_rmsnorm_split`).

# Phase N — Profile-driven micro-optimization audit (one win, two dead ends)

Fresh 25-step profile of the post-Phase-M defaults, as a share of GPU time:

| kernel group | share | per-step cost |
|---|---:|---:|
| `mm_f32_db64` (LM head fwd / dA / dB) | 24.5% | 36.5 ms |
| `counter_gemm_db` (this card has no tensor cores / no fp16 math) | 22.1% | 33.0 ms |
| `mm_f32_db32` (dense expert/shared GEMMs) | 17.1% | 25.5 ms |
| GQA (fwd causal4 4.6%, bwd_probs 8.0%, bwd_dkv 4.2%) | 16.8% | 25.0 ms |
| CE stats + cached backward | 4.2% | 6.3 ms |
| everything else (norms, swiglu, rope, adds, Adam) | 15.3% | 22.8 ms |

Three experiments came out of reading that table. Only one is kept.

## N.1 CE backward with vec4 (rejected)

`ce_bwd_cached_f32` is a flat elementwise pass over 2048x32000 logits (read 262
MB, write 262 MB) and measured 3.73 ms, i.e. ~140 GB/s. A vec4 variant (four
elements per invocation, guarded by `cols % 4 == 0`) was expected to save ~1 ms
of instruction issue. It measured **4.96 ms per call** instead, and the end-to-end
A/B was flat (GPU 142.58-143.39 new vs 142.73-142.97 baseline). The pass is
already bandwidth-bound; wider accesses only changed the access granularity.
Reverted, along with the test-side profile-name plumbing it needed.

## N.2 The [288, 7] narrow transpose-A matmul (rejected)

Twelve GEMMs per step are [in=288, experts=7] (six router weight gradients plus the
six auxiliary balance gradients) and go through the tiled `mm_f32_tn` kernel at
0.19 ms each — 2.3 ms/step for 4M MAC. A dedicated width-8 kernel (one lane per
output row, eight register accumulators) looked like an easy 10x. It measured
**~0.8 ms per call**, i.e. four times slower than the kernel it replaced, and cost
10 ms of GPU per step. The reason is the launch shape: `ceil(288/64) = 5`
workgroups means one wave per busy CU, and the inner loop has a dependent global
load per k, so 2048 iterations pay full memory latency with nothing to interleave.
The tiled kernel exists precisely because its LDS staging and multiple waves hide
that latency. Reverted; a real win here needs a k-split reduction (hundreds of
workgroups plus a partial buffer), which was not worth the complexity for 2.3 ms.

## N.3 GQA causal forward: eight query rows by default (kept)

`gqa_fwd_causal8_f32` keeps eight causal query rows per workgroup instead of four,
which halves the K/V re-reads; the existing witness already showed byte-identical
output against the four-row kernel across the trainer shapes. It is now the default
(`MOTIFCL_VK_GQA_FWD_ROWS=4` forces the old kernel), and the end-to-end evidence
is stronger than the micro-benchmark:

- two 6-step full runs, one per kernel, produce **430 byte-identical checkpoint
  files** and the same FINAL VAL (6.4265);
- 25-step A/B, three pairs: GPU forward+backward 142.16-143.01 -> 141.48-142.46 ms
  (~0.5%), tok/s within host noise (11385-11467 vs 11395-11475).

The lesson from the two rejections is that the remaining per-kernel gains on this
card are no longer about instruction count: the kernels that could still move are
either bandwidth-bound at ~200 GB/s (dense GEMMs, CE, elementwise) or limited by
occupancy/wave count (GQA, the tiled matmuls, the [288,7] shape).

# Phase O — The step-time accounting was wrong; wide GEMM tiles rejected

## O.1 The host idle is 6-9 ms, not 30

Every throughput comparison in Phases K-N used the trainer's window average, which
includes the first step. That step pays the driver's first-use cost for 115
pipeline objects and cost **~530 ms** in the same run that reported 177.5 ms/step;
with `LOG_EVERY=25` it alone inflates the mean by ~14 ms. A per-step
instrumentation run (temporary, `MOTIFCL_STEP_BREAKDOWN=1`) shows the real steady
state:

| | steady value |
|---|---:|
| step wall time | 150.7-153.8 ms |
| GPU forward+backward | 141.4-141.5 ms |
| GPU optimizer batch | 4.0 ms |
| GPU idle inside both submits | 0.3-0.4 ms each |
| host record (forward logits / CE / backward / optim) | 1.85 / 0.02 / 3.3-4.4 / 1.0 ms |

So the pipeline is ~94% GPU-busy, and the split-mode "forward-length GPU stall
while the host records the backward" that Phase D reasoned about is really the
~6 ms of host recording, not the 30 ms a warmup-polluted average suggested.
Consequence for the roadmap: host/device overlap (async submissions with deferred
buffer recycling, sketched after Phase L) is worth 2-4%, not the 8-10% estimated
from the window means, and it does not fit the 165 MB of free VRAM without a
double-buffered activation set. It stays a lower priority than GPU-side work.

The trainer now prints a warmup-excluded mean next to the window mean,
`steady <tok/s> <ms/step> (n, warmup)`; `MOTIFCL_WARMUP_STEPS` (default 1) sets the
skip. Same run: window mean 11512-11536 tok/s (177.5 ms) vs steady 13317-13586
tok/s (150.7-153.8 ms). The window mean is kept unchanged so older numbers stay
comparable.

## O.2 Wide 128x64 / 64x128 dense GEMM tiles (rejected)

The traffic model says the three LM-head GEMMs re-read their large operand:
head forward reads the 36.9 MB weight once per 64-row m tile, head dA re-reads
the 262 MB dlogits five times, head dB re-reads it once per m tile. Rectangular
tiles halve the m-tile or n-tile count, which the model scores at ~25% fewer bytes.

Implementation: a second body (`mm_f32_db_wide_body.glsl`) with flat tile staging,
because the original body's `AQ = 64/TILE_N`, `BQ = 64/TILE_M` formulas cap tiles at
64x64. Both 128x64 and 64x128 kernels were verified **bit-identical** to the 64x64
kernel on small shapes and on the three LM-head shapes (17 assertions, including
automatic selection and the `MOTIFCL_VK_MATMUL_DB_WIDE=0` kill switch), and a
6-step training run with and without them produced the same 430 checkpoint files.

It is still slower, on every shape (bench, batched, 15 reps):

| shape | 64x64 | 128x64 | 64x128 |
|---|---:|---:|---:|
| head_fwd 2048x288x32000 | 10.97 ms | 13.23 ms | 13.18 ms |
| head_dA 2048x32000x288 | 15.30 ms | 21.95 ms | 45.39 ms |
| head_dB 288x2048x32000 | 11.44 ms | 16.32 ms | 13.90 ms |

Trainer A/B, 25 steps: GPU forward+backward 141.4-142.5 ms with the wide tiles
disabled vs 151.2-151.4 ms with them on. The first version was much worse still
(head dA 46.6 ms) because the staged tile travelled through `inout vec4[N]`
parameters and the driver kept those arrays in scratch memory; single-register
staging fixed that but did not make the wide tiles competitive. The 64x64 kernel is
a local optimum on this card: 256 threads = four waves and 16 KiB of LDS = four
resident workgroups per CU buy more latency hiding than the traffic reduction is
worth. Reverted, including its test.

Also re-confirmed that the shipped tile heuristic is the right one: forcing
`MOTIFCL_VK_MATMUL_DB_TILE=64` costs GPU 145.9-146.3 ms and forcing `32` costs
164.9-165.5 ms, against 141.4-141.7 ms on auto.

# Phase P — Eight query rows in attention backward stage 1; what the counter path really is

## P.1 The backward stage-1 kernel, parameterized by rows (kept)

Phase M's forward counterpart (`gqa_fwd_causal8_f32`) showed that doubling the
number of causal query rows per workgroup halves the K/V re-reads. The backward
stage 1 (`gqa_bwd_probs_causal4_f32`) had the same structure hardcoded to four
rows, streaming K twice and V once per workgroup. It is now generated from one
body with a `ROWS` macro (`gqa_bwd_probs_causal_rows_f32`), so the eight-row form
shares the code instead of being another copy.

Because every per-element operation and every reduction order is unchanged, the
steered forms must agree bit for bit. They do:

- two 6-step full runs, one per kernel, produce **430 byte-identical checkpoint
  files** and the same FINAL VAL (6.4265);
- 25-step A/B, three pairs, using the new steady-state line: 153.9, 154.2, 154.0 ms
  with four rows vs 153.3, 153.3, 153.2 ms with eight rows (~0.7 ms, 0.5%), steady
  throughput 13281-13308 -> 13355-13367 tok/s.

Eight rows is the default now; `MOTIFCL_VK_GQA_BWD_ROWS=4` restores the four-row
kernel. `test_vulkan_gqa_training` accepts either stage-1 kernel name depending on
that variable.

## P.2 The counter path is dense over experts, and that is most of its cost

`counter_gemm_db` was the second largest block (22% of GPU time). A temporary
shape log (2 steps, not committed) shows the counter matmul is called ~840 times
per step with only three shapes, all `C=11` counters:

| shape (counters, in, out, batch) | calls/step |
|---|---:|
| 11, 288, 384, 2048 | 336 |
| 11, 192, 288, 2048 | 336 |
| 11, 288, 288, 2048 | 168 |

336 = 6 layers x 7 experts x 8, i.e. the MoE forward is *dense over experts*: each
layer projects the full 2048-row batch through every expert (plus the shared
expert), and masking/routing happens outside the GEMM. The kernel is then at
~25% of peak because each call is a 2.5G-MAC-on-paper shape that first has to
decode 11 counters per weight element (384x288x11 decodes) and only then stream
the GEMM. The per-10-step profile puts this at 345 ms of 1514 ms (22.8%), i.e.
34.5 ms per step.

That reframes the remaining large wins: they are architectural (dense-over-experts
cost, LM-head traffic), not kernel micro-tuning. The decode mirror path
(`counter_decode_dense_enabled`, `kDecodeDenseMinRows`) already exists in
`CounterStateLinear` for exactly this shape of problem and is a candidate for the
next phase.

# Phase Q — Decode-then-dense counter projections by default: +3.0%

Phase P named the counter path as the next lever and pointed at the existing
`MOTIFCL_COUNTER_DECODE_DENSE` mirror path, which had been A/B-only and off by
default because an earlier phase measured its host overhead (the naive Tensor-op
version) as large as its GPU win.

Measured on the current pipeline it is a clear win, and it is exact:

- two 6-step full runs, one with the fused decode-inside-GEMM kernels and one with
  decode-then-dense, produce **430 byte-identical checkpoint files** and the same
  FINAL VAL (6.4265). The decoded mirror is not an approximation: it is the same
  weights, and both GEMM paths sum k in ascending order.
- 25-step A/B, three pairs, steady-state line:
  fused 153.2 / 153.7 / 154.5 ms (13260-13368 tok/s) vs decode-dense
  **149.0 / 149.3 / 149.3 ms (13718-13748 tok/s)**, i.e. -4.4 ms per step (+3.0%).
- kernel profile, 10-step window: the counter block drops from 34.5 ms/step of
  `counter_gemm_db` to 2.9 ms of `counter_decode_weight_f32` plus the expert
  projections folded into `mm_f32_db32`/`mm_f32_db64` (total GPU time 1514 -> 1462
  ms per 10 steps).

`MOTIFCL_COUNTER_DECODE_DENSE=0` restores the fused path; the row-count gate
(`kDecodeDenseMinRows = 128`) still keeps one-row decode steps on the fused kernel.
`test_vulkan_counter_fused` now derives the expected kernel names from the same
rule. Full CTest stays green.

What is left inside this path is now launch structure rather than kernel work: the
dense-over-experts forward still issues one small GEMM per expert (48 projections
per step for the up matrix, 48 for the down matrix, plus the gradient directions)
where a single wide GEMM over the concatenated 8-expert weight matrix would cut the
launch count and give the GEMM a 3072-wide output. That needs MoE7FFN to consume a
concatenated expert output, so it is a separate phase.

# Phase R — Attention-backward width search, and a screen of every off-by-default gate

Two thin, cheap experiments after Phase Q came back negative; both are worth
recording because they close questions that would otherwise be re-opened.

## R.1 Stage-1 attention backward: one body, widths 4/8/16

The rows-parameterized kernel from Phase P was split into a shared body
(`gqa_bwd_probs_causal_body.glsl`) plus thin wrappers that only pick `ROWS`, and the
original hand-written four-row kernel was dropped in favour of the wrapper (all
widths are bit-identical: two 6-step runs, rows8 default vs forced rows4, produce
430 identical checkpoint files).

Width 16 was measured and rejected. 15-step steady means: rows8 150.5 ms (13605
tok/s), rows4 151.2 ms (13544), **rows16 160.3-160.6 ms (12754-12776)**. The wider
form needs 40 KiB of shared memory for the probability caches, which leaves one
workgroup per CU, and the sixteen live row accumulators spill; the halved K/V
re-reads do not pay for either.

## R.2 Screening every off-by-default gate

Phase Q's win came from flipping L. The pipeline has changed enough since the other
gates were added that several could be stale, so all four remaining off-by-default
switches were run through the trainer (15 steps, steady line):

| gate | steady ms | vs baseline |
|---|---:|---:|
| baseline | 150.6 | - |
| `MOTIFCL_VK_CE_STATS_SUBGROUP=1` | 151.3 | worse |
| `MOTIFCL_ENABLE_GQA_NO_TMP_BWD=1` | 150.8 | neutral |
| `MOTIFCL_ENABLE_GQA_PARTIAL_KV_BWD=1` | 150.6 | neutral |
| `MOTIFCL_VK_COUNTER_SWIGLU=1` | 154.9 | much worse |

None qualifies. The subgroup CE reduction also changes the reduction order, which is
exactly what the cached-vs-legacy bit-exactness witness pins, so it stays opt-in.
That closes the "maybe an old switch is now a win" hypothesis: the remaining large
items are the ones listed at the end of Phase Q (concatenated expert projections and
the LM-head traffic), not dormant flags.

# Phase S — Counter bookkeeping: a cheaper decode and a mirror cache

Phase Q moved the counter projections onto the decoded dense mirror, which left two
bookkeeping costs: `counter_decode_weight_f32` (226 rebuilds per step, 2.8 ms) and
`counter_update_apply_f32` (2.4 ms). Before optimizing the concatenation idea from
Phase Q, its traffic was re-modelled: stacking the seven expert up-matrices into one
2688-row GEMM leaves the byte counts unchanged (per-expert: 7 x (6 n-tiles x 2.36 MB
of A + 32 m-tiles x 0.44 MB of B); stacked: 42 n-tiles x 2.36 MB of A + 32 m-tiles x
3.1 MB of B = the same 198 MB), so it would only save launches, not work. That
cancelled the project before it was built.

The bookkeeping itself does have room:

- the decode kernel divided by (2C-1) four times per invocation and stored four
  scalars; the quotient only depends on a 6-bit code, so it is now tabulated once per
  workgroup in shared memory and the four outputs leave as one vec4: 12.4 -> 11.3 us
  per rebuild. What is left is the launch count, not the kernel;
- `CounterStateLinear` gained an **opt-in** decoded-mirror cache
  (`set_dense_mirror_cached`), invalidated by its own `apply_update_backward`, so a
  mirror is rebuilt once per update instead of once per projection: 226 -> 96
  rebuilds per step. The trainer enables it for its counter layers
  (`MOTIFCL_COUNTER_MIRROR_CACHE=0` disables it). It is deliberately not the default
  for library users: `state`/`scale` are public tensors and writing them directly
  would bypass the invalidation.

Both changes are exact: a 6-step full run with the new decode kernel and the cache
enabled produces the same 430 checkpoint files as before, and the cache A/B isolates
its effect: 148.8 ms/step with the cache off vs **147.6-147.9 ms** with it on
(13761-13762 -> 13843-13879 tok/s, +0.7%). Together with the kernel change this
round is 149.0 -> 147.7 ms per steady step.

# Phase T — RoPE by pair; three more elementwise/bookkeeping attempts rejected

## T.1 RoPE one invocation per pair (kept, bit-identical)

`rope_f32` ran one invocation per element and, for the rotary half, computed `pow`,
`cos` and `sin` for *both* halves of each even/odd pair and read the pair twice. The
pair kernel (`rope_pair_f32`) owns the pair: one angle, one `vec2` load, one `vec2`
store, same arithmetic. Measured 58.9 -> **32.3 us per call** (24 calls per step), and
it is bit-identical: the 6-step checkpoint set is unchanged (430 files). Steady step
147.6-147.9 -> **147.2-147.3 ms**. Partial-rotary shapes (tail elements copied
through) keep the per-element kernel, which the host selects.

## T.2 Rejected: memory placement knobs, staged counter-update row, vec4 Adam

- Memory placement, 15-step steady means against a 149.2 ms baseline:
  `MOTIFCL_VK_SMALL_HOST_BYTES=65536` 149.1, `=262144` 150.1,
  `MOTIFCL_VK_PREFER_DEVICE_LOCAL=1` 149.2, `MOTIFCL_VK_HOST_VISIBLE=1` 819.7
  (catastrophic). Nothing to gain; the 128 KiB host-visible rule is right.
- Counter update kernel: staging the packed row and its grad_w row in shared memory
  (one global pass instead of two) made it *slower*, 23.2 -> 27.3 us per layer,
  because the row is 216 bytes and the second pass was already cache-resident while
  the extra shared traffic, barrier and registers cost more. Reverted.
- Adam as a vec4 kernel: 132.7 -> 137.6 us per call. The update is already
  bandwidth-bound and perfectly coalesced, so four elements per invocation only add
  per-thread work. Reverted (the scalar kernel carries a comment recording this).

At this point the elementwise family is at 139-190 GB/s and the remaining kernels are
either bandwidth-bound or limited by the tiny output shape of the [.,7] router GEMMs
(193 us per call at 5 workgroups of parallelism, where the only fix is a k-split that
would change the summation order).

# Phase U — dK/dV staged through shared memory: 2.3x on that kernel

The stage-2 attention backward (`gqa_bwd_dkv_causal_tile_f32`) keeps one lane per key
and walks the whole row history with a direct global load of the q (or dout) slice for
every row: twelve-to-sixteen broadcast vec4 loads whose latency its accumulator FMAs
cannot hide, and a runtime `if (i < dmax)` guard in the innermost loop. It measured
1028 us per layer, ~14% of peak.

The new body (`gqa_bwd_dkv_causal_tile_staged_body.glsl` plus `hd48`/`hd64` wrappers)
stages the row history in chunks of 16 into shared memory: every lane copies a fixed
number of vec4 per chunk, so the loads are coalesced across the workgroup, and the
chunk is double-buffered so the next one loads while the current one is reduced.
`DMAX = head_dim / 4` becomes a compile-time constant, which removes the inner guard.
Rows still ascend through the same accumulators with the same products, so the output
is bit-identical: a 6-step run produces the same 430 checkpoint files as before.

Kernel time 1028 -> **442.7 us per layer** (2.3x), i.e. 6.2 -> 2.7 ms per step; the
10-step kernel window drops from 1445 to 1411 ms (3.4 ms per step) and the trainer's
steady step from 147.2-147.3 to **143.9-144.0 ms** (13908 -> 14221-14232 tok/s). The
trainer's head width is 48 (288 / 6 heads), so the hd48 wrapper is the one in use;
hd64 covers the other common width and any other width still takes the generic tile
kernel.

# Phase V — Stage-1 attention backward reads K/V rows as vec4: another 21%

The stage-1 backward (`gqa_bwd_probs_causal_rows8_f32`, 11.9 ms/step, ~5% of peak) has
one lane per key and reads that key's whole K row, then V row, then K again for dQ.
The rows are contiguous runs of head_dim floats, but the body read them one scalar at a
time: 48 dependent 4-byte loads per key per pass, each consumed immediately by the
accumulator FMAs.

Because head_dim and every K/V row start are multiples of four, the row of a key is a
run of head_dim/4 aligned vec4. The eight-row body now takes a `KV_VEC4` variant
(`gqa_bwd_probs_causal_rows8_vec_f32`) that reads them that way and adds the four
components in ascending d order - same products, same order, so the result is
bit-identical (6-step run: the same 430 checkpoint files).

Kernel time 1983.8 -> **1574.8 us per layer** (-21%), i.e. 11.9 -> 9.4 ms per step; the
10-step window drops 1411 -> 1388 ms and the trainer's steady step 143.9-144.0 ->
**141.7-142.0 ms** (14221-14232 -> 14427-14449 tok/s). The scalar body stays for
head widths that are not a multiple of four (the test's D7 case), and the host chooses
between them from head_dim; test_vulkan_gqa_training derives the expected name.

What is left in this kernel is latency, not bandwidth: after the vec4 change the
remaining gap to the traffic bound is ~2x, and closing it needs the K/V tiles staged
through shared memory (a 64-key tile is 12 KiB at head_dim 48, which together with the
existing 20 KiB of probability/row caches is exactly the 32 KiB budget - the experiment
belongs to its own phase because it trades occupancy for latency).

# Phase W — The same vec4 K-row read on the causal forward: -16%

`gqa_fwd_causal8_f32` had the same shape of problem as the stage-1 backward: one lane per
key, and the K row for that key read one float at a time (48 dependent 4-byte loads per
key) with the eight accumulator chains unable to hide the latency.

The new `gqa_fwd_causal8_vec_f32` declares the K buffer as vec4 and adds the four
components in ascending d order, so every row is byte-identical to the scalar body
(6-step run: the same 430 checkpoint files). Non-multiple-of-four head widths keep the
scalar kernel; the host chooses from head_dim, and
`test_vulkan_gqa_causal8_fwd` now matches the rows-8 family by its common prefix so it
accepts either variant while still comparing its output bitwise against the four-row
kernel.

1059 -> **888.7 us per layer** (-16%), i.e. 6.3 -> 5.4 ms per step; the 10-step kernel
window 1388 -> 1371.9 ms and the trainer's steady step 141.7-142.0 ->
**140.9-141.1 ms** (14427-14449 -> 14519-14534 tok/s).

# Phase X — Two shared-memory staging attempts that lost, and a GEMM shape census

Phase V left the stage-1 backward at 1575 us per layer, about twice its traffic bound,
and the obvious next step was to stage K/V through shared memory the way Phase U did for
dK/dV. That was built and measured, and it lost.

## X.1 Staging the K/V history in stage 1: rejected

A dedicated hd48 body staged each 64-key K/V tile into 12 KiB of shared memory (the
existing probability/row caches are 20 KiB, so the workgroup comes to 31 KiB) and had
every lane reduce its key's dot from the tile, in the same key assignment and the same
d component order as the inline body. The output stayed bit-identical, but the kernel
went from **1575 to 2460.7 us per layer**. Two effects: 31 KiB of shared memory admits
two workgroups per CU instead of three, so there are fewer waves to cover the tile
barriers, and the causal shape makes the tiles lopsided - a workgroup near the start of
the sequence stages a tile with only eight active keys, so most of the staging is paid
for nothing. Reverted; the inline vec4 body is the better trade on this card.

## X.2 Larger staging chunks in the dK/dV kernel: rejected

Phase U's staged dK/dV kernel stages the row history 16 rows at a time, double-buffered
(3 KiB per buffer at head_dim 48). Doubling to 32 rows (6 KiB per buffer, six loads per
lane instead of three) halved the barrier count and doubled the loads in flight, but
measured **441 -> 842 us per layer**: the bigger chunks and the doubled staging buffers
cost more than the barriers saved. CHUNK 16 stands.

## X.3 GEMM shape census (temporary instrumentation, not committed)

A shape log on the generic nn GEMM path (two steps) shows 84 calls per step over five
distinct shapes:

| shape (m, k, n) | calls/step | what it is |
|---|---:|---|
| 2048, 288, 192 | 48 | expert down projection |
| 2048, 384, 288 | 48 | expert up projection |
| 2048, 288, 7 | 42 | router logits |
| 2048, 288, 288 | 24 | attention projections |
| 2048, 288, 32000 | 3.5 | LM-head forward |

The router projection is the forward twin of the [288, 7] transposed gradient GEMMs
noted in Phase R: a 2048-row batch against a seven-column output, which the generic
kernel runs at a few workgroups of parallelism. Both are candidates for one batched
narrow-shape kernel rather than four hundred small dispatches per step, but that work
changes summation order and so is a separate decision.

*Correction (Phase Y).* Those counts came from runs that also executed the final
validation forward, so the per-step numbers are inflated - the shape set is right, the
call counts are not. The per-kernel profile remains the authority for time.

# Phase Y — The cached cross-entropy backward is at its bandwidth ceiling

`ce_bwd_cached_f32` (4.6 ms/step, 3.3%) reads logits and writes grad_logits, one element
per invocation, at an effective 115 GB/s. It looked like the vectorisation wins of
Phases V and W: four columns per invocation moves both sides as vec4 and loads the
per-row statistics and target once per four elements instead of once per element.

It does not help. Measured 4562 us (scalar) vs **4673 us** (vec4, 128 threads per
workgroup) and **4586 us** (vec4, 256 threads) - all inside the same band. The kernel is
not instruction-bound: the read/write mix plus the per-row statistics indirection caps it
at ~115 GB/s and neither wider accesses nor larger workgroups move that. Reverted, like
the earlier attempt recorded in Phase M.

What this leaves: the LM head's 37 ms (bandwidth-bound at 163 GB/s with wide tiles already
rejected) and the CE pair (7.2 ms together) can only be improved by *removing* traffic -
fusing the CE backward into the head gradient GEMMs so the 262 MB dlogits buffer is never
materialised - not by making the existing kernels faster. That fusion needs the CE
backward node and the head matmul backward to become one node, so it is a graph-level
change rather than a kernel one.

# Phase Z — CE backward folded into the head GEMMs: exact, and it frees 262 MB

Phase Y's conclusion was that the head plus cross-entropy can only get cheaper by not
materialising dlogits, which is a graph change. This phase made that change.

Two 64x64 double-buffered GEMM wrappers share the existing body: `mm_f32_db64_ce_a`
folds the CE backward into the A staging read (the head's input gradient) and
`mm_f32_db64_ce_b` into the B staging read (the head's weight gradient). Their staging
read applies, per element of the operand that used to be dlogits,

    dlogits = grad_scale * (exp(logits - max[row]) * inv_sum[row] - (col == target[row]))

with `grad_scale` read on the device from the loss gradient, and the per-row statistics
taken from the same `ce_fwd_stats` output the cached CE backward used. Every value and
operation is the one the standalone cached kernel performs, in the same order, so the
products - and therefore both gradients - are bit-identical to multiplying a
materialised dlogits.

The graph change: a new op `head_ce_fused(xf, weight, targets)` (weight in the nn
`[in, classes]` layout) produces the same scalar loss as
`softmax_cross_entropy(matmul(xf, weight), targets)`. It computes the logits and the CE
statistics once (outside the graph, under a no-grad guard) and installs one node whose
backward runs the two fused GEMMs and hands the gradients to `xf` and `weight`; the
dlogits tensor never exists. The trainer uses it by default for the untied head,
`MOTIFCL_HEAD_CE_FUSED=0` restores the separate CE node.

Verification:

- `tests/test_head_ce_fused.cpp` (new, in CTest): the fused loss, input gradient and
  weight gradient are bit-identical to the unfused path and match a double-precision CPU
  reference within 2e-5;
- a 6-step full training run produces the same 430 checkpoint files as before.

Cost and benefit:

| | before | after |
|---|---:|---:|
| head input gradient | 15.3 ms | 17.09 ms (fused) |
| head weight gradient | 11.4 ms | 13.56 ms (fused) |
| cached CE backward | 4.56 ms | gone |
| VRAM peak / used | 2400 / 2233 MB | **2144 / 1983 MB** |

The fused GEMMs pay the recomputation of `exp` five times for the input gradient (its A
operand is re-read once per n tile) and five times for the weight gradient, which eats
most of what removing the CE kernel saves: measured 127.6-128.1 ms of GPU per step
against 128.1-128.4 ms, i.e. about 0.4 ms (~0.3%). The real prize is memory: the
262 MB dlogits buffer is gone, so the trainer runs with 250 MB of headroom where it had
170 MB. That headroom is what a double-buffered activation set for host/device overlap
would need, which is the remaining known 2-4%.

# Phase AA — Where the last ~9 ms of every step goes, and why hiding it needs a pipeline

With the head fused (Phase Z) the steady step is 140.8-141.4 ms while the GPU executes
127.6-128.1 ms of forward+backward and 4.02 ms of optimizer, i.e. 9.8 ms of the step is
not GPU time. The trainer's own accounting puts it there:

| phase | ms/step |
|---|---:|
| upload | 0.004 |
| forward recording | 0.7 |
| forward+backward recording | 6.6 |
| backward recording (split mode only) | 4.4 |
| submit wait | 5.1 |
| balance / optimizer | 0.3 |
| device sync (`loss.item()`) | 0.02 |

The structure of a step explains it: the host records the whole forward+backward graph
(~7.3 ms) with the GPU idle because nothing has been submitted yet; the batch is
submitted, the host waits for it, reads the loss and the routing counts back, records
the optimizer batch, submits it, and only then starts recording the next step. The
optimizer's 4 ms of GPU time does overlap the start of the next recording, which is why
the gap is ~9.8 ms rather than ~13 ms.

Two experiments to move that gap without restructuring, both neutral within noise over
15-step windows: `MOTIFCL_MOE_SPLIT_FWD_BWD=1` (142.8 ms against 142.7-143.0 ms - it
trades the forward+backward recording for a forward-length stall, and the zero-gradient
expert skip pays for it) and `MOTIFCL_MOE_GPU_BALANCE=1` (143.2 ms), alone and together.

The only way to hide the recording is a two-batch pipeline: submit step N's
forward+backward, and while the GPU runs it, record step N+1's forward+backward into a
second in-flight batch, then submit it behind step N's optimizer batch. Queue order
already enforces the data dependency (step N+1's forward reads the parameters the
optimizer writes), so the pipeline needs no extra synchronisation - but it does need two
activation generations alive at once, which before Phase Z did not fit in VRAM and now
does (250 MB of headroom). It also needs the buffer pool to keep a submitted-but-running
batch's buffers out of the free list, which is the part to build and verify first; the
acceptance test is the same as every other change here - the 6-step run must keep
producing the same 430 checkpoint files, since a pipeline changes no arithmetic.

Expected payoff if it works: up to 7 ms of the 9.8 ms (~5%), taking the step from ~141 ms
to ~134 ms.

# Phase AB — Two more kernels that are already at the memory system's limit

Two costs looked like the vec4 wins of Phases V/W and are not:

- `add_f32` (3.31 ms/step over 89 calls, the MoE expert accumulation) moves two operands
  and one result of 2.36 MB each in 37.24 us, i.e. **190 GB/s** - the same rate the
  best-behaved elementwise kernels reach on this card. A four-elements-per-invocation
  form measured 37.24 us against 37.2 us, unchanged: the kernel is at the memory system's
  limit, not instruction-bound. Not merged.
- `ce_fwd_stats_f32` (2.54 ms/step) reads the 262 MB logits twice (one pass for the row
  max, one for the exp sum, both required for a bit-exact result) in 2.54 ms, i.e.
  **206 GB/s**. Also closed.

# Dispatch inventory of one training step (~990 dispatches)

Recording 990 dispatches costs 7.3 ms of host time (~7.4 us each), so the count matters
as much as the GPU time. Per step:

| kernel | calls/step | GPU ms/step |
|---|---:|---:|
| `mm_f32_db32` | 284 | 46.0 |
| `counter_decode_weight_f32` | 132 | 1.5 |
| `counter_update_apply_f32` | 106 | 2.5 |
| `add_f32` | 89 | 3.3 |
| `mm_f32_db64` | 49 | 17.4 |
| `swiglu_f32` | 48 | 1.4 |
| `swiglu_bwd_f32` | 41 | 1.9 |
| `fog_hard_route7_candidate_bwd_f32` | 35 | 0.8 |
| `rope_pair_f32` | 24 | 0.7 |
| `adam_update_fast_f32` | 21 | 2.8 |
| rest (rmsnorm, counter decay, zero, ce, gqa, ...) | ~160 | ~13 |

The counter subsystem alone is **238 dispatches (24% of the step's dispatches) for 4.0 ms
of GPU**; its state is one 216-byte row plus a 1.2 KiB weight slice per layer, so every
one of those launches is latency, not throughput. Packing a block's counter layers into one
arena would let each block update and decode with a single dispatch (238 -> ~24), which is
the concrete next target after the pipeline: worth roughly 1.8 ms of recording and 0.5-1 ms
of GPU launch overhead.

# Phase AC — Deferred submit built; the pipeline it was for cannot pay

Phase AA proposed a two-batch pipeline to hide the ~7 ms of host recording. The runtime now
has the machinery: `batch_end_async()` submits a batch and returns without waiting, and
`batch_wait()` waits for every pending submission. Two submission slots carry their own
command buffer, fence, descriptor sets and kept-alive buffers, so a command buffer that the
GPU still owns is never reset or recycled, and a descriptor set or buffer referenced by an
in-flight submission is never handed out again. The synchronous `batch_end()` is unchanged
(async + wait), and with the whole trainer still on it the refactor is invisible: the 6-step
run produces the same 430 checkpoint files, and CTest is 40/30/0.

Then the pipeline was measured out of existence. A step's host work is only ~7 ms of which
the forward recording is 0.7 ms; the backward recording (4.4 ms) is what is worth hiding -
and the backward cannot be recorded until the routing counts are read back, because the
zero-gradient expert skip decides *which* kernels to record. That skip is not optional:

| configuration | steady step |
|---|---:|
| default (split, zero-grad skip) | **143.7 ms** |
| split, skip off | 152.8 ms |
| one batch | 153.7 ms |
| one batch, skip off | 155.6 ms |

The skip is worth **9-10 ms per step**, twice what the pipeline could ever hide, so any
scheme that records the backward before the routing readback is a large regression. The
one-batch mode is 10 ms worse for the same reason (it records the backward with no routing
information at all, so nothing is skipped).

Conclusion: the remaining ~7 ms of host work per step is not reachable by pipelining, and a
step of ~141 ms against ~132 ms of GPU work is close to this design's floor. The remaining
honest target is the counter subsystem's 238 dispatches (Phase AB) and the ~9.4 ms stage-1
attention backward.

# Phase AD — GEMM inventory per step, and the narrow-GEMM dead end

A shape log on the double-buffered GEMM path (capped at one step's worth of dispatches)
resolves what the 284 `mm_f32_db32` and 49 `mm_f32_db64` calls per step actually are:

| tile | m, k, n | layouts | calls/step | GMAC/step | what |
|---|---|---|---:|---:|---|
| db64 | 2048, 288, 384 | 0, 1 | 48 | 10.9 | expert/up forward (8 per layer) |
| db32 | 2048, 192, 288 | 0, 1 | 48 | 5.4 | expert/down forward |
| db32 | 2048, 288, 192 | 0, 0 | 40 | 4.5 | shared-expert forward |
| db32 | 288, 2048, 192 | 1, 0 | 40 | 4.5 | shared-expert weight gradient |
| db32 | 2048, 384, 288 | 0, 0 | 40 | 9.1 | expert/up input gradient |
| db32 | 384, 2048, 288 | 1, 0 | 40 | 9.1 | expert/up weight gradient |
| db32 | 2048, 288, 288 | 0, 1 | 24 | 4.1 | attention projections forward |
| db32 | 2048, 288, 288 | 0, 0 | 20 | 3.4 | attention input gradient |
| db32 | 288, 2048, 288 | 1, 0 | 19 | 3.2 | attention weight gradient |
| db64 | 2048, 288, 32000 | 0, 0 | 1 | 18.9 | head forward |

So ~73 GMAC per step of double-buffered GEMM (plus 37.7 GMAC in the two frozen-CE head
gradients from Phase Z), every shape at least 64 columns or rows wide, and all of them in a
tile that the heuristic already picked. A forced-tile screen inside one session confirms the
heuristic: auto 159.0/159.6 ms, forced 64 164.9 ms, forced 32 165.8 ms.

The narrow-output GEMM did not survive measurement. The router and balance gradients
(`m=288, k=2048, n=7`) run at 193 us per call in the generic kernel, which pads n to 16
columns, so one lane per output row looked like a tenfold win: 288 lanes, each walking k once
with eight accumulators, reading A coalesced and B broadcast, accumulating in strict
ascending k so the result would be bit-identical (it was - the 6-step run reproduced the same
430 checkpoint files). It measured **1718 us per call**, nine times slower: 288 lanes are five
workgroups on a 36-CU card, and each lane's 2048-step walk with four loads in flight is
latency the sixteen-column tile hides by keeping nineteen workgroups resident and its B tile
broadcast out of shared memory. Reverted. This is the second narrow-GEMM attempt to lose
(Phase R's was the first); on this card a narrow output is better served by padding the tile
than by trading tiles for lanes.

One process note from the same session: the identical binary measured 143.7 ms per step in
the morning and 159.2 ms an hour later with the tree unchanged, so the card throttles under
sustained load. Only same-session A/B pairs are evidence here; absolute tok/s figures are
per-session.

# Phase AE — The deferred-submit API was broken and is now witnessed

Phase AC added `batch_end_async()` / `batch_wait()` but nothing exercised them, and the new
`tests/test_vulkan_batch_async.cpp` immediately found that they never worked: the deferred
branch of `submit_fast_commands()` returned before setting `result.success`, so every
`batch_end_async()` reported failure with an empty error string. Fixed by setting the flag on
that path.

The test now covers what a wrong implementation breaks first:

- submission order: two deferred batches where the second consumes the first's output must
  match the synchronous chain (1e-5);
- buffer reuse: a same-sized buffer is allocated, downloaded and dropped between the two
  deferred submissions, so a pool that recycled it while the first batch was still running
  would corrupt the second result;
- slot recycling: a plain synchronous batch after deferred ones still produces the right
  answer (the slots hand their fences, descriptor sets and kept-alive buffers back);
- the elementwise path through a deferred slot.

With the trainer unchanged (still on the synchronous `batch_end()`), the 6-step run keeps
producing the same 430 checkpoint files, and CTest is now 41 passed / 30 skipped / 0 failed
of 71.

# Phase AF — The fused head finally pays: statistics cached in shared memory

Phase Z's folded CE transform cost more than it should because every staged vec4 of the
input-gradient operand also loaded `(max, inv_sum)` and the target from global memory:
the A staging re-reads each element once per n tile, so a 2048-row, 32000-column operand
issued tens of millions of those loads per call for two arrays that are only
2048 x 12 bytes. The transform now copies the workgroup's row statistics and targets into
shared memory once (64 rows, 640 bytes) before the first staging read and indexes them
locally; the weight-gradient side keeps the global form, because its rows are the batch's
tokens - every workgroup walks all 2048 of them once, so there is nothing to cache and the
24 KiB table stays in L2.

The input-gradient GEMM drops 17.09 -> **15.99 ms per step** (-6.4%) with the transform's
arithmetic untouched, so the results stay bit-identical: the 6-step run reproduces the same
430 checkpoint files and `test_head_ce_fused` still reports both gradients bit-identical to
the unfused path. Same-session A/B (15 steps each, two pairs): fused head **141.6 / 142.1 ms**
against unfused **142.8 / 142.8 ms**, i.e. folding the CE backward now wins by ~1 ms per step
where Phase Z measured it neutral - the cache is what makes the fusion pay.

## AF.1 Packing the statistics into one vec4 per row: measured, rejected

The obvious follow-up was to pack `(max, inv_sum, target, 0)` into a single vec4 table (a
one-dispatch 2048-row kernel per step) so both transform sides load one value instead of a
statistics pair plus a target. Implemented and measured over two profile runs per side:

| kernel | Phase AF | packed table |
|---|---:|---:|
| input gradient (`ce_a`) | 15.99 ms | 16.27 / 16.62 ms |
| weight gradient (`ce_b`) | 13.34 ms | 13.06 / 13.42 ms |

The input gradient got slower: its rows already come from the shared-memory cache, so the
change only traded an `ivec` read for a float-to-int conversion per staged element. The
weight gradient gained nothing measurable: its row index is shared by sixteen lanes per
k tile, so the second (target) load was already a broadcast that costs nothing. Net
+0.2-0.4 ms, so the packed table was reverted and Phase AF's form stands.

# Phase AH — The counter update path is 19% of the step, and the profiler hides it

Every earlier budget in this report came from the kernel profiler, which attributes
2.5 ms/step to `counter_update_apply_f32` and ~1.5 ms to `counter_decode_weight_f32`. A
direct experiment disagrees: with the update path short-circuited (a temporary env knob
that returns from `apply_update_backward` before it dispatches anything, so training is
wrong but the rest of the step is untouched), the step drops **142.4 -> 115.6 ms**, and the
decode knob changes nothing.

    142.4 ms   baseline
    115.6 ms   update path skipped
    145.5 ms   decode skipped (noise: stale weights reroute the experts)
    114.2 ms   both skipped

So the counter update path costs **26.8 ms per step - 19% of the step**, twenty times what
the profiler reports. It dispatches, per counter layer, one transposed-A GEMM
(`m=384, k=2048, n=288` for the expert projections) plus one update kernel, and there are
96 counter layers per step, so the shaped work is 96 x 226 GMAC. At 64x64 tiles each of
those GEMMs moves ~30 MB (the weight-gradient operand is re-read once per n tile, the
activation once per m tile), i.e. 2.9 GB per step, which is ~15 ms at the card's 190 GB/s -
most of the 26.8 ms, with the rest being dispatch and barrier overhead across 192 tiny
launches that each only fill part of the card (30 workgroups at the 64 tile, 108 at the 32
tile).

Two things follow. First, the counter subsystem is by far the largest remaining target,
not the 4 ms the profile suggested. Second, its cost is traffic, not tiles: the 64x64 tile
already moves the least data, which is why forcing the 64x64 tile on these small
traffic-heavy shapes measured **worse** (146.4 against 143.5 ms) - fewer workgroups means
the memory system is fed less well. Cutting the traffic needs bigger output tiles, and a
bigger tile only pays if there are enough workgroups to keep the card busy, which for these
shapes means batching the seven identical expert-up weight gradients of a block into one
dispatch (grid.z = layer) - a change that needs the per-layer operands reachable from one
descriptor set, i.e. the arena the counter subsystem has so far avoided.
