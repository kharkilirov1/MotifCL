# Changelog

## Unreleased

### GEMM inventory per step, and the narrow-GEMM dead end (2026-10-24, no code change)

- Shape log of the double-buffered GEMM path resolves the 333 dispatches of a step: expert
  and shared-expert projections (up/down forward, input gradient, weight gradient) at 48/48/
  40/40/40/40 calls, attention projections 24 forward and 39 for the two gradients, and one
  head forward - ~73 GMAC in total, every shape already in the tile the heuristic picks. A
  forced-tile screen inside one session agrees: auto 159.0/159.6 ms, forced 64 164.9 ms,
  forced 32 165.8 ms.
- A one-lane-per-row kernel for the narrow router/balance gradients (`n=7`, 193 us per call
  in the generic 16-column-tile kernel) measured 1718 us per call and was reverted: 288 lanes
  are five workgroups, and the padded tile hides that latency better than lane-per-row does.
  Bit-identical while it lasted (same 430 checkpoint files). Second narrow-GEMM attempt to
  lose; recorded in `reports/vulkan-moe-sparse-2026-10-21.md`, Phase AD.
- Process note recorded there too: the same binary measured 143.7 ms and 159.2 ms an hour
  apart on an unchanged tree, so the card throttles and only same-session A/B pairs count.

### Deferred batch submission; the pipeline it enables cannot pay (2026-10-24)

- `VulkanRuntime::batch_end_async()` submits an open batch and returns without waiting,
  and `batch_wait()` waits for every pending submission. The runtime keeps two submission
  slots with their own command buffer, fence, descriptor sets and kept-alive buffers: a
  command buffer the GPU still owns is never reset, and a set or buffer referenced by an
  in-flight submission is never recycled early. `batch_end()` is unchanged (submit + wait);
  with the trainer still on it the 6-step run produces the same 430 checkpoint files and
  CTest stays 40 passed / 30 skipped / 0 failed of 70.
- The pipeline the API was built for is rejected on measurement: the backward cannot be
  recorded before the routing readback because the zero-gradient expert skip decides which
  kernels to record, and that skip is worth 9-10 ms per step (default 143.7 ms against
  152.8 ms with the skip off, 153.7 ms in one-batch mode, 155.6 ms one-batch without the
  skip) - twice what the pipeline could hide. Recorded in
  `reports/vulkan-moe-sparse-2026-10-21.md`, Phase AC, together with the conclusion that
  ~141 ms against ~132 ms of GPU work is close to this design's floor.

### Two kernels confirmed at the memory limit; dispatch inventory (2026-10-24, no code change)

- `add_f32` (3.31 ms/step, 89 calls, MoE expert accumulation) already runs at 190 GB/s; a
  four-elements-per-invocation form measured 37.24 us against 37.2 us. Not merged.
- `ce_fwd_stats_f32` (2.54 ms/step) reads the 262 MB logits twice - the row max and the
  exp sum need separate passes for a bit-exact result - at 206 GB/s. Closed.
- Dispatch inventory: ~990 dispatches per step, 7.3 ms of host recording. The counter
  subsystem is 238 of them (24%) for 4.0 ms of GPU although each updates a 216-byte row
  and a 1.2 KiB slice, i.e. pure launch latency; packing a block's counter layers into one
  arena would collapse it to ~24 dispatches (worth ~1.8 ms of recording and 0.5-1 ms of
  GPU), the next target after the pipeline.
  `reports/vulkan-moe-sparse-2026-10-21.md`, Phase AB.

### Step-gap accounting and pipeline design (2026-10-24, no code change)

- With the head fused, a steady step is 140.8-141.4 ms while the GPU runs 127.6-128.1 ms
  of forward+backward plus 4.02 ms of optimizer: 9.8 ms of the step is not GPU time, and
  the trainer's phase counters place it in the host recording of the forward+backward
  graph (7.3 ms, done with the GPU idle because nothing has been submitted yet) plus the
  submit wait (5.1 ms), with the optimizer's 4 ms of GPU time already overlapping the
  next recording.
- Two ways to close it without restructuring measured neutral over 15-step windows:
  `MOTIFCL_MOE_SPLIT_FWD_BWD=1` (142.8 ms vs 142.7-143.0 ms) and
  `MOTIFCL_MOE_GPU_BALANCE=1` (143.2 ms), separately and together.
- Recorded design for the next attempt: a two-batch pipeline that records step N+1's
  forward+backward while the GPU runs step N's batch and submits it behind step N's
  optimizer batch. Queue order already carries the parameter dependency, so the work is
  a second activation generation (now affordable: 250 MB of VRAM headroom after Phase Z)
  and a buffer pool that keeps an in-flight batch's buffers out of the free list.
  Acceptance test: the same 430 checkpoint files. Expected up to ~7 ms (~5%).
  `reports/vulkan-moe-sparse-2026-10-21.md`, Phase AA.

### Cross-entropy backward folded into the head GEMMs (2026-10-24)

- New `head_ce_fused(xf, weight, targets)` op (`include/motifcl/ops/loss.hpp`) returns the
  same loss as `softmax_cross_entropy(matmul(xf, weight), targets)` but its single
  autograd node computes both head gradients with `mm_f32_db64_ce_a` /
  `mm_f32_db64_ce_b`, which recover `dlogits = grad_scale * (softmax(logits) - onehot)`
  from the stats already produced by `ce_fwd_stats` while staging their operand. The
  `[rows, classes]` dlogits buffer is never materialised. The trainer uses it by default
  for the untied head; `MOTIFCL_HEAD_CE_FUSED=0` restores the separate CE node.
- Exact, not approximate: `tests/test_head_ce_fused.cpp` proves the loss and both
  gradients are bit-identical to the unfused path and match a double-precision CPU
  reference (2e-5), and a 6-step training run produces the same 430 checkpoint files.
- Cost: the fused input-gradient GEMM pays the exp recomputation once per n tile
  (17.09 ms against 15.3 + a share of the 4.56 ms CE kernel) and the weight-gradient one
  likewise (13.56 against 11.4), so the GPU saving is ~0.4 ms/step (127.6-128.1 against
  128.1-128.4 ms of GPU). The real gain is memory: VRAM peak 2400 -> 2144 MB and used
  2233 -> 1983 MB, headroom a double-buffered activation set would need.
  `reports/vulkan-moe-sparse-2026-10-21.md`, Phase Z.

### Cached cross-entropy backward: vec4 form measured, no gain (2026-10-24, no code change)

- A four-columns-per-invocation form of `ce_bwd_cached_f32` (vec4 logits and gradient,
  per-row statistics and target loaded once per four elements, element-identical
  arithmetic) measured 4673 us per call against 4562 us scalar, and 4586 us with 256
  threads per workgroup - the kernel is not instruction-bound, the read/write mix plus
  the per-row statistics indirection caps it at ~115 GB/s. Reverted.
- Consequence recorded: the remaining large items (LM head 37 ms, CE pair 7.2 ms) can only
  improve by removing traffic - fusing the CE backward into the head gradient GEMMs so the
  262 MB dlogits buffer is never materialised - which is a graph-level change.
- The Phase X GEMM shape census is corrected: its counts came from runs that also executed
  the final validation forward, so per-step figures are inflated (the shape set is right).
  `reports/vulkan-moe-sparse-2026-10-21.md`, Phase Y.

### Two staging attempts rejected; GEMM shape census (2026-10-24, no code change)

- Staging the stage-1 attention backward K/V history through shared memory (12 KiB of
  tiles on top of the 20 KiB of caches, 31 KiB per workgroup) is bit-identical but
  slower: 1575 -> 2460.7 us per layer, because 31 KiB admits two workgroups per CU
  instead of three and the causal tiles are lopsided (a workgroup near the sequence start
  stages a tile with eight active keys). Reverted.
- Doubling the staged dK/dV kernel's chunk from 16 to 32 rows: 441 -> 842 us per layer.
  Reverted; CHUNK 16 stands.
- Shape census of the generic nn GEMM path (temporary instrumentation): 84 calls per step
  over five shapes, including the [2048, 288, 7] router projection (42 calls) - the
  forward twin of the [288, 7] transposed gradient GEMMs from Phase R. A batched
  narrow-shape kernel is the only way to help those, and it changes summation order.
  `reports/vulkan-moe-sparse-2026-10-21.md`, Phase X.

### Causal attention forward: K rows read as vec4, -16% on that kernel (2026-10-24)

- `gqa_fwd_causal8_vec_f32` declares the K buffer as vec4 and adds the four components in
  ascending d order, so every row is byte-identical to the scalar body (6-step run: the
  same 430 checkpoint files). Non-multiple-of-four head widths keep the scalar kernel;
  the host chooses from head_dim.
- 1059 -> 888.7 us per layer (-16%), 6.3 -> 5.4 ms per step; the 10-step kernel window
  1388 -> 1371.9 ms and the trainer's steady step 141.7-142.0 -> 140.9-141.1 ms
  (14427-14449 -> 14519-14534 tok/s).
- `test_vulkan_gqa_causal8_fwd` matches the rows-8 family by its common prefix, so it
  accepts either variant while still comparing bitwise against the four-row kernel.
  `reports/vulkan-moe-sparse-2026-10-21.md`, Phase W.

### Attention backward stage 1: K/V rows read as vec4, -21% on that kernel (2026-10-24)

- `gqa_bwd_probs_causal_rows8_vec_f32` reads each key's K and V row as aligned vec4
  (head_dim and every row start are multiples of four) and adds the four components in
  ascending d order, so the result is bit-identical to the scalar body (6-step run: the
  same 430 checkpoint files). Non-multiple-of-four head widths keep the scalar kernel.
- 1983.8 -> 1574.8 us per layer (-21%), 11.9 -> 9.4 ms per step; the 10-step kernel
  window 1411 -> 1388 ms and the trainer's steady step 143.9-144.0 -> 141.7-142.0 ms
  (14221-14232 -> 14427-14449 tok/s).
- `test_vulkan_gqa_training` derives the expected stage-1 kernel from head_dim.
  `reports/vulkan-moe-sparse-2026-10-21.md`, Phase V.

### Attention backward dK/dV: row history staged through shared memory, 2.3x (2026-10-24)

- New `gqa_bwd_dkv_causal_tile_staged_body.glsl` with `hd48`/`hd64` wrappers. The old
  tile kernel loaded the q (or dout) slice for every row straight from global memory -
  twelve to sixteen broadcast vec4 loads per row whose latency the accumulator FMAs
  could not hide - and guarded the innermost loop on a runtime `dmax`. The staged
  version copies rows in chunks of 16 into shared memory (coalesced, double-buffered)
  and makes `DMAX = head_dim / 4` a compile-time constant.
- 1028 -> 442.7 us per layer (2.3x), 6.2 -> 2.7 ms per step; the 10-step kernel window
  drops 1445 -> 1411 ms and the steady step 147.2-147.3 -> 143.9-144.0 ms
  (13908 -> 14221-14232 tok/s). Bit-identical: the 6-step checkpoint set is unchanged
  (430 files). The trainer uses the hd48 wrapper (288 / 6 heads); other widths keep the
  generic tile kernel.
- `reports/vulkan-moe-sparse-2026-10-21.md`, Phase U.

### RoPE: one invocation per rotated pair (2026-10-24)

- `rope_pair_f32` computes the angle (and its `pow`) once per even/odd pair instead of
  once per element for each half, and moves the pair as one `vec2` in and one `vec2`
  out. 58.9 -> 32.3 us per call, bit-identical (6-step run: the same 430 checkpoint
  files), steady step 147.6-147.9 -> 147.2-147.3 ms. Partial-rotary layouts keep the
  per-element kernel.
- Rejected on measurement and reverted: staging the counter-update row in shared
  memory (23.2 -> 27.3 us per layer; the row is 216 bytes and already cache-resident),
  a vec4 Adam kernel (132.7 -> 137.6 us; already bandwidth-bound and coalesced), and
  the memory-placement knobs (`SMALL_HOST_BYTES` 64K/256K, `PREFER_DEVICE_LOCAL`,
  `HOST_VISIBLE` = 819.7 ms/step). `reports/vulkan-moe-sparse-2026-10-21.md`, Phase T.

### Counter bookkeeping: cheaper decode kernel and an opt-in mirror cache (2026-10-24)

- `counter_decode_weight_f32` tabulates `code/(2C-1) - 1` once per workgroup in
  shared memory (the quotient only depends on a 6-bit code) and writes the four
  decoded weights as one vec4: 12.4 -> 11.3 us per rebuild, same values.
- `CounterStateLinear::set_dense_mirror_cached(true)` keeps the decoded dense mirror
  across calls and invalidates it in `apply_update_backward`, so the trainer rebuilds
  a mirror once per update instead of once per projection (226 -> 96 rebuilds per
  step). The trainer enables it (`MOTIFCL_COUNTER_MIRROR_CACHE=0` disables); it stays
  opt-in for library users because `state`/`scale` are public tensors.
- Exact: a 6-step full run with both changes produces the same 430 checkpoint files
  as before. Cache A/B: 148.8 ms/step off vs 147.6-147.9 ms on (+0.7%); with the
  kernel change the round is 149.0 -> 147.7 ms per steady step.
- Also re-modelled the concatenated expert up-projection from Phase Q and cancelled
  it: stacking the seven up-matrices leaves the byte counts identical, so it would
  only save launches. `reports/vulkan-moe-sparse-2026-10-21.md`, Phase S.

### Attention backward: one shared body for the stage-1 widths (2026-10-24)

- `gqa_bwd_probs_causal_body.glsl` now holds the stage-1 kernel and thin wrappers
  pick `ROWS` (4 or 8, default 8; `MOTIFCL_VK_GQA_BWD_ROWS`). The original
  hand-written four-row file is gone; both widths are bit-identical (two 6-step
  runs, 430 identical checkpoint files).
- Width 16 measured and rejected: rows8 150.5 ms, rows4 151.2 ms, rows16
  160.3-160.6 ms per step - 40 KiB of shared memory leaves one workgroup per CU and
  the sixteen row accumulators spill.
- Screened every remaining off-by-default gate on the current pipeline (15-step
  steady means): CE subgroup stats 151.3 ms against a 150.6 baseline, GQA no-temp
  backward 150.8, GQA partial-K/V backward 150.6, counter-fused SwiGLU 154.9 - none
  is a win, so the "stale flag" hypothesis is closed.
  `reports/vulkan-moe-sparse-2026-10-21.md`, Phase R.

### Counter projections: decode-then-dense by default, +3.0% (2026-10-24)

- `MOTIFCL_COUNTER_DECODE_DENSE` is now the default (set it to `0` to restore the
  fused decode-inside-GEMM kernels). The mirror path decodes the 11-counter packed
  weights into a dense FP32 mirror and runs the tuned GEMM; on the trainer shapes
  the fused kernel was only ~25% of peak because it re-decodes every weight element
  on every call.
- Bit-identical, not approximate: two 6-step full runs, one per path, produce 430
  byte-identical checkpoint files with the same FINAL VAL (6.4265).
- 25-step A/B, three pairs, steady-state line: fused 153.2/153.7/154.5 ms
  (13260-13368 tok/s) vs decode-dense 149.0/149.3/149.3 ms (13718-13748 tok/s),
  i.e. -4.4 ms per step (+3.0%). Kernel profile: `counter_gemm_db` 34.5 ms/step
  collapses to 2.9 ms of `counter_decode_weight_f32` plus the expert projections in
  the dense GEMM family.
- `test_vulkan_counter_fused` derives the expected kernel names from the same rule.
  `reports/vulkan-moe-sparse-2026-10-21.md`, Phase Q.

### Attention backward: eight query rows by default (2026-10-24)

- The causal attention backward stage 1 is now generated from a rows-parameterized
  body (gqa_bwd_probs_causal_rows_f32, shared by ROWS=4 and ROWS=8) instead of a
  four-row copy. Eight rows halve the workgroup count and therefore the K/V
  re-reads (the kernel streams K twice and V once per workgroup).
  MOTIFCL_VK_GQA_BWD_ROWS=4 restores the four-row kernel.
- Verified bit-identical, not merely close: two 6-step full runs, one per kernel,
  produce 430 byte-identical checkpoint files with the same FINAL VAL (6.4265).
- 25-step A/B, three pairs, steady-state metric: 153.9/154.2/154.0 ms with four
  rows vs 153.3/153.3/153.2 ms with eight rows (~0.5%, steady 13281-13308 ->
  13355-13367 tok/s).
- 
eports/vulkan-moe-sparse-2026-10-21.md (Phase P) also characterizes the
  second largest GPU block: counter_gemm_db is called ~840 times per step for
  only three shapes, because the MoE forward is dense over experts (6 layers x 7
  experts x 8 projections of the full 2048-row batch), at ~25% of peak with an
  11-counter decode before each GEMM.

### Trainer: steady-state throughput line; step-time accounting corrected (2026-10-24)

- `motifcl_train_moe15m` now prints a warmup-excluded mean next to the existing
  window mean: `steady <tok/s> <ms/step> (n, warmup)`, with
  `MOTIFCL_WARMUP_STEPS` (default 1). The first step pays the driver's first-use
  cost for every pipeline and cost ~530 ms on a 25-step run, so the window mean
  read 177.5 ms/step where the steady state is 150.7-153.8 ms (13317-13586 tok/s
  vs 11512-11536). The window mean is unchanged so earlier numbers stay comparable.
- Corrected accounting (`reports/vulkan-moe-sparse-2026-10-21.md`, Phase O): the
  trainer is ~94% GPU-busy in steady state (GPU 141.4-141.5 ms forward+backward,
  4.0 ms optimizer, 0.3-0.4 ms idle per submit, 6.4 ms host recording). The
  "~30 ms of host idle per step" quoted in earlier phases was warmup-polluted, so
  host/device overlap is worth 2-4%, not 8-10%.
- Rejected on measurement: rectangular 128x64 / 64x128 dense GEMM tiles. They cut
  the LM-head operand re-reads by ~25% in the traffic model and were verified
  bit-identical to the 64x64 kernel (17 assertions plus identical checkpoints),
  but measured slower on every head shape (head_fwd 10.97 -> 13.2 ms, head_dA
  15.30 -> 21.95 ms, head_dB 11.44 -> 16.3 ms; trainer GPU 141.4-142.5 -> 151.2-
  151.4 ms). The 64x64 tile's four-waves/four-workgroups-per-CU occupancy is a
  local optimum. Reverted with its test.

### GQA causal forward: eight query rows by default (2026-10-24)

- `gqa_fwd_causal8_f32` (eight causal query rows per workgroup, half the K/V
  re-reads) is now the kernel the causal fast path selects;
  `MOTIFCL_VK_GQA_FWD_ROWS=4` restores the four-row kernel.
- End-to-end evidence: two 6-step full runs, one per kernel, produce 430
  byte-identical checkpoint files with the same FINAL VAL (6.4265); 25-step A/B,
  three pairs, GPU forward+backward 142.16-143.01 -> 141.48-142.46 ms (~0.5%),
  tok/s within host noise.
- Also recorded in `reports/vulkan-moe-sparse-2026-10-21.md` (Phase N): two
  audited dead ends. A vec4 cached-CE backward was flat (the pass is already
  bandwidth-bound at ~140 GB/s and the kernel itself got slower), and a dedicated
  width-8 register kernel for the twelve [288,7] router/balance-gradient GEMMs cost
  10 ms of GPU per step because five workgroups cannot hide memory latency that
  the tiled kernel's LDS staging does. Both were reverted.

### Split RMSNorm grad_w by default: +1.5% trainer throughput (2026-10-24)

- The non-split `rmsnorm_bwd_w_f32` launches one workgroup per 64 columns (5 for a
  288-wide norm) and walks all 2048 rows in each, so 13 calls per step cost 2.5% of
  the GPU at ~16 GB/s. The already-implemented split form
  (`rmsnorm_bwd_w_partial_f32` + `rmsnorm_bwd_w_reduce_f32`, 64 x 5 = 320
  workgroups) is now the default for training shapes (rows 256..65535,
  cols <= 1024); `MOTIFCL_VK_RMS_DW_SPLIT=0` restores the old launch, a number pins
  the split count.
- 25-step A/B, three ABBA pairs, same executable: 11219-11304 -> 11420-11432 tok/s
  (+1.5%, disjoint ranges), GPU forward+backward 145.2-146.1 -> 142.8-143.0 ms.
- Two identical 6-step full runs stay 430 byte-identical checkpoint files with
  FINAL VAL 6.4265. Evidence: `reports/vulkan-moe-sparse-2026-10-21.md`, Phase M.

### Reproducible training: deterministic embedding backward (2026-10-24)

- Root cause of the run-to-run checkpoint differences: on GCN4 / RX 580 (no float
  atomics) the embedding weight gradient was accumulated with a float
  compare-and-swap scatter, so duplicate token ids were summed in thread
  scheduling order. Bisection showed the drift was confined to the embedding row
  and its Adam state; `MOTIFCL_DISABLE_EMBEDDING_SCATTER=1` made runs identical.
- New deterministic path (default): `embedding_backward_dup_count_f32_i32` counts
  tokens per row with order-independent integer atomics and writes the
  single-token rows, publishing the duplicated rows; `embedding_backward_dup_sum_f32_i32`
  re-sums those rows in ascending token order. One writer per row per phase.
  `MOTIFCL_EMBEDDING_NONDETERMINISTIC=1` restores the scatter.
- Cost: ~0.4 ms GPU per step (~0.3%) measured A/B. Two identical 6-step full
  runs now produce 430 byte-identical checkpoint files with the same FINAL VAL
  (6.4265), so checkpoint hashes become a usable exactness witness.
- New witness `tests/test_vulkan_embedding_backward_det.cpp`: four fixtures
  (unique/duplicate/all-equal ids and the 32000x288x2048 trainer shape) match an
  independent CPU reference exactly (max_abs 0.0), repeat byte-identically, and
  the invalid-argument controls mutate nothing.
- Evidence: `reports/vulkan-moe-sparse-2026-10-21.md`, Phase L.

### Split forward/backward and GPU balance by default: +5.8% trainer throughput (2026-10-24)

- `MOTIFCL_MOE_SPLIT_FWD_BWD` and `MOTIFCL_MOE_GPU_BALANCE` now default to on.
  The split submits the forward batch, reads the routing counts back, and records
  the backward afterwards, which lets the MoE backward skip losing experts (their
  gradient is identically zero; `tests/test_vulkan_moe7_zero_grad.cpp` proves the
  dense and skip backwards byte-identical). The GPU balance-bias update joins the
  optimizer batch instead of a host round trip between the two batches.
- 25-step A/B, three ABBA pairs, same executable: 10538-10679 -> 11199-11290
  tok/s (+5.8%, disjoint ranges), step 192.7 -> 182.1 ms, GPU forward+backward
  155.1 -> 145.5 ms. Every printed step loss and the final validation loss are
  identical (10.4277, 10.2058, 8.8452, 7.2389, 6.7947, 6.5197; FINAL VAL 6.4265).
  The trainer is not run-to-run bit-reproducible, so the skip's exactness rests
  on its dedicated byte-for-byte witness, not on checkpoint hashes.
- `=0` env overrides keep the previous one-batch / host-bias path for A/B.
- Evidence: `reports/vulkan-moe-sparse-2026-10-21.md`, Phase K.

### Device-side MoE balance gradient: +2.0% trainer throughput (2026-10-24)

- `moe_balance_grad_f32` computes the auxiliary load-balancing router gradient
  on the device from the forward's logits, instead of downloading [N,7] logits
  and uploading [N,7] gradients per MoE block per step
  (`MOTIFCL_MOE_HOST_BALANCE_GRAD=1` restores the host path). The balance phase
  drops 8.5 -> 2.0 ms/step; 25-step A/B 10302-10394 -> 10569-10597 tok/s (+2.0%,
  disjoint ranges), with identical training losses at printed precision.
- New standalone witness `tests/test_vulkan_moe_balance_grad.cpp`: four fixtures
  against an independent CPU implementation (max_abs 1.3e-12..3.3e-11) plus
  invalid-argument controls (rows, NaN/Inf, aliasing, undersized buffers).
- Evidence: `reports/vulkan-moe-sparse-2026-10-21.md`, Phase J.

### Counter GEMM full-tile pipeline: +12.1% trainer throughput (2026-10-24)

- `counter_gemm_db` selects a bounds-free, hoisted-address pipeline for tiles
  that are completely inside the matrix (all trainer shapes except edge tiles,
  which keep the guarded pipeline in the same shader). The choice is a
  per-workgroup uniform branch, so there is no extra dispatch and the arithmetic
  order is unchanged: 1.48-1.95x on the counter GEMM at the trainer shapes and
  +12.1% end to end (10707 -> 12007 tok/s, step 191.3 -> 170.5 ms), with
  bit-identical training losses and final validation loss.
  `MOTIFCL_COUNTER_GEMM_GUARDED=1` forces the guarded pipeline for A/B in one
  binary. Evidence: `reports/vulkan-moe-sparse-2026-10-21.md`, Phase H.

### MoE sparse execution, counter-GEMM measurement, allocation classes (2026-10-21/22)

- Sparse Vulkan MoE7 forward: `moe_route_top1_f32` (router dot product, bias,
  strict `>` argmax, atomic per-expert compaction on device),
  `counter_gemm_gather_db` (gather/scatter row addressing around the list, same
  tile and accumulation order as the dense kernel), `moe_swiglu_bounded_f32`
  (gated on the expert's live row count). Full MoE forward: prefill T=2048
  3663.8 -> 1037.1 us (3.53x), decode T=1 1164.2 -> 228.8 us (5.09x), 28 -> 23
  dispatches, `rel_err 0` against the dense path. A sparse forward cannot train
  (the router gradient needs the losing experts' values); the sparse backward is
  scoped, not enabled.
- Zero-gradient expert skip (default on): a losing top-1 expert's backward graph
  is skipped and its counter second moment is decayed explicitly, proven
  bit-identical across 16 counter layers x 5 steps
  (`tests/test_vulkan_moe7_zero_grad.cpp`). It only engages with the opt-in
  `MOTIFCL_MOE_SPLIT_FWD_BWD=1` ordering, which submits the forward, reads the
  routing back, then records the backward.
- Buffer pool size classes are the next eighth of an octave instead of the next
  power of two: reserved 3343 -> 2360 MB, slow allocations 28 -> 0 per step,
  step 258.7 -> 220.1 ms, 7917 -> 9305 tok/s (+21.3% over the Phase D baseline).
  Adds `buffer_bytes_requested()` / `buffer_bytes_requested_peak()`; the trainers
  print both.
- Counter GEMM investigation: `MOTIFCL_VK_SHADER_INFO=1` prints driver-reported
  VGPR/SGPR/LDS and waves per SIMD through `VK_AMD_shader_info`;
  `MOTIFCL_VK_SHADER_ISA=<kernel>|all` dumps the GCN ISA and opcode histogram.
  They show `counter_gemm_db` (29.6% of a training step) is issue-bound at
  ~5.4 instructions per FMA. The measured workaround,
  `MOTIFCL_COUNTER_DECODE_DENSE=1`, decodes the packed state into a reusable
  dense mirror and runs the tuned FP32 GEMM: 1.5-2.0x per projection, +6.9% end
  to end (9146 -> 9760 tok/s), validation perplexity unchanged within the
  measured spread; gated to >=128 rows because the decode is a fixed ~15 us.
- Evidence: `reports/vulkan-moe-sparse-2026-10-21.md` (phases A-G, including the
  process error that invalidated one A/B and how the correctness test caught it).

### Vulkan hardening: cached CE, bounded workspaces, guards (2026-10-18)

- Training cross-entropy caches each row's max and inverse exponential sum on
  device and uses a coalesced cached backward; portable (non-subgroup) statistics
  are the default and are bit-exact against the legacy composition.
  `MOTIFCL_VK_CE_CACHE=0` forces the old path.
- One wave64 predicate (`supports_wave64_arithmetic()`) now gates every
  subgroup-only reduction (counter update, CE, softmax, RMSNorm, GQA).
- Counter update validates dtype, shape, backend and optimizer parameters before
  dispatch; the constructor restricts C to [1,11].
- Command-ordered weak workspace reuse replaces per-layer dense `grad_w`
  scratch and slightly growing shapes reuse one capacity bucket (65 live
  requests: 67.6 MB -> 1.6 MB, 65 -> 2 allocations).
- Optional RMSNorm weight-gradient row split for rows >=256 / cols <=1024.
- Python timing-window and benchmark-runner tests are now registered as CTest
  targets, so trainer log-format changes cannot silently break them.
- Evidence: `reports/vulkan-hardening-performance-2026-10-18.md`.

### Tiled causal attention + counter decode LUT (3rd pass)

- New tiled causal GQA attention fast path (causal, no mask, query_offset 0,
  sliding_window 0, key == query length <= 256, head_dim == v_head_dim <= 64;
  the general kernels remain the fallback):
  `gqa_fwd_causal4_f32` keeps four query rows per workgroup so each K/V vector is
  loaded once per four rows (2.99 -> 1.15 ms per layer),
  `gqa_bwd_probs_causal4_f32` does the same for the recompute + ds + dQ stage
  (5.42 -> 1.94 ms), and `gqa_bwd_dkv_causal_tile_f32` gives each workgroup a
  64-key tile with one lane per key (1.72 -> 1.02 ms). Attention block:
  61 -> 25 ms per training step.
- Counter decode now uses a 64-entry ternary table built once per workgroup in
  shared memory instead of a branch chain per decoded weight: 0.257 -> 0.228 ms
  per counter projection, 62 -> 55 ms per step.
- Trainers report the CPU record phase split into forward and backward
  (`GPU_TIMING=1`), which locates the remaining host cost at 18.4 ms forward +
  31.5 ms backward per step.
- MoE trainer (7x192 + shared 192, B8xT256): ~7000 -> **8500 tok/s**
  (239 ms/step, GPU fwd+bwd 175 ms); dense control (hidden 768): 9542 ->
  **12380 tok/s**. Full CTest 51/51, counter exactness and convergence witnesses
  intact. Evidence: `reports/vulkan-training-perf-2026-10-17.md`.

### Vulkan GEMM and counter kernels: double buffering, LDS.128 operands (2nd pass)

- New double-buffered register-blocked FP32 GEMM family
  (`kernels/vulkan/mm_f32_db_body.glsl` with `mm_f32_db32.comp` and
  `mm_f32_db64.comp`): the next K tile is loaded into registers before the
  current one is consumed, and A/B are read as `vec4` (`LDS.128`) instead of
  eight scalar shared-memory operands per k step. One body covers A*B, A*B^T and
  A^T*B. Measured on the training shapes: LM-head forward 25.3% -> **62.4%** of
  the RX 580 FP32 peak (24.2 ms -> 9.8 ms), head dA 15.2% -> 41.8%, head dB
  25.9% -> 57.9%, expert/attention projections 22% -> 47-57%. The kernels are
  the default for shapes with all dimensions divisible by four; the previous
  scalar/register-blocked kernels stay as the fallback.
  `MOTIFCL_VK_MATMUL_VARIANT=base|rb4|db` and `MOTIFCL_VK_MATMUL_DB_TILE=32|64`
  force a variant for A/B runs.
- New `kernels/vulkan/counter_gemm_db.comp`: double-buffered fused 6-bit counter
  decode + GEMM for both counter training projections (forward and
  backward-input, one mode flag). Decode of the next tile overlaps the current
  tile's FMA loop. Counter projections: ~24% -> ~30% of peak, 77 -> 62 ms per
  training step.
- New `benchmarks/bench_vulkan_gemm_shapes`: measures the exact trainer GEMM
  shapes with VK timestamp queries and reports TFLOP/s against the Polaris FP32
  peak. New `tests/test_vulkan_gemm_parity`: CPU-reference parity for all three
  layouts across 13 shapes (tile multiples, partial tiles, partial K tiles, odd
  dimensions taking the scalar fallback).
- MoE trainer (7x192 + shared 192, B8xT256): 5666-5701 -> **~7000 tok/s**
  (289 ms/step, GPU 294 -> 222 ms); dense control (hidden 768): 8800 -> 9542
  tok/s. Full CTest 51/51, counter exactness and convergence witnesses intact.
  Evidence: `reports/vulkan-training-perf-2026-10-17.md`.

### Vulkan training throughput (RX 580, measured 8.3x on the memory-native MoE trainer)

- Device-local VRAM is now the default storage-buffer placement for the Vulkan
  runtime (was opt-in via `MOTIFCL_VK_PREFER_DEVICE_LOCAL=1`); allocation falls
  back to BAR then host memory when VRAM is full, and
  `MOTIFCL_VK_PREFER_DEVICE_LOCAL=0` restores the previous host-visible/GTT
  policy. Measured 2.2x on the MoE trainer with bit-identical losses.
- Buffers up to 128 KiB are allocated host-visible by default
  (`MOTIFCL_VK_SMALL_HOST_BYTES` overrides, `0` disables) so per-step scalar,
  router-logit and balance-gradient transfers are memcpy instead of a staging
  submit+fence. Measured: balance phase 40-160 ms -> 8.5-13.8 ms per step.
- Raised the Vulkan buffer-pool budget from 2 GiB / 2048 allocations to
  6 GiB / 4096 / 256 per bucket. The old cap was below the trainer's ~3.4 GiB
  working set, so the 262 MB vocabulary-logit buffers were re-allocated through
  `vkAllocateMemory` every step (measured CPU recording phase 960 ms -> 50 ms).
- New `kernels/vulkan/counter_update_apply_f32.comp`: one wave64 workgroup per
  output row folds the counter row statistics, commits `v`/`scale`, and runs the
  finite-state stochastic-rounding tick, with `grad_w = grad_out^T * x` formed
  once by the tiled transposed GEMM into a reusable transient scratch. Replaces
  two scalar update kernels that recomputed that GEMM with strided reads, and
  drops the separate row-scale copy dispatch: 2.9 ms -> 0.30 ms per counter layer
  (49% of a training step's GPU time). Layers whose dense gradient would exceed
  32 MiB keep the allocation-free recompute path.
- Added an opt-in per-kernel GPU profiler (`MOTIFCL_VK_PROFILE=1`,
  `VulkanRuntime::kernel_profile()`), submit/fence wall-time accounting
  (`last_submit_time_us`, `submit_count`), a generated kernel-name table
  (`src/runtime/vulkan_spirv_names.inc`, `tools/gen_vulkan_spirv_names.py`), and
  per-phase step instrumentation in the trainers (`GPU_TIMING=1`).
- `tools/gen_vulkan_spirv.py` now writes the generated SPIR-V header as UTF-8.
- Evidence: `reports/vulkan-training-perf-2026-10-17.md`. Memory-native MoE
  (7x192 + shared 192, B8xT256) 635 tok/s -> **5666-5701 tok/s** (120-step steady
  state, 357-365 ms/step, flat device memory); dense control (hidden 768) 8800
  tok/s. Full CTest 50/50.

- Added `ModernModelSpec` / `LayerSpec` graph extraction, long-context runtime
  metadata plus paged KV/DeltaNet state-cache containers, Gemma4 dense text-core parsing with per-layer local/global
  attention windows, Qwen3.5 DeltaNet/Gated-Attention/MoE and Mixtral MoE graph
  execution through `HybridGPTModel`, and optional `tools/parity_harness.py`.
- Added GGUF `Q4_0`/`Q8_0` native quant repacking through
  `gguf::File::read_tensor_quantized()` and packed GGUF `Q4_K`/`Q5_K` tensor
  reads, plus direct OpenCL `q8_0 x Q4_K/Q5_K` matmul so those payloads can be
  represented as MotifCL quant tensors without expanding to F32.
- Added a modern-model runner front door: architecture registry and readiness
  probe APIs, `motifcl_generate_transformer --inspect`,
  `--list-architectures`, built-in chat prompt templates (`chatml`, `llama3`,
  `mistral`, `gemma`, `generic`), Qwen3.5/Mixtral hybrid routing, and explicit
  blocker reporting for detected but unsupported families such as Gemma 2/3, Qwen3, Phi,
  DeepSeek, Falcon, GPT-NeoX, and Mamba.
- Added a GGUF loader foundation for reading metadata, tokenizer token arrays,
  tensor directory entries, tensor alignment, tensor type/block sizes, and raw
  tensor payloads; mapping GGUF metadata to `HFTransformerConfig`; mapping
  common `blk.*` LLaMA-style tensor names; loading F16/F32/BF16 plus direct
  packed GGUF `Q4_0`, `Q8_0`, `Q4_K`, and `Q5_K` payloads into split
  `ModernGPTModel` projections/LM head with safe packed transpose/repack; and accepting `.gguf` files through
  `motifcl_generate_transformer --model`.

## 1.1.1 - 2026-05-20

- Hardened quantized tensor metadata validation so serialized `.mclt` and checkpoint-loaded scale tensors must cover the target tensor shape before any GPU kernel can consume them.
- Added per-callsite CPU-side validation contracts for quantized matmul, attention, KV-cache, paged attention, and quant tensor dispatch paths.
- Added malformed-artifact regression coverage for `.mclt`, GGUF metadata/tensor tables, and quantized checkpoint payloads so invalid inputs fail with controlled `motifcl::Error` paths instead of crash/OOB behavior.
- Added explicit experimental/stable boundaries for compressed KV and microkernel backend selection; `native`/`asm` backends remain opt-in fallback paths until correctness and perf gates pass.
- Added GitHub Actions release gate covering kernel contracts, release build, full CTest, Python build/pytest, release check, sanitizer, and platform build smoke.
- Fixed backend teardown ordering by draining the OpenCL queue before clearing pooled buffers and releasing backend lifetime state.

## 1.1.0 - 2026-05-10

- Added generic HF-style modern decoder compatibility APIs: `HFArchitecture`,
  `HFTransformerConfig`, config normalization, common LLaMA/Gemma/Mistral/Qwen2
  safetensors weight loading wrappers, tokenizer loading, and `generate_hf_text`.
- Added Q4_0/Q8_0 no-grad modern transformer inference by quantizing `Linear`
  projection weights and the LM head while keeping the FP32 training path intact,
  plus mixed per-layer `QuantizationPolicy` and policy-aware checkpoint export.
- Changed generation to prefill the prompt through one cached forward pass by
  default, with token-by-token prefill kept as a debug option.
- Added `motifcl_generate_transformer` CLI for launching HF-style model
  directories through the existing `ModernGPTModel`/`KVCache` stack, including
  `--quant q4|q8` and `--no-prefill`.
- Added C++ and Python smoke coverage for the generic modern HF compatibility
  layer plus optional real-HF smoke coverage for tiny LLaMA/Mistral/Qwen2/Gemma
  repos, so Gemma remains a compatibility profile rather than a separate branch.
- Added a PyTorch-like `motifcl.nn` Python facade with Python-side `Module`,
  recursive parameter collection, `state_dict`/`load_state_dict`, `train`/`eval`,
  `no_grad`, and Tensor operator overloads.
- Added GPU-side greedy/top-k/top-p/temperature sampling, fused padded
  variable-length batch generation with per-row positioned decode cache writes,
  binary `tokenizer.model` pieces, `vocab.json`/`merges.txt` and tokenizer.json
  BPE/Unigram support, whole-model quantized transformer checkpoint manifests,
  and persisted Q4_0/Q8_0 tensor metadata in tensor serialization.
- Added checked-in performance baseline scaffolding and CI smoke hooks for
  perf-regression tooling with required metric/metadata gates, install/consumer
  verification, selected clang-tidy diagnostics, and local `tools/release_check.py`.

## 1.0.0 - 2026-05-04

Initial productionized MotifCL source release.

- C++17/OpenCL runtime, tensor API, ops, autograd, NN modules, examples, tools, tests, and CMake install/export.
- Python package and wheel build via scikit-build-core / pybind11.
- Quantization coverage for Q8_0/Q4_0, mixed quantized matmul, per-axis/blockwise metadata.
- Graph capture/replay with kernel replay, host-reduction replay for scalar losses, tensor metadata, buffer-planning analysis, and shape-polymorphic signatures.
- Optimized kernels for register-blocked matmul, FlashAttention-style tiled attention, and workgroup row reductions for norm/reduce paths.
