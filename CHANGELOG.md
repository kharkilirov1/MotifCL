# Changelog

## Unreleased

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
