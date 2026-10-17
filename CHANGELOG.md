# Changelog

## Unreleased

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
- Evidence: `reports/vulkan-training-perf-2026-10-17.md`. MoE (7x192 + shared
  192, B8xT256) 635 tok/s -> 4787-5396 tok/s; dense control (hidden 768) 8800
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
