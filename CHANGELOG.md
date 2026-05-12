# Changelog

## Unreleased

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
