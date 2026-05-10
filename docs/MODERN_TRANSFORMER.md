# Modern transformer API

MotifCL has a modern decoder-only transformer path for current GPT-like models. Prefer it over the legacy `GPTModel` when testing current architecture ideas. HF-style model loading now targets this same path; Gemma is an adapter/profile, not a separate model branch.

## Components

- `TransformerConfig`: single config object for model dimensions and feature switches.
- `qkv_split`: splits a fused `[Q | K | V]` projection into Q, K, V tensors with backward support.
- `rope`: rotary positional embeddings with `token_offset` for cached inference.
- `grouped_query_attention`: attention with `n_kv_head <= n_head`; `n_kv_head == 1` is MQA.
- `grouped_query_attention_masked`: GQA/MQA attention with flexible binary masks or F32 additive attention bias.
- `ModernSelfAttention`: fused QKV projection, optional RoPE, GQA/MQA, output projection.
- `ModernMLP`: GELU or SwiGLU feed-forward path.
- `ModernTransformerBlock`: RMSNorm + attention + residual + RMSNorm + MLP + residual.
- `ModernGPTModel`: token embedding, modern blocks, final RMSNorm, LM head.
- `KVCache`: preallocated K/V cache for incremental inference.
- `rope_positions` / `kv_cache_append_positions`: per-row decode positions for
  variable-length batch generation after padded prompt prefill.
- `HFTransformerConfig`: HF-style config normalization and architecture dispatch for Gemma/LLaMA/Mistral/Qwen2-style decoder-only models.
- GPU `dropout` and GPU `masked_fill`: no CPU round-trip in the normal path; both support autograd for the input tensor.

## Mask API

Binary masks use the same convention as `masked_fill`: nonzero means “blocked”. Supported mask shapes:

- `[Q, K]` — one query/key mask shared by the batch;
- `[B, K]` — key padding mask shared by all query tokens in each batch row;
- `[B*Q, K]` or `[B, Q, K]` — full per-batch/per-query mask;
- `[B, 1, K]` — broadcast query dimension;
- `[1, Q, K]` — broadcast batch dimension.

F32 additive masks are also supported with `additive_mask=true`; values are added to attention scores after scaling, so `-1e30` behaves like a blocked key.

## C++ smoke

```cpp
#include <motifcl/motifcl.hpp>

auto backend = motifcl::Backend::create_opencl();

motifcl::nn::TransformerConfig cfg;
cfg.vocab_size = 32000;
cfg.block_size = 256;
cfg.n_embd = 512;
cfg.n_head = 8;
cfg.n_kv_head = 2; // GQA; use 1 for MQA, n_head for normal MHA
cfg.n_layer = 6;
cfg.use_rope = true;
cfg.use_swiglu = true;

motifcl::nn::ModernGPTModel model(backend, cfg);
std::int32_t ids_host[] = {1, 2, 3, 4};
auto ids = motifcl::Tensor::from_cpu(backend, {1, 4}, motifcl::DType::I32, ids_host);
auto logits = model.forward(ids); // [1, 4, vocab_size]
```

## Python smoke

```python
import motifcl as mcl

b = mcl.Backend.create()
cfg = mcl.TransformerConfig()
cfg.vocab_size = 32000
cfg.block_size = 256
cfg.n_embd = 512
cfg.n_head = 8
cfg.n_kv_head = 2
cfg.n_layer = 6
cfg.use_rope = True
cfg.use_swiglu = True

model = mcl.ModernGPTModel(b, cfg)
tokens = mcl.tensor_i32(b, [1, 4], [1, 2, 3, 4])
logits = model.forward(tokens)

mask = mcl.tensor_i32(b, [4, 4], [
    0, 1, 1, 1,
    0, 0, 1, 1,
    0, 0, 0, 1,
    0, 0, 0, 0,
])
masked_logits = model.forward_masked(tokens, mask)
```

## KV-cache inference

For incremental decode, create one cache per layer through `ModernGPTModel::create_kv_cache` in C++ or `model.create_kv_cache(...)` in Python, or construct `KVCache` directly for standalone attention tests.

`ModernSelfAttention::forward_with_cache(x, cache, batch, seq)` appends the new K/V rows and uses `rope(..., token_offset=old_cache_length)` so rotary positions line up with the total decoded length.
`ModernGPTModel::forward_with_cache(token_ids, caches)` runs the whole model over only the new tokens and updates every layer cache.

## HF-style launch path

```cpp
auto cfg = motifcl::nn::load_hf_transformer_config_json("model/config.json");
auto model = motifcl::nn::make_hf_transformer_model(backend, cfg);
auto report = motifcl::nn::load_hf_transformer_weights(backend, model, shards, cfg);
motifcl::nn::enable_hf_transformer_quantized_inference(model, motifcl::DType::Q4_0);
auto tokenizer = motifcl::nn::load_hf_tokenizer("model", cfg);
auto text = motifcl::nn::generate_hf_text(backend, model, tokenizer, "Hello");
```

The `motifcl_generate_transformer` tool wraps the same path for command-line smoke/inference runs. Use `--quant q4|q8` for inference-only quantized projections/LM head. Prompt prefill uses one cached forward pass by default; `--no-prefill` preserves token-by-token debugging behavior. Greedy, temperature, top-k, and top-p decoding use GPU `rowwise_sample` by default; `--cpu-sampling` forces the CPU logits path.

## Current limits

The modern path now uses staged GQA/MQA backward kernels for common GPT sizes (`key_len <= 128`, `head_dim <= 128`), including masked attention. Larger or unsupported shapes fall back to the correctness-first fused backward kernel. The old equal-head MHA path still uses the faster tiled attention kernels where applicable. HF compatibility currently covers the common LLaMA/Gemma/Mistral/Qwen2-style `model.layers.*` weight layout. Tokenizer support handles binary `tokenizer.model` pieces, `vocab.json` + `merges.txt`, HF `tokenizer.json` BPE merges, and Unigram/SentencePiece-style `▁` vocab arrays; full SentencePiece normalizer parity remains future work. Q4/Q8 projection quantization is inference-only; tensor serialization preserves quant metadata and `save_quantized_transformer_checkpoint()` / `load_quantized_transformer_checkpoint()` persist whole-model inference manifests, including mixed Q4/Q8 policy metadata. Batch generation uses a fused padded prefill and per-row positioned decode cache writes; fully compact packed ragged KV-cache layouts remain future performance work. GPU top-p is implemented with a 128-candidate cap for large vocabularies unless `top_k` narrows the candidate set. Mixed precision and vendor-specific attention kernels remain separate performance work.

Speed regression coverage lives in `benchmarks/bench_modern_transformer.cpp`.
