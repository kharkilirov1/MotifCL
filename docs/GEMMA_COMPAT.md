# Gemma compatibility profile

Gemma support is not a separate MotifCL model branch. It is the first profile of
the generic modern HF-style decoder compatibility layer described in
`docs/HF_COMPAT.md`, and it runs on top of `nn::ModernGPTModel`.

Implemented:

- `nn::GemmaConfig` and `load_gemma_config_json()` for backwards-compatible callers;
- generic `nn::HFTransformerConfig`, `load_hf_transformer_config_json()`, and
  `make_hf_transformer_model()`;
- `SafeTensorsFile` and `load_safetensors()` for `.safetensors` metadata and tensor loading;
- HF Gemma weight-name mapper and `load_gemma_hf_weights()`;
- generic `load_hf_transformer_weights()` wrapper for common LLaMA/Gemma/Mistral/Qwen2-style layouts;
- byte-fallback / simple-vocab / binary `tokenizer.model` pieces /
  tokenizer.json BPE-Unigram `GemmaTokenizer`;
- `nn::QuantizedLinear` for Q8/Q4 inference matmul;
- `nn::generate()` / `nn::generate_text()` and `nn::generate_hf_text()` with `KVCache`.

## C++ sketch

```cpp
auto backend = motifcl::Backend::create_opencl();
auto cfg = motifcl::nn::load_hf_transformer_config_json("config.json");
auto model = motifcl::nn::make_hf_transformer_model(backend, cfg);

auto report = motifcl::nn::load_hf_transformer_weights(
    backend,
    model,
    {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"},
    cfg,
    false,
    false);

motifcl::nn::QuantizationPolicy policy;
policy.default_dtype = motifcl::DType::Q4_0;
policy.lm_head_dtype = motifcl::DType::Q8_0;
model.enable_quantized_inference(policy);
auto tokenizer = motifcl::nn::load_hf_tokenizer(".", cfg);
motifcl::nn::GenerateOptions opts;
opts.max_new_tokens = 64;
opts.eos_token_id = cfg.eos_token_id;

auto text = motifcl::nn::generate_hf_text(backend, model, tokenizer, "Hello", opts);
```

## Current limitations

- The tokenizer supports byte fallback, text vocab maps, binary
  `tokenizer.model` pieces, HF `tokenizer.json` BPE merges, Unigram vocab
  arrays, SentencePiece-style `▁` normalization, and added tokens. Full
  SentencePiece normalizer parity remains future work.
- Weight loading targets the common HF Gemma tensor names and packs Q/K/V plus
  gate/up projections into MotifCL fused internal weights.
- Quantized linear is inference-only: it quantizes activations to Q8 rows and uses
  Q8/Q4 quantized matmul kernels for quantized weights.
- `generate()` uses model-side `KVCache`; `generate_batch()` runs a fused
  padded prefill and per-row positioned decode cache path. Greedy, temperature,
  top-k, and top-p selection use GPU `rowwise_sample()` unless CPU sampling is
  explicitly requested. Fully compact packed ragged KV storage remains future
  performance work.
