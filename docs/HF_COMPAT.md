# Modern HF-style transformer compatibility

MotifCL's model-loading direction is a generic **modern decoder-only transformer**
path, not a dedicated Gemma branch. The public flow is:

1. read a HF-style `config.json` into `nn::HFTransformerConfig`;
2. map that config to the existing `nn::TransformerConfig`;
3. instantiate `nn::ModernGPTModel`;
4. optionally load `.safetensors` weights through an architecture adapter;
5. tokenize and generate through `KVCache`-backed modern inference.

Gemma is currently one adapter/profile on this path. LLaMA/Mistral/Qwen2-style
configs are also accepted when they use the common decoder-only schema:

- `vocab_size`
- `max_position_embeddings` / `block_size` / `seq_length`
- `hidden_size`
- `intermediate_size`
- `num_hidden_layers`
- `num_attention_heads`
- `num_key_value_heads`
- `rms_norm_eps`
- `rope_theta`
- `attention_bias`
- `bos_token_id`, `eos_token_id`, `pad_token_id`

## C++ API

```cpp
auto backend = motifcl::Backend::create_opencl();
auto cfg = motifcl::nn::load_hf_transformer_config_json("model/config.json");
auto model = motifcl::nn::make_hf_transformer_model(backend, cfg);

auto report = motifcl::nn::load_hf_transformer_weights(
    backend,
    model,
    {"model/model-00001-of-00002.safetensors", "model/model-00002-of-00002.safetensors"},
    cfg,
    false,
    false);

motifcl::nn::enable_hf_transformer_quantized_inference(model, motifcl::DType::Q4_0);
auto tokenizer = motifcl::nn::load_hf_tokenizer("model", cfg);
motifcl::nn::GenerateOptions opts;
opts.max_new_tokens = 64;
auto text = motifcl::nn::generate_hf_text(backend, model, tokenizer, "Hello", opts);
```

## CLI

```bash
motifcl_generate_transformer \
  --model-dir ./model \
  --arch auto \
  --quant q4 \
  --prompt "Hello" \
  --max-new-tokens 64
```

Use `--weights` repeatedly to pass explicit shards, or let the tool discover
`*.safetensors` files in `--model-dir`. Use `--random-init` only for smoke tests.

## Current adapter coverage

Implemented:

- architecture dispatch: `auto`, `gemma`, `llama`, `mistral`, `qwen2`, `generic_decoder`;
- config normalization into `TransformerConfig`;
- common LLaMA/Gemma/Mistral/Qwen2-style `model.layers.*` weight names;
- Q/K/V and gate/up weight packing into MotifCL fused internal parameters;
- Q4_0/Q8_0 quantized modern transformer inference for `Linear` projections and
  the LM head through `enable_hf_transformer_quantized_inference()` / `--quant`,
  plus `QuantizationPolicy` for mixed per-layer Q4/Q8 and separate LM-head dtype;
- batch-1 cached generation through `ModernGPTModel` + `KVCache`, plus fused
  padded variable-length `generate_batch()` / `generate_batch_text()` /
  `generate_hf_batch_text()` paths that run one model forward per decode step
  for the whole batch instead of looping prompt-by-prompt; decode updates use
  per-row cache positions after the padded prefill so shorter rows do not keep
  appending at the global max prompt length;
- prompt prefill through one cached forward pass by default, with `prefill_prompt=false`
  / `--no-prefill` available for debugging;
- GPU-side `rowwise_sample()` for greedy, temperature, top-k, and top-p decode,
  avoiding full-vocab CPU logits download unless `gpu_greedy_sampling=false`
  / `--cpu-sampling` is requested;
- persisted quantized tensor metadata in `save_tensor()` / `load_tensor()`, so
  Q4_0/Q8_0 tensors retain per-tensor/per-axis/blockwise scales across disk
  round-trips;
- whole-model inference checkpoint helpers
  `save_quantized_transformer_checkpoint()` /
  `save_quantized_transformer_checkpoint()` plus policy-aware export that
  persist FP32 embeddings/norms plus mixed Q4/Q8 projection and LM-head weights
  under a JSON manifest;
- tokenizer loading from binary `tokenizer.model`, `vocab.json`/`merges.txt`,
  `tokenizer.json`, and `vocab.txt`, including BPE merge ranks, HF Unigram vocab
  arrays, SentencePiece-style `▁` space normalization, added tokens, sibling
  `merges.txt`, and byte fallback;
- optional real-HF integration smoke (`tools/hf_integration_smoke.py`) covering
  tiny LLaMA, Mistral, Qwen2, and Gemma repositories. It is offline/skipped by
  default and runs only with `--run` or `MOTIFCL_RUN_HF_INTEGRATION=1`.

Still intentionally limited:

- the binary SentencePiece reader is a minimal `ModelProto` parser for pieces
  and trainer model type; full normalizer/pre-tokenizer parity with
  SentencePiece is still future work;
- GPU top-p is exact for small vocabularies or `top_k <= 128`; for very large
  vocabularies without `top_k`, the current kernel samples from the best 128
  candidates;
- variable-length batch prefill is still padded/masked; fully compact packed
  ragged KV-cache layouts remain future performance work;
- architecture-specific variants that diverge from the common `model.layers.*`
  safetensors layout need their own mapper;
- quantized linear/model projection is inference-only and is activated only for no-grad generation/inference;
- full FP16 backward coverage remains separate training-stack work.
