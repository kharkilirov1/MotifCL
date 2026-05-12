# Modern HF-style transformer compatibility

MotifCL's model-loading direction is a generic **modern decoder-only transformer**
path, not a dedicated Gemma branch. The public flow is:

1. read a HF-style `config.json` into `nn::HFTransformerConfig`;
2. derive a `nn::ModernModelSpec` graph (`LayerSpec` entries for attention,
   FFN, projector, MoE, and recurrent-state layers);
3. map the runnable text core to the existing `nn::TransformerConfig`;
4. instantiate `nn::ModernGPTModel` for dense attention+SwiGLU graphs or
   `nn::HybridGPTModel` for runnable DeltaNet/Gated-Attention/MoE graphs;
5. optionally load `.safetensors` weights through an architecture adapter or
   F16/F32/BF16 and common quantized tensors from a `.gguf` file through the
   GGUF adapter;
6. tokenize and generate through `KVCache`-backed modern inference.

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

Probe and registry front door:

```bash
motifcl_generate_transformer --list-architectures
motifcl_generate_transformer --model ./model.gguf --inspect
motifcl_generate_transformer --model-dir ./model --inspect
```

The probe reports detected format, architecture, shape/config summary,
tokenizer/weight availability, GGUF tensor counts, and explicit blockers for
families that are recognized but not runnable by the current `ModernGPTModel`
runner.

GGUF smoke/inference path:

```bash
motifcl_generate_transformer \
  --model ./model.gguf \
  --arch auto \
  --prompt "Hello" \
  --max-new-tokens 32
```

Chat-style prompting is available without adding a separate server layer:

```bash
motifcl_generate_transformer \
  --model ./qwen2.gguf \
  --system "You are concise." \
  --user "Explain MotifCL in one paragraph." \
  --chat-template auto \
  --max-new-tokens 64
```

Built-in templates: `chatml`, `llama3`, `mistral`, `gemma`, `generic`, and
`none`. `auto` uses tokenizer_config markers when present, then falls back by
architecture.

The direct GGUF path currently maps common LLaMA-style metadata/tensors:
`general.architecture`, `<arch>.context_length`, `<arch>.embedding_length`,
`<arch>.block_count`, `<arch>.feed_forward_length`,
`<arch>.attention.head_count(_kv)`, `tokenizer.ggml.tokens`, and `blk.*`
F16/F32/BF16 tensor payloads. GGUF `Q4_0`, `Q8_0`, `Q4_K`, and `Q5_K`
payloads are also accepted and dequantized to F32 parameters on model load.
For native quant plumbing, `gguf::File::read_tensor_quantized()` can repack GGUF
`Q4_0`/`Q8_0` blocks into MotifCL quant tensors with block scales and can keep
GGUF `Q4_K`/`Q5_K` payloads packed as MotifCL K-quant tensors. Direct OpenCL
`q8_0 activation x Q4_K/Q5_K weight` matmul kernels now cover the packed
K-quant projection path without F32 expansion.

## Current adapter coverage

Implemented:

- architecture dispatch/registry: runnable now for `auto`, `gemma`, `gemma4`
  dense text core, `llama`, `mistral`, `qwen2`, `qwen3.5`, `mixtral`,
  `generic_decoder`; probe/blocker reporting remains for Gemma multimodal
  variants, Qwen3, Phi, DeepSeek, Falcon, GPT-NeoX, Mamba;
- config normalization into `TransformerConfig`;
- `ModernModelSpec` / `LayerSpec` graph extraction with layer kinds for full
  attention, sliding attention, Gated DeltaNet, Gated Attention, MoE FFN,
  vision/audio projector, and SwiGLU FFN;
- Gemma4 dense text-core parsing, Gemma-style chat/tokenizer routing, common
  Gemma4 text weight mapping, and per-layer local/global attention windows wired
  into `ModernGPTModel` for unmasked forward/cached generation;
- Qwen3.5/Mixtral hybrid graph execution through `HybridGPTModel`, including
  DeltaNet state-cache layers, Gated Attention layers, MoE router/expert FFN
  execution, recurrent state-cache lifecycle, and the hybrid scheduler that
  interleaves DeltaNet and standard attention layers;
- long-context runtime planning metadata for paged KV, sliding-window cache, and
  separate recurrent state-cache layers;
- model readiness probe API and CLI `--inspect`;
- built-in chat prompt templates for common current instruct model families;
- common LLaMA/Gemma/Mistral/Qwen2-style `model.layers.*` weight names;
- common LLaMA-style GGUF `blk.*` tensor names with F16/F32/BF16 fallback loading
  plus native packed `Q4_0`/`Q8_0`/`Q4_K`/`Q5_K` installation into MotifCL
  `Linear`/LM-head inference weights; packed tensors already in `[in,out]`
  are installed directly, and `[out,in]` packed tensors are safely
  dequantized/transposed/repacked instead of being reinterpreted;
- Q/K/V and gate/up weight packing into fused F32 parameters, with split
  q/k/v and gate/up projection mode used for packed GGUF quant tensors;
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
  arrays, parsed SentencePiece normalizer flags (`add_dummy_prefix`,
  `remove_extra_whitespaces`, `escape_whitespaces`), NFKC-lite fullwidth/space
  normalization, added tokens, sibling `merges.txt`, and byte fallback;
- optional real-HF integration smoke (`tools/hf_integration_smoke.py`) covering
  tiny LLaMA, Mistral, Qwen2, and Gemma repositories. It is offline/skipped by
  default and runs only with `--run` or `MOTIFCL_RUN_HF_INTEGRATION=1`;
- optional parity harness (`tools/parity_harness.py`) for tokenizer/prompt/logits
  shape checks and HF numerical comparison when `torch`/`transformers` are
  installed. It is skipped unless `--run` or `MOTIFCL_RUN_PARITY=1` is used.

Still intentionally limited:

- the binary SentencePiece reader parses pieces, trainer model type, and the
  common normalizer flags used by current tiny/standard fixtures; exact
  byte-for-byte parity for arbitrary `precompiled_charsmap` ICU rules remains
  a compatibility-extension task unless a SentencePiece/ICU dependency is added;
- GPU top-p is exact for small vocabularies or `top_k <= 128`; for very large
  vocabularies without `top_k`, the current kernel samples from the best 128
  candidates;
- variable-length batch prefill is still padded/masked; fully compact packed
  ragged KV-cache layouts remain future performance work;
- architecture-specific variants that diverge beyond the implemented
  LLaMA/Gemma/Mistral/Qwen/Qwen3.5/Mixtral aliases or common GGUF `blk.*`
  layout still need dedicated mapper entries;
- `ModernModelSpec` represents MoE/projector/DeltaNet layers; runnable text-core
  dense graphs execute through `ModernGPTModel`, while runnable recurrent/MoE
  hybrid graphs execute through `HybridGPTModel`;
- quantized linear/model projection is inference-only and is activated only for no-grad generation/inference;
- full FP16 backward coverage remains separate training-stack work.
