# Long context and TurboQuant notes

This repository can already read the native context advertised by GGUF/HF metadata.
For Gemma 4 E2B GGUF the inspected value is `context_length=131072`, matching
Google's public Gemma 4 statement that edge models use a 128K context window.

## What is implemented now

- `motifcl run` / `motifcl up` default to `--ctx-size auto`.
  - `auto` means a safe 8K runtime cap, or the model native context if smaller.
  - Override the auto cap with `MOTIFCL_CTX_AUTO_MAX`.
- `--ctx-size max`, `--ctx-size native`, and `--ctx-size model` use the model-native
  context length from metadata.
- Integer `--ctx-size N` caps runtime cache/context to `min(N, native_context)`.
- The background launcher now restarts its owned server when context/cache-affecting
  options change, instead of reusing a stale hot server with the wrong context.
- `/health` reports `ctx_size`, generation defaults, chat template, and paged-KV flags.
- `motifcl inspect` shows the resolved context policy without starting a server.
- `motifcl_generate_transformer` and `motifcl_warm_decode_bench` accept
  `--ctx-size auto|max|native|model|N`.
- `--paged-kv --kv-page-size N` is exposed through the one-command launcher for
  measured long-context experiments, but it is not a default.
- `--kv-cache-dtype f32|q8|q4` is an explicit compressed-KV baseline for the
  regular KV cache path. It uses row-wise scales, quantized append kernels, and
  q8/q4 decode attention fast paths:
  - a single-workgroup local-score path while the attended key range fits device
    local memory;
  - a global-score two-kernel fallback for longer decode contexts, so q8/q4 does
    not fall back to the old per-output-element scalar dequant path.
  It is intentionally **not** default. Paged-KV exposes the same `q8`/`q4`
  cache dtype knob for experiments, but it still needs broader long-context
  performance and quality gates before becoming a recommended default.

Examples:

```powershell
motifcl inspect
motifcl run --ctx-size auto
motifcl run --ctx-size 32768
motifcl run --ctx-size max
motifcl run --kv-cache-dtype q8
motifcl run --kv-cache-dtype q4
$env:MOTIFCL_CTX_AUTO_MAX = "16384"; motifcl run --ctx-size auto
```

Warm benchmark:

```powershell
.\build_kquant_prefill\tools\motifcl_warm_decode_bench.exe `
  --model .\build\models\gemma-4-E2B-it-GGUF\gemma-4-E2B-it-Q4_K_M.gguf `
  --ctx-size auto `
  --prompt "Hello" `
  --max-new-tokens 16 `
  --warmup 1 `
  --iters 3 `
  --ignore-eos `
  --kv-cache-dtype q8
```

The benchmark prints `load_ms`, `native_context_length`, `ctx_size`,
`kv_cache_dtype`, `prompt_eval_ms`, and `decode_tok_s`.

Measured smoke on Gemma 4 E2B Q4_K_M, `ctx_size=512`, prompt length 112 tokens,
`max_new_tokens=8`, `--ignore-eos`, cold single iteration:

| KV cache | prompt_eval_ms | decode_tok_s | status |
| --- | ---: | ---: | --- |
| f32 | 20225.915 | 7.581 | baseline |
| q8 row-wise | 19624.466 | 9.058 | decode faster on this smoke |
| q8 row-wise, forced global-score fallback | 20968.414 | 8.830 | scalable long-context fallback path |
| q4 row-wise | 20994.819 | 8.847 | decode faster, prompt still slower |

Interpretation: compressed regular KV is now a real decode-speed path, not just
a memory-saving path. However, it still is not a default because the measured win
is decode-side only: long prompt/prefill, long-context quality checks, paged-KV
interaction, and broader model coverage still need wall-speed gates.

## Research summary

- Native context first: Gemma 4 E2B/E4B already advertise 128K context. For this
  model, the first correct step is to honor native metadata and allocate/cache
  accordingly; RoPE extrapolation is not needed just to reach 128K.
- RoPE extension is a separate feature: vLLM's current long-context examples use
  HF `rope_parameters`/YaRN-style overrides plus a serving-side max length. HF
  supports per-layer RoPE settings for models that mix full and sliding attention.
- Paged attention/paged KV is a memory-management feature, not automatic quality
  extension. It becomes important when long contexts cause contiguous KV allocation
  pressure or fragmented server workloads.
- TurboQuant targets KV-cache compression. Google's paper/blog describe
  PolarQuant plus a 1-bit QJL residual stage, with reported near-lossless KV
  quantization around 3.5 bits/channel and speedups for attention-logit computation.

Sources:

- Gemma 4 announcement: https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/
- vLLM context extension: https://docs.vllm.ai/en/latest/features/context_extension/
- HF RoPE parameters: https://huggingface.co/docs/transformers/main/internal/rope_utils
- TurboQuant announcement: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- TurboQuant paper: https://arxiv.org/abs/2504.19874

## TurboQuant decision

Do **not** switch the runtime to TurboQuant by default yet.

Reason: TurboQuant helps the long-context KV bottleneck, but it is not just a CLI
flag. MotifCL now has a baseline compressed KV path, but a real TurboQuant path
still requires rotation/PolarQuant/QJL-specific packing plus tiled attention
kernels and quality/perf gates.

Practical staged implementation:

1. Keep native-context `auto/max` and paged-KV toggles working and benchmarkable.
2. Baseline compressed-KV path (`q8` and simple symmetric `q4`) exists behind
   `--kv-cache-dtype`; decode now has q8/q4 tiled/global-score fast paths, but
   keep it benchmark-only until long-context prompt/decode and quality gates win.
3. Implement TurboQuant-style per-vector rotation/PolarQuant + QJL residual only
   after the compressed-KV abstraction is proven by wall benchmark.
4. Gate any compressed KV default by:
   - long-context prompt wall speed,
   - decode tokens/s,
   - memory use,
   - deterministic short-prompt parity,
   - long-context retrieval/needle smoke.

In short: native 128K support and paged-KV are the right product-level step now;
TurboQuant is a real next project, not a safe one-pass default.
