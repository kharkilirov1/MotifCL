# MotifCL architecture

MotifCL is split into a small set of layers:

1. **Runtime** (`include/motifcl/runtime`, `src/runtime`) owns OpenCL context, queue, programs, kernels, events, profiling, and kernel source discovery.
2. **Tensor engine** (`include/motifcl/tensor`, `src/tensor`) owns shapes, dtype, storage, host upload/download, views, quant metadata, RNG-backed factories, and autograd handles.
3. **Ops** (`include/motifcl/ops`, `src/ops`, `kernels`) provide OpenCL kernels plus CPU-backed correctness fallbacks for broad API coverage such as broadcasting, slicing, masking, dropout, scalar reductions, RoPE, fused QKV splitting, GQA/MQA attention, KV-cache append, and SwiGLU.
4. **Autograd / graph capture** (`include/motifcl/autograd`, `src/autograd`) records eager backward nodes and static graph metadata. `CapturedGraph::compile_runtime_plan` validates runtime tensor specs and materializes tensor-to-allocation bindings for dynamic-shape planning.
5. **NN / training** (`include/motifcl/nn`, `include/motifcl/train`) provides modules, optimizers, dataloaders, LR schedulers, gradient clipping, training history, and checkpoint helpers.
6. **Python** (`python`) exposes the native extension and thin wrappers/stubs.

## Modern transformer path

The legacy `GPTModel` remains as a compact smoke-test architecture. New transformer experiments should use `TransformerConfig` with `ModernGPTModel`/`ModernTransformerBlock`/`ModernSelfAttention`:

- fused/flexible QKV projection: one linear projection produces `[Q | K | V]`, then `qkv_split` creates separate tensor views/copies with autograd;
- RoPE: `rope` rotates per-head channels and supports `token_offset` for incremental decode;
- GQA/MQA: `grouped_query_attention` allows `n_kv_head <= n_head`, including MQA with one KV head; backward uses staged kernels for common GPT sizes and falls back to correctness-first kernels outside that envelope;
- SwiGLU MLP: `ModernMLP` can use `swiglu(gate_up_proj(x))` before the down projection;
- mask API: `masked_fill` runs on GPU for F32/I32/U8 masks; `grouped_query_attention_masked` supports binary masks with `[Q,K]`, `[B,K]`, `[B*Q,K]`, `[B,Q,K]`, `[B,1,K]`, `[1,Q,K]`, plus F32 additive attention-bias masks;
- KV cache: `KVCache` stores preallocated K/V tensors and attention/block/model `forward_with_cache` appends new tokens for inference;
- dropout: `dropout` is GPU-backed and has an autograd node that reuses the generated mask in backward.

## Dynamic graph status

Graph capture can replay kernel launches bound to the original tensors, emit schedules, tensor specs, shape-polymorphic specs, liveness buffer plans, and runtime allocation bindings. Arbitrary kernel-argument rewiring for every captured op remains an explicit runtime-op integration point; CPU-backed fallback ops are intentionally recorded as non-replayable.

## Performance tuning

The default F32 matmul path uses the register-blocked OpenCL kernel. `motifcl_kernel_tuner` records timings in `motifcl_tuning.json`. For controlled regression/perf experiments, `MOTIFCL_MATMUL_F32_TILE=4|8|16` forces the generated tiled F32 matmul variant at runtime.
