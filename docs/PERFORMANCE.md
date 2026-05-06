# Performance and regression workflow

1. Build benchmarks:

```bash
cmake --preset release
cmake --build --preset release
```

2. Capture local tuning:

```bash
./build/release/tools/motifcl_kernel_tuner
```

3. Compare generated F32 tile variants manually:

```bash
MOTIFCL_MATMUL_F32_TILE=4  ./build/release/benchmarks/bench_matmul
MOTIFCL_MATMUL_F32_TILE=8  ./build/release/benchmarks/bench_matmul
MOTIFCL_MATMUL_F32_TILE=16 ./build/release/benchmarks/bench_matmul
```

4. Keep a checked-in or artifact-stored baseline JSON per device/driver. Treat regressions above 10-15% as suspicious unless explained by driver/device changes.

```bash
python tools/perf_regression.py motifcl_tuning.json baselines/ellesmere.json --tolerance 0.15
```

Current portable kernels are OpenCL C and do not include hand-written vendor ISA. Vendor-specific optimization should be added behind capability gates and backed by correctness/perf regression tests.

## Fast training mode currently implemented

The current fast path is still FP32/eager, but avoids several prototype bottlenecks:

- output tensors in hot GPU ops use `Tensor::empty` instead of CPU-zero-uploaded `Tensor::zeros`;
- scalar losses reduce on GPU via `mean_reduce_f32` instead of synchronously downloading partial losses;
- `Storage` uses an OpenCL buffer memory pool by default;
- Adam uses `adam_update_f32_fast`, which precomputes bias-correction scalars once per parameter update instead of calling `pow()` per element in the kernel;
- MLP uses fused `add_bias_gelu_rows_f32` for the first projection bias+GELU forward path;
- FP16 foundation is available for inference-style experiments: `cast_f32_to_f16`, `cast_f16_to_f32`, and `matmul_f16_accum_f32` when the OpenCL device exposes `cl_khr_fp16`;
- `GraphExecutor` wraps captured replayable graphs with runtime planning metadata and execution counters;
- autograd uses a topological backward scheduler, accumulating all fan-in gradients before executing each backward node once;
- GPT-sized attention backward uses a staged softmax-gradient path (`P`, `dP`, `dS`, then `dQ/dK/dV`) for seq_len <= 128, avoiding the previous repeated softmax recomputation inside `dK/dV`.
- RMSNorm weight-gradient uses a cached per-row inverse-RMS path instead of recomputing row RMS once per output column;
- transposed F32 matmul backward paths use register-blocked kernels for `A^T @ B` and `A @ B^T`.

Knobs:

```bash
# disable pooled storage for debugging
MOTIFCL_DISABLE_MEMORY_POOL=1

# cap cached OpenCL buffers
MOTIFCL_MEMORY_POOL_MAX_BLOCKS=256
```

On the local Ellesmere/RX 580 run, the 10.57M GPT benchmark improved from ~4.03 tok/s to ~5.37 tok/s at batch=1 after the first fast-path changes. After the topological backward scheduler, batch=2 Release step 2 improved to ~0.99 s / ~258 tok/s. After staged attention backward, cached RMSNorm weight-gradient, register-blocked transposed matmul, and coalesced transposed-tile loads, the latest checked batch=2 Release run reports warmed steps at ~0.056 s / ~4.53k tok/s.

Important boundary: this is not yet full production mixed-precision training. Missing pieces are FP32 master weights/loss scaling, FP16 training kernels with backward coverage, true multi-buffer fused Adam, fused MLP/residual/norm backward kernels, and dynamic kernel-argument rebinding for graph execution.

## Operator profiling

The runtime profiler can be enabled from C++ through `backend.profiler.set_enabled(true)` or from Python through:

```python
backend.profile_set_enabled(True)
backend.profile_clear()
# run workload
for row in backend.profile_summary():
    print(row.name, row.count, row.total_ms, row.avg_ms, row.max_ms)
```

The 10.57M Shakespeare GPT example also supports:

```powershell
$env:MOTIFCL_PROFILE=1
$env:MOTIFCL_PROFILE_TOP=30
build\release\examples\cpp\08_shakespeare_10m_train.exe build\datasets\tiny_shakespeare.txt 1 2
```

Before the topological backward scheduler, local profiled batch=2 step showed the real bottleneck was not Adam or transfer time:

- `multihead_attention_backward_flash_f32`: ~40.0 s over 146 launches;
- `rmsnorm_backward_weight_f32`: ~3.8 s over 512 launches;
- `matmul_flags_f32`: ~3.0 s over 1462 launches;
- uploads/downloads: ~0.12 ms total.

After the scheduler, the same profiled batch=2 step was:

- `multihead_attention_backward_flash_f32`: ~0.82 s over 3 launches;
- `rmsnorm_backward_weight_f32`: ~0.053 s over 7 launches;
- `matmul_flags_f32`: ~0.092 s over 38 launches;
- profiled GPU/transfer time: ~0.985 s total;
- wall time: ~3.77 s for first/warmup step and ~0.99 s for the second step without profiling.

After replacing the monolithic attention-backward kernel with the staged path for seq_len <= 128, the local profiled batch=2 step was:

- `attention_backward_probs_f32`: ~0.0058 s over 3 launches;
- `attention_backward_dp_f32`: ~0.0035 s over 3 launches;
- `attention_backward_apply_{q,k,v}_f32` + `attention_backward_ds_f32`: ~0.0014 s total over 12 launches;
- `matmul_flags_f32`: ~0.092 s over 38 launches;
- `rmsnorm_backward_weight_f32`: ~0.053 s over 7 launches;
- profiled GPU/transfer time: ~0.175 s total;
- warmed wall time without profiling: ~0.183 s/step, ~1.40k tok/s at batch=2.

The remaining high-impact targets are now matmul variants used by GPT projections and `rmsnorm_backward_weight_f32`.

After adding cached RMSNorm weight-gradient, register-blocked transposed matmul, and coalesced transposed-tile loads, the local profiled batch=2 step is:

- `matmul_transb_rb4_f32`: ~0.011 s over 19 launches;
- `matmul_register_block4_f32`: ~0.010 s over 19 launches;
- `attention_backward_probs_f32`: ~0.006 s over 3 launches;
- `matmul_transa_rb4_f32`: ~0.005 s over 19 launches;
- `multihead_attention_flash_f32`: ~0.005 s over 3 launches;
- `attention_backward_dp_f32`: ~0.003 s over 3 launches;
- `rmsnorm_backward_weight_cached_f32`: ~0.0007 s over 7 launches;
- profiled GPU/transfer time: ~0.048 s total;
- warmed wall time without profiling: ~0.056 s/step, ~4.53k tok/s at batch=2.

The remaining high-impact targets are now smaller: forward/backward matmul variants, attention forward/probs, and host-side first-step/program-build overhead.

## GPT-shape matmul microbench

Run:

```powershell
build\release\benchmarks\bench_matmul_gpt_shapes.exe 3
```

Latest local results on Ellesmere/RX 580:

| shape M,K,N | best F32 | FP16 accum F32 |
| --- | ---: | ---: |
| 256,512,2304 | default ~0.68 ms / ~889 GFLOP/s | ~1.26 ms / ~480 GFLOP/s |
| 256,512,512 | tile16 ~0.14 ms / ~953 GFLOP/s | ~0.92 ms / ~146 GFLOP/s |
| 256,512,256 | tile16 ~0.094 ms / ~715 GFLOP/s | ~0.73 ms / ~92 GFLOP/s |

Conclusion for this card/driver: current naive OpenCL FP16 matmul is slower than F32. Mixed precision should not be promoted as the default fast path until FP16 kernels are rewritten and benchmarked again.
