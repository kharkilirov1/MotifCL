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

4. Keep a checked-in or artifact-stored baseline JSON per device/driver. `baselines/ellesmere.json` is the RX 580/Ellesmere seed baseline. Treat regressions above 10-15% as suspicious unless explained by driver/device changes.

```bash
python tools/perf_regression.py motifcl_tuning.json baselines/ellesmere.json \
  --tolerance 0.15 \
  --min-metrics 9 \
  --require-key matmul_q8_256_ms \
  --require-key matmul_q4_256_ms \
  --require-field metadata.device_family \
  --require-baseline-field metadata.profile
```

`--min-metrics` and repeated `--require-key` make the script useful as a CI
gate instead of only a best-effort comparison: missing critical Q4/Q8 metrics now
fail fast even if the JSON file is otherwise syntactically valid. Metadata gates
(`--require-field`, `--require-baseline-field`, `--match-field`, and
`--require-field-contains`) let a release check reject results captured on the
wrong device/driver before comparing timings.

For the full local release sweep, use:

```bash
python tools/release_check.py --wheel --hf-run --require-clean-git
```

Drop `--wheel`, `--hf-run`, or `--require-clean-git` for faster local iteration.

Current portable kernels are OpenCL C and do not include hand-written vendor ISA. Vendor-specific optimization should be added behind capability gates and backed by correctness/perf regression tests.

## Fast training mode currently implemented

The current fast path is still FP32/eager, but avoids several prototype bottlenecks:

- output tensors in hot GPU ops use `Tensor::empty` instead of CPU-zero-uploaded `Tensor::zeros`;
- scalar losses reduce on GPU via `mean_reduce_f32` instead of synchronously downloading partial losses;
- `Storage` uses an OpenCL buffer memory pool by default;
- Adam uses `adam_update_f32_fast`, which precomputes bias-correction scalars once per parameter update instead of calling `pow()` per element in the kernel; `optim::Adam` now groups up to four parameter buffers into `adam_update_f32_fast4` and supports decoupled AdamW-style `weight_decay`;
- MLP uses fused `add_bias_gelu_rows_f32` for the first projection bias+GELU forward path;
- RMSNorm residual backward has a fused helper kernel (`rmsnorm_backward_x_residual_{wg,}f32`) for the common pre-norm residual fan-in case;
- Modern SwiGLU transformer MLP blocks without bias/dropout can use `fused_swiglu_mlp_rmsnorm_residual` when `MOTIFCL_ENABLE_FUSED_MLP_BACKWARD=1`; it is a custom autograd node that keeps the branch as one backward unit and fuses `grad_out @ down_weight^T` with SwiGLU backward via register-blocked `swiglu_down_backward_packed_f32`;
- `MOTIFCL_ENABLE_HIGH_LEVEL_MLP_FUSION=1` additionally enables the higher-level experimental path: RMSNorm row inverse is cached, `rmsnorm_matmul_rb4_f32` computes the gate/up projection without materializing the normalized activation, and backward reuses the cached row inverse through `rmsnorm_matmul_transa_rb4_f32` plus cached RMSNorm residual/weight kernels;
- FP16 foundation is available for inference-style experiments: `cast_f32_to_f16`, `cast_f16_to_f32`, `matmul_f16_accum_f32`, and a `matmul_f16_accum_f32_vec4` path for `K % 4 == 0` when the OpenCL device exposes `cl_khr_fp16`;
- mixed-precision training infrastructure now includes `DynamicLossScaler` plus `MixedPrecisionAdam` with FP32 master weights, grad unscale/overflow checks, and F16<->F32 parameter synchronization;
- `GraphExecutor` wraps captured replayable graphs with runtime planning metadata, execution counters, same-shape `cl_mem` rebinding through `GraphExecutor::bind_tensor(captured_tensor_id, runtime_tensor)`, and an extension-gated driver command-buffer path;
- `Backend::command_buffer_mode()` reports `cl_khr_command_buffer_mutable_dispatch`, `cl_khr_command_buffer_replay`, or `host_replay_fallback`; when the ICD exposes `cl_khr_command_buffer`, pure captured kernel graphs are recorded/finalized once and replayed with `clEnqueueCommandBufferKHR`, and mutable dispatch uses `clUpdateMutableCommandsKHR` to update rebound buffer arguments;
- autograd uses a topological backward scheduler, accumulating all fan-in gradients before executing each backward node once;
- GPT-sized attention backward uses a staged softmax-gradient path (`P`, `dP`, `dS`, then `dQ/dK/dV`) for seq_len <= 128, avoiding the previous repeated softmax recomputation inside `dK/dV`.
- RMSNorm weight-gradient uses a cached per-row inverse-RMS path instead of recomputing row RMS once per output column;
- transposed F32 matmul backward paths use register-blocked kernels for `A^T @ B` and `A @ B^T`.
- unscaled Q4_0 matmul selects an integer dot4-unrolled register-blocked kernel when `K % 4 == 0`;
- scalar losses reduce partial buffers on GPU with separate `partial_count` and denominator arguments, avoiding the old MSE partial-read overrun when `numel != partial_chunks`.

Knobs:

```bash
# disable pooled storage for debugging
MOTIFCL_DISABLE_MEMORY_POOL=1

# cap cached OpenCL buffers
MOTIFCL_MEMORY_POOL_MAX_BLOCKS=256

# profiler-guided FlashAttention compile-time knobs
MOTIFCL_FA_TILE=8|16|32
MOTIFCL_FA_WG=64|128|256
```

On the local Ellesmere/RX 580 run, the 10.57M GPT benchmark improved from ~4.03 tok/s to ~5.37 tok/s at batch=1 after the first fast-path changes. After the topological backward scheduler, batch=2 Release step 2 improved to ~0.99 s / ~258 tok/s. After staged attention backward, cached RMSNorm weight-gradient, register-blocked transposed matmul, and coalesced transposed-tile loads, the latest checked batch=2 Release run reports warmed steps at ~0.056 s / ~4.53k tok/s.

Important boundary: this is still not a fully FP16-covered training stack. Implemented pieces are same-shape captured-kernel `cl_mem` rebinding, driver-level command-buffer record/replay/update when the OpenCL ICD exposes `cl_khr_command_buffer(_mutable_dispatch)`, grouped multi-buffer Adam/AdamW updates, vectorized FP16 inference matmul selection, FP32-master/loss-scaling optimizer infrastructure, a fused RMSNorm+residual backward helper, experimental env-gated fused SwiGLU/down-input backward for biasless modern MLP blocks, Q4 dot4-unrolled unscaled matmul, and env-gated FlashAttention tile/workgroup rebuilds. Remaining pieces are an optimized register-blocked fused MLP backward kernel spanning the down-input GEMM, a literal monolithic transformer-block kernel spanning both MLP matmul weight-gradient reductions, broad FP16 backward kernels for all training ops, and external RGP/driver-counter tuning.

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

Conclusion for this card/driver: the old naive OpenCL FP16 matmul was slower than F32. A `half4`/`float4` vectorized path now exists for `K % 4 == 0`; rerun this benchmark before promoting FP16 as a default fast path. Mixed-precision optimizer infrastructure now exists, but full mixed-precision training still requires FP16 forward/backward coverage for the whole model graph.
