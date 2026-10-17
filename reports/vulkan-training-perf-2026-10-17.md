# Vulkan FP32 training throughput — RX 580, 2026-10-17

Measured, same host, same data, same binaries family. This record documents the
training-speed work and the evidence behind each number.

## Device and stack

| | |
|---|---|
| GPU | Radeon RX 580 Series (Polaris 20 / Ellesmere / gfx803), 8 GB |
| Driver | AMD proprietary Adrenalin 31.0.21925.1001 (2026-05-20), `driverInfo "26.5.2"` |
| Vulkan ICD | AMDVLK `amdvlk64.dll` 9.2.10.279, apiVersion 1.3.260 (no RADV/Mesa on the box) |
| Shader compiler | AMDLLPC / LLVM 17.0.0, target table `POLARIS10:gfx803` |
| Runtime | `Backend::create_vulkan()`, FP32 only (shaderFloat16 = false) |
| Build | `cmake --build build/release` (GCC 13.2, Release), CTest 50/50 |

Device facts that shaped the work (read from `vulkaninfo` on this box):
`subgroupSize = 64` (wave64 only), `maxComputeSharedMemorySize = 32768`,
`timestampPeriod = 40` ns, `timestampValidBits = 64`, `maxMemoryAllocationSize = 2 GiB`,
only 256 MiB of device-local memory is host-visible (no resizable BAR), 7.75 GiB VRAM heap,
`shaderFloat16 = false`, `shaderBufferFloat32AtomicAdd = false` (no float atomicAdd).

## Measurement protocol

Model: `mn-moe15m` (`tools/train_moe15m.cpp`) and its dense control
(`tools/train_ternary15m.cpp`): dim 288, 6 layers, 6 heads, vocab 32000, tied/untied LM
head, all linear layers are `CounterStateLinear` (finite-state 6-bit counter synapses),
B = 8 × T = 256 = 2048 tokens/step TinyStories. Wall time is the runner's own
`steady_clock` window average printed as `tok/s`; GPU time is the batch submission's
`VK_QUERY_TYPE_TIMESTAMP` delta (`GPU_TIMING=1`).

## Headline

| arm | config | ms/step | tok/s |
|---|---|---:|---:|
| before (HEAD, default config) | host-visible/GTT buffers | 3 224 | 635 |
| before (HEAD, `MOTIFCL_VK_PREFER_DEVICE_LOCAL=1`) | VRAM | 1 443 | 1 419 |
| **after (this work, defaults)** | VRAM + pool + tiled update + small-buffer policy | **370–424** | **4 787–5 396** |
| dense control HIDDEN=768, after | same | 233 | 8 800 |

MoE steady-state windows: step 20 `4787 tok/s` (369.0 ms), step 30 `5396 tok/s` (423.9 ms),
step 40 `5247 tok/s` (370.1 ms). vs the original default configuration that is **~8.3x**;
vs the best configuration previously recorded in `docs/MOE_PRETRAIN_RESULTS.md`
(2018 tok/s with `MOTIFCL_VK_PREFER_DEVICE_LOCAL=1`) it is **2.6x**.

Loss trajectory is unchanged for the same budget: at the 20-step mark this run reports
step 20 loss 6.3836 / ema 9.7998 against 6.4081 / 9.8034 for the pre-change run of the
same command, and the 4-bit-exactness / convergence witnesses below are bit-identical
where the arithmetic order was preserved.

## Root causes found, in order of measured impact

### 1. Default memory placement put every activation in system RAM (2.2x)

`VulkanRuntime::create_buffer` preferred `DEVICE_LOCAL|HOST_VISIBLE` (a 256 MiB BAR
aperture) and then `HOST_VISIBLE|HOST_COHERENT` (GTT), and only tried pure VRAM last — so
training activations lived in system memory and every dispatch read/wrote over PCIe. The
pre-FOG ordering (VRAM first, staged transfers) was available only through the opt-in
`MOTIFCL_VK_PREFER_DEVICE_LOCAL=1`.

Measured A/B on the same binary, 60 steps, bit-identical losses: 635 tok/s (default) vs
1 419 tok/s (device-local), i.e. 2.2x. **Fix:** device-local is now the default and
allocation still falls back BAR → host memory when VRAM is full, so oversized models
degrade instead of failing. `MOTIFCL_VK_PREFER_DEVICE_LOCAL=0` restores the old policy.

### 2. The buffer pool cap was smaller than the training working set (~19x on the CPU phase)

The buffer pool was capped at 2 GiB / 2048 allocations while the MoE trainer's live peak is
~3.4 GiB, so every step destroyed and re-allocated the vocabulary-logit buffers
(2048 × 32000 × 4 B = 262 MB each) through `vkAllocateMemory`. The step-phase
instrumentation showed the forward+backward *recording* phase at **960 ms/step** with the
GPU phase only 650-720 ms: the CPU was blocked in allocation.

**Fix:** pool cap 6 GiB / 4096 allocations / 256 per bucket. Measured: recording phase
960 ms → 50 ms; step 1 919 ms → 758 ms in the same profile build.

### 3. The counter update recomputed the grad_w GEMM twice with strided reads (50% of GPU time)

Per-op GPU profile before (`MOTIFCL_VK_PROFILE=1`), 120 counter layers/step:

| kernel | ms/layer | share of step GPU |
|---|---:|---:|
| `counter_row_stats_fused_f32_subgroup` | 1.7–1.8 | 29–30% |
| `counter_apply_update_fused_f32` | 1.1–1.2 | 20% |
| `counter_forward_f32_rb4` | 0.40 | 7.5% |
| `counter_backward_input_f32_rb4` | 0.40 | 7.4% |

Both update kernels recomputed `grad_w = grad_outᵀ·x` (one workgroup per output row,
`grad_out[r·out + row]` strided by `out_features` = one 4-byte element per 128-byte line)
and the forward GEMM of the same shape runs 4x faster. That is ~50% of a step's GPU time
spent on two passes of a GEMM that can be done once.

**Fix (new kernel `kernels/vulkan/counter_update_apply_f32.comp` + tiled grad_w):** the
gradient is formed once by the existing transposed register-blocked GEMM
(`matmul_transpose_a`, 32x32 tile / 4x4 per lane) into a **reusable transient** grad_w
scratch, then one wave64 workgroup per output row folds the row statistics, commits
`v`/`scale`, and runs the stochastic-rounding tick. At save it also folds the previous
separate row-scale copy dispatch. The layer's *persistent* cost is unchanged (packed
6-bit state + O(out) scale/v); layers whose dense gradient would exceed 32 MiB keep the
old strictly allocation-free recompute path, so the 1B-class probes do not grow.

Measured after: `mm_f32_tn` (grad_w) **0.28 ms**/layer + `counter_update_apply_f32`
**0.02 ms**/layer = 0.30 ms vs 2.9 ms → **9.7x** on the update, GPU step 653 ms → 287 ms.

Arithmetic order is preserved: grad_w accumulates k sequentially, and the row statistics
keep the previous per-lane group order + `subgroupAdd`, so
`tests/test_vulkan_counter_fused` still demands and gets an **exact** state/scale/v match,
and `tests/test_vulkan_memory_native` reproduces the recorded convergence witness
first_loss 7.60164 → last_loss 0.0642764 exactly.

### 4. Small host↔device transfers cost a staging submit + fence each (~2-4 ms)

With device-local storage, every `Tensor::from_cpu` / `to_vector` on a small tensor runs a
staging command buffer: `vkQueueSubmit` + `vkWaitForFences`. The MoE step performs ~24 such
transfers (router logits download per layer, balance-gradient upload, per-expert bias
download/upload, loss scalar, token ids). Measured balance phase 40–160 ms and loss sync
2-8 ms per step.

**Fix:** buffers ≤ 128 KiB are allocated host-visible by default
(`MOTIFCL_VK_SMALL_HOST_BYTES` overrides; 0 disables), so those transfers are a memcpy; the
PCIe cost of a few tens of KiB inside a kernel is far below one fence wait. Measured:
balance phase 40–160 ms → **8.5–13.8 ms**, sync 2.3 ms → 0.03 ms, upload 4.9 ms → 0.1 ms.

## Instrumentation added (kept)

- `VulkanRuntime::set_gpu_timing_enabled()` / `last_gpu_time_us()` existed; the runtime now
  also reports `last_submit_time_us()` and `submit_count()` (wall time inside
  `vkQueueSubmit` + `vkWaitForFences`), so GPU work and submit/fence latency are separable.
- **Per-kernel GPU profiler**: `MOTIFCL_VK_PROFILE=1` records one timestamp per dispatch
  inside the batch plus one after the last, accumulates GPU time per embedded kernel and
  reports it through `VulkanRuntime::kernel_profile()` (sorted by total) /
  `reset_kernel_profile()`. Kernel names come from the generated
  `src/runtime/vulkan_spirv_names.inc` (`tools/gen_vulkan_spirv_names.py`); the pointer →
  name table is derived from `vulkan_spirv_kernels.inc`.
- Trainers: `GPU_TIMING=1` prints per-step `ms fwd+bwd gpu / opt gpu / sync` and the
  `upload / record / submit / balance / optim / total` phase split; `MOTIFCL_VK_PROFILE=1`
  additionally prints the top-20 kernels by GPU time per log window.
- `tools/gen_vulkan_spirv.py` now writes the generated header as UTF-8.

## Post-change GPU profile (MoE, per step, 287-330 ms total)

| kernel | ms/step | note |
|---|---:|---|
| counter_forward_f32_rb4 ×120 | 36 | decode-in-GEMM forward |
| counter_backward_input_f32_rb4 ×120 | 36 | grad_x from packed state |
| mm_f32_tn (grad_w) ×120 | 34 | tiled transposed GEMM |
| head GEMMs (nn/nt/tn _rb4) | ~91 | 288 → 32000 vocabulary projection + backward |
| GQA fwd/bwd (3 kernels) | ~65 | MHA via the general GQA path |
| counter_update_apply_f32 ×120 | 2.3 | finalize + tick |
| everything else | ~50 | elementwise, norms, rope, swiglu, Adam |

## Remaining opportunities (not done here)

1. **Sparse MoE expert execution.** `MoE7FFN::forward` computes all 7 experts for all 2048
   rows and then keeps the routed winner per row, so 7-of-8 expert GEMMs are discarded —
   26% of the step's modelled FLOPs. It needs a device-side argmax/count + row gather, a
   segmented counter GEMM, and a scatter-add. The routing decision only depends on the
   normalized input, so it can run before the experts. `glslc` is available on this box
   (Android NDK shader-tools), so this is buildable; `docs/MOE_PRETRAIN_RESULTS.md` records
   the 2.6x MoE-vs-dense wall-time penalty that this would remove.
2. **Head projection.** The 288 → 32000 LM head is ~40% of the modelled FLOPs and runs at
   ~1.0-1.4 TFLOP/s. A vocabulary-chunked fused cross-entropy would cut the 262 MB logits
   and grad-logits working set (and the bucket pressure) but needs a new kernel.
3. **Dispatch recording.** `record` is still 55-190 ms/step (~770 dispatches): descriptor
   set re-acquire + `vkUpdateDescriptorSets` per dispatch and a global memory barrier
   between every dispatch. Caching descriptor sets by buffer tuple and using per-buffer
   barriers would attack both; `VK_KHR_push_descriptor` and `VK_EXT_descriptor_buffer` are
   advertised by this ICD.
4. **Driver-side introspection.** `VK_KHR_pipeline_executable_properties` and
   `VK_AMD_shader_info` are advertised, which would expose per-kernel VGPR/LDS/spill
   statistics from AMDLLPC — the next step for tuning the remaining GEMMs.

## Reproduce

```powershell
cmake --build build/release --target motifcl_train_moe15m motifcl_train_ternary15m -j 10

# MoE, throughput + GPU time + phase split
$env:DATA_DIR="data/tinystories"; $env:OUT_DIR="out/perf/moe"
$env:STEPS="40"; $env:BATCH="8"; $env:SEQ="256"; $env:LOG_EVERY="10"
$env:EVAL_EVERY="100000"; $env:CKPT_EVERY="100000"; $env:SAVE_CKPT="0"; $env:RESUME="0"
$env:GPU_TIMING="1"
build\release\tools\motifcl_train_moe15m.exe

# same run with the per-kernel GPU profile
$env:MOTIFCL_VK_PROFILE="1"; build\release\tools\motifcl_train_moe15m.exe

# dense control at equal active width (hidden 768)
$env:HIDDEN="768"; build\release\tools\motifcl_train_ternary15m.exe

# correctness gates
ctest --test-dir build/release --output-on-failure
```

Regenerate shaders and the profiler name table after editing `kernels/vulkan/*.comp`:

```bash
python tools/gen_vulkan_spirv.py        # needs glslc (MOTIFCL_GLSLC or Android NDK)
python tools/gen_vulkan_spirv_names.py
```
