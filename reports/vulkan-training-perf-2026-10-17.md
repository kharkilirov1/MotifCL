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
| **after (this work, defaults)** | VRAM + pool + tiled update + small-buffer policy | **357–365** | **5 666–5 701** |
| dense control HIDDEN=768, after | same | 233 | 8 800 |

120-step steady-state run (defaults, `GPU_TIMING=1`), steps 30/60/90/120 all
`5 179 / 5 701 / 5 666 / 5 678 tok/s` at 357–365 ms per step with device memory flat at
3390/3406 MB live/peak and loss falling 6.31 → 5.65 → 4.67 → 4.56 (final val loss 4.4564,
ppl 86.2 at 120 steps × 2048 tokens). vs the original default configuration that is
**~8.9x**; vs the best configuration previously recorded in
`docs/MOE_PRETRAIN_RESULTS.md` (2018 tok/s with `MOTIFCL_VK_PREFER_DEVICE_LOCAL=1`) it is
**2.8x**.

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

## Challenge pass: double-buffered GEMM kernels (2026-10-18)

Second optimization pass, same device and harness. The first pass removed the
structural waste; this pass attacks the kernels themselves and the measured
absolute utilization.

### Where the time was (per MoE step, GPU, after the first pass)

| block | ms/step | achieved | % of RX 580 FP32 peak |
|---|---:|---:|---:|
| baseline (first pass) | 291-294 | 0.93 TFLOP/s | 15% |
| LM-head GEMMs (nn/nt/tn) | 99 | 1.14 TFLOP/s | 18% |
| counter projections (fwd+bwd) | 77 | ~1.5 TFLOP/s | 24% |
| counter grad_w GEMM | 30 | 1.2 TFLOP/s | 20% |
| GQA attention (fwd + 2 bwd) | 68 | 0.18 TFLOP/s | 3% |

### What changed

1. **Double-buffered register-blocked GEMM** (`kernels/vulkan/mm_f32_db_body.glsl`
   plus the 32x32/64-thread `mm_f32_db32.comp` and 64x64/256-thread
   `mm_f32_db64.comp` specializations). One body covers all three operand
   layouts (A*B, A*B^T, A^T*B). Two changes mattered:
   * the K loop walks 16-deep tiles through a two-buffer LDS ring and loads the
     next tile into registers **before** consuming the current one, so global
     latency and decode arithmetic overlap the FMA loop instead of stalling the
     workgroup at each barrier;
   * A and B are viewed as `vec4` arrays and read with `LDS.128`, so each k step
     is two vector LDS reads for sixteen FMA (the old kernels issued eight scalar
     reads for the same sixteen FMA and were LDS-issue bound).
   Tile selection is measured, not guessed: 64x64 when there are at least ~96
   output tiles *and* the shape is wide (n >= 384) or deep (k >= 1024), else
   32x32 (`MOTIFCL_VK_MATMUL_DB_TILE` / `MOTIFCL_VK_MATMUL_VARIANT` override for
   A/B runs).
2. **Double-buffered counter decode + GEMM** (`kernels/vulkan/counter_gemm_db.comp`).
   One kernel serves both training projections (forward `y = x*W` and
   backward-input `grad_x = grad_out*W^T`) with a mode flag; the packed 6-bit
   state is decoded into registers for the next K tile while the current tile is
   being multiplied. This replaces the two scalar-operand kernels that measured
   ~24% of peak.
3. **Tooling**: `benchmarks/bench_vulkan_gemm_shapes` measures the exact training
   shapes (nn/nt/tn) with GPU timestamps and reports TFLOP/s against the 6.17
   TFLOPS FP32 peak, and `tests/test_vulkan_gemm_parity` checks all three layouts
   against a host reference (the pre-existing Vulkan matmul parity smoke needs
   OpenCL tensors and skips on this box).

### Kernel-level result (bench_vulkan_gemm_shapes, RX 580)

| shape | op | before | after | % peak after |
|---|---|---:|---:|---:|
| 2048x288x32000 (LM head fwd) | nn | 24.17 ms / 25.3% | **9.81 ms** | **62.4%** |
| 2048x32000x288 (LM head dA) | nt | 40.28 ms / 15.2% | **14.6 ms** | **41.8%** |
| 288x2048x32000 (LM head dB) | tn | 23.59 ms / 25.9% | **10.6 ms** | **57.9%** |
| 2048x288x384 (expert up) | nn | 0.331 ms / 22.2% | **0.144 ms** | **51.2%** |
| 2048x192x288 (expert down) | nn | 0.168 ms / 21.8% | **0.078 ms** | **47.2%** |
| 2048x288x288 (attn qkv) | nn | 0.249 ms / 22.1% | **0.111 ms** | **49.8%** |
| 2048x288x768 (dense up) | nn | 0.652 ms / 22.5% | **0.260 ms** | **56.5%** |
| 384x2048x288 (counter grad_w) | tn | 0.370 ms / 19.9% | **0.246 ms** | 29.9% |

Everything but the counter grad_w shape is now in the 42-62% band; the head
projection (34% of the modelled step FLOPs when counting fwd+bwd) sits at 62.4%.

### Step-level result

| arm | ms/step | tok/s |
|---|---:|---:|
| first pass (end of 2026-10-17 work) | 357-365 | 5666-5701 |
| **this pass** | **289** | **~7000** (measured 6969 at step 30; 30-step steady state) |
| GPU fwd+bwd | 294 -> **222** | |
| LM-head GEMM block | 99 -> **37** | |
| counter projections | 77 -> **62** | |
| counter grad_w | 30 -> **29** | |
| GQA attention | 68 -> **61** | |
| dense control (hidden 768) | 233 -> **215** | 8800 -> **9542** |

Correctness: full CTest **51/51**; `test_vulkan_counter_fused` still demands and
gets an exact state/scale/v match, `test_vulkan_memory_native` still reproduces
the recorded convergence witness (7.60164 -> 0.0642764), and the new
`test_vulkan_gemm_parity` covers all 13 shape/layout combinations including
partial tiles, partial K tiles and the odd-dimension scalar fallback.

### Remaining headroom (measured, not estimated)

* **GQA attention is now the worst block**: 61 ms/step at ~3% of peak. The
  kernels run one 64-lane workgroup per (batch, head, query row) and therefore
  re-read the whole K/V history for every query row (per layer ~600 MB of
  redundant K/V traffic). The fix is tiling several query rows per workgroup with
  K/V tiles resident in LDS (the same structure that produced the GEMM win);
  expected 61 -> ~20 ms.
* **Counter grad_w (29 ms/step, 30% peak)** is bandwidth-bound because the output
  tile count is small (108 workgroups) while K = 2048: it re-reads both operands
  `n/32` and `m/32` times. A 64x32 output tile keeps the same wave count while
  cutting operand traffic ~25%; a larger redesign would need k-splitting.
* **Counter decode (62 ms/step)** still reads packed state bytes with scattered
  1-byte loads (32 B sector per 1-3 useful bytes). Staging each out row's 12
  contiguous state bytes cooperatively into LDS before decode should recover
  another 10-15 ms.
* **CE backward (5.0 ms) and the elementwise/norm block (~20 ms)** are un-fused;
  folding SwiGLU and residual-add epilogues into the neighbouring GEMMs and
  chunking the vocabulary CE would remove most of that plus ~200 dispatches.
* **CPU recording is 48-51 ms/step** (~17%) with ~700 dispatches; the measured
  Vulkan fast-path dispatch is ~15 us, so most of that is autograd/Tensor CPU
  work rather than command recording. Needs a CPU-side scope profile before
  more kernel work.
* **Sparse MoE backward** would remove ~35-40 ms of discarded expert work, but
  today each expert owns separate state/scale buffers and the runtime binds
  descriptors at offset 0, so one segmented dispatch cannot index per-expert
  weights: it needs either a shared expert-state buffer or descriptor-offset
  support first. That is a refactor, not a kernel tweak.

## Challenge round 2: tiled attention and counter decode (2026-10-18)

Third pass, same device/harness. Two of the five remaining items landed; the
other three are measured and characterised below rather than guessed at.

### Item 1: GQA attention, 61 -> 25 ms/step

The general attention kernels run one 64-lane workgroup per (batch, head, query
row), so the whole K/V history is streamed again for every query row (~600 MB of
redundant traffic per layer, 3% of FP32 peak). Three specializations now cover
the trainer's shape (causal, no mask, query_offset 0, sliding_window 0,
key == query length <= 256, head_dim == v_head_dim <= 64); the general kernels
stay as the fallback and every attention parity test still passes.

| kernel | before | after | |
|---|---:|---:|---|
| `gqa_fwd_general_f32_subgroup` -> `gqa_fwd_causal4_f32` | 2.99 ms | 1.15 ms | 4 query rows/workgroup, one K load per key for four dots, vec4 subgroup max/sum, V reused across rows |
| `gqa_bwd_probs_f32_subgroup` -> `gqa_bwd_probs_causal4_f32` | 5.42 ms | 1.94 ms | same row tiling; dp formed once per key for four rows; dQ reuses K across rows |
| `gqa_bwd_dkv_f32` -> `gqa_bwd_dkv_causal_tile_f32` | 1.72 ms | 1.02 ms | one 64-key tile/workgroup, one lane per key (coalesced ds/p, broadcast q/dout), constant-indexed vec4 accumulators to avoid spills |
| **attention block per step** | **61 ms** | **25 ms** | |

### Item 3: packed-state decode, 62 -> 55 ms/step

The counter projections decode the 6-bit state inside the GEMM. Each ternary
code went through a branch chain (`ternary_from_code`) per weight; a 64-entry
ternary table built once per workgroup in shared memory replaces it. Measured
0.257 -> 0.228 ms per projection (-11%), 62 -> 55 ms per step. The remaining
decode cost is dominated by the scattered one-byte packed-state reads (a 32 B
sector per 1-3 useful bytes); staging each out row's 12 contiguous bytes
cooperatively into LDS is the next step there.

### Items 2, 4, 5: measured, not yet taken

* **Counter grad_w (29 ms/step, 30% peak)** is bandwidth-bound: the output tile
  count is small (108 workgroups) while K = 2048, so both operands are re-read
  `n/32` and `m/32` times (56 MB per layer at ~230 GB/s). Fixing it needs a
  rectangular 64x32 tile specialization to cut the traffic ~25% or a k-split
  with a reduction pass; both are measured-plan items, not kernel tweaks.
* **Elementwise/CE block (~20 ms/step)**: the residual `add_f32` is already
  bandwidth-bound (96 calls, 0.04 ms each at ~76-100 GB/s), so vectorization
  alone buys little; the win is fusion (add+rmsnorm, SwiGLU epilogue in the
  counter GEMM, chunked vocabulary CE), which needs new fused kernels.
* **CPU recording is 48-54 ms/step (20%)**, now split by the new instrumentation
  into **forward 18.4 ms / backward 31.5 ms** plus balance 8 ms and optimizer
  6 ms. The measured Vulkan fast-path dispatch is ~15 us, so ~700 dispatches are
  only ~11 ms of that: the bulk is autograd/Tensor CPU work (node graph, grad
  accumulation, per-op allocations). Attacking it means either fusing ops (fewer
  nodes) or a Tensor/node freelist plus a leaner tape walk.

### Round result

| arm | ms/step | tok/s |
|---|---:|---:|
| first pass (2026-10-17) | 357-365 | 5666-5701 |
| GEMM/kernel pass | 289 | ~7000 |
| **this pass** | **239** | **8500** |
| dense control (hidden 768) | 233 -> **165** | 9542 -> **12380** |

GPU fwd+bwd 222 -> 175 ms. Full CTest 51/51 with the counter exactness witness
and the recorded convergence witness intact (7.60164 -> 0.0642764).
