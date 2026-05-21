# Microkernel backend boundary

MotifCL keeps OpenCL as the stable default backend. Native CPU, ASM CPU, and
native GPU kernels should enter only through an explicit microkernel layer, not
by rewriting model or op code.

Current boundary:

- `include/motifcl/runtime/microkernel.hpp`
  - `MatmulBackend`
  - `AttentionBackend`
  - `QuantBackend`
  - backend kind: `OpenCL`, `Native`, `Asm`, `Vulkan`
  - stability: `Stable`, `Experimental`
  - descriptor registry: `microkernel_descriptors(domain, backend)`
  - availability gate: `microkernel_backend_available(domain, backend)`
  - typed selectors: `selected_matmul_backend()`,
    `selected_attention_backend()`, `selected_quant_backend()`,
    `selected_norm_backend()`, `selected_activation_backend()`
- `src/runtime/microkernel.cpp`
  - normalizes backend names;
  - exposes a concrete descriptor registry for the currently implemented
    OpenCL matmul/attention/quant/norm/activation families plus narrow opt-in
    Native/Vulkan slices;
  - reads per-domain env overrides;
  - resolves typed backend selections to available descriptors;
  - emits one warning per domain when an experimental/unknown backend is
    requested;
  - keeps OpenCL as the effective backend for unavailable experimental
    domains.

Opt-in selectors:

- `MOTIFCL_MATMUL_BACKEND=opencl|native|asm|vulkan`
- `MOTIFCL_ATTENTION_BACKEND=opencl|native|asm|vulkan`
- `MOTIFCL_QUANT_BACKEND=opencl|native|asm|vulkan`
- `MOTIFCL_NORM_BACKEND=opencl|native|asm|vulkan`
- `MOTIFCL_ACTIVATION_BACKEND=opencl|native|asm|vulkan`

The selectors are now touched by the real matmul, attention/KV-cache,
quantization, norm, and activation callsites. `native`, `asm`, and `vulkan` are
accepted only as explicit experimental requests. Implemented descriptor slices
become effective; unimplemented domains warn and fall back to OpenCL rather than
silently taking an invalid path.

The typed backend structs are intentionally real but conservative: OpenCL has
registered descriptors for the stable path. Native currently has narrow
implemented descriptors for opt-in host decode matmul:

- `native.matmul.f32_m1` — F32 x F32 M=1.
- `native.matmul.f32_q4_0_m1` — F32 x packed `Q4_0` M=1 with scalar
  `quant_scale`; tensors with external quant scale metadata intentionally fall
  back to the validated OpenCL path until native per-axis/per-block scale
  handling lands.

Their callable scalar/SIMD cores live behind
`include/motifcl/runtime/native_matmul.hpp` and
`src/runtime/native_matmul.cpp`. The current dispatch uses an SSE column-vector
kernel when the compiler target exposes SSE and keeps scalar baselines for
correctness/perf comparison, so later wider SIMD/ASM kernels can replace the
cores without changing Tensor/op dispatch. Other Native domains and all ASM
domains stay unavailable and descriptor-empty until a measured implementation
is landed. A new backend becomes selectable only after
`microkernel_backend_available(domain, backend)` returns true.

`Vulkan` is the native GPU backend line. It is intentionally separate from the
CPU `native`/`asm` path. The repository has a dependency-free Vulkan loader probe in
`include/motifcl/runtime/vulkan_backend.hpp` and
`src/runtime/vulkan_backend.cpp`; it dynamically loads `vulkan-1.dll` on
Windows or `libvulkan.so`/`libvulkan.so.1` on Linux, creates a minimal Vulkan
instance, enumerates physical devices, and exposes `motifcl_dump_vulkan_info`.
It also contains the first real Vulkan compute smoke path:
`run_vulkan_smoke_compute()` creates a device/queue, storage buffer,
descriptor set, shader module, compute pipeline, command buffer, submits a
tiny SPIR-V kernel, and verifies that the GPU wrote `42.0f` into host-visible
memory. `motifcl_dump_vulkan_info` prints this smoke result when a Vulkan
device is available. This keeps CI/builds independent of the Vulkan SDK while
proving whether the machine has a viable native-GPU Vulkan runtime and
execution path.

The Vulkan line also has a first fixed-shape F32 matmul smoke:
`run_vulkan_f32_matmul_smoke()` uses three storage buffers (`A`, `B`, `C`) and
an embedded SPIR-V compute pipeline to execute a `1x4 * 4x4 -> 1x4` multiply on
the GPU. It verifies the output `{90, 100, 110, 120}`. This is deliberately a
small correctness/data-path milestone, not yet a general matmul kernel.
The next step has also landed as a bounded specialized helper:
`run_vulkan_f32_m1_matmul(a, b, k, n)` validates `A=K`, `B=K*N`, bounds `K,N`
to a small smoke-safe specialization range, generates a tiny unrolled SPIR-V
kernel at runtime, and executes `1xK * KxN -> 1xN` through the same
storage-buffer compute path. The test suite exercises both malformed metadata
rejection and a `1x3 * 3x2 -> 1x2` GPU result `{22, 28}`.

`MOTIFCL_MATMUL_BACKEND=vulkan` now resolves to the narrow
`vulkan.matmul.f32_m1` and `vulkan.matmul.f32` descriptors. The real `matmul()`
callsite dispatches:

- F32 `M=1`, no-autograd, rank-2 tensors with `K,N <= 64` through
  `run_vulkan_f32_m1_matmul()`.
- General F32, no-autograd, rank-2 tensors with specialized `K <= 256` and
  exact-dispatch `M,N <= 4096` through `run_vulkan_f32_matmul()`.

The general path emits a tiny runtime SPIR-V compute shader that reads
`GlobalInvocationId`, maps one Vulkan work item to one output cell, uses
storage buffers for `A/B/C`, and dispatches exactly `N x M x 1`. If Vulkan
runtime/compute is unavailable the callsite falls back to the validated OpenCL
path; strict testing can set `MOTIFCL_REQUIRE_VULKAN_COMPUTE=1` or
`MOTIFCL_REQUIRE_VULKAN_MATMUL=1` to make that fallback fail loudly.

`MOTIFCL_ATTENTION_BACKEND=vulkan` now resolves to the narrow
`vulkan.attention.softmax_rows_f32` descriptor. The real `softmax_rows()`
callsite dispatches F32 rank-2 row softmax with `rows <= 4096` and
`cols <= 256` through `run_vulkan_softmax_rows()`. This emits a runtime SPIR-V
shader using `GlobalInvocationId`, `GLSL.std.450` `FMax`/`Exp`, one work item
per row, and storage buffers for input/output. If Vulkan compute is unavailable
the callsite falls back to OpenCL; strict testing can set
`MOTIFCL_REQUIRE_VULKAN_ATTENTION=1`.

`MOTIFCL_NORM_BACKEND=vulkan` now resolves to the narrow
`vulkan.norm.rmsnorm_f32` descriptor. The real `rmsnorm()` callsite dispatches
F32 rank-2 input plus F32 rank-1 weight with `rows <= 4096`, `cols <= 256`,
finite positive `eps`, same backend, and no autograd through
`run_vulkan_rmsnorm()`. The runtime SPIR-V shader uses one work item per row,
storage buffers for input/weight/output, and `GLSL.std.450` `InverseSqrt` for
the RMS reciprocal. If Vulkan compute is unavailable the callsite falls back to
OpenCL; strict testing can set `MOTIFCL_REQUIRE_VULKAN_NORM=1`.

`MOTIFCL_ACTIVATION_BACKEND=vulkan` now resolves to the narrow
`vulkan.activation.swiglu_f32` descriptor. The real `swiglu()` callsite
dispatches F32 rank-2 packed `[rows, 2*hidden]` inputs with `rows <= 4096`,
`hidden <= 4096`, and no autograd through `run_vulkan_swiglu()`. The runtime
SPIR-V shader maps one work item to one output element, computes
`gate / (1 + exp(-gate)) * up`, and writes `[rows, hidden]`. If Vulkan compute
is unavailable the callsite falls back to OpenCL; strict testing can set
`MOTIFCL_REQUIRE_VULKAN_ACTIVATION=1`.

This is still not a full transformer replacement. Vulkan Attention/Quant
full GQA/paged-KV kernels, LayerNorm/backward norm kernels, remaining
activation/unary kernels, Quant domains, autograd, packed quant weights, and
large/perf-gated production shapes remain descriptor-empty or fall back to
OpenCL until validated storage-buffer inputs, shape/stride contracts, and
correctness/perf gates land against the matching OpenCL kernels.

Policy for new fast paths:

1. Keep OpenCL as the fallback.
2. Add backend descriptors for the exact domain/backend pair.
3. Native/ASM/Vulkan paths are opt-in until they have correctness and
   performance evidence.
4. Add a per-callsite validator before every kernel launch.
5. Add correctness coverage to CTest/pytest.
6. Capture baseline and candidate performance artifacts.
7. For Vulkan GPU paths: add the SPIR-V/compute pipeline plus descriptor-set
   validation under the Vulkan backend, then compare it against the matching
   OpenCL kernel on the same tensor shape.
8. Run `tools/perf_truth_gate.py baseline.json candidate.json ...`.
9. If candidate is slower or unstable, dispatch must fall back to OpenCL.

This lets us add custom kernels incrementally while preserving the existing
runtime contract and release gate.
