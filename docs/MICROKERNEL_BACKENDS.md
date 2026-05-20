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
    `selected_attention_backend()`, `selected_quant_backend()`
- `src/runtime/microkernel.cpp`
  - normalizes backend names;
  - exposes a concrete descriptor registry for the currently implemented
    OpenCL matmul/attention/quant families;
  - reads per-domain env overrides;
  - resolves typed backend selections to available descriptors;
  - emits one warning per domain when an experimental/unknown backend is
    requested;
  - keeps OpenCL as the effective backend until a native/ASM implementation is
    actually wired and benchmark-gated.

Opt-in selectors:

- `MOTIFCL_MATMUL_BACKEND=opencl|native|asm|vulkan`
- `MOTIFCL_ATTENTION_BACKEND=opencl|native|asm|vulkan`
- `MOTIFCL_QUANT_BACKEND=opencl|native|asm|vulkan`

The selectors are now touched by the real matmul, attention/KV-cache, and
quantization callsites. `native`, `asm`, and `vulkan` are accepted only as
explicit experimental requests; this build warns and falls back to OpenCL rather
than silently taking an unimplemented path.

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
CPU `native`/`asm` path: selecting `MOTIFCL_MATMUL_BACKEND=vulkan` currently
falls back to OpenCL with an explicit warning until a Vulkan compute descriptor
lands. The repository now has a dependency-free Vulkan loader probe in
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

This is still not a matmul replacement. Vulkan remains unavailable as a
Matmul/Attention/Quant descriptor until the next layer lands: validated
storage-buffer inputs, shape/stride contracts, and correctness/perf gates
against the matching OpenCL kernels.

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
