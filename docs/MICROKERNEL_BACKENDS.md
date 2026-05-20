# Microkernel backend boundary

MotifCL keeps OpenCL as the stable default backend. Native/ASM kernels should
enter only through an explicit microkernel layer, not by rewriting model or op
code.

Current boundary:

- `include/motifcl/runtime/microkernel.hpp`
  - `MatmulBackend`
  - `AttentionBackend`
  - `QuantBackend`
  - backend kind: `OpenCL`, `Native`, `Asm`
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

- `MOTIFCL_MATMUL_BACKEND=opencl|native|asm`
- `MOTIFCL_ATTENTION_BACKEND=opencl|native|asm`
- `MOTIFCL_QUANT_BACKEND=opencl|native|asm`

The selectors are now touched by the real matmul, attention/KV-cache, and
quantization callsites. `native` and `asm` are accepted only as explicit
experimental requests; this build warns and falls back to OpenCL rather than
silently taking an unimplemented path.

The typed backend structs are intentionally real but conservative: OpenCL has
registered descriptors, while Native/ASM stay unavailable and descriptor-empty
until a measured implementation is landed. A new backend becomes selectable
only after `microkernel_backend_available(domain, backend)` returns true.

Policy for new fast paths:

1. Keep OpenCL as the fallback.
2. Add backend descriptors for the exact domain/backend pair.
3. Native/ASM path is opt-in until it has correctness and performance evidence.
4. Add a per-callsite validator before every kernel launch.
5. Add correctness coverage to CTest/pytest.
6. Capture baseline and candidate performance artifacts.
7. Run `tools/perf_truth_gate.py baseline.json candidate.json ...`.
8. If candidate is slower or unstable, dispatch must fall back to OpenCL.

This lets us add custom kernels incrementally while preserving the existing
runtime contract and release gate.
