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
- `src/runtime/microkernel.cpp`
  - normalizes backend names;
  - reads per-domain env overrides;
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

Policy for new fast paths:

1. Keep OpenCL as the fallback.
2. Native/ASM path is opt-in until it has correctness and performance evidence.
3. Add a per-callsite validator before every kernel launch.
4. Add correctness coverage to CTest/pytest.
5. Capture baseline and candidate performance artifacts.
6. Run `tools/perf_truth_gate.py baseline.json candidate.json ...`.
7. If candidate is slower or unstable, dispatch must fall back to OpenCL.

This lets us add custom kernels incrementally while preserving the existing
runtime contract and release gate.
