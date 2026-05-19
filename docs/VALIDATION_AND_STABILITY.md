# Validation and stability contract

MotifCL treats model artifacts (`.mclt`, GGUF, safetensors, quantized
checkpoint directories) as untrusted input. Host code must validate shapes,
dtypes, byte lengths, quant metadata, and kernel launch invariants before any
OpenCL kernel sees a buffer.

## Tensor and quant metadata invariants

- `Tensor` shape dimensions are non-negative internally and serialized
  dimensions must produce the exact expected dtype byte count.
- External quant scales are only valid for MotifCL quantized storage dtypes that
  consume an out-of-band scale tensor: `Q8_0`, `Q4_0`, and `Q4_0_COL`.
- Quant scale tensors must be valid `F32`, rank-1, on the same backend.
- Axis contracts:
  - axis `0`: one scale per row, `scales.numel() >= shape[0]`;
  - axis `1`: one scale per column, `scales.numel() >= shape[1]`;
  - axis `2`: flat blockwise scale, `scales.numel() >= ceil(numel/block_size)`;
  - axis `3`/`4`: `Q4_0_COL` column/tile8 block scales,
    `scales.numel() >= shape[1] * ceil(shape[0]/block_size)`.
- Serialized quant tensors must fail during load/attach if these invariants do
  not hold. They must not be deferred to kernel execution.

## Artifact parser negative coverage

Regression coverage should include malformed but bounded artifacts:

- `.mclt` tensors with valid qweight shape but too-short quant scales;
- quantized transformer checkpoints whose `.qweight.mclt` payloads violate the
  same scale coverage invariant;
- GGUF duplicate metadata keys, unknown metadata/tensor types, incomplete quant
  blocks, and tensor ranges that run past EOF.

These cases are covered by `tests/test_quant.cpp`, `tests/test_gemma_compat.cpp`,
`tests/test_gguf.cpp`, and the bounded artifact fuzzer
`tests/test_artifact_fuzz.cpp`.

## OpenCL kernel input contract

OpenCL kernels are not the trust boundary. Every kernel family must have a
host-side validation owner that checks:

1. tensor validity and same backend;
2. dtype-specific dispatch;
3. rank/shape/stride assumptions used by the kernel indexing formula;
4. scalar launch parameters such as `M/N/K`, head counts, page size, block size,
   and offsets;
5. packed-quant byte length and block divisibility;
6. external metadata coverage, especially quant scale tensors and page tables.

`docs/kernel_validation_contracts.json` records the current family-level owner
matrix. `python tools/check_kernel_contracts.py` verifies that every concrete
`__kernel` entry in `kernels/*.cl` is assigned to at least one contract family.

Per-callsite helpers now enforce the hot-path contract before launch:

- `validate_matmul_args(...)`;
- `validate_attention_args(...)`;
- `validate_kv_cache_args(...)`;
- `validate_quant_tensor_args(...)`.

## Stable vs experimental API

Stable by default:

- `Tensor`, core F32 ops, standard quantize/dequantize, Q4/Q8 matmul paths;
- `.mclt` tensor serialization with strict dtype/shape/metadata validation;
- common GGUF F16/F32/BF16 and supported packed quant payload loading;
- `motifcl_generate_transformer --inspect`, `--list-architectures`, and normal
  dense text generation for implemented families.

Experimental / benchmark-gated:

- compressed regular KV cache (`q8`/`q4`);
- paged-KV and compressed paged-KV;
- hybrid Qwen3.5/Mixtral-style runtime pieces;
- fused MLP/attention variants behind env or explicit flags;
- future vendor-specific or handwritten assembly kernels.

Experimental features must stay opt-in, retain safe fallbacks, and must not
weaken artifact parsing or kernel launch validation.

Paged KV, compressed KV, and non-OpenCL microkernel backend requests emit
runtime warnings when explicitly enabled/requested. `native`/`asm`
microkernels currently resolve to a controlled OpenCL fallback until correctness
and performance gates prove a concrete implementation.

## CI gate

Use the local gate before tagging or before merging high-risk runtime changes:

```powershell
python tools/ci_gate.py --parallel 2
```

The gate runs:

- `git diff --check`;
- kernel contract coverage check;
- release configure/build;
- full release `ctest`;
- Python configure/build and `pytest`;
- tool syntax checks;
- `tools/release_check.py`.

New fast paths must also pass the performance truth gate:

```powershell
python tools/perf_truth_gate.py baselines/ellesmere.json candidate.json --tolerance 0.15
```

The gate is a policy wrapper over `tools/perf_regression.py`: correctness tests
come first, then measured wall-time/token-speed evidence, then fallback if the
candidate loses.
