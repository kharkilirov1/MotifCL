# FOG v3 MotifCL Vulkan port status

Base: **MotifCL `main`** from the user-provided archive. Experimental branches are intentionally out of scope.
Target hardware: **Radeon RX 580 / Polaris**, native Vulkan path.

| FOG component | MotifCL main status | Port action | Current status |
|---|---|---|---|
| token embedding | Vulkan native | reuse | ready |
| learned positions | Vulkan native | reuse | ready |
| 4-layer causal backbone | Vulkan GQA/MHA | reuse | ready |
| RMSNorm / GELU | Vulkan native | reuse | ready |
| tied LM head | `matmul_transpose_b` | reuse | ready |
| CE loss / Adam | Vulkan native | reuse | ready |
| elementwise multiply | API existed, Vulkan gap | Vulkan fwd path | implemented; SPIR-V + RX580 witness pending |
| SiLU | API existed, Vulkan gap | Vulkan fwd/bwd | implemented; SPIR-V + RX580 witness pending |
| sigmoid | portable Vulkan gap | Vulkan fwd/bwd | implemented; SPIR-V + RX580 witness pending |
| BLOCK_PRODUCT | FOG-specific | fused Vulkan fwd/bwd | implemented; RX580 witness pending |
| hard ST operator routing | FOG-specific | Vulkan fwd/bwd | implemented; RX580 witness pending |
| 4 flexible bilinear ops | dense MotifCL primitives | module | ready |
| query-conditioned binder | reuse Vulkan GQA; full-width address map | module | ready; hardware witness pending |
| typed value/control/scratch registers | missing | new register cell | ready; hardware witness pending |
| recurrent `value -> next address` | missing | structured step API | ready; hardware witness pending |
| HALT probability | sigmoid + linear | register cell | ready; hardware witness pending |
| cosine tied direct readout | matmul + fixed normalization | module | ready; hardware witness pending |
| lexical RX580 trainer | missing | new executable | ready |
| operator recurrence hardware gate | missing | new executable | ready, not yet run on RX580 |
| full binder->generate->re-address gate | missing | new executable | ready, not yet run on RX580 |
| Python `.pt` import | layout differs | optional converter after port freeze | not implemented |

## Static evidence completed in the development container

The following source slices compile with GCC C++17 and `-Wall -Wextra -Wpedantic -Werror`. In addition, a complete CMake/Ninja **link build** of `motifcl`, the Vulkan runtime, `test_fog_ops`, and examples `09/10/11` succeeded using temporary declarations for the new SPIR-V symbols. Those declarations were restored immediately afterward and are not part of the port:

- `src/runtime/vulkan_backend.cpp` using temporary declarations for the not-yet-regenerated new SPIR-V symbols;
- `src/ops/basic_ops.cpp`;
- `src/ops/activation.cpp`;
- `src/ops/fog.cpp`;
- `src/nn/fog_v3.cpp`;
- `tests/test_fog_ops.cpp`;
- examples `09`, `10`, `11`.

The real new GLSL shaders could not be compiled in this container because `glslc` is not installed here. The Windows build script regenerates the real SPIR-V bundle before compiling MotifCL.

## Claim boundary

The port now contains the **full structured v3 machine path** needed for controlled training: binder, typed registers, hard finite operator grammar, recurrent generated-state feedback, HALT head and direct tied readout. It is also ready for lexical pretraining.

What is **not** claimed yet:

1. a successful runtime witness on the user's physical RX 580 — run `setup-and-check-fog-rx580.ps1`;
2. bit/parameter equivalence to the PyTorch v3 checkpoint;
3. natural-language semantic reasoning quality — that is a training result, not a port property;
4. bit-exact optimizer resume (weights resume, Adam moments currently do not).
