# Applying the method to MotifCL

The C++ side is an extension of the MotifCL engine. To build and run it, integrate these
sources into a MotifCL checkout.

## 1. Copy sources into MotifCL

| From this repo | Into MotifCL |
|---|---|
| `engine/kernels/compact_counter.cl` | `kernels/compact_counter.cl` |
| `engine/include/compact_counter.hpp` | `include/motifcl/nn/compact_counter.hpp` |
| `engine/src/compact_counter.cpp` | `src/nn/compact_counter.cpp` |
| `tests/test_counter_state.cpp` | `tests/test_counter_state.cpp` |
| `tests/test_counter_memory_truth.cpp` | `tests/test_counter_memory_truth.cpp` |
| `tests/test_reversible_attn.cpp` | `tests/test_reversible_attn.cpp` |
| `tests/test_f16_matmul_autograd.cpp` | `tests/test_f16_matmul_autograd.cpp` |

`compact_counter.cl` now carries both the original `*_f32` update kernels (which consume a
dense `[out,in]` grad_w) and the memory-native `counter_row_stats_fused_f32` /
`counter_apply_update_fused_f32` kernels, which form grad_w in registers from `grad_out` and
`x`. The layer uses the fused path; the dense-grad path is kept only for cross-checking.

## 2. Apply the engine patches

```bash
cd <motifcl>
git apply engine/patches/backend_kernel_route.patch   # routes "counter" kernels to compact_counter.cl
git apply engine/patches/matmul_f16_autograd.patch     # adds F16MatMulBackward, enables f16 matmul autograd
```

If `git apply` rejects (upstream drift), apply manually:
- **backend**: in `KernelCache::source_file_for_kernel`, add as the FIRST branch:
  `if (contains(kernel_name, "counter")) return "compact_counter.cl";`
- **matmul**: add the `F16MatMulBackward` node and, in `matmul()`'s f16 branch, attach it
  when `requires_grad` instead of throwing.

## 3. Register in CMake

- `CMakeLists.txt` → `MOTIFCL_SOURCES`: add `src/nn/compact_counter.cpp`.
- `tests/CMakeLists.txt` → `MOTIFCL_TESTS`: add `test_counter_state`,
  `test_counter_memory_truth`, `test_reversible_attn`, `test_f16_matmul_autograd`.

## 4. Build + test

```bash
# Windows: Ninja + clang-cl (warnings not errors)
cmake -B build/dev -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DMOTIFCL_WARNINGS_AS_ERRORS=OFF
cmake --build build/dev
ctest --test-dir build/dev -R "counter_state|reversible_attn|f16_matmul_autograd" --output-on-failure
```

## 5. GPU benchmarks (optional)

`benchmarks/gpu/*.cpp` are standalone (compile against `motifcl.lib`). The `build_*.bat`
scripts show the exact clang-cl invocation (`/MD`, `/I include`, `/I third_party/opencl`,
link `build/dev/motifcl.lib` + `build/dev/OpenCL.lib`). Set
`MOTIFCL_KERNEL_DIR=<motifcl>/kernels` when running.

## Notes / constraints
- Requirement: `in_features % 4 == 0` (6-bit packing groups of 4).
- In-backward weight update is eager-only; incompatible with grad-accumulation,
  weight-sharing, DDP all-reduce, activation-checkpointing without an explicit scheduler.
- The layer self-updates whenever autograd is enabled and `update_enabled()` is true —
  it does **not** require its input to have `requires_grad` (a counter layer fed raw input
  still updates its state). `grad_x` is only propagated when the input asks for it.
- Memory-native backward: no dense `[out,in]` grad_w is allocated and no decoded weight is
  carried across forward→backward. `test_counter_memory_truth` enforces this. The fused
  in-kernel gradient costs ~2× the grad_w GEMM, so re-measure perf on device before quoting
  the old 0.97×-dense number for this path.
- Tested on AMD RX 580 (Polaris) via OpenCL; needs `cl_khr_fp16` for the f16 test (else it skips).
