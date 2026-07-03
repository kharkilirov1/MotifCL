# MotifCL repository study — 2026-06-28

Рабочая копия: `C:\Users\Kharki\Desktop\motifcl_production\motifcl_production`

## 1. Как изучалось

Использованные источники и проверки:

- CPL CLI fallback, потому что CPL MCP не был доступен, а HTTP `http://127.0.0.1:3878/health` не отвечал.
- `cpl scan`, `cpl skeleton`, `cpl retrieve`, `cpl grep`, `cpl symbols`, `cpl references`.
- Structural CPL index: `.cpl/index.sqlite`, 255 indexed files, 2545 symbols, 30391 references, 2683 chunks.
- Semantic vector DB: попытка `cpl embed-index --backend ollama --model nomic-embed-text --dimensions 768` не уложилась в 300s RPC timeout; зависший `cargo/cpl` процесс был остановлен. Vector DB не создан.
- `git status`, `git diff --stat`, `git diff --check`.
- CMake/package/test inventory через `CMakePresets.json`, `pyproject.toml`, `tests/CMakeLists.txt`, CTest.
- Независимая временная GCC/Ninja сборка: `build/codex-study-gcc`.

## 2. Идентичность проекта

MotifCL — C++17/OpenCL neural compute framework, ориентированный на legacy AMD GPUs и исследовательские transformer/quantization workflows.

Пакетная идентичность:

- Root Python package: `motifcl`, version `1.1.1`.
- Build backend: `scikit-build-core`, `pybind11`.
- CMake project: `MotifCL VERSION 1.1.1 LANGUAGES CXX`.
- Core runtime: C++17 + OpenCL.
- Python binding layer: `python/bindings.cpp` + wrappers under `python/motifcl`.
- Отдельный experimental подпроект: `memory-native`, version `0.1.0`, pure Python/PyTorch/Numpy design for finite-state counter synapses and memory-efficient training.

## 3. Размер и структура

CPL scan:

- 325 files total.
- 243 source files.
- ~3.0 MB indexed project bytes.
- Languages: C++, C/C++, Python.
- Recommended CPL mode: Hybrid.
- Complexity estimate: ~759k tokens.

Tracked Git files:

- 248 tracked files.
- Top tracked areas:
  - `include`: 67
  - `src`: 54
  - `tests`: 32
  - `tools`: 19
  - `docs`: 16
  - root: 14
  - `kernels`: 12
  - `examples`: 10
  - `benchmarks`: 9
  - `python`: 9

Важно: untracked `memory-native/` содержит собственную `.venv/` и `results/`; внутри `memory-native/.gitignore` есть `.venv/`, `*.pt`, `build/`, `dist/`, но при review/staging нужно не делать blind `git add memory-native/` без проверки.

## 4. Основные подсистемы

### 4.1 Build/package layer

Главная конфигурация:

- `CMakeLists.txt`
- `CMakePresets.json`
- `cmake/CompilerOptions.cmake`
- `pyproject.toml`
- `python/CMakeLists.txt`
- `tests/CMakeLists.txt`
- `benchmarks/CMakeLists.txt`
- `examples/cpp/CMakeLists.txt`
- `tools/CMakeLists.txt`

Root CMake options:

- `MOTIFCL_BUILD_TESTS`
- `MOTIFCL_BUILD_PYTHON`
- `MOTIFCL_BUILD_EXAMPLES`
- `MOTIFCL_BUILD_TOOLS`
- `MOTIFCL_BUILD_BENCHMARKS`
- `MOTIFCL_WARNINGS_AS_ERRORS`
- `MOTIFCL_INSTALL`
- `MOTIFCL_ENABLE_SANITIZERS`
- `MOTIFCL_ENABLE_FAST_MATH`
- `MOTIFCL_ENABLE_NATIVE_ARCH`
- `MOTIFCL_ENABLE_LTO`
- `MOTIFCL_OPENCL_FAST_RELAXED_MATH`
- `MOTIFCL_ANDROID_MI9_TUNING`

CMake presets observed:

- configure: `dev`, `release`, `rx580-release`, `android-mi9-arm64`, `asan`, `python`.
- build: `dev`, `release`, `rx580-release`, `android-mi9-arm64`, `macos-release`, `asan`, `python`.
- test: `dev`, `asan`, `rx580-release`.

### 4.2 Public C++ API surface

Main umbrella header:

- `include/motifcl/motifcl.hpp`

Major API namespaces/modules:

- `include/motifcl/core`: dtype, device, status, error, shape, logging.
- `include/motifcl/runtime`: OpenCL context, buffers, program/kernel wrappers, backend, command buffer, profiler, microkernel/native matmul/Vulkan hooks.
- `include/motifcl/tensor`: storage, allocator, tensor object.
- `include/motifcl/autograd`: node, tape, backward, graph capture/replay.
- `include/motifcl/ops`: basic ops, activation, matmul, attention, norm, reduction, indexing, quant, loss, optim, fused transformer kernels.
- `include/motifcl/nn`: module, parameter, linear, embedding, RMSNorm, attention, transformer, Gemma/HF compatibility, compact counter.
- `include/motifcl/motif`: motif/research layers.
- `include/motifcl/train`: training utilities/static train step/checkpoint/history helpers.

Notable public classes/symbol families:

- Runtime: `Backend`, `KernelCache`, `OpenCLContext`, `Buffer`, `Program`, `Kernel`, `DriverCommandBuffer`, `Profiler`.
- Tensor: `Tensor`, `TensorImpl`, `Storage`, `Allocator`.
- Autograd: `Node`, `Tape`, `CapturedGraph`, `GraphExecutor`, `GraphCaptureGuard`.
- NN: `Parameter`, `Module`, `Linear`, `QuantizedLinear`, `Embedding`, `RMSNorm`, `SelfAttention`, `TransformerBlock`, `GPTModel`.
- Modern transformer path: `TransformerConfig`, `ModernSelfAttention`, `ModernMLP`, `ModernTransformerBlock`, `ModernGPTModel`, `KVCache`, `PagedKVCache`, `DeltaStateCache`.
- HF/Gemma path: `HFTransformerConfig`, architecture registry/probe helpers, safetensors/GGUF loaders, tokenizer/generation helpers, quantization policies.
- Research/motif path: `MotifLinear`, `MotifLoRA`, `Router`, `MotifTransformerBlock`, `SARCResidual`.
- New memory-native C++ layer: `CounterStateLinear`.

### 4.3 Runtime/backend model

Core files:

- `src/runtime/opencl_context.cpp`
- `src/runtime/backend.cpp`
- `src/runtime/buffer.cpp`
- `src/runtime/program.cpp`
- `src/runtime/kernel.cpp`
- `src/runtime/command_buffer.cpp`
- `src/runtime/microkernel.cpp`
- `src/runtime/native_matmul.cpp`
- `src/runtime/vulkan_backend.cpp`
- `src/runtime/profiler.cpp`

Observed behavior:

- `Backend::create_opencl()` constructs OpenCL runtime with kernel directory.
- `KernelCache` maps kernel names to `.cl` source files and caches `Program` objects.
- Runtime has capability probes for integer dot mode and command-buffer support.
- `Buffer` owns OpenCL `cl_mem`, guards upload/download ranges, checks context liveness, and now exposes device memory byte accounting functions.
- Backend lifecycle cleanup calls `clear_memory_pool_for_context(ctx)` in move/destruction/reset paths.

### 4.4 Tensor/autograd model

Core files:

- `include/motifcl/tensor/tensor.hpp`
- `src/tensor/tensor.cpp`
- `src/tensor/storage.cpp`
- `src/tensor/allocator.cpp`
- `include/motifcl/autograd/node.hpp`
- `include/motifcl/autograd/tape.hpp`
- `include/motifcl/autograd/graph.hpp`
- `src/autograd/backward.cpp`
- `src/autograd/tape.cpp`
- `src/autograd/graph.cpp`

Observed model:

- `TensorImpl` stores backend pointer/lifetime, storage, shape/strides/offset, dtype, quantization metadata, requires-grad flag, grad tensor, and grad function.
- `Tensor` supports views, contiguous conversion, CPU transfer, gradients, `backward()`, and common construction helpers.
- Autograd is dynamic-node based, with graph capture/replay support for runtime plans, tensor specs, graph buffer plans, runtime bindings, and command-buffer replay when supported.

### 4.5 Ops and OpenCL kernels

Core files:

- `src/ops/basic_ops.cpp`
- `src/ops/matmul.cpp`
- `src/ops/attention.cpp`
- `src/ops/fused_transformer.cpp`
- `src/ops/activation.cpp`
- `src/ops/norm.cpp`
- `src/ops/reduce.cpp`
- `src/ops/indexing.cpp`
- `src/ops/fp16.cpp`
- `src/ops/quant.cpp`
- `src/ops/loss.cpp`
- `src/ops/optim.cpp`
- `kernels/*.cl`

Implemented/advertised stack from README/docs:

- Elementwise/scalar ops.
- Row reductions.
- Register-blocked F32 matmul.
- Generated F32 matmul tile variants.
- Q8/Q4 quantize/dequantize and mixed quantized matmul paths.
- Integer-dot accelerated Q8 path with fallback.
- Q4 dot4-unrolled specialization.
- Activations including SwiGLU.
- GPU dropout/masked fill.
- Softmax rows and causal mask.
- RoPE variants.
- Fused QKV split.
- GQA/MQA attention and backward paths.
- KV-cache append.
- FlashAttention-style tiled multi-head attention forward/backward.
- RMSNorm/LayerNorm including fused residual helpers.
- MSE, softmax cross entropy.
- Adam and SGD kernels.

Current branch includes split value-head support:

- `TransformerConfig.v_head_dim`.
- `ModernSelfAttention::v_head_dim()`.
- `qkv_split(packed, q_dim, k_dim, v_dim)` overload.
- Attention validation computes `v_head_dim` separately from `head_dim`.
- Output channels for GQA/MQA use `n_head * v_head_dim`.

### 4.6 Modern transformer/HF/Gemma layer

Core files:

- `include/motifcl/nn/transformer.hpp`
- `src/nn/transformer.cpp`
- `include/motifcl/nn/hf_compat.hpp`
- `src/nn/hf_compat.cpp`
- `include/motifcl/nn/gemma.hpp`
- `src/nn/gemma.cpp`

Architecture direction from docs:

- Legacy `GPTModel` remains a smoke-test architecture.
- New experiments should use `TransformerConfig` + `ModernGPTModel`/`ModernTransformerBlock`/`ModernSelfAttention`.
- Modern path includes packed QKV projection, RoPE, GQA/MQA, RMSNorm, SwiGLU MLP, KV cache, paged KV cache, HF-style model adapter coverage, GGUF/safetensors loading, tokenizer/generation tooling, and mixed Q4/Q8 policies.

### 4.7 Python bindings/wrappers

Core files:

- `python/bindings.cpp`
- `python/motifcl/__init__.py`
- `python/motifcl/functional.py`
- `python/motifcl/nn.py`
- `python/motifcl/optim.py`

Observed binding surface:

- Runtime/backends.
- Tensor constructors and ops.
- Memory pool helpers: `clear_memory_pool`, `memory_pool_cached_blocks`, `memory_pool_cached_bytes`.
- Modern transformer ops including `qkv_split` overloads.
- `TransformerConfig` and modern model classes.
- NN wrappers: `Module`, `Linear`, `GELU`, `Sequential`, `QuantizedLinear`, `GPTModel`.

### 4.8 `memory-native` experimental package

Path:

- `memory-native/`

Package metadata:

- Name: `memory-native`
- Version: `0.1.0`
- Description: finite-state counter synapses + reversible activations for memory-efficient training in pure PyTorch.
- Runtime dependency: `numpy>=1.21`.
- Optional deps:
  - `torch`: `torch>=2.1`
  - `dev`: `pytest>=7`, `torch>=2.1`

Key modules:

- `counter.py`: `_FusedCounterLinearFn`, `CompactCounterLinear`, `RMSCounterLinear`, finite-state encode/decode, stochastic rounding.
- `actquant.py`: packed int4 and stochastic activation quantization helpers.
- `baselines.py`: `TernaryQATLinear`.
- `fused_qkv.py`: `CounterQKVLinear`.
- `fused_update.py`: hashed stochastic-rounding update helpers.
- `int8_compute.py`: int8 quantization/matmul experiments.
- `memory.py`: memory reporting/peak training memory estimates.
- `models.py`: small GPT model using counter/reversible ideas.
- `np_native.py`: NumPy-native counter/GPT implementation.
- `optimizers.py`: GaLore/LoMo optimizer wrappers.
- `reversible.py`: reversible blocks/activation recompute patterns.

Test suite exists under `memory-native/tests`, but current local Python environments do not have `pytest`, `numpy`, or `torch`, so these tests were not executed.

## 5. Current Git/worktree state

Branch:

- `fog-qkv-split`

HEAD:

- `29f2702 WIP split qk_head_dim/v_head_dim — forward OK, backward training segfault`

Working tree:

- 10 modified tracked files.
- 14 untracked entries before generated-report additions.

Tracked diff stat:

- `CMakeLists.txt`: +18
- `CMakePresets.json`: +92
- `cmake/CompilerOptions.cmake`: +20
- `docs/ROADMAP.md`: +8/-4
- `include/motifcl/runtime/buffer.hpp`: +8
- `src/ops/attention.cpp`: +13
- `src/ops/matmul.cpp`: +26/-1
- `src/runtime/backend.cpp`: +33/-? small refactor
- `src/runtime/buffer.cpp`: +23
- `tests/CMakeLists.txt`: +4

Untracked relevant product files:

- `docs/BITNET_TERNARY_VULKAN_PLAN.md`
- `docs/COUNTER_STATE_NATIVE_DESIGN.md`
- `docs/MEMORY_NATIVE_TRAINING_METHOD.md`
- `docs/PORTABILITY.md`
- `include/motifcl/nn/compact_counter.hpp`
- `kernels/compact_counter.cl`
- `memory-native/`
- `ports/android/*`
- `ports/apple/*`
- `ports/rx580/*`
- `src/nn/compact_counter.cpp`
- `tests/test_counter_state.cpp`
- `tests/test_f16_matmul_autograd.cpp`
- `tests/test_memory_truth_gate.cpp`
- `tests/test_reversible_attn.cpp`

Diff hygiene:

- `git diff --check` produced no whitespace/conflict-marker errors.
- It did warn that `CMakeLists.txt` and `docs/ROADMAP.md` CRLF will be replaced by LF when Git touches them.

## 6. Current change themes

### 6.1 Portability/performance build knobs

Tracked CMake changes add:

- Fast math toggle.
- Native arch toggle.
- LTO toggle.
- OpenCL fast relaxed math define.
- Android MI9 tuning toggle.
- LTO/IPO support probing.
- New presets for RX 580, Android MI 9 arm64, macOS release.

### 6.2 Compact counter / memory-native training

New C++ path:

- `CounterStateLinear` is a self-updating finite-state counter layer.
- State is packed; comments describe 6-bit packing and no persistent dense `[out,in]` gradient allocation in memory-native backward path.
- `decode_weight()` exists as a dense reference/forward helper.
- `apply_update_backward(grad_out, x, seed)` recomputes update inside kernels.
- `backward_input_from_state()` computes grad input without materializing dense weight.
- `parameters()` returns empty because the layer updates internal state.

New kernel:

- `kernels/compact_counter.cl`
- Includes decode, fused RMS update, and backward-input support.

Verification tests added:

- `test_counter_state`
- `test_memory_truth_gate`
- `test_f16_matmul_autograd`
- `test_reversible_attn`

### 6.3 Device memory accounting

`include/motifcl/runtime/buffer.hpp` adds:

- `device_bytes_current()`
- `device_bytes_peak()`
- `device_bytes_reset_peak()`

Purpose stated in comments: memory-native truth gate should detect accidental dense weight-sized buffer materialization.

### 6.4 Attention/QKV split and `v_head_dim`

Current code supports `v_head_dim` separately from q/k `head_dim`.

Important observed points:

- `TransformerConfig` has `v_head_dim`.
- `normalize_config` defaults `v_head_dim` to `head_dim` when <= 0.
- `ModernSelfAttention` computes `v_dim = n_kv_head * v_head_dim`.
- QKV split call uses `qkv_split(packed, q_dim_, kv_dim_, v_dim_)`.
- Attention validation checks q/k head dim match and v head dim separately.
- For `v_head_dim != head_dim`, current backward/fallback paths insert `backend.finish()` synchronization to bound temporary chains/driver queue pressure while staged split kernels are absent.

### 6.5 FP16 matmul autograd

Tracked `src/ops/matmul.cpp` change adds an F16 backward node:

- F16 matmul forward remains F16/F16.
- Backward casts to F32 for gradient matmuls, then casts gradients back to F16.
- Covered by `tests/test_f16_matmul_autograd.cpp`.

## 7. Verification results

### 7.1 CPL

Verified:

- CPL CLI works.
- CPL structural index refresh succeeded.
- `index-db`: 255 files, 2545 symbols, 30391 references, 2683 chunks.

Not verified:

- CPL MCP server was not available in this tool context.
- CPL HTTP server at `127.0.0.1:3878` was not reachable.
- CPL semantic vector DB was not built; `embed-index` exceeded 300s and was stopped.

### 7.2 Existing `dev` preset

Current shell lacks `clang-cl` and `cl`:

- `where clang-cl`: not found.
- `where cl`: not found.
- `CXX`/`CC`: empty.

`cmake --preset dev` result:

- Fails because cached/selected `CMAKE_CXX_COMPILER` is `clang-cl`, but `clang-cl` is not in PATH.

`cmake --build --preset dev` result:

- Fails because `build/dev/build.ninja` includes missing `CMakeFiles\rules.ninja`.

Important nuance:

- Existing old test executables under `build/dev` still run, but that is not a fresh build witness for the current source.

### 7.3 Fresh temporary GCC/Ninja build

Configured a separate temporary build dir:

```powershell
cmake -S . -B build/codex-study-gcc -G Ninja `
  -DCMAKE_CXX_COMPILER=C:/Strawberry/c/bin/g++.exe `
  -DCMAKE_BUILD_TYPE=Debug `
  -DMOTIFCL_BUILD_TESTS=ON `
  -DMOTIFCL_BUILD_EXAMPLES=OFF `
  -DMOTIFCL_BUILD_TOOLS=OFF `
  -DMOTIFCL_BUILD_BENCHMARKS=OFF `
  -DMOTIFCL_BUILD_PYTHON=OFF `
  -DMOTIFCL_WARNINGS_AS_ERRORS=OFF
```

Configure result:

- GNU C++ 13.2.0.
- Vendored/minimal OpenCL headers with `C:/WINDOWS/System32/OpenCL.dll`.
- Configure and generation succeeded.

Build result:

- `cmake --build build/codex-study-gcc --parallel 4` succeeded.
- Repeated no-op build exit code `0`, output: `[1/1] Copying MotifCL OpenCL kernels to the build tree`.

CTest inventory:

- 31 tests.

Full CTest result:

- Command: `ctest --test-dir build/codex-study-gcc --output-on-failure --timeout 90`
- Result: `100% tests passed, 0 tests failed out of 31`
- Total time: `233.09 sec`

Focused changed-area tests also passed:

- `test_shape`
- `test_buffer`
- `test_tensor`
- `test_counter_state`
- `test_memory_truth_gate`
- `test_f16_matmul_autograd`
- `test_reversible_attn`

### 7.4 Python/memory-native tests

Blocked by environment:

- `memory-native/.venv/Scripts/python.exe --version`: Python 3.11.14.
- `memory-native/.venv/Scripts/python.exe -m pytest --version`: `No module named pytest`.
- system `python -m pytest --version`: `No module named pytest`.
- system import availability: `torch=False`, `numpy=False`, `pytest=False`.

No Python dependencies were installed during this study.

## 8. Risks and engineering notes

1. **Dirty tree is substantial.** It mixes build-system portability, native memory training, docs, platform ports, and attention/F16/autograd changes. Review should be split by theme if this is meant to become mergeable.
2. **Commit message says backward training segfault, but fresh GCC full CTest is green.** This means either the segfault is outside current test coverage, hardware/compiler-specific, or already mitigated by the working tree changes. Do not infer GPU training stability from CTest alone.
3. **Dev preset is not currently reproducible in this shell.** The main blocker is missing `clang-cl`/MSVC environment plus a stale/incomplete `build/dev`. Fresh GCC build works, but that is not the same as the intended `dev` preset.
4. **`memory-native/` is large as a working directory.** It contains `.venv` and results. Its nested `.gitignore` protects `.venv` if respected, but staging must be deliberate.
5. **CPL `.cpl/` is not ignored by root `.gitignore`.** The structural index was generated for this study only and should not be staged.
6. **Python tests are unverified.** The memory-native Python package needs dev deps (`pytest`, `torch`, likely `numpy`) before its tests can run.
7. **Attention split `v_head_dim` path uses conservative queue synchronization.** That is a correctness/memory-pressure safety measure, but likely a performance tradeoff until staged split backward kernels exist.
8. **F16 autograd is intentionally narrow.** Current new test proves matmul backward, not full FP16 training coverage for transformer/fused paths.

## 9. Suggested next actions

If the next objective is to make this branch reviewable:

1. Add `.cpl/` to root `.gitignore` or keep deleting local CPL cache before status snapshots.
2. Split work into review chunks:
   - portability/CMake presets,
   - device memory accounting,
   - compact counter native layer,
   - F16 matmul autograd,
   - `v_head_dim`/QKV attention path,
   - memory-native Python prototype,
   - ports/docs.
3. Decide canonical compiler path on Windows:
   - fix `dev` preset shell setup via MSVC/LLVM developer environment, or
   - add/keep a documented GCC/MinGW local verification path.
4. Install memory-native dev deps in an isolated env and run:
   - `python -m pytest memory-native/tests -q`
5. Add a regression reproducer for the branch HEAD warning: “backward training segfault” if it still exists outside current CTest.
6. Run platform-specific verification where intended:
   - RX 580 preset,
   - Android MI9 toolchain path,
   - macOS/Metal probe path.

## 10. Bottom line

The repository is a broad C++17/OpenCL DL framework with a serious amount of transformer, quantization, HF/Gemma compatibility, autograd, and runtime work already present. Current working tree adds a second major research direction: memory-native finite-state counter training plus portability/performance scaffolding.

Fresh GCC build and all 31 C++/OpenCL CTests pass in this environment. The main unresolved local issues are: broken/stale `dev` preset due missing `clang-cl`, unverified Python `memory-native` tests due missing deps, and broad dirty working tree scope that should be split before review.
