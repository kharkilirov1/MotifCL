# Vulkan Port Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining MotifCL OpenCL-to-Vulkan port gaps with verified Vulkan-native training/update paths, Python runtime selection, and OpenCL-OFF build gates.

**Architecture:** Keep OpenCL as the legacy backend while adding explicit Vulkan-native execution paths with no hidden OpenCL fallback. Every Vulkan claim must have a device-resident Tensor/runtime witness and an `MOTIFCL_ENABLE_OPENCL=OFF` witness where the code belongs in the standalone Vulkan runtime. Do not remove OpenCL until all tests prove the Vulkan replacement.

**Tech Stack:** C++17, generated SPIR-V compute shaders, Vulkan storage buffers, MotifCL Tensor/Backend runtime, pybind11 bindings when available, CMake/Ninja/CTest.

## Global Constraints

- Repository root: `C:\Users\Kharki\Desktop\motifcl_production\motifcl_production`.
- Current active branch: `fog-qkv-split`; current workspace is dirty and contains the active Vulkan port state, so implementation continues in-place unless the user explicitly requests a new worktree.
- No completion claim without fresh witness: build/test output, link command, benchmark output, or runtime log.
- Vulkan-native means device-resident `VulkanBuffer` inputs/outputs; host upload/download is allowed only at test boundary.
- `MOTIFCL_ENABLE_OPENCL=OFF` must never link `OpenCL::OpenCL` into `motifcl_vulkan_runtime` or `test_vulkan_runtime_standalone`.
- Python bindings cannot be claimed verified unless pybind11 target/config exists and the module imports locally.
- Existing scoped Vulkan paths must remain green: `test_vulkan_backend`, `test_vulkan_runtime_standalone`, full CTest suite.
- Do not rewrite generated or unrelated files; change only files named in a task.

---

## File Structure

- `include/motifcl/runtime/vulkan_backend.hpp` — public Vulkan runtime APIs for compact-counter update, backward kernels, and Python-safe backend probing.
- `src/runtime/vulkan_backend.cpp` — generated SPIR-V and persistent `VulkanRuntime` dispatch wrappers.
- `src/nn/compact_counter.cpp` — `CounterStateLinear` backend routing for decode, backward-input, and fused update.
- `src/ops/matmul.cpp` — Vulkan autograd-compatible forward/backward building blocks for F32 matmul variants.
- `src/ops/basic_ops.cpp` — Vulkan add backward primitive and direct add reuse in autograd.
- `src/ops/norm.cpp` — Vulkan RMSNorm backward route or explicit verified unsupported gate.
- `src/ops/activation.cpp` — Vulkan SwiGLU backward route or explicit verified unsupported gate.
- `src/ops/attention.cpp` — Vulkan GQA backward route or explicit verified unsupported gate for training.
- `include/motifcl/autograd/graph.hpp` and `src/autograd/graph.cpp` — backend-neutral graph capture metadata replacing OpenCL-only assumptions where needed.
- `python/bindings.cpp`, `python/motifcl/__init__.py`, `CMakeLists.txt` — Vulkan runtime selection and optional Vulkan-only Python target.
- `tests/test_vulkan_backend.cpp` — Tensor-level Vulkan witnesses.
- `tests/test_vulkan_runtime_standalone.cpp` — OpenCL-free Vulkan runtime witnesses.
- `tests/test_memory_truth_gate.cpp` — compact-counter memory-native truth gate.
- `docs/VULKAN_PORT_PROTOCOL.md` — status and exact verification evidence.

---

### Task 1: Production U8 compact-counter fused update on Vulkan

**Files:**
- Modify: `include/motifcl/runtime/vulkan_backend.hpp`
- Modify: `src/runtime/vulkan_backend.cpp`
- Modify: `src/nn/compact_counter.cpp`
- Test: `tests/test_vulkan_runtime_standalone.cpp`
- Test: `tests/test_vulkan_backend.cpp`
- Test: `tests/test_memory_truth_gate.cpp`
- Docs: `docs/VULKAN_PORT_PROTOCOL.md`

**Interfaces:**
- Consumes: existing `VulkanRuntime`, `VulkanBuffer`, `run_vulkan_compact_counter_decode_weight`, `run_vulkan_compact_counter_backward_input_u8`.
- Produces:
  - `VulkanOpResult run_vulkan_compact_counter_apply_update_backward_u8(VulkanRuntime&, VulkanBuffer& state, VulkanBuffer& scale, VulkanBuffer& v, const VulkanBuffer& grad_out, const VulkanBuffer& x, std::size_t batch, std::size_t in_features, std::size_t out_features, std::size_t C, float lr, float lr_scale, float rms_beta, float rms_eps, std::uint32_t seed);`
  - `CounterStateLinear::apply_update_backward(...)` works for Vulkan tensors without calling `Tensor::buffer()` or `backend_->kernels.get(...)`.

- [ ] **Step 1: Add failing runtime witness for production U8 update**

Add to `tests/test_vulkan_runtime_standalone.cpp` inside the `runtime.supports_storage_buffer_i8()` block. The test must allocate U8 3-byte packed state, F32 scale/v/grad_out/x buffers, call `run_vulkan_compact_counter_apply_update_backward_u8`, download state/scale/v, and compare against a CPU reference copied from `kernels/compact_counter.cl` formulas:

```cpp
// Reference formulas to implement in the test helper:
// gw(row,i) = sum_r grad_out[r*out_features + row] * x[r*in_features + i]
// gsq = sum_i gw(row,i)^2 / in_features
// gs = sum_i gw(row,i) * t(row,i) / sqrt(in_features)
// vv = rms_beta * v[row] + (1 - rms_beta) * gsq
// denom = max(sqrt(vv), rms_eps)
// scale_new = clamp(scale[row] - lr_scale * gs, 1e-5f, 10.0f)
// tick = -lr * (gw / denom) * (C / scale_new)
// c_reb = c_old * (scale_old / scale_new)
// val = c_reb + tick
// stochastic rounding uses cc_uniform01(seed ^ cc_hash_u32(elem))
```

Run:

```powershell
cmake --build build/codex-vulkan-off --target test_vulkan_runtime_standalone -j 4
ctest --test-dir build/codex-vulkan-off -R test_vulkan_runtime_standalone --output-on-failure --timeout 240
```

Expected before implementation: FAIL mentioning missing symbol or failed update expectation.

- [ ] **Step 2: Implement Vulkan row-stats/update shader decomposition**

In `src/runtime/vulkan_backend.cpp`, add generated SPIR-V helpers:

```cpp
std::vector<std::uint32_t> counter_row_stats_fused_u8_spirv(std::size_t batch, std::size_t in_features, std::size_t out_features, std::size_t C, float lr_scale, float rms_beta, float rms_eps);
std::vector<std::uint32_t> counter_apply_update_fused_u8_spirv(std::size_t batch, std::size_t in_features, std::size_t out_features, std::size_t C, float lr, std::uint32_t seed);
std::vector<std::uint32_t> copy_f32_spirv();
```

Use three persistent Vulkan dispatches:
1. row-stats updates `v`, writes `scale_new`, `denom`.
2. apply-update mutates `state` using old `scale`, `scale_new`, `denom`.
3. copy commits `scale_new -> scale`.

For initial correctness, use one work-item per output row for row-stats instead of local reductions; cap `in_features <= 4096`, `batch <= 4096`, `out_features <= 4096`, and reject larger shapes with an explicit `VulkanOpResult.error`.

- [ ] **Step 3: Add public runtime API**

Declare and implement:

```cpp
VulkanOpResult run_vulkan_compact_counter_apply_update_backward_u8(
    VulkanRuntime& runtime,
    VulkanBuffer& state,
    VulkanBuffer& scale,
    VulkanBuffer& v,
    const VulkanBuffer& grad_out,
    const VulkanBuffer& x,
    std::size_t batch,
    std::size_t in_features,
    std::size_t out_features,
    std::size_t C,
    float lr,
    float lr_scale,
    float rms_beta,
    float rms_eps,
    std::uint32_t seed);
```

Validation must reject zero dimensions, `in_features % 4 != 0`, invalid `C`, missing `VK_KHR_8bit_storage`, non-finite hyperparameters, undersized buffers, and dispatch dimensions above uint32 range.

- [ ] **Step 4: Wire `CounterStateLinear::apply_update_backward`**

In `src/nn/compact_counter.cpp`, replace the Vulkan rejection in `apply_update_backward` with a call to the new API. Keep `apply_update_seed(grad_w)` explicitly unsupported on Vulkan unless a dense-grad update API is added in the same task. Validate `grad_out` and `x` are Vulkan tensors on the same backend.

- [ ] **Step 5: Add Tensor-level training witness**

Extend `tests/test_vulkan_backend.cpp` to:
1. Construct `nn::CounterStateLinear` on `Backend::create_vulkan()`.
2. Save initial `state`, `scale`, `v`.
3. Run `apply_update_backward(grad_out, x, seed)`.
4. Compare updated `state`, `scale`, `v` to the same CPU reference used in standalone test.
5. Assert `forward` before and after update changes according to decoded weights.

Run:

```powershell
cmake --build build/codex-study-gcc --target test_vulkan_backend test_vulkan_runtime_standalone -j 4
ctest --test-dir build/codex-study-gcc -R "test_vulkan_backend|test_vulkan_runtime_standalone" --output-on-failure --timeout 240
```

Expected after implementation: both tests pass.

- [ ] **Step 6: Run memory truth gate and full regression**

Run:

```powershell
ctest --test-dir build/codex-study-gcc -R "test_memory_truth_gate|test_counter_state|test_vulkan_backend" --output-on-failure --timeout 300
ctest --test-dir build/codex-study-gcc --output-on-failure --timeout 300
ctest --test-dir build/codex-vulkan-off -R test_vulkan_runtime_standalone --output-on-failure --timeout 240
```

Expected: all selected and full suite tests pass; OFF test remains OpenCL-free.

---

### Task 2: Vulkan autograd graph capture metadata

**Files:**
- Modify: `include/motifcl/autograd/graph.hpp`
- Modify: `src/autograd/graph.cpp`
- Modify: `tests/test_vulkan_backend.cpp`
- Docs: `docs/VULKAN_PORT_PROTOCOL.md`

**Interfaces:**
- Consumes: existing `autograd::record_op` and Tensor ids.
- Produces: graph metadata that can represent Vulkan-backed Tensor ops without requiring `cl_mem` handles.

- [ ] **Step 1: Add failing graph-capture witness**

In `tests/test_vulkan_backend.cpp`, begin graph capture around Vulkan Tensor `add`, `matmul`, `sgd_update`, and compact-counter update. Assert every node records `backend == "vulkan"` or equivalent enum and does not expose OpenCL-only memory handles.

Run:

```powershell
cmake --build build/codex-study-gcc --target test_vulkan_backend -j 4
ctest --test-dir build/codex-study-gcc -R test_vulkan_backend --output-on-failure --timeout 240
```

Expected before implementation: FAIL due absent backend metadata or OpenCL-only capture fields.

- [ ] **Step 2: Add backend-neutral captured tensor descriptor**

Change `TensorSpec` in `include/motifcl/autograd/graph.hpp` to include:

```cpp
enum class CapturedBackendKind { Unknown, OpenCL, Vulkan };
CapturedBackendKind backend = CapturedBackendKind::Unknown;
std::uintptr_t native_handle = 0;
```

OpenCL nodes may keep legacy `cl_mem` through `native_handle`; Vulkan nodes use a stable non-owning identifier only for debugging and scheduling. Do not expose raw secrets or driver pointers to Python by default.

- [ ] **Step 3: Populate metadata from Tensor backend**

In `src/autograd/graph.cpp`, when registering tensors, set backend kind from `tensor.backend().kind()`. Preserve existing OpenCL behavior for command buffer tests.

- [ ] **Step 4: Verify old and new graph tests**

Run:

```powershell
ctest --test-dir build/codex-study-gcc -R "test_graph_capture|test_vulkan_backend" --output-on-failure --timeout 240
```

Expected: graph capture and Vulkan backend tests pass.

---

### Task 3: Vulkan core backward kernels for training smoke

**Files:**
- Modify: `include/motifcl/runtime/vulkan_backend.hpp`
- Modify: `src/runtime/vulkan_backend.cpp`
- Modify: `src/ops/basic_ops.cpp`
- Modify: `src/ops/matmul.cpp`
- Modify: `src/ops/norm.cpp`
- Modify: `src/ops/activation.cpp`
- Test: `tests/test_vulkan_backend.cpp`

**Interfaces:**
- Consumes: `run_vulkan_add`, `run_vulkan_f32_matmul`, `run_vulkan_f32_matmul_transpose_b`, `run_vulkan_sgd_update`.
- Produces: Vulkan-backed backward for add and matmul first; RMSNorm/SwiGLU either ported with tests or explicitly rejected in training with clear error.

- [ ] **Step 1: Add failing Vulkan autograd smoke**

Add a test that creates Vulkan tensors `A`, `B`, runs `C = matmul(A, B)`, `D = add(C, target)`, calls backward with explicit grad output, and verifies `A.grad()` and `B.grad()` against CPU references. Use small shapes: `A=[2,3]`, `B=[3,2]`.

Run:

```powershell
cmake --build build/codex-study-gcc --target test_vulkan_backend -j 4
ctest --test-dir build/codex-study-gcc -R test_vulkan_backend --output-on-failure --timeout 240
```

Expected before implementation: FAIL because Vulkan matmul currently rejects autograd or backward falls to OpenCL.

- [ ] **Step 2: Allow Vulkan forward ops to attach backward nodes**

In `src/ops/matmul.cpp`, remove blanket `!requires_grad` from Vulkan F32 supported predicates only after adding matching backward. The backward must compute:

```cpp
if (a.requires_grad()) a.backward(matmul_transpose_b(grad_output, b));
if (b.requires_grad()) b.backward(matmul_transpose_a(a, grad_output));
```

Add or route Vulkan `matmul_transpose_a` support before enabling `b.requires_grad()` for Vulkan.

- [ ] **Step 3: Add Vulkan transpose-A matmul runtime API**

Add:

```cpp
VulkanOpResult run_vulkan_f32_matmul_transpose_a(VulkanRuntime&, const VulkanBuffer& a, const VulkanBuffer& b, VulkanBuffer& c, std::size_t m, std::size_t k, std::size_t n);
```

Shape contract: `a` is `[K,M]`, `b` is `[K,N]`, `c` is `[M,N]`, output element `c[row,col] = sum_kk a[kk*M + row] * b[kk*N + col]`.

- [ ] **Step 4: Port add backward path**

Ensure Vulkan `add` backward returns grad to both inputs with no OpenCL allocation. If broadcasting is unsupported, reject non-identical shapes with an explicit message.

- [ ] **Step 5: Verify targeted and full tests**

Run:

```powershell
ctest --test-dir build/codex-study-gcc -R "test_backward|test_vulkan_backend" --output-on-failure --timeout 240
ctest --test-dir build/codex-study-gcc --output-on-failure --timeout 300
```

Expected: old OpenCL backward tests and new Vulkan backward smoke pass.

---

### Task 4: Vulkan attention/GQA training boundary

**Files:**
- Modify: `include/motifcl/runtime/vulkan_backend.hpp`
- Modify: `src/runtime/vulkan_backend.cpp`
- Modify: `src/ops/attention.cpp`
- Test: `tests/test_vulkan_backend.cpp`
- Docs: `docs/VULKAN_PORT_PROTOCOL.md`

**Interfaces:**
- Consumes: existing non-causal unmasked F32 GQA forward.
- Produces: either a tested Vulkan backward for scoped GQA or a strict training gate that refuses `requires_grad` with a precise error.

- [ ] **Step 1: Add training-boundary test**

Add two assertions:
1. Inference scoped GQA still passes on Vulkan.
2. If any of `q/k/v.requires_grad()` is true, the op either returns correct gradients from Vulkan or throws `vulkan grouped_query_attention backward is not implemented` before allocating OpenCL buffers.

- [ ] **Step 2: Choose implementation based on complexity**

If implementing backward in this task, add generated SPIR-V for softmax-attention backward scoped to `query_tokens,key_tokens,head_dim <= 64`, `batch=1`, non-causal, unmasked. If not implementing, enforce the explicit gate and document the remaining gap.

- [ ] **Step 3: Verify**

Run:

```powershell
ctest --test-dir build/codex-study-gcc -R "test_attention|test_vulkan_backend" --output-on-failure --timeout 240
```

Expected: inference witness passes and training boundary is explicit.

---

### Task 5: Vulkan-only Python bindings and OpenCL-OFF package gate

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `python/bindings.cpp`
- Modify: `python/motifcl/__init__.py`
- Test: add `tests/test_python_vulkan_runtime.py` only if Python test infra exists; otherwise add CTest command around import.
- Docs: `docs/VULKAN_PORT_PROTOCOL.md`

**Interfaces:**
- Consumes: `Backend::create_vulkan`, `Backend::kind`, `is_vulkan`, `is_opencl`.
- Produces: Python import/runtime selection works when OpenCL is enabled; Vulkan-only Python target is configured or explicitly skipped with a CMake message when pybind11 is unavailable.

- [ ] **Step 1: Add CMake configure witness**

Run current configure with Python requested:

```powershell
cmake -S . -B build/codex-vulkan-off -G Ninja -DMOTIFCL_ENABLE_OPENCL=OFF -DMOTIFCL_BUILD_TESTS=ON -DMOTIFCL_BUILD_PYTHON=ON -DMOTIFCL_BUILD_EXAMPLES=ON -DMOTIFCL_BUILD_TOOLS=ON -DMOTIFCL_BUILD_BENCHMARKS=OFF
```

Record whether pybind11 is found. If not found, do not claim Python build verified.

- [ ] **Step 2: Split pybind module by backend availability**

Make `python/bindings.cpp` compile when `MOTIFCL_ENABLE_OPENCL=OFF` by guarding OpenCL-only classes/functions with compile definitions. Always expose:

```python
Backend.vulkan()
Backend.create()
BackendKind.Vulkan
backend.kind
backend.is_vulkan()
backend.is_opencl()
probe_vulkan_runtime()
```

- [ ] **Step 3: Add Vulkan-only pybind target**

In `CMakeLists.txt`, if `MOTIFCL_ENABLE_OPENCL=OFF` and pybind11 exists, link the Python module to `motifcl_vulkan_runtime` plus only source files that do not include OpenCL-only symbols. If full Tensor bindings are not available in OFF mode, expose a limited runtime module and name this limitation in docs.

- [ ] **Step 4: Verify import when pybind11 exists**

Run:

```powershell
cmake --build build/codex-vulkan-off --target motifcl_python -j 4
python -c "import motifcl; b=motifcl.Backend.vulkan(); print(b.is_vulkan())"
```

Expected: prints `True`. If pybind11 is unavailable, expected verified status is `Python target skipped because pybind11 is unavailable`, not success.

---

### Task 6: Final verification, performance status, and docs

**Files:**
- Modify: `docs/VULKAN_PORT_PROTOCOL.md`
- Modify: `docs/ROADMAP.md` if status table is stale.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: final evidence-backed status and remaining-gap list.

- [ ] **Step 1: Run fresh ON/OFF verification**

```powershell
cmake --build build/codex-study-gcc --target test_vulkan_backend test_vulkan_runtime_standalone -j 4
ctest --test-dir build/codex-study-gcc -R "test_vulkan_backend|test_vulkan_runtime_standalone" --output-on-failure --timeout 240
ctest --test-dir build/codex-study-gcc --output-on-failure --timeout 300
cmake --build build/codex-vulkan-off --target test_vulkan_runtime_standalone -j 4
ctest --test-dir build/codex-vulkan-off -R test_vulkan_runtime_standalone --output-on-failure --timeout 240
```

- [ ] **Step 2: Run fresh speed witness**

```powershell
if (Test-Path build/codex-vulkan-off/vk_microbench.exe) { build/codex-vulkan-off/vk_microbench.exe }
```

- [ ] **Step 3: Check OpenCL-free link command**

```powershell
ninja -C build/codex-vulkan-off -t commands test_vulkan_runtime_standalone | Select-String 'OpenCL|motifcl_vulkan_runtime|test_vulkan_runtime_standalone'
```

Expected: link command includes `libmotifcl_vulkan_runtime.a` and no `OpenCL::OpenCL`/OpenCL library.

- [ ] **Step 4: Run diff hygiene**

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; status lists only intentional files.

- [ ] **Step 5: Update status docs**

Update `docs/VULKAN_PORT_PROTOCOL.md` with exact commands and outputs from Steps 1-4. State performance honestly: current generated scalar shaders are correctness witnesses, not optimized throughput.
