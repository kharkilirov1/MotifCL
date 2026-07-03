# Vulkan port protocol

Цель: перевести MotifCL с OpenCL-first runtime на Vulkan-first/Vulkan-only runtime без скрытого OpenCL fallback в путях, которые заявлены как портированные.

## Инварианты портирования

1. **Не считать OpenCL-backed Tensor путём Vulkan-портом.** Если операция скачивает данные из OpenCL tensor на host, запускает standalone Vulkan helper и возвращает результат через `Tensor::from_cpu(opencl_backend, ...)`, это transition shim, а не Vulkan-native runtime.
2. **Каждый портовый срез должен иметь OpenCL-free witness.** Минимальный witness — target/test, который не линкуется с `OpenCL::OpenCL` и не включает `motifcl.hpp`, `Backend::create_opencl`, `Tensor`, `Buffer`, `Kernel`, `Program` или `OpenCLContext`.
3. **Fallback должен быть явным.** Экспериментальные `MOTIFCL_*_BACKEND=vulkan` paths могут временно fallback to OpenCL, но Vulkan-native tests должны покрывать отдельный OpenCL-free путь.
4. **Операция считается Vulkan-native только после device-resident I/O.** Входы и выходы должны жить в Vulkan buffer allocation/descriptor set path; host roundtrip допускается только для тестового upload/download boundary.
5. **Полное удаление OpenCL делается последним.** До переноса `TensorStorage`, autograd buffers, kernel dispatch и основных ops OpenCL остаётся legacy backend для regression parity.

## Срезы

### Slice 0 — standalone Vulkan runtime target

Статус: выполнен и проверен в `build/codex-study-gcc`.

Требования:

- `motifcl_vulkan_runtime` собирается из Vulkan runtime source без OpenCL link dependency.
- `test_vulkan_runtime_standalone` линкуется только с `motifcl_vulkan_runtime`.
- Test проверяет Vulkan loader/probe, validation failures и compute smoke when device is available.

Witness:

- `cmake --build build/codex-study-gcc --target test_vulkan_runtime_standalone --parallel 4`
- `ctest --test-dir build/codex-study-gcc -R "test_vulkan_runtime_standalone" --output-on-failure --timeout 90`
- `ninja -C build/codex-study-gcc -t commands test_vulkan_runtime_standalone` shows link line with `libmotifcl_vulkan_runtime.a` and no `OpenCL`.

### Slice 1 — persistent Vulkan runtime objects

Статус: выполнен и проверен в `build/codex-study-gcc`.

Готово:

- `VulkanRuntime` owns a persistent loader/instance/device/compute queue/function table.
- `VulkanRuntime::dispatch_storage_buffers(...)` reuses that device context for upload/dispatch/download.
- `test_vulkan_runtime_standalone` dispatches multiple smoke workloads through one `VulkanRuntime`.
- `VulkanBuffer` is explicit move-only device storage with upload/download helpers and runtime-owned lifetime.
- Persistent-buffer dispatch path binds existing `VulkanBuffer` storage instead of allocating per op.

Вынести из current one-shot helpers reusable objects:

- `VulkanRuntime`
- `VulkanBuffer`
- `VulkanPipeline` / `VulkanCommandRunner` остаются будущей internal split/cache task, но Slice 1 witness уже не зависит от one-shot allocations.

Требование witness:

- Allocate/upload/dispatch/download/free в одном persistent device context.
- No `OpenCL::OpenCL` link for the test.

Current witness:

- `ctest --test-dir build/codex-study-gcc -R "test_vulkan_runtime_standalone" --output-on-failure --timeout 90`
- `ninja -C build/codex-study-gcc -t commands test_vulkan_runtime_standalone` shows no `OpenCL`.

### Slice 2 — Vulkan tensor storage

Статус: выполнен для базового Tensor storage и проверен в `test_vulkan_backend`.

Добавить backend-neutral storage layer:

- `RuntimeBackendKind::{Vulkan, OpenCL}`
- device buffer vtable/variant
- Vulkan `Tensor::empty/from_cpu/to_cpu` path

Требование witness:

- `Tensor::from_cpu(Backend::create_vulkan(), ...)` works without OpenCL context.
- Vulkan `Tensor` paths cover `from_cpu`, `to_cpu`, `matmul`, `softmax_rows`, `rmsnorm`, `swiglu`, and `add` in `test_vulkan_backend`.

### Slice 3 — core ops Vulkan-native

Статус: частично выполнен для forward core ops.

Порядок:

1. elementwise add — выполнено для Vulkan-backed F32 tensors
2. matmul f32 — выполнено для Vulkan-backed F32 tensors (`M=1` and generic)
3. softmax rows — выполнено для Vulkan-backed F32 tensors
4. RMSNorm — выполнено для Vulkan-backed F32 tensors
5. SwiGLU — выполнено для Vulkan-backed F32 tensors
6. attention/GQA — выполнен Vulkan-backed Tensor path для non-causal, unmasked, batch=1 F32 GQA (`head_dim,key_tokens <= 64`); causal/windowed/masked GQA остаются OpenCL legacy/unsupported на Vulkan Tensor.
7. quantized matmul — выполнен Vulkan-backed Tensor path для scalar-scale `Q8_0 x Q8_0 -> F32` через `VK_KHR_8bit_storage`; остальные quant layouts/scaled tensor scales остаются OpenCL legacy.
8. compact counter decode/backward-input — выполнен Vulkan-backed Tensor path для production `CounterStateLinear` U8/3-byte packed state: `decode_weight`, inference `forward` (`matmul_transpose_b`) и `backward_input_from_state`; fused state update пока не перенесён.

Дополнительный проверенный переходный witness:

- `test_vulkan_runtime_standalone` теперь покрывает OpenCL-free staged runtime helpers для:
  - `run_vulkan_grouped_query_attention` — GQA forward в одном generated SPIR-V dispatch без OpenCL.
  - `run_vulkan_i8_scaled_matmul` — quantized API boundary; true byte `int8` storage path when `VK_KHR_8bit_storage`/`uniformAndStorageBuffer8BitAccess` is available, with int32-upload fallback for standalone vector API on devices without that feature.
  - `run_vulkan_compact_counter_backward_input` — fused compact state decode + backward-input SPIR-V dispatch for the standalone `uint32_t` packed layout.
  - `run_vulkan_compact_counter_decode_weight` / `run_vulkan_compact_counter_backward_input_u8` — production U8/3-byte packed `CounterStateLinear` decode/backward-input dispatches on persistent Vulkan buffers.
  - `run_vulkan_compact_counter_increment` — packed 6-bit counter increment primitive как standalone SPIR-V storage-buffer update.
  - `run_vulkan_sgd_update` — direct SPIR-V SGD update, including a persistent `VulkanRuntime`/`VulkanBuffer` overload.

Эти staged helpers являются OpenCL-free Vulkan runtime witnesses, но по инварианту 4 не закрывают full Vulkan-native Tensor/device-resident порт там, где остаётся host staging.

Требование witness для каждой операции:

- Vulkan Tensor inputs/outputs.
- No OpenCL buffer allocation.
- Parity against CPU reference and old OpenCL regression where available.

### Slice 4 — autograd/training

Статус: ВЫПОЛНЕН (2026-07-03) с оптимизированными кернелами (инварианты 6-10 из docs/PORT_PROMPT.md) и перф-записями в `reports/vulkan-perf/`.

Портировано Vulkan-native (device-resident Tensor I/O, cached pipelines, push constants; GLSL источники в `kernels/vulkan/*.comp`, embedded SPIR-V в `src/runtime/vulkan_spirv_kernels.inc`, генератор `tools/gen_vulkan_spirv.py`):

- matmul forward+backward: tiled 16x16 shared-memory NN/NT/TN + wave-per-output M=1 формы; `MatMulBackward` подключён к Vulkan-путям (`dA = dC*B^T`, `dB = A^T*dC`).
- softmax rows forward+backward (wave64-per-row, shared reductions).
- RMSNorm forward+backward_x (wave-per-row) + двухстадийный backward_weight (row-inv scratch + column reduce); residual-вариант скомпонован из backward_x + add.
- SwiGLU и GELU forward+backward (elementwise, формулы бит-в-бит с kernels/activation.cl).
- add + SGD update (elementwise; SGD in-place capable).
- softmax cross-entropy forward (per-row + mean reduce) + backward.
- GQA non-causal batch=1 forward (retrofit: wave-per-(tq,head), shared softmax, key_tokens<=1024) + backward (3 стадии: probs+ds -> dQ -> dK/dV); causal/windowed/masked остаются OpenCL legacy, явно отклоняются на Vulkan-тензорах с MCL_CHECK.
- compact-counter fused update (`apply_update_backward`): row-stats + stochastic tick + scale commit, бит-в-бит с OpenCL при одинаковом seed (проверено сравнением байтов состояния OpenCL vs Vulkan); требует VK_KHR_8bit_storage. `apply_update_seed` (плотный grad_w) остаётся OpenCL-only.
- capture/replay без `cl_mem`: `VulkanRuntime::capture_begin/capture_end/replay` записывает cached-path диспетчи (буферы удерживаются shared ownership) и реплеит их одним command buffer / одним submit. `register_tensor` для Vulkan-тензоров не записывает `cl_mem`. OpenCL `GraphExecutor` (rebinding, арены) остаётся legacy-механизмом OpenCL-пути.
- Инвариант 9 (батчирование): `VulkanRuntime::batch_begin/batch_end` — eager-опы записываются в один primary command buffer с compute-барьерами и submit'ятся один раз; батч удерживает все буферы до fence (временные тензоры autograd безопасно умирают до submit).

Witness (все выполняются и зелёные):

- `ctest --test-dir build/port-vk -R test_vulkan_train_step --output-on-failure` — полный SGD-шаг трансформер-блока (rmsnorm -> qkv matmul -> GQA -> residual -> rmsnorm -> SwiGLU MLP -> residual -> lm_head -> CE loss) на тензорах `Backend::create_vulkan()`, forward одним submit, backward+optimizer вторым; loss падает 3.457 -> 0.376 за 30 шагов; чистый skip (77) без Vulkan-устройства.
- `ctest --test-dir build/port-vk -R test_vulkan_backend` — Tensor-level parity backward (matmul multi-tile, rmsnorm dx/dw, swiglu, gelu, GQA dQ/dK/dV vs CPU-референсы с указанными допусками; counter fused update бит-в-бит vs OpenCL).
- `ctest --test-dir build/port-vk -R test_vulkan_runtime_standalone` — OpenCL-free parity (multi-tile matmul NN/NT/TN/M=1, wave-per-row backward-кернелы, cols<64), плюс capture/replay c 1000-реплеевым flat-timing доказательством инвариантов 6/7/9.
- Perf: `build/port-vk/benchmarks/bench_vulkan_perf.exe` из корня репо -> `reports/vulkan-perf/<op>.json` + `SUMMARY.md`; все опы в бюджете (см. SUMMARY; end-to-end train step ratio ~3.5-3.8x против OpenCL при цели >=0.40).
- memory truth gate (`test_memory_truth_gate`) остаётся зелёным в полном прогоне.

### Slice 5 — make OpenCL legacy/optional

Статус: ВЫПОЛНЕН (2026-07-03).

- `MOTIFCL_ENABLE_OPENCL=OFF` собирает ПОЛНУЮ библиотеку `motifcl` (Tensor/ops/autograd/nn): исходники компилируются против vendored CL-типов, а вместо OpenCL-загрузчика линкуется локальный no-op stub (`src/runtime/opencl_disabled_stub.cpp`); `Backend::create_opencl()` падает штатной ошибкой "no OpenCL platforms found", OpenCL-зависимые тесты корректно скипаются (exit 77).
- Витнесс-команды OFF-сборки (все зелёные на RX 580):
  - `cmake -S . -B build/port-vk-off -G Ninja -DCMAKE_CXX_COMPILER=C:/Strawberry/c/bin/g++.exe -DCMAKE_BUILD_TYPE=Release -DMOTIFCL_ENABLE_OPENCL=OFF -DMOTIFCL_BUILD_TESTS=ON -DMOTIFCL_BUILD_PYTHON=ON`
  - `cmake --build build/port-vk-off -j 6`
  - `ctest --test-dir build/port-vk-off --output-on-failure --timeout 300` — 34/34 (Vulkan-тесты проходят, OpenCL-тесты Skipped).
  - `ninja -C build/port-vk-off -t commands test_vulkan_runtime_standalone | grep -i opencl` — пусто (линковка без OpenCL).
  - Линковка `test_vulkan_train_step`/`test_vulkan_backend` не содержит OpenCL-загрузчика (только собственные исходники `opencl_context.cpp`/`opencl_disabled_stub.cpp` против vendored-заголовков).
- CI: job `vulkan-off-smoke` в `.github/workflows/ci.yml` (OpenCL-off сборка трёх witness-таргетов, grep линковки standalone на отсутствие OpenCL, прогон тестов со skip без GPU).
- Docs: README/SPEC переписаны как "Vulkan-first, OpenCL optional" c legacy-секцией; `Device::type` по умолчанию `DeviceType::Vulkan`.

## Current transition status

Vulkan является первичным backend'ом: persistent runtime (cached pipelines/descriptor pools, device-local pooled буферы со staging, батчированные command buffers, timestamp-тайминг) + Vulkan-backed Tensor storage, forward И backward:

- probe / smoke compute
- F32 matmul: NN, NT (`matmul_transpose_b`), TN (`matmul_transpose_a`), M=1 формы — tiled/wave-оптимизированные, с autograd
- softmax rows fwd+bwd; RMSNorm fwd+bwd (x и weight); SwiGLU fwd+bwd; GELU fwd+bwd; add (с autograd); SGD update
- softmax cross-entropy fwd+bwd
- scalar-scale Q8_0 x Q8_0 matmul (без autograd; остальные quant-layouts — OpenCL legacy)
- non-causal unmasked F32 GQA batch=1 fwd+bwd (causal/windowed/masked — OpenCL legacy, явно отклоняются)
- production `CounterStateLinear` U8: decode/inference/backward-input + fused state update (`apply_update_backward`)
- dispatch capture/replay (без `cl_mem`), батчированный тренировочный шаг (invariant 9)

OpenCL остаётся optional legacy backend (регрессионный паритет + непортированные пути); `-DMOTIFCL_ENABLE_OPENCL=OFF` собирает полную библиотеку со стабом загрузчика. Остающиеся transition-гэпы: causal/masked/windowed attention, quant-layouts кроме scalar-scale Q8, `apply_update_seed` (dense grad_w), OpenCL `GraphExecutor` rebinding/арены (Vulkan-реплей — fixed-binding), embedding/rope/dropout и другие не перечисленные выше опы. Отдельно: scalar-scale Q8 matmul и standalone vector-API хелперы всё ещё идут через legacy one-shot dispatch (per-call pipeline, generated SPIR-V) — корректны, но не перенесены на кэшированный путь; это помеченный transition-остаток, не «done» по перф-инвариантам.

Python bindings now expose `Backend.vulkan()`, `Backend.create()` runtime selection via `MOTIFCL_BACKEND=vulkan|vk`, `Backend.kind`, `is_vulkan()`, and `is_opencl()`.  Local verification note: current machine has no `pybind11` Python package/dev config, so the CMake Python target is skipped by configure and the binding source was not compiled here.

Latest verification (2026-07-03, RX 580):

- Vulkan device: Radeon RX 580 Series; `VK_KHR_8bit_storage` with `storageBuffer8BitAccess=true`, `uniformAndStorageBuffer8BitAccess=true`; timestampPeriod 40 ns, timestampValidBits 64.
- ON-сборка: `cmake -S . -B build/port-vk -G Ninja -DCMAKE_CXX_COMPILER=C:/Strawberry/c/bin/g++.exe -DCMAKE_BUILD_TYPE=Release -DMOTIFCL_BUILD_TESTS=ON -DMOTIFCL_BUILD_PYTHON=OFF -DMOTIFCL_BUILD_EXAMPLES=OFF -DMOTIFCL_BUILD_TOOLS=OFF -DMOTIFCL_BUILD_BENCHMARKS=ON -DMOTIFCL_WARNINGS_AS_ERRORS=OFF`; `ctest --test-dir build/port-vk --output-on-failure --timeout 300` — 34/34 passed (включая новые `test_vulkan_train_step`, `test_split_qkv_train_repro`).
- OFF-сборка: команды в Slice 5 выше — 34/34 (Vulkan passed / OpenCL skipped), grep линковки standalone чист.
- Perf-записи: `reports/vulkan-perf/SUMMARY.md` (регенерируется `build/port-vk/benchmarks/bench_vulkan_perf.exe` из корня; медиана 50 прогонов после 5 warmup; Vulkan GPU-время дополнительно из timestamp queries). Все опы PASS против бюджетов из docs/PORT_PROMPT.md; исторический скалярный microbench-результат 83.79 ms на 64^3 matmul устранён кэшем пайплайнов + tiled-кернелами (теперь ~0.2 ms wall, ~7 us GPU).
- Инвариант 6/7/9 доказан флэт-таймингом 1000 реплеев в `test_vulkan_runtime_standalone` (вторая половина <= 1.5x первой).
