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
- embedding gather (`nn::Embedding::forward`) + `token_position_embedding` (token + position tables) — forward И backward (embedding weight gradient, position gradient) device-resident на Vulkan-тензорах
- RoPE: `rope` (interleaved) с forward+backward через inverse-flag reuse, `rope_split_half`, `rope_positions`, `rope_positions_split_half` — все четыре варианта device-resident на Vulkan-тензорах
- dispatch capture/replay (без `cl_mem`), батчированный тренировочный шаг (invariant 9)

OpenCL остаётся optional legacy backend (регрессионный паритет + непортированные пути); `-DMOTIFCL_ENABLE_OPENCL=OFF` собирает полную библиотеку со стабом загрузчика. Остающиеся transition-гэпы: causal/masked/windowed attention, quant-layouts кроме scalar-scale Q8, `apply_update_seed` (dense grad_w), OpenCL `GraphExecutor` rebinding/арены (Vulkan-реплей — fixed-binding), dropout и другие не перечисленные выше опы. Отдельно: scalar-scale Q8 matmul и standalone vector-API хелперы всё ещё идут через legacy one-shot dispatch (per-call pipeline, generated SPIR-V) — корректны, но не перенесены на кэшированный путь; это помеченный transition-остаток, не «done» по перф-инвариантам.

Python bindings now expose `Backend.vulkan()`, `Backend.create()` runtime selection via `MOTIFCL_BACKEND=vulkan|vk`, `Backend.kind`, `is_vulkan()`, and `is_opencl()`.  Local verification note: current machine has no `pybind11` Python package/dev config, so the CMake Python target is skipped by configure and the binding source was not compiled here.

Latest verification (2026-07-03, RX 580):

- Vulkan device: Radeon RX 580 Series; `VK_KHR_8bit_storage` with `storageBuffer8BitAccess=true`, `uniformAndStorageBuffer8BitAccess=true`; timestampPeriod 40 ns, timestampValidBits 64.
- ON-сборка: `cmake -S . -B build/port-vk -G Ninja -DCMAKE_CXX_COMPILER=C:/Strawberry/c/bin/g++.exe -DCMAKE_BUILD_TYPE=Release -DMOTIFCL_BUILD_TESTS=ON -DMOTIFCL_BUILD_PYTHON=OFF -DMOTIFCL_BUILD_EXAMPLES=OFF -DMOTIFCL_BUILD_TOOLS=OFF -DMOTIFCL_BUILD_BENCHMARKS=ON -DMOTIFCL_WARNINGS_AS_ERRORS=OFF`; `ctest --test-dir build/port-vk --output-on-failure --timeout 300` — 34/34 passed (включая новые `test_vulkan_train_step`, `test_split_qkv_train_repro`).
- OFF-сборка: команды в Slice 5 выше — 34/34 (Vulkan passed / OpenCL skipped), grep линковки standalone чист.
- Perf-записи: `reports/vulkan-perf/SUMMARY.md` (регенерируется `build/port-vk/benchmarks/bench_vulkan_perf.exe` из корня; медиана 50 прогонов после 5 warmup; Vulkan GPU-время дополнительно из timestamp queries). Все опы PASS против бюджетов из docs/PORT_PROMPT.md; исторический скалярный microbench-результат 83.79 ms на 64^3 matmul устранён кэшем пайплайнов + tiled-кернелами (теперь ~0.2 ms wall, ~7 us GPU).
- Инвариант 6/7/9 доказан флэт-таймингом 1000 реплеев в `test_vulkan_runtime_standalone` (вторая половина <= 1.5x первой).

Latest verification (2026-07-05, embedding + RoPE slices E1–E2):

- ON-сборка (`build/port-vk`): `ctest --test-dir build/port-vk --timeout 300` — **36/36 passed** (включая расширенные `test_vulkan_backend` / `test_vulkan_runtime_standalone` с embedding-gather/weight-backward/token+pos и четырьмя RoPE-вариантами, forward+backward parity vs CPU reference, inverse-roundtrip).
- OFF-сборка (`build/codex-vulkan-off`, `-DMOTIFCL_ENABLE_OPENCL=OFF`): **36/36 passed** (Vulkan passed, OpenCL skipped), `ninja -t commands test_vulkan_runtime_standalone | grep -ci opencl` → 0 (OpenCL-free standalone witness, инвариант 2).

## Memory-native method на Vulkan (Slices R1-R3, 2026-07-05)

**Контекст:** memory-native training method (см. `docs/MEMORY_NATIVE_TRAINING_METHOD.md`) — две
совместные компоненты: (A) finite-state counter synapse (`nn::CounterStateLinear`, 0.75 байт/вес
packed state, fused in-backward update); (B) reversible activations (forward без хранения
активаций, backward через inverse-recovery + recompute). Цель — обе компоненты полностью
работают на Vulkan-тензорах без OpenCL-контекста.

**Статус:** ВЫПОЛНЕНО для Linear/Counter coupling; attention coupling требует batched+causal
Vulkan GQA (отдельный план). Все инварианты 1-10 из `docs/PORT_PROMPT.md` выполнены для
каждого среза.

### Slice R1 — `sub` Vulkan device path

Требовался для inverse coupling: `x2 = y2 − G(y1)`, `x1 = y1 − F(x2)`. Без Vulkan-пути `sub`
обратное восстановление входов в reversible backward уходило бы в OpenCL.

- `kernels/vulkan/sub_f32.comp` — elementwise coalesced, 1-element-per-lane (local_size=64).
- `src/runtime/vulkan_backend.cpp` — `run_vulkan_sub` (device-resident, `dispatch_cached`).
- `src/ops/basic_ops.cpp` — `sub()` теперь имеет `is_vulkan()` branch + `vulkan_sub_supported`
  gate (зеркало `add`). Autograd через `SubBackward` (scale(grad, -1) на `b`).
- Standalone vector-API overload намеренно опущен (host-staging shim не на hot path;
  см. комментарий в `vulkan_backend.cpp`).

Witness (R1):
- `tests/test_vulkan_runtime_standalone.cpp` — OpenCL-free (инвариант 2), device-resident
  parity vs CPU (инвариант 5), pipeline-cache через 1000-replay flat-timing (инвариант 6/7/9).
- `tests/test_vulkan_backend.cpp` — Tensor-level parity через `motifcl::sub`.
- Perf: `reports/vulkan-perf/sub_f32.json` — vk p50 176.6 us, gpu 29.44 us, opencl 302.5 us,
  ratio 1.71×, бюджет 0.6 PASS.

### Slice R2 — `nn::ReversibleBlock` Module

- `include/motifcl/nn/reversible.hpp`, `src/nn/reversible.cpp` — Module с
  `std::pair<Tensor,Tensor> forward(x1, x2)` (non-virtual; базовый `forward(const Tensor&)`
  падает через `MCL_CHECK`).
- Реализация: forward под `NoGradGuard` (активации F/G не хранятся); после NoGrad attach
  per-output `ReversibleBackwardNode` с `part`-дискриминатором (как `QKVSplitBackwardNode` —
  без изменения базового `autograd::Node`). Backward накапливает оба grad'а (через
  shared `PendingGrads`), затем под `NoGradGuard` делает inverse-recovery
  (`x2r = sub(y2, g->forward(y1)); x1r = sub(y1, f->forward(x2r))`), recompute-forward под
  `autograd::set_enabled(true)` + `IsolatedBackwardScope` (свежий `BackwardEngine`), и
  диспатчит восстановленные грады в исходные `x1`/`x2` через стандартный `x.backward(...)`.
- `include/motifcl/autograd/node.hpp` + `src/tensor/tensor.cpp` + `src/autograd/tape.cpp` —
  новый `autograd::IsolatedBackwardScope` (RAII save/clear/restore `g_active_backward_engine`
  через внутренние accessor'ы `_save_and_clear_active_backward_engine` /
  `_restore_active_backward_engine`). Необходим т.к. `BackwardEngine::run` оборачивает topo-loop
  в `NoGradGuard` (`src/tensor/tensor.cpp:75`) — без scope nested backward в Node::backward
  либо не запускался бы (filled pending map активного engine, не содержащего свежий граф),
  либо не мог бы attach grad_fn к forward-опам recompute.

Witness (R2):
- `tests/test_vulkan_reversible.cpp` — 2-block reversible stack (Linear+GELU coupling),
  parity vs grad-enabled reference, rel-err ~3e-5 (tolerance 1e-4). OpenCL-free.

### Slice R3 — Counter coupling (memory-native full A+B на Vulkan)

End-to-end memory-native training loop на Vulkan без OpenCL: reversible stack с
`nn::CounterStateLinear` coupling.

Witness (R3):
- `tests/test_vulkan_memory_native.cpp`:
  - Witness A: single `CounterStateLinear` teacher-recovery на Vulkan — тернарный teacher
    восстанавливается, last_loss → 0, ternary-acc 100% (бит-в-бит паритет с OpenCL regression
    `test_counter_state.cpp`).
  - Witness B: 4-block reversible stack с counter coupling, loss 7.60 → 0.064 за 50 шагов
    (100× уменьшение) — pipeline (forward + recompute-backward + counter state update)
    полностью на Vulkan device-resident тензорах.

### Memory-native — что НЕ сделано (явно)

- Attention coupling: `nn::multihead_attention` OpenCL-only; Vulkan-GQA покрывает только
  batch=1 non-causal. Для reversible attention coupling нужен batched+causal Vulkan GQA
  (отдельный план).
- `apply_update_seed` reference path (`dense grad_w`): намеренно OpenCL-only; per
  `docs/MEMORY_NATIVE_TRAINING_METHOD.md` §2.7 memory-native метод не материализует ∇W, и
  reference path нужен только для debug/unit-тестов.
- Числовой замер пиковой памяти активаций (сейчас — качественный): forward в NoGrad не хранит
  активации между forward и backward; backward создаёт кратковременный recompute-graph.
- Concat/slice ops: обойдены pair API; нужны только если reversible block стекать в
  `nn::Sequential` (требует single-Tensor forward).

## Embedding + RoPE на Vulkan (Slices E1–E2, 2026-07-05)

**Контекст:** `nn::Embedding::forward` (token embedding gather), `token_position_embedding`
(token + position tables, используется transformer forward), и четыре варианта RoPE
(`rope`, `rope_split_half`, `rope_positions`, `rope_positions_split_half` — выбираются
`ModernSelfAttention::apply_rope`/`apply_rope_positions` по конфигу) были последним
блоком transformer forward, остававшимся на OpenCL-пути `kernels.get(...)` +
`.buffer()`. Эти срезы переносят их на Vulkan-тензорное, device-resident API с
cached-pipeline dispatch и full autograd, тем же шаблоном что `sub`/`add`/`rmsnorm`
(Slices R1–R4).

**Статус:** ВЫПОЛНЕНО. Все инварианты 1–5 из основного protocol выполнены для каждого
среза.

### Slice E1 — embedding (gather + weight backward + token+position + position backward)

- `kernels/vulkan/embedding_gather_f32_i32.comp` — elementwise coalesced gather
  (1-element-per-lane, `local_size=64`), бит-в-бит с
  `kernels/embedding.cl:embedding_gather_f32_i32` (0 на OOB индексе).
- `kernels/vulkan/embedding_weight_backward_f32_i32.comp` — per-`(vocab, d)` lane с
  loop over `token_count`, суммирует град в выбранные строки. Зеркало
  `kernels/embedding.cl:embedding_weight_backward_f32_i32`.
- `kernels/vulkan/token_position_embedding_f32_i32.comp` — gather token + position
  table, `pos = token_linear % seq_len`, бит-в-бит с OpenCL-кернелом.
- `kernels/vulkan/position_embedding_backward_f32_i32.comp` — суммирует град по
  batch-измерению в position table; host пред-zero'ит таблицу (контракт
  `position_shape[0] >= seq_len`).
- `src/runtime/vulkan_backend.cpp` — `run_vulkan_embedding_gather`,
  `run_vulkan_embedding_weight_backward`, `run_vulkan_token_position_embedding`,
  `run_vulkan_position_embedding_backward` через `dispatch_cached` против embedded
  SPIR-V (`vulkan_spirv_kernels.inc`); validate shapes, buffer sizes, конечность
  гиперпараметров.
- `src/nn/embedding.cpp` — `Embedding::forward`, `token_position_embedding`, и обе
  backward helper'а (`embedding_weight_backward`, `position_embedding_backward`)
  имеют `if (backend.is_vulkan())` ветку поверх существующего OpenCL-пути; autograd
  nodes (`EmbeddingBackwardNode`, `TokenPositionEmbeddingBackwardNode`) сохранены.

Witness (E1):
- `tests/test_vulkan_runtime_standalone.cpp` — OpenCL-free (инвариант 2),
  device-resident parity vs CPU reference (инвариант 5) для всех четырёх операций;
  weight backward OOB-row zeroing, position backward batch-summation.
- `tests/test_vulkan_backend.cpp` — Tensor-level parity: `nn::Embedding` forward +
  `weight.backward(grad_out)` vs CPU gather/scatter, `token_position_embedding`
  forward + backward (token + position grads) vs CPU.

### Slice E2 — RoPE (interleaved + split-half, fixed-offset + per-token positions)

- `kernels/vulkan/rope_f32.comp` — interleaved pair layout с `inverse` push-const flag;
  переиспользуется для backward через инверсию знака угла (тот же трюк, что OpenCL
  `rope_impl` helper). Бит-в-бит с `kernels/attention.cl:rope_f32`.
- `kernels/vulkan/rope_split_half_f32.comp` — split-half layout (первые `head_dim/2`
  вращаются против вторых), с `inverse` flag; зеркало `rope_split_half_f32`.
- `kernels/vulkan/rope_positions_f32.comp` — per-token i32 positions table (forward
  only, соответствует OpenCL — backward отсутствует).
- `kernels/vulkan/rope_positions_split_half_f32.comp` — split-half + positions
  (forward only).
- `src/runtime/vulkan_backend.cpp` — `run_vulkan_rope`, `run_vulkan_rope_positions`,
  `run_vulkan_rope_split_half`, `run_vulkan_rope_positions_split_half` через
  `dispatch_cached`; validate `head_dim*n_head == channels`, положительность theta.
- `src/ops/attention.cpp` — `rope_impl` (покрывает `rope` + `rope_split_half` через
  `split_half`/`inverse` флаги) и `rope_positions`/`rope_positions_split_half` имеют
  `if (x.backend().is_vulkan())` ветки; `RopeBackwardNode::backward` переиспользует
  ту же ветку через `inverse=true`.

Witness (E2):
- `tests/test_vulkan_runtime_standalone.cpp` — OpenCL-free, device-resident: rope
  forward parity, rope inverse roundtrip (`rope(rope(x), inverse=true) == x`),
  rope_split_half / rope_positions / rope_positions_split_half forward parity.
- `tests/test_vulkan_backend.cpp` — Tensor-level: `rope` forward + backward (vs CPU
  ref, угол — позиция в последовательности `t`, не flat row), `rope_split_half`,
  `rope_positions`, `rope_positions_split_half` forward parity vs CPU refs.

### Embedding/RoPE — что НЕ сделано (явно)

- Fused decode kernels (`qk_norm_rope_decode_f32`,
  `qk_norm_rope_cache_append_decode_f32`, `rope_cache_append_decode_f32`) —
  inference-only оптимизации, остаются OpenCL legacy (помечены в remaining gaps).
- Quantized embedding (`embedding_gather_transposed_q4_k_i32` /
  `q5_k_i32`) — отдельный quant-layout срез.
- `rope_positions` / `rope_positions_split_half` backward — отсутствует в OpenCL,
  Vulkan соответствует этому scope.
- Dropout (отдельный gap).

## Bug-audit fixes (Slice F: scalar ops + OpenCL-only guards, 2026-07-05)

Полное ревью Vulkan-порта (см. `docs/superpowers/plans/` и git history) выявило
класс latent-багов: ops, которые на Vulkan-тензорах тихо доходили до `kernels.get(...)`
и падали на null OpenCL context, либо пропускали validation. Slice F закрывает
самые критичные находки ревью.

### F1 — Elementwise scalar device-path (`scale`/`mul_scalar`/`add_scalar`)

`SubBackward`/`MulBackward`/`DivBackward`/`ScalarBackward`/`DropoutBackward(p==0)`
все вызывают `scale`/`mul_scalar`, у которого раньше не было Vulkan-ветки —
backward chain падал на Vulkan-тензорах. Добавлено:

- `kernels/vulkan/mul_scalar_f32.comp`, `kernels/vulkan/add_scalar_f32.comp` —
  elementwise coalesced (1-element-per-lane, `local_size=64`), бит-в-бит с
  `kernels/basic.cl`.
- `run_vulkan_mul_scalar`/`run_vulkan_add_scalar` (vulkan_backend.cpp) —
  `dispatch_cached`, validate finite scalar + element count + buffer sizes.
- `src/ops/basic_ops.cpp::elementwise_scalar` теперь имеет `is_vulkan()` ветку.

Witness: `tests/test_vulkan_backend.cpp` — `scale`/`add_scalar` forward parity
vs CPU; **`sub` backward с `b.requires_grad=true`** (chain `SubBackward → scale(-1)`)
теперь проходит на Vulkan (раньше crash).

### F2 — Явные guards для OpenCL-only ops (CRITICAL review findings)

Раньше эти ops падали с null-context crash на Vulkan-тензорах. Теперь они
`MCL_CHECK(!is_vulkan(), ...)` с понятным сообщением, указывающим обход
(use OpenCL backend / use_bias=false / dropout_p=0). Список:

- `nn::Embedding::forward` quantized-transposed (Q4_K/Q5_K) gather (M3).
- `add_bias_rows` (C1) — закрывает `nn::Linear(use_bias=true)` на Vulkan.
- `mul`/`div` binary elementwise через `elementwise_binary` (C2 partial).
- `scale_inplace`/`add_inplace` (in-place нужен отдельный device kernel —
  readonly+writeonly aliasing в одном Vulkan dispatch это UB).
- `qkv_split` (C3).
- `kv_cache_append` / `kv_cache_append_positions` / `paged_kv_cache_append` /
  `paged_grouped_query_attention` (C5/C6).
- `dropout(p>0)` (C6).
- `unary`/`unary_backward_kernel` → покрывает `relu`/`silu`/`exp`/`sqrt`/`rsqrt`
  и `relu_backward` (gelu имеет собственные Vulkan-ветки, не страдает) (H4).
- `sum_rows` / `mul_rows` / `add_bias_gelu_rows` (H5).
- Fused decode predicates `can_use_fused_qk_norm_rope_decode`,
  `can_use_fused_qk_norm_rope_cache_append_decode`,
  `can_use_fused_rope_cache_append_decode`,
  `can_use_fused_packed_qkv_q4_0_decode`,
  `can_use_fused_packed_swiglu_q4_0_decode` — все возвращают false на Vulkan
  тензорах (C4).

Witness: `tests/test_vulkan_backend.cpp::expect_vulkan_guard` — 11 ops
проверены на то, что они громко `MCL_CHECK`-падают с упоминанием Vulkan (а не
crash'ат на null context).

### F3 — Validation gaps

- `run_vulkan_compact_counter_apply_update_fused` теперь проверяет
  `std::isfinite(lr/lr_scale/rms_beta)` и `rms_eps >= 0` (раньше NaN/Inf
  тихо отравляли весь update) (M4).
- `run_vulkan_i8_scaled_matmul` теперь reject'ит вызов при активном
  `batch_active()` / `capture_active()` — `dispatch_storage_buffers` делает
  собственный submit и не участвует в batch/capture recording (H1).
- `run_vulkan_rope[_positions][_split_half]` reject'ят shape product overflow
  перед `static_cast<uint32_t>((total+63)/64)` (L1).
- `run_vulkan_embedding_gather` / `run_vulkan_embedding_weight_backward` /
  `run_vulkan_token_position_embedding` reject'ят `int32` push-constant
  overflow для `n = a*b` products (L2).

### Slice F — что НЕ сделано (явно)

Реальные Vulkan device-kernels для перечисленных в F2 ops — это следующий срез
(отдельные .comp + run_vulkan_* + wiring). Slice F только превращает silent
crash в громкое `MCL_CHECK` с понятным сообщением + закрывает backward chain
через scalar device-path. Пользователь может:

- использовать OpenCL backend для неподдерживаемых ops, либо
- убрать проблемную фичу (`use_bias=false`, `dropout_p=0`, не-Q4_K embedding).

Latest verification (2026-07-05, после Slice F):

- ON-сборка (`build/port-vk`): `ctest --test-dir build/port-vk --timeout 300` —
  **36/36 passed** (включая новые scalar-parity + sub-backward-chain + 11
  guard-enforcement свидетелей в `test_vulkan_backend`, и mul_scalar/add_scalar
  device-resident свидетелей + non-fine-validation reject'ы в
  `test_vulkan_runtime_standalone`).
- OFF-сборка (`build/codex-vulkan-off`): **36/36 passed** (Vulkan passed /
  OpenCL skipped), `ninja -t commands test_vulkan_runtime_standalone | grep -ci opencl` → **0**.
