# Слайс Q2 — интеграция квант-ядра в Tensor/ops (Vulkan quant inference end-to-end)

Пререквизит: слайс Q1 принят (кернелы quantize_q8_rowwise / dequant q8/q4 scaled / mm_q8q8 / mm_q8q4 + диспетчеры run_vulkan_* + standalone-тесты зелёные).

Цель: `Linear::enable_quantized_inference(Q8_0|Q4_0)` работает на Vulkan-бэкенде end-to-end без OpenCL и без скрытого фолбэка.

## Обязательное чтение
1. `.hermes.md`, `docs/superpowers/plans/vulkan-quant-q1.md` (что уже есть).
2. `src/ops/quant.cpp` — существующие OpenCL-пути (формулы choose_*_scale/choose_axis_scales, семантика _set_quant_scales).
3. `src/ops/matmul.cpp` — диспетчер matmul() и место квант-веток; как выбирается vulkan-путь для f32 (vulkan_matmul_f32_supported).
4. `src/nn/linear.cpp` — квант-ветка forward (НЕ менять её).
5. `tests/test_quant.cpp` — допуски и паттерны parity.
6. `tests/test_vulkan_backend.cpp` — стиль тестов Tensor-уровня на Vulkan.

## Изменения

### 1. src/ops/quant.cpp — Vulkan-ветки
- `quantize_q8_symmetric_rows(x)`: если `x.backend().is_vulkan()` — НЕ звать OpenCL kernels.get; вызвать `run_vulkan_quantize_q8_rowwise` (device-скейлы, без host readback): создать out Tensor Q8_0 [M,K] и scales Tensor F32 [M] (Tensor::empty на том же бэкенде), диспатч, `out._set_quant_scales(scales, /*axis=*/0, 0)`, `autograd::record_op` как в OpenCL-ветке.
- `quantize_q8_symmetric_cols(x)` и `quantize_q4_symmetric_cols(x)`: Vulkan-ветка через ХОСТ (веса квантуются один раз при enable_quantized_inference — скорость не критична): `x.to_vector<float>()` → посчитать col-scales (формула choose_axis_scales axis=1) → кванты на хосте (ФОРМУЛА раундинга — та же, что в kernels/quant.cl scaled-кернелах; для Q4_0 упаковка нибблов как в Q1) → `Tensor::from_cpu(..., Q8_0/Q4_0, packed_data)` + scales from_cpu + `_set_quant_scales(scales, 1, 0)`.
- `dequantize_q8(x)` / `dequantize_q4(x)`: Vulkan-ветка: has_quant_scales с axis 0/1 → `run_vulkan_dequantize_q8_scaled` / `_q4_scaled`; скалярный scale (_set_quant_scale) → допустимо через host (to_vector байтов → host dequant → from_cpu), это тестовый путь.
- ВСЕ остальные квант-варианты на Vulkan (axis=2/blocks, quantize_q8_symmetric без axis, axis-функции с axis!=нужного): `MCL_CHECK(false, "... not ported to Vulkan yet")` — явный отказ, НИКАКОГО тихого фолбэка (протокол порта, .hermes.md).

### 2. src/ops/matmul.cpp — Vulkan квант-ветки в matmul()
- В начале квант-диспетчера (до OpenCL веток): если `a.backend().is_vulkan()`:
  - a.dtype()==Q8_0 && b.dtype()==Q8_0 && a scales axis==0 && b scales axis==1 → `run_vulkan_matmul_q8q8_scaled`, выход f32 [M,N].
  - a.dtype()==Q8_0 && b.dtype()==Q4_0 && те же оси → `run_vulkan_matmul_q8q4_scaled`.
  - Любая другая квант-комбинация на Vulkan → `MCL_CHECK(false, "vulkan quant matmul supports Q8_0[rows]xQ8_0/Q4_0[cols] only (yet)")`.
- Семантику autograd скопировать с OpenCL квант-пути (квант-матмул — inference-only; проверь, что делает OpenCL ветка: record_op и/или запрет при is_enabled — повтори 1:1).
- Валидация как в validate_scaled_quant_matmul_tensor — используй её же, если применима.

### 3. Linear — НЕ МЕНЯТЬ
`Linear::forward` квант-ветка должна заработать сама поверх 1+2. Если не работает — разберись ПОЧЕМУ и почини в quant.cpp/matmul.cpp, а не в linear.cpp.

### 4. Витнес — tests/test_vulkan_backend.cpp (добавить блок)
1. `quantize_q8_symmetric_rows` на Vulkan: parity квант-байтов и скейлов vs host-эмуляция (bit-identical), включая строку из нулей (scale=1).
2. `matmul(q8_rows, q8_cols)` parity vs host dequant→naive f32 matmul: M=33,K=47,N=65 и M=1,K=128,N=96. Допуск как в test_quant.cpp для соответствующих кейсов (посмотри и возьми их значения).
3. `matmul(q8_rows, q4_cols)` — те же шейпы (N=65 покрывает нечётный packed-хвост).
4. Linear end-to-end: два Linear (bias on/off), enable_quantized_inference(Q8_0) затем отдельный с (Q4_0), forward [5×K] на Vulkan; сравнить с f32-forward того же слоя ДО квантизации, допуск квант-погрешности (подбери по факту, старт 3e-2 relative max-diff; зафиксируй фактический max diff в отчёте).
5. Негатив: квант-матмул с axis=2 скейлами на Vulkan → выброс MCL error (expect_motifcl_error паттерн из test_quant.cpp).

### 5. Сборка и прогоны (вывод в отчёт)
- `cmake --build build/rx580-release --parallel 12`
- `ctest --test-dir build/rx580-release -R "test_vulkan_backend" --output-on-failure --timeout 300`
- `ctest --test-dir build/rx580-release -R "vulkan|quant" --output-on-failure --timeout 300`
- Полный: `ctest --test-dir build/rx580-release --output-on-failure --timeout 300` (регрессии OpenCL-путей недопустимы).

## Запреты
- Не менять linear.cpp, existing кернелы, Q1-файлы (если Q1-диспетчер неудобен — СТОП и отчёт, поправим Q1 отдельно).
- Никакого тихого OpenCL/host фолбэка в vulkan-путях matmul/quantize_rows.
- Git — только read-only.

## Формат отчёта
1. Файлы + суть правок.
2. Хвосты фактического вывода всех прогонов.
3. Фактический max-diff в Linear end-to-end тесте.
4. Что НЕ проверено.
