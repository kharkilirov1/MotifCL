# Слайс Q1 — Vulkan quant core: кернелы + диспетчеры + standalone-витнес

Цель: низкоуровневое ядро квант-инференса на Vulkan (Q8_0 активации × Q8_0/Q4_0 веса, axis-scales), OpenCL-free, по протоколу порта. Интеграция в Tensor/ops — СЛЕДУЮЩИЙ слайс, здесь только VulkanBuffer-уровень.

## Обязательное чтение перед кодом
1. `.hermes.md` (корень) — правила сборки и уроки RX 580.
2. `kernels/quant.cl` — формулы квантизации: `quantize_f32_to_q8_0_rowwise_fused` (строка ~93), `dequantize_q8_0_to_f32_scaled` (~131), `dequantize_q4_0_to_f32_scaled` (~174). Точная формула rounding и nibble-упаковки Q4_0 — ПЕРЕПИСАТЬ, не выдумывать.
3. `kernels/matmul.cl` — только `matmul_q8_0_scaled_f32` (~3447) и `matmul_q8_q4_scaled_f32` (~3501): как читаются scales по axis-режимам.
4. Образцы стиля Vulkan-кернелов: `kernels/vulkan/mm_f32_nn.comp` (16x16 тайл, push constants), `kernels/vulkan/rmsnorm_f32.comp` (WG-редукция).
5. `tools/gen_vulkan_spirv.py` — как кернелы попадают в `src/runtime/vulkan_spirv_kernels.inc`.
6. `include/motifcl/runtime/vulkan_backend.hpp` — паттерн деклараций `run_vulkan_*` (см. run_vulkan_f32_matmul, run_vulkan_embedding_weight_backward_scatter как образцы валидации/push/dispatch).
7. НЕ трогать legacy `i8_scaled_matmul_spirv` / `run_vulkan_i8_scaled_matmul` — старый per-shape путь, оставить как есть.

## Scale-конвенция слайса (упрощённая, под путь Linear)
- Активации A: int8 [M,K], scales_a f32 [M] (per-row, axis=0).
- Веса B: int8/packed-int4 [K,N], scales_b f32 [N] (per-col, axis=1).
- C[m,n] = scales_a[m] * scales_b[n] * sum_k( a[m,k] * b[k,n] ) — аккумулятор float, значения int8 конвертируются во float при загрузке тайла.
- Q4_0: упаковка нибблов и смещение — ТОЧНО как в quant.cl (проверь по dequantize_q4_0: какой ниббл младший, каков сдвиг знака).

## Новые файлы кернелов (kernels/vulkan/)
1. `quantize_q8_rowwise_f32.comp` — in f32 [M,K] → out int8 [M,K], scales f32 [M]. Один WG (local_size_x=64) на строку: shared-редукция max|x| по строке, потом квант всей строки. Формула квант-раундинга — из quant.cl rowwise_fused (включая обработку max_abs==0 → scale=1).
2. `dequantize_q8_scaled_f32.comp` — int8 + scales + push(mode: 0=scalar-в-scales[0],1=rows,2=cols) → f32. Для parity-тестов.
3. `dequantize_q4_scaled_f32.comp` — packed q4 → f32, те же режимы.
4. `mm_q8q8_scaled_f32.comp` — тайловый 16x16 (как mm_f32_nn): грузим тайлы A(int8) и B(int8) в shared как float, аккум float; на выходе умножение на scales_a[row]*scales_b[col]. Push: M,K,N.
5. `mm_q8q4_scaled_f32.comp` — то же, но B packed q4 [K,N]: байт на два соседних n (или как в референсе — проверь по matmul_q8_q4_scaled_f32, где именно пакуется пара: по n или по k!). Распаковка при загрузке тайла.
- Требование шейдеров: `#extension GL_EXT_shader_8bit_storage : require` (для int8_t SSBO) — работает на цели (см. .hermes.md). Буфер q4 — uint8_t.
- Все кернелы: guard `if (gid >= total) return;` и корректные хвосты при M,K,N не кратных тайлу.

## SPIR-V
- Добавить кернелы в tools/gen_vulkan_spirv.py по существующей схеме, запустить `python tools/gen_vulkan_spirv.py`, убедиться что src/runtime/vulkan_spirv_kernels.inc перегенерился и содержит новые символы `k_<name>` / `k_<name>_words`. Если glslc не найдётся — остановись и сообщи, НЕ пиши .inc руками.

## Диспетчеры (src/runtime/vulkan_backend.cpp + декларации в hpp)
По образцу существующих (fail-lambda, валидация nbytes, push, dispatch_cached, ceil-div):
1. `run_vulkan_quantize_q8_rowwise(runtime, in_f32, out_i8, out_scales, M, K)` — группы: M (по WG на строку).
2. `run_vulkan_dequantize_q8_scaled(runtime, in_i8, scales, out_f32, count, mode, rows, cols)`.
3. `run_vulkan_dequantize_q4_scaled(...)` — count элементов (не байт), валидация packed-размера ceil(count/2).
4. `run_vulkan_matmul_q8q8_scaled(runtime, a_i8, a_scales, b_i8, b_scales, c_f32, M, K, N)`.
5. `run_vulkan_matmul_q8q4_scaled(...)` — валидация b: ceil по упаковке.
- int32 push-overflow проверки как в образцах (kMaxInt32).

## Витнес (tests/test_vulkan_runtime_standalone.cpp — OpenCL-free по протоколу!)
Добавить блок тестов по образцу существующих в этом файле:
1. Host-эмуляция в тесте: quantize_q8_rowwise (C++), dequant, наивный quant-matmul — маленькие функции прямо в тесте.
2. quantize parity: сравнение квант-БАЙТОВ и scales device vs host BIT-IDENTICAL (формула одна).
3. mm_q8q8 parity: M=33,K=47,N=65 (пересекает тайлы, хвосты) + M=1,K=128,N=96. Допуск: |diff| <= 1e-4 * max(1,|ref|) (порядок суммирования тайла отличается).
4. mm_q8q4 parity: те же шейпы. Плюс нечётное N (packed-хвост) — например N=65 уже нечётное, хорошо.
5. dequant q8/q4 parity: точное сравнение float.
6. Негатив: неверные размеры буферов → fail (по образцу invalid_q8 теста ~строка 81).

## Сборка и прогон (обязательно, вывод в отчёт)
- `cmake --build build/rx580-release --parallel 12`
- `ctest --test-dir build/rx580-release -R "test_vulkan_runtime_standalone" --output-on-failure --timeout 300`
- Затем полный vulkan-набор: `ctest --test-dir build/rx580-release -R vulkan --output-on-failure --timeout 300`

## Запреты
- Не менять существующие кернелы/диспетчеры/тесты (только добавлять).
- Не трогать OpenCL файлы вообще.
- Никаких git-команд кроме read-only.
- Если что-то не получается после 2 попыток — СТОП и отчёт, не изобретать обходы.

## Формат отчёта
1. Список файлов с суммарным diff-объёмом.
2. Фактические хвосты вывода: gen_vulkan_spirv.py, сборка, оба ctest.
3. Что НЕ проверено / известные ограничения.
