# Слайс Q3 — decode-путь M=1 (f32 × Q4_0) + бенч-кейсы кванта

Пререквизит: Q1+Q2 приняты (квант-ядро + Tensor/ops интеграция, Linear Q8_0/Q4_0 работает на Vulkan).

Цель: быстрый decode-путь инференса: `matmul(x_f32 [1,K], W_q4 [K,N] cols-scales) → f32 [1,N]` на Vulkan + квант-кейсы в bench_vulkan_perf. Это путь `Linear::forward` при M=1 с Q4_0 весом (см. src/nn/linear.cpp:33-42: matmul(x, decode_weight) без квантизации активации).

## Обязательное чтение
1. `.hermes.md`, ТЗ Q1/Q2 (docs/superpowers/plans/).
2. `kernels/matmul.cl`: `matmul_f32_q4_0_scaled_m1_f32` (~3528) и `matmul_f32_q4_0_scaled_m1_wg64x4_f32` (~3558) — референс математики и wave64-схемы.
3. `kernels/vulkan/mm_f32_m1n.comp` — существующий Vulkan f32 m1-кернел (схема редукции/раскладки — образец стиля).
4. `kernels/vulkan/mm_q8q4_scaled_f32.comp` — q4_load уже написан в Q1 (переиспользуй ту же распаковку).
5. `benchmarks/bench_vulkan_perf.cpp` — как устроены кейсы (BenchCase, vulkan_iter/opencl_iter, fixtures), кейс matmul_f32_m1 как образец.

## Изменения

### 1. Кернел `kernels/vulkan/mm_f32q4_m1_wg64_f32.comp`
- Вход: A f32 [1,K]; B packed Q4_0 [K,N] (линейная упаковка k*N+n как в Q1); scales_b f32 [N] (cols).
- Схема по образцу OpenCL wg64x4: local_size_x=64; каждый WG считает 4 выходных колонки; потоки страйдят по K, частичные суммы в shared [4][64], tree-редукция; потоки 0..3 пишут C[col+j] = dot * scales_b[col+j].
- Push: K, N. Guard'ы на хвостах N (col+j < N) и K.
- Требование: результат для каждой колонки математически = sum_k(a[k] * (q4(k,col)-8)) * scale_b[col] — сверь порядок с OpenCL референсом (там scale внутри или снаружи суммы? перенеси точно).

### 2. Диспетчер `run_vulkan_matmul_f32q4_m1(runtime, a_f32, b_q4, b_scales, c_f32, K, N)`
- По образцу Q1-диспетчеров: валидация nbytes (a: K*4, b: ceil(K*N/2), scales: N*4, c: N*4), push-overflow, dispatch_cached, группы = ceil(N/4).
- Декларация в hpp рядом с Q1-блоком.

### 3. matmul.cpp — ветка декода
- В квант-диспетчере: Vulkan && a.dtype()==F32 && b.dtype()==Q4_0 && b scales axis==1 && M==1 → run_vulkan_matmul_f32q4_m1. (Прочие f32×квант шейпы на Vulkan → MCL_CHECK not-ported, как в Q2.)
- После этого Linear decode-ветка (linear.cpp:33-42, НЕ трогать) заработает на Vulkan для Q4_0.

### 4. Витнес
- tests/test_vulkan_runtime_standalone.cpp: parity mm_f32q4_m1 vs host: K=47,N=65 и K=128,N=96 и K=1024,N=256 (страйд-петля по K с несколькими итерациями). Допуск относительный 1e-4 (f32×int4 — точная арифметика в float, порядок редукции отличается).
- tests/test_vulkan_backend.cpp: Tensor-уровень: Linear c enable_quantized_inference(Q4_0), forward [1,K] (decode-путь!) vs f32-референс до квантизации, допуск как в Q2-тесте; проверь через каким-нибудь способом, что пошёл именно m1-путь (например, MOTIFCL_LOG? если нет дешёвого способа — отметь в отчёте, что путь подтверждён только шейп-условием).

### 5. Бенч-кейсы (benchmarks/bench_vulkan_perf.cpp)
Добавить 3 кейса по образцу существующих (vulkan_iter через run_vulkan_*; opencl_iter через Tensor matmul с квант-тензорами на OpenCL, backend->finish()):
- `matmul_q8q8_scaled` 512x512x512 (prefill), target_ratio 0.33.
- `matmul_f32q4_m1` 1x1024x1024 и 1x2048x2048 (decode), target_ratio 0.33.
- В work_bytes посчитай реально читаемые байты (q4: K*N/2 + K*4 + N*4).

### 6. Сборка и прогоны (вывод в отчёт)
- regen SPIR-V, сборка, ctest -R "test_vulkan_runtime_standalone|test_vulkan_backend", полный ctest -R vulkan.
- Бенч НЕ запускай (GPU занят другими прогонами оркестратора; запустит оркестратор).

## Запреты
- Не менять Q1/Q2 файлы кроме добавления новых веток/тестов, не трогать linear.cpp, OpenCL-файлы.
- Git read-only.

## Формат отчёта
Файлы+суть; фактические хвосты прогонов; max-diff тестов; что не проверено.
