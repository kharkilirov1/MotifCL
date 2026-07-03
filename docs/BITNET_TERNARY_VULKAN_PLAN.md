# План: перенос тернарного (1.58-bit) W2A8-инференса BitNet в Vulkan-бэкенд MotifCL

> Статус документа: финальный, готовый к реализации. Все замечания ревью обработаны. Несогласованности источников истины (адреса `block_i2_s`, формат scale, type id, опкоды, layout P0/P1) устранены и приведены к единым формулам. Замечания, неустранимые без данных реальной модели, вынесены в раздел 10 как блокирующие предусловия фазы 0.

## 0. Фаза 0 — блокирующие предусловия (выполнить ДО среза 1)

Ревью корректно указало: разделы 3-6 зависят от фактической структуры целевого GGUF (тип квантования, расположение и гранулярность scale). Если реальная модель использует `TQ2_0` вместо `I2_S`, или per-channel scale вместо per-tensor — дизайн буферов (§4.3) и загрузчик (§6) меняются. Поэтому фаза 0 — обязательна и блокирует начало разработки.

**Z-0.1. Получить и инспектировать реальный артефакт.**
- Сгенерировать dummy-модель именно с типом `I2_S` (а не TL1/TL2/TQ2_0):
  ```
  python C:\Users\Kharki\BitNet\utils\generate-dummy-bitnet-model.py --model-size 125M --outtype i2_s --outfile dummy_i2s.gguf dummy_dir
  ```
  Если `generate-dummy-bitnet-model.py` не поддерживает `i2_s` как outtype — установить фактически поддерживаемые типы (см. OQ-1, OQ-7) и принять решение по §6.5 на их основе.
- Дополнительно — реальный BitNet-b1.58-2B-4T-gguf, если доступен локально.

**Z-0.2. Выпустить утилиту инспекции `tools/gguf_inspect.py`** (новый файл), которая дампит по каждому тензору: имя, `ggml_type` (число и имя), shape, размер блока, число блоков, on-disk nbytes, и эвристику расположения scale (наличие соседнего тензора `<name>_scale`; либо размер данных тензора, не кратный `nb_blocks*64`, что указывает на хвостовой scale). Вывод — таблица + JSON.

**Z-0.3. Зафиксировать факты в `docs/BITNET_GGUF_FACTS.md`** (новый файл): фактический type id весовых тензоров; расположение scale (хвост блочных данных vs отдельный тензор; точное смещение в байтах); гранулярность scale (per-tensor / per-channel / per-block); кратность K (для §4.8). Эти факты — вход для §3.5, §3.8, §6.1, §6.5, §10.

**Критерий выхода из фазы 0:** OQ-1, OQ-3, OQ-6, OQ-7 (§10) закрыты подтверждёнными данными; `BITNET_GGUF_FACTS.md` заполнен. Только после этого — срез 1.

> Если фактический тип окажется не `I2_S`, а, например, `TQ2_0` (тоже 2 бита/элемент, но иной layout и наличие f16-scale внутри блока, см. `ggml-common.h:243-247`), §3.3/§6.1 переписываются под него, но архитектура срезов 1-4 и Vulkan-дизайн §4 остаются в силе (меняются только битовые формулы распаковки и парсинг scale). Это явно отражено в OQ-7.

## 1. Резюме и цель

Цель — реализовать нативные GPU-ядра инференса тернарных (1.58-bit) весов BitNet в Vulkan-бэкенде MotifCL для AMD Radeon RX 580 (Polaris/GCN4), без зависимости от CUDA и без аппаратного int8 dot-product (dp4a). Технология-донор — Microsoft BitNet: формат `I2_S` (тернарные веса {−1, 0, +1}, 2 бита/элемент, одна глобальная f32-шкала), схема W2A8 (2-bit Weight, 8-bit Activation).

Подход — инкрементальные вертикальные срезы с TDD (reference + тест прежде ядра):
- **Срез 1** — формат `DType::I2_S` + GGUF-загрузка + CPU-reference dequant + **нативный** I2_S-Tensor (host-распаковка обязательна, см. §6.5).
- **Срез 2** — Vulkan GEMV `F32 × I2_S → F32` (M=1), ручная распаковка 2-бит, F32-аккумуляция. Доказательство пути распаковки.
- **Срез 3** — полный W2A8: `I8 × I2_S`, целочисленная аккумуляция, ручной MAD без dp4a.
- **Срез 4** — снятие ограничений (M>1, большие K/N), perf-gate, e2e-валидация на BitNet-2B.

Конечный критерий — Vulkan-путь даёт численно эквивалентный baseline-результат (cosine ≥ 0.99 на логитах) при замедлении ≤ 20% относительно baseline (определение baseline уточнено в §7.6) и проходит CTest + `perf_truth_gate.py`.

Срезы строго последовательны (§8): `1 → 2 → 3 → 4`. Каждый срез, будучи завершён, самостоятельно мержабелен (opt-in через `MOTIFCL_MATMUL_BACKEND=vulkan`, OpenCL/CPU-fallback сохранён), но функционально опирается на предыдущий.

## 2. Контекст: репозитории, железо, мотивация

**Репозитории.**
- Донор: `C:\Users\Kharki\BitNet`. CPU-ядра-образцы: `src/ggml-bitnet-mad.cpp` (`quantize_i2_s`, `ggml_vec_dot_i2_i8_s_*`). GPU-ядро-образец (CUDA, неприменимо напрямую): `gpu/bitnet_kernels/bitnet_kernels.h` (`decode_i2s_to_i8s`). Формат: `3rdparty/llama.cpp/ggml/include/ggml.h`, `3rdparty/llama.cpp/ggml/src/ggml-common.h`.
- Цель: `C:\Users\Kharki\Desktop\motifcl_production\motifcl_production`. Работаем только с `src/`, `include/motifcl/`, `kernels/`, `tools/`, `tests/`, `docs/`, `benchmarks/`. Игнорируем `build*`, `.agent_backup*`, `dist`, `build_wheel_smoke_venv*`.

**Железо: AMD Radeon RX 580 (Polaris/GCN4, 8 ГБ VRAM).** NVIDIA нет. Доступны Vulkan 1.3 и OpenCL (AMD APP). **Аппаратного int8 dot-product (dp4a / `VK_KHR_shader_integer_dot_product` с пакетным режимом) на Polaris НЕТ** — это центральное ограничение: целочисленный MAD придётся раскручивать вручную (`OpShiftRightLogical` + `OpBitwiseAnd` + `OpIMul` + `OpIAdd`).

**Почему Vulkan.** Стратегия пользователя — уход от OpenCL к собственным нативным GPU-ядрам. В MotifCL это линия Vulkan с runtime-генерируемым SPIR-V. Текущее состояние Vulkan-бэкенда: готовы только F32-ядра (matmul m1/general, softmax, rmsnorm, swiglu, add) через единый диспетчер `run_vulkan_storage_buffer_compute()`; квантованных Vulkan-ядер нет, integer-dot не используется — только float storage buffers. Это и есть точка, в которую врезается тернарная технология.

**Почему тернарность даёт выигрыш (memory-bound).** Инференс LLM на одном GPU при генерации токенов (M=1, GEMV) ограничен пропускной способностью VRAM, а не ALU. Тернарные веса занимают 2 бита/элемент вместо 16 (f16) — это в ~8 раз меньше трафика VRAM на чтение весов. Даже без dp4a, при ручной распаковке, экономия bandwidth доминирует, и Polaris-ядро остаётся выгодным относительно f16/f32. Дополнительный ALU-overhead распаковки скрывается за ожиданием памяти.

## 3. Формат данных и кодировка весов

### 3.1 Семантика I2_S
W2A8 в BitNet кодирует **тернарные веса {−1, 0, +1}** в 2 бита/элемент с **одной глобальной (per-tensor) f32-шкалой** — абсолютным максимумом исходных весов (`i2_scale = max|src|`, `ggml-bitnet-mad.cpp:63,107`). Декодирование: `W_actual[i] = scale × ternary(code[i])`.

> Замечание о смысле scale: в `quantize_i2_s` `i2_scale` = `max|src|` (НЕ обратный максимум). При прямом восстановлении `W ≈ scale × ternary`. Формула умножения на CPU-стороне инференса завершается простым `(float)sumi` (`ggml-bitnet-mad.cpp:295`) — масштабирование на `i2_scale` выполняется вне dot-product, на уровне графа. В MotifCL мы храним `i2_scale` как `quant_scale()` и применяем его после матмула (§4.6/§4.7). Точную семантику (нужно ли дополнительно делить на что-либо для конкретной модели) подтвердить в фазе 0 (OQ-1) — для dummy-модели достаточно `W = scale × ternary`.

### 3.2 Кодировка тернарных значений (источник истины — `ggml-bitnet-mad.cpp:65-72`, `108-116`)
Промежуточный код `q8 ∈ {0,1,2}` формируется так: `q8 = 1`, если `|src| < 1e-6` (ноль); иначе `q8 = (src·i2_scale > 0 ? 2 : 0)` (т.е. знак `src`, т.к. `i2_scale ≥ 0`).

| Ternary | code (q8) | 2-bit | примечание |
|---|---|---|---|
| −1 | 0 | `00` | negative |
| 0 | 1 | `01` | zero |
| +1 | 2 | `10` | positive |
| (unused) | 3 | `11` | не используется |

**Декодирование (единое для всех ядер и reference MotifCL):** `ternary = int(code) − 1` (0→−1, 1→0, 2→+1). Это алгебраически корректное отображение, и оно **ОБЯЗАНО** совпадать в §4 (шейдер), §6.4 (CPU-dequant) и в любом host-распаковщике.

> **Важное замечание о расхождении с GPU-донором (документировать в коде).** CUDA-ядро `decode_i2s_to_i8s` (`bitnet_kernels.h:42`) делает `__vsubss4(x, 0x02020202)`, т.е. отображает `code → code − 2 → {−2,−1,0}`, и компенсирует это смещением/постобработкой в формуле инференса. CPU-ядро `ggml_vec_dot_i2_i8_s` (`ggml-bitnet-mad.cpp:242-245`), напротив, использует **unsigned** `_mm256_maddubs_epi16` с кодами `{0,1,2}` напрямую (без вычитания), а вычитание единицы (переход `{0,1,2}→{−1,0,+1}`) эквивалентно поправке `Σ(code−1)·a = Σcode·a − Σa`, которая на уровне ggml учитывается структурой графа/нормировок. **MotifCL использует чистое `code − 1`** в каждом ядре и reference. Это снимает необходимость в парном смещении активаций и упрощает SPIR-V. Эквивалентность чистого `code−1` пути результату ggml-пути ОБЯЗАНА быть подтверждена численным тестом (§7.4a, OQ-5) до завершения среза 3.

### 3.3 Точная битовая раскладка GGML I2_S (источник истины — `ggml-bitnet-mad.cpp:80-91`, MAD-распаковка `225-234` и `ggml-common.h:271-273`)

Это критическая часть. **Простое «16 элементов на uint32, LSB-first» — НЕВЕРНО для нативного GGML-формата I2_S. Нативный формат — MSB-first со страйдом 32.** Реальная упаковка (источники истины, дословно):

- Блок = `QK_K = 256` элементов = `QK_K/4 = 64` байта. Структура: `typedef struct { uint8_t qs[QK_K/4]; } block_i2_s;` (`ggml-common.h:271-272`), и `static_assert(sizeof(block_i2_s) == QK_K/4)` (`ggml-common.h:273`) — т.е. **внутри блока scale НЕ хранится** (в отличие от `block_tq2_0` на строках 243-247, где есть `ggml_half`; не путать — это разные типы).
- Упаковка (`ggml-bitnet-mad.cpp:81-88`, путь `QK_I2_S = 128`): элемент `j ∈ [0,128)` пишется в байт `i*32 + (j%32)` со сдвигом `6 − 2·(j/32)`. Один байт содержит 4 элемента со **страйдом 32**: `{e_g, e_{g+32}, e_{g+64}, e_{g+96}}`, где `g = j%32`, разложенные **MSB-first** по битам `[7:6], [5:4], [3:2], [1:0]` для групп `g/…` = 0,1,2,3 соответственно.
- MAD-распаковка подтверждает MSB-first (`ggml-bitnet-mad.cpp:226-228`): `xq8_0 = (b >> 6) & 3`, `xq8_1 = (b >> 4) & 3`, `xq8_2 = (b >> 2) & 3`, `xq8_3 = (b >> 0) & 3` — т.е. старшие биты соответствуют меньшему индексу группы.
- GPU-донор подтверждает interleaving (`bitnet_kernels.h:29`): `i2s = {e0,e4,e8,e12,e1,e5,e9,e13,...}`.

**Каноническая формула извлечения кода элемента `j` из нативного GGML-блока (MSB-first, страйд-32):**
```glsl
// qs — байтовый указатель на начало блока (64 байта); j ∈ [0,128) внутри полу-блока QK_I2_S
uint group        = j / 32;                 // 0..3  -> определяет битовый сдвиг
uint byte_in_blk  = j % 32;                 // 0..31 -> номер байта внутри 32-байтовой полосы
uint b            = uint(qs[byte_in_blk]);  // (для второго полу-блока: qs[32 + byte_in_blk])
uint code         = (b >> (6u - 2u*group)) & 0x3u;   // MSB-first
int  ternary      = int(code) - 1;          // {-1,0,+1}
```
**Соответствие битов (для одного логического элемента после/до переупаковки):** старшая пара GGML `[7:6]` (group 0) ↔ младшая пара P0 `[1:0]` (см. §3.4) того же логического элемента после переупаковки.

### 3.4 Стратегия layout для MotifCL: P0 vs P1 (решение + алгоритм переупаковки)

Есть два варианта подачи весов в шейдер. **Решение: срез 2 использует P0** (переупаковка при host-загрузке, простой шейдер); срезы 3-4 могут перейти на P1 (см. OQ-2). Оба описаны формально, с алгоритмом переупаковки.

- **P0 (плоский, LSB-first, 16 элементов/uint32) — используется срезом 2.**
  Формула извлечения: `code = (packed_u32 >> (2*i)) & 0x3`, `i ∈ [0,16)`. Элемент `e_m` хранится в слове `m/16`, в паре бит `[2*(m%16)+1 : 2*(m%16)]`. Этот layout получается **переупаковкой из нативного GGML** на host (см. ниже).
- **P1 (нативный GGML, MSB-first, страйд-32) — кандидат для срезов 3-4.**
  Шейдер читает GGML напрямую по формуле §3.3, без переупаковки (нулевая стоимость загрузки, но сложнее индексация в шейдере).

**Алгоритм переупаковки GGML → P0 (host, часть GGUF-загрузчика, §6.5; покрывается отдельным unit-тестом ДО ядра).**

Псевдокод (C-уровень), на блок 256 элементов = 64 GGML-байта → 16 P0-uint32:
```c
// in:  qs_ggml[64]  — нативный GGML-блок (MSB-first, страйд-32), два полу-блока по 128 элементов
// out: qs_p0[16]    — 256 кодов, упакованных LSB-first по 16/uint32
// Шаг 1: распаковать 256 кодов в естественном порядке индекса элемента e[0..255]
uint8_t code[256];
for (int half = 0; half < 2; ++half) {          // два полу-блока QK_I2_S=128
    const uint8_t* qs = qs_ggml + half*32;       // каждый полу-блок занимает 32 байта
    for (int j = 0; j < 128; ++j) {
        int group       = j / 32;                // 0..3
        int byte_in_blk = j % 32;                // 0..31
        uint8_t b       = qs[byte_in_blk];
        uint8_t c       = (b >> (6 - 2*group)) & 0x3;   // MSB-first, §3.3
        code[half*128 + j] = c;                  // естественный индекс элемента
    }
}
// Шаг 2: упаковать 256 кодов LSB-first по 16 на uint32
for (int w = 0; w < 16; ++w) {
    uint32_t packed = 0;
    for (int i = 0; i < 16; ++i)
        packed |= (uint32_t)(code[w*16 + i] & 0x3) << (2*i);
    qs_p0[w] = packed;
}
```

**Пример на 4 элементах (демонстрация перехода).** Пусть для полу-блока 0, байта `byte_in_blk=0` нативное `qs[0] = 0b10_01_00_01 = 0x91`. По §3.3:
- `j=0`  (group 0): `(0x91 >> 6) & 3 = 0b10 = 2` → ternary +1; это `e[0]`.
- `j=32` (group 1): `(0x91 >> 4) & 3 = 0b01 = 1` → ternary 0; это `e[32]`.
- `j=64` (group 2): `(0x91 >> 2) & 3 = 0b00 = 0` → ternary −1; это `e[64]`.
- `j=96` (group 3): `(0x91 >> 0) & 3 = 0b01 = 1` → ternary 0; это `e[96]`.

После Шага 2 эти коды попадут в P0-слова `e[0]→qs_p0[0] биты [1:0]`, `e[32]→qs_p0[2] биты [1:0]`, `e[64]→qs_p0[4] биты [1:0]`, `e[96]→qs_p0[6] биты [1:0]`, и шейдер прочитает их формулой P0 `(packed>>0)&3`. Таким образом GGML `[7:6]` элемента `e[0]` = P0 `[1:0]` элемента `e[0]` — биты «развёрнуты», порядок элементов нормализован к естественному.

> **Согласованность §3.3 ↔ §3.4 ↔ §4 ↔ §6:** ровно один из путей подаётся в ядро среза 2 — **P0**, который производит host-переупаковщик §6.5. Шейдер среза 2 (§4.5/§4.6) использует **только формулу P0** (`(w>>(2*i))&3`). Reference-dequant §6.4 и переупаковщик §3.4 используют **формулу GGML §3.3**. Все три обязаны давать одинаковые тернарные значения для одних и тех же логических элементов — это проверяется white-box тестом §7.4b. Тест (§7) обязан явно фиксировать, какой layout подан в ядро.

### 3.5 Масштаб (scale)
Per-tensor, один скаляр f32 (для I2_S; подтвердить per-tensor в фазе 0, OQ-3). В нативном GGML I2_S хранится **f32 в конце данных тензора**: `float* scale_ptr = (float*)((char*)i2_weight + n/4); scale_ptr[0] = i2_scale;` (`ggml-bitnet-mad.cpp:90-91`; тот же паттерн `143-144`). То есть scale — это f32, записанный сразу за упакованными весами (смещение `n/4` байт от начала данных тензора), **не f16**. В MotifCL хранится отдельно — как `Tensor::quant_scale()` (скаляр f32). При GGUF-загрузке scale берётся либо из хвоста блочных данных тензора, либо из отдельного тензора `<name>_scale` — какая схема используется фактически, фиксирует фаза 0 (OQ-1).

### 3.6 Размер хранилища
- GGML on-disk (нативный I2_S): `nb_blocks · 64` байт упакованных весов **+ хвостовой f32 scale (4 байта) + выравнивание до 32 Б per-row** (`ggml-bitnet-mad.cpp:96`: `return nrow * row_size/4 + 32`). Это влияет на парсинг nbytes тензора в §6.
- MotifCL `dtype_storage_nbytes(I2_S, numel)`: упакованные веса = `(numel + 3) / 4` байт; масштаб хранится отдельно (не входит в эту величину).

### 3.7 DType::I2_S в MotifCL (контракт для §5)
- `include/motifcl/core/dtype.hpp` (enum `DType`, строки 8-22, последний элемент `Q4_0_COL` на строке 21 **без хвостовой запятой**): добавить запятую после `Q4_0_COL` и новый элемент `I2_S` — `// Ternary BitNet 1.58-bit, 2 bits/elem, per-tensor f32 scale`.
- `src/core/types.cpp`:
  - `dtype_size()` (строки 6-20, перед `default`): `case DType::I2_S: return 1;` (условный «байтовый» размер; физически 2 бита).
  - `dtype_storage_nbytes()` (строки 22-37, в начало, рядом с веткой Q4_0 на строке 23): `if (dtype == DType::I2_S) return (numel + 3) / 4;`
  - `dtype_name()` (строки 39-53, перед `default`): `case DType::I2_S: return "i2_s";`

### 3.8 Итоговая справочная таблица формата

| Параметр | Значение | Источник истины |
|---|---|---|
| Ternary | {−1,0,+1} ← `code−1`, code∈{0,1,2}; `11` unused | `ggml-bitnet-mad.cpp:65-72,76-78` |
| 2-bit encoding | `00`→−1, `01`→0, `10`→+1 | там же |
| GGML block | 256 элементов / 64 байта (`block_i2_s.qs[QK_K/4]`); scale НЕ внутри блока | `ggml-common.h:271-273` |
| GGML bit-order | **MSB-first, страйд-32**: байт = {e_g,e_{g+32},e_{g+64},e_{g+96}}, сдвиги 6/4/2/0 | `ggml-bitnet-mad.cpp:81-88,226-228` |
| GGML type id | **36** (`GGML_TYPE_I2_S`) | `ggml.h:393` |
| Scale | per-tensor **f32**, = `max|src|`; в GGML — хвост данных тензора (смещение `n/4`) | `ggml-bitnet-mad.cpp:63,90-91` |
| MotifCL storage | `(numel+3)/4` Б, scale отдельно (`quant_scale()`) | план §3.7 |
| P0-layout ядра (срез 2) | 16 элементов/uint32, **LSB-first**: `(w>>(2*i))&3`; получается host-переупаковкой §3.4 | план §3.4 |
| Сопутствующие type id | TQ1_0=34, TQ2_0=35, I8_S=37, TL1=38, TL2=39 | `ggml.h:391-396` |

## 4. Дизайн Vulkan SPIR-V ядра

### 4.1 Двухуровневый подход
- **Срез 2 (GEMV, доказательство пути):** `F32 activations × I2_S weights`, M=1, F32-аккумулятор. Минимум сложности для валидации конвейера распаковки. Layout весов — **P0**.
- **Срез 3 (W2A8 full):** `int8 activations × I2_S weights`, int32-аккумулятор, ручной MAD без dp4a. Production для реальных размеров.

Стиль генерации — как существующие f32-ядра в `vulkan_backend.cpp` (генератор `f32_matmul_spirv`, тело ~строки 805-972; хелперы `append_spirv_inst`, `append_spirv_string`). Новые ядра пишутся теми же хелперами. Перед написанием генератора выпускается ручной SPIR-V-эталон (см. §4.10).

### 4.2 Новые SPIR-V опкоды

**Источник опкодов — официальный `SPIRV-Headers`, заголовок `spirv.hpp` (`enum class Op`), версия, совместимая с Vulkan 1.3 (например, SPIRV-Headers ревизии, соответствующей Vulkan SDK 1.3.x; брать ровно ту, против которой собирается проект).** В генераторе использовать `constexpr` из `spirv.hpp`/собственной таблицы, **не magic numbers**. Десятичные/hex значения ниже приведены для удобства сверки и соответствуют каноническим значениям SPIR-V (стабильны начиная с 1.0; в 1.3 не менялись для перечисленных опкодов).

Уже используются f32-ядрами (для опоры): `OpLoad`(61/0x3D), `OpStore`(62/0x3E), `OpAccessChain`(65/0x41), `OpFMul`(133/0x85), `OpFAdd`(129/0x81), `OpIMul`(132/0x84), `OpIAdd`(128/0x80), `OpCompositeExtract`(81/0x51), `OpLabel`(248/0xF8), `OpReturn`(253/0xFD), `OpFunctionEnd`(56/0x38).

**Новые для распаковки и integer-MAD:**

| Опкод | dec | hex | назначение |
|---|---|---|---|
| `OpShiftRightLogical` | 194 | 0xC2 | сдвиг uint32 для извлечения 2-бит слайса |
| `OpBitwiseAnd` | 199 | 0xC7 | маска `0x3` |
| `OpISub` | 130 | 0x82 | `code − 1` целочисленно (предпочтительный путь) |
| `OpSelect` | 169 | 0xA9 | branch-free выбор (запасной путь восстановления) |
| `OpIEqual` | 171 | 0xAB | сравнение кода (для Select-цепочки) |
| `OpConvertSToF` | 111 | 0x6F | int32-аккумулятор → float (срез 3) |
| `OpConvertFToS` | 110 | 0x6E | квантование активаций в шейдере (если потребуется, срез 3) |

> Внимание к ранее перепутанным значениям (исправлено относительно черновика): `OpIEqual = 171` (0xAB), а не 170; `OpUEqual = 173` (0xAD). `OpBitwiseAnd = 199` (0xC7), `OpShiftRightLogical = 194` (0xC2). При реализации все коды сверять с подключённым `spirv.hpp`; расхождение заголовка и таблицы — повод доверять заголовку и обновить таблицу плана.

Все опкоды входят в capability **Shader** (Vulkan 1.0) — расширений не требуется. **dp4a-опкоды (`OpSDot`/`OpUDot` из `SPV_KHR_integer_dot_product`) сознательно НЕ используются** — на Polaris нет ускорения, и пакетный режим не гарантирован.

### 4.3 Descriptor set layout

**Срез 2 (F32 × I2_S, M=1):**
| binding | содержимое | тип |
|---|---|---|
| 0 | F32 activations a[K] | storage, f32 runtime array |
| 1 | packed I2_S weights (**P0-layout**, uint32) | storage, u32 runtime array |
| 2 | output c[N] | storage, f32 runtime array |

scale передаётся как **push-constant (f32)** — единый скаляр, не буфер (per-tensor; см. §3.5). Push-constant ≥ 128 байт гарантированы спецификацией Vulkan — одного f32 заведомо достаточно.

**Срез 3 (I8 × I2_S, W2A8):**
| binding | содержимое | тип |
|---|---|---|
| 0 | int8 activations a[M·K] (упакованы по 4 в i32-слова) | storage, u32/i32 |
| 1 | packed I2_S weights (P0 или P1) | storage, u32 |
| 2 | output c[M·N] | storage, f32 |

`scale_w` (per-tensor f32) — push-constant. `scale_a` — push-constant, если глобальный/per-tensor; **если per-row** (типично для динамического int8-квантования активаций per-token), он становится массивом длины M и переносится в **binding 3 (storage f32-массив)**, смещение в шейдере `scale_a_buf[m]` (см. §4.7, расширение). Решение по гранулярности `scale_a` — в фазе реализации среза 3 (зависит от схемы квантования активаций, которую выберем).

### 4.4 Маппинг dispatch
- Срез 2: `gl_GlobalInvocationID.x → n ∈ [0,N)`; dispatch `(N,1,1)`, local_size `(1,1,1)` (как f32-m1), затем local_size 64 с bounds-check (как оптимизация).
- Срез 3: `.x→n, .y→m`; dispatch `(N,M,1)` или тайлами.

### 4.5 Алгоритм распаковки одного uint32 (P0-layout, 16 значений) — для среза 2
```
packed (uint32) = [w15|w14|...|w1|w0], каждый wi — 2 бита, LSB-first
for i in 0..16:
    code    = (packed >> (2*i)) & 0x3        // OpShiftRightLogical + OpBitwiseAnd
    ternary = int(code) - 1                  // {-1,0,+1}; OpISub(code,1)
    acc    += float(ternary) * a[k_base+i]   // OpConvertSToF -> OpFMul -> OpFAdd (срез 2)
```
Восстановление в SPIR-V — предпочтительно **одной инструкцией** `OpISub(code, const_1)` с последующим `OpConvertSToF`. Запасной branch-free путь: `v = OpSelect(OpIEqual(code,1), 0, OpSelect(OpIEqual(code,2), +1, -1))`. Хвост `K % 16` обрабатывается отдельным batch с проверкой `k_base + i < K` (см. §4.8).

### 4.6 Скелет генератора среза 2 (`f32_i2_gemv_spirv(K, N)`)
Структура повторяет `f32_matmul_spirv`: header (magic `0x07230203`, version Vulkan-1.0-совместимая, bound, schema=0) → `OpCapability Shader` → `OpMemoryModel Logical GLSL450` → `OpEntryPoint GLCompute "main"` + `GlobalInvocationId` → `OpExecutionMode LocalSize` → декорации (`BuiltIn 28` для `gl_GlobalInvocationID`; `ArrayStride=4`; `Block`; `DescriptorSet=0`, `Binding=0..2`; push-constant block для scale) → типы (`void, fn, f32, u32, i32, v3u32`, runtime arrays, struct-Block, pointers StorageBuffer/PushConstant) → константы (0,1,2, mask=0x3, 0.0f, 1.0f, −1.0f) → переменные (3 binding'а + input `gl_GlobalInvocationID` + push-constant) → тело.

Тело: `OpLoad gid; OpCompositeExtract col,0` → цикл по `k_word ∈ [0, ceil(K/16))`: `OpAccessChain + OpLoad packed` → внутренний разворот по 16 (или `<K` хвост): `OpShiftRightLogical`, `OpBitwiseAnd 0x3`, `OpISub(code,1)`, `OpConvertSToF`, `OpAccessChain + OpLoad a[k]`, `OpFMul`, аккумуляция `OpFAdd` → после цикла `acc · scale` (push-constant, `OpFMul`) → `OpAccessChain + OpStore c[col]`.

Объём генератора ~350-450 строк C++. **Обязательная структура — явные хелперы** (снижают копипаст и облегчают отладку битовых формул): `emit_unpack_2bit(packed_id, lane_idx, out_code_id)`, `emit_ternary_from_code(code_id, out_float_id)`, `emit_f32_mul_accumulate(ternary_id, a_id, acc_id)`. Wrapper `run_vulkan_ternary_m1_matmul` ~50-80 строк (по образцу `run_vulkan_f32_m1_matmul`, рядом ~строка 2576; см. фактические границы при реализации).

### 4.7 Скелет среза 3 (W2A8)
Отличия от среза 2: аккумулятор int32 (`OpIAdd` целочисленно), активации int8 (распаковка из i32-слов: `OpShiftRightLogical` + `OpBitwiseAnd 0xFF` + знаковое расширение через `OpBitFieldSExtract` или сдвиговую пару), после цикла `OpConvertSToF(acc) · scale_a · scale_w`. +250-350 строк.

> **Расширение под per-row scale_a (note для совместимости).** Если активации квантуются per-row (per-token) в int8 с индивидуальным `scale_a[m]`, заменить push-constant `scale_a` на binding 3 (storage f32-массив), и в шейдере вычислять `c[m,n] = float(acc) · scale_a_buf[m] · scale_w`. Descriptor-сет среза 3 проектировать так, чтобы добавление binding 3 не ломало остальные привязки. Это будущее расширение, но интерфейс закладывается сразу.

### 4.8 Ограничения v1 и план снятия
- K кратна 16 для основного цикла P0; хвост `K % 16` (1-15 весов) — отдельным batch с проверкой границы `k_base+i<K` (zero-pad по сути: пропуск загрузки за границей). Подтвердить кратность K для целевой модели (OQ-6): если все K кратны 16/256, хвост не активируется; иначе хвостовая ветка обязательна с первого среза.
- v1: N, M ≤ 4096 на инвокацию; снятие — 2D dispatch + тайлинг 8×8/16×16.
- Срез 2: M=1; обобщение на M>1 — 2D dispatch (срез 4).

### 4.9 Риски ядра
- Ручная сборка байткода → выделить хелперы `emit_unpack_2bit`/`emit_ternary_from_code`/`emit_mad_loop`; валидировать каждый сгенерированный модуль через `spirv-val`; выводить дизассемблер `.spvasm` для визуального review (см. §4.10).
- Нет dp4a → loop unrolling, в перспективе LUT-распаковка в shared memory; ожидание 30-50% пиковой ALU (приемлемо, путь memory-bound).
- Несоответствие битовой раскладки → white-box unit-тест: распаковать известный байт и сверить с ручным расчётом и CPU-reference (§7.4b). **Особое внимание: P0 (LSB-first) генерируется переупаковщиком из GGML (MSB-first, страйд-32); шейдер видит только P0.**
- Bounds: проверка размеров буферов уже в `run_vulkan_storage_buffer_compute` (проверяет `buffer.nbytes` — сверить точную строку при реализации, ~1834).
- Точность float-масштаба → tolerance-тест против CPU baseline (§7.9).

### 4.10 Ручной SPIR-V-эталон (предусловие генератора)
Перед реализацией генератора §4.6 выпустить **ручной эталонный SPIR-V** (короткий, ~20-40 инструкций после преамбулы) для smoke-кейса `[1,16]×[16,8]` и сохранить его дизассемблер `tests/fixtures/ternary_m1_16x8.spvasm`. Генератор обязан воспроизвести функционально эквивалентный модуль (проверка: `spirv-val` чист И результат совпадает с reference на этом кейсе). В тестах включить дамп сгенерированного модуля в `.spvasm` для diff-review против эталона. Это ловит логические ошибки (инвертированный сдвиг для MSB/LSB, перепутанный operand-порядок), которые `spirv-val` не детектирует.

## 5. Интеграция в MotifCL (точки врезки)

Чек-лист правок по файлам (номера строк — по текущему состоянию; перед правкой перечитывать файл — он мог измениться).

### 5.1 DType и типизация
- `include/motifcl/core/dtype.hpp` (enum `DType`, строки 8-22): после `Q4_0_COL` (строка 21) поставить запятую и добавить `I2_S, // Ternary BitNet 1.58-bit`.
- `src/core/types.cpp` — три ветки (§3.7):
  - `dtype_size()` (6-20): `case DType::I2_S: return 1;`
  - `dtype_storage_nbytes()` (22-37): `if (dtype == DType::I2_S) return (numel + 3) / 4;` (рядом со строкой 23).
  - `dtype_name()` (39-53): `case DType::I2_S: return "i2_s";`

### 5.2 Дескриптор микроядра
- `src/runtime/microkernel.cpp`, `microkernel_descriptors()` (строки 144-237), блок `if (backend == Vulkan && domain == Matmul)` (~154): добавить дескриптор `"vulkan.matmul.ternary_m1"` (после `vulkan.matmul.f32_m1`, ~157) с описанием: opt-in, rank-2, same-backend, no autograd, bounded K/N, OpenCL/CPU fallback. Хелпер `descriptor()` (59-70) — как есть.

### 5.3 Предикат и раннеры в matmul.cpp
- `src/ops/matmul.cpp`:
  - Новый предикат `vulkan_ternary_matmul_f32_i2_s_m1_supported(a,b)` (вставить рядом с прочими `vulkan_*_supported`, перед `vulkan_matmul_f32_supported`): `a.dtype()==F32 && b.dtype()==I2_S && оба ndim()==2 && a.backend_ptr()==b.backend_ptr() && a.shape()[0]==1 && a.shape()[1]==b.shape()[0] && K>0 && N>0 && K<=spec_limit && N<=spec_limit && !requires_grad`. На срезе 2 `spec_limit` мал (напр. 64) — это **намеренное ограничение smoke-валидации**, не production-предел; на срезе 4 поднять до реальных размеров (4096+) после 2D dispatch/тайлинга.
  - Новый раннер `run_vulkan_ternary_m1_tensor_matmul(a,b)`: MCL_CHECK'и формы/dtype, выгрузить `a.to_vector<float>()`, `b` → host-буфер packed-весов (P0) + `b.quant_scale()`, вызвать `run_vulkan_ternary_m1_matmul(a_host, b_packed, K, N, scale)`.
  - Диспетчер `matmul()` (ближе к концу файла, после ветки F32×Q4_0_COL m1; в текущем файле ~1102-1144 — сверить): вставить ветку `F32×I2_S, ndim==2, M==1`: проверить autograd; при `selected_backend==Vulkan && predicate` вызвать раннер, при успехе — собрать результат-тензор (по образцу `matmul_vulkan_f32_m1_from_result`); иначе — учесть `strict_vulkan_matmul_required()` (если строгий режим — ошибка) и перейти на **fallback** (§10 OQ-4: на этапе среза 2 — CPU-dequant→F32-matmul; см. §6.5/§5.9).

### 5.4 Vulkan backend
- `include/motifcl/runtime/vulkan_backend.hpp` (87 строк; после `VulkanF32MatmulSmokeResult`, рядом со строкой 51): struct `VulkanTernaryMatmulResult { bool success; std::vector<float> output; std::string device_name; std::string error; };`. После `run_vulkan_f32_m1_matmul` (рядом ~83): объявление `VulkanTernaryMatmulResult run_vulkan_ternary_m1_matmul(const std::vector<float>& a, const std::vector<uint8_t>& b_packed_p0, size_t k, size_t n, float scale);`.
- `src/runtime/vulkan_backend.cpp` (2658 строк): генератор `f32_i2_gemv_spirv(k,n)` (§4.6) рядом с `run_vulkan_f32_m1_matmul` (~2576); реализация `run_vulkan_ternary_m1_matmul` (~2618): проверки `k>0,n>0`, `k<=limit`, `n<=limit`, `a.size()==k`, `b_packed_p0.size()==(k*n+3)/4`; собрать буферы `{a, b_packed_p0, c}`, scale → push-constant; вызвать `run_vulkan_storage_buffer_compute(shader, ..., buffers, output_binding=2)`; скопировать результат в `VulkanTernaryMatmulResult::output`.

### 5.5 Linear
- `src/nn/linear.cpp` (101 строка), `Linear::forward` (строки 29-50), условие выбора decode-пути для квантованных типов (~36): добавить `|| decode_dtype == DType::I2_S` в список квантованных типов (наряду с Q4_0/Q4_0_COL и пр.).

### 5.6 CMake
- `CMakeLists.txt`: все затрагиваемые исходники (`types.cpp`, `microkernel.cpp`, `vulkan_backend.cpp`, `matmul.cpp`, `linear.cpp`, `gguf.cpp`, `hf_compat.cpp`) уже в `MOTIFCL_SOURCES`. **Изменений в основном таргете не требуется.** Новые сущности — только новый тест (§7.7) и, опционально, утилита `tools/gguf_inspect.py` (Python, вне CMake). Перед правкой — сверить, что в текущем `CMakeLists.txt` нет per-file перечисления, требующего ручного добавления (если перечисление пофайловое — добавить новые .cpp нет нужды, мы не создаём новых .cpp в src/, кроме изменений существующих).

### 5.7 Соответствие политике `docs/MICROKERNEL_BACKENDS.md`
Keep OpenCL/CPU fallback (§5.9); backend descriptor (§5.2); opt-in через `MOTIFCL_MATMUL_BACKEND=vulkan` + per-callsite validator (§5.3); per-callsite validator в `matmul()` (§5.3); SPIR-V/compute validation через `run_vulkan_storage_buffer_compute` (§4) + `spirv-val` (§4.10). Correctness coverage и performance baseline/gating — §7.

### 5.8 Окружение
- `MOTIFCL_MATMUL_BACKEND=vulkan` — включить Vulkan-путь.
- `MOTIFCL_REQUIRE_VULKAN_MATMUL=1` → `strict_vulkan_matmul_required()` (проверка уже есть для F32×F32; распространяется на новый путь). В строгом режиме отсутствие Vulkan-поддержки I2_S — контролируемая ошибка, не молчаливый fallback.

### 5.9 Замечания и fallback-политика (исправлено)
- `f32_i2_gemv_spirv` в срезе 2 реализуется полностью (§4.6); пустой/невалидный результат = сигнал на fallback.
- **OpenCL-путь для I2_S отсутствует.** На время срезов 1-3 fallback — **CPU-dequant I2_S→F32 (host, через тот же reference §6.4) + существующий F32-matmul** (на выбранном backend). Это корректно (не crash), но медленно — допустимо как fallback. В non-strict режиме при недоступности Vulkan этот путь обеспечивает правильный результат. Решение по постоянному production-fallback — OQ-4 (варианты: полный OpenCL-kernel, либо постоянный dequant-fallback с предупреждением, либо строгий отказ). **НЕ использовать Q4_0-перенаправление как fallback по умолчанию** — оно меняет численную семантику; черновиковый «временный Q4_0-fallback» отвергнут в пользу dequant→F32.
- Если потребуется per-row/per-block scale вместо скаляра — расширить Tensor API (`quant_scale_axis`/`quant_block_size`) и валидацию (§10 OQ-3) + binding/push-constant в ядре (§4.3/§4.7).

## 6. GGUF-загрузка BitNet i2_s

### 6.1 Форматные константы (источники истины)
- Type id: `I2_S = 36` (`ggml.h:393`). **Подтвердить, что весовые тензоры целевой модели имеют именно этот тип** (фаза 0, OQ-7).
- Block size: `QK_K = 256` (`ggml-common.h:72`).
- Block nbytes: `64` (`block_i2_s { uint8_t qs[QK_K/4]; }`, `ggml-common.h:271-273`; scale НЕ внутри блока).
- Bit-order внутри блока — §3.3 (MSB-first, страйд-32).
- Scale — хвостовой **f32** данных тензора (`ggml-bitnet-mad.cpp:90-91`) либо тензор `<name>_scale` (фаза 0, OQ-1).

Сопутствующие BitNet-типы (для полноты enum): `TQ1_0=34`, `TQ2_0=35`, `I8_S=37`, `TL1=38`, `TL2=39` (`ggml.h:391-396`).

### 6.2 Правки `include/motifcl/gguf.hpp`
`enum class TensorType` (строки ~55-87, элементы вида `Name = N`, последний значимый `BF16 = 30` на строке 85, затем `Unknown = 0xffffffffu` на строке 86): перед `Unknown` добавить `TQ1_0 = 34, TQ2_0 = 35, I2_S = 36, I8_S = 37, TL1 = 38, TL2 = 39,`.

### 6.3 Правки `src/gguf.cpp` (847 строк)
| Функция | Ориентир строк | Правка |
|---|---|---|
| `tensor_type_from_u32()` | 269-302 | case'ы 34..39 → соответствующие enum |
| `tensor_type_name()` | 304-338 | имена `"tq1_0"`…`"tl2"` |
| `tensor_type_block_size()` | 344-383 | `I2_S/TQ1_0/TQ2_0/TL1/TL2 → 256`; `I8_S → 1` |
| `tensor_type_block_nbytes()` | 385-418 | `I2_S → 64`; `I8_S → 1`; `TQ*/TL* → 0` (variable/deferred — не парсим как блочные на MVP) |
| `tensor_type_can_dequantize_to_f32()` | 425-439 | `I2_S → true` |
| `dequantize_tensor_data_to_f32()` | блок после ~629 | реализация CPU-reference I2_S (§6.4) |
| `read_tensor_quantized()` | 779-845 | поддержка I2_S (нативный путь, §6.5) |

### 6.4 CPU-reference dequant I2_S (источник истины для §7) — явный псевдокод
Эталон, против которого валидируется и переупаковщик P0, и Vulkan-ядро. **Битовая формула здесь ОБЯЗАНА совпадать с §3.3 (MSB-first, страйд-32).**

```c
// in:  data — указатель на упакованные данные тензора (nb_blocks*64 байт), scale (f32, из §3.5/OQ-1)
// out: out[numel] (float)
// numel кратно 256 (QK_K); nb_blocks = numel/256
void cpu_dequantize_i2_s_ggml(const uint8_t* data, float scale, float* out, size_t numel) {
    size_t nb_blocks = numel / 256;
    for (size_t blk = 0; blk < nb_blocks; ++blk) {
        const uint8_t* qs = data + blk * 64;            // 64 байта на блок
        for (int half = 0; half < 2; ++half) {          // два полу-блока QK_I2_S=128
            const uint8_t* p = qs + half * 32;          // 32 байта на полу-блок
            for (int j = 0; j < 128; ++j) {
                int   group       = j / 32;             // 0..3
                int   byte_in_blk = j % 32;             // 0..31
                uint8_t b         = p[byte_in_blk];
                uint8_t code      = (b >> (6 - 2*group)) & 0x3;   // MSB-first (§3.3)
                int   ternary     = (int)code - 1;                // {-1,0,+1}
                out[blk*256 + half*128 + j] = (float)ternary * scale;
            }
        }
    }
}
```
**Unit-тест на бумаге (§7.4b):** для известного блока (например, первый байт `0x91`, см. пример §3.4) вручную вычислить 4 значения и сверить с функцией.

> Альтернатива порядку индекса: если фаза 0 покажет, что целевой ggml пишет элементы в ином естественном порядке (interleaving по `bitnet_kernels.h:29`), скорректировать индекс `out[...]` соответственно. На dummy-модели, сгенерированной тем же `quantize_i2_s`, порядок выше корректен.

### 6.5 `read_tensor_quantized()` — стратегия (ИСПРАВЛЕНО: нативный путь — основной)
Ревью верно указало: если на срезах 1-2 идти через Q4_0-переквантизацию, то распаковка I2_S в шейдере (§4) никогда не тестируется на реальном тернаре. Поэтому:

- **ОСНОВНОЙ путь (обязателен с среза 1):** создать **нативный `DType::I2_S`-тензор**. Загрузить упакованные веса, выполнить **host-переупаковку GGML → P0** (§3.4) либо сохранить нативный GGML (P1) — выбор по OQ-2; масштаб → `quant_scale()`. На срезе 2 шейдер потребляет именно P0-веса этого тензора. Это единственный способ протестировать §4 на реальных данных.
- **MVP-/fallback-распаковка (для среза 1, пока нет Vulkan-ядра):** для корректного e2e уже на срезе 1 — `dequantize_tensor_data_to_f32()` (§6.4) даёт F32-тензор (медленно, но верно). Это fallback-путь (§5.9), не подмена нативного I2_S-тензора.
- **Q4_0-переквантизация — НЕ основной путь.** Допустима лишь как опциональная совместимость со старой инфраструктурой, но численно меняет семантику, поэтому из дефолта исключена (см. §5.9). Если всё же используется — требует хелпера `f32_to_f16()` и помечается как отдельный экспериментальный режим.

### 6.6 `src/nn/hf_compat.cpp` (3063 строки) — статус I2_S (исправлено, без противоречия)
- `gguf_type_is_native_quant()` (строки ~469-475): **после среза 1** добавить `I2_S` (нативный I2_S-тензор готов). **До завершения среза 1** I2_S туда НЕ входит и обрабатывается через `dequantize_tensor_data_to_f32` → F32-тензор (медленно, но корректно). Комментарий в коде: `// TODO(slice-1): I2_S native tensor; until then dequant->F32 fallback`.
- `apply_gguf_quantized_linear()` — изменений не требует: после среза 1 I2_S проходит как нативный quant через `read_tensor_quantized()`; до того — как F32 (dense) через базовый путь. Поведение детерминировано стадией, противоречия нет.

### 6.7 Тестовые модели
```
# dummy-модель BitNet с типом I2_S (приоритет; для фазы 0 и среза 1)
python C:\Users\Kharki\BitNet\utils\generate-dummy-bitnet-model.py --model-size 125M --outtype i2_s --outfile dummy_i2s.gguf dummy_dir
# если i2_s не поддерживается генератором — зафиксировать поддерживаемые типы (OQ-7) и адаптировать §3/§6
# конверсия HF (для реального e2e)
python C:\Users\Kharki\BitNet\utils\convert-hf-to-gguf-bitnet.py --outtype i2_s model_dir -o out.gguf
# либо готовый HF BitNet-b1.58-2B-4T-gguf
```
Инспекция артефакта — `tools/gguf_inspect.py` (§0, Z-0.2).

## 7. Тестирование и perf-gate

Политика — `docs/VALIDATION_AND_STABILITY.md` + `docs/MICROKERNEL_BACKENDS.md`. Принцип: **reference + white-box bit-test прежде ядра**, затем black-box numeric.

### 7.1 Reference прежде ядра (TDD)
До генерации SPIR-V реализуется и тестируется CPU-reference dequant I2_S (§6.4) и host-переупаковщик GGML→P0 (§3.4). Ядро валидируется против reference.

### 7.2 Unit-тест `tests/test_vulkan_ternary_matmul.cpp` (black-box numeric)
Сравнение GPU-результата с CPU-reference (распаковка 1.58-bit → F32 + обычный F32-matmul). Формы: smoke `[1,16]×[16,8]`; validation `[1,64]×[64,32]`, `[1,128]×[128,64]`; реальные кратные 16 `[1,4096]×[4096,2048]` (после снятия `spec_limit`, срез 4); (если M>1, срез 4) `[4,256]×[256,128]`. Инварианты: packed-размер = `(numel+3)/4`; scale = f32-скаляр; все выходы finite; `|GPU−CPU|/max(|CPU|,1e-3) ≤ tol` (§7.9). **Тест явно фиксирует поданный layout (P0 для среза 2).**

### 7.3 Smoke-тест
Детерминированный кейс с известным результатом (по образцу `run_vulkan_f32_matmul_smoke()` в `test_vulkan_backend.cpp:89-97`), встроить как `run_vulkan_ternary_matmul_smoke()`. Использует фикстуру `[1,16]×[16,8]` из §4.10.

### 7.4 Malformed-input regression
По образцу `test_vulkan_backend.cpp:53-76`: короткий packed-буфер, неверный scale, рассогласование K → все → controlled `motifcl::Error`, не segfault, непустое сообщение.

### 7.4a Численная валидация кодировки `code−1` vs ggml (`test_ternary_code_mapping`) — закрывает OQ-5
Цель — доказать, что чистое `code−1` численно эквивалентно ggml-пути (`ggml_vec_dot_i2_i8_s`), несмотря на то что ggml использует unsigned `maddubs` с кодами `{0,1,2}`.
1. Синтезировать вектор тернарных весов `w ∈ {−1,0,+1}^K` (K=16/128) и активаций.
2. Закодировать `w` в I2_S (коды `{0,1,2}`, scale, например 0.5) тем же отображением, что `quantize_i2_s`.
3. Путь A (MotifCL): распаковать через `code−1`, посчитать dot-product скалярным циклом.
4. Путь B (ggml-эмуляция): воспроизвести `Σ q8·a` через unsigned-семантику и применить поправку `−Σa` (эквивалент `code−1`); при наличии — вызвать реальный `ggml_vec_dot_i2_i8_s` на тех же данных.
5. Сверить A == B в пределах численной точности (отн. ≤ 0.1%). Зафиксировать в коде комментарий: «MotifCL uses code−1, NOT CUDA's code−2 (bitnet_kernels.h:42)».

### 7.4b White-box bit-test распаковки (`test_bitwise_unpack_i2_s`) — закрывает риск «общей ошибки reference и шейдера»
Чёрный ящик §7.2 не поймает ошибку, если И reference, И шейдер используют одну и ту же неверную формулу (например, оба LSB-first вместо GGML MSB-first). Поэтому — явная white-box проверка:
1. Взять известный GGML-блок (минимум первый байт `0x91` из примера §3.4 + ещё 4-9 элементов).
2. Вызвать `cpu_dequantize_i2_s_ggml` (§6.4) и сверить **побитово** с ручным расчётом на бумаге (значения {+1,0,−1,0,…}).
3. Вызвать host-переупаковщик GGML→P0 (§3.4) и сверить, что P0-распаковка `(w>>(2*i))&3` даёт те же тернарные значения для тех же логических элементов.
4. Запустить Vulkan-ядро на микро-входе, где веса = этот блок, активации = единичный орт `e_t`; выход `c[t]` обязан равняться `scale·ternary(e_t)`. Это изолирует битовую формулу шейдера от арифметики.
Тест гарантирует, что ошибка в bit-formula будет поймана даже при «согласованной» ошибке reference/шейдера, т.к. эталон — ручной расчёт, а не другая реализация.

### 7.5 Kernel validation contract
В `docs/kernel_validation_contracts.json` — запись `vulkan_quantized_matmul`, pattern `^matmul.*_vulkan_.*_(q4_0|q8_0|i2_s).*$`, validated_by `src/ops/matmul.cpp`, `src/runtime/vulkan_backend.cpp`; контракт: rank-2, A=F32, B∈{Q4_0,Q8_0,I2_S}, `A.shape[1]==B.shape[0]`, packed-длина = `(numel+3)/4` для I2_S, scale f32 (rank-0/скаляр для per-tensor), output M×N, K≤spec-limit, M=1 для opt-in m1-пути.

### 7.6 Perf-gate (`tools/perf_truth_gate.py`) — baseline уточнён
**Проблема baseline (ревью):** OpenCL-ядра для I2_S в MotifCL нет, поэтому `baselines/polaris_opencl_i2s.json` нечем сгенерировать напрямую. Решение (выбрано):
- **Первичный baseline — CPU-dequant→F32-matmul на том же backend** (host-распаковка §6.4 + существующий F32-путь). Это честный нижний ориентир «во сколько раз Vulkan-тернарное ядро лучше/хуже наивного пути». Файл `baselines/polaris_dequant_f32_i2s.json`.
- **Опциональный baseline — OpenCL F32** (распакованные веса как F32, существующий OpenCL F32-matmul), если нужен прямой OpenCL-ориентир: `baselines/polaris_opencl_f32_dequant_i2s.json`.
- Полноценный OpenCL **тернарный** baseline появляется только если будет реализовано OpenCL-ядро (OQ-4, вариант a) — тогда `baselines/polaris_opencl_i2s.json`. До тех пор perf-gate сравнивает Vulkan-кандидата с dequant→F32-baseline.

Формы BitNet-2B: `(1,4096)×(4096,2048)`, `(1,11008)×(11008,4096)`, `(8,256)×(256,4096)`, `(1,2048)×(2048,2048)`.
```
MOTIFCL_MATMUL_BACKEND=vulkan bench_matmul --k 4096 --n 2048 --json candidate_vulkan_i2s.json
python tools/perf_truth_gate.py baselines/polaris_dequant_f32_i2s.json candidate_vulkan_i2s.json \
    --tolerance 0.20 --require-key matmul_f32_i2s_m1_ms
```
Допуск **0.20** — пояснение: для первого Vulkan-ядра без dp4a целевой ориентир — не медленнее baseline более чем на 20%; ожидаемо Vulkan-тернарное ядро **быстрее** dequant→F32-baseline (за счёт bandwidth), поэтому гейт по сути защищает от регрессий. Метаданные: `metadata.kernel="matmul_vulkan_i2s_m1"`, `device_name="AMD Radeon RX 580"`, `vulkan_version="1.3"`, ISO-8601 timestamp.

### 7.7 CTest + `tools/ci_gate.py`
`tests/CMakeLists.txt`: `add_executable(test_vulkan_ternary_matmul tests/test_vulkan_ternary_matmul.cpp)`, link `motifcl`, применить `motifcl_apply_compiler_options`, `add_test`, `set_tests_properties(... PROPERTIES SKIP_RETURN_CODE 77)` (skip при отсутствии Vulkan-устройства). Тестовая модель: переменная `BITNET_DUMMY_GGUF_PATH` (по умолчанию пусто → загрузочные under-test секции скипаются), прокинуть как compile-define в тест; вспомогательный скрипт `tests/generate_test_models.sh`, вызывающий `generate-dummy-bitnet-model.py` (§6.7). `ci_gate.py`: после build добавить вызов `perf_truth_gate.py` для тернарного кандидата (срез 4).

### 7.8 End-to-end на BitNet-2B
OpenCL/CPU-baseline и Vulkan прогон `motifcl_generate_transformer --dump-logits` на одном prompt → `compare_logits.py --min-cosine 0.99`; perplexity (WikiText) совпадает в пределах ≤ 0.1% отн.

### 7.9 Допуски (обоснование добавлено) — закрывает замечание «допуски из воздуха»
Допуски ниже — стартовые; **уточняются по результату анализа error-propagation** (см. ниже). Обоснование: тернарная квантизация с per-tensor f32-scale даёт детерминированную ошибку восстановления весов; основной источник численного расхождения GPU↔CPU — порядок суммирования (float-ассоциативность) при больших K, поэтому абсолютный допуск растёт с K, а относительный держится 2-3%.

| Проверка | Допуск (старт) |
|---|---|
| Unit vs CPU ref (малые K ≤ 64) | отн. 2%, абс. 0.01 |
| Unit vs CPU ref (средние K ≤ 512) | отн. 3%, абс. 0.05 |
| Unit vs CPU ref (большие K ≥ 4096) | отн. 2%, абс. 0.1 |
| `code−1` vs ggml (§7.4a) | отн. ≤ 0.1% (численная точность) |
| White-box bit (§7.4b) | точное равенство кодов; выход = `scale·ternary` побитово в пределах f32 |
| Perf Vulkan vs baseline (§7.6) | ≤ 20% slowdown |
| Malformed | 100% controlled rejection |
| E2E logits | cosine ≥ 0.99 |
| E2E perplexity | отн. ≤ 0.1% |

**Анализ error-propagation (выполнить в срезе 1, до фиксации финальных допусков):** (1) взять F32-веса, квантовать в I2_S, восстановить `scale·ternary`; (2) измерить относительную ошибку матмула по слоям (среднее, дисперсия) на репрезентативных активациях; (3) при необходимости скорректировать таблицу. Ожидание: если базовая ошибка матмула ~2%, то e2e-cosine на логитах ~0.995 (выше порога 0.99) — порог реалистичен.

### 7.10 Артефакты
`baselines/polaris_dequant_f32_i2s.json`, опц. `baselines/polaris_opencl_f32_dequant_i2s.json`, `baselines/polaris_vulkan_i2s_v1.0.json`; CTest-executable; `tests/fixtures/ternary_m1_16x8.spvasm` (§4.10); `tools/gguf_inspect.py`, `docs/BITNET_GGUF_FACTS.md` (§0); опц. `docs/PERF_TERNARY_VULKAN_NOTES.md`.

## 8. Дорожная карта по срезам (1→4)

**Зависимости строго последовательны: `1 → 2 → 3 → 4`** (исправлено относительно черновика: срезы НЕ независимы). Каждый срез — вертикальный (reference + white-box bit-test + numeric → реализация → верификация) и, будучи завершён, мержабелен (opt-in, fallback сохранён), но функционально опирается на предыдущий. Fallback на CPU-dequant→F32 (§5.9) срабатывает, если путь не поддерживается на текущем backend.

### Срез 0 — Предусловия (фаза 0, §0)
**Задачи:** Z-0.1…Z-0.3 (артефакт I2_S-gguf, `gguf_inspect.py`, `BITNET_GGUF_FACTS.md`); закрыть OQ-1, OQ-3, OQ-6, OQ-7.
**Трудозатраты:** ~1-2 дня.
**Критерий:** факты формата подтверждены; тип, scale-location, scale-гранулярность, кратность K зафиксированы.

### Срез 1 — Формат и загрузка (фундамент, нативный I2_S обязателен)
**Задачи:** §3.7 (`DType::I2_S`); §6.2-6.4 (`TensorType::I2_S` + метаданные + CPU-reference dequant §6.4); host-переупаковщик GGML→P0 (§3.4); §6.5 **нативный I2_S-Tensor** (основной путь) + dequant→F32 fallback; unit-тесты: white-box bit (§7.4b), `code−1` vs ggml (§7.4a), dequant против ручного эталона; анализ error-propagation (§7.9); загрузка dummy_i2s.gguf.
**Зависимости:** Срез 0.
**Трудозатраты:** ~4-5 дней.
**Критерий готовности:** dummy_i2s.gguf грузится как нативный I2_S-тензор; CPU-dequant и переупаковщик побитово совпадают с ручным эталоном (MSB-first/страйд-32); `code−1` численно эквивалентно ggml-пути; `dtype_storage_nbytes(I2_S,n)==(n+3)/4`; все тесты зелёные.

### Срез 2 — Vulkan GEMV F32 × I2_S (M=1), доказательство пути
**Задачи:** §4.10 ручной SPIR-V-эталон; §4.6 генератор `f32_i2_gemv_spirv` (P0) с хелперами; §5.4 wrapper + push-constant scale; §5.3 предикат + раннер + ветка в `matmul()` + dequant-fallback; §5.2 дескриптор; §5.5 Linear; §7.2-7.4 + §7.4b на ядре; `spirv-val` + `.spvasm` diff каждого модуля.
**Зависимости:** Срез 1 (нативный I2_S-тензор + reference + переупаковщик).
**Трудозатраты:** ~5-7 дней (основной риск — ручной SPIR-V).
**Критерий готовности:** `[1,K]×[K,N]→[1,N]` на Vulkan совпадает с CPU-reference в допусках §7.9; white-box bit-test ядра зелёный; malformed → controlled error; `spirv-val` чист; `.spvasm` совпадает с эталоном по смыслу; CTest зелёный (или skip 77).

### Срез 3 — W2A8 (I8 × I2_S), целочисленный MAD
**Задачи:** §4.7 генератор W2A8 (int32-аккумулятор, ConvertSToF, scale_w·scale_a); квантование активаций per-row/глобально в int8 (host или шейдер; решение по scale_a-гранулярности → §4.3); расширение предиката/раннера/дескриптора на W2A8; тесты корректности на формах кратных 16; повтор §7.4a на W2A8-пути.
**Зависимости:** Срез 2 (распаковка отлажена и провалидирована).
**Трудозатраты:** ~5-6 дней.
**Критерий готовности:** I8×I2_S совпадает с CPU-reference (full W2A8) в допусках; покрыты формы BitNet-2B GEMV; численная эквивалентность `code−1` подтверждена и на W2A8.

### Срез 4 — Снятие ограничений + perf-gate + e2e
**Задачи:** §4.8 M>1 (2D dispatch), большие K/N (тайлинг), хвост K%16; снятие `spec_limit` в предикате (§5.3); §7.6 baseline (dequant→F32) + candidate + `perf_truth_gate.py`; §7.7 интеграция в `ci_gate.py`; §7.8 e2e на BitNet-2B (logits cosine, perplexity); §7.5 запись контракта; опц. OQ-2 переход на P1; опц. OQ-4 OpenCL-ядро.
**Зависимости:** Срез 3.
**Трудозатраты:** ~6-8 дней.
**Критерий готовности:** general-путь корректен для M>1 и больших N; Vulkan ≤20% медленнее baseline (ожидаемо — быстрее); e2e cosine≥0.99, perplexity-дельта ≤0.1%; `ci_gate.py` зелёный.

**Итого ориентир:** ~21-28 рабочих дней (включая фазу 0 и срез 1 с обязательным нативным I2_S). Срезы последовательны; каждый завершённый — приготовлен к мержу с merge-gate тестами.

## 9. Риски и митигации

| Риск | Влияние | Митигация |
|---|---|---|
| **Ручная сборка SPIR-V** (неверные ID, operand counts, type-forwards, коды опкодов) | ядро не валидируется/падает | хелперы `emit_unpack_2bit`/`emit_ternary_from_code`/`emit_mad_loop`; ручной SPIR-V-эталон §4.10; `spirv-val` + `.spvasm` diff на каждом модуле; опкоды из `spirv.hpp` (§4.2), НЕ magic numbers |
| **Нет dp4a на Polaris** | медленнее CUDA-донора | путь memory-bound — выигрыш от 2-bit bandwidth доминирует; loop unrolling; цель ≤20% к baseline; LUT-распаковка в shared memory как будущая оптимизация |
| **Битовая раскладка** (нативный GGML MSB-first/страйд-32 vs P0 LSB-first; «общая ошибка» reference и шейдера) | тихо неверные веса, ложная зелёность black-box | единый источник истины §3.3; **white-box bit-test §7.4b против ручного расчёта** (не против другой реализации); reference §6.4 ↔ переупаковщик §3.4 ↔ шейдер §4.5 сверены; тест фиксирует layout |
| **Кодировка `code−1` vs CUDA `code−2`** | дрейф/неверный знак | чистое `code−1` везде; численный тест §7.4a против ggml-пути; документировано в коде |
| **Точность 1.58-bit** | дрейф от CPU/float | per-tensor f32-scale; анализ error-propagation §7.9 → откалиброванные допуски; e2e cosine≥0.99, perplexity≤0.1% |
| **Тип целевой модели ≠ I2_S** (может быть TQ2_0/TL1) | §3-§6 неприменимы | **фаза 0 §0**: инспекция реального gguf до разработки; OQ-7; архитектура срезов устойчива, меняются лишь битовые формулы |
| **Локация/гранулярность scale** (хвост f32 vs `_scale`-тензор; per-tensor vs per-row) | NaN/неверный масштаб; слом дизайна буферов | фаза 0 §0 (OQ-1, OQ-3); `gguf_inspect.py`; обе схемы проверяются; per-row → расширение §4.3/§4.7 (binding 3) уже заложено |
| **Точный GGUF type id / адрес block_i2_s** | неверная загрузка | подтверждено: `I2_S=36` (`ggml.h:393`), `block_i2_s` на `ggml-common.h:271-273` (НЕ 243-244 — там tq2_0); block 256/64Б; сопутствующие 34/35/37/38/39 в enum |
| **Отсутствует OpenCL-fallback I2_S** | strict-режим падает; non-strict медленный | дефолтный fallback — **CPU-dequant→F32** (корректно, §5.9), НЕ Q4_0-перенаправление; OQ-4 для постоянного решения |
| **baseline для perf-gate отсутствует** | нечего сравнивать | baseline = dequant→F32 на том же backend (§7.6); OpenCL-тернарный baseline только при реализации OQ-4(a) |
| **Тестовая модель не найдена в CI** | тест падает «model not found» | `BITNET_DUMMY_GGUF_PATH` (пусто → skip), `tests/generate_test_models.sh`, `SKIP_RETURN_CODE 77` (§7.7) |

## 10. Открытые вопросы (требуют подтверждения)

Помечены приоритетом: **[BLOCK-0]** — закрыть в фазе 0 до среза 1; **[S1]/[S3]** — закрыть к соответствующему срезу.

1. **[BLOCK-0] Локация масштаба I2_S в целевых GGUF.** В `quantize_i2_s` scale пишется как **f32** в хвост данных тензора (`ggml-bitnet-mad.cpp:90-91`, смещение `n/4`). Но конвертеры BitNet (`convert-hf-to-gguf-bitnet.py`, ~строка 831 для TL-типов) могут писать отдельный тензор `<name>_scale` (возможно bf16). **Подтвердить на реальном артефакте** (`gguf_inspect.py`), какая схема и какой dtype scale используются для весовых тензоров, и поддержать её в `read_tensor_quantized()`. Также подтвердить семантику применения scale (§3.1): `W = scale·ternary` достаточно, или есть доп. нормировка.

2. **[S3] Layout P0 vs P1 для production-ядра.** Срез 2 использует P0 (host-переупаковка §3.4, простой шейдер). Для срезов 3-4 решить: оставить P0-переупаковку (стоимость один раз при загрузке) или перейти на нативный GGML (P1, MSB-first/страйд-32) в шейдере ради нулевой стоимости загрузки и меньшего пика памяти. Влияет на §4.5/§4.7 и §6.5. Решение принять по результату профилирования среза 2.

3. **[BLOCK-0] Гранулярность scale.** План исходит из per-tensor (один f32). Если целевая модель использует per-channel/per-block — потребуется расширение Tensor API (`quant_scale_axis`/`quant_block_size`), валидации, и дополнительный binding/push-constant в ядре (§4.3 закладывает binding 3). **Подтвердить per-tensor в фазе 0** (`gguf_inspect.py`); при per-channel/per-block — расширить §6.5-6.6 и §4.3/§4.7 до начала среза 2.

4. **[S1-решение] OpenCL-fallback для I2_S.** Сейчас отсутствует. Дефолт плана — CPU-dequant→F32 (§5.9). Выбрать постоянный production-вариант: (a) полная OpenCL-реализация `matmul_f32_i2_s_m1` (даёт прямой OpenCL perf-baseline §7.6), (b) постоянный dequant→F32-fallback с предупреждением о производительности, или (c) строгий `MCL_CHECK(false, "I2_S requires Vulkan")` в OpenCL-режиме. **НЕ** использовать Q4_0-перенаправление (меняет семантику). Решение — к концу среза 1.

5. **[S3] Корректность маппинга `code−1` в W2A8.** Подтверждено алгебраически (§3.2) и проверяется тестом §7.4a (срез 1, и повтор на W2A8 в срезе 3): чистое `code−1` (без CUDA-смещения `−2`, `bitnet_kernels.h:42`) даёт идентичный результат CPU-пути `ggml_vec_dot_i2_i8_s` (unsigned `maddubs` с `q8∈{0,1,2}` + структурная поправка). Финальное подтверждение — на реальных весах в срезе 3.

6. **[BLOCK-0] K%16 и хвост.** Подтвердить (`gguf_inspect.py` по shape тензоров), что все matmul-формы целевой модели имеют K кратно 16/256. Если нет — хвостовая обработка §4.8 обязательна с первого среза, а не опционально в срезе 4.

7. **[BLOCK-0] Фактический тип квантования целевой модели.** План построен под `I2_S=36`. Реальная модель может использовать `TQ2_0=35` (тоже 2 бита, но с f16-scale внутри блока, иной layout, `ggml-common.h:243-247`), `TQ1_0=34`, `TL1=38`, `TL2=39`. **Определить в фазе 0** (`gguf_inspect.py`). Если тип ≠ I2_S — переписать §3.3/§6.1/§6.4 под фактический формат (битовые формулы и парсинг scale), сохранив архитектуру §4-§8.

8. **[S3] Гранулярность `scale_a` (квантование активаций).** Для W2A8 активации квантуются в int8. Если per-row (per-token) — `scale_a` становится массивом длины M и требует binding 3 (§4.3/§4.7); если глобальный — push-constant. Решение — при проектировании квантования активаций в срезе 3.

---

**Сводка артефактов изменений (файлы):**
- Фаза 0: `tools/gguf_inspect.py` (новый), `docs/BITNET_GGUF_FACTS.md` (новый)
- Формат: `include/motifcl/core/dtype.hpp`, `src/core/types.cpp`
- GGUF: `include/motifcl/gguf.hpp`, `src/gguf.cpp`, `src/nn/hf_compat.cpp`
- Ядро: `src/runtime/vulkan_backend.cpp`, `include/motifcl/runtime/vulkan_backend.hpp`
- Диспетч/интеграция: `src/ops/matmul.cpp`, `src/runtime/microkernel.cpp`, `src/nn/linear.cpp`
- Тесты/гейты: `tests/test_vulkan_ternary_matmul.cpp` (новый), `tests/CMakeLists.txt`, `tests/generate_test_models.sh` (новый), `tests/fixtures/ternary_m1_16x8.spvasm` (новый), `docs/kernel_validation_contracts.json`, `tools/ci_gate.py`, `baselines/*`

**Сводка источников истины (донор, перепроверено):** `ggml.h:391-396` (type id: TQ1_0=34, TQ2_0=35, **I2_S=36**, I8_S=37, TL1=38, TL2=39); `ggml-common.h:72` (QK_K=256), `ggml-common.h:271-273` (**block_i2_s = qs[QK_K/4] = 64 Б, scale НЕ внутри блока**; строки 243-247 — это block_tq2_0, не путать); `ggml-bitnet-mad.cpp:63,65-72,76-78` (i2_scale=max|src|, кодировка q8∈{0,1,2}); `ggml-bitnet-mad.cpp:81-91` (упаковка MSB-first/страйд-32, **хвостовой f32 scale на смещении n/4**); `ggml-bitnet-mad.cpp:226-228` (MAD-распаковка сдвигами 6/4/2/0, подтверждает MSB-first); `ggml-bitnet-mad.cpp:242-245,295` (unsigned maddubs, `s[row]=(float)sumi`); `bitnet_kernels.h:29,42` (interleaving; GPU-смещение `code−2` — НЕ копируем). SPIR-V опкоды — из `spirv.hpp` (SPIRV-Headers, Vulkan-1.3-совместимая ревизия), значения сверены в §4.2.
