# Дизайн: нативный finite-state counter-слой (`CounterStateLinear`) в MotifCL

> Статус: **РЕАЛИЗОВАНО (срезы 1–5)**. Основан на провалидированном PyTorch-прототипе
> (`counter_state_fused_v2.py` + `counter_state_rms.py`) и эмпирических гейтах
> на tinyshakespeare (см. раздел 1). Точки врезки сверены с текущим деревом
> MotifCL (autograd / nn / ops / kernels / runtime).

## 0. Статус реализации (срезы 1–5 готовы и verified на RX 580)

| Срез | Что | Статус |
|---|---|---|
| 1 | `encode/decode_state` (C++) | ✅ bit-exact vs PyTorch |
| 2 | forward (decode → `matmul_transpose_b`) | ✅ |
| 3 | fused-backward update (SR + row-RMS) | ✅ teacher-recovery MSE→0 |
| 4 | `nn::CounterStateLinear` + autograd-узел + CTest | ✅ `tests/test_counter_state.cpp` зелёный |
| 4.5 | deep nonlinear MLP / char-LM | ✅ char-LM parity 1.02× dense; deep MLP > ternary-QAT |
| 5 | **packed 6-bit** (0.75 байт/вес) + параллельные ядра + perf-gate | ✅ **0.97× dense** (быстрее!) — decode-кэш + work-group row_stats; 16× память |

Файлы: `kernels/compact_counter.cl` (decode/row-stats/apply, packed 4-кода/3-байта),
`include/motifcl/nn/compact_counter.hpp`, `src/nn/compact_counter.cpp`; kernel-route в
`src/runtime/backend.cpp`; сборка в `CMakeLists.txt`. Требование: `in_features % 4 == 0`.

Изоляция (PyTorch, тот же стек): counter+RMS **лучше** ternary-QAT (0.53× на MLP-регрессии);
на регрессии узкое место — тернаризация (62× даже у QAT+Adam), не counter. На языковых
(дискретных) задачах counter parity-class с FP32. Замеры памяти обучения: **11.8–13.6×**
меньше FP32+Adam (uint8), ~16× при packed 6-bit.

**Reversible-активации (раздел 6) — доказаны в движке:** recompute-backward корректен
(Linear grad-err 2e-7, **через attention** 2.8e-7, witnesses), forward в NoGrad не хранит
активации, naive-float recovery (~3e-3) training-нейтрален (Δce 0%). Регрессия
`tests/test_reversible_attn.cpp`. Counter в полном attention-GPT — проверен (parity 1.28×),
row-stats распараллелен (work-group reduction).
Осталось: оформить reversible как `nn::ReversibleBlock` Module (нужны concat/slice + multi-
output autograd-node) и числовой замер пиковой памяти активаций.

## 1. Зачем и что уже доказано

Цель — обучать тернарную (1.58-bit) модель так, чтобы **состояние оптимизатора
жило внутри самого веса**: каждый синапс — конечный автомат в 1 байте (uint8,
45/63 достижимых состояния = тернар `t∈{−1,0,+1}` + остаток-счётчик `c`), а
обновление вшито в backward по тайлам — полный `∇W` и Adam-моменты никогда не
материализуются.

Эмпирически подтверждено (CPU, tinyshakespeare, char-LM):

| Вопрос | Результат |
|---|---|
| Тернаризация vs FP32 | +2.1% (почти даром) |
| Vanilla counter vs ternary-QAT (d=128) | +16.6% — counter слаб |
| **+ row-RMS adaptive scaling (d=128)** | **+2.5%** — закрывает 85% разрыва |
| **+ row-RMS на d=256 (4× параметров)** | **−5.8%** — обгоняет ternary-QAT |
| Память (веса+оптимизатор) при обучении | **11.8× → 13.6×** меньше |
| Масштабирование | gap **сужается** с ростом, не растёт |

**Вывод для дизайна: `row-RMS` (per-row второй момент `v`) — не опция, а
обязательная часть update. Без него слой не конкурентен.**

Связь с уже собранным i2_s-путём: обученная counter-модель тривиально
экспортируется — `state → t` даёт тернарные веса, которые пакуются в GGUF `i2_s`
и исполняются существующим `ggml-bitnet-mad.cpp` / будущим Vulkan-ядром. То есть
counter-training и i2_s-inference — две стороны одного формата.

## 2. Хранение состояния (на слой `[out, in]`)

| Буфер | DType | Shape | Роль | Регистрация |
|---|---|---|---|---|
| `state` | `U8` (`dtype.hpp:12`) | `[out, in]` | автомат: `code = (t+1)*(2C−1) + (c+C−1)` | `Parameter` (но обновляется вручную в backward, не оптимизатором) |
| `scale` | `F32` | `[out, 1]` | per-row масштаб `s_i`, `w = s_i·t` | буфер (Tensor, не в `parameters()`) |
| `v` | `F32` | `[out, 1]` | per-row RMS второй момент | буфер |

`scale`/`v` — `O(out_features)`, ничтожны. Кодировка `encode/decode_state`
повторяет прототип: `levels = 2C−1`, `t = code/levels − 1`, `c = code%levels −
(C−1)`. По умолчанию `C=11` (63 состояния, лучший по абляции).

Замечание: 1 байт/вес — это память **обучения** (включает оптимизатор). Для
сравнения, BF16+Adam стоит ~12 байт/вес. Инференс упаковывается в 1.58 бит
(i2_s) отдельно.

## 3. Forward

`src/nn/compact_counter.cpp`, по образцу `src/nn/linear.cpp:29-50`:

1. `decode_counter_state` (OpenCL) → тернарный `t` (F32 или I8) из `state`.
2. `w = scale ⊙ t` (broadcast по строке) — либо слить в decode-ядро (выдавать
   сразу `w_row = s_i·t`).
3. `y = x · wᵀ` через существующий `matmul` (`src/ops/matmul.cpp`).
4. `out._set_grad_fn(make_shared<CounterBackwardNode>(x, module_ref))`.

Forward не материализует `∇W`; сохраняет только `x` (как `ReluBackwardNode`,
`activation.cpp:59-66`).

## 4. Backward (fused tile-update) — ядро дизайна

`CounterBackwardNode::backward(grad_out)` по образцу
`FusedSwiGLUMLPRMSNormResidualBackwardNode` (`src/ops/fused_transformer.cpp:254-319`):

```
grad_x = grad_out · w           // СТАРЫМ весом (декод из state до апдейта)
{ NoGradGuard guard;            // апдейт state не пишется в autograd-граф
  for tile [lo:hi] of rows:
      grad_w_tile = grad_outᵀ[:,lo:hi] · x        // FP32, только [tile, in]
      fused_counter_update(state[lo:hi], scale[lo:hi], v[lo:hi],
                           grad_w_tile, seed, lr, lr_scale, C, rms_beta, rms_eps)
}
if x.requires_grad: x.backward(grad_x)
```

Транзакционная семантика (критично, из раздела 6 research-дока): `grad_x`
считается **старым** весом до того, как любой тайл `state` обновлён. В прототипе
это гарантируется тем, что `grad_x.add_(go_i @ w_i)` идёт перед `_update_tile`.
В MotifCL — либо считать весь `grad_x` до цикла апдейта, либо держать апдейт в
`NoGradGuard` после полного прохода `grad_x`.

`seed` (uint32) генерируется на CPU за слой/шаг и передаётся скаляром в ядро
(как в dropout, `basic.cl:123-135`).

## 5. OpenCL-ядра — `kernels/compact_counter.cl`

Зарегистрировать маршрут в `KernelCache::source_file_for_kernel`
(`src/runtime/backend.cpp:155-174`): имена, содержащие `counter` → этот файл.

### 5.1 `decode_counter_state_f32`
Распаковка `uint8 → s_i·t`. Один work-item на элемент. Использует `levels`,
`C`, `scale[row]`. Образец вызова — `src/ops/quant.cpp:247-268`.

### 5.2 `fused_counter_update_rms` — главное ядро
Один work-item (или work-group) на строку тайла. На вход: `state`, `scale`,
`v`, `grad_w_tile` (FP32, `[tile, in]`), скаляры `lr, lr_scale, C, rms_beta,
rms_eps, seed, in_features`. Логика per-row (точная калька `RMSCounterLinear._update_tile`):

```
g_sq = mean_j(grad_w[row,j]^2)
v[row] = rms_beta*v[row] + (1-rms_beta)*g_sq
denom  = max(sqrt(v[row]), rms_eps)
grad_s = sum_j(grad_w[row,j]*t[row,j]) / sqrt(in_features)
s_new  = clamp(scale[row] - lr_scale*grad_s, 1e-5, 10)
for j in row:
    decode state[row,j] -> (t, c)
    tick = -lr * (grad_w[row,j]/denom) * (C / s_new)
    c2   = c * (scale[row]/s_new)                 // rebase остатка под новый scale
    cc   = stochastic_round(c2 + tick, rng(seed,row,j))   // mcl_uniform01, basic.cl:119
    carry = trunc(cc / C); rem = cc - carry*C
    t' = clamp(t + carry, -1, 1)
    if blocked: rem = sign(cc)*(C-1)
    state[row,j] = encode(t', clamp(rem, -(C-1), C-1))
scale[row] = s_new
```

Stochastic rounding обязателен и должен быть несмещённым (`mcl_uniform01`,
`basic.cl:110-121`). `denom`-floor `rms_eps` защищает ранние шаги от взрыва (в
прототипе подтверждено: с RMS нужен малый `lr≈3e-3`, не `0.04`).

## 6. Точки врезки (чек-лист файлов)

| Что | Файл | Опора |
|---|---|---|
| Класс слоя | `include/motifcl/nn/compact_counter.hpp` (новый) | `nn/linear.hpp:8-40` |
| Forward + Node | `src/nn/compact_counter.cpp` (новый) | `linear.cpp:29-50`, `fused_transformer.cpp:254-319` |
| Op-обёртки | `src/ops/compact_counter.cpp` (новый) | `quant.cpp:247-268`, `matmul.cpp:442-451` |
| Ядра | `kernels/compact_counter.cl` (новый) | `quant.cl`, RNG `basic.cl:110-135` |
| Маршрут ядра | `src/runtime/backend.cpp:155-174` | + ветка `counter→compact_counter.cl` |
| Сборка | `CMakeLists.txt` / `src/CMakeLists.txt` | добавить новые `.cpp` |
| Тесты | `tests/` (CTest) | сверка с CPU-reference (раздел 8) |

## 7. Дорожная карта (вертикальные срезы, TDD)

1. **Срез 1 — CPU-reference + encode/decode.** Перенести `encode/decode_state`
   как чистые C++ функции, юнит-тест против PyTorch-значений (битовая
   эквивалентность). Без GPU.
2. **Срез 2 — forward.** `decode_counter_state_f32` + `matmul`; тест: forward
   совпадает с прототипом на фиксированном `state` (допуск ~1e-5).
3. **Срез 3 — fused backward update.** `fused_counter_update_rms`; тест:
   teacher-recovery (как `counter_state_C_ablation`) — MSE→~0, тернар-acc высокий;
   и `grad_x` совпадает с reference.
4. **Срез 4 — слой + обучение.** `CounterStateLinear` как `Module`; повторить
   parity-gate (QAT vs counter+RMS) уже на MotifCL, сверить с PyTorch-числами.
5. **Срез 5 — packed 6-bit + perf.** Только после parity: упаковка `4 states/3
   bytes`, perf-gate против Q8/Q4 пути, снятие 3× замедления прототипа.

## 8. Верификация

- Reference: тот же `counter_state_rms.py` как эталон значений.
- Корректность: encode/decode битово; forward численно; update — teacher-recovery
  + `grad_x` сверка; обучение — parity vs ternary-QAT (целевая изоляция ≤ +3%).
- Stochastic: фиксировать `seed`, проверять несмещённость средним по прогонам.
- Perf-gate: `tools/perf_truth_gate.py` (срез 5), baseline — F32/Q8 linear.

## 9. Ограничения и открытые вопросы

- **In-backward update = side-effect.** Несовместимо с graph-capture replay
  (`autograd/graph.hpp:104-111`, `ARCHITECTURE.md:28`): апдейт держать в
  `NoGradGuard`, ядро помечать non-replayable. Для eager-training (целевой режим)
  — ок.
- **Несовместимо с** grad-accumulation, weight-sharing, DDP all-reduce,
  activation-checkpointing — как и прототип (явный отказ). Нужен один
  forward→backward на слой за шаг; при шаринге — explicit scheduler.
- **RNG-качество** `mcl_hash_u32` для несмещённого SR — проверить на срезе 3
  (хвостовая полировка чувствительна к смещению).
- **Хвостовой остаток** ~+2.5% (d=128): кандидаты — per-row momentum (первый
  момент, ещё `O(d)`) и lr-schedule; добавить как опции после среза 4.
- **scale/v как буферы**, но изменяются в backward — убедиться, что `state_dict`
  их сохраняет (persistent buffers), иначе чекпоинт потеряет оптимизатор.
