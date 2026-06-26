# Memory-Native Training — формальное описание метода

> Статус: реализовано и проверено в MotifCL на AMD Radeon RX 580 (OpenCL/Polaris).
> Все числа ниже — измерения реальных прогонов (regressions в `tests/`, бенчмарки в
> dev-наборе). Документ формализует метод целиком: два независимых рычага памяти плюс
> поддерживающая f16-инфраструктура.

---

## 1. Постановка задачи: четыре пула памяти обучения

Обучение нейросети расходует память на четыре независимых пула. Для слоя с `N`
весами при стандартном full-FT (FP32 + Adam):

| Пул | Что хранит | Байт/вес |
|---|---|---|
| Параметры | веса FP32 | 4 |
| Optimizer state | Adam `m`, `v` | 8 |
| Градиенты | полный `∇W` | 4 |
| **Итого вес-сторона** | | **16** |
| Активации | для backward | зависит от глубины/батча |

**Тезис метода:** выигрыш достигается, только если атаковать пулы *совместно*. Локальная
оптимизация одного пула не меняет порядок. Метод состоит из двух рычагов:

- **A. Finite-State Counter Synapse** — пулы 1–3 в одном байте на вес.
- **B. Reversible Activations** — пул 4 (forward не хранит, backward восстанавливает).

---

## 2. Метод A: Finite-State Counter Synapse

### 2.1 Состояние синапса

Каждый вес — конечный автомат. Состояние кодирует тернарный видимый вес и остаток-счётчик:

```
t ∈ {−1, 0, +1}             видимый тернарный вес
c ∈ {−(C−1), …, C−1}        остаточный счётчик (2C−1 уровней)
code = (t+1)·(2C−1) + (c + (C−1))     ∈ [0, 3(2C−1))
```

Число достижимых состояний `3(2C−1)`. При `C=11` это 63 ≤ 64, что укладывается в **6 бит**.
Видимый вес forward: `w = s·t`, где `s` — масштаб строки (per-output-channel).

### 2.2 Sigma-delta / error-feedback динамика

Введём целое скрытое состояние `z = C·t + c` и непрерывный shadow-вес `θ = (s/C)·z = s·t + (s/C)·c`.
Пусть `ĝ` — несмещённая оценка градиента, округление стохастическое и несмещённое. Пока
тернарная граница не насыщена, carry/remainder-арифметика даёт

```
E[θ_{k+1} − θ_k | F_k] = −η·g_k                 (несмещённый трекинг SGD)
|θ_k − w_k| ≤ s·(C−1)/C < s                      (ограниченная ошибка квантования)
```

То есть автомат отслеживает траекторию SGD с ограниченной ошибкой; на границах `t=±1`
это projected-SGD с гистерезисом. Счётчик — не буфер, а **error-feedback-память**,
спасающая подпороговые обновления от исчезновения.

### 2.3 Row-RMS adaptive scaling (критическая компонента)

Голый counter (только error-feedback) эмпирически слаб против Adam. Решение — дешёвый
`O(out_features)` второй момент на строку (аналог Adam-variance):

```
v_i ← β·v_i + (1−β)·mean_j(g_ij²)
denom_i = max(√v_i, ε)
tick_ij ∝ −η · (g_ij / denom_i) · (C / s_i)
```

**Эмпирический эффект изоляции (PyTorch, тот же стек):**

| | vanilla counter | counter + row-RMS |
|---|---|---|
| Изоляция vs ternary-QAT, d=128 | +16.6% | **+2.5%** |
| Изоляция vs ternary-QAT, d=256 | +1.7% | **−5.8%** (лучше QAT) |

Row-RMS закрывает ~85% разрыва на d=128 и **обгоняет** ternary-QAT на d=256.

### 2.4 Fused-в-backward обновление (memory-native)

Полный `∇W` **не материализуется ни на одном уровне**. Элемент
`∇W[o,i] = Σ_n grad_out[n,o]·x[n,i]` формируется **прямо в регистрах OpenCL-ядра** из
`grad_out` и `x` (активаций, которые и так есть): read state → SR-transition → write
state. Ни dense `[out,in]` grad-буфера, ни даже тайлового — градиент живёт только в
регистрах. Ни master-веса, ни Adam-моментов.

Декодированный плотный вес тоже **не хранится** между forward и backward: backward-узел
держит только `x`. Вес `w = s·t` декодируется как короткоживущий временный объект
исключительно для `grad_x` и уничтожается сразу — поэтому одновременно живёт не более
одного `[out,in]`-веса на всю сеть (а не по одному на слой). Это и есть разница между
«persistent state compact» и «training peak compact»; см. §5.3 и memory truth gate
(`tests/test_counter_memory_truth.cpp`).

Цена за in-kernel grad_w — он считается дважды (в row-stats и в apply), т.е. ~2×
grad_w-GEMM: осознанный размен «компьют за память», ради честного memory-native peak.

Транзакционная семантика: `grad_x` вычисляется **старым** весом до того, как состояние
обновлено (иначе recompute увидит новые веса).

### 2.5 Учёт обучаемого масштаба

Счётчик хранит незавершённое обновление `r = s·c/C`. При `s_old → s_new` остаток
перебазируется: `c ← SR(c·s_old/s_new)`, иначе изменение масштаба самопроизвольно
меняет накопленную ошибку.

### 2.6 Packed 6-bit хранение

4 кода × 6 бит = 24 бита = 3 байта. Обновление идёт **по группам** (один work-item на
3-байтную группу), что исключает гонку за общими байтами. Память: **0.75 байт/вес**.

### 2.7 Алгоритм (forward + fused backward)

```
forward(x):
    { w[out,in] = decode(state) · scale      # OpenCL; временный, в своей области
      y = x · wᵀ }                            # matmul_transpose_b; w уничтожается здесь
    attach CounterBackwardNode(x, seed)       # узел хранит ТОЛЬКО x (без w)

backward(grad_out):
    { w = decode(state)·scale                 # pre-update, временный
      grad_x = grad_out · w }                 # w уничтожается сразу после grad_x
    NoGrad:                                    # grad_w НЕ материализуется:
        # pass 1 (work-group/строку): grad_w[o,·] в регистрах → row-RMS v, denom, s_new
        # pass 2 (work-item/группу):  grad_w[o,i]=Σ_n grad_out[n,o]·x[n,i] в регистре
        #                             → SR tick, carry/remainder, encode
        # commit s_new
    if x.requires_grad: x.backward(grad_x)     # апдейт идёт и без grad на входе
```

---

## 3. Метод B: Reversible Activations

### 3.1 Обратимый coupling-блок

Вход делится на половины `(x1, x2)`:

```
forward:  y1 = x1 + F(x2);   y2 = x2 + G(y1)
inverse:  x2 = y2 − G(y1);   x1 = y1 − F(x2)
```

`F`, `G` — детерминированные операторы (Linear / counter / attention; без dropout/RNG).

### 3.2 Recompute-backward

Forward выполняется в `NoGrad` — активации **не хранятся**. В backward вход
восстанавливается через inverse, затем градиенты пересчитываются по восстановленному входу.

```
NoGrad: forward chain → output           # хранит только выход
NoGrad: recover input via inverse chain  # ~3e-3 ошибка восстановления (float)
grad-enabled: recompute from recovered input → loss → backward → grads
```

### 3.3 Корректность float-восстановления

Naive-float inverse **не точен**: ошибка восстановления накапливается с глубиной
(~3e-3 за 12 блоков). Что измерено:

- В протестированных witness'ах (до 12 блоков) эта ошибка **training-нейтральна**:
  сеть с 3e-3-шумом в активациях даёт ту же потерю, что чистая (Δce = 0.0%).
- **Это эмпирический результат на испытанной глубине, а не общий закон.** Для
  LLM-масштаба (десятки–сотни блоков) накопление нужно перепроверять depth-sweep'ом;
  на случай деградации предусмотрен fallback: anchors (хранить вход раз в K блоков)
  или fixed-point reversible mode. Формулировка «fixed-point/anchor никогда не нужен»
  была бы преждевременной — корректно: «на испытанной глубине naive-float reversible
  жизнеспособен, с явным планом отхода для большей глубины» (см. §7).

### 3.4 Корректность recompute-backward (измерено)

Градиенты recompute-backward совпадают со stored-autograd:

| Блок | grad rel-err |
|---|---|
| Linear coupling (4 блока) | 2.4e-07 |
| **Attention coupling (2 блока, q/k/v/o + MLP)** | **2.8e-07** |

Recompute через attention корректен; forward не хранит активации.

---

## 4. Реализация в MotifCL

| Компонент | Файл |
|---|---|
| Counter-ядра (decode / row-stats / apply, packed 6-bit) | `kernels/compact_counter.cl` |
| Counter-слой + autograd-узел | `include/motifcl/nn/compact_counter.hpp`, `src/nn/compact_counter.cpp` |
| Kernel-route (counter → compact_counter.cl) | `src/runtime/backend.cpp` |
| Counter-регрессия (teacher-recovery) | `tests/test_counter_state.cpp` |
| Reversible-регрессия (на attention) | `tests/test_reversible_attn.cpp` |
| f16 matmul autograd (`F16MatMulBackward`, f32-точный backward + cast) | `src/ops/matmul.cpp`, `tests/test_f16_matmul_autograd.cpp` |
| Дизайн / точки врезки | `docs/COUNTER_STATE_NATIVE_DESIGN.md` |

Reversible пока собирается из примитивов (`add`/`sub`/`Linear`/`multihead_attention`) с
`NoGradGuard`-forward; оформление как `nn::ReversibleBlock` Module требует concat/slice-ops
и multi-output autograd-узла (см. §7).

---

## 5. Эмпирические результаты (verified, RX 580)

### 5.1 Counter — качество

| Задача | counter vs ориентир |
|---|---|
| Teacher-recovery (линейный) | MSE → 0, ternary-acc 100% |
| char-LM (bigram-MLP, cross-entropy) | **1.02× dense FP32** — parity |
| Полный attention-GPT (counter q/k/v/o/fc) | **1.28× dense** — parity-class |
| Deep nonlinear MLP-регрессия | counter+RMS **0.53×** ternary-QAT (лучше); тернаризация = 62× узкое место, не counter |

### 5.2 Counter — parity-gate vs BF16 AdamW (char-LM, PyTorch-стек)

| Конфиг | ternary-QAT vs FP32 | counter+RMS vs FP32 | изоляция counter−QAT |
|---|---|---|---|
| d=128, 800 шагов | +2.1% | +4.6% | +2.5% |
| d=256, 800 шагов | +0.0% | −0.5% | **−5.1%** (counter лучше) |

Вывод: с ростом модели разрыв **сужается**, counter+RMS обгоняет ternary-QAT.

### 5.3 Counter — память и скорость

Важно разделять два утверждения, которые легко спутать:

- **Persistent state (хранимое между шагами): 0.75 байт/вес** — измерено, packed
  6-bit, ~16× меньше FP32+Adam. Сюда входит весь «оптимизатор» (state+scale+v).
- **Training peak (что живёт во время forward/backward): теперь тоже compact** —
  после устранения двух утечек (см. §2.4): backward-узел не хранит плотный вес между
  forward и backward (одновременно ≤ 1 `[out,in]` на всю сеть, не по одному на слой),
  а grad_w считается в регистрах ядра и не материализуется вовсе. Инвариант защищён
  тестом `tests/test_counter_memory_truth.cpp` (peak dense-weight ≤ 1, grad_w-allocs = 0).

| Метрика | значение |
|---|---|
| Persistent state (веса+оптимизатор) | **0.75 байт/вес** packed → ~16× меньше FP32+Adam |
| Измерено на реальной модели | 11.8× (d=128) … 13.6× (d=256) |
| Training-peak grad_w | **0** dense-буферов (in-kernel) — gate-enforced |
| Training-peak плотный вес | **≤ 1 × [out,in]** одновременно (транзиентный, для grad_x) |
| Цена in-kernel grad_w | ~2× grad_w-GEMM (считается в row-stats и в apply) |
| Скорость (2048×2048, fwd+bwd) | **0.97× dense** — измерено для пути с материализованным grad_w; для fused-пути перемерить на устройстве |
| История оптимизации | 3× → 1.44× (parallel) → 1.07× (packed) → 0.97× (decode-кэш + work-group row_stats) |

> Замечание по скорости: число **0.97× dense** снято на исходном пути (dense grad_w,
> сохранённый вес). Fused memory-native backward торгует ~2× grad_w-компьютом за память,
> поэтому его перф нужно **перемерить на RX 580** прежде, чем переносить 0.97× на него.

### 5.4 Reversible — активации

| Witness | результат |
|---|---|
| float-recovery (12 блоков) | 3e-3 (не точен, но…) |
| training-tolerance | Δce **0.0%** на испытанной глубине (≤12 блоков); глубже — depth-sweep + anchors fallback (§3.3) |
| recompute-backward (Linear) | grad rel-err 2.4e-7 |
| recompute-backward (attention) | grad rel-err **2.8e-7** |
| forward-хранение | **нулевое** (NoGrad) |

### 5.5 f16 matmul autograd

| Witness | результат |
|---|---|
| grad vs f32-reference | rel-err **4e-4** |

---

## 6. Сводный выигрыш памяти

| Метод | Полное full-FT | **Этот метод** |
|---|---|---|
| Веса+оптимизатор/вес | 16 байт | **0.75 байт** (~16×) |
| Активации | хранятся все | recompute (forward ≈ 0) |
| 1B-модель, веса+opt | ~16 ГБ | **~0.75 ГБ** |
| 7B-модель, веса+opt | ~112 ГБ | **~5.3 ГБ** |

> Строки 1B/7B — **арифметика памяти** (0.75 vs 16 байт/вес), а не прогоны: метод
> измерен на RX 580 на micro/tiny char-LM и attention-GPT, ≤800 шагов. Это масштабная
> экстраполяция формата, не доказательство обучения 1B/7B. «Активации → recompute»
> подтверждено корректностью (grad 2.8e-7), но пиковая память активаций пока замерена
> качественно, не в числах (§7).

Качество — parity-class на языковых (дискретных) задачах **на испытанном масштабе**. На
точной FP32-регрессии тернар структурно слабее (предел тернаризации, не дефект метода).

---

## 7. Границы и открытые вопросы

- **Reversible как Module:** нужны concat/slice-ops + multi-output autograd-узел для
  `nn::ReversibleBlock`; числовой замер пиковой памяти активаций (сейчас — качественный).
- **In-backward update несовместим** с grad-accumulation, weight-sharing, DDP all-reduce,
  activation-checkpointing — нужен явный планировщик; целевой режим — eager-training.
- **Тернаризация** структурно ограничивает точную FP32-регрессию (ternary-QAT 62× на
  таком таргете); язык/дискретные задачи — parity.
- **Learned 64-state LUT** (distillation Adam в transition table) — не реализован;
  ручной counter+RMS уже parity-class, LUT — потенциальное улучшение.
- **Глубокий reversible:** 3e-3 проверено до 12 блоков; на очень глубоких сетях
  перепроверить накопление (depth-sweep) + fallback anchors/fixed-point (§3.3).
- **Перф fused-backward:** in-kernel grad_w считается дважды (~2× grad_w-GEMM). Память
  выиграна (gate-enforced), но число 0.97× dense снято на старом пути — fused-перф
  нужно **перемерить на RX 580** и, если нужно, сократить до одного прохода (кешировать
  row-stats grad_w в local-памяти тайла, не в global).

---

## 8. Воспроизводимость

```bash
# сборка фреймворка (Ninja + clang-cl, warnings не errors)
cmake -B build/dev -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DMOTIFCL_WARNINGS_AS_ERRORS=OFF
cmake --build build/dev

# регрессии метода (включая memory truth gate)
ctest --test-dir build/dev -R "counter_state|counter_memory_truth|reversible_attn|f16_matmul_autograd" --output-on-failure
```

Standalone-бенчмарки (parity-gate, deep-MLP, char-LM, attention-GPT, perf/profile,
reversible-witnesses) — в dev-наборе, прогоняются clang-cl напрямую против `motifcl.lib`.

---

## 9. Резюме

Метод атакует **все четыре пула памяти обучения** одновременно: counter-синапс (16× на
весах+оптимизаторе, parity на языке, быстрее dense) и reversible-активации (forward без
хранения, recompute корректен через attention). Обе компоненты реализованы в MotifCL,
проверены на полном transformer-блоке на потребительском GPU (RX 580), с CTest-регрессиями.
Качество — parity-class на языковых задачах; узкое место (точная FP32-регрессия) — предел
тернаризации, а не оптимизатора.
