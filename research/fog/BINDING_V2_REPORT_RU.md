# FOG Binding v2: точное связывание факта в latent-памяти

Дата: 2026-08-13.

## Короткий итог

Дополнительные probes уточнили прежний диагноз. Провал legacy FOG был не
только проблемой нового answer-BOS: в обученных checkpoints линейный probe,
малый MLP и query-conditioned slot probe не обнаружили обобщаемого сигнала
целевого значения уже в состояниях writer — от query-state backbone до
proposal и persistent memory. Одновременно старый reader игнорировал даже
заведомо правильный oracle-вектор, пока его отдельно не обучили.

FOG Binding v2 исправляет оба места контролируемым способом:

1. query используется как адрес и сравнивается с адресами строк до
   permutation-invariant pooling;
2. выбранный payload переносится без обучаемой V/O-ротации;
3. первый payload-slot защищён от workspace mixing, а production memory
   переиспользует фиксированные `K=4` slots без компрессии;
4. первый токен ответа читается непосредственно из primary latent, без нового
   пустого BOS, который должен заново найти факт.

На operator-disjoint exact-code lookup это дало **4032/4032 на закрытом test во
всех трёх seed**. На отдельной задаче переноса четырёх цифр — **4096/4096 exact
match во всех трёх seed**. Контрфактические вмешательства уничтожают результат,
то есть ответ причинно зависит от query и protected payload.

После финальной доработки тот же принцип перенесён в полный production pipeline
на **10,000,039 параметров**: реальные token embeddings, четырёхслойный
backbone, reusable workspace `K=4`, cosine tied direct head и checkpoint
loader. Три независимых seed дали **4032/4032 на locked test**; zero,
target-deranged и query-cyclic дали **0/4032**. Обучался только один scalar
резкости адресного softmax, 40 шагов; мигрированные lexical weights оставались
заморожены.

Это важная положительная проверка механизма точного связывания, но не
доказательство полноценного reasoning и не решение арифметики. Задачи ниже —
синтетический lookup. Structured gates используют ортогональные коды, а полный
10M gate — фиксированные токены из реального 8192-элементного embedding/LM-head
codebook. Успешный 10M результат получен только при `R=1`.

## 1. Что показали probes старой архитектуры

Диагностика использовала seed-0 checkpoints `fog_full` и `fog_strict` из
предыдущего matched-эксперимента. Writer был полностью заморожен. Probes
обучались только на train mappings и оценивались на 1024 validation-примерах с
непересекающимися mapping operators.

В validation распределение классов не идеально равномерно: majority baseline
равен 155/1024 = 15.14%, а macro chance остаётся 12.5%. Поэтому отдельные
значения около 15% не являются найденным сигналом.

| представление | FOG full | FOG strict | интерпретация |
|---|---:|---:|---|
| mean всех rows + query, linear | 12.89% | 12.60% | pooling не сохраняет mapping |
| mean всех rows + query, MLP | 12.89% | 11.82% | нелинейный probe не помогает |
| точный query-conditioned выбор row, linear | **100.00%** | **100.00%** | payload существует во входе |
| dot-product query-conditioned выбор row, linear | **100.00%** | **100.00%** | достаточно обычной адресации |
| final memory, learned slot-attention | 12.11% | 13.48% | после writer доступный binding не найден |

Mean rows действительно почти не зависит от перестановки: максимальное
отклонение составило `8.94e-8` для full и `5.96e-8` для strict. Это ожидаемо,
поскольку

\[
\sum_i (k_i + v_{\pi(i)}) = \sum_i k_i + \sum_i v_i,
\]

то есть сумма знает набор ключей и значений, но не знает пары
`key → value`.

Линейные и MLP-probes были также поставлены на:

- query-позицию и последний hidden state backbone на обоих reasoning-шагах;
- proposal каждого шага;
- persistent memory после каждого шага;
- mean и flatten представления финальных proposal/memory.

Их validation accuracy лежала в диапазоне 10.64–15.14%, а macro accuracy —
около 12.5%. Это не доказывает математическое отсутствие любой информации в
векторах, но показывает, что точный target не доступен проверенным линейным,
небольшим нелинейным или learned-attention reader. Важнее всего, что контроль
с адресным выбором той же исходной строки достигает 100%.

### Oracle-reader

Вместо writer в strict decoder подавался точный код
`sqrt(d_model) · one_hot(target)`, повторённый по memory slots.

До адаптации reader результаты normal/zero/target-deranged совпадали с majority
baseline 15.14%: старый reader игнорировал даже идеальную память. После
обучения reader-side `backbone + classifier + answer_bos + neutral` при
замороженных input embeddings, planner и compressor:

| checkpoint-источник | oracle normal | oracle zero | oracle target-deranged |
|---|---:|---:|---:|
| FOG full | **100.00%** | 15.14% | **0.00%** |
| FOG strict | **100.00%** | 13.38% | **0.00%** |

Следовательно, интерфейс выразительно способен прочитать точный код после
специального обучения, но прежний end-to-end objective не сформировал ни
надёжный writer, ни использующий память reader.

Источник: `binding_diagnostics_seed0.json`.

## 2. Изменение архитектуры

### 2.1 Разделение address и payload

Вместо общего контекстного pooling сначала строится адресное распределение:

\[
a_i(q)=\operatorname{softmax}_i\!\left(
\frac{(W_q q)^\top(W_k k_i)}{\sqrt r}
\right),
\qquad
p(q)=\sum_i a_i(q)v_i.
\]

`AddressedPayloadBinding` обучает только Q/K-проекции для сравнения. Payload
`v_i` копируется теми же attention weights без обучаемых V/O-проекций. Это
важное отличие: writer может научиться адресу, не изобретая одновременно
произвольную систему координат для самого факта.

### 2.2 Protected payload

Slot 0 содержит только выбранный payload. Query определяет attention weights,
но не прибавляется residual-путём к содержимому primary slot. Этот slot:

- не проходит через auxiliary slot mixing;
- не перезаписывается workspace FFN;
- помещается в `ReusableLatentMemory` как slot 0 без компрессии;
- на следующей итерации заменяется новым primary напрямую, тогда как только
  auxiliary tail мягко смешивается с предыдущим состоянием;
- проверяется тестом на точное равенство `memory[:, 0] == primary`.

Размер памяти остаётся `K=4` при любой глубине: identities четырёх slots
переиспользуются, а не растут как `K × R`. Content-independent compression в
production v2 отсутствует. Остальные slots сохраняют обычный параллельный
workspace и могут обучаться для более сложной обработки. Таким образом, точный
carrier и приблизительное латентное вычисление больше не обязаны использовать
один и тот же канал.

### 2.3 Direct first-token readout

В legacy-схеме decoder получал новый answer-BOS и должен был заново извлечь из
памяти уже найденный факт. В `query_bound_v2` первый токен ответа вычисляется
непосредственно из primary latent через `DirectLatentReadout` и tied LM head.
Для токенов 2+ используется реальный предыдущий ответ в одной последовательной
timeline вместе с prompt, memory и contentful readout state. Пустого
BOS-retrieval hop нет.

Production-контракт реализован полями:

- `architecture_version="query_bound_v2"`;
- `binding_mode="query_conditioned"`;
- `readout_mode="direct_latent"`;
- `protected_binding_slots=1`;
- `binding_offsets=(2,)` в текущем 10M preset: в сериализации gate value token
  расположен через две позиции после key token.

Текущий preset содержит 10,000,039 уникальных параметров и reusable `K=4`.
Это отдельная архитектура; legacy checkpoints не переименовываются в v2.
Миграция переносит совместимые lexical/backbone tensors и tied token codebook,
а новые binding-компоненты получает как явную инициализацию, не как результат
lexical pretraining.

`DirectLatentReadout` нормирует направление primary, а `CosineTiedHead`
сравнивает его с нормированными строками той же token embedding matrix. Поэтому
извлечённый token embedding остаётся в собственной системе координат словаря;
нет обучаемого decoder, который мог бы скрыто переименовать latent-коды.

## 3. Exact-code operator-disjoint gate

Контрольная задача та же: новая перестановка восьми состояний, перемешанные
строки и запрос одного соответствия. Split определяется хешем всей
перестановки: train/validation/test mappings не пересекаются. Случайный уровень
— 12.5%.

В строгом exact-code варианте:

- keys — замороженный ортогональный codebook;
- values — другой замороженный ортогональный codebook;
- classifier tied с value codebook;
- primary переносит payload без V/O-ротации;
- обучаются адресация и вспомогательные компоненты, а не произвольный decoder
  latent-кода.

Размер модели — 49,664 уникальных параметра, из них 48,640 trainable. Каждый
seed обучался 200 шагов с batch 64. Validation до открытия test: 1024/1024 во
всех трёх seed.

Locked test содержит 4032 примера и был вычислен из сохранённых checkpoints:

| seed | normal | zero primary | target-deranged primary | query-deranged |
|---:|---:|---:|---:|---:|
| 0 | **4032/4032** | 531/4032 | **0/4032** | **0/4032** |
| 1 | **4032/4032** | 531/4032 | **0/4032** | **0/4032** |
| 2 | **4032/4032** | 531/4032 | **0/4032** | **0/4032** |

У всех режимов один test stream SHA-256:
`30d79d0feacba5a79898a9c42547e8b339ffbed4fda482990faad5fc0dadd095`.
Zero primary даёт 13.17%, потому что нулевые logits выбирают один класс и
повторяют его частоту в конечной выборке. Target-deranged и query-deranged дают
ровно 0%, поскольку перестановка имеет уникальное значение для каждого ключа.

Источники:

- `artifacts/binding_v2_exactcode_validation/summary.json`;
- `artifacts/binding_v2_exactcode_locked_test_seed{0,1,2}.json`.

## 4. Точный перенос четырёх цифр

Чтобы проверить, не ограничивается ли carrier одним из восьми классов, создан
отдельный gate. Каждая из восьми строк содержит случайную последовательность
из четырёх десятичных цифр. Query должен выбрать одну строку, а модель —
сохранить все четыре позиции. Split хешируется по всей таблице.

Ключи и цифры используют замороженные непересекающиеся ортогональные codebooks.
Одна address distribution выбирает целый payload `[4, d_model]`; каждая цифра
читается тем же frozen digit codebook. Trainable параметров — 1,152. Обучение:
200 шагов, batch 64.

Validation exact match был 1024/1024 во всех трёх seed. Locked test:

| seed | normal exact | digit accuracy | zero exact | target-deranged exact | query-deranged exact |
|---:|---:|---:|---:|---:|---:|
| 0 | **4096/4096** | **100.00%** | 1/4096 | **0/4096** | **0/4096** |
| 1 | **4096/4096** | **100.00%** | 1/4096 | **0/4096** | **0/4096** |
| 2 | **4096/4096** | **100.00%** | 1/4096 | **0/4096** | **0/4096** |

NLL на цифру в normal равен 0.000617 / 0.000613 / 0.000622 для seed 0/1/2.
При zero digit accuracy равен 10.39%, то есть практически десятичному chance.

Источники:

- `artifacts/binding_digits_decisive_validation/summary.json`;
- `artifacts/binding_digits_locked_test_seed{0,1,2}.json`.

## 5. Что доказано, а что нет

### Подтверждено

- Непрерывный latent-вектор способен хранить точное дискретное значение; дело
  не в недостатке битовой ёмкости самого вектора.
- Обычный permutation-invariant pooling уничтожает связь `key → value`, даже
  если помнит все ключи и все значения.
- Query-conditioned address selection сохраняет эту связь.
- Protected payload способен безошибочно перенести одну метку и четыре
  позиционно различимые цифры на unseen tables.
- Полный 10M token/backbone pipeline способен точно перенести выбранную
  идентичность из tied 8192-token codebook при замороженных lexical weights.
- Zero/deranged/query interventions показывают причинную зависимость результата
  от primary payload и выбранного адреса.

### Не подтверждено

- Lookup не равен многошаговому reasoning. Здесь нет композиции нескольких
  отношений, выбора алгоритма или проверки промежуточного вывода.
- Случайная четырёхзначная строка не является арифметикой: модель ничего не
  складывает и не переносит разряды, а выбирает уже присутствующий payload.
- Ортогональные frozen codebooks значительно чище настоящих token embeddings.
- 10M gate использует заданный offset и синтетическую сериализацию; он не
  подтверждает, что модель самостоятельно выделит semantic address/payload
  relations в свободном естественном тексте.
- Успех первого protected slot не подтверждает полезность остальных parallel
  workspace slots или recurrent latent depth.

## 6. Статус 10M v2

### 6.1 Что именно прошло

Финальный token-level gate использует текущую production-модель на 10,000,039
параметров. Prompt проходит через token embedding и четырёхслойный backbone,
query-conditioned writer выбирает value-token при единственном отношении
`binding_offsets=(2,)`, primary переносится через reusable `K=4`, а ответ
выбирается cosine head, tied с тем же словарём из 8192 embeddings.

Главный BF16 checkpoint:
`checkpoints/fog_binding_v2_10m_token_lookup_bf16.pt`, SHA-256
`d9a5523adc85a049d970f51c2bd75c6f88c64a533d8614077334999a1f4c5960`.
Legacy TinyStories BF16 checkpoint сохранён как secondary lexical artifact.

Перед обучением проверен полный oracle vocabulary-copy contract:
**8192/8192 embeddings декодируются в собственный token**, минимальный logit
margin — 0.579111. Таким образом, writer должен найти и скопировать правильный
лексический код, а не обучить новую классификационную таблицу.

Во всех трёх запусках:

- `reasoning_steps=1`;
- 40 optimizer-шагов, batch 32, всего 1280 binding-примеров;
- train scope — `sharpness`;
- единственный trainable parameter — scalar `planner.bind.logit_scale`;
- migrated token embeddings/backbone, отдельно инициализированные Q/K
  directions, workspace и оба tied vocabulary heads не обновлялись;
- validation была открыта до test и дала 1024/1024 normal;
- затем сохранённый checkpoint отдельно оценён на 4032 test-примерах.

Locked-test результаты:

| model seed | normal | zero | target-deranged | query-cyclic | address hit | correct-address mass | address entropy | NLL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42, primary | **4032/4032** | **0/4032** | **0/4032** | **0/4032** | **100%** | 98.4234% | 0.11891 | 6.41837 |
| 0 | **4032/4032** | **0/4032** | **0/4032** | **0/4032** | **100%** | 96.6230% | 0.22108 | 6.42868 |
| 1 | **4032/4032** | **0/4032** | **0/4032** | **0/4032** | **100%** | 99.8313% | 0.01563 | 6.41170 |

Все три locked evaluations использовали один stream SHA-256:
`344b99f6f49ff0f56d4c567a92dfa58433b9931b934d18f9d8691776e01103f8`.
Соответствующие correct-address masses на validation были 98.4456%, 96.4910%
и 99.8273%; address hit — 100% во всех случаях.

Источники:

- seed 42: `artifacts/binding_v2_10m_token_lookup_validation.json` и
  `artifacts/binding_v2_10m_token_lookup_locked_test.json`;
- seed 0: `artifacts/fog_binding_v2_10m_token_lookup_seed0_validation.json` и
  `artifacts/fog_binding_v2_10m_token_lookup_seed0_locked_test.json`;
- seed 1: `artifacts/binding_v2_10m_token_lookup_seed1_validation.json` и
  `artifacts/binding_v2_10m_token_lookup_seed1_locked_test.json`.

### 6.2 Почему при 100% accuracy NLL остаётся высоким

Normal NLL около 6.42 нельзя скрывать. Для 8192 классов uniform NLL равен
`ln(8192)=9.01091`: правильный токен стабильно занимает первое место, но frozen
cosine head распределяет значительную вероятность между тысячами соседних
лексических embeddings. Обучение меняло только резкость address selection, а
не output temperature, token codebook или lexical calibration. Поэтому этот
gate подтверждает **точный top-1 перенос идентичности токена**, но не уверенный
или хорошо откалиброванный языковой прогноз.

### 6.3 Граница результата

Это первый положительный перенос binding v2 в полный 10M token/backbone
pipeline и сильный причинный результат: обнуление carrier, память от примера с
другим target и циклическая замена query уничтожают все 4032 ответа во всех
seed. Но проверено только одно адресное извлечение за **R=1**.

Не проверены:

1. полезность recurrent refinement при `R>1`;
2. композиция нескольких отношений или нескольких lookup-шагов;
3. вычисление нового значения, которого нет в prompt;
4. настоящее сложение с разрядами и carry;
5. semantic binding в свободном естественном тексте;
6. GSM8K и общее reasoning-качество.

Следующие gates: двухшаговая композиция при R>1, затем арифметика с unseen
числами, и только после этого реальный SFT/GSM8K. Текущий честный статус:
**точное token-level binding в полной 10M модели подтверждено; полноценный 10M
latent reasoner и арифметика не подтверждены**.
