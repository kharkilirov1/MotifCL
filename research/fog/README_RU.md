# FOG Latent Reasoner: обученный 10M-кандидат

Пакет содержит полностью запускаемые legacy и binding-v2 варианты примерно на
10M параметров, BPE-токенизатор, реальные обученные checkpoints, скрипты
воспроизводимого обучения и оценки, тесты и экспериментальные отчёты.

Главный checkpoint —
`checkpoints/fog_binding_v2_10m_token_lookup_bf16.pt`, SHA-256
`d9a5523adc85a049d970f51c2bd75c6f88c64a533d8614077334999a1f4c5960`.
Это 10,000,039-параметрический `query_bound_v2`: он наследует lexical
embedding/backbone от 1,600-шагового TinyStories pretrain и дополнительно
обучил один address-sharpness scalar 40 шагов на token lookup. Checkpoint прошёл
locked binding test, но остаётся исследовательской `R=1` lookup-моделью, а не
готовой разговорной или reasoning-моделью.

Предыдущий lexical checkpoint
`checkpoints/fog_10m_tinystories_1_36m_pretrained_bf16.pt` остаётся secondary
legacy-вариантом на 10,035,848 параметров. Он увидел 1,358,852 target-токена на
закреплённом срезе TinyStories. Последующие legacy GSM8K/addition эксперименты
не решили задачи: строгая GSM8K модель в основном выдавала частый ответ `12`.

Дополнительный matched-эксперимент отделил размер модели от интерфейса памяти.
Малый обычный Transformer при тех же данных и ширине почти идеально решил
новую структурированную lookup-задачу в 2 из 3 seed, тогда как FOG с большим
числом параметров остался на уровне случайного ответа во всех 6 запусках
full/strict. Подробности и отрицательные controls — в
`MATCHED_EXPERIMENT_REPORT_RU.md`.

Последующая probe-диагностика уточнила причину: legacy writer не формировал
доступное точное связывание `key → value`, а старый BOS-reader не использовал
даже oracle-memory без отдельного обучения. Реализован отдельный
`query_bound_v2`: query-conditioned адресация до pooling, payload без V/O-ротации,
один защищённый memory slot и direct first-token readout. В контролируемом
exact-code gate v2 прошёл locked test `4032/4032` во всех трёх seed; отдельный
четырёхзначный payload gate прошёл `4096/4096` во всех трёх seed. Эти результаты
подтверждают механизм точного binding.

Финальная проверка перенесла механизм в полный **10,000,039-параметрический**
token/backbone pipeline. Три seed независимо дали `4032/4032` на locked test;
zero, target-deranged и query-cyclic — `0/4032`. Address hit равен 100%, а
средняя масса правильного адреса — 98.42% / 96.62% / 99.83%. Обучался лишь
один scalar sharpness 40 шагов; мигрированные lexical weights оставались
замороженными. Это подтверждает точный token-level binding при `R=1`, но **не**
полноценное reasoning или арифметику. Подробности — в
`BINDING_V2_REPORT_RU.md`.

Weights в релизных checkpoints экспортированы из FP32 в BF16: это переносимые
inference checkpoints без optimizer state и с небольшой необратимой потерей
точности. Полные training checkpoints создаются командами ниже.


## Register Machine v3: модель уже можно собирать и обучать

В репозитории теперь есть отдельная **model-ready** архитектура
`register_machine_v3` на **10 245 433 параметра**. Это не замена и не
переименование обученного v2 checkpoint: текущий v3 checkpoint — честная
инициализация исследовательской модели, а не готовая LLM.

В v3 добавлены типизированные регистры `value/control/scratch`, recurrent
`value -> next address`, конечная грамматика операторов `READ / IDENTITY /
BLOCK_PRODUCT / 4 learned bilinear operators`, hard forward routing со
straight-through gradient, необязательный HALT и публичный one-step transition
для JVP-диагностики.

Команда:

```bash
python build_fog_machine.py \
  --output checkpoints/fog_machine_v3_10m_init.pt \
  --reasoning-steps 8 --jvp-probes 2 --seed 42
```

проверяет полный цикл `instantiate -> forward/backward -> machine gradients ->
JVP -> save -> strict reload`. Текущий checkpoint проходит этот gate. В EXP-037
полная v3-машина на контролируемой задаче создаёт промежуточное значение,
которого нет среди payload prompt, и повторно использует его как следующий
адрес: **100% на глубинах 1,2,3,4,6,8 на 3/3 seed**. Это доказательство
механизма generated latent computation, но не natural-language reasoning.

Полная спецификация и training ladder: `research/MODEL_READY_V3.md`.

## 1. Состав legacy-модели

| параметр | значение |
|---|---:|
| словарь | 8 192 |
| ширина `d_model` | 320 |
| decoder-блоки / головы | 4 / 5 |
| FFN | 1 344 |
| latent slots `K` | 4 |
| latent-итерации `R` | 4 |
| память `N` | 8 |
| compare rank | 80 |
| максимальная общая длина | 512 |
| **уникальные параметры** | **10 035 848** |

На каждом шаге планировщик создаёт четыре непрерывных latent-состояния. Память
растёт `4 → 8`, затем обучаемо сжимается `12 → 8` на шагах 3 и 4. Словарная
голова применяется только к позициям ответа; промежуточные мысли не декодируются
в текст. Embedding и LM head используют одну матрицу весов.

`max_seq_len=512` одновременно включает prompt, восемь latent memory позиций и
teacher-forcing prefix ответа.

Текущий v2 preset — отдельная архитектура на 10,000,039 параметров. Он
сохраняет lexical geometry, но использует query-conditioned cosine binder,
один protected slot, reusable workspace `K=4` без накопления/компрессии,
`binding_offsets=(2,)` и cosine tied direct head. Legacy checkpoint нельзя
считать обученным v2 checkpoint: миграция переносит совместимые
lexical/backbone weights, после чего token-lookup gate обучает только один
sharpness scalar. Production preset допускает другую recurrent depth, но
положительный эксперимент проведён только с `reasoning_steps=1`.

## 2. Установка и проверка релиза

Нужен Python 3.10+ и PyTorch 2.3+.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python verify_release.py
python verify_release.py --forward-backward
```

`verify_release.py` по умолчанию загружает главный binding-v2 checkpoint и
совместимый `tokenizer/tinystories_3k_bpe.json`, проверяет 10,000,039
параметров, tied LM/direct heads, hashes и UTF-8 round trip. Флаг
`--forward-backward` дополнительно проверяет gradients binding Q/K. Secondary
legacy checkpoint можно передать явно через `--checkpoint`.
`demo.py` остаётся маленькой архитектурной демонстрацией, а не генератором
текста из обученного checkpoint.

## 3. Какие реальные данные использованы

Через Hugging Face Dataset Viewer были закреплены только следующие строки:

| данные | строки | SHA-256 JSONL |
|---|---:|---|
| TinyStories `train[0:3000]` | 3 000 | `445c316f…ecf5fce` |
| TinyStories `validation[0:300]` | 300 | `e1412321…c8a04a` |
| GSM8K `train[0:1024]` | 1 024 | `bfa19819…775057f` |

Полные hashes и число строк источника записаны в соседних файлах
`data_cache/*.manifest.json`. Официальный GSM8K `test` не скачивался и не
использовался. Внутри 1 024 train-строк seed 42 создаёт split 896/128.

Архив не перераспространяет исходные JSONL TinyStories/GSM8K: в нём оставлены
манифесты с хешами и скрипт ниже, который заново получает те же диапазоны строк.
Синтетические addition-файлы также воспроизводятся локальным генератором.

Команды получения тех же срезов:

```bash
python download_viewer_subset.py \
  --dataset roneneldan/TinyStories --config default --split train \
  --limit 3000 --output data_cache/tinystories_train_3000.jsonl

python download_viewer_subset.py \
  --dataset roneneldan/TinyStories --config default --split validation \
  --limit 300 --output data_cache/tinystories_validation_300.jsonl

python download_viewer_subset.py \
  --dataset openai/gsm8k --config main --split train \
  --limit 1024 --output data_cache/gsm8k_train_1024.jsonl
```

## 4. Что получилось

| эксперимент | teacher-forced validation | greedy exact match |
|---|---:|---:|
| TinyStories, первые 400 шагов | loss 4.4758, PPL 87.86 | — |
| TinyStories, суммарно 1 600 шагов | loss 3.7194, PPL 41.24, token acc. 32.93% | — |
| GSM8K, strict memory-only/final | loss 1.7380, token acc. 40.14% | 1/128 = 0.78% |
| GSM8K, full prompt/full CoT | loss 3.4921, token acc. 31.85% | 0/128 |
| сложение 0…19, strict memory-only | loss 0.8249, token acc. 68.56% | 5/80 = 6.25% |

Отдельная оценка на **всех** 300 validation-историях показала изменение от
случайной инициализации к финальному checkpoint: loss `9.0445 → 3.7529`, PPL
`8471.95 → 42.64`, token accuracy `0.0016% → 32.96%`. Значения 3.7194/41.24
выше — checkpoint-selection метрика на первых десяти validation-батчах.

Для строгого GSM8K ablation `normal / zero / shuffled memory` дал одинаковые
`1/128`; модель почти всегда отвечала `12`. На full-CoT генерации появились
повторы и повреждённый текст. Это отрицательный результат, а не рабочий
математический reasoner.

На pair-disjoint сложении held-out exact match составил `6.25% / 5.00% /
1.25%` для normal/zero/shuffled memory. На train — `7.50% / 3.75% / 2.50%`.
Падение при перемешивании — слабый признак зависимости от содержания памяти, но
6.25% слишком мало, чтобы считать сложение выученным. Высокая teacher-forced
token accuracy частично отражает предсказание EOS и отдельных цифр; решающая
метрика здесь — exact match.

### Matched-диагностика размера и архитектуры

На новой задаче поиска значения по ключу использованы непересекающиеся множества
таблиц, одинаковые minibatches и name-stable общая инициализация. Locked test
открывался один раз после выбора протокола:

| модель | параметры | test seed 0 / 1 / 2 |
|---|---:|---:|
| обычный Transformer, ответ на query | 69 184 | 100.00% / 26.46% / 99.98% |
| FOG full-prompt | 139 204 | 12.28% / 12.35% / 12.28% |
| FOG strict memory-only | 139 204 | 12.28% / 12.28% / 12.28% |

Случайный уровень — 12.5%. Обнуление и target-deranged перемешивание latent
memory почти не меняли ответы FOG. Однако lossless-контроль без planner и
сжатия — два прохода общей backbone с переносом всех hidden states и чтением
через новый BOS — также остался на случайном уровне после 1 000 шагов. Более
простой одно-проходный контроль с новым answer-BOS тоже провалился. Поэтому
первоначально провал был локализован шире компрессора — в latent-interface.

Frozen probes затем разделили writer и reader. Mean-pooling строк остался около
chance, а точный или обычный dot-product query-conditioned выбор строки дал
100% с linear и MLP probes. В то же время probes на query hidden, proposals и
persistent memory legacy FOG остались около macro chance 12.5%. Старый reader
не реагировал даже на точный oracle-код до отдельного обучения. Значит legacy
run содержит два дефекта: writer не сохраняет доступный binding, reader не
обучается его читать.

Исправленный controlled v2 дал следующие locked-test результаты:

| gate | seed 0 | seed 1 | seed 2 | causal intervention |
|---|---:|---:|---:|---:|
| exact-code lookup, 4032 примера | **100%** | **100%** | **100%** | target/query deranged: 0% |
| 4-digit payload, 4096 примеров | **100% exact** | **100% exact** | **100% exact** | target/query deranged: 0% exact |

В v2 query адресует rows до pooling; выбранный payload переносится без
обучаемой V/O-ротации, primary slot защищён от workspace mixing, reusable
`K=4` не применяет компрессию, а первый ответ читается напрямую без нового BOS
retrieval. Это положительный
результат для точного binding на unseen tables. Lookup не равен рассуждению, а
перенос четырёх случайных цифр не равен арифметике.

Полный 10,000,039-параметрический v2 прошёл mapping-disjoint token lookup:

| seed | locked normal | zero | target-deranged | query-cyclic | correct-address mass |
|---:|---:|---:|---:|---:|---:|
| 42 | **4032/4032** | 0/4032 | 0/4032 | 0/4032 | 98.42% |
| 0 | **4032/4032** | 0/4032 | 0/4032 | 0/4032 | 96.62% |
| 1 | **4032/4032** | 0/4032 | 0/4032 | 0/4032 | 99.83% |

Oracle vocabulary copy перед обучением — `8192/8192`. Каждый run обучал один
address-sharpness scalar 40 шагов при замороженных migrated embeddings и
backbone, а также замороженных инициализированных Q/K directions и cosine tied
head. Высокий normal NLL `6.42` при
100% top-1 означает слабую калибровку frozen 8192-way head; uniform NLL равен
`ln(8192)=9.01091`.

Итог:

- **получилось**: реальный BPE, измеримый lexical pretrain, переносимый
  checkpoint, воспроизводимый локальный pipeline и causal memory ablations;
- **подтверждено отдельно**: binding v2 точно переносит один label и
  четырёхзначный payload в controlled gates и value-token в полном 10M
  token/backbone pipeline;
- **не получилось**: надёжное рассуждение и обобщение на GSM8K/сложении;
- **статус**: exact binding current 10M v2 подтверждён только для одношагового
  lookup (`R=1`); композиция, `R>1`, арифметика и reasoning не подтверждены.
  Ни один checkpoint не является готовой LLM или доказанной reasoning-системой.

## 5. Точное воспроизведение выполненного обучения

BPE и исходная инициализация:

```bash
python train_real.py tokenizer \
  --local-data data_cache/tinystories_train_3000.jsonl \
  --max-samples 3000 --vocab-size 8192 \
  --output tokenizer/tinystories_3k_bpe.json

python train_real.py init-model \
  --tokenizer tokenizer/tinystories_3k_bpe.json \
  --architecture legacy_v1 \
  --output checkpoints/fog_10m_tinystories_3k_init.pt
```

TinyStories stage 1 и stage 2 (CPU/FP32; замените временные каталоги на свои):

```bash
python train_real.py pretrain \
  --tokenizer tokenizer/tinystories_3k_bpe.json \
  --init-checkpoint checkpoints/fog_10m_tinystories_3k_init.pt \
  --local-data data_cache/tinystories_train_3000.jsonl \
  --local-eval-data data_cache/tinystories_validation_300.jsonl \
  --checkpoint-dir runs/tinystories_stage1 \
  --device cpu --precision fp32 --sequence-length 128 \
  --batch-size 4 --gradient-accumulation 1 \
  --max-steps 400 --warmup-steps 20 --lr 3e-4 \
  --eval-every 100 --eval-batches 10 --save-every 100 --log-every 20

python train_real.py pretrain \
  --tokenizer tokenizer/tinystories_3k_bpe.json \
  --init-checkpoint runs/tinystories_stage1/best.pt \
  --local-data data_cache/tinystories_train_3000.jsonl \
  --local-eval-data data_cache/tinystories_validation_300.jsonl \
  --checkpoint-dir runs/tinystories_stage2 \
  --device cpu --precision fp32 --sequence-length 128 \
  --batch-size 4 --gradient-accumulation 1 \
  --max-steps 1200 --warmup-steps 20 --lr 1.5e-4 \
  --eval-every 200 --eval-batches 10 --save-every 200 --log-every 50
```

Строгий GSM8K run:

```bash
python train_real.py sft \
  --tokenizer tokenizer/tinystories_3k_bpe.json \
  --init-checkpoint runs/tinystories_stage1/best.pt \
  --local-data data_cache/gsm8k_train_1024.jsonl \
  --checkpoint-dir runs/gsm8k_strict \
  --device cpu --precision fp32 --seed 42 \
  --target-mode final --decoder-mode memory-only \
  --validation-size 128 --max-prompt-length 128 --max-answer-length 12 \
  --batch-size 8 --gradient-accumulation 1 \
  --max-steps 400 --warmup-steps 20 --lr 3e-4 --weight-decay 0.01 \
  --eval-every 100 --eval-batches 16 --save-every 100 --log-every 20
```

Full-prompt/full-CoT diagnostic, начатый от сильнейшего lexical checkpoint:

```bash
python train_real.py sft \
  --tokenizer tokenizer/tinystories_3k_bpe.json \
  --init-checkpoint runs/tinystories_stage2/best.pt \
  --local-data data_cache/gsm8k_train_1024.jsonl \
  --checkpoint-dir runs/gsm8k_full_cot \
  --device cpu --precision fp32 --seed 42 \
  --target-mode full --decoder-mode full \
  --validation-size 128 --max-prompt-length 128 --max-answer-length 128 \
  --batch-size 4 --gradient-accumulation 1 \
  --max-steps 400 --warmup-steps 20 --lr 2e-4 --weight-decay 0.01 \
  --eval-every 100 --eval-batches 16 --save-every 100 --log-every 20
```

Строгий addition gate, начатый от более сильного lexical checkpoint:

```bash
python generate_arithmetic_data.py
python train_real.py sft \
  --tokenizer tokenizer/tinystories_3k_bpe.json \
  --init-checkpoint runs/tinystories_stage2/best.pt \
  --local-data data_cache/addition_train.jsonl \
  --local-eval-data data_cache/addition_validation.jsonl \
  --checkpoint-dir runs/addition_strict \
  --device cpu --precision fp32 --target-mode final --decoder-mode memory-only \
  --max-prompt-length 64 --max-answer-length 8 \
  --batch-size 16 --gradient-accumulation 1 \
  --max-steps 800 --warmup-steps 20 --lr 3e-4 --weight-decay 0.01 \
  --eval-every 100 --eval-batches 5 --save-every 100 --log-every 20
```

Оценка memory interventions:

```bash
python eval_local_sft.py \
  --data data_cache/gsm8k_train_1024.jsonl \
  --tokenizer tokenizer/tinystories_3k_bpe.json \
  --checkpoint checkpoints/fog_10m_gsm8k_896_best_bf16.pt \
  --validation-size 128 --split validation --decoder-mode memory-only \
  --max-prompt-length 128 --max-answer-length 12 --max-new-tokens 8 \
  --interventions normal zero shuffle
```

## 6. Скрипт для более крупного реального датасета

`train_real.py` умеет стримить закреплённые ревизии TinyStories и GSM8K либо
читать локальные TXT/JSONL. Для серьёзного запуска обучите BPE минимум на 50 000
историй, создайте новый init checkpoint и затем увеличьте объём данных и число
шагов. Hash токенизатора входит в checkpoint и проверяется при загрузке.

```bash
python train_real.py tokenizer \
  --dataset-id roneneldan/TinyStories \
  --dataset-config default \
  --revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 \
  --max-samples 50000 --vocab-size 8192 \
  --output tokenizer/tinystories_50k_bpe.json

python train_real.py init-model \
  --tokenizer tokenizer/tinystories_50k_bpe.json \
  --architecture query_bound_v2 \
  --output checkpoints/fog_10m_50k_bpe_init.pt

python train_real.py pretrain \
  --tokenizer tokenizer/tinystories_50k_bpe.json \
  --init-checkpoint checkpoints/fog_10m_50k_bpe_init.pt \
  --checkpoint-dir checkpoints/pretrain \
  --sequence-length 256 --batch-size 8 --gradient-accumulation 4 \
  --max-steps 10000 --warmup-steps 500 --lr 3e-4 --precision auto
```

Обе training-команды поддерживают `--resume .../last.pt`. Для точного resume не
меняйте batch geometry, seed, scheduler или `max_steps`.

Точный binding-v2 checkpoint из архива воспроизводится из secondary lexical
checkpoint отдельно: миграция не переносит optimizer state и не выдаёт новые
модули за уже обученные.

```bash
python migrate_to_binding_v2.py \
  checkpoints/fog_10m_tinystories_1_36m_pretrained_bf16.pt \
  checkpoints/fog_binding_v2_10m_migrated_init_fp32.pt --seed 42

python train_token_binding_v2.py \
  --init-checkpoint checkpoints/fog_binding_v2_10m_migrated_init_fp32.pt \
  --output-checkpoint checkpoints/fog_binding_v2_10m_token_lookup_fp32.pt \
  --output-metrics artifacts/binding_v2_10m_token_lookup_validation.json \
  --steps 40 --batch-size 32 --learning-rate 0.1 --weight-decay 0 \
  --train-scope sharpness --reasoning-steps 1 --eval-examples 1024

python train_token_binding_v2.py \
  --evaluate-checkpoint checkpoints/fog_binding_v2_10m_token_lookup_fp32.pt \
  --evaluation-split test --eval-examples 4032 \
  --reasoning-steps 1 \
  --output-metrics artifacts/binding_v2_10m_token_lookup_locked_test.json

python export_checkpoint.py \
  checkpoints/fog_binding_v2_10m_token_lookup_fp32.pt \
  checkpoints/fog_binding_v2_10m_token_lookup_bf16.pt \
  --tokenizer tokenizer/tinystories_3k_bpe.json
```

`train_real.py init-model` теперь имеет явный `--architecture`.
`query_bound_v2` используется для нового v2 запуска, а
`--architecture legacy_v1` сохраняет точное воспроизведение старых опытов.

Локальный pretrain принимает `.txt` (документ на строку) или JSONL с полем
`text`. Локальный SFT принимает JSONL `{"prompt": "...", "response": "..."}`;
имена полей задаются через `--prompt-field` и `--response-field`. Для
`target-mode=final` response должен содержать маркер `####`.

## 7. Файлы

- `checkpoints/fog_binding_v2_10m_token_lookup_bf16.pt` — главный
  10,000,039-параметрический seed-42 token-binding checkpoint; это `R=1`
  lookup-модель, не готовый reasoner;
- `checkpoints/fog_10m_tinystories_1_36m_pretrained_bf16.pt` — secondary legacy
  lexical checkpoint, экспортированный в BF16;
- `checkpoints/fog_10m_gsm8k_896_best_bf16.pt` — неудачный strict GSM8K
  checkpoint, сохранённый для анализа;
- `checkpoints/fog_10m_addition_strict_best_bf16.pt` — слабый addition
  ablation;
- `tokenizer/tinystories_3k_bpe.json` — реально обученный BPE;
- `train_real.py` — tokenizer/init/pretrain/SFT/официальная evaluation;
- `download_viewer_subset.py` — малые воспроизводимые Dataset Viewer срезы;
- `eval_local_sft.py` — local exact match и memory interventions;
- `artifacts/real_training/` — фактические greedy evaluation JSON;
- `matched_structured_lookup_experiment.py` — воспроизводимый matched gate;
- `MATCHED_EXPERIMENT_PROTOCOL.md` и `MATCHED_EXPERIMENT_REPORT_RU.md` —
  заранее зафиксированный протокол и итоговая диагностика;
- `binding_diagnostics.py` и `binding_diagnostics_seed0.json` — frozen probes и
  oracle-reader локализация;
- `binding_v2_experiment.py` и `binding_digits_experiment.py` — exact-code и
  multi-digit binding gates;
- `BINDING_V2_REPORT_RU.md` — уточнённый диагноз, устройство v2, locked tests и
  границы интерпретации;
- `migrate_to_binding_v2.py` и `train_token_binding_v2.py` — явная миграция
  совместимых весов и token-level training gate для current v2;
- `artifacts/binding_v2_10m_token_lookup_{validation,locked_test}.json` и
  соответствующие seed-0/seed-1 JSON — финальная трёхseedовая 10M evidence;
- `artifacts/matched_experiment_final/metrics.json` — сводные машинные метрики;
- `TRAINING_REPORT_REAL_V2.md` — полный отчёт реального обучения;
- `EXPERIMENT_REPORT_V2.md` и `MODEL_CARD.md` — архитектурный аудит и
  ограничения;
- `tests/` — автоматические gates.

## Исследовательский workflow

Активные гипотезы отделены от уже подтверждённых результатов. См. [`research/README.md`](research/README.md) для процесса IDEA -> PROTOCOL -> RESULT -> DECISION, [`research/STATUS.md`](research/STATUS.md) для реестра доказательств и [`research/ROADMAP.md`](research/ROADMAP.md) для порядка следующих этапов. Текущий production-transfer эксперимент — EXP-002 (recurrent protected binding).

## Простое обучение FOG v3 одной командой

Для model-ready `register_machine_v3` добавлен wrapper `train_v3_easy.py`.

Быстрый smoke-run:

```bash
python train_v3_easy.py --install --recipe smoke
```

Первый нормальный локальный запуск:

```bash
python train_v3_easy.py \
  --install --recipe starter --device cuda \
  --text-data /path/to/train.jsonl \
  --text-eval-data /path/to/validation.jsonl
```

Если дополнительно передать `--sft-data`, wrapper автоматически выполняет staged SFT с увеличением latent depth. Повтор той же команды автоматически возобновляет незавершённые `last.pt` и пропускает уже законченные стадии.

Полная короткая инструкция: [`TRAIN_V3_EASY_RU.md`](TRAIN_V3_EASY_RU.md).
