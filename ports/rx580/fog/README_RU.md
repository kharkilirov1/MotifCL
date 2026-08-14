# FOG v3 -> MotifCL main -> Radeon RX 580 (Vulkan)

Это порт model-ready FOG `register_machine_v3` на **MotifCL main**. Целевой backend для RX 580 — Vulkan. OpenCL в этом порте не требуется.

## Что уже портировано

### 1. 10M lexical backbone

Нативный `motifcl::nn::FogV3LexicalModel`:

- vocab 8192;
- `d_model=320`;
- 5 attention heads;
- 4 decoder blocks;
- FFN 1344 + GELU;
- RMSNorm;
- learned positions;
- tied token embedding / LM projection;
- Vulkan forward + backward + Adam.

Это позволяет выполнять первый реальный этап FOG-обучения (causal lexical pretrain) на RX 580 без PyTorch/ROCm.

### 2. FOG finite operator bank

Нативный `FogOperatorBankV3` содержит семь кандидатов:

1. READ;
2. IDENTITY;
3. BLOCK_PRODUCT;
4. FLEX_0;
5. FLEX_1;
6. FLEX_2;
7. FLEX_3.

Forward использует **hard argmax operator**, backward — straight-through softmax gradient. Это сохраняет closure operator family и повторяет ключевой принцип Python FOG v3.

### 3. Новые Vulkan kernels

Добавлены только FOG/port-specific пробелы. Matmul, attention, embedding, RMSNorm, GELU, CE и Adam уже были в MotifCL main и не переписывались.

Новые shaders:

- `mul_f32.comp`;
- `silu_f32.comp`, `silu_bwd_f32.comp`;
- `sigmoid_f32.comp`, `sigmoid_bwd_f32.comp`;
- `fog_block_product_f32.comp`, `fog_block_product_bwd_f32.comp`;
- `fog_hard_route7_f32.comp`;
- `fog_hard_route7_candidate_bwd_f32.comp`;
- `fog_hard_route7_logits_bwd_f32.comp`.

### 4. Query-conditioned register path

Добавлен `FogV3Model`, который объединяет один стабильный checkpoint layout для двух стадий. Structured machine API содержит:

- shared address projection + Vulkan GQA binder;
- value/control/two scratch registers;
- recurrent `value -> next query`;
- hard finite operator grammar;
- HALT probability head;
- cosine-tied direct vocabulary readout.

RX580 adaptation: binder использует full-width 320D address projection вместо Python 80D compare rank, потому что текущий Vulkan GQA backward MotifCL требует одинаковую q/k/v head width. Это сознательный backend trade-off (~77k дополнительных параметров против отдельного split-value attention-backward kernel).

### 5. Hardware gates

Перед длинным обучением запускаются три уровня:

- `test_fog_ops` — forward/backward новых Vulkan ops + structured read smoke;
- `10_fog_v3_operator_gate` — реально обучает hard router выбирать BLOCK_PRODUCT и проверяет recurrent generated state без decode/snap;
- `11_fog_v3_structured_machine_gate` — controlled полный путь `binding -> generated value -> recurrent re-address -> direct readout`, аналог цели Python EXP-037.

## Граница совместимости с Python v3

Machine-side structured semantics теперь присутствуют, однако C++ порт **не является побитово/параметрически идентичным** Python checkpoint:

- Python planner имеет более богатый auxiliary parallel workspace; portable Vulkan cell использует более простой typed control/scratch update поверх тех же основных register roles;
- address projection в RX580 path 320D, не 80D, по причине Vulkan GQA-backward выше;
- PyTorch `.pt` веса поэтому пока нельзя просто загрузить напрямую.

Это уже полноценный обучаемый MotifCL FOG path, но natural-language semantic reasoning всё равно остаётся задачей обучения, а не доказанным свойством порта.

## Windows: сборка

Новые GLSL shaders пока не входят в исходный MotifCL SPIR-V bundle, поэтому при первой сборке нужен `glslc`. Скрипт ищет его в `PATH`, `$env:VULKAN_SDK`, либо `$env:MOTIFCL_GLSLC`.

Из корня репозитория:

```powershell
powershell -ExecutionPolicy Bypass -File ports/rx580/fog/build-fog-vulkan.ps1
```

Скрипт:

1. компилирует/встраивает новые FOG Vulkan shaders;
2. конфигурирует `rx580-release` без OpenCL;
3. строит FOG trainer и hardware gates.

## Проверка RX 580

```powershell
powershell -ExecutionPolicy Bypass -File ports/rx580/fog/check-fog-rx580.ps1
```

Не начинайте длинный pretrain, пока этот скрипт не закончит строкой:

```text
FOG RX580 Vulkan hardware gates PASSED
```

## Подготовка TinyStories

Python-зависимости только для подготовки токенов:

```powershell
python -m pip install tokenizers datasets
```

Например 100 000 историй:

```powershell
python tools/fog/prepare_token_stream.py `
  --tinystories 100000 `
  --output data/fog_tinystories_100k.i32
```

Или свой JSONL:

```json
{"text":"Document one..."}
{"text":"Document two..."}
```

```powershell
python tools/fog/prepare_token_stream.py `
  --input data/train.jsonl `
  --output data/fog_train.i32
```

## Первый pretrain на RX 580

Безопасный старт для 8 GB VRAM:

```powershell
powershell -ExecutionPolicy Bypass -File ports/rx580/fog/train-fog-v3.ps1 `
  -TokenFile data/fog_tinystories_100k.i32 `
  -Steps 1000 `
  -Batch 1 `
  -Seq 128 `
  -Lr 0.0003
```

Если память и скорость нормальны, затем пробуйте `Batch 2`. Sequence length лучше увеличивать только после первого стабильного witness.

Повторный запуск с тем же checkpoint загружает веса. **Adam moments пока не сериализуются**, поэтому это weight-resume, а не bit-exact optimizer resume.

## Порядок запуска

```text
build-fog-vulkan.ps1
        ↓
check-fog-rx580.ps1
        ↓
prepare_token_stream.py
        ↓
train-fog-v3.ps1
```

Если `check-fog-rx580.ps1` падает, присылайте полный stdout/stderr — сначала исправляется backend/kernel path, и только потом запускается 10M training.
