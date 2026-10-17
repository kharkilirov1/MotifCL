# MN-MoE15M — MoE-обучение на RX 580 (MotifCL, нативный Vulkan, FP32)

Короткий практический гайд: как собрать, запустить и читать логи обучения
разреженно-маршрутизируемой MoE-модели **с нуля** на Radeon RX 580 (Polaris 10).
Архитектура и обоснования — в `docs/MOE_FROM_SCRATCH_DESIGN.md`, все измерения и
границы доказательств — в `docs/MOE_PRETRAIN_RESULTS.md`.

## Что это за модель

- Бэкбон как у проверенного `ternary15m`: dim 288, 6 слоёв, MHA-6, RoPE, RMSNorm,
  vocab 32000, seq 256, SwiGLU, LM-голова **не связана** (`TIE_HEAD=0`).
- Вместо плотного FFN — **7 экспертов** по `CounterStateLinear` (hidden 192 каждый,
  6-битное конечное состояние вместо FP32-весов и моментов оптимизатора) плюс
  всегда включённый shared-эксперт 192.
- Роутер: `W_r [288,7]` + bias, выбор **top-1** через `fog_hard_route7` (Vulkan-ядро
  с настоящим автоградом: кандидаты получают градиент только на своих токенах,
  логиты — straight-through softmax).
- Балансировка: Switch-aux-loss как **явный градиент роутера** (`AUX_WEIGHT`) плюс
  bias на выбор (`BALANCE_GAMMA`, кламп `BALANCE_CLAMP`).

Новые SPIR-V-ядра не нужны (и `glslc` на машине нет) — вся маршрутизация собрана на
существующих ядрах.

## Требования

- Vulkan-устройство (на этой машине проверено: `Radeon RX 580 Series`, apiVersion 1.3.260).
- CMake ≥ 3.20, Ninja, MinGW GCC (сборка проверена на GCC 13.2.0 из Strawberry).
- **Vulkan SDK не нужен** — SPIR-V закоммичен в `src/runtime/vulkan_spirv_kernels.inc`.
- Данные: `data/tinystories/{train.bin,validation.bin}` (packed uint16, запись = seq+1).

## Обязательная настройка памяти

```
MOTIFCL_VK_PREFER_DEVICE_LOCAL=1
```

Без неё текущий дефолт рантайма размещает тензорные буферы в host-visible памяти
(GTT/system RAM), и **та же самая конфигурация идёт в 2.0× медленнее на MoE и в 2.8×
на плотной модели при бит-идентичных loss** (измерено A/B, см.
`docs/MOE_PRETRAIN_RESULTS.md` §2). Причина — коммит FOG-интеграции, инвертировавший
предпочтение типа памяти.

## Сборка

```powershell
cmake --build build/release --target motifcl_train_moe15m motifcl_train_ternary15m -j 8
```

Только проверки:

```powershell
cmake --build build/release --target test_moe7_reference test_moe7_balance -j 8
ctest --test-dir build/release -R test_moe7
```

## Запуск

Один прогон MoE:

```powershell
$env:DATA_DIR='data/tinystories'; $env:OUT_DIR='out/moe_long'
$env:STEPS='20000'; $env:BATCH='8'; $env:SEQ='256'
$env:EVAL_EVERY='500'; $env:EVAL_RECORDS='96'; $env:CKPT_EVERY='2000'
$env:SAVE_CKPT='1'; $env:RESUME='1'
$env:BALANCE_GAMMA='0.01'; $env:AUX_WEIGHT='0.01'; $env:BALANCE_CLAMP='4'
$env:MOTIFCL_VK_PREFER_DEVICE_LOCAL='1'
.\build\release\tools\motifcl_train_moe15m.exe
```

Серия сравнений (MoE / плотная 768 / плотная 384 / MoE e384 / абляции балансировки),
арм'ы идут последовательно, каждый со своим `OUT_DIR`:

```powershell
.\ports\rx580\moe\run-moe-comparison.ps1 -DryRun
.\ports\rx580\moe\run-moe-comparison.ps1 -Steps 20000 -Arms moe_e192,dense_h384
```

Ключевые переменные: `HIDDEN_EXP`, `HIDDEN_SHARED` (ширина экспертов), `AUX_WEIGHT`,
`BALANCE_GAMMA`, `BALANCE_CLAMP`, `EVAL_BATCH` (по умолчанию `min(32, BATCH)` — eval не
должен быть тяжелее обучения), `SAVE_CKPT`, `RESUME`. У плотного тренажёра есть
`HIDDEN` — им задаётся контроль по равной активной ширине.

## Что смотреть в логе

```
step 500 loss 3.6250 ema 3.8372 lr 0.00100  2066 tok/s  eta 5.4 h  devmem 3327/3359 MB  load 0.139 ... spread 0.024  bias0[-1.580..2.000]  spread0 9.714 gap0 3.065
step 500 VAL loss 3.3416  ppl 28.264
```

- `devmem live/peak` — сколько device-памяти держит рантайм (учёт аллокаций).
- `load ... spread` — доля токенов на каждого из 7 экспертов и разброс max−min;
  `spread > 0.3` означает начинающийся коллапс роутера.
- `bias0[...]` — балансировочный bias слоя 0; упор в кламп означает, что он перестал
  адаптироваться (проверьте `BALANCE_CLAMP`).
- `VAL ... ppl` — валидация; считается при выключенном обновлении состояний счётчиков.

Сводка по логам:

```powershell
python tools\moe_log_summary.py out\moe_long\log.txt out\dense_long\log.txt
```

## Оценка готового чекпойнта

```powershell
$env:OUT_DIR='out/moe_long'; $env:EVAL_ONLY='1'; $env:EVAL_RECORDS='96'
$env:MOTIFCL_VK_PREFER_DEVICE_LOCAL='1'
.\build\release\tools\motifcl_train_moe15m.exe
# EVAL_ONLY steps=20000 records=96 tokens=24576 loss=2.8510 ppl=17.30 top1=0.3721 top5=0.6402
```

Режим загружает чекпойнт по `meta.txt`, обучающих шагов не делает, состояния счётчиков
не обновляет и считает те же фиксированные 96 валидационных записей, что и цикл обучения,
плюс top-1/top-5 точность предсказания следующего токена.

## Чекпойнты и возобновление

Чекпойнт (`CKPT_EVERY`) пишет состояния счётчиков всех экспертов, веса роутера, bias,
моменты Adam и `meta.txt` с номером шага. Повторный запуск с `RESUME=1` продолжает с
`meta.txt` (проверено: шаг 300 → 301–305, состояние и bias восстановлены). При падении
прогон теряет не больше `CKPT_EVERY` шагов.

## Ориентиры (измерено на этой машине)

| метрика | значение |
|---|---|
| пропускная | 2010–2066 tok/s при B8×T256, ~1.0 с/шаг |
| device peak | 3359 МБ (обучение); eval с `EVAL_BATCH=32` раздувал до 6780 МБ — исправлено |
| val loss / ppl | 3.3416 / 28.264 на шаге 500 (горизонт затем сокращён до 12000 шагов на чекпойнте 2000, см. `docs/MOE_PRETRAIN_RESULTS.md`) |
| загрузка экспертов | spread 0.014–0.040 при `AUX_WEIGHT=0.01` |
| B16×T256 | **не работает**: `VK_ERROR_DEVICE_LOST` на ~21 шаге |

## Границы

1. Вычисления плотные, маршрутизация разреженная: считаются все 7 кандидатов,
   `fog_hard_route7` выбирает строки. Ускорения от разреженности в этой версии нет.
2. Число экспертов жёстко равно 7 (ограничение существующего ядра); общий top-k
   требует новых SPIR-V и `glslc`.
3. Накопления градиента нет: один backward = одно обновление состояния счётчика.
4. Любые числа пропускной способности MotifCL приводите вместе с политикой памяти.
