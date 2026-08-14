# FOG v3 на RX 580 — быстрый старт

Это native Vulkan-порт FOG v3 поверх **MotifCL main**. ROCm/PyTorch для этого пути не нужен.

## 0. Что установить один раз (Windows)

Нужны:

- свежий AMD Radeon Vulkan driver;
- CMake;
- Ninja;
- Python 3;
- `glslc` из Vulkan SDK (либо путь к `glslc.exe` в `MOTIFCL_GLSLC`).

## 1. Собрать и сразу проверить RX 580

Открой PowerShell в корне репозитория:

```powershell
powershell -ExecutionPolicy Bypass -File ports/rx580/fog/setup-and-check-fog-rx580.ps1
```

Скрипт последовательно:

1. генерирует SPIR-V для новых FOG shaders;
2. собирает `rx580-release` с Vulkan и без OpenCL;
3. проверяет Vulkan forward/backward новых ops;
4. обучает маленький hard-operator gate;
5. обучает controlled полный путь `binder -> generated latent -> recurrent re-address`.

Не начинай длинное обучение, пока последняя строка не будет:

```text
FOG RX580 PORT IS READY FOR LEXICAL PRETRAINING
```

## 2. Подготовить TinyStories

```powershell
python -m pip install tokenizers datasets
python tools/fog/prepare_token_stream.py `
  --tinystories 100000 `
  --output data/fog_tinystories_100k.i32
```

Для первого теста можно взять 10 000 вместо 100 000.

## 3. Первый pretrain

На RX 580 8 GB начни консервативно:

```powershell
powershell -ExecutionPolicy Bypass -File ports/rx580/fog/train-fog-v3.ps1 `
  -TokenFile data/fog_tinystories_100k.i32 `
  -Steps 1000 `
  -Batch 1 `
  -Seq 128 `
  -Lr 0.0003
```

Если VRAM/драйвер ведут себя стабильно, следующий запуск можно попробовать с `-Batch 2`.

## 4. Что прислать при ошибке

Если hardware gate падает — пришли **полный stdout/stderr** от `setup-and-check-fog-rx580.ps1`. По имени упавшего gate будет сразу видно, где проблема:

- shader/Vulkan primitive;
- hard routing / BLOCK_PRODUCT;
- GQA binder / recurrent machine;
- либо уже lexical trainer.

## Важная совместимость

C++ Vulkan-порт сохраняет смысл FOG v3, но пока не совпадает с Python checkpoint по layout параметров. Главные backend-адаптации:

- address projection 320D вместо Python compare-rank 80D, чтобы переиспользовать существующий Vulkan GQA backward;
- portable typed control/scratch update проще Python auxiliary workspace;
- machine checkpoint MotifCL имеет собственный `.mclp` формат.
