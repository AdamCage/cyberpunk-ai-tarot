# Cyberpunk AI Tarot

Генерация недостающих 56 карт Младших Арканов колоды таро из Cyberpunk 2077 с помощью дообучения FLUX.1-dev через LoRA.

Пайплайн состоит из пяти шагов: генерация капшенов → тренировка LoRA → генерация 56 карт → добавление рамок и подписей.

---

## Содержание

- [Требования](#требования)
- [Структура проекта](#структура-проекта)
- [Настройка `.env`](#настройка-env)
- [Шаг 1 — Установка окружения](#шаг-1--установка-окружения)
- [Шаг 2 — Генерация капшенов](#шаг-2--генерация-капшенов)
- [Шаг 3 — Тренировка LoRA](#шаг-3--тренировка-lora)
- [Шаг 4A — Генерация карт локально (diffusers)](#шаг-4a--генерация-карт-локально-diffusers)
- [Шаг 4B — Генерация карт через ComfyUI](#шаг-4b--генерация-карт-через-comfyui)
  - [Базовый запуск](#базовый-запуск)
  - [Режим hypermeta](#режим-hypermeta)
  - [Режим benchmark](#режим-benchmark)
- [Шаг 5 — Добавление рамок и подписей](#шаг-5--добавление-рамок-и-подписей)
- [Справочник параметров `.env`](#справочник-параметров-env)
- [Справочник CLI-аргументов](#справочник-cli-аргументов)

---

## Требования

- **Python 3.11+** и [`uv`](https://docs.astral.sh/uv/) — менеджер пакетов
- **GPU с CUDA, ≥ 16 GB VRAM** (рекомендуется RTX 3090 / 4080 и выше)
- **FLUX.1-dev** — скачать через HuggingFace:
  ```powershell
  uv run huggingface-cli download black-forest-labs/FLUX.1-dev
  ```
- **ComfyUI** — только для шага 4B ([github.com/comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI))

---

## Структура проекта

```
cyberpunk-ai-tarot/
│
├── .env                             # пути к моделям и параметры (не в git)
├── pyproject.toml                   # зависимости проекта (uv)
│
├── generate_captions.py             # шаг 1: генерация .txt-капшенов для обучения
├── setup_training.py                # шаг 2: установка окружения (ai-toolkit + torch)
├── train.py                         # шаг 3: запуск LoRA-тренировки
├── generate_cards.py                # шаг 4A: генерация карт через diffusers (локально)
├── generate_cards_comfyui.py        # шаг 4B: генерация карт через ComfyUI API
├── add_card_borders.py              # шаг 5: добавление рамок и подписей
│
├── config/
│   ├── train_full.yaml              # конфиг полной тренировки (10000 шагов)
│   └── train_test.yaml              # конфиг тестового прогона (50 шагов)
│
├── data/
│   ├── originals/                   # 26 оригинальных артов (.webp) + .txt-капшены
│   ├── missing/
│   │   ├── meta.jsonl               # метаданные и промпты для 53 недостающих карт
│   │   ├── meta_v2–v7.jsonl         # история итераций промптов
│   │   └── _hyper_meta.jsonl        # индивидуальные гиперпараметры для финальных карт
│   ├── style_bible/
│   │   └── style_bible.json         # канонические правила стиля
│   ├── generated/
│   │   ├── res/                     # сырые PNG из генерации (без рамки)
│   │   ├── final/                   # финальные PNG с рамкой и подписью
│   │   └── benchmark/
│   │       ├── seed/                # результаты sweep по 10 seed
│   │       └── seeds/               # результаты произвольного sweep по seed
│   └── RussoOne-Regular.ttf         # шрифт для подписей
│
├── workflows/
│   └── generate_workflow.json       # ComfyUI-воркфлоу (FLUX + LoRA + KSampler)
│
├── output/                          # LoRA-чекпоинты полной тренировки
├── output_test/                     # LoRA-чекпоинты тестового прогона
└── vendor/
    └── ai-toolkit/                  # клонируется setup_training.py
```

---

## Настройка `.env`

Создай файл `.env` в корне проекта. Минимальный вариант:

```ini
# Путь к модели FLUX.1-dev — repo id или локальный путь к snapshot
HF_MODEL_PATH=black-forest-labs/FLUX.1-dev
# или локально:
# HF_MODEL_PATH=C:/Users/AdamCage/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/<revision>

TRIGGER_WORD=LoraTrigger

# Параметры полной тренировки
TRAINING_STEPS=10000
TRAINING_RANK=32
TRAINING_LR=1e-4
TRAINING_LR_RESTART_STEPS=1667
TRAINING_BATCH_SIZE=1
TRAINING_RESOLUTION_W=320
TRAINING_RESOLUTION_H=640
TRAINING_NUM_REPEATS=10

# Параметры тестового прогона
TEST_STEPS=50
TEST_RANK=4
TEST_TRAINING_LR=1e-3
TEST_RESOLUTION_W=256
TEST_RESOLUTION_H=256
TEST_NUM_REPEATS=5

# ComfyUI (для шага 4B)
COMFYUI_URL=http://127.0.0.1:8188
COMFYUI_FLUX_UNET=flux1-dev.safetensors
COMFYUI_CLIP_L=clip_l.safetensors
COMFYUI_T5=t5xxl_fp16.safetensors
COMFYUI_VAE=ae.safetensors
# COMFYUI_LORA=cp2077_tarot_lora-000001000.safetensors
```

Полное описание всех переменных — в разделе [Справочник параметров `.env`](#справочник-параметров-env).

---

## Шаг 1 — Установка окружения

```powershell
uv run setup_training.py
```

Скрипт выполнит:

1. Проверит наличие `git` и `uv`
2. Клонирует (или обновит) `ostris/ai-toolkit` в `vendor/ai-toolkit/`
3. Инициализирует git-сабмодули ai-toolkit
4. Установит весь ML-стек через `uv sync --extra training` (~15–20 мин, ~10 GB)
5. Установит зависимости ai-toolkit: `uv pip install -r vendor/ai-toolkit/requirements.txt`
6. Создаст папки `output/` и `output_test/`
7. Проверит корректность `HF_MODEL_PATH` в `.env`

---

## Шаг 2 — Генерация капшенов

```powershell
# Записать .txt-файл рядом с каждым изображением в data/originals/
uv run generate_captions.py

# Предпросмотр без записи файлов
uv run generate_captions.py --dry-run

# Записать в другую папку
uv run generate_captions.py --output-dir path/to/dir
```

Скрипт читает `data/originals/meta.jsonl` и для каждого изображения создаёт `<имя>.txt` с капшеном вида:

```
LoraTrigger, tarot card illustration, cyberpunk city vibe, ..., <concept из meta.jsonl>
```

---

## Шаг 3 — Тренировка LoRA

```powershell
# Тестовый прогон — убедиться, что пайплайн работает (~5–10 мин)
uv run train.py --mode test

# Полная тренировка (~1–2 ч на RTX 4080)
uv run train.py --mode full
```

Скрипт:
- читает `.env` и подставляет переменные в YAML-конфиг
- для `test` использует `config/train_test.yaml` (50 шагов, rank 4, 256×256)
- для `full` использует `config/train_full.yaml` (10 000 шагов, rank 32, 320×640)
- запускает `vendor/ai-toolkit/run.py` с подготовленным конфигом

Результаты:

| Путь | Содержимое |
|---|---|
| `output/cp2077_tarot_lora/` | LoRA-чекпоинты полной тренировки |
| `output_test/cp2077_tarot_lora/` | LoRA-чекпоинты тестового прогона |
| `output/.../samples/` | Превью-картинки, генерируемые каждые 250 шагов |

Чекпоинты сохраняются каждые 250 шагов, хранятся 4 последних. Для генерации обычно берут 2-й или 3-й из конца.

---

## Шаг 4A — Генерация карт локально (diffusers)

Вариант без ComfyUI — запускает FLUX + LoRA напрямую через библиотеку `diffusers`.

```powershell
# Авто-определить последний чекпоинт из output/, сгенерировать все карты
uv run generate_cards.py

# Указать LoRA явно
uv run generate_cards.py --lora output/cp2077_tarot_lora/cp2077_tarot_lora-000003000.safetensors

# Сгенерировать только несколько карт
uv run generate_cards.py --cards cups_01,swords_knight

# VRAM < 24 GB — CPU offload (медленнее, но работает от ~12 GB)
uv run generate_cards.py --cpu-offload

# Перезаписать уже существующие файлы
uv run generate_cards.py --overwrite

# Просмотр промптов без генерации
uv run generate_cards.py --dry-run
```

Результат: `data/generated/res/<card_id>.png`

---

## Шаг 4B — Генерация карт через ComfyUI

Более гибкий вариант: отправляет задачи в ComfyUI через REST API (`/prompt`).  
Поддерживает per-card гиперпараметры и режим benchmark.

### Подготовка

1. Запусти ComfyUI
2. Скопируй LoRA-чекпоинт в `ComfyUI/models/loras/`
3. Убедись, что `COMFYUI_*` переменные в `.env` указывают на нужные модели

### Базовый запуск

```powershell
# Минимальный запуск — указать имя LoRA-файла (как он лежит в ComfyUI/models/loras/)
uv run generate_cards_comfyui.py --lora cp2077_tarot_lora-000003000.safetensors

# Указать все параметры явно
uv run generate_cards_comfyui.py `
  --lora   cp2077_tarot_lora-000003000.safetensors `
  --url    http://127.0.0.1:8188 `
  --steps  30 `
  --guidance 4.0 `
  --lora-scale 0.85 `
  --width  320 --height 640 `
  --seed   42

# Только несколько карт
uv run generate_cards_comfyui.py --lora cp2077... --cards cups_01,wands_03,pentacles_queen

# Случайный seed для каждой карты
uv run generate_cards_comfyui.py --lora cp2077... --random-seed

# Перезаписать уже существующие
uv run generate_cards_comfyui.py --lora cp2077... --overwrite

# Просмотр промптов без запуска генерации
uv run generate_cards_comfyui.py --dry-run
```

Результат: `data/generated/res/<card_id>.png`

#### Как работает workflow

Скрипт загружает `workflows/generate_workflow.json` и патчит следующие ноды:

| Нода ComfyUI | Что патчится |
|---|---|
| `LoraLoader` | имя LoRA, `strength_model`, `strength_clip` |
| `CLIPTextEncodeFlux` | `clip_l` (стиль + теги), `t5xxl` (описание сцены), `guidance` |
| `EmptySD3LatentImage` | `width`, `height` |
| `KSampler` | `seed`, `steps`, `sampler_name`, `scheduler` |
| `SaveImage` | `filename_prefix` |

CLIP-L получает стилевые теги + триггер-слово, T5-XXL — развёрнутое описание сцены из `data/missing/meta.jsonl`.

---

### Режим hypermeta

Позволяет задать **индивидуальные гиперпараметры** для каждой карты — seed, guidance, lora_scale и steps — через файл `data/missing/_hyper_meta.jsonl`.

```powershell
uv run generate_cards_comfyui.py --lora cp2077... --hypermeta
```

Формат строки в `_hyper_meta.jsonl`:

```json
{"card_id": "cups_01", "seed": 100, "guidance": 4.5, "lora_scale": 0.85, "steps": 30}
```

Все поля опциональны, кроме `card_id`. Если поле не задано — берётся значение из CLI-аргумента (или дефолт).

Сценарий использования: после benchmark-прогона выбрать лучший seed/параметры для каждой карты, записать в `_hyper_meta.jsonl`, затем запустить финальную генерацию с `--hypermeta`.

---

### Режим benchmark

Режим автоматического подбора гиперпараметров. Запускает серию генераций с разными seed, guidance или lora_scale и сохраняет результаты + коллаж-сетку для визуального сравнения.

#### Режим `seed` — 10 прогонов, стандартный набор seed

```powershell
uv run generate_cards_comfyui.py `
  --lora cp2077... `
  --cards cups_01,cups_02 `
  --benchmark seed `
  --steps 28 `
  --guidance 3.5 `
  --lora-scale 1.0
```

Прогоняет 10 стандартных seed: `42, 100, 200, 300, 400, 500, 777, 1000, 1337, 9999`.  
Результат: `data/generated/benchmark/seed/rm_<seed>_st<steps>/<card_id>.png`  
Коллаж: `data/generated/benchmark/seed/<card_id>_benchmark.png` — сетка 5×2

#### Режим `seeds` — произвольный список seed

```powershell
uv run generate_cards_comfyui.py `
  --lora cp2077... `
  --cards cups_03 `
  --benchmark seeds `
  --benchmark-seeds 42,777,1234,5678,9999 `
  --benchmark-lora 0.85 `
  --steps 28 `
  --guidance 4.0
```

Аргументы:
- `--benchmark-seeds` — список seed через запятую
- `--benchmark-lora` — фиксированный `lora_scale` для этого прогона (по умолчанию берётся `--lora-scale`)

Результат: `data/generated/benchmark/seeds/rm_<seed>_st<steps>_l<lora>/<card_id>.png`  
Коллаж: `data/generated/benchmark/seeds/<card_id>_benchmark.png` — динамическая сетка (~5 колонок)

#### Режим `full` — 30 прогонов: seed + guidance + lora_scale

```powershell
uv run generate_cards_comfyui.py `
  --lora cp2077... `
  --cards cups_01 `
  --benchmark full `
  --steps 28
```

Три блока:

| Блок | Переменная | Фиксировано | Прогонов |
|---|---|---|---|
| A — seed | 42, 100, 200, 300, 400, 500, 777, 1000, 1337, 9999 | guidance, lora_scale | 10 |
| B — guidance | 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0 | seed=42, lora_scale | 10 |
| C — lora_scale | 0.6, 0.75, 0.85, 1.0, 1.15, 1.5, 2.0, 2.5, 3.0, 3.5 | seed=42, guidance=3.5 | 10 |

Результат: `data/generated/benchmark/full/...`  
Коллаж: сетка 10×3 с подписями `[A: seed]`, `[B: guidance]`, `[C: lora_scale]`

#### Пропуск уже готовых

По умолчанию уже существующие файлы пропускаются. Добавь `--overwrite` чтобы перезаписать:

```powershell
uv run generate_cards_comfyui.py --lora cp2077... --benchmark seed --cards cups_01 --overwrite
```

---

## Шаг 5 — Добавление рамок и подписей

После генерации добавляет декоративную киберпанк-рамку и вертикальное название карты на русском языке.

```powershell
# Обработать все карты из data/generated/res/ и data/originals/
uv run add_card_borders.py

# Только определённые карты
uv run add_card_borders.py --cards cups_01 cups_02 swords_knight

# Указать другие папки
uv run add_card_borders.py `
  --res-dir      data/generated/res `
  --originals-dir data/originals `
  --output-dir   data/generated/final

# Предпросмотр без генерации
uv run add_card_borders.py --dry-run
```

Результат: `data/generated/final/<card_id>.png`

#### Что добавляется на карту

```
┌──────────┬──────────────────────────┐
│ штрихкод │                          │
│──────────│        арт карты         │
│          │                          │
│ название │                          │
│ (верт.)  │                          │
└──────────┴──────────────────────────┘
```

- **Левая колонка (148 px)** — тёмный фон, цвет зависит от масти
- **Верхняя 1/3 колонки** — штрихкод Code 128B с названием карты (повёрнут вертикально)
- **Нижние 2/3 колонки** — вертикальное название карты на русском (шрифт Russo One)
- **Акцентная линия** — вертикальная полоса между колонкой и артом
- **Угловые маркеры** — Г-образные маркеры на углах арта и колонки
- **Глитч-разделитель** — сегментированная линия между штрихкодом и названием

#### Цветовые схемы по масти

| Масть | Акцент | Свечение |
|---|---|---|
| Кубки (cups) | Маджента | Тёмная маджента |
| Мечи (swords) | Фиолетовый | Жёлтый |
| Жезлы (wands) | Красный | Тёмно-красный |
| Пентакли (pentacles) | Золотой | Бирюзовый |
| Старшие арканы (major) | Золотой | Тёмно-золотой |

#### Источники для обработки

Скрипт объединяет два источника:
- `data/generated/res/*.png` — сгенерированные карты Младших Арканов
- `data/originals/*.webp` — оригинальные арты (Старшие Арканы + четыре Короля)

Метаданные (русское название, масть) берутся из `data/missing/meta.jsonl` и `data/originals/meta.jsonl`. Если карта не найдена в метаданных — название выводится автоматически из `card_id` (например, `cups_07` → «Семёрка Кубков»).

---

## Справочник параметров `.env`

| Переменная | Описание |
|---|---|
| `HF_MODEL_PATH` | Путь к FLUX.1-dev: repo id (`black-forest-labs/FLUX.1-dev`) или локальный путь к snapshot |
| `TRIGGER_WORD` | Триггер-слово LoRA (по умолчанию `LoraTrigger`) |
| `TRAINING_STEPS` | Число шагов полной тренировки |
| `TRAINING_RANK` | Rank LoRA |
| `TRAINING_LR` | Learning rate |
| `TRAINING_LR_RESTART_STEPS` | Шаги до рестарта cosine scheduler |
| `TRAINING_BATCH_SIZE` | Размер батча |
| `TRAINING_RESOLUTION_W` / `H` | Разрешение обучения (ширина × высота) |
| `TRAINING_NUM_REPEATS` | Число повторений датасета |
| `TEST_STEPS` | Шаги тестового прогона |
| `TEST_RANK` | Rank LoRA для теста |
| `TEST_TRAINING_LR` | Learning rate для теста |
| `TEST_RESOLUTION_W` / `H` | Разрешение тестового прогона |
| `TEST_NUM_REPEATS` | Повторения датасета для теста |
| `COMFYUI_URL` | URL ComfyUI (по умолчанию `http://127.0.0.1:8188`) |
| `COMFYUI_FLUX_UNET` | Имя файла FLUX UNet в `ComfyUI/models/unet/` |
| `COMFYUI_CLIP_L` | Имя файла CLIP-L в `ComfyUI/models/clip/` |
| `COMFYUI_T5` | Имя файла T5-XXL в `ComfyUI/models/clip/` |
| `COMFYUI_VAE` | Имя файла VAE в `ComfyUI/models/vae/` |
| `COMFYUI_LORA` | Имя LoRA-файла в `ComfyUI/models/loras/` (опционально) |
| `COMFYUI_WORKFLOW` | Путь к workflow JSON (по умолчанию `workflows/generate_workflow.json`) |

---

## Справочник CLI-аргументов

### `generate_captions.py`

| Аргумент | Описание |
|---|---|
| `--dry-run` | Показать капшены без записи файлов |
| `--output-dir PATH` | Куда записывать `.txt` (по умолчанию рядом с изображениями) |

### `train.py`

| Аргумент | Описание |
|---|---|
| `--mode test` | Тестовый прогон (50 шагов, rank 4, 256×256) |
| `--mode full` | Полная тренировка (параметры из `.env`) |

### `generate_cards.py` (diffusers)

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `--lora PATH` | авто | Путь к `.safetensors` (авто-поиск в `output/`) |
| `--output-dir PATH` | `data/generated/res` | Папка для PNG |
| `--steps N` | 28 | Шаги inference |
| `--guidance F` | 3.5 | Guidance scale |
| `--width N` | из `.env` | Ширина |
| `--height N` | из `.env` | Высота |
| `--lora-scale F` | 1.0 | Сила LoRA |
| `--seed N` | 42 | Random seed |
| `--cards LIST` | все | `card_id` через запятую |
| `--cpu-offload` | выкл | CPU offload (~12 GB VRAM, медленнее) |
| `--overwrite` | выкл | Перезаписать существующие файлы |
| `--dry-run` | выкл | Показать промпты без генерации |

### `generate_cards_comfyui.py`

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `--url URL` | `COMFYUI_URL` из `.env` | URL ComfyUI |
| `--workflow PATH` | `workflows/generate_workflow.json` | Путь к workflow JSON |
| `--lora NAME` | `COMFYUI_LORA` из `.env` | Имя LoRA в `models/loras/` |
| `--lora-scale F` | 1.0 | Сила LoRA |
| `--steps N` | 28 | Шаги inference |
| `--guidance F` | 3.5 | Guidance scale |
| `--width N` | из `.env` | Ширина |
| `--height N` | из `.env` | Высота |
| `--seed N` | 42 | Seed |
| `--random-seed` | выкл | Случайный seed для каждой карты |
| `--cards LIST` | все | `card_id` через запятую |
| `--output-dir PATH` | `data/generated/res` | Папка для PNG |
| `--overwrite` | выкл | Перезаписать существующие файлы |
| `--timeout N` | 600 | Таймаут ожидания от ComfyUI (сек) |
| `--dry-run` | выкл | Показать промпты без генерации |
| `--hypermeta` | выкл | Загружать per-card параметры из `_hyper_meta.jsonl` |
| `--benchmark MODE` | — | Режим sweep: `seed`, `seeds` или `full` |
| `--benchmark-seeds LIST` | — | Список seed для `--benchmark seeds` |
| `--benchmark-lora F` | `--lora-scale` | `lora_scale` для `--benchmark seeds` |

### `add_card_borders.py`

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `--res-dir PATH` | `data/generated/res` | Папка с сырыми PNG |
| `--originals-dir PATH` | `data/originals` | Папка с оригинальными WEBP |
| `--output-dir PATH` | `data/generated/final` | Папка для результатов |
| `--cards LIST` | все | Список `card_id` (stem файла) через пробел |
| `--dry-run` | выкл | Показать список карт без обработки |
