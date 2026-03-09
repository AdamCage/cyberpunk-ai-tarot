# Cyberpunk AI Tarot

![](data/result/all_cards.png)

Generating the missing 56 Minor Arcana cards for the Cyberpunk 2077 tarot deck by fine-tuning FLUX.1-dev with LoRA, trained on the 26 original in-game artworks.

The pipeline has five steps: caption generation → LoRA training → card generation → border & title overlay → full-deck collage.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Project Structure](#project-structure)
- [Step-by-Step Reproduction Guide](#step-by-step-reproduction-guide)
  - [0. Configure `.env`](#0-configure-env)
  - [1. Install the Environment](#1-install-the-environment)
  - [2. Generate Training Captions](#2-generate-training-captions)
  - [3. Train the LoRA](#3-train-the-lora)
  - [4A. Generate Cards Locally (diffusers)](#4a-generate-cards-locally-diffusers)
  - [4B. Generate Cards via ComfyUI](#4b-generate-cards-via-comfyui)
  - [5. Add Borders and Titles](#5-add-borders-and-titles)
- [ComfyUI — Advanced Usage](#comfyui--advanced-usage)
  - [How the Workflow is Patched](#how-the-workflow-is-patched)
  - [Hypermeta Mode](#hypermeta-mode)
  - [Benchmark Mode](#benchmark-mode)
- [`.env` Reference](#env-reference)
- [CLI Reference](#cli-reference)

---

## System Requirements

| Component | Requirement |
|---|---|
| **Python** | 3.11 or higher |
| **Package manager** | [`uv`](https://docs.astral.sh/uv/) |
| **GPU** | CUDA-capable, **≥ 16 GB VRAM** (RTX 3090 / 4080 or better recommended) |
| **Disk space** | ~30 GB (FLUX model ~25 GB + ML stack ~5 GB + outputs) |
| **OS** | Windows, Linux or macOS with CUDA drivers |

**Required models** (downloaded separately before training):

```powershell
# Download FLUX.1-dev from HuggingFace (~25 GB)
uv run huggingface-cli download black-forest-labs/FLUX.1-dev
```

**For ComfyUI generation only (Step 4B):**
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running locally or remotely
- FLUX.1-dev components placed in ComfyUI model folders:
  - `ComfyUI/models/unet/flux1-dev.safetensors`
  - `ComfyUI/models/clip/clip_l.safetensors`
  - `ComfyUI/models/clip/t5xxl_fp16.safetensors`
  - `ComfyUI/models/vae/ae.safetensors`

---

## Project Structure

```
cyberpunk-ai-tarot/
│
├── .env                             # model paths and training params (not in git)
├── pyproject.toml                   # project dependencies (managed by uv)
│
├── generate_captions.py             # step 1: generate .txt training captions
├── setup_training.py                # step 2: install the ML environment
├── train.py                         # step 3: run LoRA training via ai-toolkit
├── generate_cards.py                # step 4A: generate cards locally via diffusers
├── generate_cards_comfyui.py        # step 4B: generate cards via ComfyUI API
├── add_card_borders.py              # step 5: add cyberpunk frame + card title
│
├── config/
│   ├── train_full.yaml              # full training config template (10 000 steps)
│   └── train_test.yaml              # test run config template (50 steps)
│
├── data/
│   ├── originals/                   # 26 original card arts (.webp) + .txt captions
│   ├── missing/
│   │   ├── meta.jsonl               # metadata and prompts for 53 missing cards
│   │   ├── meta_v2–v7.jsonl         # prompt revision history
│   │   └── _hyper_meta.jsonl        # per-card inference hyperparameters
│   ├── style_bible/
│   │   └── style_bible.json         # canonical style rules for the art style
│   ├── generated/
│   │   ├── res/                     # raw generated PNGs (no border)
│   │   ├── final/                   # final PNGs with border and title
│   │   └── benchmark/
│   │       ├── seed/                # 10-seed sweep results
│   │       └── seeds/               # custom seed sweep results
│   └── RussoOne-Regular.ttf         # font used for card titles
│
├── workflows/
│   └── generate_workflow.json       # ComfyUI node graph (FLUX + LoRA + KSampler)
│
├── output/                          # LoRA checkpoints from full training
├── output_test/                     # LoRA checkpoints from test run
└── vendor/
    └── ai-toolkit/                  # cloned by setup_training.py
```

---

## Step-by-Step Reproduction Guide

> Run all commands from the project root. The full pipeline takes about 2–3 hours on a modern GPU (mostly training time).

### 0. Configure `.env`

Create a `.env` file in the project root. Copy the template below and fill in your paths:

```ini
# FLUX.1-dev — HuggingFace repo id or local snapshot path
HF_MODEL_PATH=black-forest-labs/FLUX.1-dev
# Local example (after hf download):
# HF_MODEL_PATH=C:/Users/you/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/<revision>

TRIGGER_WORD=LoraTrigger

# ── Full training params ────────────────────────────────────────────────────
TRAINING_STEPS=10000
TRAINING_RANK=32
TRAINING_LR=1e-4
TRAINING_LR_RESTART_STEPS=1667
TRAINING_BATCH_SIZE=1
TRAINING_RESOLUTION_W=320
TRAINING_RESOLUTION_H=640
TRAINING_NUM_REPEATS=10

# ── Test run params ─────────────────────────────────────────────────────────
TEST_STEPS=50
TEST_RANK=4
TEST_TRAINING_LR=1e-3
TEST_RESOLUTION_W=256
TEST_RESOLUTION_H=256
TEST_NUM_REPEATS=5

# ── ComfyUI (Step 4B only) ───────────────────────────────────────────────────
COMFYUI_URL=http://127.0.0.1:8188
COMFYUI_FLUX_UNET=flux1-dev.safetensors
COMFYUI_CLIP_L=clip_l.safetensors
COMFYUI_T5=t5xxl_fp16.safetensors
COMFYUI_VAE=ae.safetensors
# COMFYUI_LORA=cp2077_tarot_lora-000010000.safetensors
```

Full variable reference: [`.env` Reference](#env-reference).

---

### 1. Install the Environment

```powershell
uv run setup_training.py
```

This script:

1. Verifies `git` and `uv` are installed
2. Clones (or pulls) `ostris/ai-toolkit` into `vendor/ai-toolkit/`
3. Initializes ai-toolkit git submodules
4. Installs the full ML stack via `uv sync --extra training` — **~15–20 min, ~10 GB**
5. Installs ai-toolkit requirements: `uv pip install -r vendor/ai-toolkit/requirements.txt`
6. Creates `output/` and `output_test/` directories
7. Validates `HF_MODEL_PATH` in `.env`

---

### 2. Generate Training Captions

```powershell
# Write a .txt caption file next to each image in data/originals/
uv run generate_captions.py

# Preview captions without writing files
uv run generate_captions.py --dry-run
```

Reads `data/originals/meta.jsonl` and creates `<name>.txt` alongside each image with a caption in the format:

```
LoraTrigger, tarot card illustration, cyberpunk city vibe, ..., <concept from meta.jsonl>
```

---

### 3. Train the LoRA

> **Recommended:** run the test first to confirm the pipeline works end-to-end, then run the full training.

```powershell
# Test run — verifies the pipeline (~5–10 min)
uv run train.py --mode test

# Full training (~1–2 h on RTX 4080)
uv run train.py --mode full
```

What it does:
- Reads `.env` and substitutes variables into the YAML config
- `test` → `config/train_test.yaml` (50 steps, rank 4, 256×256)
- `full` → `config/train_full.yaml` (10 000 steps, rank 32, 320×640, cosine LR with restarts)
- Launches `vendor/ai-toolkit/run.py` with the resolved config

Output:

| Path | Contents |
|---|---|
| `output/cp2077_tarot_lora/` | LoRA checkpoints (`.safetensors`) |
| `output_test/cp2077_tarot_lora/` | Test run checkpoints |
| `output/.../samples/` | Preview images generated every 250 steps |

Checkpoints are saved every 250 steps; the last 4 are kept. The 2nd or 3rd from the end usually gives the best results.

---

### 4A. Generate Cards Locally (diffusers)

No ComfyUI needed — runs FLUX.1-dev + LoRA directly via `diffusers`.

```powershell
# Auto-detect the latest checkpoint from output/ and generate all cards
uv run generate_cards.py

# Specify a checkpoint explicitly
uv run generate_cards.py --lora output/cp2077_tarot_lora/cp2077_tarot_lora-000010000.safetensors

# Generate only specific cards
uv run generate_cards.py --cards cups_01,swords_knight

# VRAM < 24 GB — enable CPU offload (~12 GB VRAM, slower)
uv run generate_cards.py --cpu-offload

# Overwrite already generated files
uv run generate_cards.py --overwrite

# Preview prompts without running generation
uv run generate_cards.py --dry-run
```

Output: `data/generated/res/<card_id>.png`

---

### 4B. Generate Cards via ComfyUI

More flexible option: submits jobs to a running ComfyUI instance via its REST API (`/prompt`).
Supports per-card hyperparameters and a full benchmark sweep mode.

**Before running:**
1. Start ComfyUI
2. Copy the LoRA checkpoint to `ComfyUI/models/loras/`
3. Make sure `COMFYUI_*` variables in `.env` point to your model files

```powershell
# Minimal run — specify the LoRA filename as it sits in ComfyUI/models/loras/
uv run generate_cards_comfyui.py --lora cp2077_tarot_lora-000010000.safetensors

# With explicit parameters
uv run generate_cards_comfyui.py `
  --lora       cp2077_tarot_lora-000010000.safetensors `
  --steps      30 `
  --guidance   4.0 `
  --lora-scale 0.85 `
  --width      320 `
  --height     640 `
  --seed       42

# Generate only specific cards
uv run generate_cards_comfyui.py --lora cp2077... --cards cups_01,wands_03,pentacles_queen

# Random seed for each card
uv run generate_cards_comfyui.py --lora cp2077... --random-seed

# Overwrite existing files
uv run generate_cards_comfyui.py --lora cp2077... --overwrite

# Preview prompts without generation
uv run generate_cards_comfyui.py --dry-run
```

Output: `data/generated/res/<card_id>.png`

---

### 5. Add Borders and Titles

Adds a cyberpunk-themed decorative frame and a vertical Russian card title to every generated image. Also produces a full-deck collage (`all_cards.png`).

```powershell
# Process all cards from data/generated/res/ and data/originals/
uv run add_card_borders.py

# Only specific cards
uv run add_card_borders.py --cards cups_01 cups_02 swords_knight

# Skip collage generation
uv run add_card_borders.py --no-collage

# Custom column count for the collage (default: 14)
uv run add_card_borders.py --collage-cols 12

# Preview without processing
uv run add_card_borders.py --dry-run
```

Output:
- `data/generated/final/<card_id>.png` — individual framed cards
- `data/generated/final/all_cards.png` — 8K collage (~7700×7400 px) of the full deck

> `card_back` is copied to `final/` as-is without any border added.

**What gets added to each card:**

```
┌──────────┬──────────────────────────┐
│ barcode  │                          │
│──────────│        card art          │
│          │                          │
│  title   │                          │
│ (vert.)  │                          │
└──────────┴──────────────────────────┘
```

- **Left sidebar (148 px)** — dark background, color depends on suit
- **Top ⅓ of sidebar** — Code 128B barcode encoding the card title (rotated vertically)
- **Bottom ⅔ of sidebar** — vertical card title in Russian (Russo One font, auto-sized)
- **Accent line** — vertical stripe between sidebar and art
- **Corner markers** — L-shaped markers at art corners
- **Glitch separator** — segmented line between barcode and title zones

**Color schemes by suit:**

| Suit | Accent | Glow |
|---|---|---|
| Cups | Magenta | Dark magenta |
| Swords | Violet | Yellow |
| Wands | Red | Dark red |
| Pentacles | Gold | Teal |
| Major Arcana | Gold | Dark gold |

**Input sources:**
- `data/generated/res/*.png` — generated Minor Arcana
- `data/originals/*.webp` — original Major Arcana + four Kings

Metadata (Russian title, suit) is read from `data/missing/meta.jsonl` and `data/originals/meta.jsonl`. If a card is not found in metadata, the title is inferred from `card_id` (e.g. `cups_07` → "Семёрка Кубков").

---

## ComfyUI — Advanced Usage

### How the Workflow is Patched

The script loads `workflows/generate_workflow.json` and patches these nodes for each card:

| ComfyUI node | Patched fields |
|---|---|
| `LoraLoader` | `lora_name`, `strength_model`, `strength_clip` |
| `CLIPTextEncodeFlux` | `clip_l` (style tags + trigger), `t5xxl` (scene description), `guidance` |
| `EmptySD3LatentImage` | `width`, `height` |
| `KSampler` | `seed`, `steps`, `sampler_name`, `scheduler` |
| `SaveImage` | `filename_prefix` |

CLIP-L receives style tags and the trigger word. T5-XXL receives the full scene description from `data/missing/meta.jsonl`. This dual-prompt approach is native to FLUX and gives better results than a single combined prompt.

---

### Hypermeta Mode

Allows setting **per-card hyperparameters** — seed, guidance, lora_scale, steps — via `data/missing/_hyper_meta.jsonl`.

```powershell
uv run generate_cards_comfyui.py --lora cp2077... --hypermeta
```

Each line in `_hyper_meta.jsonl`:

```json
{"card_id": "cups_01", "seed": 100, "guidance": 4.5, "lora_scale": 0.85, "steps": 30}
```

All fields except `card_id` are optional. Missing fields fall back to CLI arguments.

**Typical workflow:** run benchmark → visually pick the best parameters for each card → write them to `_hyper_meta.jsonl` → run final generation with `--hypermeta`.

---

### Benchmark Mode

Automatic hyperparameter sweep. Runs a series of generations with varying seed, guidance, or lora_scale and saves results alongside a visual comparison grid.

#### Mode `seed` — 10 runs, standard seed set

```powershell
uv run generate_cards_comfyui.py `
  --lora       cp2077... `
  --cards      cups_01,cups_02 `
  --benchmark  seed `
  --steps      28 `
  --guidance   3.5 `
  --lora-scale 1.0
```

Runs 10 fixed seeds: `42, 100, 200, 300, 400, 500, 777, 1000, 1337, 9999`.

Output: `data/generated/benchmark/seed/rm_<seed>_st<steps>/<card_id>.png`  
Collage: `data/generated/benchmark/seed/<card_id>_benchmark.png` — 5×2 grid

#### Mode `seeds` — custom seed list

```powershell
uv run generate_cards_comfyui.py `
  --lora            cp2077... `
  --cards           cups_03 `
  --benchmark       seeds `
  --benchmark-seeds 42,777,1234,5678,9999 `
  --benchmark-lora  0.85 `
  --steps           28 `
  --guidance        4.0
```

- `--benchmark-seeds` — comma-separated list of seeds
- `--benchmark-lora` — fixed `lora_scale` for the run (default: `--lora-scale`)

Output: `data/generated/benchmark/seeds/rm_<seed>_st<steps>_l<lora>/<card_id>.png`  
Collage: dynamic grid (~5 columns)

#### Mode `full` — 30 runs: seed + guidance + lora_scale

```powershell
uv run generate_cards_comfyui.py `
  --lora      cp2077... `
  --cards     cups_01 `
  --benchmark full `
  --steps     28
```

Three blocks:

| Block | Variable | Fixed | Runs |
|---|---|---|---|
| A — seed | 42, 100, 200, 300, 400, 500, 777, 1000, 1337, 9999 | guidance, lora_scale | 10 |
| B — guidance | 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0 | seed=42, lora_scale | 10 |
| C — lora_scale | 0.6, 0.75, 0.85, 1.0, 1.15, 1.5, 2.0, 2.5, 3.0, 3.5 | seed=42, guidance=3.5 | 10 |

Output: `data/generated/benchmark/full/...`  
Collage: 10×3 grid with labels `[A: seed]`, `[B: guidance]`, `[C: lora_scale]`

By default existing files are skipped. Add `--overwrite` to regenerate:

```powershell
uv run generate_cards_comfyui.py --lora cp2077... --benchmark seed --cards cups_01 --overwrite
```

---

## `.env` Reference

| Variable | Description |
|---|---|
| `HF_MODEL_PATH` | FLUX.1-dev path: repo id (`black-forest-labs/FLUX.1-dev`) or local snapshot path |
| `TRIGGER_WORD` | LoRA trigger word (default: `LoraTrigger`) |
| `TRAINING_STEPS` | Number of steps for full training |
| `TRAINING_RANK` | LoRA rank |
| `TRAINING_LR` | Learning rate |
| `TRAINING_LR_RESTART_STEPS` | Cosine LR scheduler restart interval |
| `TRAINING_BATCH_SIZE` | Training batch size |
| `TRAINING_RESOLUTION_W` / `H` | Training image resolution (width × height) |
| `TRAINING_NUM_REPEATS` | Dataset repeat count per epoch |
| `TEST_STEPS` | Steps for test run |
| `TEST_RANK` | LoRA rank for test run |
| `TEST_TRAINING_LR` | Learning rate for test run |
| `TEST_RESOLUTION_W` / `H` | Resolution for test run |
| `TEST_NUM_REPEATS` | Dataset repeats for test run |
| `COMFYUI_URL` | ComfyUI URL (default: `http://127.0.0.1:8188`) |
| `COMFYUI_FLUX_UNET` | FLUX UNet filename in `ComfyUI/models/unet/` |
| `COMFYUI_CLIP_L` | CLIP-L filename in `ComfyUI/models/clip/` |
| `COMFYUI_T5` | T5-XXL filename in `ComfyUI/models/clip/` |
| `COMFYUI_VAE` | VAE filename in `ComfyUI/models/vae/` |
| `COMFYUI_LORA` | LoRA filename in `ComfyUI/models/loras/` (optional) |
| `COMFYUI_WORKFLOW` | Path to workflow JSON (default: `workflows/generate_workflow.json`) |

---

## CLI Reference

### `generate_captions.py`

| Argument | Description |
|---|---|
| `--dry-run` | Print captions without writing files |
| `--output-dir PATH` | Write `.txt` files here (default: next to images) |

### `train.py`

| Argument | Description |
|---|---|
| `--mode test` | Test run (50 steps, rank 4, 256×256) |
| `--mode full` | Full training (parameters from `.env`) |

### `generate_cards.py` (diffusers)

| Argument | Default | Description |
|---|---|---|
| `--lora PATH` | auto | Path to `.safetensors` (auto-detected from `output/`) |
| `--output-dir PATH` | `data/generated/res` | Output folder |
| `--steps N` | 28 | Inference steps |
| `--guidance F` | 3.5 | Guidance scale |
| `--width N` | from `.env` | Image width |
| `--height N` | from `.env` | Image height |
| `--lora-scale F` | 1.0 | LoRA strength |
| `--seed N` | 42 | Random seed |
| `--cards LIST` | all | Comma-separated `card_id` filter |
| `--cpu-offload` | off | CPU offload (~12 GB VRAM, slower) |
| `--overwrite` | off | Overwrite existing files |
| `--dry-run` | off | Print prompts without generating |

### `generate_cards_comfyui.py`

| Argument | Default | Description |
|---|---|---|
| `--url URL` | `COMFYUI_URL` from `.env` | ComfyUI URL |
| `--workflow PATH` | `workflows/generate_workflow.json` | Workflow JSON path |
| `--lora NAME` | `COMFYUI_LORA` from `.env` | LoRA filename in `models/loras/` |
| `--lora-scale F` | 1.0 | LoRA strength |
| `--steps N` | 28 | Inference steps |
| `--guidance F` | 3.5 | Guidance scale |
| `--width N` | from `.env` | Image width |
| `--height N` | from `.env` | Image height |
| `--seed N` | 42 | Seed |
| `--random-seed` | off | Random seed per card |
| `--cards LIST` | all | Comma-separated `card_id` filter |
| `--output-dir PATH` | `data/generated/res` | Output folder |
| `--overwrite` | off | Overwrite existing files |
| `--timeout N` | 600 | ComfyUI response timeout in seconds |
| `--dry-run` | off | Print prompts without generating |
| `--hypermeta` | off | Load per-card params from `_hyper_meta.jsonl` |
| `--benchmark MODE` | — | Sweep mode: `seed`, `seeds`, or `full` |
| `--benchmark-seeds LIST` | — | Comma-separated seeds for `--benchmark seeds` |
| `--benchmark-lora F` | `--lora-scale` | Fixed `lora_scale` for `--benchmark seeds` |

### `add_card_borders.py`

| Argument | Default | Description |
|---|---|---|
| `--res-dir PATH` | `data/generated/res` | Raw generated PNGs |
| `--originals-dir PATH` | `data/originals` | Original WEBP artworks |
| `--output-dir PATH` | `data/generated/final` | Output folder |
| `--cards LIST` | all | Space-separated `card_id` list (file stems) |
| `--no-collage` | off | Skip generating `all_cards.png` |
| `--collage-cols N` | 14 | Number of columns in `all_cards.png` |
| `--dry-run` | off | List cards without processing |
