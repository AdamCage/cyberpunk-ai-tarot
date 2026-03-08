# /// script
# requires-python = ">=3.11"
# dependencies = ["python-dotenv", "requests", "matplotlib", "Pillow"]
# ///
"""
Генерирует недостающие карты таро через ComfyUI API (FLUX.1-dev + LoRA).

Использует workflow-шаблон flux_dev_full_text_to_image_with_lora.json,
патчит промпт, LoRA, разрешение и seed для каждой карты,
и отправляет задачу в ComfyUI через /prompt API.

Перед запуском:
  1. Запусти ComfyUI
  2. Скопируй LoRA из output/ в ComfyUI/models/loras/
  3. Укажи имя LoRA через --lora (или задай COMFYUI_LORA в .env)

Запуск:
    uv run generate_cards_comfyui.py --lora cp2077_tarot_lora_000003000.safetensors
    uv run generate_cards_comfyui.py --lora cp2077... --cards cups_01,wands_03
    uv run generate_cards_comfyui.py --workflow path/to/other.json --lora ...
    uv run generate_cards_comfyui.py --dry-run   # показать промпты без генерации

Результат сохраняется в data/generated/<card_id>.png
"""

import argparse
import copy
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
MISSING_META = ROOT / "data" / "missing" / "meta.jsonl"
HYPER_META = ROOT / "data" / "missing" / "hyper_meta.jsonl"
OUTPUT_DIR = ROOT / "data" / "generated"
BENCHMARK_DIR = ROOT / "data" / "generated" / "benchmark"
WORKFLOW_TEMPLATE = ROOT / "workflows" / "generate_workflow.json"

# Стилевые теги — соответствуют STYLE_PREFIX в generate_captions.py
# (core_rendering + composition_rules + palette_rules + glitch_rules из style_bible.json)
STYLE_PREFIX = (
    # core_rendering
    "tarot card illustration, cyberpunk city vibe, graphic poster style, "
    "matte ink gouache look, flat posterized shapes, thick black brush outlines, "
    "messy dry-brush fills, visible paper grain, paint splatter, screenprint texture, "
    "slight color misregistration, "
    # composition_rules
    "full-bleed illustration, one dominant silhouette, "
    "big dark masses, limited bright accents, "
    # palette_rules
    "limited color palette, muted neon paint, hard edge color transitions, "
    # glitch_rules
    "print artifacts, ink bleed, offset print layers, "
    # базовые
    "high contrast, heavy blacks, worn paint edges"
)

# ──────────────────────────────────────────────────────────────────────────────


def load_meta(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[ОШИБКА] Не найден: {path}")
        sys.exit(1)
    cards = []
    with path.open(encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cards.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [!] Ошибка парсинга строки {lineno}: {e}")
    return cards


def load_hyper_meta(path: Path) -> dict[str, dict]:
    """
    Загружает hyper_meta.jsonl — гиперпараметры для каждой карты.

    Формат строки:
      {"card_id": "cups_01", "seed": 100, "guidance": 4.5, "lora_scale": 0.85, "steps": 30}

    Поддерживаемые ключи (все опциональны, кроме card_id):
      seed        — фиксированный seed (игнорируется если --random-seed)
      guidance    — guidance scale
      lora_scale  — сила LoRA
      steps       — число шагов inference

    Возвращает dict: card_id -> {параметры}.
    Если файл не найден — возвращает пустой dict с предупреждением.
    """
    if not path.exists():
        print(f"[!] hyper_meta не найден: {path}  (используются параметры CLI)")
        return {}
    result: dict[str, dict] = {}
    with path.open(encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                card_id = entry.get("card_id")
                if card_id:
                    result[card_id] = entry
            except json.JSONDecodeError as e:
                print(f"  [!] hyper_meta строка {lineno}: {e}")
    return result


def build_clip_l_prompt(entry: dict, trigger: str) -> str:
    """Ключевые слова и стиль для CLIP-L: триггер + STYLE_PREFIX + теги из meta.jsonl."""
    tags = ", ".join(entry.get("tags", []))
    parts = [trigger, STYLE_PREFIX, tags]
    return ", ".join(p for p in parts if p)


def build_t5xxl_prompt(entry: dict) -> str:
    """Развёрнутое описание сцены/концепта для T5-XXL (только concept)."""
    return entry.get("concept", "").strip().rstrip(".")


def snap_to(value: int, multiple: int = 16) -> int:
    return max(multiple, round(value / multiple) * multiple)


def apply_workflow_template(
    template_path: Path,
    *,
    lora_name: str | None,
    lora_scale: float,
    clip_l_prompt: str,
    t5xxl_prompt: str,
    guidance: float,
    width: int,
    height: int,
    steps: int,
    seed: int,
    filename_prefix: str,
    sampler_name: str | None = None,
    scheduler: str | None = None,
) -> dict:
    """
    Загружает workflow-шаблон ComfyUI, патчит параметры карты и возвращает
    граф в API-формате ({node_id: {class_type, inputs}}).

    Патчатся следующие ноды (по type):
      - LoraLoader: lora_name, strength_model, strength_clip
      - CLIPTextEncodeFlux: clip_l (теги/стиль), t5xxl (описание сцены), guidance
      - EmptySD3LatentImage: width, height
      - KSampler: seed, steps, sampler_name (опц.), scheduler (опц.)
      - SaveImage: filename_prefix

    Связи между нодами (links) берутся из шаблона без изменений.
    Конвертация workflow JSON → API-формат выполняется автоматически.

    Особенность KSampler: widgets_values содержит скрытый "control_after_generate"
    между seed и steps — при конвертации он пропускается (не входит в inputs).
    KSampler widgets_values layout: [seed, ctrl, steps, cfg, sampler_name, scheduler, denoise]
    """
    if not template_path.exists():
        print(f"[ОШИБКА] Workflow-шаблон не найден: {template_path}")
        sys.exit(1)

    with template_path.open(encoding="utf-8-sig") as f:
        wf = json.load(f)

    wf = copy.deepcopy(wf)

    # ── Патч widgets_values в нодах ───────────────────────────────────────────
    for node in wf["nodes"]:
        ntype = node["type"]
        wv: list = node.get("widgets_values", [])

        if ntype == "LoraLoader" and len(wv) >= 3:
            if lora_name:
                wv[0] = lora_name
                wv[1] = lora_scale   # strength_model
                wv[2] = lora_scale   # strength_clip

        elif ntype == "CLIPTextEncodeFlux" and len(wv) >= 3:
            wv[0] = clip_l_prompt   # clip_l  — ключевые слова и стиль (CLIP-L)
            wv[1] = t5xxl_prompt   # t5xxl   — развёрнутое описание сцены (T5-XXL)
            wv[2] = guidance

        elif ntype == "EmptySD3LatentImage" and len(wv) >= 2:
            wv[0] = width
            wv[1] = height

        elif ntype == "KSampler" and len(wv) >= 3:
            wv[0] = seed
            # wv[1] = "control_after_generate" (скрытый виджет — не трогаем)
            wv[2] = steps
            # wv layout: [seed, ctrl, steps, cfg, sampler_name, scheduler, denoise]
            if sampler_name is not None and len(wv) >= 5:
                wv[4] = sampler_name
            if scheduler is not None and len(wv) >= 6:
                wv[5] = scheduler

        elif ntype == "SaveImage" and len(wv) >= 1:
            wv[0] = filename_prefix

    # ── Конвертация в API-формат ──────────────────────────────────────────────
    # Связи: link_id → (from_node_id, from_output_slot)
    link_map: dict[int, tuple[int, int]] = {
        link[0]: (link[1], link[2]) for link in wf.get("links", [])
    }

    api: dict = {}
    for node in wf["nodes"]:
        node_id = str(node["id"])
        class_type = node["type"]
        wv = node.get("widgets_values", [])
        inputs: dict = {}
        widget_idx = 0

        for inp in node.get("inputs", []):
            name = inp["name"]
            link_id = inp.get("link")   # None если не подключено (null в JSON)

            if link_id is not None:
                # Вход подключён через link → ссылка на выход другой ноды
                if link_id in link_map:
                    from_node, from_slot = link_map[link_id]
                    inputs[name] = [str(from_node), from_slot]
            elif "widget" in inp:
                # Виджет → берём значение из widgets_values по порядку
                if widget_idx < len(wv):
                    inputs[name] = wv[widget_idx]
                widget_idx += 1
                # KSampler имеет скрытый виджет "control_after_generate" после seed
                # (он есть в widgets_values, но отсутствует в inputs — пропускаем слот)
                if class_type == "KSampler" and name == "seed":
                    widget_idx += 1

        api[node_id] = {"class_type": class_type, "inputs": inputs}

    return api


def check_comfyui(url: str) -> bool:
    """Проверяет доступность ComfyUI."""
    try:
        resp = requests.get(f"{url}/system_stats", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def submit_prompt(url: str, workflow: dict, client_id: str) -> str:
    """Отправляет задачу в ComfyUI и возвращает prompt_id."""
    payload = {"prompt": workflow, "client_id": client_id}
    resp = requests.post(f"{url}/prompt", json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"ComfyUI отклонил workflow: {data['error']}")
    return data["prompt_id"]


def wait_for_result(url: str, prompt_id: str, timeout: int = 600) -> dict:
    """Ждёт завершения задачи и возвращает outputs из истории."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(2)
        try:
            resp = requests.get(f"{url}/history/{prompt_id}", timeout=10)
        except requests.exceptions.RequestException:
            continue
        if resp.status_code != 200:
            continue
        history = resp.json()
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                raise RuntimeError(f"ComfyUI ошибка: {msgs}")
            outputs = entry.get("outputs", {})
            if outputs:
                return outputs
    raise TimeoutError(f"Таймаут {timeout}s: ComfyUI не ответил для prompt_id={prompt_id}")


def download_image(url: str, outputs: dict, save_path: Path) -> bool:
    """Скачивает первое изображение из outputs и сохраняет в save_path."""
    for node_output in outputs.values():
        images = node_output.get("images", [])
        for img_info in images:
            params = {
                "filename": img_info["filename"],
                "type": img_info.get("type", "output"),
            }
            subfolder = img_info.get("subfolder", "")
            if subfolder:
                params["subfolder"] = subfolder
            try:
                resp = requests.get(f"{url}/view", params=params, timeout=60)
                if resp.status_code == 200:
                    save_path.write_bytes(resp.content)
                    return True
            except requests.exceptions.RequestException:
                pass
    return False


# Режимы benchmark
BENCHMARK_MODES = ("full", "seed", "seeds")

# seed-варианты для блока A
BENCH_SEEDS = (42, 100, 200, 300, 400, 500, 777, 1000, 1337, 9999)

# guidance-варианты для блока B (full)
BENCH_GUIDANCES = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0)

# lora_scale-варианты для блока C (full)
BENCH_LORAS = (0.6, 0.75, 0.85, 1.0, 1.15, 1.5, 2.0, 2.5, 3.0, 3.5)


def _build_seed_matrix(
    steps: int,
    base_guidance: float,
    base_lora: float,
) -> list[tuple[str, int, int, float, float]]:
    """
    Блок A (10 прогонов): перебор seed при фиксированных steps/guidance/lora.
    Возвращает: [(dir_name, seed, steps, guidance, lora_scale), ...]
    """
    runs = []
    for rs in BENCH_SEEDS:
        runs.append((f"rm_{rs}_st{steps}", rs, steps, base_guidance, base_lora))
    return runs


def _build_custom_seeds_matrix(
    seeds: list[int],
    steps: int,
    guidance: float,
    lora_scale: float,
) -> list[tuple[str, int, int, float, float]]:
    """
    Произвольный список seed с фиксированными steps/guidance/lora_scale.
    Используется режимом --benchmark seeds вместе с --benchmark-seeds и --benchmark-lora.
    Возвращает: [(dir_name, seed, steps, guidance, lora_scale), ...]
    """
    l_tag = str(lora_scale).replace(".", "")
    runs = []
    for rs in seeds:
        runs.append((f"rm_{rs}_st{steps}_l{l_tag}", rs, steps, guidance, lora_scale))
    return runs


def _build_full_matrix(
    steps: int,
    base_guidance: float,
    base_lora: float,
) -> list[tuple[str, int, int, float, float]]:
    """
    Матрица из 30 прогонов:
      Блок A (10): seed при фиксированных steps/guidance/lora
      Блок B (10): guidance при seed=42, фикс. steps/lora
      Блок C (10): lora_scale при seed=42, фикс. steps/guidance=3.5
    Возвращает: [(dir_name, seed, steps, guidance, lora_scale), ...]
    """
    runs = []

    # Блок A — seed
    for rs in BENCH_SEEDS:
        runs.append((f"rm_{rs}_st{steps}", rs, steps, base_guidance, base_lora))

    # Блок B — guidance (seed=42, lora=base)
    for g in BENCH_GUIDANCES:
        g_tag = f"{g}".replace(".", "")
        runs.append((f"rm_42_st{steps}_g{g_tag}", 42, steps, g, base_lora))

    # Блок C — lora_scale (seed=42, guidance=3.5)
    for ls in BENCH_LORAS:
        ls_tag = f"{ls}".replace(".", "")
        runs.append((f"rm_42_st{steps}_l{ls_tag}", 42, steps, 3.5, ls))

    assert len(runs) == 30, f"Ожидалось 30 прогонов, получилось {len(runs)}"
    return runs


def _run_label(
    dir_name: str, seed: int, steps: int, guidance: float, lora_scale: float,
    base_guidance: float, base_lora: float,
) -> str:
    """Короткая подпись к варианту для коллажа и консоли."""
    if guidance != base_guidance and lora_scale == base_lora:
        return f"guidance={guidance}"
    if lora_scale != base_lora and guidance == 3.5:
        return f"lora={lora_scale}"
    return f"seed={seed}"


def _save_benchmark_grid(
    card: dict,
    runs: list[tuple[str, int, int, float, float]],
    img_paths: list[Path | None],
    out_path: Path,
    base_guidance: float,
    base_lora: float,
    mode: str,
) -> None:
    """Строит коллаж и сохраняет в out_path.

    full: сетка 10×3 (seed / guidance / lora)
    seed: сетка 5×2
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage
    import numpy as np
    import math

    n = len(runs)
    if mode == "seed":
        COLS, ROWS = 5, 2
    elif mode == "seeds":
        # Динамическая сетка: стараемся держать ~5 колонок
        COLS = min(n, 5)
        ROWS = math.ceil(n / COLS)
    else:
        COLS, ROWS = 10, 3

    THUMB_W, THUMB_H = 160, 320
    FIG_W = COLS * (THUMB_W / 100) + 0.4
    FIG_H = ROWS * (THUMB_H / 100) + 1.8

    fig, axes = plt.subplots(ROWS, COLS, figsize=(FIG_W, FIG_H),
                             squeeze=False,
                             gridspec_kw={"hspace": 0.6, "wspace": 0.06})
    fig.patch.set_facecolor("#111111")

    steps_val = runs[0][2] if runs else "?"
    lora_val = runs[0][4] if runs else "?"
    if mode == "seeds":
        title = (
            f"{card['card_id']}  —  {card.get('title_en', '')} / {card.get('title_ru', '')}  "
            f"[steps={steps_val}  lora={lora_val}  guidance={base_guidance}]"
        )
    else:
        title = (
            f"{card['card_id']}  —  {card.get('title_en', '')} / {card.get('title_ru', '')}  "
            f"[steps={steps_val}]"
        )
    fig.suptitle(title, color="white", fontsize=9, fontweight="bold", y=0.998)

    if mode == "seed":
        block_labels = ["seed"] * n
    elif mode == "seeds":
        block_labels = [f"seed={r[1]}" for r in runs]
    else:
        block_labels = (
            ["A: seed"] * len(BENCH_SEEDS) +
            ["B: guidance"] * len(BENCH_GUIDANCES) +
            ["C: lora_scale"] * len(BENCH_LORAS)
        )

    axes_flat = axes.flat
    for idx, (ax, (dir_name, seed, steps, guidance, lora_scale), img_path) in enumerate(
        zip(axes_flat, runs, img_paths)
    ):
        ax.set_facecolor("#1a1a1a")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        if img_path and img_path.exists():
            try:
                img = PILImage.open(img_path).convert("RGB")
                img.thumbnail((THUMB_W, THUMB_H), PILImage.LANCZOS)
                ax.imshow(np.array(img), aspect="auto")
            except Exception:
                ax.text(0.5, 0.5, "ошибка\nчтения", ha="center", va="center",
                        color="#ff4444", fontsize=6, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "нет\nфайла", ha="center", va="center",
                    color="#888888", fontsize=6, transform=ax.transAxes)

        label = _run_label(dir_name, seed, steps, guidance, lora_scale,
                           base_guidance, base_lora)
        block = block_labels[idx] if idx < len(block_labels) else ""
        if mode == "seeds":
            full_label = f"seed={seed}\nlora={lora_scale}"
        else:
            full_label = f"[{block}]\n{label}"
        ax.set_xlabel(full_label, color="#cccccc", fontsize=5.5,
                      labelpad=3, linespacing=1.3)

    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → коллаж сохранён: {out_path.name}")


def _run_benchmark(args, lora_name: str | None, width: int, height: int, trigger: str) -> None:
    """
    Запускает benchmark-генерации для каждой карты (или --cards).

    Режимы (--benchmark full|seed):
      full  — 30 прогонов: блок A (seed×10) + B (guidance×10) + C (lora×10)
      seed  — 10 прогонов: только блок A (seed×10), всё остальное по умолчанию

    steps берётся из --steps.
    Фиксированные базовые значения для незадействованных параметров:
      guidance   = --guidance (default 3.5)
      lora_scale = --lora-scale (default 1.0)
      seed       = 42 (в блоках B и C)

    Результаты:
      data/generated/benchmark/<mode>/rm_{seed}_st{steps}/{card_id}.png
      data/generated/benchmark/<mode>/{card_id}_benchmark.png  (коллаж)
    """
    mode = args.benchmark  # "full" или "seed"

    print(f"Проверка ComfyUI: {args.url}")
    if not check_comfyui(args.url):
        print(f"[ОШИБКА] ComfyUI недоступен по адресу {args.url}")
        sys.exit(1)
    print("  OK — ComfyUI отвечает\n")

    if not args.workflow.exists():
        print(f"[ОШИБКА] Workflow-шаблон не найден: {args.workflow}")
        sys.exit(1)

    all_cards = load_meta(MISSING_META)
    if not all_cards:
        print("[ОШИБКА] Нет карт в meta.jsonl")
        sys.exit(1)

    if args.cards:
        wanted = {c.strip() for c in args.cards.split(",")}
        cards = [c for c in all_cards if c["card_id"] in wanted]
        missing_ids = wanted - {c["card_id"] for c in cards}
        if missing_ids:
            print(f"[!] Не найдены card_id: {', '.join(sorted(missing_ids))}")
        if not cards:
            sys.exit(1)
    else:
        cards = all_cards

    BASE_G = args.guidance
    BASE_L = args.lora_scale
    steps = args.steps

    if mode == "seed":
        runs = _build_seed_matrix(steps, BASE_G, BASE_L)
        mode_label = "seed (10 прогонов)"
    elif mode == "seeds":
        custom_seeds = [int(s.strip()) for s in args.benchmark_seeds.split(",") if s.strip()]
        if not custom_seeds:
            print("[ОШИБКА] --benchmark-seeds не задан или пуст. Пример: --benchmark-seeds 42,100,777")
            sys.exit(1)
        bench_lora = args.benchmark_lora if args.benchmark_lora is not None else BASE_L
        runs = _build_custom_seeds_matrix(custom_seeds, steps, BASE_G, bench_lora)
        mode_label = f"seeds ({len(runs)} прогонов, lora_scale={bench_lora})"
    else:
        runs = _build_full_matrix(steps, BASE_G, BASE_L)
        mode_label = "full (30 прогонов: seed + guidance + lora)"

    total_runs = len(runs)
    bench_dir = BENCHMARK_DIR / mode
    bench_dir.mkdir(parents=True, exist_ok=True)
    client_id = str(uuid.uuid4())

    print(f"Benchmark [{mode_label}]: {len(cards)} карт × {total_runs} вариантов = {len(cards) * total_runs} генераций")
    print(f"  steps={steps}  guidance={BASE_G}  lora_scale={BASE_L}")
    print(f"  Выходная директория: {bench_dir}\n")

    total_ok = 0
    total_errors = 0

    for card_idx, card in enumerate(cards):
        card_id = card["card_id"]
        title = card.get("title_en", card_id)
        print(f"{'─' * 60}")
        print(f"  Карта [{card_idx + 1}/{len(cards)}]: {card_id} — {title}")
        print(f"{'─' * 60}")

        clip_l_prompt = build_clip_l_prompt(card, trigger)
        t5xxl_prompt = build_t5xxl_prompt(card)

        card_ok = 0
        card_errors: list[str] = []
        img_paths: list[Path | None] = []

        for i, (dir_name, seed, st, guidance, lora_scale) in enumerate(runs, 1):
            run_dir = bench_dir / dir_name
            run_dir.mkdir(parents=True, exist_ok=True)
            out_path = run_dir / f"{card_id}.png"
            img_paths.append(out_path)

            label = _run_label(dir_name, seed, st, guidance, lora_scale, BASE_G, BASE_L)
            print(f"  [{i:02d}/{total_runs}] {label}  ->  {dir_name}/", end="", flush=True)

            if not args.overwrite and out_path.exists():
                print("  пропуск (уже есть)")
                card_ok += 1
                continue

            workflow = apply_workflow_template(
                args.workflow,
                lora_name=lora_name,
                lora_scale=lora_scale,
                clip_l_prompt=clip_l_prompt,
                t5xxl_prompt=t5xxl_prompt,
                guidance=guidance,
                width=width,
                height=height,
                steps=st,
                seed=seed,
                filename_prefix=f"bm_{card_id}_{dir_name}_",
            )

            try:
                prompt_id = submit_prompt(args.url, workflow, client_id)
                outputs = wait_for_result(args.url, prompt_id, timeout=args.timeout)
                saved = download_image(args.url, outputs, out_path)
                if saved:
                    print(f"  OK  ({out_path.name})")
                    card_ok += 1
                else:
                    print("  ОШИБКА: изображение не найдено в outputs")
                    card_errors.append(label)
            except TimeoutError as exc:
                print(f"  ТАЙМАУТ: {exc}")
                card_errors.append(label)
            except Exception as exc:
                print(f"  ОШИБКА: {exc}")
                card_errors.append(label)

        # ── Коллаж для карты ─────────────────────────────────────────────
        grid_path = bench_dir / f"{card_id}_benchmark.png"
        try:
            _save_benchmark_grid(card, runs, img_paths, grid_path, BASE_G, BASE_L, mode)
        except Exception as exc:
            print(f"  [!] Не удалось построить коллаж: {exc}")

        print(f"  Итог карты: {card_ok}/{total_runs} OK"
              + (f", ошибки: {len(card_errors)}" if card_errors else ""))
        total_ok += card_ok
        total_errors += len(card_errors)

    grand_total = len(cards) * total_runs
    print(f"\n{'=' * 60}")
    print(f"  Всего генераций: {total_ok} / {grand_total}")
    if total_errors:
        print(f"  Ошибок: {total_errors}")
    print(f"  Результат: {bench_dir}")
    print("=" * 60)


def main() -> None:
    load_dotenv(ROOT / ".env")
    trigger = os.environ.get("TRIGGER_WORD", "LoraTrigger")
    default_w = int(os.environ.get("TRAINING_RESOLUTION_W", 960))
    default_h = int(os.environ.get("TRAINING_RESOLUTION_H", 1952))

    parser = argparse.ArgumentParser(
        description="Генерация 56 карт таро через ComfyUI API (FLUX.1-dev + LoRA)"
    )

    # Соединение
    parser.add_argument(
        "--url", default=os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188"),
        help="URL ComfyUI (default: http://127.0.0.1:8188)",
    )

    # Workflow-шаблон
    parser.add_argument(
        "--workflow", type=Path,
        default=Path(os.environ.get("COMFYUI_WORKFLOW", str(WORKFLOW_TEMPLATE))),
        help=f"Путь к workflow JSON (default: {WORKFLOW_TEMPLATE.name})",
    )

    # LoRA
    parser.add_argument(
        "--lora",
        default=os.environ.get("COMFYUI_LORA", ""),
        help="Имя LoRA в ComfyUI models/loras/ (если не задан — берётся из шаблона)",
    )
    parser.add_argument(
        "--lora-scale", type=float, default=1.0,
        help="Сила LoRA, 0.0–1.0 (default: 1.0)",
    )

    # Параметры генерации
    parser.add_argument("--steps", type=int, default=28, help="Шаги inference (default: 28)")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale (default: 3.5)")
    parser.add_argument("--width", type=int, default=default_w, help=f"Ширина (default: {default_w})")
    parser.add_argument("--height", type=int, default=default_h, help=f"Высота (default: {default_h})")
    parser.add_argument("--seed", type=int, default=42, help="Начальный seed (default: 42)")
    parser.add_argument(
        "--random-seed", action="store_true",
        help="Случайный seed для каждой карты (игнорирует --seed)",
    )

    # Фильтрация карт
    parser.add_argument(
        "--cards", type=str, default=None,
        help="card_id через запятую, например cups_01,swords_knight",
    )

    # Вывод
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"Папка для результатов (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Перезаписывать уже существующие файлы",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Таймаут ожидания от ComfyUI в секундах (default: 600)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Показать промпты без генерации")

    # Hypermeta — индивидуальные гиперпараметры для каждой карты
    parser.add_argument(
        "--hypermeta", action="store_true",
        help=(
            "Использовать data/missing/hyper_meta.jsonl: для каждой карты берутся "
            "индивидуальные seed/guidance/lora_scale/steps. "
            "Параметры CLI (--seed, --guidance и т.д.) используются как fallback "
            "если карта отсутствует в hyper_meta или поле не задано."
        ),
    )

    # Benchmark-режим
    parser.add_argument(
        "--benchmark",
        choices=BENCHMARK_MODES,
        metavar="{" + "|".join(BENCHMARK_MODES) + "}",
        default=None,
        help=(
            "Режим подбора гиперпараметров. "
            "full — 30 прогонов (seed×10 + guidance×10 + lora×10); "
            "seed — 10 прогонов (стандартный набор seed, всё остальное по умолчанию); "
            "seeds — произвольный список seed из --benchmark-seeds с фиксированным --benchmark-lora. "
            "steps берётся из --steps. Результаты в data/generated/benchmark/<mode>/."
        ),
    )
    parser.add_argument(
        "--benchmark-seeds",
        type=str,
        default="",
        metavar="42,100,777,...",
        help="Список seed через запятую для режима --benchmark seeds.",
    )
    parser.add_argument(
        "--benchmark-lora",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Фиксированный lora_scale для режима --benchmark seeds (default: --lora-scale).",
    )
    args = parser.parse_args()

    lora_name: str | None = args.lora or None
    width = snap_to(args.width)
    height = snap_to(args.height)

    # ── Загрузка hyper_meta (если задан --hypermeta) ──────────────────────────
    hyper: dict[str, dict] = {}
    if args.hypermeta:
        hyper = load_hyper_meta(HYPER_META)
        print(f"[hypermeta] загружено: {len(hyper)} карт из {HYPER_META.name}\n")

    # ── Загрузка карт ─────────────────────────────────────────────────────────
    all_cards = load_meta(MISSING_META)
    if args.cards:
        wanted = {c.strip() for c in args.cards.split(",")}
        cards = [c for c in all_cards if c["card_id"] in wanted]
        missing_ids = wanted - {c["card_id"] for c in cards}
        if missing_ids:
            print(f"[!] Не найдены card_id: {', '.join(sorted(missing_ids))}")
        if not cards:
            sys.exit(1)
    else:
        cards = all_cards

    # ── Dry-run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\nWorkflow:   {args.workflow}")
        print(f"URL:        {args.url}")
        print(f"LoRA:       {lora_name or '(из шаблона)'}")
        print(f"Разрешение: {width}×{height}  |  {args.steps} шагов  |  guidance {args.guidance}")
        print("\n── ПРОМПТЫ (CLIP-L / T5-XXL) ─────────────────────────────────────────")
        for entry in cards:
            print(f"\n[{entry['card_id']}]  {entry['title_en']}")
            print(f"  CLIP-L (теги):  {build_clip_l_prompt(entry, trigger)}")
            print(f"  T5-XXL (сцена): {build_t5xxl_prompt(entry)}")
        print()
        return

    # ── Benchmark-режим ──────────────────────────────────────────────────────
    if args.benchmark is not None:
        _run_benchmark(args, lora_name, width, height, trigger)
        return

    # ── Проверка ComfyUI ─────────────────────────────────────────────────────
    print(f"Проверка ComfyUI: {args.url}")
    if not check_comfyui(args.url):
        print(f"[ОШИБКА] ComfyUI недоступен по адресу {args.url}")
        print("         Убедись, что ComfyUI запущен и доступен.")
        sys.exit(1)
    print("  OK — ComfyUI отвечает\n")

    if not lora_name:
        print("[!] --lora не задан → используется LoRA из workflow-шаблона.")
        print(f"    Шаблон: {args.workflow}\n")

    if not args.workflow.exists():
        print(f"[ОШИБКА] Workflow-шаблон не найден: {args.workflow}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client_id = str(uuid.uuid4())

    # ── Генерация ─────────────────────────────────────────────────────────────
    print(f"Генерация {len(cards)} карт -> {args.output_dir}\n")
    ok_count = 0
    error_cards: list[str] = []
    skipped = 0

    for i, entry in enumerate(cards):
        card_id = entry["card_id"]
        title = entry.get("title_en", card_id)
        out_path = args.output_dir / f"{card_id}.png"

        if not args.overwrite and out_path.exists():
            print(f"  [{i + 1}/{len(cards)}] пропуск: {card_id}.png (уже есть)")
            skipped += 1
            continue

        print(f"  [{i + 1}/{len(cards)}] {card_id} — {title}... ", end="", flush=True)

        clip_l_prompt = build_clip_l_prompt(entry, trigger)
        t5xxl_prompt = build_t5xxl_prompt(entry)

        # ── Применение hyper_meta (если --hypermeta) ──────────────────────
        hp = hyper.get(card_id, {}) if hyper else {}
        card_guidance = hp.get("guidance", args.guidance)
        card_lora_scale = hp.get("lora_scale", args.lora_scale)
        card_steps = hp.get("steps", args.steps)
        if args.random_seed:
            seed = random.randint(0, 2**32 - 1)
        elif "seed" in hp:
            seed = hp["seed"]
        else:
            seed = args.seed + i

        if hp:
            print(f"[seed={seed} g={card_guidance} l={card_lora_scale} st={card_steps}] ", end="", flush=True)

        workflow = apply_workflow_template(
            args.workflow,
            lora_name=lora_name,
            lora_scale=card_lora_scale,
            clip_l_prompt=clip_l_prompt,
            t5xxl_prompt=t5xxl_prompt,
            guidance=card_guidance,
            width=width,
            height=height,
            steps=card_steps,
            seed=seed,
            filename_prefix=f"tarot_{card_id}_",
        )

        try:
            prompt_id = submit_prompt(args.url, workflow, client_id)
            outputs = wait_for_result(args.url, prompt_id, timeout=args.timeout)
            saved = download_image(args.url, outputs, out_path)
            if saved:
                print(f"OK  ({out_path.name})")
                ok_count += 1
            else:
                print("ОШИБКА: изображение не найдено в outputs")
                error_cards.append(card_id)
        except TimeoutError as exc:
            print(f"ТАЙМАУТ: {exc}")
            error_cards.append(card_id)
        except Exception as exc:
            print(f"ОШИБКА: {exc}")
            error_cards.append(card_id)

    # ── Итог ─────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Сгенерировано: {ok_count}")
    if skipped:
        print(f"  Пропущено:     {skipped}  (уже есть; --overwrite чтобы перезаписать)")
    if error_cards:
        print(f"  Ошибки:        {len(error_cards)}  -> {', '.join(error_cards)}")
    print(f"  Результат:     {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
