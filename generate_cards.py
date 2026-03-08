# /// script
# requires-python = ">=3.11"
# dependencies = ["python-dotenv"]
# ///
"""
Генерирует недостающие карты таро через FLUX.1-dev + LoRA (локально, diffusers).

Требует обученной LoRA в output/ и установленного ML-окружения (.venv).
Все параметры модели берутся из .env; LoRA определяется автоматически
(последний .safetensors из output/) или задаётся через --lora.

Запуск:
    uv run generate_cards.py                          # авто-LoRA, все 56 карт
    uv run generate_cards.py --lora output/cp2077_tarot_lora/cp2077_tarot_lora-000001000.safetensors
    uv run generate_cards.py --cards cups_01,swords_03  # отдельные карты
    uv run generate_cards.py --dry-run               # показать промпты без генерации
    uv run generate_cards.py --cpu-offload           # если VRAM < 24 GB

Результат сохраняется в data/generated/<card_id>.png
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def _ensure_venv() -> None:
    """Перезапускает скрипт в .venv если он уже не там."""
    if not VENV_PYTHON.exists():
        print("[ОШИБКА] .venv не найден. Сначала запусти: uv run setup_training.py")
        sys.exit(1)
    if Path(sys.executable).resolve() != VENV_PYTHON.resolve():
        sys.exit(subprocess.run([str(VENV_PYTHON), __file__] + sys.argv[1:]).returncode)


_ensure_venv()

# ── Импорт ML-зависимостей (только в .venv) ──────────────────────────────────
import torch  # noqa: E402
from diffusers import FluxPipeline  # noqa: E402
from tqdm import tqdm  # noqa: E402

# ── Константы ─────────────────────────────────────────────────────────────────

MISSING_META = ROOT / "data" / "missing" / "meta.jsonl"
OUTPUT_DIR = ROOT / "data" / "generated"

STYLE_PREFIX = (
    "tarot card illustration, cyberpunk city vibe, graphic poster style, "
    "matte ink gouache look, flat posterized shapes, thick black brush outlines, "
    "visible paper grain, paint splatter, screenprint texture, "
    "high contrast, heavy blacks, worn paint edges"
)

# ──────────────────────────────────────────────────────────────────────────────


def load_meta(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[ОШИБКА] Не найден: {path}")
        sys.exit(1)
    cards = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cards.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [!] Ошибка парсинга строки {lineno}: {e}")
    return cards


def build_prompt(entry: dict, trigger: str) -> str:
    concept = entry.get("concept", "").strip().rstrip(".")
    return f"{trigger}, {STYLE_PREFIX}, {concept}"


def find_latest_lora(output_dir: Path) -> Path | None:
    """Возвращает последний (по времени изменения) .safetensors из output/."""
    loras = sorted(
        output_dir.glob("**/*.safetensors"),
        key=lambda p: p.stat().st_mtime,
    )
    return loras[-1] if loras else None


def snap_to(value: int, multiple: int = 16) -> int:
    """Округляет до ближайшего кратного multiple."""
    return max(multiple, round(value / multiple) * multiple)


def main() -> None:
    load_dotenv(ROOT / ".env")
    hf_model_path = os.environ.get("HF_MODEL_PATH", "")
    trigger = os.environ.get("TRIGGER_WORD", "LoraTrigger")
    default_w = int(os.environ.get("TRAINING_RESOLUTION_W", 960))
    default_h = int(os.environ.get("TRAINING_RESOLUTION_H", 1952))

    parser = argparse.ArgumentParser(
        description="Генерация 56 карт таро через FLUX.1-dev + LoRA (diffusers)"
    )
    parser.add_argument(
        "--lora", type=Path, default=None,
        help="Путь к .safetensors LoRA (авто-определение из output/ если не указан)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"Папка для результатов (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--steps", type=int, default=28,
        help="Шаги inference (default: 28)",
    )
    parser.add_argument(
        "--guidance", type=float, default=3.5,
        help="Guidance scale (default: 3.5)",
    )
    parser.add_argument("--width", type=int, default=default_w, help=f"Ширина (default: {default_w})")
    parser.add_argument("--height", type=int, default=default_h, help=f"Высота (default: {default_h})")
    parser.add_argument(
        "--lora-scale", type=float, default=1.0,
        help="Сила LoRA, 0.0–1.0 (default: 1.0)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--cards", type=str, default=None,
        help="card_id через запятую, например cups_01,swords_knight",
    )
    parser.add_argument(
        "--cpu-offload", action="store_true",
        help="Включить CPU offload (снижает VRAM до ~12 GB, медленнее)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Пропускать уже существующие файлы (default: включено)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Перезаписывать уже существующие файлы",
    )
    parser.add_argument("--dry-run", action="store_true", help="Показать промпты без генерации")
    args = parser.parse_args()

    skip_existing = not args.overwrite

    # ── Поиск LoRA ────────────────────────────────────────────────────────────
    lora_path = args.lora
    if lora_path is None:
        lora_path = find_latest_lora(ROOT / "output") or find_latest_lora(ROOT / "output_test")
        if lora_path is None:
            print("[ОШИБКА] LoRA не найдена в output/. Запусти тренировку или укажи --lora <путь>.")
            sys.exit(1)
        print(f"  LoRA (авто): {lora_path}")
    else:
        if not lora_path.exists():
            print(f"[ОШИБКА] LoRA не найдена: {lora_path}")
            sys.exit(1)
        print(f"  LoRA: {lora_path}")

    if not hf_model_path:
        print("[ОШИБКА] HF_MODEL_PATH не задан в .env")
        sys.exit(1)

    width = snap_to(args.width)
    height = snap_to(args.height)

    # ── Загрузка карт ─────────────────────────────────────────────────────────
    all_cards = load_meta(MISSING_META)
    if args.cards:
        wanted = {c.strip() for c in args.cards.split(",")}
        cards = [c for c in all_cards if c["card_id"] in wanted]
        missing = wanted - {c["card_id"] for c in cards}
        if missing:
            print(f"[!] Не найдены card_id: {', '.join(sorted(missing))}")
        if not cards:
            sys.exit(1)
    else:
        cards = all_cards

    print(f"\n  Карт: {len(cards)}  |  {width}×{height}  |  {args.steps} шагов  |  guidance {args.guidance}")
    print(f"  Seed: {args.seed}  |  LoRA scale: {args.lora_scale}")

    # ── Dry-run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n── ПРОМПТЫ (dry-run) ────────────────────────────────────────────────")
        for entry in cards:
            print(f"\n[{entry['card_id']}]  {entry['title_en']}")
            print(f"  {build_prompt(entry, trigger)}")
        print()
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Загрузка модели ───────────────────────────────────────────────────────
    print(f"\nЗагрузка FLUX из: {hf_model_path}")
    pipe = FluxPipeline.from_pretrained(hf_model_path, torch_dtype=torch.bfloat16)

    print(f"Загрузка LoRA: {lora_path.name}  (scale={args.lora_scale})")
    pipe.load_lora_weights(str(lora_path), adapter_name="cp2077")
    pipe.set_adapters(["cp2077"], adapter_weights=[args.lora_scale])

    if args.cpu_offload:
        print("CPU offload включён (меньше VRAM, медленнее)...")
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    # ── Генерация ─────────────────────────────────────────────────────────────
    print(f"\nГенерация -> {args.output_dir}\n")
    errors: list[str] = []
    skipped = 0

    for i, entry in enumerate(tqdm(cards, unit="card", ncols=80)):
        card_id = entry["card_id"]
        out_path = args.output_dir / f"{card_id}.png"

        if skip_existing and out_path.exists():
            tqdm.write(f"  пропуск (уже есть): {card_id}.png")
            skipped += 1
            continue

        prompt = build_prompt(entry, trigger)
        generator = torch.Generator(device="cuda").manual_seed(args.seed + i)

        try:
            image = pipe(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                generator=generator,
            ).images[0]
            image.save(out_path)
            tqdm.write(f"  OK  {card_id}.png")
        except Exception as exc:
            tqdm.write(f"  [!] ОШИБКА {card_id}: {exc}")
            errors.append(card_id)

    # ── Итог ─────────────────────────────────────────────────────────────────
    generated = len(cards) - len(errors) - skipped
    print(f"\n{'=' * 60}")
    print(f"  Сгенерировано: {generated}")
    if skipped:
        print(f"  Пропущено:     {skipped}  (уже существуют; --overwrite чтобы перезаписать)")
    if errors:
        print(f"  Ошибки:        {len(errors)}  -> {', '.join(errors)}")
    print(f"  Результат:     {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
