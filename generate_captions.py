# /// script
# requires-python = ">=3.11"
# dependencies = ["python-dotenv"]
# ///
"""
Генерирует .txt-капшены для тренировочных изображений LoRA.

Читает data/originals/meta.jsonl и для каждого изображения создаёт
одноимённый .txt-файл рядом с ним, совместимый с форматом kohya_ss / ai-toolkit.

Настройки берутся из .env (TRIGGER_WORD). При отсутствии .env используются
значения по умолчанию.

Запуск:
    uv run generate_captions.py
    uv run generate_captions.py --dry-run      # показать капшены без записи
    uv run generate_captions.py --output-dir path/to/dir  # писать в другую папку
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Принудительно UTF-8 для Windows-консоли
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Конфигурация ──────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

import os  # noqa: E402 — после load_dotenv

ORIGINALS_DIR = ROOT / "data" / "originals"
META_FILE = ORIGINALS_DIR / "meta.jsonl"

# Trigger-слово — берётся из .env, иначе дефолт
TRIGGER = os.getenv("TRIGGER_WORD", "LoraTrigger")

# Базовые стилевые теги — покрывают все позитивные поля style_bible.json
# (core_rendering + composition_rules + palette_rules + glitch_rules),
# кроме секции avoid (отрицания в supervised learning не работают).
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
    # неизменные
    "high contrast, heavy blacks, worn paint edges"
)

IMAGE_EXTENSIONS = {".webp", ".png", ".jpg", ".jpeg"}

# ──────────────────────────────────────────────────────────────────────────────


def build_caption(entry: dict) -> str:
    """Собирает строку капшена из записи meta.jsonl."""
    concept: str = entry.get("concept", "").strip().rstrip(".")
    parts = [TRIGGER, STYLE_PREFIX, concept]
    return ", ".join(p for p in parts if p)


def load_meta(meta_file: Path) -> dict[str, dict]:
    """Возвращает словарь {card_id: entry} из .jsonl-файла."""
    mapping: dict[str, dict] = {}
    with meta_file.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [!] Ошибка парсинга строки {lineno}: {e}")
                continue
            card_id: str = entry.get("card_id", "")
            if card_id:
                mapping[card_id] = entry
    return mapping


def find_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Генератор .txt-капшенов для LoRA-тренировки")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Показать капшены в консоли без записи файлов"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Куда записывать .txt-файлы (по умолчанию — рядом с изображениями)"
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir or ORIGINALS_DIR
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем мета
    if not META_FILE.exists():
        print(f"[ОШИБКА] Не найден файл метаданных: {META_FILE}")
        raise SystemExit(1)

    meta = load_meta(META_FILE)
    print(f"Загружено записей в meta.jsonl: {len(meta)}\n")

    images = find_images(ORIGINALS_DIR)
    if not images:
        print(f"[ОШИБКА] Изображения не найдены в: {ORIGINALS_DIR}")
        raise SystemExit(1)

    ok_count = 0
    warn_count = 0

    for img_path in images:
        # card_id в meta.jsonl совпадает с именем файла (включая расширение)
        card_id = img_path.name
        entry = meta.get(card_id)

        if entry is None:
            print(f"  [!] Нет записи в meta.jsonl для: {card_id}  — пропускаем")
            warn_count += 1
            continue

        caption = build_caption(entry)
        txt_path = output_dir / img_path.with_suffix(".txt").name

        if args.dry_run:
            print(f"-- {card_id}")
            print(f"   {caption}\n")
        else:
            txt_path.write_text(caption, encoding="utf-8")
            print(f"  OK  {txt_path.name}")

        ok_count += 1

    print(f"\nГотово: {ok_count} капшенов {'показано' if args.dry_run else 'записано'}", end="")
    if warn_count:
        print(f", {warn_count} пропущено (нет в meta.jsonl)", end="")
    print()


if __name__ == "__main__":
    main()
