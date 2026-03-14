#!/usr/bin/env python3
"""
add_bleed.py — добавляет bleed-отступ для печати карт таро.

Берёт готовые карты из data/generated/final/, добавляет равномерный
отступ (по умолчанию 2 мм) на каждую из 4 сторон и сохраняет в
data/generated/to_print/.

Размер отступа в пикселях рассчитывается пропорционально физическому
размеру карты (по умолчанию 69×120 мм):

    bleed_px_x = round(card_width  / card_w_mm * bleed_mm)
    bleed_px_y = round(card_height / card_h_mm * bleed_mm)

Пример запуска:
    python add_bleed.py
    python add_bleed.py --bleed-mm 3 --fill 0,0,0
    python add_bleed.py --input-dir data/generated/final --output-dir data/generated/to_print
"""

import argparse
import io
import sys
from pathlib import Path

# UTF-8 вывод на Windows cp1251 терминале
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from PIL import Image

PROJECT_ROOT = Path(__file__).parent
DEFAULT_INPUT  = PROJECT_ROOT / "data" / "generated" / "final"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "generated" / "to_print"

CARD_W_MM = 69.0
CARD_H_MM = 120.0
BLEED_MM  = 2.0


def parse_fill(s: str) -> tuple[int, int, int]:
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Цвет должен быть в формате R,G,B (например, 0,0,0), получено: {s!r}"
        )
    try:
        r, g, b = (int(p.strip()) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Не удалось разобрать цвет: {s!r}")
    for v in (r, g, b):
        if not 0 <= v <= 255:
            raise argparse.ArgumentTypeError(f"Компоненты цвета должны быть 0–255, получено {v}")
    return (r, g, b)


def add_bleed(
    img: Image.Image,
    bleed_mm: float,
    card_w_mm: float,
    card_h_mm: float,
    fill: tuple[int, int, int],
) -> tuple[Image.Image, int, int]:
    """
    Добавляет bleed-отступ к изображению.

    Возвращает (новое_изображение, bleed_px_x, bleed_px_y).
    """
    img = img.convert("RGB")
    w, h = img.size

    bx = round(w / card_w_mm * bleed_mm)
    by = round(h / card_h_mm * bleed_mm)

    new_w = w + 2 * bx
    new_h = h + 2 * by

    canvas = Image.new("RGB", (new_w, new_h), fill)
    canvas.paste(img, (bx, by))

    return canvas, bx, by


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Добавляет bleed-отступ для печати карт таро."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT,
        help=f"Папка с готовыми картами (default: {DEFAULT_INPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help=f"Папка для результатов (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--bleed-mm", type=float, default=BLEED_MM,
        help=f"Ширина bleed-отступа в мм на каждую сторону (default: {BLEED_MM})",
    )
    parser.add_argument(
        "--card-w-mm", type=float, default=CARD_W_MM,
        help=f"Ширина карты без bleed в мм (default: {CARD_W_MM})",
    )
    parser.add_argument(
        "--card-h-mm", type=float, default=CARD_H_MM,
        help=f"Высота карты без bleed в мм (default: {CARD_H_MM})",
    )
    parser.add_argument(
        "--fill", type=parse_fill, default="0,0,0",
        metavar="R,G,B",
        help="Цвет заливки bleed-отступа (default: 0,0,0 — чёрный)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Перезаписывать уже существующие файлы",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Показать что будет сделано без сохранения файлов",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"[ОШИБКА] Папка не найдена: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    # Собираем все PNG кроме коллажа
    sources = sorted(
        f for f in args.input_dir.glob("*.png")
        if f.stem != "all_cards"
    )

    if not sources:
        print(f"[!] Нет PNG-файлов в {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"add_bleed: {len(sources)} файлов  bleed={args.bleed_mm} мм  "
          f"карта={args.card_w_mm}x{args.card_h_mm} мм  "
          f"fill=rgb{args.fill}")
    print(f"  {args.input_dir}  ->  {args.output_dir}\n")

    ok = skipped = 0

    for src in sources:
        dst = args.output_dir / src.name

        if not args.overwrite and dst.exists():
            print(f"  [SKIP] {src.name}  (уже есть; --overwrite чтобы перезаписать)")
            skipped += 1
            continue

        try:
            img = Image.open(src)
            result, bx, by = add_bleed(
                img,
                bleed_mm=args.bleed_mm,
                card_w_mm=args.card_w_mm,
                card_h_mm=args.card_h_mm,
                fill=args.fill,
            )
            orig_size = f"{img.size[0]}x{img.size[1]}"
            new_size  = f"{result.size[0]}x{result.size[1]}"
            bleed_info = f"+{bx}px / +{by}px"

            if args.dry_run:
                print(f"  [DRY] {src.name:45s} {orig_size} -> {new_size}  ({bleed_info})")
            else:
                result.save(str(dst), format="PNG", optimize=False)
                print(f"  [OK]  {src.name:45s} {orig_size} -> {new_size}  ({bleed_info})")
            ok += 1

        except Exception as e:
            print(f"  [ERR] {src.name}: {e}", file=sys.stderr)
            skipped += 1

    print()
    if args.dry_run:
        print(f"Dry-run: {ok} файлов готовы к обработке.")
    else:
        print(f"Готово: {ok} обработано, {skipped} пропущено.")
        if ok:
            sample = next((args.output_dir / s.name for s in sources
                           if (args.output_dir / s.name).exists()), None)
            if sample:
                w, h = Image.open(sample).size
                print(f"Размер с bleed: {w}x{h} px  "
                      f"~= {args.card_w_mm + 2*args.bleed_mm:.0f}x"
                      f"{args.card_h_mm + 2*args.bleed_mm:.0f} мм")


if __name__ == "__main__":
    main()
