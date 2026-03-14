#!/usr/bin/env python3
"""
to_a6.py — форматирует карты таро в A6 с bleed-отступом для печати.

Берёт готовые карты из data/generated/final/, центрирует каждую на листе A6
(105×148 мм) с чёрными полями и добавляет bleed-отступ (по умолчанию 2 мм)
на каждую из 4 сторон. Сохраняет в data/generated/a6/.

Логика:
  1. Рассчитать px/мм из размера самой карты (её физические размеры = 69×120 мм)
  2. Построить холст A6 в пикселях (105×148 мм × px/мм)
  3. Центрировать карту на этом холсте
  4. Добавить bleed-рамку (2 мм × px/мм) вокруг A6-холста

Итоговый размер файла: (105 + 2*bleed_mm) × (148 + 2*bleed_mm) мм

Пример запуска:
    python to_a6.py
    python to_a6.py --bleed-mm 3 --fill 255,255,255
    python to_a6.py --input-dir data/generated/final --output-dir data/generated/a6
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

PROJECT_ROOT  = Path(__file__).parent
DEFAULT_INPUT  = PROJECT_ROOT / "data" / "generated" / "final"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "generated" / "a6"

# Физические размеры карты (без bleed, без рамки A6)
CARD_W_MM = 69.0
CARD_H_MM = 120.0

# Размер листа A6
A6_W_MM = 105.0
A6_H_MM = 148.0

BLEED_MM = 2.0


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
            raise argparse.ArgumentTypeError(
                f"Компоненты цвета должны быть 0-255, получено {v}"
            )
    return (r, g, b)


def to_a6(
    img: Image.Image,
    card_w_mm: float,
    card_h_mm: float,
    a6_w_mm: float,
    a6_h_mm: float,
    bleed_mm: float,
    fill: tuple[int, int, int],
) -> tuple[Image.Image, dict]:
    """
    Центрирует img на листе A6 и добавляет bleed-отступ.

    Возвращает (результат, info):
      info содержит px/mm, размеры A6 и bleed в пикселях.
    """
    img = img.convert("RGB")
    card_w, card_h = img.size

    # px/мм по каждой оси карты
    ppm_x = card_w / card_w_mm
    ppm_y = card_h / card_h_mm

    # A6 в пикселях (используем среднее px/mm для изотропности)
    # Берём ppm отдельно по осям — карта уже может иметь чуть разный ppm_x/ppm_y
    a6_px_w = round(a6_w_mm * ppm_x)
    a6_px_h = round(a6_h_mm * ppm_y)

    # Bleed в пикселях
    bleed_px_x = round(bleed_mm * ppm_x)
    bleed_px_y = round(bleed_mm * ppm_y)

    # Итоговый размер холста = A6 + bleed со всех сторон
    total_w = a6_px_w + 2 * bleed_px_x
    total_h = a6_px_h + 2 * bleed_px_y

    # Смещение карты: bleed + центрирование на A6
    offset_x = bleed_px_x + (a6_px_w - card_w) // 2
    offset_y = bleed_px_y + (a6_px_h - card_h) // 2

    canvas = Image.new("RGB", (total_w, total_h), fill)
    canvas.paste(img, (offset_x, offset_y))

    info = {
        "ppm_x": ppm_x,
        "ppm_y": ppm_y,
        "a6_px": (a6_px_w, a6_px_h),
        "bleed_px": (bleed_px_x, bleed_px_y),
        "total_px": (total_w, total_h),
        "offset": (offset_x, offset_y),
    }
    return canvas, info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Форматирует карты таро в A6 с bleed-отступом для печати."
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
        help=f"Физическая ширина карты в мм (default: {CARD_W_MM})",
    )
    parser.add_argument(
        "--card-h-mm", type=float, default=CARD_H_MM,
        help=f"Физическая высота карты в мм (default: {CARD_H_MM})",
    )
    parser.add_argument(
        "--a6-w-mm", type=float, default=A6_W_MM,
        help=f"Ширина листа A6 в мм (default: {A6_W_MM})",
    )
    parser.add_argument(
        "--a6-h-mm", type=float, default=A6_H_MM,
        help=f"Высота листа A6 в мм (default: {A6_H_MM})",
    )
    parser.add_argument(
        "--fill", type=parse_fill, default="0,0,0",
        metavar="R,G,B",
        help="Цвет заполнения полей и bleed (default: 0,0,0 — чёрный)",
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

    sources = sorted(
        f for f in args.input_dir.glob("*.png")
        if f.stem != "all_cards"
    )

    if not sources:
        print(f"[!] Нет PNG-файлов в {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_mm_w = args.a6_w_mm + 2 * args.bleed_mm
    total_mm_h = args.a6_h_mm + 2 * args.bleed_mm

    print(f"to_a6: {len(sources)} файлов")
    print(f"  карта: {args.card_w_mm}x{args.card_h_mm} мм  ->  "
          f"A6: {args.a6_w_mm}x{args.a6_h_mm} мм  +  "
          f"bleed: {args.bleed_mm} мм  =  {total_mm_w}x{total_mm_h} мм")
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
            result, info = to_a6(
                img,
                card_w_mm=args.card_w_mm,
                card_h_mm=args.card_h_mm,
                a6_w_mm=args.a6_w_mm,
                a6_h_mm=args.a6_h_mm,
                bleed_mm=args.bleed_mm,
                fill=args.fill,
            )

            card_size  = f"{img.size[0]}x{img.size[1]}"
            final_size = f"{info['total_px'][0]}x{info['total_px'][1]}"
            a6_size    = f"{info['a6_px'][0]}x{info['a6_px'][1]}"
            bleed_info = f"bleed={info['bleed_px'][0]}x{info['bleed_px'][1]}px"
            offset     = f"offset={info['offset'][0]},{info['offset'][1]}"

            if args.dry_run:
                print(f"  [DRY] {src.name:45s} {card_size} -> A6:{a6_size} "
                      f"-> {final_size}  {bleed_info}  {offset}")
            else:
                result.save(str(dst), format="PNG", optimize=False)
                print(f"  [OK]  {src.name:45s} {card_size} -> {final_size}  "
                      f"{bleed_info}  {offset}")
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
            sample = next(
                (args.output_dir / s.name for s in sources
                 if (args.output_dir / s.name).exists()),
                None,
            )
            if sample:
                w, h = Image.open(sample).size
                print(f"Итоговый размер: {w}x{h} px  "
                      f"~= {total_mm_w:.0f}x{total_mm_h:.0f} мм")


if __name__ == "__main__":
    main()
