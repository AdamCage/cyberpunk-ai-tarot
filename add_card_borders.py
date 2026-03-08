#!/usr/bin/env python3
"""
add_card_borders.py — добавляет киберпанк-рамку и название карты на арты.

Лейаут:
  ┌──────────┬──────────────────────────┐
  │  левая   │                          │
  │ колонка  │        арт карты         │
  │ (название│                          │
  │ + штрих- │                          │
  │  код)    │                          │
  └──────────┴──────────────────────────┘

Источники:
  • data/generated/res/*.png   — сгенерированные карты (minor arcana)
  • data/originals/*.webp      — оригинальные арты (major arcana + kings)

Вывод: data/generated/final/
"""

import argparse
import io
import json
import os
import sys

# UTF-8 вывод на Windows cp1251 терминале
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ─── Пути ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
RES_DIR      = PROJECT_ROOT / "data/generated/res"
ORIGINALS_DIR= PROJECT_ROOT / "data/originals"
OUTPUT_DIR   = PROJECT_ROOT / "data/generated/final"
MISSING_META = PROJECT_ROOT / "data/missing/meta.jsonl"
ORIGINALS_META = PROJECT_ROOT / "data/originals/meta.jsonl"
FONT_PATH    = PROJECT_ROOT / "data/RussoOne-Regular.ttf"

# ─── Цвета по масти ────────────────────────────────────────────────────────────

SUIT_COLORS = {
    "cups":      {"accent": (220, 40, 120),  "glow": (180, 20,  90),  "dim": (45, 8, 25)},
    "swords":    {"accent": (150, 70, 255),  "glow": (215, 195, 55),  "dim": (28, 12, 50)},
    "wands":     {"accent": (210, 30,  30),  "glow": (180, 15,  15),  "dim": (42, 6,  6)},
    "pentacles": {"accent": (200, 170, 20),  "glow": (0,  180, 160),  "dim": (38, 32, 4)},
    "major":     {"accent": (210, 160, 30),  "glow": (160, 100, 10),  "dim": (38, 28, 4)},
}
DEFAULT_COLORS = {"accent": (180, 180, 180), "glow": (100, 100, 100), "dim": (25, 25, 25)}

# ─── Шрифты ────────────────────────────────────────────────────────────────────

def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Загружает Russo One, с фоллбэком на системные шрифты."""
    candidates = [FONT_PATH]
    if bold:
        candidates += [
            Path(r"C:\Windows\Fonts\ariblk.ttf"),
            Path(r"C:\Windows\Fonts\impact.ttf"),
        ]
    else:
        candidates += [
            Path(r"C:\Windows\Fonts\bahnschrift.ttf"),
            Path(r"C:\Windows\Fonts\arial.ttf"),
        ]
    for p in candidates:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                continue
    return ImageFont.load_default()


# ─── Метаданные ────────────────────────────────────────────────────────────────

_SUIT_NAMES_RU = {
    "cups":      "Кубков",
    "swords":    "Мечей",
    "wands":     "Жезлов",
    "pentacles": "Пентаклей",
}
_RANK_NAMES_RU = {
    "01": "Туз",    "1":  "Туз",
    "02": "Двойка", "2":  "Двойка",
    "03": "Тройка", "3":  "Тройка",
    "04": "Четвёрка","4": "Четвёрка",
    "05": "Пятёрка","5":  "Пятёрка",
    "06": "Шестёрка","6": "Шестёрка",
    "07": "Семёрка","7":  "Семёрка",
    "08": "Восьмёрка","8":"Восьмёрка",
    "09": "Девятка","9":  "Девятка",
    "10": "Десятка",
    "page":   "Паж",
    "knight": "Рыцарь",
    "queen":  "Королева",
    "king":   "Король",
}


def _infer_from_card_id(card_id: str) -> tuple[str, str]:
    """(title_ru, suit) из card_id вида 'cups_07', 'swords_knight'."""
    parts = card_id.rsplit("_", 1)
    if len(parts) != 2:
        return card_id, ""
    suit_key, rank_key = parts
    suit_ru = _SUIT_NAMES_RU.get(suit_key, suit_key.capitalize())
    rank_ru = _RANK_NAMES_RU.get(rank_key, rank_key.capitalize())
    return f"{rank_ru} {suit_ru}", suit_key


def load_meta(paths: list[Path]) -> dict[str, dict]:
    """Объединяет несколько meta.jsonl файлов в один словарь.
    Ключ = card_id без расширения файла (stem).
    """
    meta: dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                raw_id = entry["card_id"]
                # Нормализуем ключ: убираем расширение если есть
                key = Path(raw_id).stem if "." in raw_id else raw_id
                entry["_key"] = key
                meta[key] = entry
    return meta


def resolve_card_info(card_id: str, entry: dict | None) -> tuple[str, str]:
    """Возвращает (title_ru, suit) с фоллбэком через card_id."""
    if entry:
        title_ru = entry.get("title_ru") or entry.get("title_en", "")
        suit = entry.get("suit") or ""
        arcana = entry.get("arcana") or ""
        # Старший аркан без масти → используем "major" для цветовой схемы
        if arcana == "major" and not suit:
            suit = "major"
    else:
        title_ru, suit = "", ""

    if not title_ru or not suit:
        inferred_title, inferred_suit = _infer_from_card_id(card_id)
        if not title_ru:
            title_ru = inferred_title
        if not suit:
            suit = inferred_suit

    return title_ru, suit


# ─── Штрихкод (Code 128B, только ASCII) ───────────────────────────────────────

# Code 128 B: кодировка символов (стартовый символ = 104, стоп = 106)
_C128_PATTERNS = [
    "11011001100", "11001101100", "11001100110", "10010011000", "10010001100",
    "10001001100", "10011001000", "10011000100", "10001100100", "11001001000",
    "11001000100", "11000100100", "10110011100", "10011011100", "10011001110",
    "10111001100", "10011101100", "10011100110", "11001110010", "11001011100",
    "11001001110", "11011100100", "11001110100", "11101101110", "11101001100",
    "11100101100", "11100100110", "11101100100", "11100110100", "11100110010",
    "11011011000", "11011000110", "11000110110", "10100011000", "10001011000",
    "10001000110", "10110001000", "10001101000", "10001100010", "11010001000",
    "11000101000", "11000100010", "10110111000", "10110001110", "10001101110",
    "10111011000", "10111000110", "10001110110", "11101110110", "11010001110",
    "11000101110", "11011101000", "11011100010", "11011101110", "11101011000",
    "11101000110", "11100010110", "11110101000", "11110100010", "11110010010",
    "11011000010", "11001000010", "11110111010", "11000010100", "10001111010",
    "10100111100", "10010111100", "10010011110", "10111100100", "10011110100",
    "10011110010", "11110100100", "11110010100", "11110010010", "11011011110",
    "11011110110", "11110110110", "10101111000", "10100011110", "10001011110",
    "10111101000", "10111100010", "11110101100", "00110100010", "10001001110",
    "10011101110", "10011110110", "00010100110", "11110110100", "00110101000",
    "00110100100", "00110010100", "00110010010", "11011011100", "11011001110",
    "11011110100", "11011110010", "11001011000", "11001001100", "11001100100",
    "10011001000", "10011000100",
]
_C128_START_B = "11010010000"  # start B
_C128_STOP    = "1100011101011"


def _encode_barcode(text: str) -> str:
    """Возвращает строку из 0/1 для Code 128B."""
    # Работаем только с ASCII-printable символами
    clean = "".join(c for c in text if 32 <= ord(c) <= 126)
    if not clean:
        clean = "TAROT"

    checksum = 104  # стартовый B
    bars = _C128_START_B
    for i, ch in enumerate(clean):
        val = ord(ch) - 32  # Code B: символ 32 = индекс 0
        checksum += (i + 1) * val
        bars += _C128_PATTERNS[val]

    bars += _C128_PATTERNS[checksum % 103]
    bars += _C128_STOP
    return bars


def draw_barcode(draw: ImageDraw.ImageDraw,
                 text: str,
                 x: int, y: int, width: int, height: int,
                 color: tuple) -> None:
    """Рисует штрихкод Code 128B в заданной области."""
    bits = _encode_barcode(text)
    n = len(bits)
    if n == 0:
        return

    bar_w = max(1, width / n)
    cx = x
    for bit in bits:
        x0 = int(cx)
        x1 = max(x0 + 1, int(cx + bar_w))
        if bit == "1":
            draw.rectangle([x0, y, x1, y + height], fill=color)
        cx += bar_w


# ─── Декоративные примитивы ────────────────────────────────────────────────────

def _rect(draw: ImageDraw.ImageDraw, x0, y0, x1, y1, fill):
    draw.rectangle([min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1)], fill=fill)


def draw_corner(draw: ImageDraw.ImageDraw,
                x: int, y: int, size: int, color: tuple,
                flip_x=False, flip_y=False) -> None:
    """Г-образный угловой маркер."""
    sx, sy = (-1 if flip_x else 1), (-1 if flip_y else 1)
    thick = max(2, size // 8)
    _rect(draw, x, y, x + sx * size, y + sy * thick, color)       # горизонталь
    _rect(draw, x, y, x + sx * thick, y + sy * size, color)       # вертикаль
    r = max(2, thick - 1)
    ex, ey = x + sx * size, y + sy * size
    draw.ellipse([ex - r, ey - r, ex + r, ey + r], fill=color)    # точка


def draw_glitch_sep(draw: ImageDraw.ImageDraw,
                    x0: int, y: int, x1: int,
                    color_a: tuple, color_b: tuple,
                    segments: int = 9) -> None:
    """Сегментированная глитч-линия."""
    seg_w = (x1 - x0) / segments
    for i in range(segments):
        off = 1 if i % 2 == 0 else 0
        sx = x0 + int(i * seg_w)
        ex = x0 + int((i + 1) * seg_w) - 2
        c = color_a if i % 3 != 2 else color_b
        _rect(draw, sx, y + off, ex, y + off + 1, c)


# ─── Вертикальный текст ────────────────────────────────────────────────────────

def _text_rotated(canvas: Image.Image, text: str, font: ImageFont.FreeTypeFont,
                  cx: int, cy: int, color: tuple, angle: int = 90) -> None:
    """Рисует текст повёрнутым на angle градусов, центрированным в (cx, cy)."""
    dummy = ImageDraw.Draw(canvas)
    bbox = dummy.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Создаём временный слой под текст
    tmp = Image.new("RGBA", (tw + 4, th + 4), (0, 0, 0, 0))
    td = ImageDraw.Draw(tmp)
    td.text((2 - bbox[0], 2 - bbox[1]), text, font=font, fill=color + (255,))

    rotated = tmp.rotate(angle, expand=True)
    rw, rh = rotated.size
    paste_x = cx - rw // 2
    paste_y = cy - rh // 2

    if canvas.mode == "RGBA":
        canvas.paste(rotated, (paste_x, paste_y), rotated)
    else:
        # Конвертируем canvas временно в RGBA для paste
        rgba = canvas.convert("RGBA")
        rgba.paste(rotated, (paste_x, paste_y), rotated)
        # Вернуть обратно RGB
        result = rgba.convert("RGB")
        canvas.paste(result)


# ─── Основная функция рамки ────────────────────────────────────────────────────

# Размер левой колонки относительно ширины арта
SIDEBAR_W = 148   # px

def add_border(img: Image.Image, title_ru: str, suit: str) -> Image.Image:
    """
    Собирает финальную карту:
    - Левая колонка (SIDEBAR_W px): вертикальное название + штрихкод
    - Правая часть: арт с тонкой рамкой
    Всё на чёрном фоне.
    """
    colors = SUIT_COLORS.get(suit, SUIT_COLORS.get("major", DEFAULT_COLORS))
    accent = colors["accent"]
    glow   = colors["glow"]
    dim    = colors["dim"]

    img = img.convert("RGB")
    art_w, art_h = img.size

    # Отступы вокруг арта
    pad_t = 14    # сверху
    pad_b = 14    # снизу
    pad_r = 12    # справа
    border = 3    # толщина акцентной линии вокруг арта

    total_w = SIDEBAR_W + border + art_w + pad_r
    total_h = pad_t + art_h + pad_b

    canvas = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # ── Фон левой колонки ─────────────────────────────────────────────────────
    # Тёмный акцентный фон
    draw.rectangle([0, 0, SIDEBAR_W - 1, total_h - 1], fill=dim)

    # Вертикальная акцентная полоса — граница колонки
    _rect(draw, SIDEBAR_W, 0, SIDEBAR_W + border - 1, total_h, accent)

    # ── Арт ───────────────────────────────────────────────────────────────────
    art_x = SIDEBAR_W + border
    art_y = pad_t
    canvas.paste(img, (art_x, art_y))

    # Тонкая рамка вокруг арта
    _rect(draw, art_x, art_y - border, art_x + art_w, art_y - 1, accent)           # верх
    _rect(draw, art_x, art_y + art_h + 1, art_x + art_w, art_y + art_h + border, accent)  # низ
    _rect(draw, art_x + art_w + 1, art_y, art_x + art_w + border, art_y + art_h, accent)  # право

    # Угловые маркеры арта
    cs = 24
    draw_corner(draw, art_x, art_y, cs, accent, flip_x=False, flip_y=False)
    draw_corner(draw, art_x + art_w, art_y, cs, accent, flip_x=True, flip_y=False)
    draw_corner(draw, art_x, art_y + art_h, cs, accent, flip_x=False, flip_y=True)
    draw_corner(draw, art_x + art_w, art_y + art_h, cs, accent, flip_x=True, flip_y=True)

    # ── Верхний и нижний края колонки ─────────────────────────────────────────
    _rect(draw, 0, 0, SIDEBAR_W, border - 1, accent)               # верхняя полоса
    _rect(draw, 0, total_h - border, SIDEBAR_W, total_h - 1, accent) # нижняя полоса

    # Угловые маркеры колонки
    draw_corner(draw, 2, 2, 18, glow)
    draw_corner(draw, 2, total_h - 2, 18, glow, flip_y=True)

    # ── Горизонтальный разделитель в колонке (верхняя треть) ──────────────────
    div_y1 = total_h // 3
    draw_glitch_sep(draw, 6, div_y1, SIDEBAR_W - 6, accent, glow, segments=7)

    # ── Вертикальное название (снизу вверх, нижние 2/3 колонки) ───────────────
    text_zone_top    = div_y1 + 16
    text_zone_bottom = total_h - border - 12
    text_zone_h = text_zone_bottom - text_zone_top
    text_cx = SIDEBAR_W // 2
    text_cy = (text_zone_top + text_zone_bottom) // 2

    # Подбираем размер шрифта: текст должен вписаться в text_zone_h (после поворота)
    font_size = 56
    font_title = _load_font(font_size)
    dummy = ImageDraw.Draw(canvas)
    for _ in range(10):
        bb = dummy.textbbox((0, 0), title_ru, font=font_title)
        if bb[2] - bb[0] <= text_zone_h - 16:
            break
        font_size = max(20, font_size - 4)
        font_title = _load_font(font_size)
    for _ in range(6):
        next_size = font_size + 4
        test_font = _load_font(next_size)
        bb = dummy.textbbox((0, 0), title_ru, font=test_font)
        if bb[2] - bb[0] > text_zone_h - 16:
            break
        font_size = next_size
        font_title = test_font

    _text_rotated(canvas, title_ru, font_title,
                  text_cx + 3, text_cy + 3, glow, angle=90)
    _text_rotated(canvas, title_ru, font_title,
                  text_cx, text_cy, (255, 255, 255), angle=90)

    # ── Штрихкод — верхняя 1/3 колонки, повёрнут вертикально ─────────────────
    bc_margin = 10
    bc_zone_top    = border + bc_margin
    bc_zone_bottom = div_y1 - bc_margin
    bc_zone_h = bc_zone_bottom - bc_zone_top   # высота зоны (вдоль карты)

    # Штрихкод занимает всю зону целиком
    bc_draw_w = SIDEBAR_W - 2 * bc_margin

    if bc_zone_h > 20:
        # Рисуем штрихкод горизонтально на временном слое, затем поворачиваем
        bc_tmp = Image.new("RGBA", (bc_zone_h, bc_draw_w), (0, 0, 0, 0))
        bc_draw_tmp = ImageDraw.Draw(bc_tmp)
        draw_barcode(bc_draw_tmp, title_ru,
                     0, 0, bc_tmp.width, bc_tmp.height, accent + (255,))
        bc_rotated = bc_tmp.rotate(90, expand=True)
        # Центрируем в колонке по X, прижимаем к верху зоны
        bc_paste_x = (SIDEBAR_W - bc_rotated.width) // 2
        bc_paste_y = bc_zone_top

        rgba_canvas = canvas.convert("RGBA")
        rgba_canvas.paste(bc_rotated, (bc_paste_x, bc_paste_y), bc_rotated)
        canvas.paste(rgba_canvas.convert("RGB"))
        draw = ImageDraw.Draw(canvas)  # обновляем draw после paste

    # Дополнительный глитч-штрих поверх разделителя (декор)
    _rect(draw, 6, div_y1 + 3, SIDEBAR_W - 6, div_y1 + 4, glow)

    return canvas


# ─── Сбор источников ──────────────────────────────────────────────────────────

def collect_sources(
    res_dir: Path,
    originals_dir: Path,
    card_filter: list[str] | None,
) -> list[tuple[Path, str]]:
    """
    Возвращает список (path, card_id) из обоих источников.
    card_id для originals = имя файла без расширения.
    """
    sources: list[tuple[Path, str]] = []

    # generated/res — PNG
    if res_dir.exists():
        for p in sorted(res_dir.glob("*.png")):
            sources.append((p, p.stem))

    # originals — WEBP
    if originals_dir.exists():
        for p in sorted(originals_dir.glob("*.webp")):
            sources.append((p, p.stem))

    if card_filter:
        f = set(card_filter)
        sources = [(p, cid) for p, cid in sources if cid in f]

    return sources


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Добавляет киберпанк-рамку и название к картам таро."
    )
    parser.add_argument("--res-dir",       type=Path, default=RES_DIR)
    parser.add_argument("--originals-dir", type=Path, default=ORIGINALS_DIR)
    parser.add_argument("--output-dir",    type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cards", nargs="*",
                        help="Список card_id (stem файла) для обработки")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    meta = load_meta([MISSING_META, ORIGINALS_META])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sources = collect_sources(args.res_dir, args.originals_dir, args.cards)
    if not sources:
        print("[!] Нет файлов для обработки.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(sources)} cards -> {args.output_dir}")
    print()

    ok = skipped = 0
    for img_path, card_id in sources:
        entry = meta.get(card_id)
        title_ru, suit = resolve_card_info(card_id, entry)

        if args.dry_run:
            print(f"  [DRY] {card_id:40s}  «{title_ru}»  [{suit or 'major'}]")
            ok += 1
            continue

        try:
            img = Image.open(img_path)
            result = add_border(img, title_ru, suit)
            out_path = args.output_dir / (card_id + ".png")
            result.save(str(out_path), format="PNG", optimize=False)
            print(f"  [OK] {card_id:40s}  «{title_ru}»  -> {out_path.name}")
            ok += 1
        except Exception as e:
            print(f"  [ERR] {card_id}: {e}", file=sys.stderr)
            skipped += 1

    print()
    print(f"Done: {ok} processed, {skipped} skipped.")


if __name__ == "__main__":
    main()
