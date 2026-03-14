#!/usr/bin/env python3
"""
add_card_back_border.py — добавляет симметричные рамки и bleed к рубашке карты.

Две узкие колонки (левая + правая), идентичные по стилю колоды.
Левая колонка содержит пасхалки:
  - надпись «AdamCage cyberpunk2077-ai-tarot» вертикально, приглушённо
  - ссылка на проект чуть заметнее
Правая колонка — зеркальный декор без текста.

После добавления рамки накладывается bleed 2mm (33px по горизонтали, 32px по вертикали)
при пропорции карты 69x120mm — чёрная полоса по всем 4 сторонам для резки при печати.

Исходник:  data/generated/final/card_back.png      (960x1920)
С рамкой:                                          (1123x1948)
Результат: data/generated/card_back_to_print.png   (1189x2012)
"""

import io
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).parent
INPUT_PATH   = PROJECT_ROOT / "data" / "generated" / "final" / "card_back.png"
OUTPUT_PATH  = PROJECT_ROOT / "data" / "generated" / "card_back_to_print.png"
FONT_PATH    = PROJECT_ROOT / "data" / "RussoOne-Regular.ttf"

# Размер с рамкой совпадает с финальными картами колоды (cups_01.png и т.д.)
TARGET_W = 1123
TARGET_H = 1948

# Bleed 2mm при пропорции карты 69x120mm
CARD_W_MM = 69.0
CARD_H_MM = 120.0
BLEED_MM  = 2.0

# Цвет рубашки — нейтральное тёплое золото, не привязано к масти
ACCENT = (110,  85, 35)    # приглушённое золото (было 180,140,60 — слишком ярко)
GLOW   = ( 60,  45, 15)    # тёмное золото
DIM    = ( 18,  14,  4)    # почти чёрный с едва заметным золотым тоном

# Акцентные линии ещё тише — чтобы не отбирать внимание от арта
LINE_COLOR  = (90, 68, 28)    # линии рамки
CORNER_COLOR = (75, 56, 20)   # угловые маркеры


# ─── Шрифт ────────────────────────────────────────────────────────────────────

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for p in [FONT_PATH,
              Path(r"C:\Windows\Fonts\bahnschrift.ttf"),
              Path(r"C:\Windows\Fonts\arial.ttf")]:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                continue
    return ImageFont.load_default()


# ─── Примитивы ────────────────────────────────────────────────────────────────

def _rect(draw, x0, y0, x1, y1, fill):
    draw.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=fill)


def _draw_corner(draw, x, y, size, color, flip_x=False, flip_y=False):
    sx = -1 if flip_x else 1
    sy = -1 if flip_y else 1
    thick = max(2, size // 8)
    _rect(draw, x, y, x + sx * size, y + sy * thick, color)
    _rect(draw, x, y, x + sx * thick, y + sy * size, color)
    r = max(1, thick - 1)
    ex, ey = x + sx * size, y + sy * size
    draw.ellipse([ex - r, ey - r, ex + r, ey + r], fill=color)


def _glitch_sep(draw, x0, y, x1, color_a, color_b, segments=7):
    seg_w = (x1 - x0) / segments
    for i in range(segments):
        off = 1 if i % 2 == 0 else 0
        sx = x0 + int(i * seg_w)
        ex = x0 + int((i + 1) * seg_w) - 2
        c = color_a if i % 3 != 2 else color_b
        _rect(draw, sx, y + off, ex, y + off + 1, c)


def _text_rotated_alpha(canvas: Image.Image, text: str, font,
                         cx: int, cy: int, color: tuple, angle: int = 90,
                         alpha_mul: float = 1.0):
    """Рисует текст с поворотом и заданной прозрачностью alpha_mul ∈ [0,1]."""
    dummy = ImageDraw.Draw(canvas)
    bbox  = dummy.textbbox((0, 0), text, font=font)
    tw    = bbox[2] - bbox[0]
    th    = bbox[3] - bbox[1]

    tmp = Image.new("RGBA", (tw + 4, th + 4), (0, 0, 0, 0))
    td  = ImageDraw.Draw(tmp)
    # alpha задаётся как часть цвета
    a = int(255 * alpha_mul)
    td.text((2 - bbox[0], 2 - bbox[1]), text, font=font, fill=color + (a,))

    rotated = tmp.rotate(angle, expand=True)
    rw, rh  = rotated.size
    ox      = cx - rw // 2
    oy      = cy - rh // 2

    base = canvas.convert("RGBA")
    base.paste(rotated, (ox, oy), rotated)
    canvas.paste(base.convert("RGB"))


# ─── Декор одной боковой колонки ──────────────────────────────────────────────

def _draw_sidebar(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    col_x: int,          # левый край колонки
    col_w: int,          # ширина колонки
    total_h: int,
    border: int,
    flip: bool = False,  # True для правой колонки
    with_text: bool = False,
    name_text: str = "",
    url_text: str = "",
):
    """Рисует одну боковую колонку (левую или зеркальную правую)."""

    col_x1 = col_x + col_w   # правый край колонки

    # Фон колонки
    _rect(draw, col_x, 0, col_x1 - 1, total_h - 1, DIM)

    # Акцентная вертикальная полоса (внутренний край, примыкает к арту)
    inner_x = col_x1 - border if not flip else col_x
    _rect(draw, inner_x, 0, inner_x + border - 1, total_h, LINE_COLOR)

    # Горизонтальные полосы сверху и снизу
    _rect(draw, col_x, 0, col_x1, border - 1, LINE_COLOR)
    _rect(draw, col_x, total_h - border, col_x1, total_h - 1, LINE_COLOR)

    # Угловые маркеры
    if not flip:
        _draw_corner(draw, col_x + 2, 2, 16, CORNER_COLOR)
        _draw_corner(draw, col_x + 2, total_h - 2, 16, CORNER_COLOR, flip_y=True)
    else:
        _draw_corner(draw, col_x1 - 2, 2, 16, CORNER_COLOR, flip_x=True)
        _draw_corner(draw, col_x1 - 2, total_h - 2, 16, CORNER_COLOR, flip_x=True, flip_y=True)

    # Глитч-разделители
    div_y1 = total_h // 3
    div_y2 = total_h - total_h // 4
    pad = col_x + 4
    pad1 = col_x1 - 4
    _glitch_sep(draw, pad, div_y1, pad1, LINE_COLOR, GLOW, segments=5)
    _rect(draw, pad, div_y1 + 3, pad1, div_y1 + 4, GLOW)
    _glitch_sep(draw, pad, div_y2, pad1, GLOW, LINE_COLOR, segments=4)

    if not with_text:
        return

    cx = col_x + col_w // 2

    # ── Пасхалка 1: надпись «AdamCage cyberpunk2077-ai-tarot» ─────────────────
    # Зона — верхняя треть (над первым разделителем)
    name_zone_top    = border + 8
    name_zone_bottom = div_y1 - 8
    zone_h           = name_zone_bottom - name_zone_top
    cy_name          = name_zone_top + zone_h // 2

    font_name = _load_font(18)
    # Приглушённый цвет — виден при внимательном рассмотрении, но не бросается в глаза
    name_color = (72, 55, 20)   # темнее ACCENT, чуть теплее DIM
    _text_rotated_alpha(canvas, name_text, font_name,
                        cx, cy_name, name_color,
                        angle=90, alpha_mul=0.55)

    # ── Пасхалка 2: URL проекта — чуть заметнее ───────────────────────────────
    # Зона — нижние 2/3 (под первым разделителем)
    url_zone_top    = div_y1 + 14
    url_zone_bottom = div_y2 - 10
    zone_h2         = url_zone_bottom - url_zone_top
    cy_url          = url_zone_top + zone_h2 // 2

    font_url = _load_font(12)
    url_color = (55, 42, 15)    # чуть светлее DIM, но явно темнее name_color
    _text_rotated_alpha(canvas, url_text, font_url,
                        cx, cy_url, url_color,
                        angle=90, alpha_mul=0.75)


# ─── Основная функция ─────────────────────────────────────────────────────────

def add_back_border(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    art_w, art_h = img.size

    # Считаем отступы под целевой размер 1123x1948
    pad_t  = (TARGET_H - art_h) // 2        # 14 px
    pad_b  = TARGET_H - art_h - pad_t        # 14 px
    border = 2                               # толщина акцентной линии

    # Доступная ширина под две колонки = TARGET_W - art_w
    sides_total = TARGET_W - art_w           # 163 px
    col_w_left  = sides_total // 2           # 81 px
    col_w_right = sides_total - col_w_left   # 82 px

    total_w = TARGET_W
    total_h = TARGET_H

    canvas = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    draw   = ImageDraw.Draw(canvas)

    # ── Арт по центру ─────────────────────────────────────────────────────────
    art_x = col_w_left
    art_y = pad_t
    canvas.paste(img, (art_x, art_y))

    # Тонкая рамка вокруг арта (сверху и снизу)
    _rect(draw, art_x, art_y - border, art_x + art_w, art_y - 1, LINE_COLOR)
    _rect(draw, art_x, art_y + art_h + 1, art_x + art_w, art_y + art_h + border, LINE_COLOR)

    # Угловые маркеры арта
    cs = 20
    _draw_corner(draw, art_x, art_y, cs, CORNER_COLOR)
    _draw_corner(draw, art_x + art_w, art_y, cs, CORNER_COLOR, flip_x=True)
    _draw_corner(draw, art_x, art_y + art_h, cs, CORNER_COLOR, flip_y=True)
    _draw_corner(draw, art_x + art_w, art_y + art_h, cs, CORNER_COLOR, flip_x=True, flip_y=True)

    # ── Левая колонка — с пасхалками ──────────────────────────────────────────
    _draw_sidebar(
        canvas, draw,
        col_x=0, col_w=col_w_left,
        total_h=total_h, border=border,
        flip=False, with_text=True,
        name_text="AdamCage cyberpunk2077-ai-tarot",
        url_text="github.com/AdamCage/cyberpunk-ai-tarot",
    )

    # ── Правая колонка — зеркальный декор, без текста ─────────────────────────
    _draw_sidebar(
        canvas, draw,
        col_x=art_x + art_w, col_w=col_w_right,
        total_h=total_h, border=border,
        flip=True, with_text=False,
    )

    return canvas


# ─── Запуск ───────────────────────────────────────────────────────────────────

def main():
    if not INPUT_PATH.exists():
        print(f"[ERROR] Not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    img = Image.open(INPUT_PATH)
    print(f"Input:   {INPUT_PATH.name}  {img.size[0]}x{img.size[1]} px")

    bordered = add_back_border(img)
    print(f"Framed:  {bordered.size[0]}x{bordered.size[1]} px")

    # ── Bleed 2mm ─────────────────────────────────────────────────────────────
    bw, bh = bordered.size
    bleed_x = round(bw / CARD_W_MM * BLEED_MM)
    bleed_y = round(bh / CARD_H_MM * BLEED_MM)
    final_w = bw + 2 * bleed_x
    final_h = bh + 2 * bleed_y

    result = Image.new("RGB", (final_w, final_h), (0, 0, 0))
    result.paste(bordered, (bleed_x, bleed_y))
    print(f"Output:  {OUTPUT_PATH.name}  {final_w}x{final_h} px  "
          f"(bleed +{bleed_x}px left/right, +{bleed_y}px top/bottom)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(OUTPUT_PATH), format="PNG", optimize=False)
    print(f"Saved:   {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
