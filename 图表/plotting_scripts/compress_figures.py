# -*- coding: utf-8 -*-
"""压缩 figures/ 用于聊天展示，并生成总览 contact sheet。

输出：
  figures_preview/figXX_*.png   单图压缩版（长边 <= 1400px）
  figures_preview/sheet_a.png   Fig 1-6 总览
  figures_preview/sheet_b.png   Fig 7-12 总览
"""

import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
PREVIEW_DIR = os.path.join(ROOT, "figures_preview")

LONG_EDGE = 1400
SHEET_COLS = 3
SHEET_CELL_W = 760
SHEET_CELL_H = 480
SHEET_PAD = 16
SHEET_TITLE_H = 40


def load_font(size):
    for name in ("Microsoft YaHei", "SimHei", "Arial"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def compress_all():
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(FIG_DIR) if f.endswith(".png"))
    for f in files:
        im = Image.open(os.path.join(FIG_DIR, f)).convert("RGB")
        w, h = im.size
        scale = min(1.0, LONG_EDGE / max(w, h))
        if scale < 1.0:
            im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        im.save(os.path.join(PREVIEW_DIR, f), "PNG", optimize=True)
        print(f"{f} -> {im.size}")


def make_sheet(names, title, out_name):
    font = load_font(24)
    cells = []
    for n in names:
        im = Image.open(os.path.join(PREVIEW_DIR, n)).convert("RGB")
        s = min(SHEET_CELL_W / im.width, SHEET_CELL_H / im.height)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
        cells.append(im)
    rows = (len(cells) + SHEET_COLS - 1) // SHEET_COLS
    w = SHEET_PAD * 2 + SHEET_COLS * SHEET_CELL_W
    h = SHEET_PAD * 2 + SHEET_TITLE_H + rows * SHEET_CELL_H
    sheet = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(sheet)
    d.text((SHEET_PAD, SHEET_PAD), title, fill="#263238", font=font)
    for i, im in enumerate(cells):
        r, c = divmod(i, SHEET_COLS)
        x = SHEET_PAD + c * SHEET_CELL_W + (SHEET_CELL_W - im.width) // 2
        y = SHEET_PAD + SHEET_TITLE_H + r * SHEET_CELL_H + (SHEET_CELL_H - im.height) // 2
        sheet.paste(im, (x, y))
    path = os.path.join(PREVIEW_DIR, out_name)
    sheet.save(path, "PNG", optimize=True)
    print(f"{out_name} {sheet.size} {os.path.getsize(path)} bytes")


def main():
    compress_all()
    files = sorted(f for f in os.listdir(PREVIEW_DIR) if f.startswith("fig"))
    make_sheet(files[:6], "Fig 1-6 总览（压缩预览）", "sheet_a.png")
    make_sheet(files[6:12], "Fig 7-12 总览（压缩预览）", "sheet_b.png")


if __name__ == "__main__":
    main()
