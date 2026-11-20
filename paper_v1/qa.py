"""Inspect the compiled paper and create small review contact sheets."""

import json
import re
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
QA = ROOT / "qa"
QA.mkdir(exist_ok=True)


def source_sheet():
    names = re.findall(r"\\paperfigure\{([^}]+)\}", "\n".join(
        p.read_text(encoding="utf-8") for p in ROOT.glob("*.tex")))
    sheet = Image.new("RGB", (1380, ((len(names) + 1) // 2) * 460), "white")
    draw = ImageDraw.Draw(sheet)
    sizes = {}
    for n, name in enumerate(names):
        path = ROOT.parent / "图表" / "figures" / name
        image = Image.open(path).convert("RGB")
        sizes[name] = image.size
        image.thumbnail((670, 420))
        x, y = n % 2 * 690, n // 2 * 460
        sheet.paste(image, (x + (690 - image.width) // 2, y + 25))
        draw.text((x + 8, y + 6), name, fill="black")
    sheet.save(QA / "source_contact.png")
    print(json.dumps(sizes, ensure_ascii=False))


def audit_pdf():
    document = fitz.open(ROOT / "main.pdf")
    thumbs = []
    pages = []
    for index, page in enumerate(document):
        text = page.get_text()
        pages.append({"page": index + 1, "chars": len(text),
                      "replacement_chars": text.count("\ufffd"),
                      "images": len(page.get_images()),
                      "head": text[:70].replace("\n", " ")})
        pix = page.get_pixmap(matrix=fitz.Matrix(0.43, 0.43), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        thumbs.append(image)
    cell_w, cell_h = 280, 390
    sheet = Image.new("RGB", (1400, ((len(thumbs) + 4) // 5) * cell_h), "#dedede")
    draw = ImageDraw.Draw(sheet)
    for index, image in enumerate(thumbs):
        x, y = index % 5 * cell_w, index // 5 * cell_h
        sheet.paste(image, (x + 10, y + 20))
        draw.text((x + 11, y + 3), str(index + 1), fill="black")
    sheet.save(QA / "paper_contact.png")
    text = "\n\f\n".join(p.get_text() for p in document)
    (QA / "paper_text.txt").write_text(text, encoding="utf-8")
    source = "\n".join(p.read_text(encoding="utf-8") for p in ROOT.glob("*.tex"))
    keys = set()
    for match in re.finditer(r"\\cite\{([^}]+)\}", source):
        keys.update(match[1].split(","))
    bibkeys = set(re.findall(r"@\w+\{([^,]+),", "\n".join(
        p.read_text(encoding="utf-8") for p in ROOT.glob("*.bib"))))
    report = {"pages": len(document), "page_details": pages,
              "cited_references": len(keys), "bibliography_entries": len(bibkeys),
              "unresolved_keys": sorted(keys - bibkeys),
              "uncited_entries": sorted(bibkeys - keys),
              "figures": len(re.findall(r"\\contentsline \{figure\}", (ROOT / "main.lof").read_text(encoding="utf-8"))),
              "tables": len(re.findall(r"\\contentsline \{table\}", (ROOT / "main.lot").read_text(encoding="utf-8")))}
    (QA / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys
    if "--sources" in sys.argv:
        source_sheet()
    else:
        audit_pdf()
