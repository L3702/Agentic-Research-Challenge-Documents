"""Inspect the compiled paper and create small review contact sheets."""

import json
import re
import sys
from hashlib import sha256
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
QA = ROOT / "qa"
QA.mkdir(exist_ok=True)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def source_sheet():
    names = [item["output"] for item in figure_manifest()]
    sheet = Image.new("RGB", (1380, ((len(names) + 1) // 2) * 460), "white")
    draw = ImageDraw.Draw(sheet)
    sizes = {}
    for n, name in enumerate(names):
        path = ROOT / name
        if path.suffix == ".pdf":
            with fitz.open(path) as figure:
                pix = figure[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        else:
            with Image.open(path) as raster:
                image = raster.convert("RGB")
        sizes[name] = image.size
        image.thumbnail((670, 420))
        x, y = n % 2 * 690, n // 2 * 460
        sheet.paste(image, (x + (690 - image.width) // 2, y + 25))
        draw.text((x + 8, y + 6), name, fill="black")
    sheet.save(QA / "source_contact.png")
    print(json.dumps(sizes, ensure_ascii=False))


def figure_manifest():
    return json.loads((ROOT / "figures/manifest.json").read_text(encoding="utf-8"))


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
    labels = re.findall(r"\\label\{([^}]+)\}", source)
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", source))
    # Figure macros pass the label as their third argument.
    labels += re.findall(r"\\paperfigure\{[^}]+\}\{[^}]+\}\{([^}]+)\}", source)
    labels = [label for label in labels if label != "#3"]
    manifest = figure_manifest()
    bad_inputs = []
    for item in manifest:
        path = ROOT.parent / item["input"]
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            bad_inputs.append(item["input"])
    evidence_paths = [
        "experiments/E1/validation/human_ratings.csv",
        "experiments/E1/sentiment_dictionary_full.csv",
        "experiments/E3/e3_car_by_sentiment.csv",
        "experiments/E4/e4_oos_metrics.csv",
        "experiments/E6/e6_llm_oos_metrics.csv",
        "experiments/E8/e8_robustness_summary.csv",
    ]
    report = {"pages": len(document), "page_details": pages,
              "cited_references": len(keys), "bibliography_entries": len(bibkeys),
              "unresolved_keys": sorted(keys - bibkeys),
              "uncited_entries": sorted(bibkeys - keys),
              "unresolved_labels": sorted(refs - set(labels)),
              "unreferenced_float_or_equation_labels": sorted(
                  label for label in set(labels) - refs if label.startswith(("tab:", "fig:", "eq:"))),
              "duplicate_labels": sorted({label for label in labels if labels.count(label) > 1}),
              "figure_manifest_entries": len(manifest),
              "evidence_inputs_sha256": {
                  name: sha256((ROOT.parent / name).read_bytes()).hexdigest()
                  for name in evidence_paths},
              "figure_input_mismatches": bad_inputs,
              "missing_figure_outputs": [item["output"] for item in manifest
                                         if not (ROOT / item["output"]).is_file()],
              "figures": len(re.findall(r"\\contentsline \{figure\}", (ROOT / "main.lof").read_text(encoding="utf-8"))),
              "tables": len(re.findall(r"\\contentsline \{table\}", (ROOT / "main.lot").read_text(encoding="utf-8")))}
    (QA / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    failures = ("unresolved_keys", "uncited_entries", "unresolved_labels", "duplicate_labels",
                "unreferenced_float_or_equation_labels", "figure_input_mismatches",
                "missing_figure_outputs")
    if any(report[key] for key in failures):
        raise SystemExit(1)


if __name__ == "__main__":
    if "--sources" in sys.argv:
        source_sheet()
    else:
        audit_pdf()
