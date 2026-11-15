"""
diagnose_text_availability.py — 102 份 MPR PDF 的文本可得性诊断
============================================================
回答一个决定项目生死的问题：**哪些报告能直接抽出文本，哪些需要 OCR / 重新抓取？**

对每份 PDF 输出：
  size_kb, pages, n_image, text_ops(内容流里的 Tj/TJ 数), cjk(本仓库解析器抽到的中文字数)

判定：
  cjk >= 5000            → OK（可直接用）
  500 <= cjk < 5000      → PARTIAL（部分页可读，需人工抽查）
  cjk < 500 且 text_ops 多 → CID（字体编码问题，修解析器或换工具可救）
  cjk < 500 且 text_ops 少 → SCAN（疑似扫描件，需 OCR）

输出：docs/文本可得性诊断.md + data/processed/text_availability.csv
"""

from __future__ import annotations

import glob
import importlib.util
import os
import re
import sys

import pandas as pd

PDF_DIR = "data/raw/mpr"
OUT_MD = "docs/文本可得性诊断.md"
OUT_CSV = "data/processed/text_availability.csv"


def load_parser():
    spec = importlib.util.spec_from_file_location(
        "pdf_to_text", os.path.join(os.path.dirname(__file__), "pdf_to_text.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def count_text_ops(p2t, raw: bytes) -> int:
    """只统计"像内容流"的流里的文本绘制指令，排除内嵌字体程序。"""
    ops = 0
    for body in p2t.build_objects(raw).values():
        st = p2t.stream_bytes(body)
        if not st or st[:4] == b"\x00\x01\x00\x00" or b"glyf" in st[:2000]:
            continue
        ops += st.count(b"Tj") + st.count(b"TJ")
    return ops


def main() -> int:
    p2t = load_parser()
    rows = []
    for f in sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf"))):
        period = os.path.basename(f).replace(".pdf", "")
        raw = open(f, "rb").read()
        try:
            txt = p2t.pdf_text(raw)
        except Exception as e:  # noqa: BLE001
            txt = ""
            print(f"[警告] {period} 解析异常：{e}")
        cjk = sum(1 for ch in txt if "\u4e00" <= ch <= "\u9fff")
        ops = count_text_ops(p2t, raw)
        n_img = len(re.findall(rb"/Subtype\s*/Image", raw))
        pages = len(re.findall(rb"/Type\s*/Page[^s]", raw))
        if cjk >= 5000:
            verdict = "OK"
        elif cjk >= 500:
            verdict = "PARTIAL"
        elif ops >= 2000:
            verdict = "CID"
        else:
            verdict = "SCAN"
        rows.append({
            "period": period,
            "size_kb": round(len(raw) / 1024),
            "pages": pages,
            "n_image": n_img,
            "text_ops": ops,
            "cjk": cjk,
            "verdict": verdict,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    df["year"] = df["period"].str[:4]

    lines = []
    A = lines.append
    A("# MPR 文本可得性诊断")
    A("")
    A(f"> 对象：`data/raw/mpr/*.pdf` 共 **{len(df)}** 份")
    A("> 脚本：`src/data/diagnose_text_availability.py`")
    A("> 判定口径：抽到的中文字数 ≥5000 为 OK；500–5000 为 PARTIAL；")
    A("> 另有文字指令但抽不出字为 CID（字体编码问题）；文字指令也很少为 SCAN（疑扫描件）。")
    A("")
    A("## 总览")
    A("")
    vc = df["verdict"].value_counts()
    A("| 判定 | 份数 | 含义 |")
    A("|---|---|---|")
    A(f"| OK | {int(vc.get('OK', 0))} | 可直接抽取使用 |")
    A(f"| PARTIAL | {int(vc.get('PARTIAL', 0))} | 部分可读，需抽查 |")
    A(f"| CID | {int(vc.get('CID', 0))} | 有文字层但字体编码未解，换解析器可救 |")
    A(f"| SCAN | {int(vc.get('SCAN', 0))} | 疑似扫描件，需 OCR |")
    A("")
    A("## 按年份")
    A("")
    A("| 年份 | 份数 | OK | PARTIAL | CID | SCAN | 平均中文字数 |")
    A("|---|---|---|---|---|---|---|")
    for y, g in df.groupby("year"):
        gv = g["verdict"].value_counts()
        A(f"| {y} | {len(g)} | {int(gv.get('OK', 0))} | {int(gv.get('PARTIAL', 0))} | "
          f"{int(gv.get('CID', 0))} | {int(gv.get('SCAN', 0))} | {int(g['cjk'].mean())} |")
    A("")
    A("## 逐份明细")
    A("")
    A("| 期次 | 大小KB | 页数 | 图片数 | 文字指令 | 中文字数 | 判定 |")
    A("|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        A(f"| {r['period']} | {r['size_kb']} | {r['pages']} | {r['n_image']} | "
          f"{r['text_ops']} | {r['cjk']} | {r['verdict']} |")
    A("")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"诊断完成 -> {OUT_MD}")
    print(df["verdict"].value_counts().to_string())
    print(f"可用（OK）占比：{vc.get('OK', 0) / len(df):.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
