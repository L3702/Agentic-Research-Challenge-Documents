"""
build_sentiment.py — 从 102 期 MPR PDF 构建"报告级情绪序列（P/R 两段）"
=====================================================================
输入
    data/raw/mpr/*.pdf          102 期报告
    data/processed/mpr_meta.csv 期次与发布日
    --dict <词表>               姜富伟中文金融情感词典（xlsx / csv / txt）

输出
    data/processed/sentiment.csv
        period, pub_date, chars_R, chars_P,
        pos_R, neg_R, pos_P, neg_P,
        tone_R, tone_P, tone_all, z_tone_P, z_tone_R

    以及 docs/情绪序列诊断.md（逐期体检，用来判断哪几期抽坏了）

为什么要 P/R 两段
    姜富伟等（2021）发现：市场只对"政策指引"部分有反应，"宏观回顾"部分不显著
    （因为报告里的宏观数据发布前已公布）。所以情绪必须分段度量：
        R 段 = 回顾（报告开头到"货币政策趋势"之前）
        P 段 = 政策指引（"货币政策趋势"/"下一阶段主要政策思路"之后）
    R 段同时充当**文内安慰剂**：如果 R 段也显著，说明测的不是"新信息"。

文本抽取后端（自动选择，可用 --backend 指定）
    pypdf      —— 推荐，队友实测 102/102 可读
    pymupdf    —— 次选
    builtin    —— 本仓库自带的纯标准库解析器，只作兜底（2011–2018 部分期次会失败）

用法
    python src/data/build_sentiment.py --dict "路径/姜富伟词典.xlsx"
    python src/data/build_sentiment.py --dict 词典.xlsx --backend pypdf --limit 10   # 先试 10 期
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd

PDF_DIR = "data/raw/mpr"
TXT_DIR = "data/processed/mpr_text"
META = "data/processed/mpr_meta.csv"
OUT_CSV = "data/processed/sentiment.csv"
OUT_MD = "docs/情绪序列诊断.md"
DEFAULT_DICT = "data/external/mpr_sentiment_dict.csv"

# 否定词：出现在情感词前 3 个字符内则翻转极性
NEGATORS = ("不", "没", "无", "未", "非", "否", "别", "勿", "莫", "难以", "并未", "不再", "不会", "不能")

# P 段标题的候选写法（取最后出现的一次，避开目录）
P_HEAD_PATTERNS = [
    r"货币政策趋势",          # 2005 年后报告"第五部分"的标题
    r"预测与政策建议",        # 2001–2004 年报告
    r"下一阶段[^\n]{0,12}政策思路",
    r"下一阶段货币政策",
    r"政策建议",
]
# P 段之后若出现"附录/统计表"，截断
P_TAIL_PAT = r"\n\s*(附录|附表|统计表|专栏目录)"

# 句末标点（情感单元法：以句为单元）
SENT_SPLIT = re.compile(r"[。！？；!?;\n]+")


# ------------------------------------------------------------------ 文本抽取
def _load_builtin():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("pdf_to_text", os.path.join(here, "pdf_to_text.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def make_extractor(backend: str):
    """返回 pdf_path -> text 的函数。"""
    if backend in ("pypdf", "auto"):
        try:
            from pypdf import PdfReader

            def _pypdf(path: str) -> str:
                reader = PdfReader(path)
                return "\n".join((pg.extract_text() or "") for pg in reader.pages)

            if backend == "pypdf":
                return _pypdf, "pypdf"
            pypdf_fn = _pypdf
        except Exception:  # noqa: BLE001
            pypdf_fn = None
    else:
        pypdf_fn = None

    if backend in ("pymupdf", "auto"):
        try:
            import fitz  # PyMuPDF

            def _fitz(path: str) -> str:
                with fitz.open(path) as doc:
                    return "\n".join(p.get_text() for p in doc)

            if backend == "pymupdf":
                return _fitz, "pymupdf"
            fitz_fn = _fitz
        except Exception:  # noqa: BLE001
            fitz_fn = None
    else:
        fitz_fn = None

    if backend == "auto":
        if pypdf_fn:
            return pypdf_fn, "pypdf"
        if fitz_fn:
            return fitz_fn, "pymupdf"

    mod = _load_builtin()

    def _builtin(path: str) -> str:
        return mod.pdf_text(open(path, "rb").read())

    return _builtin, "builtin"


# ------------------------------------------------------------------ 文本清洗
CJK = r"\u4e00-\u9fff"


def clean_text(raw: str) -> str:
    t = raw.replace("\u3000", " ")
    t = re.sub(r"-\s*\n\s*", "", t)                       # 连字符断行
    t = re.sub(rf"(?<=[{CJK}])\s+(?=[{CJK}])", "", t)     # 中文字符间多余空格（2001–2003 常见）
    t = re.sub(r"\n{2,}", "\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


GOOD_PUNCT = set("，。、；：？！“”‘’（）《》〈〉【】—…·％％")


def _is_garbage_line(s: str) -> bool:
    """判断一行是否是乱码（解析器把字体二进制也当成文本时会出现）。"""
    if len(s) < 8:
        return False
    weird = 0
    for ch in s:
        if "\u4e00" <= ch <= "\u9fff":
            continue
        if ch in GOOD_PUNCT:
            continue
        if 32 <= ord(ch) < 127:
            continue
        weird += 1
    return weird / len(s) > 0.25


def strip_noise(t: str) -> str:
    """去页眉页脚、纯页码行与乱码行。"""
    lines = []
    for ln in t.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if _is_garbage_line(s):
            continue
        # 目录行：大量点线（"第一部分货币政策分析........"）
        if len(s) < 200 and (s.count(".") + s.count("…") + s.count("·")) > 5:
            continue
        if re.fullmatch(r"[-—\s]*\d{1,3}[-—\s]*", s):                      # 纯页码
            continue
        if re.fullmatch(r"(中国人民银行)?\s*货币政策执行报告\s*\d{0,4}", s):
            continue
        if re.fullmatch(r"\d{4}\s*年\s*(第[一二三四五六七八九十]+季度)?", s):
            continue
        lines.append(s)
    return "\n".join(lines)


# ------------------------------------------------------------------ P/R 分段
def split_pr(t: str) -> tuple[str, str, str, str]:
    """返回 (R 段, P 段, 分段状态)。

    为保证不同年份可比，按**优先级**尝试 P 段起点：
        1. 货币政策趋势（2005 年后"第五部分"的标题）
        2. 预测与政策建议（2001–2003 年报的写法）
        3. 下一阶段主要政策思路 / 下一阶段货币政策（兜底）

    每种模式取**最后一次**出现（避开目录与内容摘要）；若某模式切出的 P 段
    过短（<800 个汉字，说明命中了小标题而不是章节标题），则退到下一模式。
    """
    cands: list[tuple[str, str, str, int, str]] = []     # (状态, r, p, P汉字数, 锚点)
    for pat in P_HEAD_PATTERNS:
        hits = [m.start() for m in re.finditer(pat, t)]
        if not hits:
            continue
        best = hits[-1]
        p = t[best:]
        m = re.search(P_TAIL_PAT, p)
        if m:
            p = p[: m.start()]
        r = t[:best]
        n = sum(1 for ch in p if "\u4e00" <= ch <= "\u9fff")
        cands.append(("ok" if n >= 300 else "tail_only", r, p, n, pat))

    if not cands:
        return t, "", "no_heading", ""
    total = max(1, sum(1 for ch in t if "\u4e00" <= ch <= "\u9fff"))
    # P 段应占全篇 3%–60%；过短说明命中小标题，过长说明命中了目录/正文开头
    good = [c for c in cands if 0.02 * total <= c[3] <= 0.60 * total]
    if good:
        st, r, p, _, ap = good[0]                        # 优先级最高的合格候选
        return r, p, st, ap
    tgt = 0.25 * total
    st, r, p, _, ap = min(cands, key=lambda c: abs(c[3] - tgt))   # 都不合格：取最接近 25% 的
    return r, p, "check", ap


# ------------------------------------------------------------------ 词典
def load_dict(path: str) -> tuple[set[str], set[str]]:
    """返回 (积极词集合, 消极词集合)。支持 xlsx / csv / txt。"""
    pos: set[str] = set()
    neg: set[str] = set()

    def _add(target: set[str], words: Iterable) -> None:
        for w in words:
            if isinstance(w, str):
                w = w.strip()
                if 1 < len(w) <= 12:
                    target.add(w)

    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm"):
        import openpyxl

        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        for ws in wb.worksheets:
            name = ws.title
            is_pos = any(k in name for k in ("积", "正", "pos", "POS", "积极"))
            is_neg = any(k in name for k in ("消", "负", "neg", "NEG", "消极"))
            words = []
            for row in ws.iter_rows(values_only=True):
                for c in row:
                    if isinstance(c, str):
                        words.append(c)
            if is_pos:
                _add(pos, words)
            elif is_neg:
                _add(neg, words)
            else:  # 无法从表名判断：按列配对处理
                rows = list(ws.iter_rows(values_only=True))
                if rows and len(rows[0]) >= 2:
                    _add(pos, [r[0] for r in rows if r and isinstance(r[0], str)])
                    _add(neg, [r[1] for r in rows if r and len(r) > 1 and isinstance(r[1], str)])
    else:
        import csv as _csv

        with open(path, encoding="utf-8-sig", errors="ignore") as fh:
            rows = [r for r in _csv.reader(fh) if r and any(c.strip() for c in r)]
        # 格式一：word,polarity（1 / -1）
        pol_rows = [r for r in rows if len(r) >= 2 and r[1].strip() in ("1", "-1", "+1")]
        if len(pol_rows) >= 20:
            for r in pol_rows:
                w = r[0].strip()
                if 1 < len(w) <= 12:
                    (pos if r[1].strip() in ("1", "+1") else neg).add(w)
            return pos, neg
        # 格式二：带"积极/消极/正面/负面"表头的一列词
        lines = [c.strip() for r in rows for c in r if c.strip()]
        mode = None
        for ln in lines:
            if any(k in ln for k in ("积极", "正面", "positive", "Positive")):
                mode = pos
                continue
            if any(k in ln for k in ("消极", "负面", "negative", "Negative")):
                mode = neg
                continue
            if mode is not None:
                _add(mode, [ln])
        if not pos and not neg:          # 没有表头：前一半当积极，后一半当消极
            half = len(lines) // 2
            _add(pos, lines[:half])
            _add(neg, lines[half:])
    return pos, neg


# ------------------------------------------------------------------ 情绪打分
def score(text: str, pos: set[str], neg: set[str]) -> tuple[int, int]:
    """情感单元法：按句切分，逐词匹配，否定词翻转极性。"""
    n_pos = n_neg = 0
    for sent in SENT_SPLIT.split(text):
        if not sent:
            continue
        i = 0
        L = len(sent)
        while i < L:
            hit = None
            for w in (pos, neg):
                for size in (4, 3, 2):
                    cand = sent[i:i + size]
                    if len(cand) == size and cand in w:
                        hit = (cand, w is pos)
                        break
                if hit:
                    break
            if hit:
                word, is_pos = hit
                prefix = sent[max(0, i - 3):i]
                flipped = any(n in prefix for n in NEGATORS)
                if flipped:
                    is_pos = not is_pos
                n_pos += int(is_pos)
                n_neg += int(not is_pos)
                i += len(word)
            else:
                i += 1
    return n_pos, n_neg


def tone(pos: int, neg: int) -> float:
    tot = pos + neg
    return (pos - neg) / tot if tot else np.nan


# ------------------------------------------------------------------ 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", required=False, help="情感词典路径（xlsx/csv/txt）；不给则用内置 MPR 词典")
    ap.add_argument("--txt-dir", default=TXT_DIR, help="已有的 MPR 纯文本目录（优先于 PDF）")
    ap.add_argument("--backend", default="auto", choices=["auto", "pypdf", "pymupdf", "builtin"])
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 期（调试用）")
    ap.add_argument("--demo-dict", action="store_true", help="用内置演示小词典自测")
    args = ap.parse_args()

    if args.demo_dict:
        pos = set("增长 稳定 支持 宽松 适度 灵活 降低 改善 积极 向好 加大 扩大 增加 有利 缓解 回升 稳健".split())
        neg = set("风险 上升 上涨 紧缩 抑制 过热 泡沫 压力 困难 波动 违约 收紧 限制 减少 不利 恶化 下滑".split())
        print(f"[词典] 演示词典：积极 {len(pos)} / 消极 {len(neg)}")
    else:
        dict_path = args.dict if args.dict else (DEFAULT_DICT if os.path.exists(DEFAULT_DICT) else None)
        if not dict_path or not os.path.exists(dict_path):
            print("错误：未找到词典。请用 --dict 指定，或确认 "
                  f"{DEFAULT_DICT} 存在（可用 --demo-dict 自测）")
            return 1
        pos, neg = load_dict(dict_path)
        print(f"[词典] {dict_path} → 积极 {len(pos)} / 消极 {len(neg)}")
        if len(pos) < 100 or len(neg) < 100:
            print("[警告] 词典规模偏小，可能没读对表格结构，请检查")

    # ---------- 文本来源：优先用已抽取的 txt，其次才是 PDF ----------
    txt_files = sorted(glob.glob(os.path.join(args.txt_dir, "*.txt"))) if os.path.isdir(args.txt_dir) else []
    use_txt = len(txt_files) >= 5
    if use_txt:
        print(f"[文本] 使用已抽取的纯文本：{args.txt_dir}（{len(txt_files)} 份），跳过 PDF 解析")
        extract, backend_name = None, "txt"
    else:
        extract, backend_name = make_extractor(args.backend)
        print(f"[文本] 未找到 {args.txt_dir}，改用 PDF 解析，后端 = {backend_name}")

    meta = pd.read_csv(META)
    meta["period"] = meta["period"].astype(str)
    pmap = dict(zip(meta["period"], meta["pub_date"]))

    if use_txt:
        sources = [(os.path.basename(f).replace(".txt", ""), f) for f in txt_files]
    else:
        sources = [(os.path.basename(f).replace(".pdf", ""), f) for f in
                   sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))]
    if args.limit:
        sources = sources[: args.limit]

    rows = []
    for period, path in sources:
        try:
            if use_txt:
                try:
                    raw = open(path, encoding="utf-8").read()
                except UnicodeDecodeError:
                    raw = open(path, encoding="gbk", errors="ignore").read()
            else:
                raw = extract(path)
        except Exception as e:  # noqa: BLE001
            rows.append({"period": period, "status": f"extract_error:{e}", "chars": 0})
            continue
        t = strip_noise(clean_text(raw))
        cjk = sum(1 for ch in t if "\u4e00" <= ch <= "\u9fff")
        r, p, st, anchor = split_pr(t)
        npos_r, nneg_r = score(r, pos, neg)
        npos_p, nneg_p = score(p, pos, neg)
        rows.append({
            "period": period,
            "pub_date": pmap.get(period),
            "chars": cjk,
            "chars_R": sum(1 for ch in r if "\u4e00" <= ch <= "\u9fff"),
            "chars_P": sum(1 for ch in p if "\u4e00" <= ch <= "\u9fff"),
            "pos_R": npos_r, "neg_R": nneg_r,
            "pos_P": npos_p, "neg_P": nneg_p,
            "tone_R": tone(npos_r, nneg_r),
            "tone_P": tone(npos_p, nneg_p),
            "status": st,
            "anchor": anchor,
        })

    df = pd.DataFrame(rows)
    df["pos_all"] = df["pos_R"].fillna(0) + df["pos_P"].fillna(0)
    df["neg_all"] = df["neg_R"].fillna(0) + df["neg_P"].fillna(0)
    df["tone_all"] = (df["pos_all"] - df["neg_all"]) / (df["pos_all"] + df["neg_all"]).replace(0, np.nan)

    for col in ("tone_P", "tone_R", "tone_all"):
        s = df[col]
        df["z_" + col.replace("tone", "tone")] = (s - s.mean()) / s.std(ddof=0)

    ordered = ["period", "pub_date", "chars", "chars_R", "chars_P",
               "pos_R", "neg_R", "pos_P", "neg_P", "tone_R", "tone_P", "tone_all",
               "z_tone_R", "z_tone_P", "z_tone_all", "status", "anchor"]
    df = df[[c for c in ordered if c in df.columns]].sort_values("period")
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.6f")

    # ---------------- 诊断 ----------------
    bad = df[(df["status"] != "ok") | (df["chars"] < 8000) | (df["chars_P"] < 500)]
    lines = ["# 情绪序列诊断", "", f"- 后 端：{backend_name}", f"- 期数：{len(df)}",
             f"- 分段成功：{(df['status'] == 'ok').sum()}", f"- 需人工复核：{len(bad)}", ""]
    quant = df[["tone_R", "tone_P", "tone_all"]].describe().round(4)
    lines += ["## 情绪得分分布", "", "```", quant.to_string(), "```", ""]
    corr = df[["tone_R", "tone_P"]].corr().iloc[0, 1]
    lines += [f"R 段与 P 段情绪相关系数：**{corr:.3f}**（预期 0.2–0.6；过高说明两段没分开）", ""]
    if len(bad):
        lines += ["## 需人工复核的期次", "", "| 期次 | 字数 | R 段字数 | P 段字数 | 状态 |", "|---|---|---|---|---|"]
        for _, x in bad.iterrows():
            lines.append(f"| {x['period']} | {int(x['chars'])} | {int(x.get('chars_R', 0) or 0)} | "
                         f"{int(x.get('chars_P', 0) or 0)} | {x['status']} |")
        lines.append("")
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"[完成] 写出 {OUT_CSV}（{len(df)} 行）与 {OUT_MD}")
    print(f"       分段成功 {int((df['status'] == 'ok').sum())}/{len(df)}；"
          f"R-P 相关 {corr:.3f}")
    if len(bad):
        print("       需复核期次：" + ", ".join(bad["period"].head(12).tolist()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
