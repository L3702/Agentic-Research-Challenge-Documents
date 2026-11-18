"""
build_paper.py — 把 docs/ 下的 LaTeX 章节组装成一篇完整的 Word 论文
==================================================================
以 .tex 为唯一数据源，解析章节、段落、公式与表格，生成排版规范的中文论文稿：
    - A4，页边距 2.5cm，正文宋体小四、1.5 倍行距、首行缩进两字符
    - 一级标题黑体三号、二级标题黑体四号
    - 表格为三线表，表题在表上方，表注在表下方
    - 页脚居中页码

输出：docs/论文_完整稿.docx
"""

from __future__ import annotations

import os
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

SRC = "docs"
OUT = os.environ.get("PAPER_OUT", "docs/论文_完整稿.docx")

TITLE = "央行书面沟通影响收益还是波动？"
SUBTITLE = "来自 102 期《货币政策执行报告》文本情绪的证据"

# (所属章节, 锚定的表题片段, 图片, 图题, 图注)
FIGURES = [
    ("ch3_data.tex", "报告语料与分段统计", "docs/figures/fig1_tone_series.png",
     "图 3.1　政策指引段与宏观回顾段的情绪走势（2001Q1--2026Q2）",
     "阴影为 2008 年金融危机、2015 年股市异常波动与 2020 年疫情冲击；竖直虚线为 2014 年断点。"),
    ("ch5_empirical.tex", "政策指引段情绪与市场波动", "docs/figures/fig2_scatter.png",
     "图 5.1　政策指引段情绪与事件日市场波动",
     "散点为 98 期报告；实线为线性拟合，阴影为 95\\% 置信区间。"),
    ("ch5_empirical.tex", "政策指引段情绪与市场波动", "docs/figures/fig3_quantile.png",
     "图 5.2　按情绪三分位分组的市场反应",
     "误差棒为 95\\% 置信区间；括号内为各组样本量。"),
    ("ch5_empirical.tex", "制度断点检验", "docs/figures/fig4_break.png",
     "图 5.3　2014 年前后的系数对比",
     "点估计来自分期子样本回归，横线为 95\\% 置信区间。"),
    ("ch5_empirical.tex", "补充检验：危机期、安慰剂窗口与政策操作", "docs/figures/fig5_robust.png",
     "图 5.4　稳健性检验：不同设定下的系数",
     "左图为振幅方程的情绪系数，右图为断点交互项；横线为 95\\% 置信区间。"),
]


# --------------------------------------------------------------- 文本清洗
MATH_MAP = [
    (r"\\text\{([^}]*)\}", r"\1"),
    (r"\\alpha", "α"), (r"\\beta", "β"), (r"\\gamma", "γ"),
    (r"\\mu", "μ"), (r"\\lambda", "λ"), (r"\\varepsilon", "ε"),
    (r"\\sigma", "σ"), (r"\\times", "×"), (r"\\leq", "≤"), (r"\\geq", "≥"),
    (r"\\in", "∈"), (r"\\cdot", "·"), (r"\\%", "%"),
    (r"\\eqref\{[^}]*\}", "(式)"),
    (r"\\ref\{tab:desc\}", "表 5.1"), (r"\\ref\{tab:vol\}", "表 5.2"),
    (r"\\ref\{tab:break\}", "表 5.3"), (r"\\ref\{tab:breaksens\}", "表 5.4"),
    (r"\\ref\{tab:post14\}", "表 5.5"), (r"\\ref\{tab:robust\}", "表 5.6"),
    (r"\\ref\{tab:corpus\}", "表 3.1"), (r"\\ref\{tab:industry\}", "表 3.2"),
    (r"\\ref\{tab:var\}", "表 3.3"),
]


def clean(s: str) -> str:
    s = s.replace("~", " ")
    s = s.replace("--", "–")
    s = s.replace("``", "“").replace("''", "”")
    # 交叉引用 → 显式编号
    refs = {"tab:desc": "表 5.1", "tab:vol": "表 5.2", "tab:break": "表 5.3",
            "tab:breaksens": "表 5.4", "tab:post14": "表 5.5", "tab:robust": "表 5.6",
            "tab:corpus": "表 3.1", "tab:industry": "表 3.2", "tab:var": "表 3.3"}
    eqs = {"eq:twoway": "（4.1）", "eq:baseline": "（4.2）", "eq:break": "（4.3）"}
    # 先去掉“表~”“式~”这类重复前缀，再替换引用本身
    s = re.sub(r"表\s*~?\s*(?=\\ref\{tab:)", "", s)
    s = re.sub(r"式\s*~?\s*(?=\\eqref\{eq:)", "", s)
    for k, v in refs.items():
        s = s.replace("\\ref{" + k + "}", v)
    for k, v in eqs.items():
        s = s.replace("\\eqref{" + k + "}", v)
    s = re.sub(r"\\ref\{[^}]*\}", "", s)
    s = re.sub(r"\\eqref\{[^}]*\}", "", s)
    # 先处理公式，避免花括号被提前删掉
    s = re.sub(r"\$([^$]*)\$", lambda m: math_inline(m.group(1)), s)
    # 常见排版命令直接去掉
    for cmd in (r"\\label\{[^}]*\}", r"\\centering", r"\\small", r"\\footnotesize",
                r"\\noindent", r"\\item", r"\\bigskip", r"\\medskip", r"\\par"):
        s = re.sub(cmd, "", s)
    # 字体命令：保留内容
    for _ in range(3):
        s = re.sub(r"\\(textbf|emph|textit|text|mathrm)\{([^{}]*)\}", r"\2", s)
    # 分式
    for _ in range(2):
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    # 转义字符
    s = s.replace("\\_", "_").replace("\\%", "%").replace("\\&", "&")
    s = s.replace("\\ ", " ")
    # 剩余的反斜杠命令
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "")
    # 换行与空格：先并成单空格，再去掉中文字符之间的空格（否则两端对齐会把短行拉散）
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"[ \t\u3000]+", " ", s)
    CJK = r"[\u4e00-\u9fff，。、；：？！“”‘’（）《》〈〉—…·]"
    s = re.sub(rf"(?<={CJK}) (?={CJK})", "", s)
    return s.strip()


def math_inline(expr: str) -> str:
    e = expr
    e = re.sub(r"\\text\{([^}]*)\}", r"\1", e)
    e = e.replace("^{***}", "***").replace("^{**}", "**").replace("^{*}", "*")
    # 先把上下标的括号压平，否则分式的正则会被内层花括号挡住
    for _ in range(3):
        e = re.sub(r"_\{([^{}]*)\}", r"_\1", e)
        e = re.sub(r"\^\{([^{}]*)\}", r"^\1", e)
    # 分式：\frac{A}{B} -> (A)/(B)
    for _ in range(2):
        e = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", e)
    for a, b in [("\\alpha", "α"), ("\\beta", "β"), ("\\gamma", "γ"), ("\\mu", "μ"),
                 ("\\lambda", "λ"), ("\\varepsilon", "ε"), ("\\sigma", "σ"),
                 ("\\times", "×"), ("\\cdot", "·"), ("\\leq", "≤"), ("\\geq", "≥"),
                 ("\\in", "∈"), ("\\%", "%"), ("\\qquad", "  "), ("\\,", " "),
                 ("\\ ", " "), ("\\left", ""), ("\\right", "")]:
        e = e.replace(a, b)
    e = e.replace("\\frac", "")
    for a, b in (("I_t0", "I(t0)"), ("I_t1", "I(t1)"), ("I_t5", "I(t5)"),
                 ("I_t-1", "I(t-1)"), ("R^(0)", "|R0|"), ("t_-1", "t-1")):
        e = e.replace(a, b)
    e = re.sub(r"\\[a-zA-Z]+", "", e)
    e = e.replace("{", "").replace("}", "")
    e = re.sub(r"\s+", " ", e)
    return e.strip()


# --------------------------------------------------------------- 章节解析
def read(name: str) -> str:
    return open(os.path.join(SRC, name), encoding="utf-8").read()


def strip_comments(t: str) -> str:
    return "\n".join(ln for ln in t.split("\n") if not ln.lstrip().startswith("%"))


def parse_tables(chapter: str) -> dict[str, dict]:
    """提取该章所有 table 环境，返回 {label: {...}}。"""
    out = {}
    for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", chapter, re.S):
        blk = m.group(0)
        cap = re.search(r"\\caption\{([^}]*)\}", blk)
        lab = re.search(r"\\label\{([^}]*)\}", blk)
        tab = re.search(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", blk, re.S)
        notes = re.search(r"\\begin\{tablenotes\}\[?[^\]]*\]?(.*?)\\end\{tablenotes\}", blk, re.S)
        if not tab:
            continue
        ncol = len(re.findall(r"[lcr]", tab.group(1)))
        rows = []
        for raw in tab.group(2).split("\\\\"):
            line = raw.strip()
            if not line:
                continue
            # 先去掉横线命令，再判断是否为空（表头行常与 \toprule 同处一段）
            line = re.sub(r"\\(toprule|midrule|bottomrule|cmidrule)(\([^)]*\))?(\{[^}]*\})?", "", line)
            if not line.strip():
                continue
            mc = re.search(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*?)\}\s*&?\s*(.*)", line, re.S)
            if mc:
                span = int(mc.group(1))
                text = clean(mc.group(2) + (" " + mc.group(3) if mc.group(3) else ""))
                rows.append(("multi", span, text))
                continue
            cells = [clean(c) for c in line.split("&")]
            while len(cells) < ncol:
                cells.append("")
            rows.append(("row", ncol, cells[:ncol]))
        ntxt = ""
        if notes:
            ntxt = clean(re.sub(r"\\item", "", notes.group(1)))
        key = lab.group(1) if lab else f"noid{len(out)}"
        out[key] = {"caption": clean(cap.group(1)) if cap else "",
                    "ncol": ncol, "rows": rows, "notes": ntxt,
                    "index": len(out) + 1}
    return out


def parse_chapter(fn: str) -> list[dict]:
    """把一章解析成块序列：section / subsection / paragraph / equation / table。"""
    t = strip_comments(read(fn))
    tables = parse_tables(t)
    t = re.sub(r"\\begin\{table\}.*?\\end\{table\}", "@@TABLE@@", t, flags=re.S)

    blocks: list[dict] = []
    table_iter = iter(tables.items())

    def flush_para(buf: list[str]) -> None:
        txt = clean(" ".join(buf))
        if txt:
            blocks.append({"type": "para", "text": txt})

    buf: list[str] = []
    eqbuf: list[str] | None = None
    for line in t.split("\n"):
        s = line.strip()
        if s.startswith("\\begin{equation}"):
            flush_para(buf); buf = []
            eqbuf = []
            continue
        if s.startswith("\\end{equation}"):
            if eqbuf:
                blocks.append({"type": "eq", "text": math_inline(" ".join(eqbuf))})
            eqbuf = None
            continue
        if eqbuf is not None:
            if s and not s.startswith("\\label"):
                eqbuf.append(s)
            continue
        if s.startswith("\\section{"):
            flush_para(buf); buf = []
            blocks.append({"type": "section", "text": clean(re.search(r"\\section\{(.*)\}", s).group(1))})
            continue
        if s.startswith("\\subsection{"):
            flush_para(buf); buf = []
            blocks.append({"type": "subsection", "text": clean(re.search(r"\\subsection\{(.*)\}", s).group(1))})
            continue
        if s.startswith("\\paragraph{"):
            flush_para(buf); buf = []
            name = clean(re.search(r"\\paragraph\{(.*?)\}", s).group(1))
            rest = s[s.index("}") + 1:].strip()
            blocks.append({"type": "para", "text": name + "　" + clean(rest), "runin": True})
            continue
        if s == "@@TABLE@@":
            flush_para(buf); buf = []
            try:
                key, tb = next(table_iter)
                blocks.append({"type": "table", "data": tb})
            except StopIteration:
                pass
            continue
        if not s:
            flush_para(buf); buf = []
            continue
        buf.append(s)
    flush_para(buf)
    return blocks


# --------------------------------------------------------------- 排版工具
def set_font(run, cn: str = "宋体", en: str = "Times New Roman", size: float | None = None,
             bold: bool | None = None) -> None:
    run.font.name = en
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), en)
    rfonts.set(qn("w:hAnsi"), en)
    rfonts.set(qn("w:eastAsia"), cn)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold


def para(doc, text: str, *, size: float = 11.5, indent: bool = True,
         align=WD_ALIGN_PARAGRAPH.JUSTIFY, cn="宋体", space_after: float = 0,
         line: float = 1.35, bold: bool = False):
    p = doc.add_paragraph()
    p.paragraph_format.alignment = align
    p.paragraph_format.line_spacing = line
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    if indent:
        p.paragraph_format.first_line_indent = Pt(size * 2)
    run = p.add_run(text)
    set_font(run, cn=cn, size=size, bold=bold)
    return p


def heading(doc, text: str, level: int):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(12 if level == 1 else 9)
    pf.space_after = Pt(6 if level == 1 else 4)
    pf.line_spacing = 1.3
    pf.keep_with_next = True
    run = p.add_run(text)
    set_font(run, cn="黑体", size=14 if level == 1 else 12, bold=True)
    if level == 1:
        run.font.color.rgb = RGBColor(0, 0, 0)
    return p


def _border(el, edge: str, sz: int):
    tcPr = el
    borders = tcPr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tcPr.append(borders)
    e = OxmlElement(f"w:{edge}")
    e.set(qn("w:val"), "single")
    e.set(qn("w:sz"), str(sz))
    e.set(qn("w:space"), "0")
    e.set(qn("w:color"), "000000")
    borders.append(e)


def three_line_table(doc, data: dict, prefix: str):
    # 表题
    cap = doc.add_paragraph()
    cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(8)
    cap.paragraph_format.space_after = Pt(3)
    cap.paragraph_format.keep_with_next = True
    r = cap.add_run(f"{prefix}{data['caption']}")
    set_font(r, cn="黑体", size=9.5, bold=True)

    rows = [x for x in data["rows"]]
    ncol = data["ncol"]
    table = doc.add_table(rows=0, cols=ncol)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    header_done = False
    for kind, span, payload in rows:
        cells = table.add_row().cells
        if kind == "multi":
            merged = cells[0].merge(cells[min(span, ncol) - 1])
            r = merged.paragraphs[0].add_run(payload)
            set_font(r, cn="宋体", size=9.5, bold=True)
            merged.paragraphs[0].paragraph_format.space_before = Pt(1)
        else:
            header_done = header_done or False
            for i, txt in enumerate(payload):
                p = cells[i].paragraphs[0]
                p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.line_spacing = 1.0
                run = p.add_run(txt)
                set_font(run, cn="宋体", size=9.5, bold=not header_done)
            header_done = True

    # 让整张表尽量不跨页：除最后一行外，所有单元格段落设 keep_with_next
    for ri, row in enumerate(table.rows):
        if ri == len(table.rows) - 1:
            continue
        for cell in row.cells:
            for pp in cell.paragraphs:
                pp.paragraph_format.keep_with_next = True

    # 三线：外框上下粗线，表头下细线
    tbl = table._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge, sz in (("top", 12), ("bottom", 12), ("left", 0), ("right", 0),
                     ("insideH", 0), ("insideV", 0)):
        e = OxmlElement(f"w:{edge}")
        if sz == 0:
            e.set(qn("w:val"), "none")
        else:
            e.set(qn("w:val"), "single")
            e.set(qn("w:sz"), str(sz))
            e.set(qn("w:color"), "000000")
        borders.append(e)
    tblPr.append(borders)

    # 表头行下加细线
    if len(table.rows) > 0:
        for cell in table.rows[0].cells:
            tcPr = cell._tc.get_or_add_tcPr()
            b = OxmlElement("w:tcBorders")
            e = OxmlElement("w:bottom")
            e.set(qn("w:val"), "single")
            e.set(qn("w:sz"), "6")
            e.set(qn("w:color"), "000000")
            b.append(e)
            tcPr.append(b)

    # 固定布局，避免 LibreOffice/Word 重新均分列宽
    table.autofit = False
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tbl.tblPr.append(layout)

    # 按各列最长内容分配列宽（总宽 16cm）
    lens = []
    for ci in range(ncol):
        L = [len(r[2][ci]) for r in rows if r[0] == "row" and isinstance(r[2], list)]
        lens.append(max(4, max(L) if L else 4))
    tot = sum(lens)
    widths_cm = [16.0 * lens[ci] / tot for ci in range(ncol)]
    for ci in range(ncol):
        for cell in table.columns[ci].cells:
            cell.width = Cm(widths_cm[ci])
    # 同步 tblGrid，否则 LibreOffice 会按等宽重排
    grid = tbl.find(qn("w:tblGrid"))
    if grid is not None:
        for gc, wcm in zip(grid.findall(qn("w:gridCol")), widths_cm):
            gc.set(qn("w:w"), str(int(wcm * 567)))

    if data["notes"]:
        note = doc.add_paragraph()
        note.paragraph_format.space_before = Pt(2)
        note.paragraph_format.space_after = Pt(8)
        note.paragraph_format.line_spacing = 1.2
        r = note.add_run("注：" + data["notes"])
        set_font(r, cn="宋体", size=8.5)


def insert_figure(doc, img: str, caption: str, note: str) -> None:
    if not os.path.exists(img):
        print("  [跳过] 缺图", img)
        return
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    p.add_run().add_picture(img, width=Cm(14.6))

    cap = doc.add_paragraph()
    cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(2)
    r = cap.add_run(caption)
    set_font(r, cn="黑体", size=9.5, bold=True)

    if note:
        nt = doc.add_paragraph()
        nt.paragraph_format.space_after = Pt(8)
        nt.paragraph_format.line_spacing = 1.2
        r = nt.add_run("注：" + note)
        set_font(r, cn="宋体", size=8.5)


def add_page_number(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld1 = OxmlElement("w:fldChar"); fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve"); instr.text = "PAGE"
    fld2 = OxmlElement("w:fldChar"); fld2.set(qn("w:fldCharType"), "end")
    run._r.append(fld1); run._r.append(instr); run._r.append(fld2)
    set_font(run, size=9)


# --------------------------------------------------------------- 组装
def main() -> int:
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.top_margin = sec.bottom_margin = Cm(2.5)
    sec.left_margin = sec.right_margin = Cm(2.5)
    add_page_number(sec)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    # ---------- 标题 ----------
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(TITLE)
    set_font(r, cn="黑体", size=17, bold=True)

    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(14)
    r = p.add_run(SUBTITLE)
    set_font(r, cn="黑体", size=12)

    # ---------- 摘要 ----------
    ab = strip_comments(read("abstract.tex"))
    cn_abs = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", ab, re.S).group(1)
    head = doc.add_paragraph()
    head.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    head.paragraph_format.space_after = Pt(6)
    r = head.add_run("摘　要")
    set_font(r, cn="黑体", size=14, bold=True)

    for para_txt in [x for x in cn_abs.split("\n\n") if x.strip()]:
        para(doc, clean(para_txt.replace("\\noindent", "")), size=11.5, indent=True, space_after=0)

    kw = re.search(r"\\textbf\{关键词：\}(.*)", ab)
    jel = re.search(r"\\textbf\{JEL 分类号：\}(.*)", ab)
    if kw:
        para(doc, "关键词：" + clean(kw.group(1)), size=10.5, indent=False, space_after=2,
             align=WD_ALIGN_PARAGRAPH.LEFT)
    if jel:
        para(doc, "JEL 分类号：" + clean(jel.group(1)), size=10.5, indent=False,
             align=WD_ALIGN_PARAGRAPH.LEFT)

    # ---------- 英文摘要 ----------
    doc.add_page_break()
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run("Abstract")
    set_font(r, cn="Times New Roman", size=14, bold=True)
    en_abs = re.findall(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", ab, re.S)
    if len(en_abs) > 1:
        for x in [y for y in en_abs[1].split("\n\n") if y.strip()]:
            txt = clean(x.replace("\\noindent", ""))
            pr = doc.add_paragraph()
            pr.paragraph_format.line_spacing = 1.5
            pr.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            run = pr.add_run(txt)
            set_font(run, cn="Times New Roman", size=11)
    kw_en = re.search(r"\\textbf\{Keywords:\}(.*)", ab)
    if kw_en:
        pr = doc.add_paragraph()
        pr.paragraph_format.space_before = Pt(6)
        run = pr.add_run("Keywords: " + clean(kw_en.group(1)))
        set_font(run, cn="Times New Roman", size=10.5, bold=False)

    # ---------- 正文 ----------
    chapters = ["ch1_intro.tex", "ch2_literature.tex", "ch3_data.tex",
                "ch4_design.tex", "ch5_empirical.tex", "ch6_conclusion.tex"]
    table_no = {"ch3_data.tex": 3, "ch5_empirical.tex": 5}
    sec = 0
    for fn in chapters:
        sub = 0
        for blk in parse_chapter(fn):
            if blk["type"] == "section":
                sec += 1
                heading(doc, f"{sec}　{blk['text']}", 1)
            elif blk["type"] == "subsection":
                sub += 1
                heading(doc, f"{sec}.{sub}　{blk['text']}", 2)
            elif blk["type"] == "para":
                if blk.get("runin") and "　" in blk["text"]:
                    head_txt, body_txt = blk["text"].split("　", 1)
                    pp = doc.add_paragraph()
                    pp.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                    pp.paragraph_format.line_spacing = 1.35
                    pp.paragraph_format.space_before = Pt(4)
                    pp.paragraph_format.space_after = Pt(2)
                    pp.paragraph_format.first_line_indent = Pt(23)
                    r1 = pp.add_run(head_txt + "　")
                    set_font(r1, cn="黑体", size=11.5, bold=True)
                    r2 = pp.add_run(body_txt)
                    set_font(r2, cn="宋体", size=11.5)
                else:
                    para(doc, blk["text"], space_after=3)
            elif blk["type"] == "eq":
                pr = doc.add_paragraph()
                pr.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
                pr.paragraph_format.space_before = Pt(4)
                pr.paragraph_format.space_after = Pt(4)
                run = pr.add_run(blk["text"])
                set_font(run, cn="宋体", en="Cambria Math", size=11)
            elif blk["type"] == "table":
                n = table_no.get(fn, 0)
                prefix = f"表 {n}.{blk['data']['index']}　"
                three_line_table(doc, blk["data"], prefix)
                cap_txt = blk["data"]["caption"]
                for ch, anchor_txt, img, caption, note in FIGURES:
                    if ch == fn and anchor_txt in cap_txt:
                        insert_figure(doc, img, caption, note)

    # ---------- 参考文献 ----------
    doc.add_page_break()
    heading(doc, "参考文献", 1)
    refs = strip_comments(read("references.tex"))
    body = re.search(r"\\begin\{thebibliography\}\{\d+\}(.*?)\\end\{thebibliography\}",
                     refs, re.S).group(1)
    items = re.split(r"\\bibitem\{[^}]*\}", body)[1:]
    for i, it in enumerate(items, 1):
        txt = clean(it)
        pr = doc.add_paragraph()
        pr.paragraph_format.left_indent = Pt(24)
        pr.paragraph_format.first_line_indent = Pt(-24)
        pr.paragraph_format.line_spacing = 1.3
        pr.paragraph_format.space_after = Pt(3)
        pr.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        run = pr.add_run(f"[{i}]　{txt}")
        set_font(run, size=10.5)

    doc.save(OUT)
    print(f"[完成] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
