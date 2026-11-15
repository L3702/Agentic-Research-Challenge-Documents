"""
audit_strict.py — 数据审计（严格版）
====================================
目的：用可复现的代码，逐条核对「数据准备已完成」这一说法，
     并输出 docs/数据审计_自动检查.md。

运行：
    python src/data/audit_strict.py

设计原则：
    1. 只读，不改任何原始数据；
    2. 每一项检查给出 PASS / WARN / FAIL 三种状态；
    3. FAIL 必须附带「为什么这人命关天」，而不是只报错。

注意：本脚本兼容 pandas 3.x（不使用已被移除的 'Q' 频率别名）。
"""

from __future__ import annotations

import glob
import io
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_SW = os.path.join(ROOT, "data", "raw", "sw")
RAW_MPR = os.path.join(ROOT, "data", "raw", "mpr")
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "docs", "数据审计_自动检查.md")

# 申万一级行业代码 → 名称（2021 版口径）
# 仅填入高置信度映射；None 为旧口径代码，名称须从数据源核实。
# 本脚本不做猜测（科研诚信要求：不编造）。
SW_NAME = {
    "801010": "农林牧渔", "801020": "采掘", "801030": "基础化工", "801040": "钢铁",
    "801050": "有色金属", "801080": "电子", "801110": "家用电器", "801120": "食品饮料",
    "801130": "纺织服饰", "801140": "轻工制造", "801150": "医药生物", "801160": "公用事业",
    "801170": "交通运输", "801180": "房地产", "801200": "商贸零售", "801210": "社会服务",
    "801230": "综合", "801710": "建筑材料", "801720": "建筑装饰", "801730": "电力设备",
    "801740": "国防军工", "801750": "计算机", "801760": "传媒", "801770": "通信",
    "801780": "银行", "801790": "非银金融", "801880": "汽车", "801890": "机械设备",
    "801950": "煤炭", "801960": "石油石化", "801970": "环保", "801980": "美容护理",
    "801060": None, "801090": None, "801100": None, "801190": None, "801220": None,
}

buf = io.StringIO()
results: list[tuple[str, str, str]] = []  # (编号, 状态, 结论)


def w(line: str = "") -> None:
    buf.write(line + "\n")


def h(level: int, text: str) -> None:
    w()
    w("#" * level + " " + text)
    w()


def status(s: str) -> str:
    return {"PASS": "✅ PASS", "WARN": "⚠️ WARN", "FAIL": "❌ FAIL"}[s]


def record(code: str, st: str, msg: str) -> None:
    results.append((code, st, msg))


# ---------------------------------------------------------------- C0 文件清单
def check_files() -> dict:
    h(2, "C0 · 文件清单")
    pdfs = sorted(glob.glob(os.path.join(RAW_MPR, "*.pdf")))
    csvs = sorted(glob.glob(os.path.join(RAW_SW, "*.csv")))
    proc = sorted(glob.glob(os.path.join(PROC, "*")))
    texts = sorted(glob.glob(os.path.join(PROC, "text", "*")))
    w(f"- MPR PDF：**{len(pdfs)}** 份")
    w(f"- 行业日行情 CSV：**{len(csvs)}** 份")
    w("- processed 文件：" + (", ".join(os.path.basename(p) for p in proc) or "（空）"))
    w(f"- 文本语料（processed/text/）：**{len(texts)}** 个文件")
    w()

    if len(pdfs) == 102 and len(csvs) == 37:
        record("C0", "PASS", "原始数据齐备：102 份 PDF + 37 份行业行情")
    else:
        record("C0", "WARN", f"原始数据数量异常：PDF={len(pdfs)}, CSV={len(csvs)}")

    if len(texts) == 0:
        record("C1", "FAIL", "**核心解释变量不存在**：没有任何文本语料，情绪序列无法生成")
    else:
        record("C1", "PASS", f"已找到 {len(texts)} 个文本文件")
    return {"pdfs": pdfs, "csvs": csvs}


# ------------------------------------------------------- C2 行业口径分段问题
def check_vintage(rp: pd.DataFrame) -> None:
    h(2, "C2 · 行业口径：37 个行业是三代分类拼接")
    cov = rp.groupby("industry").agg(n=("ret", "size"), first=("qtr", "min"), last=("qtr", "max"))
    cov = cov.sort_values(["first", "last"])
    w("| 代码 | 名称 | 起始季度 | 终止季度 | 观测季度数 |")
    w("|---|---|---|---|---|")
    for code, r in cov.iterrows():
        nm = SW_NAME.get(code)
        w(f"| {code} | {nm if nm else '**待核实**'} | {r['first']} | {r['last']} | {r['n']} |")
    w()

    ended_early = cov[cov["last"] < pd.Period("2026Q3", "Q")]
    started_late = cov[cov["first"] > pd.Period("2000Q1", "Q")]
    alive = rp.groupby("qtr")["industry"].nunique()
    w(f"- 在 2026Q3 之前就终止的行业：**{len(ended_early)}** 个（{', '.join(ended_early.index)}）")
    w(f"- 在 2000Q1 之后才开始的行业：**{len(started_late)}** 个（{', '.join(started_late.index)}）")
    w(f"- 单个时点上真正存在的行业数：最少 **{int(alive.min())}**、最多 **{int(alive.max())}**"
      f"（不是 37；出现于 {alive.idxmin()} 与 {alive.idxmax()}）")
    w()
    w("> 含义：同一列 `industry=801060` 在 2014 年前后代表的是不同东西；")
    w("> 把 37 行当成固定截面，等于把三把尺子的刻度混在一起。")
    w()

    if len(ended_early) > 0 and len(started_late) > 0:
        record("C2", "FAIL",
               f"行业口径跨三代：{len(ended_early)} 个中途终止、{len(started_late)} 个中途开始，"
               "必须重建平衡面板")
    else:
        record("C2", "PASS", "行业覆盖一致")


# ------------------------------------------------------------ C3 平衡集选项
def check_balanced(rp: pd.DataFrame, mm: pd.DataFrame) -> None:
    h(2, "C3 · 可用的平衡样本（三选一）")
    cov = rp.groupby("industry").qtr.agg(["min", "max"])

    def bal(lo: str, hi: str) -> list[str]:
        lo_p, hi_p = pd.Period(lo, "Q"), pd.Period(hi, "Q")
        return sorted(i for i, r in cov.iterrows() if r["min"] <= lo_p and r["max"] >= hi_p)

    mm = mm.copy()
    mm["pub_q"] = pd.to_datetime(mm["pub_date"]).dt.to_period("Q")
    options = [("2001Q2", "2026Q3"), ("2014Q1", "2026Q3"), ("2021Q4", "2026Q3")]
    names = {"2001Q2": "长样本", "2014Q1": "中样本", "2021Q4": "短样本"}
    w("| 方案 | 窗口 | 行业数 | 季度数 | 面板观测 | 可用 MPR 事件 |")
    w("|---|---|---|---|---|---|")
    for lo, hi in options:
        inds = bal(lo, hi)
        qs = len(pd.period_range(lo, hi, freq="Q"))
        ev = int(((mm["pub_q"] >= pd.Period(lo, "Q")) & (mm["pub_q"] <= pd.Period(hi, "Q"))).sum())
        w(f"| {names[lo]} | {lo}–{hi} | {len(inds)} | {qs} | {len(inds) * qs} | {ev} |")
    w()
    w("> 三个方案都合法，但只能选一个。混用会得到「看起来有 2809 行、"
      "实际每期只有 16–31 个行业」的假面板。")
    w()
    record("C3", "PASS", "已列出三种平衡样本方案，需团队明确取舍（见 docs/01 第 12 节问题 1）")


# --------------------------------------------------- C4 空季度与不完整季度
def check_quarters(rp: pd.DataFrame) -> None:
    h(2, "C4 · 空季度被静默算成 0、末期季度不完整")
    zeros = rp[np.isclose(rp["ret"], 0.0)]
    zq = zeros["qtr"].value_counts().to_dict() if len(zeros) else {}
    w(f"- `ret` 恰好等于 0.0 的行：**{len(zeros)}** 行，分布："
      + (", ".join(f"{k}×{v}" for k, v in sorted(zq.items())) if zq else "无"))
    w("- 原因：`(1 + NaN).prod() - 1` 中 pandas 的 `prod` 默认跳过 NaN，"
      "空季度得到 `1 - 1 = 0`，因此「缺失率 0%」是 bug 造出来的。")
    w()

    sample = os.path.join(RAW_SW, "801030.csv")
    d = pd.read_csv(sample)
    dd = pd.to_datetime(d["日期"])
    last_q = rp["qtr"].max()
    in_q = dd[(dd >= last_q.start_time) & (dd <= last_q.end_time)]
    n_days = len(in_q)
    w(f"- 样本末期 `{last_q}`：覆盖 {in_q.min().date()} 至 {in_q.max().date()}，"
      f"仅含 **{n_days} 个交易日**（完整季度约 60 个）")
    w(f"- 全样本行情区间：{dd.min().date()} 至 {dd.max().date()}")
    w()

    bad = (len(zeros) > 0) or (n_days < 55)
    if bad:
        record("C4", "FAIL",
               f"存在 {len(zeros)} 行零收益 + 末期季度只有 {n_days} 个交易日，必须先清洗再建模")
    else:
        record("C4", "PASS", "季度结构正常")


# ------------------------------------------------- C5 quarter 列的排序可靠性
def check_quarter_dtype(rp: pd.DataFrame) -> None:
    h(2, "C5 · `quarter` 列是字符串，且无法正确排序")
    s = rp["quarter"].astype(str)
    w(f"- 样例值：`{s.iloc[0]}`、`{s.iloc[-1]}`")
    ok_format = bool(re.match(r"^\d{4}Q[1-4]$", s.iloc[0]))
    w(f"- 是否为规范的 `YYYYQn` 格式：**{'是' if ok_format else '否'}**")
    w(f"- 按字符串字典序排序的前 4 个值：{sorted(s.unique())[:4]}")
    if not ok_format:
        w("- 字典序会把 `2000/12/31` 排在 `2000/3/31` 之前（因为 1 < 3）。")
        w("  任何基于排序的滞后项、滚动窗口、时间趋势都会错位。")
    w()
    if not ok_format:
        record("C5", "FAIL", "quarter 为字符串时间戳，无法正确排序 → 滞后项会错位")
    else:
        record("C5", "PASS", "quarter 格式规范")


# ------------------------------------------------------ C6 MPR 发布日滞后
def check_mpr_timing(mm: pd.DataFrame, rp: pd.DataFrame) -> None:
    h(2, "C6 · MPR 发布时间：全部滞后一个季度")
    mm = mm.copy()
    mm["pub"] = pd.to_datetime(mm["pub_date"])
    mm["report_q"] = pd.PeriodIndex(mm["period"], freq="Q")
    mm["pub_q"] = mm["pub"].dt.to_period("Q")
    gap = pd.Series([(b - a).n for a, b in zip(mm["report_q"], mm["pub_q"])])
    dist = {int(k): int(v) for k, v in gap.value_counts().sort_index().items()}
    months = {int(k): int(v) for k, v in mm["pub"].dt.month.value_counts().sort_index().items()}
    w(f"- 报告期与发布季度的间隔分布：**{dist}**")
    w(f"- 发布月份分布：{months}")
    w(f"- 首个发布日：{mm['pub'].min().date()}；最后发布日：{mm['pub'].max().date()}")
    w()
    w("> 含义：2026Q1 的报告 2026-05-11 才发布。若把「2026Q1 报告情绪」和")
    w("> 「2026Q1 行业收益」放同一行，就是用 5 月的信息解释 1–3 月的收益（前视偏差）。")
    w("> 正确做法：情绪对齐到发布后的收益窗口。")
    w()
    if int((gap == 1).sum()) == len(gap):
        record("C6", "FAIL", "102/102 期报告均滞后一期发布，但现有面板未做任何时间对齐")
    else:
        record("C6", "PASS", "发布时序正常")


# ------------------------------------------------- C7 事件窗口的交易日检查
def check_event_windows(mm: pd.DataFrame, csvs: list[str]) -> None:
    h(2, "C7 · 事件窗口：用自然日会撞上春节 / 国庆")
    calendar: set = set()
    for f in csvs:
        try:
            calendar |= set(pd.to_datetime(pd.read_csv(f, usecols=["日期"])["日期"]))
        except Exception:
            continue
    cal = pd.Series(sorted(calendar))
    w(f"- 交易日历（从行业行情合并）：{len(cal)} 个交易日，"
      f"{cal.iloc[0].date()} 至 {cal.iloc[-1].date()}")
    w()

    rows, bad = [], []
    nat_bad = []
    for _, r in mm.iterrows():
        t0c = pd.Timestamp(r["pub_date"])
        nat = int(((cal >= t0c) & (cal <= t0c + pd.Timedelta(days=7))).sum())
        if nat < 4:
            nat_bad.append((r["period"], nat))
        later = cal[cal >= t0c]
        if not len(later):
            rows.append((r["period"], t0c.date(), None, None, 0))
            bad.append(r["period"])
            continue
        t0 = later.iloc[0]
        idx = int(cal[cal == t0].index[0])
        t5 = cal.iloc[idx + 5] if idx + 5 < len(cal) else pd.NaT
        n = 6 if pd.notna(t5) else 0
        rows.append((r["period"], t0c.date(), t0.date(), t5.date() if pd.notna(t5) else None, n))
        if n < 4:
            bad.append(r["period"])

    w("| 报告期 | 发布日 | t0（首个交易日） | t+5 | 窗口交易日数 |")
    w("|---|---|---|---|---|")
    for p_, pub, t0, t5, n in rows[-8:]:
        w(f"| {p_} | {pub} | {t0} | {t5} | {n} |")
    w(f"（仅展示最后 8 期；共 {len(rows)} 期）")
    w()
    w(f"- 用交易日历推进窗口，交易日 <4 的事件：**{bad if bad else '无'}**")
    w(f"- 若错误地用「发布日 + 7 自然日」：交易日 <4 的事件有 **{len(nat_bad)}** 个 "
      f"→ {nat_bad if nat_bad else '无'}")
    if nat_bad:
        w("  > 典型原因：春节休市（如 2016-02-08 前后）、国庆长假。用自然日会得到")
        w("  > 「窗口里没有价格」的空 CAR：程序不报错，但结果是错的。")
    w()
    if bad:
        record("C7", "WARN", f"{len(bad)} 个事件用交易日历推进仍不足 4 个交易日：{bad}")
    else:
        record("C7", "PASS",
               f"交易日历下全部事件窗口合格；若用自然日 +7 天，{len(nat_bad)} 个事件会踩空")


# ------------------------------------------------ C8 单日极端跳变（口径断裂）
def check_jumps(csvs: list[str]) -> None:
    h(2, "C8 · 极端季度收益：是真实行情还是换尺子？")
    w("检验思路：若某季度 +110% 来自指数成分或基准调整，通常表现为单日跳变；")
    w("若是真实行情，则分散在多日。阈值取单日绝对收益 > 20%。")
    w()
    w("| 代码 | 单日最大涨跌幅 | 出现日期 | 该季度累计 |")
    w("|---|---|---|---|")
    n_flag = 0
    for code, qend_s in [("801020", "2007-09-30"), ("801790", "2014-12-31")]:
        f = os.path.join(RAW_SW, f"{code}.csv")
        d = pd.read_csv(f)
        d["日期"] = pd.to_datetime(d["日期"])
        d = d.sort_values("日期").reset_index(drop=True)
        d["r"] = d["收盘"].pct_change()
        qend = pd.Timestamp(qend_s)
        qstart = qend - pd.DateOffset(months=3) + pd.Timedelta(days=1)
        sub = d[(d["日期"] >= qstart) & (d["日期"] <= qend)]
        rr = sub["r"].abs()
        i = rr.idxmax()
        cum = (1 + sub["r"].fillna(0)).prod() - 1
        w(f"| {code} | {sub.loc[i, 'r']:+.2%} | {sub.loc[i, '日期'].date()} | {cum:+.2%} |")
        if abs(sub.loc[i, "r"]) > 0.20:
            n_flag += 1

    allmax = []
    for f in csvs:
        d = pd.read_csv(f)
        t = pd.DataFrame({"d": pd.to_datetime(d["日期"]), "c": d["收盘"]}).sort_values("d")
        allmax.append(float(t["c"].pct_change().abs().max()))
    w(f"\n- 全部 37 个行业的最大单日绝对涨跌：**{max(allmax):.2%}**；"
      f"超过 20% 的行业数：**{sum(x > 0.2 for x in allmax)}**")
    w()
    if n_flag == 0:
        w("> 结论：两个极端季度都是多日累积形成，没有单日跳变，")
        w("> 因此「真实历史行情」的判断可以接受。但请把这个检验写进论文附录。")
        record("C8", "PASS", "极端值经单日拆解，未发现口径断裂痕迹")
    else:
        record("C8", "WARN", "存在单日跳变，需核查指数基准调整")


# ---------------------------------------------------------- C9 超额收益基准
def check_excess() -> None:
    h(2, "C9 · 超额收益的基准随样本构成变化")
    re_ = pd.read_csv(os.path.join(PROC, "ret_ex.csv"))
    re_["industry"] = re_["industry"].astype(str).str.zfill(6)
    s = re_.groupby("quarter")["ret_ex"].sum()
    n_ind = re_.groupby("quarter")["industry"].nunique()
    w(f"- 每季度 `ret_ex` 之和的绝对值最大值：**{s.abs().max():.2e}**（恒等于 0，属机械约束）")
    w(f"- 每季度参与计算的行业数：最小 **{n_ind.min()}**、最大 **{n_ind.max()}**")
    w()
    w("> 基准 = 当期样本内所有行业的等权平均。参与计算的行业数在变化，")
    w("> 基准本身在变，超额收益的「超额」含义就不稳定。")
    w("> 建议：改用固定篮子（如始终用同一组行业）或外部宽基指数（中证全指 / 沪深300）。")
    w()
    if n_ind.nunique() > 1:
        record("C9", "FAIL",
               f"基准组合随季度变化（{n_ind.min()}–{n_ind.max()} 个行业），超额收益口径不稳定")
    else:
        record("C9", "PASS", "基准组合固定")


# ---------------------------------------------------- C10 方差分解（识别空间）
def check_variance_share(rp: pd.DataFrame) -> None:
    h(2, "C10 · 识别空间：季度固定效应吃掉了多少方差？")
    cov = rp.groupby("industry").qtr.agg(["min", "max"])
    inds = [i for i, r in cov.iterrows()
            if r["min"] <= pd.Period("2014Q1", "Q") and r["max"] >= pd.Period("2026Q3", "Q")]
    s = rp[(rp["qtr"] >= pd.Period("2014Q1", "Q"))
           & (rp["qtr"] <= pd.Period("2026Q3", "Q"))
           & (rp["industry"].isin(inds))].copy()
    s["ret_ex"] = s["ret"] - s.groupby("qtr")["ret"].transform("mean")
    qd = pd.get_dummies(s["qtr"], drop_first=True).astype(float)
    X = np.column_stack([np.ones(len(s)), qd.values])
    y = s["ret"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - resid.var() / y.var()
    w(f"- 干净子样本：{len(inds)} 个行业 × {s['qtr'].nunique()} 个季度 = **{len(s)}** 个观测")
    w(f"- 季度固定效应 R² = **{r2:.3f}** → 横截面可解释部分只剩 **{1 - r2:.3f}**")
    w(f"- 行业超额收益标准差 = **{s['ret_ex'].std():.4f}**"
      f"（约 {s['ret_ex'].std() * 100:.1f} 个百分点 / 季度）")
    w()
    w("> 含义：MPR 情绪每季度只有一个数，与季度固定效应完全共线。")
    w("> 一旦加入季度 FE，情绪主效应会被标记为 omitted。")
    w("> 因此识别只能来自「情绪 × 行业特征」的交互项——这是路线 A 的核心。")
    w()
    record("C10", "PASS", f"季度 FE 吸收 {r2:.0%} 方差，确认必须用交互项识别")


# ---------------------------------------------------------- C11 脚本可复现性
def check_reproducible() -> None:
    h(2, "C11 · 提交的代码能否复现交付的数据？")
    p = os.path.join(ROOT, "src", "data", "build_returns.py")
    src = open(p, encoding="utf-8").read()
    problems = []
    if "resample('Q')" in src or 'resample("Q")' in src:
        problems.append("使用 `resample('Q')`，pandas 3.x 已移除该别名（会直接报错）")
    if "%YQ%q" in src:
        problems.append("使用 `strftime('%YQ%q')`，`%q` 不是合法指令，无法产出 `2020Q1` 格式")
    for x in problems:
        w(f"- ❌ {x}")
    w()
    w("> 因此 `ret_panel.csv` 不是由当前提交的脚本生成的（CSV 里 quarter 是")
    w("> `2000/3/31 0:00` 这类时间戳，而脚本产不出这种格式）。")
    w("> 这在复现审查中是硬伤：代码与数据不对应。")
    w()
    if problems:
        record("C11", "FAIL", "build_returns.py 在当前环境无法运行，且与交付 CSV 格式不符")
    else:
        record("C11", "PASS", "脚本可运行")


# ------------------------------------------------------------------ 主流程
def main() -> None:
    w("# 数据审计（自动检查）")
    w()
    w(f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M}")
    w("- 生成脚本：`src/data/audit_strict.py`")
    w("- 结论标记：✅ PASS / ⚠️ WARN / ❌ FAIL")
    w()
    w("---")
    w()

    info = check_files()
    csvs = info["csvs"]

    rp = pd.read_csv(os.path.join(PROC, "ret_panel.csv"))
    rp["industry"] = rp["industry"].astype(str).str.zfill(6)
    rp["qtr"] = pd.to_datetime(rp["quarter"]).dt.to_period("Q")
    mm = pd.read_csv(os.path.join(PROC, "mpr_meta.csv"))

    check_vintage(rp)
    check_balanced(rp, mm)
    check_quarters(rp)
    check_quarter_dtype(rp)
    check_mpr_timing(mm, rp)
    check_event_windows(mm, csvs)
    check_jumps(csvs)
    check_excess()
    check_variance_share(rp)
    check_reproducible()

    summary = io.StringIO()
    summary.write("## 结论总览\n\n| 编号 | 状态 | 结论 |\n|---|---|---|\n")
    for code, st, msg in results:
        summary.write(f"| {code} | {status(st)} | {msg} |\n")
    summary.write("\n---\n\n")

    body = buf.getvalue()
    head, sep, rest = body.partition("---\n\n")
    final = head + sep + summary.getvalue() + rest

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(final)

    n_fail = sum(1 for _, st, _ in results if st == "FAIL")
    n_warn = sum(1 for _, st, _ in results if st == "WARN")
    print(f"审计完成 -> {OUT}")
    print(f"FAIL={n_fail}  WARN={n_warn}  PASS={len(results) - n_fail - n_warn}")
    if n_fail:
        print("\n必须先处理的项：")
        for code, st, msg in results:
            if st == "FAIL":
                print(f"  [{code}] {msg}")


if __name__ == "__main__":
    sys.exit(main())
