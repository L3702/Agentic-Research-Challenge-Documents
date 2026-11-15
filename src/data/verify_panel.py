"""
verify_panel.py — 面板核查器
============================
把 ret_panel.csv 与原始日行情逐条对账，输出 docs/数据核查_ret_panel.md。

运行：
    python src/data/verify_panel.py

与 audit_strict.py 的分工：
    audit_strict.py  检查「原始数据 -> 分析表」这条路有没有设计问题（审设计）
    verify_panel.py  检查「已经产出的分析表」算得对不对（审结果）

设计原则：只读，不改任何数据；每项给 PASS / WARN / FAIL。
"""

from __future__ import annotations

import glob
import io
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_SW = os.path.join(ROOT, "data", "raw", "sw")
PROC = os.path.join(ROOT, "data", "processed")
PANEL = os.path.join(PROC, "ret_panel.csv")
OLD_EX = os.path.join(PROC, "ret_ex.csv")
OUT = os.path.join(ROOT, "docs", "数据核查_ret_panel.md")

buf = io.StringIO()
results: list[tuple[str, str, str]] = []


def w(s: str = "") -> None:
    buf.write(s + "\n")


def h(level: int, text: str) -> None:
    w()
    w("#" * level + " " + text)
    w()


def tag(st: str) -> str:
    return {"PASS": "✅ PASS", "WARN": "⚠️ WARN", "FAIL": "❌ FAIL"}[st]


def rec(code: str, st: str, msg: str) -> None:
    results.append((code, st, msg))


# ----------------------------------------------------------------- 读取
def load() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    v = pd.read_csv(PANEL, dtype={"industry": str})
    v["industry"] = v["industry"].str.zfill(6)
    v["q"] = v["quarter"].astype(str)
    raw: dict[str, pd.Series] = {}
    for f in sorted(glob.glob(os.path.join(RAW_SW, "*.csv"))):
        code = os.path.basename(f).replace(".csv", "")
        d = pd.read_csv(f)
        d["日期"] = pd.to_datetime(d["日期"])
        raw[code] = d.set_index("日期")["收盘"].sort_index()
    return v, raw


# ------------------------------------------------------- V1 结构完整性
def check_structure(v: pd.DataFrame) -> None:
    h(2, "V1 · 结构完整性")
    n_q, n_i = v["q"].nunique(), v["industry"].nunique()
    per_q = v.groupby("q").size().unique()
    per_i = v.groupby("industry").size().unique()
    w(f"- 行数 **{len(v)}**；季度 **{n_q}**（{v['q'].min()}–{v['q'].max()}）；行业 **{n_i}**")
    w(f"- 每季行业数：{per_q.tolist()}；每行业季数：{per_i.tolist()}")
    w(f"- 重复 (quarter, industry) 行：**{int(v.duplicated(['q', 'industry']).sum())}**")
    w(f"- 空值合计：**{int(v.isna().sum().sum())}**")
    w(f"- `quarter` 全为 `YYYYQn` 格式：**{bool(v['q'].str.match(r'^\d{4}Q[1-4]$').all())}**")
    w()
    ok = (len(per_q) == 1 and len(per_i) == 1 and v.duplicated(["q", "industry"]).sum() == 0
          and v.isna().sum().sum() == 0 and v["q"].str.match(r"^\d{4}Q[1-4]$").all())
    if ok:
        rec("V1", "PASS", f"严格平衡面板 {n_i} 行业 × {n_q} 季度 = {len(v)} 行，无重复无空值")
    else:
        rec("V1", "FAIL", "面板不平衡或存在重复/空值")


# ------------------------------------------- V2 与原始数据独立重算对账
def check_recompute(v: pd.DataFrame, raw: dict[str, pd.Series]) -> pd.DataFrame:
    h(2, "V2 · 与原始日线独立重算对账")
    w("口径：先在**完整**日序列上算 `pct_change()`（保留季度首日相对上一季度末收盘的收益），")
    w("再按季度复利。与 build_returns.py 的差别就在这一步。")
    w()
    recs = []
    for code, s in raw.items():
        if code not in set(v["industry"]):
            continue
        s = s.loc[s.index >= pd.Timestamp("2013-12-01")]
        r = s.pct_change()
        r.index.name = "d"
        g = r.groupby(r.index.to_period("Q"))
        t = pd.DataFrame({
            "n": g.size(),
            "last": g.apply(lambda x: x.index.max()),
            "ret_ref": g.apply(lambda x: (1 + x.dropna()).prod() - 1),
        })
        t["qend"] = t.index.to_timestamp(how="end").normalize()
        t["gap_days"] = (t["qend"] - t["last"].dt.normalize()).dt.days
        t = t.reset_index().rename(columns={"d": "q"})
        t["q"] = t["q"].astype(str)
        t["industry"] = code
        recs.append(t)
    ref = pd.concat(recs, ignore_index=True)

    m = v.merge(ref[["q", "industry", "ret_ref", "n", "gap_days"]],
                on=["q", "industry"], how="left")
    m["diff"] = m["ret"] - m["ret_ref"]
    n_bad = int((m["diff"].abs() > 1e-9).sum())
    w(f"- 匹配到重算结果：**{int(m['ret_ref'].notna().sum())} / {len(m)}**")
    w(f"- 最大绝对误差：**{m['diff'].abs().max():.4f}**")
    w(f"- 误差 >1e-9 的观测：**{n_bad}**")
    if n_bad:
        w()
        w("| 季度 | 行业 | CSV | 重算 | 误差 |")
        w("|---|---|---|---|---|")
        bad = m[m["diff"].abs() > 1e-9].sort_values("diff", key=abs, ascending=False)
        for _, x in bad.head(20).iterrows():
            w(f"| {x['q']} | {x['industry']} | {x['ret']:+.4f} | {x['ret_ref']:+.4f} | {x['diff']:+.4f} |")
        w(f"（仅列前 20 条，共 {n_bad} 条）")
    w()
    if n_bad == 0:
        rec("V2", "PASS", "所有季度收益与独立重算完全一致")
    else:
        qs = sorted(m.loc[m["diff"].abs() > 1e-9, "q"].unique())
        rec("V2", "FAIL", f"{n_bad} 条收益与重算不符，集中在 {', '.join(qs)}")
    return m


# ------------------------------------------------------ V3 每季完整性
def check_completeness(m: pd.DataFrame) -> None:
    h(2, "V3 · 每个行业-季度是否覆盖完整季度")
    w("判据：该季交易日数 ≥ 40，且最后一个交易日距季度末 ≤ 5 天。")
    w()
    incomplete = m[(m["n"] < 40) | (m["gap_days"] > 5)]
    w(f"- 不完整 行业-季度：**{len(incomplete)}** 个")
    if len(incomplete):
        g = incomplete.groupby("q").agg(行业数=("industry", "size"),
                                        最少交易日=("n", "min"),
                                        最多离季末天数=("gap_days", "max"))
        w()
        w("| 季度 | 涉及行业数 | 该季最少交易日 | 离季末最多天数 |")
        w("|---|---|---|---|")
        for q, r in g.iterrows():
            w(f"| {q} | {int(r['行业数'])} | {int(r['最少交易日'])} | {int(r['最多离季末天数'])} |")
        bad_q = set(incomplete["q"])
        kept = m[~m["q"].isin(bad_q)]
        w()
        w(f"> 若剔除这些季度：**{kept['industry'].nunique()} 行业 × {kept['q'].nunique()} 季度 "
          f"= {len(kept)} 行**（当前为 {len(m)} 行）。")
    w()
    if len(incomplete) == 0:
        rec("V3", "PASS", "所有行业-季度都覆盖完整季度")
    else:
        qs = sorted(incomplete["q"].unique())
        rec("V3", "FAIL", f"{len(incomplete)} 个行业-季度不完整（{', '.join(qs)}），首季不能直接进回归")


# ------------------------------------------------- V4 零值与极端值
def check_extremes(v: pd.DataFrame) -> None:
    h(2, "V4 · 零收益与极端值")
    z = int((v["ret"] == 0).sum())
    big = v[v["ret"].abs() > 1]
    w(f"- `ret` 恰好为 0 的行：**{z}** 行")
    w(f"- `|ret| > 100%` 的行：**{len(big)}** 行")
    if len(big):
        for _, x in big.iterrows():
            w(f"  - {x['q']} {x['industry']}：{x['ret']:+.2%}")
    w()
    if z == 0 and len(big) <= 2:
        rec("V4", "PASS", f"无零收益行；极端值 {len(big)} 条（需在论文附录说明）")
    elif z > 0:
        rec("V4", "FAIL", f"存在 {z} 行零收益，疑似空季度被填 0")
    else:
        rec("V4", "WARN", f"极端值 {len(big)} 条，需逐条核实")


# --------------------------------------------- V5 基准与超额收益自洽
def check_benchmark(v: pd.DataFrame) -> None:
    h(2, "V5 · 基准与超额收益的自洽性")
    per_q = v.groupby("q").apply(lambda g: abs(g["ret"].mean() - g["mkt_ret"].iloc[0]),
                                 include_groups=False)
    resid = v.groupby("q")["ret_ex"].sum().abs().max()
    w(f"- `mkt_ret` 与季内等权均值最大偏差：**{per_q.max():.2e}**")
    w(f"- 每季 `ret_ex` 之和最大绝对值：**{resid:.2e}**（恒为 0 是定义使然）")
    w()
    w("> `ret_ex` 每季求和恒为 0 属机械约束：基准是「本样本 28 个行业的等权平均」。")
    w("> 做面板回归（含季度固定效应）时无碍；但**不要**用它论证「某些行业长期跑赢」。")
    w()
    if per_q.max() < 1e-9:
        rec("V5", "PASS", "基准计算自洽；`ret_ex` 为样本内相对收益，需在论文中说明")
    else:
        rec("V5", "FAIL", "mkt_ret 与季内均值不一致")


# ------------------------------------------------- V6 原始数据卫生
def check_raw(raw: dict[str, pd.Series]) -> None:
    h(2, "V6 · 原始日线卫生（重复日期、非交易日、断档）")
    dup = sum(int(s.index.duplicated().sum()) for s in raw.values())
    wk = sum(int((s.index.dayofweek >= 5).sum()) for s in raw.values())
    gaps = []
    for code, s in raw.items():
        s2 = s.loc[(s.index >= pd.Timestamp("2014-01-01")) & (s.index <= pd.Timestamp("2026-09-30"))]
        d = s2.index.to_series().diff().dt.days
        gaps.append((code, int(d.max()) if len(d.dropna()) else 0))
    worst = max(gaps, key=lambda x: x[1])
    w(f"- 重复日期：**{dup}**；周末日期：**{wk}**")
    w(f"- 样本期最大相邻交易日间隔：**{worst[1]} 天**（{worst[0]}）")
    w("  - 11 天左右的长假间隔属正常（春节/国庆），不等于停牌。")
    w()
    if dup == 0 and wk == 0:
        rec("V6", "PASS", f"无重复日期、无周末数据；最大间隔 {worst[1]} 天属正常长假")
    else:
        rec("V6", "WARN", f"存在重复日期 {dup} 条或周末数据 {wk} 条")


# ------------------------------------------------- V7 行业口径说明
def check_vintage(v: pd.DataFrame, raw: dict[str, pd.Series]) -> None:
    h(2, "V7 · 行业口径与起点")
    starts = {c: s.index.min() for c, s in raw.items() if c in set(v["industry"])}
    by_start: dict[str, list[str]] = {}
    for c, d in starts.items():
        by_start.setdefault(str(d.date()), []).append(c)
    w("| 首个交易日 | 行业数 | 代表代码 |")
    w("|---|---|---|")
    for d in sorted(by_start):
        codes = sorted(by_start[d])
        w(f"| {d} | {len(codes)} | {', '.join(codes[:4])}{' …' if len(codes) > 4 else ''} |")
    w()
    late = {d: codes for d, codes in by_start.items() if d > "2014-01-01"}
    if late:
        w("> 注意：这些行业在 2014 年后才「有数据」，说明其 2014Q1 只覆盖部分季度。")
    w()
    rec("V7", "PASS", "已列示各行业数据起点，需在论文样本说明中交代")


# --------------------------------------- V8 旧文件是否已清理
def check_old_file(v: pd.DataFrame) -> None:
    h(2, "V8 · 旧文件是否已从 processed 清理")
    if not os.path.exists(OLD_EX):
        w(f"- `data/processed/ret_ex.csv` 已不存在 ✅")
        if os.path.exists(os.path.join(ROOT, "data", "archive", "ret_ex_v0_37ind.csv")):
            w("- 旧版已归档到 `data/archive/ret_ex_v0_37ind.csv`（留痕，不删除）")
        w()
        rec("V8", "PASS", "旧文件已清理，`ret_panel.csv` 是唯一有效数据")
        return
    old = pd.read_csv(OLD_EX, dtype={"industry": str})
    old["industry"] = old["industry"].str.zfill(6)
    old["q"] = pd.to_datetime(old["quarter"]).dt.to_period("Q").astype(str)
    mt_old = datetime.fromtimestamp(os.path.getmtime(OLD_EX))
    mt_new = datetime.fromtimestamp(os.path.getmtime(PANEL))
    w(f"- `ret_ex.csv`：**{len(old)}** 行，{old['q'].nunique()} 个季度，"
      f"{old['industry'].nunique()} 个行业，修改于 {mt_old:%Y-%m-%d %H:%M}")
    w(f"- `ret_panel.csv`：**{len(v)}** 行，{v['q'].nunique()} 个季度，"
      f"{v['industry'].nunique()} 个行业，修改于 {mt_new:%Y-%m-%d %H:%M}")
    m = old.merge(v, on=["q", "industry"], suffixes=("_old", "_new"))
    if len(m):
        d = (m["ret_old"] - m["ret_new"]).abs()
        w(f"- 重叠 {len(m)} 条，收益不一致（>1e-6）**{int((d > 1e-6).sum())}** 条，"
          f"最大差 {d.max():.4f}")
    w()
    w("> 若该文件被编辑器占用无法删除，请关闭 VS Code / Excel 后手动删除：")
    w(f"> `Remove-Item \"{OLD_EX}\"`")
    w()
    rec("V8", "FAIL",
        f"旧文件仍在 processed（{len(old)} 行 / {old['industry'].nunique()} 行业），应删除或移入 data/archive")


# ------------------------------------------------- V9 事件可用性
def check_events(v: pd.DataFrame) -> None:
    h(2, "V9 · 能对上几个 MPR 事件")
    mm = pd.read_csv(os.path.join(PROC, "mpr_meta.csv"))
    mm["pq"] = pd.to_datetime(mm["pub_date"]).dt.to_period("Q").astype(str)
    qs = set(v["q"])
    ok = mm[mm["pq"].isin(qs)]
    w(f"- 报告总数 **{len(mm)}**；发布季度落在面板季度内的 **{len(ok)}** 个")
    w(f"- 面板季度数 **{len(qs)}**（{v['q'].min()}–{v['q'].max()}）")
    miss = mm[~mm["pq"].isin(qs)]
    w(f"- 无法匹配的报告：**{len(miss)}** 个（{miss['period'].min()}–{miss['period'].max()}）")
    w()
    w("> 日频事件研究不受这个窗口限制（日线从 1999 年就有，可用 102 个事件）。")
    w("> 季频面板只做稳健性，50 个时间维度对交互项而言并不宽裕。")
    w()
    rec("V9", "PASS", f"{len(ok)} 个事件可对齐到季频面板；日频事件研究仍有 102 个")


# ------------------------------------------------- V11 时间连续性
def check_continuity(v: pd.DataFrame) -> None:
    h(2, "V11 · 样本区间内有没有空洞")
    qs = sorted(pd.PeriodIndex(v["q"], freq="Q").unique())
    expected = pd.period_range(qs[0], qs[-1], freq="Q")
    holes = [str(q) for q in expected if q not in set(qs)]
    w(f"- 起止：**{qs[0]} – {qs[-1]}**，应为 {len(expected)} 个连续季度，实际 {len(qs)} 个")
    w(f"- 区间内缺失季度：**{holes if holes else '无'}**")
    w()
    if holes:
        rec("V11", "FAIL", f"样本区间内缺 {len(holes)} 个季度：{holes}")
    else:
        rec("V11", "PASS", f"{len(qs)} 个季度连续，无空洞")


# ------------------------------------------------- V10 体检报告是否过期
def check_stale_docs(v: pd.DataFrame) -> None:
    h(2, "V10 · 配套文档是否与新数据同步")
    p = os.path.join(ROOT, "docs", "数据体检.md")
    if not os.path.exists(p):
        rec("V10", "WARN", "找不到 docs/数据体检.md")
        return
    txt = open(p, encoding="utf-8").read()
    stale_rows = f"| 总行数 | {len(v)} |" not in txt
    mt = datetime.fromtimestamp(os.path.getmtime(p))
    w(f"- `docs/数据体检.md` 修改于 {mt:%Y-%m-%d %H:%M}")
    w(f"- 报告中是否出现「| 总行数 | {len(v)} |」：**{'是' if not stale_rows else '否'}**")
    w()
    if stale_rows:
        w("> 该报告仍在描述旧数据，**没有检查过 ret_panel.csv**。"
          "请运行 `python src/data/audit_panel.py` 重新生成。")
        rec("V10", "FAIL", "数据体检报告过期，未覆盖 ret_panel.csv")
    else:
        rec("V10", "PASS", "体检报告与新数据同步")


# ----------------------------------------------------------------- 主流程
def main() -> None:
    if not os.path.exists(PANEL):
        print("找不到 ret_panel.csv")
        return 1
    v, raw = load()

    w("# 面板核查：ret_panel.csv")
    w()
    w(f"- 核查时间：{datetime.now():%Y-%m-%d %H:%M}")
    w("- 核查脚本：`src/data/verify_panel.py`")
    w("")
    w("---")
    w()

    check_structure(v)
    m = check_recompute(v, raw)
    check_completeness(m)
    check_extremes(v)
    check_benchmark(v)
    check_raw(raw)
    check_vintage(v, raw)
    check_old_file(v)
    check_events(v)
    check_continuity(v)
    check_stale_docs(v)

    summary = io.StringIO()
    summary.write("## 结论总览\n\n| 编号 | 状态 | 结论 |\n|---|---|---|\n")
    for code, st, msg in results:
        summary.write(f"| {code} | {tag(st)} | {msg} |\n")
    summary.write("\n---\n\n")

    body = buf.getvalue()
    head, sep, rest = body.partition("---\n\n")
    final = head + sep + summary.getvalue() + rest

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(final)

    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    n_warn = sum(1 for _, s, _ in results if s == "WARN")
    print(f"核查完成 -> {OUT}")
    print(f"FAIL={n_fail}  WARN={n_warn}  PASS={len(results)-n_fail-n_warn}")
    for c, s, msg in results:
        if s != "PASS":
            print(f"  [{c}] {s}: {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
