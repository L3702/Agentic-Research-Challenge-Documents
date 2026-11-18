"""
placebo_window.py — 安慰剂窗口检验
=================================
问题：正文的波动效应，是不是「报告发布日附近恰好有其他信息」造成的？
做法：把事件日整体前移或后移若干个交易日，重算收益与振幅，再用同一套模型估计。
      若效应只在真实发布日出现、平移后消失，说明它确实来自报告；
      若平移后依然显著，说明我们测到的是某种与日历位置相关的共同因素。

输入
    data/raw/sw/*.csv             行业日行情
    data/processed/event_map.csv  102 期发布日 → 交易日
    data/processed/sentiment.csv  情绪序列
输出
    docs/安慰剂窗口检验.md
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("bme", os.path.join(HERE, "build_market_events.py"))
bme = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bme)

spec2 = importlib.util.spec_from_file_location("h1", os.path.join(HERE, "run_h1.py"))
h1 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(h1)

SHIFTS = [-20, -10, -5, 0, 5, 10, 20]
OUT_MD = "docs/安慰剂窗口检验.md"


def build_panel(shift: int) -> pd.DataFrame:
    """按 shift 个交易日平移事件日，重算被解释变量。"""
    close = bme.load_close()
    idx = bme.market_index(close, bme.IND16).reindex(close.index).ffill()
    rng = bme.market_range(bme.IND16).reindex(close.index)
    ret = idx.pct_change()

    cal = close.index
    em = pd.read_csv(bme.EVENT_MAP)
    rows = []
    for _, r in em.iterrows():
        pos = cal.searchsorted(pd.Timestamp(r["t0"]))
        q = pos + shift
        if q < 61 or q + 1 >= len(cal):
            continue
        d0, d1 = cal[q], cal[q + 1]
        rows.append({
            "period": r["period"],
            "r_1d": float(idx.iloc[q + 1] / idx.iloc[q] - 1),
            "rng": float(rng.iloc[q]),
            "prevol": float(ret.iloc[q - 60:q].std()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    s = pd.read_csv("data/processed/sentiment.csv")[["period", "z_tone_P"]]

    lines = ["# 安慰剂窗口检验", "",
             "> 把事件日整体平移若干交易日，重算收益与振幅，再用与正文相同的模型估计。",
             "> 若效应只在真实发布日（shift=0）出现，说明它来自报告本身。", "",
             "| 平移 | 振幅方程 z_tone_P | 加事前波动 | 断点交互项 | 观测数 |",
             "|---|---|---|---|---|"]

    for k in SHIFTS:
        d = build_panel(k).merge(s, on="period").dropna(subset=["z_tone_P", "rng"])
        d["year"] = d.period.str[:4].astype(int)
        d["post14"] = (d.year >= 2014).astype(float)
        d["zx"] = d.z_tone_P * d.post14

        def cell(y, xs, var):
            res = h1.fit(d, y, xs, 4)
            if res is None:
                return "—"
            row = res[res["var"] == var].iloc[0]
            return f"{row.coef*10000:+.1f}{row.star} ({row.t:+.2f})"

        b3 = cell("rng", ["z_tone_P"], "z_tone_P")
        b4 = cell("rng", ["z_tone_P", "prevol"], "z_tone_P")
        c1 = cell("r_1d", ["z_tone_P", "zx", "post14"], "zx")
        tag = "**真实发布日**" if k == 0 else f"{k:+d} 个交易日"
        lines.append(f"| {tag} | {b3} | {b4} | {c1} | {len(d)} |")
        print(f"shift={k:+3d}  B3 {b3:<18} B4 {b4:<18} C1 {c1:<18} N={len(d)}")

    lines += ["", "注：系数单位为基点，括号内为 Newey--West 稳健 $t$ 值；"
                  "振幅与收益均按平移后的事件日重新计算，情绪变量不变。", ""]

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[完成] {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
