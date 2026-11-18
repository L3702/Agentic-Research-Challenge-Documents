"""
build_policy_rate.py — 政策利率历史与季度操作强度
================================================
用途：把"央行报告期内是否有降息/加息"做成控制变量，检验正文的情绪效应
      是否只是货币政策操作的代理。

数据来源（唯一，官方）
    中国人民银行·货币政策司·利率政策·历史数据
    《金融机构人民币贷款基准利率》  2015-10-24 更新
    https://wuhan.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/125838/125888/2968985/index.html

覆盖范围
    1991-04-21 至 2015-10-24。2015 年 10 月之后贷款基准利率未再调整，
    政策利率转向 LPR（2019-08 起）与 7 天逆回购，这两段**未纳入本表**，
    因此本脚本只对 2015Q4 及以前的样本提供完整的控制变量。

输出
    data/external/policy_rate_history.csv   逐次调整记录（含幅度）
    data/processed/policy_actions.csv       按报告期汇总的季度操作强度
"""

from __future__ import annotations

import os
import sys

import pandas as pd

OUT_HIST = "data/external/policy_rate_history.csv"
OUT_Q = "data/processed/policy_actions.csv"

# 官方原表的"六个月至一年（含一年）"档；2014-11-22 起期限并档为"一年以内"
RAW = [
    ("1991-04-21", 8.64), ("1993-05-15", 9.36), ("1993-07-11", 10.98),
    ("1995-01-01", 10.98), ("1995-07-01", 12.06), ("1996-05-01", 10.98),
    ("1996-08-23", 10.08), ("1997-10-23", 8.64), ("1998-03-25", 7.92),
    ("1998-07-01", 6.93), ("1998-12-07", 6.39), ("1999-06-10", 5.85),
    ("2002-02-21", 5.31), ("2004-10-29", 5.58), ("2006-04-28", 5.85),
    ("2006-08-19", 6.12), ("2007-03-18", 6.39), ("2007-05-19", 6.57),
    ("2007-07-21", 6.84), ("2007-08-22", 7.02), ("2007-09-15", 7.29),
    ("2007-12-21", 7.47), ("2008-09-16", 7.20), ("2008-10-09", 6.93),
    ("2008-10-30", 6.66), ("2008-11-27", 5.58), ("2008-12-23", 5.31),
    ("2010-10-20", 5.56), ("2010-12-26", 5.81), ("2011-02-09", 6.06),
    ("2011-04-06", 6.31), ("2011-07-07", 6.56), ("2012-06-08", 6.31),
    ("2012-07-06", 6.00), ("2014-11-22", 5.60), ("2015-03-01", 5.35),
    ("2015-05-11", 5.10), ("2015-06-28", 4.85), ("2015-08-26", 4.60),
    ("2015-10-24", 4.35),
]


def main() -> int:
    df = pd.DataFrame(RAW, columns=["date", "rate_1y"])
    df["date"] = pd.to_datetime(df["date"])
    df["change"] = df.rate_1y.diff().round(4)
    df["direction"] = df.change.apply(
        lambda x: "降息" if x < 0 else ("加息" if x > 0 else "—"))
    os.makedirs("data/external", exist_ok=True)
    df.to_csv(OUT_HIST, index=False, float_format="%.4f")
    print(f"[写出] {OUT_HIST}，{len(df)} 次调整")

    # 只保留 2001 年以后（样本期），按季度汇总
    sub = df[(df.date >= "2001-01-01") & df.change.notna()].copy()
    sub["period"] = sub.date.dt.to_period("Q").astype(str)
    q = sub.groupby("period").agg(
        n_cut=("direction", lambda s: (s == "降息").sum()),
        n_hike=("direction", lambda s: (s == "加息").sum()),
        delta_rate=("change", "sum"),
    ).reset_index()
    # 补齐没有操作的季度
    allq = pd.period_range("2001Q1", "2026Q2", freq="Q").astype(str)
    q = pd.DataFrame({"period": allq}).merge(q, on="period", how="left").fillna(
        {"n_cut": 0, "n_hike": 0, "delta_rate": 0.0})
    q["n_cut"] = q.n_cut.astype(int)
    q["n_hike"] = q.n_hike.astype(int)
    q["has_policy"] = ((q.n_cut + q.n_hike) > 0).astype(int)
    q["covered"] = (q.period <= "2015Q4").astype(int)
    q.to_csv(OUT_Q, index=False, float_format="%.4f")
    print(f"[写出] {OUT_Q}，{len(q)} 个季度")

    print("\n样本期内的调整（2001-01-01 起）：")
    for _, r in sub.iterrows():
        print(f"  {r.date.date()}  {r.direction}  {r.change:+.2f}  → {r.rate_1y:.2f}%")
    print(f"\n合计：降息 {(sub.direction == '降息').sum()} 次，"
          f"加息 {(sub.direction == '加息').sum()} 次")
    print(f"有操作的季度 {(q.has_policy == 1).sum()} 个（2001Q1–2015Q4）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
