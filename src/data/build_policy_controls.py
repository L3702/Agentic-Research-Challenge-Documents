"""
build_policy_controls.py — 合并三类货币政策操作为季度控制变量
==========================================================
输入
    data/external/rrr_changes.csv    存款准备金率调整（公告日、幅度、是否定向）
    data/external/lpr_changes.csv    LPR 报价变动
    data/external/omo7d_changes.csv  7 天逆回购利率调整
输出
    data/processed/policy_controls.csv

口径
    · 按公告日所在季度归集（报告在下一季度发布，操作在报告期内已被市场知晓）
    · 全面降准计入 rrr_cut；定向降准单列 rrr_cut_t，供稳健性使用
    · LPR 只取 1 年期（5 年期以上与地产相关，不属总量政策利率）
    · 幅度统一为百分点，宽松为负
"""

from __future__ import annotations

import os
import sys

import pandas as pd


def quarterly(df: pd.DataFrame, date_col: str, value_col: str, cond=None) -> pd.Series:
    d = df.copy()
    if cond is not None:
        d = d[cond(d)]
    d["period"] = pd.to_datetime(d[date_col]).dt.to_period("Q").astype(str)
    return d.groupby("period")[value_col].sum()


def main() -> int:
    rrr = pd.read_csv("data/external/rrr_changes.csv", encoding="utf-8-sig")
    lpr = pd.read_csv("data/external/lpr_changes.csv", encoding="utf-8-sig")
    omo = pd.read_csv("data/external/omo7d_changes.csv", encoding="utf-8-sig")

    allq = pd.period_range("2001Q1", "2026Q3", freq="Q").astype(str)
    out = pd.DataFrame({"period": allq})

    # 全面降准（剔除定向）
    r_all = quarterly(rrr, "announce_date", "change_pp", lambda d: d.targeted == 0)
    r_tgt = quarterly(rrr, "announce_date", "change_pp", lambda d: d.targeted == 1)
    out["rrr_cut"] = out.period.map(r_all).fillna(0.0)
    out["rrr_cut_t"] = out.period.map(r_tgt).fillna(0.0)

    # LPR 1 年期
    l1 = quarterly(lpr, "date", "change_pp", lambda d: d.tenor == "1年期")
    out["lpr_cut"] = out.period.map(l1).fillna(0.0)

    # 7 天逆回购
    o7 = quarterly(omo, "date", "change_pp")
    out["omo_cut"] = out.period.map(o7).fillna(0.0)

    out["n_ops"] = (
        out.period.map(quarterly(rrr, "announce_date", "change_pp").notna()).fillna(False).astype(int)
        + out.period.map(l1).notna().astype(int)
        + out.period.map(o7).notna().astype(int)
    )
    out["any_easing"] = ((out.rrr_cut + out.lpr_cut + out.omo_cut) < 0).astype(int)
    out["any_action"] = ((out[["rrr_cut", "rrr_cut_t", "lpr_cut", "omo_cut"]].abs().sum(axis=1)) > 0).astype(int)

    out.to_csv("data/processed/policy_controls.csv", index=False, float_format="%.4f")
    print(f"[写出] data/processed/policy_controls.csv，{len(out)} 个季度")
    sub = out[(out.period >= "2001Q1") & (out.period <= "2026Q2")]
    print(f"  有操作的季度: {int(sub.any_action.sum())} / {len(sub)}")
    print(f"  全面降准 {int((sub.rrr_cut < 0).sum())} 次 | 定向降准 {int((sub.rrr_cut_t < 0).sum())} 次 | "
          f"LPR 下调 {int((sub.lpr_cut < 0).sum())} 次 | 逆回购下调 {int((sub.omo_cut < 0).sum())} 次")
    print(f"  2016 年之后有操作的季度: "
          f"{int(sub[(sub.period >= '2016Q1')].any_action.sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
