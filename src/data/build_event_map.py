"""
build_event_map.py — MPR 发布日 → 交易日窗口对齐表
=================================================
输入：data/processed/mpr_meta.csv（102 期报告的发布日）
      data/raw/sw/*.csv（用全部行业行情的并集作为 A 股交易日历）
输出：data/processed/event_map.csv

为什么必须做这一步：
    102 期报告**全部**在报告期结束后的下一个季度发布（如 2026Q1 报告 2026-05-11 发布）。
    若直接把"本期报告情绪"和"本期行业收益"放在同一行，就是用 5 月的信息解释 1–3 月的收益，
    属于前视偏差。事件研究必须把情绪对齐到"发布之后"的收益窗口。

两个容易踩的坑（本脚本已处理）：
    1. 发布日可能不是交易日（周末/假期），必须顺延到下一个交易日；
    2. 窗口必须用**交易日**推进，不能用"+7 自然日"。例如 2015Q4 报告发布于 2016-02-10，
       正是春节休市，自然日窗口里一个交易日都没有。

关于"发布当日能不能算"：
    mpr_meta.csv 的 pub_time 字段 102 行全为空，无法判断是盘中还是盘后发布。
    MPR 通常在收盘后发布，因此发布日当天的收益不包含信息。
    本表同时给出 t0（发布当交易日）与 t1（下一交易日），供两种口径使用，
    建议主表用 [t1, t5]，稳健性用 [t0, t5]。
"""

from __future__ import annotations

import glob
import os
import sys

import pandas as pd

RAW_SW = "data/raw/sw"
META = "data/processed/mpr_meta.csv"
OUT = "data/processed/event_map.csv"

HORIZONS = (1, 5, 10)       # 需要 t+1 / t+5 / t+10
LONG_HOLIDAY_DAYS = 9       # 自然日跨度超过 9 天视为长假（春节/国庆）


def trading_calendar() -> pd.DatetimeIndex:
    days: set[pd.Timestamp] = set()
    for f in glob.glob(os.path.join(RAW_SW, "*.csv")):
        d = pd.read_csv(f, usecols=["日期"])
        days |= set(pd.to_datetime(d["日期"]))
    return pd.DatetimeIndex(sorted(days))


def main() -> int:
    if not os.path.exists(META):
        print(f"找不到 {META}")
        return 1

    cal = trading_calendar()
    meta = pd.read_csv(META)
    meta["pub"] = pd.to_datetime(meta["pub_date"])
    last_day = cal.max()

    rows = []
    for _, r in meta.iterrows():
        pub = r["pub"]
        pos = cal.searchsorted(pub, side="left")     # 第一个 >= pub 的交易日
        if pos >= len(cal):
            print(f"[警告] {r['period']} 的发布日 {pub.date()} 超出交易日历范围")
            continue

        t0 = cal[pos]
        rec = {
            "period": r["period"],
            "pub_date": pub.date().isoformat(),
            "pub_weekday": "一二三四五六日"[pub.dayofweek],
            "is_trading_day": bool(pub.normalize() == t0.normalize()),
            "t_m1": cal[pos - 1].date().isoformat() if pos >= 1 else "",
            "t0": t0.date().isoformat(),
            "t1": cal[pos + 1].date().isoformat() if pos + 1 < len(cal) else "",
        }
        for h in HORIZONS:
            rec[f"t{h}"] = cal[pos + h].date().isoformat() if pos + h < len(cal) else ""
            rec[f"n{h}"] = min(h + 1, len(cal) - pos)

        t5 = cal[pos + min(5, len(cal) - pos - 1)]
        rec["cal_days_0_5"] = int((t5 - t0).days)
        # 从"发布"到"窗口末"的自然日跨度：同时覆盖两种长假情形
        #   ① 发布后立刻休市（如 2015Q4，2016-02-06 发布，2-15 才开盘）
        #   ② 窗口中间休市（如 2023Q4，2024-02-08 至 02-23 跨春节）
        rec["cal_days_pub_5"] = int((t5 - pub).days)
        rec["long_holiday"] = bool(rec["cal_days_pub_5"] >= LONG_HOLIDAY_DAYS)
        rec["near_data_end"] = bool(t0 >= last_day - pd.Timedelta(days=20))
        rec["pub_quarter"] = f"{t0.year}Q{t0.quarter}"
        rows.append(rec)

    em = pd.DataFrame(rows)
    em["pub_to_t0_days"] = (pd.to_datetime(em["t0"]) - pd.to_datetime(em["pub_date"])).dt.days

    em.to_csv(OUT, index=False)

    # ---------------- 报告 ----------------
    print("=" * 66)
    print(f"交易日历：{cal.min().date()} – {cal.max().date()}，共 {len(cal)} 个交易日")
    print(f"报告期数：{len(em)}")
    print(f"发布日不是交易日（已顺延）：{int((~em['is_trading_day']).sum())} 期")
    if (~em["is_trading_day"]).any():
        for _, x in em[~em["is_trading_day"]].iterrows():
            print(f"    {x['period']}  发布 {x['pub_date']}（周{x['pub_weekday']}） → t0 {x['t0']}")
    print(f"t+5 窗口跨越长假（发布→窗口末 >= {LONG_HOLIDAY_DAYS} 自然日）："
          f"{int(em['long_holiday'].sum())} 期")
    if em["long_holiday"].any():
        for _, x in em[em["long_holiday"]].iterrows():
            print(f"    {x['period']}  发布 {x['pub_date']} → {x['t5']}  "
                  f"（{x['cal_days_pub_5']} 个自然日，含 {x['pub_to_t0_days']} 天休市）")
    print(f"窗口交易日数 n5 的最小值：{int(em['n5'].min())}（正常为 6）")
    print(f"靠近数据末端（t0 距最后交易日 < 20 天）：{em['near_data_end'].sum()} 期")
    print(f"输出文件：{OUT}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
