"""
build_market_events.py — 构建"事件窗市场收益"
============================================
H1 的被解释变量是「报告发布后市场怎么走」。这份脚本把它算出来。

输入
    data/raw/sw/*.csv          行业日行情
    data/processed/event_map.csv  102 期发布日 → 交易日窗口
输出
    data/processed/market_events.csv

做法
    1. 行业日收益 → 逐日等权平均 = 市场日收益（无风险、无新增数据依赖）
    2. 累积成市场指数（起点 1.0）
    3. 用 event_map 里的 t_m1 / t0 / t1 / t5 取指数水平，算区间收益

为什么用"行业等权"当市场
    工作区里没有沪深 300 的日线，而等权行业组合：
      · 用现有数据就能算，零新增依赖；
      · 与后面 H2 的行业面板口径一致；
      · 学术上"等权组合"是常见市场代理（Fama-French 也用）。
    稳健性里可换成 28 行业口径或补抓沪深 300。

两套口径同时输出
    r16_*  ：16 个连续行业（2001 年起可得）→ 覆盖全部 102 期
    r28_*  ：28 个行业（2014-02-21 起）→ 只覆盖 2014 年后的 50 期
    主表用 r16（样本长），r28 做稳健性（口径新）。
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

RAW_SW = "data/raw/sw"
EVENT_MAP = "data/processed/event_map.csv"
OUT = "data/processed/market_events.csv"

# 16 个 2001 年起可得的行业
IND16 = ["801010", "801030", "801040", "801050", "801080", "801110", "801120",
         "801130", "801140", "801150", "801160", "801170", "801180", "801200",
         "801210", "801230"]
# 2014 年后可得的 12 个
IND12 = ["801710", "801720", "801730", "801740", "801750", "801760", "801770",
         "801780", "801790", "801880", "801890", "801950"]
IND28 = IND16 + IND12


def load_field(field: str) -> pd.DataFrame:
    """读某个字段（收盘/最高/最低），按日期对齐。"""
    series = {}
    for f in sorted(glob.glob(os.path.join(RAW_SW, "*.csv"))):
        code = os.path.basename(f).replace(".csv", "")
        d = pd.read_csv(f)
        d["日期"] = pd.to_datetime(d["日期"])
        if field not in d.columns:
            continue
        s = d.set_index("日期")[field].sort_index()
        s = s[~s.index.duplicated(keep="first")]
        series[code] = s
    return pd.DataFrame(series).sort_index()


def load_close() -> pd.DataFrame:
    return load_field("收盘")


def market_range(universe: list[str]) -> pd.Series:
    """市场日内振幅：(最高-最低)/前收盘，等行业等权平均。
    比 |收益| 更干净的波动代理——它不依赖方向，也不受跳空影响。"""
    hi, lo, cl = load_field("最高"), load_field("最低"), load_field("收盘")
    cols = [c for c in universe if c in hi.columns and c in lo.columns]
    prev = cl[cols].shift(1)
    rng = (hi[cols] - lo[cols]) / prev
    return rng.mean(axis=1, skipna=True).dropna()


def market_index(close: pd.DataFrame, universe: list[str]) -> pd.Series:
    """等权行业组合：逐日对可得行业取平均收益，再累积成指数。"""
    sub = close[[c for c in universe if c in close.columns]]
    rets = sub.pct_change()
    mkt_ret = rets.mean(axis=1, skipna=True)          # 等权（每日再平衡）
    mkt_ret = mkt_ret[mkt_ret.notna()]
    return (1 + mkt_ret).cumprod()


def window_return(idx: pd.Series, d0: str, d1: str) -> float:
    """从 d0 收盘到 d1 收盘的收益；日期不存在返回 NaN。"""
    v0, v1 = idx.get(pd.Timestamp(d0)), idx.get(pd.Timestamp(d1))
    if v0 is None or v1 is None or pd.isna(v0) or pd.isna(v1):
        return np.nan
    return float(v1 / v0 - 1)


def main() -> int:
    close = load_close()
    print(f"[数据] 行业数 {close.shape[1]}，日期 {close.index.min().date()} – {close.index.max().date()}")

    idx16 = market_index(close, IND16)
    idx28 = market_index(close, IND28)
    # 少数日期上个别行业文件缺行 → 指数当天空缺。
    # 用全样本交易日历 reindex + 前值填充，保证每个发布日都能取到水平。
    cal = close.index
    idx16 = idx16.reindex(cal).ffill()
    idx28 = idx28.reindex(cal).ffill()
    print(f"[指数] 16 行业口径 {idx16.index.min().date()} 起，{len(idx16)} 天")
    print(f"[指数] 28 行业口径 {idx28.index.min().date()} 起，{len(idx28)} 天")

    rng16 = market_range(IND16)
    rng28 = market_range(IND28)
    ret16 = idx16.pct_change().dropna()
    ret28 = idx28.pct_change().dropna()
    print(f"[振幅] 16 行业 {len(rng16)} 天，28 行业 {len(rng28)} 天")

    em = pd.read_csv(EVENT_MAP)
    rows = []
    for _, r in em.iterrows():
        rec = {"period": r["period"], "t0": r["t0"], "t1": r["t1"], "t5": r["t5"]}
        for tag, rng, ret in (("16", rng16, ret16), ("28", rng28, ret28)):
            rec[f"rng{tag}_evt"] = float(rng.get(pd.Timestamp(r["t0"]), np.nan))
            rec[f"rng{tag}_5d"] = float(rng.reindex([pd.Timestamp(r["t0"]), pd.Timestamp(r["t1"]),
                                                    pd.Timestamp(r["t5"])]).mean())
            pm1 = ret.index.searchsorted(pd.Timestamp(r["t_m1"]))
            rec[f"prevol{tag}"] = float(ret.iloc[max(0, pm1 - 60):pm1].std())
        for tag, idx in (("16", idx16), ("28", idx28)):
            rec[f"lvl_{tag}_t0"] = idx.get(pd.Timestamp(r["t0"]), np.nan)
            rec[f"r{tag}_m1_0"] = window_return(idx, r["t_m1"], r["t0"])   # 事件日当天
            rec[f"r{tag}_0_1"] = window_return(idx, r["t0"], r["t1"])
            rec[f"r{tag}_0_5"] = window_return(idx, r["t0"], r["t5"])
            rec[f"r{tag}_1_5"] = window_return(idx, r["t1"], r["t5"])
            rec[f"car{tag}_m1_p1"] = window_return(idx, r["t_m1"], r["t1"])  # [-1,+1]
            rec[f"car{tag}_m1_p5"] = window_return(idx, r["t_m1"], r["t5"])  # [-1,+5]
        # t1 之后的"一个交易日"：t1 当天买入、下一交易日收盘卖出 → 用 r_1_1 近似需 t2
        rows.append(rec)

    out = pd.DataFrame(rows)
    # 发布后一日的收益（姜富伟口径）：t0 → t1
    out = out.rename(columns={"r16_0_1": "r16_1d", "r16_0_5": "r16_5d",
                              "r28_0_1": "r28_1d", "r28_0_5": "r28_5d"})
    out.to_csv(OUT, index=False, float_format="%.8f")

    print("-" * 60)
    print(f"[完成] {OUT}，{len(out)} 行")
    for c in ("r16_1d", "r16_5d", "r28_1d", "r28_5d"):
        s = out[c].dropna()
        print(f"  {c}: N={len(s)}, 均值 {s.mean():+.4f}, 标准差 {s.std():.4f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
