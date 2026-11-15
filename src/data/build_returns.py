"""
build_returns.py — 行业季度收益面板构建（修正版）
=================================================
输入：data/raw/sw/*.csv      37 个申万行业指数日行情
输出：data/processed/ret_panel.csv
      列：quarter, industry, ret, mkt_ret, ret_ex, n_days

本版修正的三件事（对应 docs/03_审阅意见_ret_panel.md）：

  1. 【先算收益，再切窗口】
     旧版先把数据切到 2014-01-01 之后，再算 pct_change()，
     导致 2014 年第一个交易日没有"前一日收盘价"，整季少算 1 天。
     例：801080 电子 2014Q1 由 +0.92% 被算成 -2.30%（符号翻转）。

  2. 【用"完整季度"判据，代替 dropna() + !=0 过滤 + 硬编码剔除】
     一季算不算数，只看两个条件：交易日数 >= 40，且最后一个交易日距季度末 <= 5 天。
     这样 2014Q1（新口径行业只有 27 天）和 2026Q4（只有 24 天）都会被自动排除，
     不需要再用 `quarter < '2026Q4'` 这种硬编码。
     注意：绝不能用 `ret != 0` 来"剔除空季度"——它会静默删掉真实的零收益季度。

  3. 【只保留所有行业都完整的季度】
     申万 2014 版分类于 2014-02-21 生效，12 个行业（801710-801950）的指数从那天才有数据。
     因此 2014Q1 是"16 个行业 58 个交易日 + 12 个行业 27 个交易日"的混合季度，
     横截面不可比，整季剔除。样本因此从 2014Q2 起。

行业清单为什么是这 28 个：
    1999-12-30 起连续可得的 16 个 + 2014-02-21 起可得的 12 个。
    排除 801020 采掘（2021Q4 终止）、801960 石油石化 / 801970 环保 / 801980 美容护理
    （2021-12-13 才有数据，纳入则样本只剩 20 个季度）。
    注意：801950 煤炭在 2014-2021 期间是采掘（801020）下属的二级行业，
    2021 年改版后升为一级行业并被回填到 2014-02-21。
    因此本面板的"能源"口径只含煤炭、不含石油石化，需在论文中说明（见 docs/04_数据修复记录.md）。
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

RAW_DIR = "data/raw/sw"
OUT_FILE = "data/processed/ret_panel.csv"

# 前提条件：一季至少 40 个交易日、且季末最后一个交易日距季度末不超过 5 天
MIN_TRADING_DAYS = 40
MAX_GAP_TO_QUARTER_END = 5

# 样本起点：不写死"2014-01-01"，而是取"所有行业都有完整数据的第一个季度"，
# 但为了可复现，仍然给出一个下限，防止历史上某个残缺季度意外混进来。
SAMPLE_FLOOR = pd.Period("2014Q1", "Q")

INDUSTRIES = [
    # 1999-12-30 起连续可得（16 个）
    "801010", "801030", "801040", "801050", "801080", "801110", "801120",
    "801130", "801140", "801150", "801160", "801170", "801180", "801200",
    "801210", "801230",
    # 2014-02-21 起可得（12 个）
    "801710", "801720", "801730", "801740", "801750", "801760", "801770",
    "801780", "801790", "801880", "801890", "801950",
]


def quarter_returns(code: str) -> pd.DataFrame:
    """单个行业的季度收益；只返回"完整季度"，并附该季交易日数。"""
    df = pd.read_csv(os.path.join(RAW_DIR, f"{code}.csv"))
    df["日期"] = pd.to_datetime(df["日期"])
    close = df.set_index("日期")["收盘"].sort_index()
    close = close[~close.index.duplicated(keep="first")]

    # 关键：在【完整】日序列上计算收益，首日 NaN 只影响全样本第一行（1999-12-30）
    r = close.pct_change()

    g = r.groupby(r.index.to_period("Q"))
    out = pd.DataFrame({
        "n_days": g.size(),
        "last_day": g.apply(lambda x: x.index.max()),
        "ret": g.apply(lambda x: (1 + x.dropna()).prod() - 1 if x.notna().any() else np.nan),
    })
    out.index.name = "q"
    out["qend"] = out.index.to_timestamp(how="end").normalize()
    out["gap"] = (out["qend"] - out["last_day"].dt.normalize()).dt.days
    out["complete"] = (out["n_days"] >= MIN_TRADING_DAYS) & (out["gap"] <= MAX_GAP_TO_QUARTER_END)

    return out.loc[out["complete"], ["ret", "n_days"]]


def main() -> int:
    print("=" * 66)
    print("构建行业季度收益面板")
    print("=" * 66)

    missing = [c for c in INDUSTRIES if not os.path.exists(os.path.join(RAW_DIR, f"{c}.csv"))]
    if missing:
        print(f"[错误] 缺少原始文件：{missing}")
        return 1

    frames = {}
    for code in INDUSTRIES:
        try:
            frames[code] = quarter_returns(code)
        except Exception as e:  # noqa: BLE001
            print(f"[错误] {code} 处理失败：{e}")
            return 1

    ret = pd.DataFrame({c: f["ret"] for c, f in frames.items()}).sort_index()
    nd = pd.DataFrame({c: f["n_days"] for c, f in frames.items()}).sort_index()

    ret = ret.loc[ret.index >= SAMPLE_FLOOR]
    nd = nd.loc[ret.index]

    # 只保留"28 个行业都完整"的季度
    full = ret.notna().all(axis=1) & nd.notna().all(axis=1)
    dropped = sorted(set(ret.index[~full]))
    ret, nd = ret.loc[full], nd.loc[full]

    if ret.empty:
        print("[错误] 没有可用季度")
        return 1

    if dropped:
        print(f"[剔除] 以下季度无法做到 28 个行业全覆盖，已剔除：")
        for q in dropped:
            detail = []
            for c in INDUSTRIES:
                if q not in frames[c].index:
                    detail.append(c)
            print(f"    {q}  缺失行业 {len(detail)} 个"
                  + (f"（如 {', '.join(detail[:3])}…）" if detail else ""))

    # 长表
    long = (ret.reset_index()
            .melt(id_vars="q", var_name="industry", value_name="ret"))
    nd_long = (nd.reset_index()
               .melt(id_vars="q", var_name="industry", value_name="n_days"))
    panel = long.merge(nd_long, on=["q", "industry"], how="left")

    # 基准：本样本 28 个行业的等权平均（固定篮子，不随季度变化）
    panel["mkt_ret"] = panel.groupby("q")["ret"].transform("mean")
    panel["ret_ex"] = panel["ret"] - panel["mkt_ret"]

    panel["quarter"] = panel["q"].astype(str)
    panel = panel[["quarter", "industry", "ret", "mkt_ret", "ret_ex", "n_days"]]
    panel = panel.sort_values(["quarter", "industry"]).reset_index(drop=True)

    # 连续性自检：样本区间内不允许出现空洞
    expected = pd.period_range(ret.index.min(), ret.index.max(), freq="Q")
    hole = [str(q) for q in expected if q not in ret.index]
    if hole:
        print(f"[警告] 样本区间内存在缺失季度：{hole}")

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    panel.to_csv(OUT_FILE, index=False, float_format="%.10f")

    print("-" * 66)
    print(f"行业数    : {panel['industry'].nunique()}")
    print(f"季度数    : {panel['quarter'].nunique()}  "
          f"({panel['quarter'].min()} – {panel['quarter'].max()})")
    print(f"总行数    : {len(panel)}  "
          f"（应等于 {panel['industry'].nunique()} × {panel['quarter'].nunique()} "
          f"= {panel['industry'].nunique() * panel['quarter'].nunique()}）")
    print(f"每季交易日: {int(panel['n_days'].min())} – {int(panel['n_days'].max())}")
    print(f"零收益行  : {int((panel['ret'] == 0).sum())}")
    print(f"输出文件  : {OUT_FILE}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
