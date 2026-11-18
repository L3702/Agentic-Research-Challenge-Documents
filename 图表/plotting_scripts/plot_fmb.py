# -*- coding: utf-8 -*-
"""Fig 16 Fama-MacBeth 系数时序（待数据）。

输入：e4_coefficient_series.csv（列 period, beta, se）
输出：figures/fig16_fmb_coefficient.png
说明：只读真实 CSV；--self-test 用临时合成数据验证绘图流程。
"""

import argparse
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import DATA_DIR, PALETTE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
INPUT_NAME = "e4_coefficient_series.csv"
SEARCH_DIRS = [os.path.join(ROOT, "results"), DATA_DIR, ROOT, os.path.join(ROOT, "data")]
COLUMN_MAP = {
    "报告期": "period", "期次": "period", "period": "period",
    "beta": "beta", "系数": "beta",
    "se": "se", "标准误": "se",
}
REQUIRED = ["period", "beta", "se"]


def find_input():
    for d in SEARCH_DIRS:
        p = os.path.join(d, INPUT_NAME)
        if os.path.exists(p):
            return p
    return None


def load(path=None):
    p = path or find_input()
    if not p:
        sys.exit(f"[Fig16] 找不到输入文件 {INPUT_NAME}；已查找：" + ", ".join(SEARCH_DIRS))
    df = pd.read_csv(p)
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        sys.exit(f"[Fig16] 缺少必需列 {miss}；现有列：{list(df.columns)}")
    if df.empty:
        sys.exit("[Fig16] 输入文件为空。")
    df = df.dropna(subset=["beta", "se"]).reset_index(drop=True)
    parsed = pd.to_datetime(df["period"], errors="coerce")
    if parsed.notna().sum() >= max(1, int(0.8 * len(df))):
        df["_x"] = parsed
        df = df.sort_values("_x").reset_index(drop=True)
    else:
        df["_x"] = np.arange(len(df))
    return df


def plot(df, outdir=None):
    x = df["_x"]
    is_date = pd.api.types.is_datetime64_any_dtype(x)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    lo = df["beta"] - 1.96 * df["se"]
    hi = df["beta"] + 1.96 * df["se"]
    ax.axhline(0, color=PALETTE["neutral"], lw=0.8, ls="--", alpha=0.6)
    ax.fill_between(x, lo, hi, color=PALETTE["primary"], alpha=0.2, label="95% 置信带")
    ax.plot(x, df["beta"], color=PALETTE["primary"], lw=1.6, marker="o", ms=3.6,
            label="Fama-MacBeth 系数")
    ax.set_ylabel("情绪变量的横截面系数 β")
    ax.set_xlabel("期次")
    ax.set_title("图 16　Fama-MacBeth 横截面系数时序（情绪 → 行业收益）", fontsize=12, pad=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.35)
    ax.set_axisbelow(True)
    if is_date:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.text(0.01, -0.13,
            "注：置信带 = β ± 1.96 × 标准误；系数口径由 E4 固定（控制变量与样本划分一致）。",
            transform=ax.transAxes, fontsize=8.2, color=PALETTE["neutral"])
    _save(fig, "fig16_fmb_coefficient.png", outdir)


def _save(fig, name, outdir=None):
    out = outdir or FIG_DIR
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path, format="png", dpi=200)
    plt.close(fig)
    print(f"saved: {path}")
    return path


def _self_test():
    tmp = tempfile.mkdtemp(prefix="fig16_selftest_")
    dates = pd.date_range("2016-03-31", periods=24, freq="QE")
    rng = np.random.default_rng(11)
    beta = 0.002 + 0.001 * rng.normal(size=24).cumsum() / 3
    df = pd.DataFrame({"period": dates.strftime("%Y-%m-%d"), "beta": beta,
                       "se": rng.uniform(0.001, 0.003, 24)})
    p = os.path.join(tmp, INPUT_NAME)
    df.to_csv(p, index=False)
    plot(load(p), outdir=tmp)
    return tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        tmp = _self_test()
        print(f"self-test ok: {tmp}")
        return
    plot(load())


if __name__ == "__main__":
    main()
