# -*- coding: utf-8 -*-
"""Fig 13 MPR 情绪得分时间序列（待数据）。

输入：sentiment_dictionary_full.csv（列 period, sentiment, coverage）
输出：figures/fig13_sentiment.png
说明：只读真实 CSV，不生成任何假数据；--self-test 仅用临时目录合成数据验证绘图流程。
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
from _style import CAT_COLORS, DATA_DIR, PALETTE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
INPUT_NAME = "sentiment_dictionary_full.csv"
SEARCH_DIRS = [os.path.join(ROOT, "results"), DATA_DIR, ROOT, os.path.join(ROOT, "data")]
COLUMN_MAP = {
    "报告期": "period", "期次": "period", "period": "period",
    "情绪得分": "sentiment", "sentiment": "sentiment", "score": "sentiment",
    "覆盖率": "coverage", "coverage": "coverage",
}
REQUIRED = ["period", "sentiment", "coverage"]


def find_input():
    for d in SEARCH_DIRS:
        p = os.path.join(d, INPUT_NAME)
        if os.path.exists(p):
            return p
    return None


def load(path=None):
    p = path or find_input()
    if not p:
        sys.exit(f"[Fig13] 找不到输入文件 {INPUT_NAME}；已查找：" + ", ".join(SEARCH_DIRS))
    df = pd.read_csv(p)
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        sys.exit(f"[Fig13] 缺少必需列 {miss}；现有列：{list(df.columns)}")
    if df.empty:
        sys.exit("[Fig13] 输入文件为空。")
    df = df.dropna(subset=["sentiment"]).reset_index(drop=True)
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
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.12},
    )
    ax1.axhline(0, color=PALETTE["neutral"], lw=0.8, ls="--", alpha=0.6)
    ax1.plot(x, df["sentiment"], color=PALETTE["primary"], lw=1.4, label="情绪得分")
    ax1.fill_between(x, df["sentiment"], 0,
                     where=(df["sentiment"] >= 0), color=PALETTE["primary"],
                     alpha=0.25, interpolate=True)
    ax1.fill_between(x, df["sentiment"], 0,
                     where=(df["sentiment"] < 0), color=PALETTE["secondary"],
                     alpha=0.25, interpolate=True)
    ax1.set_ylabel("情绪得分（词典法，标准化）")
    ax1.set_title("图 13　MPR 文本情绪得分与词覆盖率时间序列", fontsize=12, pad=12)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.35)
    ax1.set_axisbelow(True)

    ax2.bar(x, df["coverage"], width=20 if is_date else 0.6,
            color=CAT_COLORS[2], alpha=0.85)
    ax2.set_ylabel("正负词覆盖率")
    ax2.set_xlabel("期次")
    ax2.grid(alpha=0.35)
    ax2.set_axisbelow(True)
    if is_date:
        ax2.xaxis.set_major_locator(mdates.YearLocator(2))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.text(0.01, -0.32,
             "注：情绪得分口径由 E1 固定（词典版本、停用词表一致）；覆盖率 = 命中正负词数 / 有效词数。",
             transform=ax2.transAxes, fontsize=8.2, color=PALETTE["neutral"])
    _save(fig, "fig13_sentiment.png", outdir)


def _save(fig, name, outdir=None):
    out = outdir or FIG_DIR
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path, format="png", dpi=200)
    plt.close(fig)
    print(f"saved: {path}")
    return path


def _self_test():
    tmp = tempfile.mkdtemp(prefix="fig13_selftest_")
    n = 20
    dates = pd.date_range("2016-03-31", periods=n, freq="QE")
    rng = np.random.default_rng(7)
    sentiment = np.cumsum(rng.normal(0, 0.25, n))
    df = pd.DataFrame({
        "period": dates.strftime("%Y-%m-%d"),
        "sentiment": sentiment,
        "coverage": rng.uniform(0.15, 0.55, n),
    })
    p = os.path.join(tmp, INPUT_NAME)
    df.to_csv(p, index=False)
    load(p)
    plot(load(p), outdir=tmp)
    return tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="用临时合成数据验证绘图，不写 figures/")
    args = ap.parse_args()
    if args.self_test:
        tmp = _self_test()
        print(f"self-test ok: {tmp}")
        return
    plot(load())


if __name__ == "__main__":
    main()
