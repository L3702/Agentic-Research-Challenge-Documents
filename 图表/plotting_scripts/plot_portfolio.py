# -*- coding: utf-8 -*-
"""Fig 18 高低情绪多空组合净值曲线（待数据）。

输入：e7_portfolio.csv（列 period, long, short, ls, benchmark）
输出：figures/fig18_portfolio_equity.png
说明：只读真实 CSV；--self-test 用临时合成数据验证绘图流程。
输入列既可为「净值」也可为「收益率」：脚本按数值特征自动判断并在图注中说明。
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
INPUT_NAME = "e7_portfolio.csv"
SEARCH_DIRS = [os.path.join(ROOT, "results"), DATA_DIR, ROOT, os.path.join(ROOT, "data")]
COLUMN_MAP = {
    "报告期": "period", "期次": "period", "period": "period",
    "多头": "long", "long": "long",
    "空头": "short", "short": "short",
    "多空": "ls", "ls": "ls", "long_short": "ls",
    "基准": "benchmark", "benchmark": "benchmark",
}
REQUIRED = ["period", "long", "short", "ls", "benchmark"]
SERIES = [("long", "多头（情绪最高 5 行业）"), ("short", "空头（情绪最低 5 行业）"),
          ("ls", "多空组合（多 - 空，净成本）"), ("benchmark", "基准（行业等权）")]


def find_input():
    for d in SEARCH_DIRS:
        p = os.path.join(d, INPUT_NAME)
        if os.path.exists(p):
            return p
    return None


def load(path=None):
    p = path or find_input()
    if not p:
        sys.exit(f"[Fig18] 找不到输入文件 {INPUT_NAME}；已查找：" + ", ".join(SEARCH_DIRS))
    df = pd.read_csv(p)
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        sys.exit(f"[Fig18] 缺少必需列 {miss}；现有列：{list(df.columns)}")
    if df.empty:
        sys.exit("[Fig18] 输入文件为空。")
    df = df.dropna(subset=["ls"]).reset_index(drop=True)
    parsed = pd.to_datetime(df["period"], errors="coerce")
    if parsed.notna().sum() >= max(1, int(0.8 * len(df))):
        df["_x"] = parsed
        df = df.sort_values("_x").reset_index(drop=True)
    else:
        df["_x"] = np.arange(len(df))
    return df


def _to_nav(s):
    """净值列直接返回；收益率列用 cumprod 转净值（起点 1000）。"""
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return s
    if (v > 0).all() and v.median() > 0.5:
        return s.astype(float)
    return (1 + pd.to_numeric(s, errors="coerce")).cumprod() * 1000.0


def plot(df, outdir=None):
    x = df["_x"]
    is_date = pd.api.types.is_datetime64_any_dtype(x)
    nav = {c: _to_nav(df[c]) for c, _ in SERIES}
    mode = "净值" if (pd.to_numeric(df["ls"], errors="coerce") > 0).all() else "收益率转净值"

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (col, label) in enumerate(SERIES):
        if col not in nav:
            continue
        color = CAT_COLORS[i % len(CAT_COLORS)]
        lw = 2.0 if col == "ls" else 1.3
        ls = "-" if col != "benchmark" else "--"
        ax.plot(x, nav[col], lw=lw, ls=ls, color=color, label=label)
    ax.set_ylabel("累计净值（起点 1000）")
    ax.set_xlabel("期次")
    ax.set_title("图 18　高低情绪多空组合累计净值（E7，等权、规则事先固定）", fontsize=12, pad=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.35)
    ax.set_axisbelow(True)
    if is_date:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.text(0.01, -0.13,
            f"注：输入列识别为{mode}；多空组合为季度调仓的等权规则，ls 已扣单边 10bp 交易成本；"
            "研究组合，不作为可执行策略。",
            transform=ax.transAxes, fontsize=8.2, color=PALETTE["neutral"])
    _save(fig, "fig18_portfolio_equity.png", outdir)


def _save(fig, name, outdir=None):
    out = outdir or FIG_DIR
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path, format="png", dpi=200)
    plt.close(fig)
    print(f"saved: {path}")
    return path


def _self_test():
    tmp = tempfile.mkdtemp(prefix="fig18_selftest_")
    dates = pd.date_range("2016-03-31", periods=24, freq="QE")
    rng = np.random.default_rng(9)
    ret = rng.normal(0.02, 0.05, (24, 4))
    df = pd.DataFrame({
        "period": dates.strftime("%Y-%m-%d"),
        "long": ret[:, 0],
        "short": ret[:, 1],
        "ls": ret[:, 0] - ret[:, 1],
        "benchmark": ret[:, 2],
    })
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
