# -*- coding: utf-8 -*-
"""Fig 17 样本外预测指标时序（待数据）。

输入：e4_oos_metrics.csv（列 period, ic, icir, oos_r2）
输出：figures/fig17_oos_metrics.png
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
from _style import CAT_COLORS, DATA_DIR, PALETTE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
INPUT_NAME = "e4_oos_metrics.csv"
SEARCH_DIRS = [os.path.join(ROOT, "results"), DATA_DIR, ROOT, os.path.join(ROOT, "data")]
COLUMN_MAP = {
    "报告期": "period", "期次": "period", "period": "period",
    "ic": "ic", "IC": "ic",
    "icir": "icir", "ICIR": "icir",
    "oos_r2": "oos_r2", "OOS_R2": "oos_r2", "oos_r²": "oos_r2",
}
REQUIRED = ["period", "ic", "icir", "oos_r2"]
PANELS = [("ic", "IC（横截面 Rank IC）"), ("icir", "ICIR（5 期滚动）"),
          ("oos_r2", "样本外 R²（逐季）")]


def find_input():
    for d in SEARCH_DIRS:
        p = os.path.join(d, INPUT_NAME)
        if os.path.exists(p):
            return p
    return None


def load(path=None):
    p = path or find_input()
    if not p:
        sys.exit(f"[Fig17] 找不到输入文件 {INPUT_NAME}；已查找：" + ", ".join(SEARCH_DIRS))
    df = pd.read_csv(p)
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        sys.exit(f"[Fig17] 缺少必需列 {miss}；现有列：{list(df.columns)}")
    if df.empty:
        sys.exit("[Fig17] 输入文件为空。")
    df = df.dropna(subset=["ic"]).reset_index(drop=True)
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
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True)
    for ax, (col, label) in zip(axes, PANELS):
        s = df[col]
        ax.axhline(0, color=PALETTE["neutral"], lw=0.8, ls="--", alpha=0.6)
        ax.plot(x, s, color=CAT_COLORS[0], lw=1.5, marker="o", ms=3.4, label=label)
        ax.axhline(s.mean(), color=CAT_COLORS[1], lw=1, ls=":", alpha=0.9,
                   label=f"均值 {s.mean():.4f}")
        ax.set_ylabel(label, fontsize=9.5)
        ax.grid(alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=8.4)
        if is_date:
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("期次")
    axes[0].set_title("图 17　样本外预测指标时序（时间顺序划分，无未来信息）", fontsize=12, pad=12)
    fig.text(0.01, -0.02,
             "注：IC 为 E4 样本外预测值与实际收益的逐季横截面 Rank IC；OOS R² 以各季截面均值为基准；"
             "ICIR 为 5 期滚动 IC 均值 / 标准差。汇总口径见 Table 9。",
             fontsize=8.2, color=PALETTE["neutral"])
    _save(fig, "fig17_oos_metrics.png", outdir)


def _save(fig, name, outdir=None):
    out = outdir or FIG_DIR
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path, format="png", dpi=200)
    plt.close(fig)
    print(f"saved: {path}")
    return path


def _self_test():
    tmp = tempfile.mkdtemp(prefix="fig17_selftest_")
    dates = pd.date_range("2019-03-31", periods=20, freq="QE")
    rng = np.random.default_rng(5)
    ic = rng.normal(0.03, 0.08, 20)
    df = pd.DataFrame({
        "period": dates.strftime("%Y-%m-%d"),
        "ic": ic,
        "icir": ic / 0.08,
        "oos_r2": rng.normal(0.01, 0.02, 20),
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
