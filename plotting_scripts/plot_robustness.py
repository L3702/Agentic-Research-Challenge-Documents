# -*- coding: utf-8 -*-
"""Fig 20 稳健性检验森林图（待数据）。

输入：e8_robustness_summary.csv（列 setting, beta, ci_low, ci_high）
输出：figures/fig20_robustness.png
说明：只读真实 CSV；--self-test 用临时合成数据验证绘图流程。
"""

import argparse
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import DATA_DIR, PALETTE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
INPUT_NAME = "e8_robustness_summary.csv"
SEARCH_DIRS = [os.path.join(ROOT, "results"), DATA_DIR, ROOT, os.path.join(ROOT, "data")]
COLUMN_MAP = {
    "设定": "setting", "setting": "setting", "稳健性设定": "setting",
    "beta": "beta", "系数": "beta",
    "ci_low": "ci_low", "下限": "ci_low",
    "ci_high": "ci_high", "上限": "ci_high",
}
REQUIRED = ["setting", "beta", "ci_low", "ci_high"]


def find_input():
    for d in SEARCH_DIRS:
        p = os.path.join(d, INPUT_NAME)
        if os.path.exists(p):
            return p
    return None


def load(path=None):
    p = path or find_input()
    if not p:
        sys.exit(f"[Fig20] 找不到输入文件 {INPUT_NAME}；已查找：" + ", ".join(SEARCH_DIRS))
    df = pd.read_csv(p)
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        sys.exit(f"[Fig20] 缺少必需列 {miss}；现有列：{list(df.columns)}")
    if df.empty:
        sys.exit("[Fig20] 输入文件为空。")
    df = df.dropna(subset=["beta", "ci_low", "ci_high"]).reset_index(drop=True)
    if df.empty:
        sys.exit("[Fig20] 有效行（beta 与置信区间非空）为 0。")
    return df


def plot(df, outdir=None):
    # 森林图自上而下阅读：最后一行画在最上方
    d = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(d))
    fig_h = max(4.6, 0.42 * len(d) + 1.4)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    ax.errorbar(d["beta"], y, xerr=[d["beta"] - d["ci_low"], d["ci_high"] - d["beta"]],
                fmt="s", color=PALETTE["primary"], ecolor=PALETTE["primary"],
                elinewidth=1.3, capsize=3.2, ms=5.2, alpha=0.9)
    ax.axvline(0, color=PALETTE["neutral"], lw=0.9, ls="--", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(d["setting"], fontsize=9)
    ax.set_xlabel("核心系数 β（情绪 → 行业收益）")
    ax.set_ylabel("稳健性设定")
    ax.set_title("图 20　稳健性检验：不同设定下的核心系数（E8）", fontsize=12, pad=12)
    ax.grid(axis="x", alpha=0.4)
    ax.set_axisbelow(True)
    # 标注主设定（输入第一行）
    main_setting = str(df["setting"].iloc[0])
    ax.text(0.99, 0.02, f"主设定：{main_setting}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.4, color=PALETTE["secondary"])
    ax.text(0.01, -0.12,
            "注：每个设定只改变一个主要条件；误差棒为 95% 置信区间，跨 0 表示该设定下不显著。",
            transform=ax.transAxes, fontsize=8.2, color=PALETTE["neutral"])
    _save(fig, "fig20_robustness.png", outdir)


def _save(fig, name, outdir=None):
    out = outdir or FIG_DIR
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path, format="png", dpi=200)
    plt.close(fig)
    print(f"saved: {path}")
    return path


def _self_test():
    tmp = tempfile.mkdtemp(prefix="fig20_selftest_")
    settings = ["主设定：词典A + [0,+5]", "词典B", "事件窗 [0,+1]", "事件窗 [0,+10]",
                "子样本 2015 前", "基准：中证全指", "滞后 2 期", "Winsorize 1%"]
    rng = np.random.default_rng(17)
    beta = rng.normal(0.002, 0.0012, len(settings))
    df = pd.DataFrame({"setting": settings, "beta": beta,
                       "ci_low": beta - rng.uniform(0.001, 0.003, len(settings)),
                       "ci_high": beta + rng.uniform(0.001, 0.003, len(settings))})
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
