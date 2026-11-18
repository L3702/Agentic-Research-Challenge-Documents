# -*- coding: utf-8 -*-
"""Fig 19 行业敏感度排序（待数据）。

输入：e5_heterogeneity.csv（列 industry, beta, ci_low, ci_high 或 ci）
输出：figures/fig19_heterogeneity.png
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
INPUT_NAME = "e5_heterogeneity.csv"
SEARCH_DIRS = [os.path.join(ROOT, "results"), DATA_DIR, ROOT, os.path.join(ROOT, "data")]
COLUMN_MAP = {
    "行业": "industry", "industry": "industry",
    "beta": "beta", "系数": "beta", "敏感度": "beta",
    "ci_low": "ci_low", "下限": "ci_low",
    "ci_high": "ci_high", "上限": "ci_high",
    "ci": "ci", "标准误": "se",
}
REQUIRED = ["industry", "beta"]


def find_input():
    for d in SEARCH_DIRS:
        p = os.path.join(d, INPUT_NAME)
        if os.path.exists(p):
            return p
    return None


def load(path=None):
    p = path or find_input()
    if not p:
        sys.exit(f"[Fig19] 找不到输入文件 {INPUT_NAME}；已查找：" + ", ".join(SEARCH_DIRS))
    df = pd.read_csv(p)
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        sys.exit(f"[Fig19] 缺少必需列 {miss}；现有列：{list(df.columns)}")
    if df.empty:
        sys.exit("[Fig19] 输入文件为空。")
    df = df.dropna(subset=["beta"]).sort_values("beta").reset_index(drop=True)
    if "ci_low" not in df or "ci_high" not in df:
        if "ci" in df:
            df["ci_low"] = df["beta"] - df["ci"]
            df["ci_high"] = df["beta"] + df["ci"]
        else:
            df["ci_low"] = np.nan
            df["ci_high"] = np.nan
    return df


def plot(df, outdir=None):
    y = np.arange(len(df))
    fig_h = max(5.2, 0.30 * len(df) + 1.4)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    colors = [PALETTE["secondary"] if b >= 0 else PALETTE["primary"] for b in df["beta"]]
    ax.barh(y, df["beta"], color=colors, alpha=0.88, height=0.62)
    if df["ci_low"].notna().any():
        ax.errorbar(df["beta"], y, xerr=[df["beta"] - df["ci_low"], df["ci_high"] - df["beta"]],
                    fmt="none", ecolor=PALETTE["dark"], elinewidth=1.0, capsize=2.4, alpha=0.8)
    ax.axvline(0, color=PALETTE["neutral"], lw=0.9, ls="--", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(df["industry"], fontsize=8.8)
    ax.set_xlabel("行业敏感度 β（情绪 × 行业收益）")
    ax.set_ylabel("申万一级行业")
    ax.set_title("图 19　行业对 MPR 情绪的敏感度排序（E5）", fontsize=12, pad=12)
    ax.grid(axis="x", alpha=0.4)
    ax.set_axisbelow(True)
    ax.text(0.01, -0.11,
            "注：误差棒为 95% 置信区间；分组与行业特征由 E5 事先确定，未按结果挑选行业。",
            transform=ax.transAxes, fontsize=8.2, color=PALETTE["neutral"])
    _save(fig, "fig19_heterogeneity.png", outdir)


def _save(fig, name, outdir=None):
    out = outdir or FIG_DIR
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path, format="png", dpi=200)
    plt.close(fig)
    print(f"saved: {path}")
    return path


def _self_test():
    tmp = tempfile.mkdtemp(prefix="fig19_selftest_")
    inds = ["银行", "计算机", "食品饮料", "医药生物", "煤炭", "有色金属", "房地产", "国防军工"]
    rng = np.random.default_rng(13)
    beta = rng.normal(0, 0.004, len(inds))
    df = pd.DataFrame({"industry": inds, "beta": beta,
                       "ci_low": beta - 0.003, "ci_high": beta + 0.003})
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
