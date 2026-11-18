# -*- coding: utf-8 -*-
"""Fig 15 行业 × 事件窗 CAR 热力图（待数据）。

输入：e3_event_results.csv（列 industry, window, car）
输出：figures/fig15_event_heatmap.png
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
INPUT_NAME = "e3_event_results.csv"
SEARCH_DIRS = [os.path.join(ROOT, "results"), DATA_DIR, ROOT, os.path.join(ROOT, "data")]
COLUMN_MAP = {
    "行业": "industry", "industry": "industry",
    "事件窗": "window", "window": "window",
    "car": "car", "CAR": "car",
}
REQUIRED = ["industry", "window", "car"]


def _win_key(w):
    s = str(w).strip()
    try:
        return (0, float(s))
    except ValueError:
        pass
    digits = "".join(ch if ch.isdigit() or ch == "+" else " " for ch in s).split()
    if digits:
        return (0, float(digits[-1]))
    return (1, s)


def find_input():
    for d in SEARCH_DIRS:
        p = os.path.join(d, INPUT_NAME)
        if os.path.exists(p):
            return p
    return None


def load(path=None):
    p = path or find_input()
    if not p:
        sys.exit(f"[Fig15] 找不到输入文件 {INPUT_NAME}；已查找：" + ", ".join(SEARCH_DIRS))
    df = pd.read_csv(p)
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        sys.exit(f"[Fig15] 缺少必需列 {miss}；现有列：{list(df.columns)}")
    if df.empty:
        sys.exit("[Fig15] 输入文件为空。")
    df["_wk"] = df["window"].map(_win_key)
    df = df.dropna(subset=["car"])
    if df.empty:
        sys.exit("[Fig15] car 列全为空。")
    return df


def plot(df, outdir=None):
    wide = df.pivot_table(index="industry", columns="_wk", values="car", aggfunc="mean")
    wide = wide.sort_index(axis=1)
    order = df.drop_duplicates("_wk").sort_values("_wk")
    labels = [str(w) for w in order["window"]]
    vals = wide.values
    vmax = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 1.0
    vmax = vmax if vmax > 0 else 1.0

    n_rows, n_cols = vals.shape
    fig_h = max(5.0, 0.34 * n_rows + 1.6)
    fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * n_cols + 3.2), fig_h))
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(wide.index, fontsize=8.5)
    if n_rows * n_cols <= 500:
        for i in range(n_rows):
            for j in range(n_cols):
                v = vals[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.2%}", ha="center", va="center", fontsize=6.4,
                        color="white" if abs(v) > 0.55 * vmax else PALETTE["dark"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("CAR", fontsize=9)
    ax.set_xlabel("事件窗（MPR 发布日 = 0）")
    ax.set_ylabel("申万一级行业")
    ax.set_title("图 15　行业 × 事件窗累计异常收益（CAR）热力图", fontsize=12, pad=12)
    _save(fig, "fig15_event_heatmap.png", outdir)


def _save(fig, name, outdir=None):
    out = outdir or FIG_DIR
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path, format="png", dpi=200)
    plt.close(fig)
    print(f"saved: {path}")
    return path


def _self_test():
    tmp = tempfile.mkdtemp(prefix="fig15_selftest_")
    inds = ["银行", "计算机", "食品饮料", "医药生物", "煤炭", "有色金属"]
    windows = ["[0,+1]", "[0,+3]", "[0,+5]", "[0,+10]"]
    rng = np.random.default_rng(3)
    rows = [{"industry": i, "window": w, "car": rng.normal(0, 0.01)} for i in inds for w in windows]
    df = pd.DataFrame(rows)
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
