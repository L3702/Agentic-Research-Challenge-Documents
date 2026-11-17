# -*- coding: utf-8 -*-
"""Fig 14 事件研究 CAR 曲线（待数据）。

输入：e3_car_by_sentiment.csv（列 window, group, car, t, ci_low, ci_high）
输出：figures/fig14_event_study_car.png
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
from _style import CAT_COLORS, DATA_DIR, PALETTE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
INPUT_NAME = "e3_car_by_sentiment.csv"
SEARCH_DIRS = [os.path.join(ROOT, "results"), DATA_DIR, ROOT, os.path.join(ROOT, "data")]
COLUMN_MAP = {
    "事件窗": "window", "window": "window",
    "分组": "group", "group": "group",
    "car": "car", "CAR": "car",
    "t": "t", "t值": "t",
    "ci_low": "ci_low", "ci_high": "ci_high",
    "下限": "ci_low", "上限": "ci_high",
}
REQUIRED = ["window", "group", "car"]
GROUP_ORDER = ["高", "中", "低", "high", "mid", "low", "高情绪", "中情绪", "低情绪"]


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
        sys.exit(f"[Fig14] 找不到输入文件 {INPUT_NAME}；已查找：" + ", ".join(SEARCH_DIRS))
    df = pd.read_csv(p)
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        sys.exit(f"[Fig14] 缺少必需列 {miss}；现有列：{list(df.columns)}")
    if df.empty:
        sys.exit("[Fig14] 输入文件为空。")
    df["_wk"] = df["window"].map(_win_key)
    df = df.sort_values("_wk")
    if df["car"].isna().all():
        sys.exit("[Fig14] car 列全为空。")
    return df


def _group_sort_key(g):
    s = str(g).strip()
    for i, name in enumerate(GROUP_ORDER):
        if s == name:
            return (i, s)
    return (len(GROUP_ORDER), s)


def plot(df, outdir=None):
    groups = sorted(df["group"].dropna().unique(), key=_group_sort_key)
    fig, ax = plt.subplots(figsize=(10.5, 6))
    for i, g in enumerate(groups):
        sub = df[df["group"] == g].sort_values("_wk")
        x = np.arange(len(sub))
        color = CAT_COLORS[i % len(CAT_COLORS)]
        ax.plot(x, sub["car"], marker="o", ms=4.5, lw=1.8, color=color, label=str(g))
        if "ci_low" in sub and "ci_high" in sub:
            lo = sub["ci_low"].fillna(sub["car"])
            hi = sub["ci_high"].fillna(sub["car"])
            ax.fill_between(x, lo, hi, color=color, alpha=0.16)
    # x 轴标签用第一组的 window 原文
    first = df[df["group"] == groups[0]].sort_values("_wk")["window"].tolist() if groups else []
    ax.set_xticks(range(len(first)))
    ax.set_xticklabels([str(w) for w in first], fontsize=9)
    ax.axhline(0, color=PALETTE["neutral"], lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("事件窗（MPR 发布日 = 0）")
    ax.set_ylabel("累计异常收益 CAR")
    ax.set_title("图 14　不同情绪组的发布后累计异常收益（CAR）", fontsize=12, pad=12)
    ax.legend(title="情绪分组", fontsize=9, title_fontsize=9)
    ax.grid(alpha=0.35)
    ax.set_axisbelow(True)
    ax.text(0.01, -0.13,
            "注：阴影为 95% 置信区间；事件日为 MPR 发布日期；估计窗与事件窗不重叠（E2/E3 设定）。",
            transform=ax.transAxes, fontsize=8.2, color=PALETTE["neutral"])
    _save(fig, "fig14_event_study_car.png", outdir)


def _save(fig, name, outdir=None):
    out = outdir or FIG_DIR
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path, format="png", dpi=200)
    plt.close(fig)
    print(f"saved: {path}")
    return path


def _self_test():
    tmp = tempfile.mkdtemp(prefix="fig14_selftest_")
    windows = ["[0,+1]", "[0,+3]", "[0,+5]", "[0,+10]"]
    rows = []
    for g, base in [("高", 0.004), ("中", 0.0), ("低", -0.005)]:
        for i, w in enumerate(windows):
            car = base * (i + 1) / 2
            rows.append({"window": w, "group": g, "car": car,
                         "t": car / 0.004, "ci_low": car - 0.006, "ci_high": car + 0.006})
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
