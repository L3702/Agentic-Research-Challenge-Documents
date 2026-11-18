# -*- coding: utf-8 -*-
"""Fig 9 行业风险-收益散点（2014-02-21 起，28 个可比行业，真实行情）。"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

from _style import DATA_DIR, PALETTE, save

START = "2014-02-21"
ANN = 252


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "sw_industries", "sw_industry_daily.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df["ret"] = df.groupby("code")["收盘"].pct_change()
    df = df[df["date"] >= START].sort_values(["code", "date"])
    df = df.dropna(subset=["ret"])
    stats = df.groupby("name")["ret"].agg(
        ann_ret=lambda s: (1 + s).prod() ** (ANN / len(s)) - 1,
        ann_vol=lambda s: s.std() * np.sqrt(ANN),
        n="count",
    ).reset_index()
    stats = stats[~stats["name"].isin(["石油石化", "环保", "美容护理"])]
    stats = stats.sort_values("ann_ret")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(stats["ann_vol"], stats["ann_ret"], s=42, color=PALETTE["primary"],
               alpha=0.85, edgecolor="white", linewidth=0.6, zorder=3)
    texts = [
        ax.text(r["ann_vol"], r["ann_ret"], r["name"], fontsize=8.2, color=PALETTE["dark"],
                ha="left", va="center")
        for _, r in stats.iterrows()
    ]
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color=PALETTE["neutral"], lw=0.5))
    ax.axhline(0, color=PALETTE["neutral"], lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("年化波动率（日收益标准差 × √252）")
    ax.set_ylabel("年化收益（复利）")
    ax.set_title("图 9　行业风险-收益分布（2014-02-21 至 2026-11-17）", fontsize=12, pad=12)
    ax.grid(alpha=0.35)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.text(0.01, -0.11, "注：点数 n = 28；收益与波动均由申万一级行业指数日收盘计算，未剔除异常值。",
            transform=ax.transAxes, fontsize=8.2, color=PALETTE["neutral"])
    save(fig, "fig09_risk_return.png")


if __name__ == "__main__":
    main()
