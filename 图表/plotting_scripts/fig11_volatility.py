# -*- coding: utf-8 -*-
"""Fig 11 行业年化波动率对比（2014-02-21 起，28 个可比行业）。"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _style import CAT_COLORS, DATA_DIR, PALETTE, save

START = "2014-02-21"
ANN = 252


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "sw_industries", "sw_industry_daily.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df["ret"] = df.groupby("code")["收盘"].pct_change()
    df = df[df["date"] >= START].sort_values(["code", "date"])
    df = df.dropna(subset=["ret"])
    vol = df.groupby("name")["ret"].std() * np.sqrt(ANN)
    vol = vol.drop(["石油石化", "环保", "美容护理"], errors="ignore").sort_values()

    fig, ax = plt.subplots(figsize=(10, 6.4))
    colors = [PALETTE["secondary"] if v > vol.median() else PALETTE["primary"] for v in vol.values]
    ax.barh(range(len(vol)), vol.values, color=colors, alpha=0.88, height=0.66)
    ax.set_yticks(range(len(vol)))
    ax.set_yticklabels(vol.index, fontsize=9)
    for i, v in enumerate(vol.values):
        ax.text(v + 0.003, i, f"{v:.1%}", va="center", fontsize=8, color=PALETTE["dark"])
    ax.axvline(vol.median(), color=PALETTE["neutral"], lw=1, ls="--")
    ax.text(vol.median() + 0.002, len(vol) - 0.5, f"中位数 {vol.median():.1%}",
            fontsize=8.4, color=PALETTE["neutral"])
    ax.set_xlabel("年化波动率（日收益标准差 × √252）")
    ax.set_title("图 11　行业年化波动率排序（2014-02-21 至 2026-11-17）", fontsize=12, pad=12)
    ax.set_xlim(0, vol.max() * 1.12)
    ax.grid(axis="x", alpha=0.4)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.text(0.01, -0.11, "注：高于中位数的行业用赭色标出；已剔除 2021 年后新分类的 3 个行业。",
            transform=ax.transAxes, fontsize=8.2, color=PALETTE["neutral"])
    save(fig, "fig11_volatility.png")


if __name__ == "__main__":
    main()
