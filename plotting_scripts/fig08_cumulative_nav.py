# -*- coding: utf-8 -*-
"""Fig 8 行业累计净值曲线（2014-02-21 起，28 个可比行业，真实行情）。"""

import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from _style import DATA_DIR, PALETTE, save

START = "2014-02-21"
HIGHLIGHT = {"801780": "银行", "801750": "计算机", "801120": "食品饮料", "801950": "煤炭"}


def load():
    df = pd.read_csv(os.path.join(DATA_DIR, "sw_industries", "sw_industry_daily.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= START].sort_values(["code", "date"])
    df["ret"] = df.groupby("code")["收盘"].pct_change()
    df = df.dropna(subset=["ret"])
    df["nav"] = df.groupby("code")["ret"].transform(lambda s: (1 + s).cumprod() * 1000)
    return df


def main():
    df = load()
    pivot = df.pivot(index="date", columns="name", values="nav")
    pivot = pivot.loc[:, [c for c in pivot.columns if c not in ("石油石化", "环保", "美容护理")]]

    fig, ax = plt.subplots(figsize=(11, 6))
    for col in pivot.columns:
        if col in HIGHLIGHT.values():
            continue
        ax.plot(pivot.index, pivot[col], color="#9aa7b0", lw=0.7, alpha=0.75, zorder=1)
    for code, name in HIGHLIGHT.items():
        if name in pivot.columns:
            ax.plot(pivot.index, pivot[name], lw=1.9, label=name, zorder=3)
    ax.set_yscale("log")
    ax.set_yticks([500, 750, 1000, 1500, 2000, 3000, 5000])
    ax.set_yticklabels(["500", "750", "1000", "1500", "2000", "3000", "5000"])
    ax.axhline(1000, color=PALETTE["neutral"], lw=0.8, ls="--", alpha=0.6)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("累计净值（2014-02-21 = 1000，对数轴）")
    ax.set_xlabel("日期")
    ax.set_title("图 8　申万一级行业累计净值（2014-02-21 至 2026-11-17，28 个可比行业）", fontsize=12, pad=12)
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.grid(alpha=0.35)
    ax.set_axisbelow(True)
    ax.text(0.01, -0.11, "注：灰色细线为其余行业；已剔除 2021-12-13 后新分类的石油石化、环保、美容护理。",
            transform=ax.transAxes, fontsize=8.2, color=PALETTE["neutral"])
    save(fig, "fig08_cumulative_nav.png")


if __name__ == "__main__":
    main()
