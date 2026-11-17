# -*- coding: utf-8 -*-
"""Fig 12 市场基准走势（沪深300 / 中证全指，真实行情）。"""

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from _style import DATA_DIR, PALETTE, save

EVENTS = [
    ("2015-06-12", "2015 股灾"),
    ("2018-01-26", "2018 贸易摩擦"),
    ("2020-01-23", "2020 疫情"),
    ("2024-09-24", "2024 政策转向"),
]


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "benchmark", "benchmark_daily.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
    df["nav"] = df.groupby("code")["close"].transform(lambda s: s / s.iloc[0] * 1000)
    pivot = df.pivot(index="date", columns="name", values="nav")

    fig, ax = plt.subplots(figsize=(11, 5.4))
    for col, color in zip(["沪深300", "中证全指"], [PALETTE["primary"], PALETTE["secondary"]]):
        if col in pivot.columns:
            ax.plot(pivot.index, pivot[col], lw=1.5, label=col, color=color)
    for d, label in EVENTS:
        x = pd.Timestamp(d)
        if x in pivot.index:
            y = pivot.loc[x].mean()
            ax.axvline(x, color=PALETTE["neutral"], lw=0.7, ls=":", alpha=0.7)
            ax.text(x, ax.get_ylim()[1] * 0.02 + 60, label, rotation=90, fontsize=7.6,
                    va="bottom", ha="right", color=PALETTE["neutral"])
    ax.set_ylabel("归一化净值（起始 = 1000）")
    ax.set_xlabel("日期")
    ax.set_title("图 12　市场基准走势（真实行情，全样本起点归一）", fontsize=12, pad=12)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(alpha=0.35)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save(fig, "fig12_benchmark.png")


if __name__ == "__main__":
    main()
