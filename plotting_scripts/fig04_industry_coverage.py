# -*- coding: utf-8 -*-
"""Fig 4 申万一级行业数据覆盖：真实抓取的日期范围。"""

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from _style import DATA_DIR, PALETTE, save


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "sw_industries", "sw_industry_daily.csv"))
    df["date"] = pd.to_datetime(df["date"])
    g = df.groupby(["code", "name"])["date"].agg(["min", "max", "count"]).reset_index()
    g = g.sort_values(["min", "code"], ascending=[False, True]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 9))
    for i, row in g.iterrows():
        y = len(g) - 1 - i
        start, end = row["min"], row["max"]
        if start.year <= 2000:
            color = PALETTE["primary"]
            label = "1999 起（15 个老行业）"
        elif start.year <= 2015:
            color = PALETTE["secondary"]
            label = "2014-02-21 起（13 个分类调整行业）"
        else:
            color = PALETTE["tertiary"]
            label = "2021-12-13 起（3 个新分类行业）"
        ax.barh(y, (end - start).days, left=start, height=0.62, color=color, alpha=0.85)
        ax.text(end + pd.Timedelta(days=40), y, f"{row['count']:,} 条",
                va="center", fontsize=7.4, color=PALETTE["neutral"])

    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([f"{r['name']} {r['code']}" for _, r in g.iterrows()][::-1], fontsize=8.6)
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(pd.Timestamp("1999-01-01"), pd.Timestamp("2028-06-01"))
    ax.set_xlabel("数据区间（起止日期）")
    ax.set_title("图 4　申万一级 31 行业日行情数据覆盖（akshare 真实抓取）", fontsize=12, pad=12)
    ax.grid(axis="x", alpha=0.4)
    ax.set_axisbelow(True)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["primary"], alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["secondary"], alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["tertiary"], alpha=0.85),
    ]
    ax.legend(handles, ["1999 起（15 个老行业）", "2014-02-21 起（13 个）", "2021-12-13 起（3 个）"],
              loc="lower right", fontsize=8.6, frameon=True)
    ax.text(0.01, -0.09, "注：2021 年起的 3 个行业（石油石化、环保、美容护理）历史较短，描述统计与实验需单独说明口径。",
            transform=ax.transAxes, fontsize=8.2, color=PALETTE["neutral"])
    save(fig, "fig04_industry_coverage.png")


if __name__ == "__main__":
    main()
