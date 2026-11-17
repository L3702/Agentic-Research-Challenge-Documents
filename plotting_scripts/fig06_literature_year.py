# -*- coding: utf-8 -*-
"""Fig 6 文献年份分布：领域研究热度随时间变化。"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from _style import PALETTE, ROOT, save


def main():
    path = os.path.join(ROOT, "documents", "Literature_Matrix.xlsx")
    df = pd.read_excel(path, sheet_name="文献矩阵")
    df["年份"] = pd.to_numeric(df["年份"], errors="coerce")
    counts = df["年份"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(counts.index.astype(int).astype(str), counts.values,
           color=PALETTE["primary"], alpha=0.85, width=0.66)
    for x, v in zip(range(len(counts)), counts.values):
        ax.text(x, v + 0.15, str(v), ha="center", fontsize=9, color=PALETTE["dark"])
    ax.set_xlabel("发表年份")
    ax.set_ylabel("文献数量（篇）")
    ax.set_title("图 6　文献年份分布（2007–2026，n = 38）", fontsize=12, pad=12)
    ax.set_ylim(0, counts.max() + 1.6)
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)
    ax.text(0.01, -0.12, "注：2025–2026 年文献集中，反映该方向近两年活跃度上升；样本为工作区已整理文献，非全量检索结果。",
            transform=ax.transAxes, fontsize=8.2, color=PALETTE["neutral"])
    save(fig, "fig06_literature_year.png")


if __name__ == "__main__":
    main()
