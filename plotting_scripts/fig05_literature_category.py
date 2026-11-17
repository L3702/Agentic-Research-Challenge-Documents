# -*- coding: utf-8 -*-
"""Fig 5 文献分类分布：来自 Literature_Matrix.xlsx 的 38 篇文献。"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from _style import CAT_COLORS, PALETTE, ROOT, save


def main():
    path = os.path.join(ROOT, "documents", "Literature_Matrix.xlsx")
    df = pd.read_excel(path, sheet_name="文献矩阵")
    order = ["01 核心：MPR文本与股票市场", "02 央行沟通传导：宏观风险与预期",
             "03 货币政策与资产价格", "04 文本情绪度量与LLM方法",
             "05 情绪因子与量化策略", "06 量化模型与行业轮动", "07 国外文献"]
    counts = df["分类"].value_counts().reindex(order)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    y = range(len(counts))[::-1]
    bars = ax.barh(list(y), counts.values, color=CAT_COLORS[:len(counts)], alpha=0.88, height=0.62)
    ax.set_yticks(list(y))
    ax.set_yticklabels([c.split(" ", 1)[1] for c in counts.index], fontsize=9.6)
    for b, v in zip(bars, counts.values):
        ax.text(v + 0.15, b.get_y() + b.get_height() / 2, str(v),
                va="center", fontsize=9.5, color=PALETTE["dark"])
    ax.set_xlabel("文献数量（篇）")
    ax.set_title("图 5　文献分类分布（n = 38，2026-11-17 版矩阵）", fontsize=12, pad=12)
    ax.set_xlim(0, counts.max() + 1.6)
    ax.grid(axis="x", alpha=0.4)
    ax.set_axisbelow(True)
    ax.text(0.01, -0.12, "注：分类来自工作区 documents/Literature_Matrix.xlsx「分类」字段，7 类互斥。",
            transform=ax.transAxes, fontsize=8.2, color=PALETTE["neutral"])
    save(fig, "fig05_literature_category.png")


if __name__ == "__main__":
    main()
