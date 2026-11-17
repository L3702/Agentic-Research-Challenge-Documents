# -*- coding: utf-8 -*-
"""Fig 1 研究框架图：研究问题 → 数据/度量 → 方法 → 假设与边界。"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from _style import PALETTE, save


def box(ax, x, y, w, h, text, fc="#f4f7f9", ec=PALETTE["primary"], fs=9.5, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                       linewidth=1.0, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", color=PALETTE["dark"])


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, linewidth=1.0, color=PALETTE["neutral"]))


def main():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("图 1　研究框架：MPR 文本情绪 × A 股行业收益", fontsize=13, pad=14)

    # 列标题
    heads = ["研究问题", "数据与度量", "方法与检验", "假设与边界"]
    xs = [0.3, 3.1, 5.9, 8.7]
    for x, h in zip(xs, heads):
        ax.text(x + 1.15, 5.75, h, ha="center", va="center", fontsize=11,
                fontweight="bold", color=PALETTE["primary"])

    # 行 1：Q1
    box(ax, 0.3, 4.55, 2.3, 0.85, "Q1 情绪能否被\n稳定可复现地度量？", fc=PALETTE["light"], bold=True)
    box(ax, 3.1, 4.55, 2.3, 0.85, "MPR 全文文本\n词典 / LLM 双轨评分\n人工抽检")
    box(ax, 5.9, 4.55, 2.3, 0.85, "词典间相关\n人工-自动一致性\n重复评分稳定性")
    box(ax, 8.7, 4.55, 2.0, 0.85, "H1 度量成立\n否则收益结论\n不可解释", fc="#fdf1e7")

    # 行 2：Q2
    box(ax, 0.3, 3.15, 2.3, 0.85, "Q2 情绪与随后行业收益\n是否关联、有预测信息？", fc=PALETTE["light"], bold=True)
    box(ax, 3.1, 3.15, 2.3, 0.85, "31 申万一级行业\n日/季收益、市场基准\n事件日 = 发布日")
    box(ax, 5.9, 3.15, 2.3, 0.85, "事件研究 CAR\nFama-MacBeth\n样本外 IC / OOS R²")
    box(ax, 8.7, 3.15, 2.0, 0.85, "H2 关联 + 预测\n不作因果表述", fc="#fdf1e7")

    # 行 3：Q3
    box(ax, 0.3, 1.75, 2.3, 0.85, "Q3 哪些行业更敏感？\n敏感度由什么决定？", fc=PALETTE["light"], bold=True)
    box(ax, 3.1, 1.75, 2.3, 0.85, "行业特征分组\n利率/融资/政策依赖\n（特征数据待定）")
    box(ax, 5.9, 1.75, 2.3, 0.85, "分行业回归\n情绪 × 特征交互项\n敏感度排序")
    box(ax, 8.7, 1.75, 2.0, 0.85, "H3 异质性关联\n非因果识别", fc="#fdf1e7")

    # 行 4：方法对照
    box(ax, 0.3, 0.35, 10.4, 0.85,
        "方法对照：词典法 vs LLM 评分（H4）　｜　Baseline：无文本模型、市场模型事件研究、行业历史均值　｜　边界：关联/预测，不做严格因果识别",
        fc="#eef3f0", ec=PALETTE["tertiary"], fs=9)

    # 纵向箭头
    for y in [5.4, 4.0]:
        arrow(ax, 1.45, y + 0.15, 1.45, y - 0.25)
    arrow(ax, 1.45, 2.6, 1.45, 2.2)

    # 横向箭头
    for y in [4.98, 3.58, 2.18]:
        for x1, x2 in [(2.62, 3.08), (5.42, 5.88), (8.22, 8.68)]:
            arrow(ax, x1, y, x2, y)

    save(fig, "fig01_research_framework.png")


if __name__ == "__main__":
    main()
