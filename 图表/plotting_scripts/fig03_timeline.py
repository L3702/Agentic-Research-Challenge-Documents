# -*- coding: utf-8 -*-
"""Fig 3 数据管线与时间对齐示意：事件日 = MPR 发布日期。"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _style import PALETTE, save


def main():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(-1.2, 11.2)
    ax.set_ylim(-0.4, 4.2)
    ax.axis("off")
    ax.set_title("图 3　MPR 发布日与行业收益的时间对齐", fontsize=13, pad=12)

    # 时间轴
    ax.annotate("", xy=(10.8, 2.4), xytext=(-1.0, 2.4),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color=PALETTE["dark"]))
    ax.text(10.9, 2.4, "时间", va="center", fontsize=9, color=PALETTE["dark"])

    def tick(x, label, color=PALETTE["dark"], dy=0.0):
        ax.plot([x, x], [2.32, 2.48], color=color, lw=1.2)
        ax.text(x, 2.62 + dy, label, ha="center", va="bottom", fontsize=9, color=color)

    tick(0.0, "报告期截止")
    tick(1.2, "估计窗\n(前120个交易日)", color=PALETTE["neutral"], dy=0.28)
    tick(5.0, "发布日 t=0\n(事件日)", color=PALETTE["secondary"], dy=0.28)
    tick(7.4, "事件窗\n[0,+1]…[0,+10]", color=PALETTE["primary"], dy=0.28)
    tick(10.2, "下一季度\n行业收益", color=PALETTE["tertiary"], dy=0.28)

    # 估计窗阴影
    ax.axvspan(1.2, 5.0, color=PALETTE["light"], alpha=0.7, zorder=0)
    ax.text(3.1, 2.1, "正常收益模型估计窗", ha="center", fontsize=8.4, color=PALETTE["neutral"])

    # 事件窗括号
    ax.plot([5.0, 7.4], [2.05, 2.05], color=PALETTE["primary"], lw=1.2)
    ax.plot([5.0, 5.0], [2.05, 2.12], color=PALETTE["primary"], lw=1.2)
    ax.plot([7.4, 7.4], [2.05, 2.12], color=PALETTE["primary"], lw=1.2)
    ax.text(6.2, 1.92, "事件窗（发布后）", ha="center", fontsize=8.4, color=PALETTE["primary"])

    # 两个说明框
    box1 = FancyBboxPatch((0.1, 0.55), 5.2, 1.1, boxstyle="round,pad=0.02,rounding_size=0.02",
                          linewidth=1.0, edgecolor=PALETTE["secondary"], facecolor="#fdf1e7")
    ax.add_patch(box1)
    ax.text(2.7, 1.35, "事件日 = MPR 发布日期", ha="center", fontsize=10,
            fontweight="bold", color=PALETTE["secondary"])
    ax.text(2.7, 0.92, "报告期截止日 ≠ 事件日；解释变量须为发布日前已完成的文本度量",
            ha="center", fontsize=8.2, color=PALETTE["dark"])

    box2 = FancyBboxPatch((5.8, 0.55), 5.2, 1.1, boxstyle="round,pad=0.02,rounding_size=0.02",
                          linewidth=1.0, edgecolor=PALETTE["primary"], facecolor=PALETTE["light"])
    ax.add_patch(box2)
    ax.text(8.4, 1.35, "被解释变量 = 发布后收益", ha="center", fontsize=10,
            fontweight="bold", color=PALETTE["primary"])
    ax.text(8.4, 0.92, "事件窗 CAR、下一季度行业收益；宏观控制取发布日前最近可得值",
            ha="center", fontsize=8.2, color=PALETTE["dark"])

    save(fig, "fig03_timeline.png")


if __name__ == "__main__":
    main()
