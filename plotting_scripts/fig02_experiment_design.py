# -*- coding: utf-8 -*-
"""Fig 2 实验设计总览：E0–E10 按 MVP / 正式 / 补充分组。"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from _style import PALETTE, save

GROUPS = [
    ("MVP（必须）", PALETTE["primary"], [
        ("E0", "数据链路 MVP\nMPR PDF → 文本 → 行业收益对齐", "data_sample/"),
        ("E1", "词典情绪度量\n+ 人工抽检 Baseline", "sentiment_dictionary_*.csv"),
        ("E2", "事件研究 Baseline\n无文本对照", "e2_car_plot.png"),
    ]),
    ("正式（必须）", PALETTE["secondary"], [
        ("E3", "正式事件研究\n高/中/低情绪组 CAR", "e3_car_plot.png"),
        ("E4", "横截面预测\nFama-MacBeth + 样本外", "e4_fmb_results.csv"),
        ("E5", "行业敏感度\n异质性检验", "e5_sensitivity_plot.png"),
    ]),
    ("增强（可选）", PALETTE["tertiary"], [
        ("E6", "LLM vs 词典\n增量信息比较", "e6_llm_vs_dict.csv"),
        ("E7", "高低情绪\n多空组合", "e7_equity_curve.png"),
    ]),
    ("补充（稳健性）", PALETTE["quaternary"], [
        ("E8", "稳健性\n词典/窗/子样本/基准", "e8_robustness_summary.csv"),
        ("E9", "LLM 敏感性\nPrompt/模型/解析", "e9_sensitivity_summary.csv"),
        ("E10", "误差分析\n异常与失败案例", "e10_error_analysis.md"),
    ]),
]


def main():
    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.set_title("图 2　实验设计总览（E0–E10）", fontsize=13, pad=14)

    y = 6.5
    for title, color, items in GROUPS:
        ax.text(0.25, y + 0.28, title, fontsize=11.5, fontweight="bold", color=color, va="center")
        n = len(items)
        width = 11.4 / n
        for i, (code, desc, out) in enumerate(items):
            x = 0.3 + i * width
            p = FancyBboxPatch((x, y - 0.75), width - 0.18, 1.25,
                               boxstyle="round,pad=0.02,rounding_size=0.02",
                               linewidth=1.0, edgecolor=color, facecolor="#f7f9fa")
            ax.add_patch(p)
            ax.text(x + (width - 0.18) / 2, y + 0.28, code, ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color=color)
            ax.text(x + (width - 0.18) / 2, y - 0.08, desc, ha="center", va="center",
                    fontsize=8.4, color=PALETTE["dark"])
            ax.text(x + (width - 0.18) / 2, y - 0.52, out, ha="center", va="center",
                    fontsize=7.2, color=PALETTE["neutral"])
        y -= 1.75

    ax.text(0.3, 0.12, "执行顺序：E0–E2 → E3–E5 → E6–E7（可选）→ E8–E10　｜　优先级以 EXPERIMENT_PLAN.md 为准",
            fontsize=9, color=PALETTE["neutral"])
    save(fig, "fig02_experiment_design.png")


if __name__ == "__main__":
    main()
