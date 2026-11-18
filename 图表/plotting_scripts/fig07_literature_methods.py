# -*- coding: utf-8 -*-
"""Fig 7 文献方法族分布：对「研究方法与模型」字段做关键词归类（可多选）。

归类规则显式列出，便于复核；一篇文献可计入多个方法族。
"""

import os
import re

import matplotlib.pyplot as plt
import pandas as pd

from _style import CAT_COLORS, PALETTE, PROJECT_ROOT, save

RULES = {
    "词典法与文本特征": r"词典|词频|情感单元|关键措辞|Harvard|Loughran|TF-IDF|可读性|相似度|措辞|语调",
    "主题模型": r"LDA|主题模型|HDMR|rSCHOLAR|SCHOLAR",
    "预训练模型": r"BERT|FinBERT|RoBERTa|Word2Vec|预训练",
    "大语言模型": r"LLM|大模型|大语言模型|DeepSeek|Qwen|ChatGPT|GPT|通义",
    "事件研究": r"事件研究|事件窗|CAR|异常收益",
    "面板与回归": r"面板|固定效应|回归|Fama-MacBeth|OLS|中介效应|调节效应",
    "结构模型与VAR": r"SVAR|TVP-VAR|TVP-FAVAR|VAR|DSGE|脉冲响应|方差分解",
    "波动率模型": r"GARCH|EGARCH|DCC|MIDAS",
    "机器学习": r"随机森林|XGBoost|LightGBM|SVM|支持向量|逻辑回归|机器学习",
    "深度学习": r"LSTM|BiLSTM|Informer|TCN|图神经|强化学习|DDPG|PPO|深度学习",
}


def main():
    path = os.path.join(PROJECT_ROOT, "documents", "Literature_Matrix.xlsx")
    df = pd.read_excel(path, sheet_name="文献矩阵")
    text = df["研究方法与模型"].fillna("").astype(str)
    counts = {}
    for name, pat in RULES.items():
        counts[name] = int(text.str.contains(pat, regex=True).sum())
    s = pd.Series(counts).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = range(len(s))
    bars = ax.barh(list(y), s.values, color=CAT_COLORS[:len(s)], alpha=0.88, height=0.62)
    ax.set_yticks(list(y))
    ax.set_yticklabels(s.index, fontsize=9.6)
    for b, v in zip(bars, s.values):
        ax.text(v + 0.15, b.get_y() + b.get_height() / 2, str(v),
                va="center", fontsize=9.5, color=PALETTE["dark"])
    ax.set_xlabel("命中文献数（篇，可多选）")
    ax.set_title("图 7　文献方法族分布（n = 38，关键词归类）", fontsize=12, pad=12)
    ax.set_xlim(0, s.max() + 2.2)
    ax.grid(axis="x", alpha=0.4)
    ax.set_axisbelow(True)
    fig.subplots_adjust(bottom=0.16)
    fig.text(0.01, 0.02,
             "注：规则为脚本内显式关键词表；单篇文献可计入多个方法族，故合计大于 38。",
             fontsize=8.2, color=PALETTE["neutral"])
    save(fig, "fig07_literature_methods.png")


if __name__ == "__main__":
    main()
