# -*- coding: utf-8 -*-
"""Fig 10 行业日收益相关性热力图（2014-02-21 起，28 个可比行业）。"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _style import DATA_DIR, PALETTE, save

START = "2014-02-21"
SHORT = {
    "农林牧渔": "农林牧渔", "基础化工": "基础化工", "钢铁": "钢铁", "有色金属": "有色",
    "电子": "电子", "家用电器": "家电", "食品饮料": "食品饮料", "纺织服饰": "纺服",
    "轻工制造": "轻工", "医药生物": "医药", "公用事业": "公用", "交通运输": "交运",
    "房地产": "地产", "商贸零售": "商贸", "社会服务": "社服", "综合": "综合",
    "建筑材料": "建材", "建筑装饰": "建装", "电力设备": "电设", "国防军工": "军工",
    "计算机": "计算机", "传媒": "传媒", "通信": "通信", "银行": "银行",
    "非银金融": "非银", "汽车": "汽车", "机械设备": "机械", "煤炭": "煤炭",
}


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "sw_industries", "sw_industry_daily.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df["ret"] = df.groupby("code")["收盘"].pct_change()
    df = df[df["date"] >= START].sort_values(["code", "date"])
    df = df.dropna(subset=["ret"])
    wide = df.pivot(index="date", columns="name", values="ret")
    wide = wide[[c for c in wide.columns if c not in ("石油石化", "环保", "美容护理")]]
    wide = wide[[c for c in SHORT if c in wide.columns]]
    corr = wide.corr()
    labels = [SHORT[c] for c in corr.columns]

    fig, ax = plt.subplots(figsize=(10.5, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.55 else PALETTE["dark"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson 相关系数", fontsize=9)
    ax.set_title("图 10　行业日收益相关性（2014-02-21 至 2026-11-17，n = 28）", fontsize=12, pad=12)
    save(fig, "fig10_correlation.png")


if __name__ == "__main__":
    main()
