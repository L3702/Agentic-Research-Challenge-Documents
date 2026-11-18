# -*- coding: utf-8 -*-
"""统一样式：字体、配色、导出设置。所有绘图脚本共享。"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 项目根：脚本位于 <项目根>/图表/plotting_scripts/，需向上三级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 向后兼容别名：旧脚本仍写 ROOT
ROOT = PROJECT_ROOT
FIG_DIR = os.path.join(PROJECT_ROOT, "图表", "figures")
TABLE_DIR = os.path.join(PROJECT_ROOT, "图表", "tables")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "external")

# 中文字体：优先微软雅黑 / 等线 / 黑体
for _f in ["Microsoft YaHei", "DengXian", "SimHei", "Adobe Song Std"]:
    try:
        rcParams["font.sans-serif"] = [_f]
        break
    except Exception:  # noqa: BLE001
        continue
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 10.5
rcParams["axes.titlesize"] = 12
rcParams["axes.labelsize"] = 10.5
rcParams["legend.fontsize"] = 9
rcParams["xtick.labelsize"] = 9
rcParams["ytick.labelsize"] = 9
rcParams["figure.dpi"] = 150
rcParams["savefig.dpi"] = 300
rcParams["savefig.bbox"] = "tight"
rcParams["axes.edgecolor"] = "#4a4a4a"
rcParams["axes.linewidth"] = 0.8
rcParams["grid.color"] = "#d9d9d9"
rcParams["grid.linewidth"] = 0.6

# 学术配色：靛青主色 + 赭橙对照 + 灰蓝辅助，避免单色系与渐变
PALETTE = {
    "primary": "#2f5d7c",
    "secondary": "#c46a3d",
    "tertiary": "#5b8c5a",
    "quaternary": "#8a6d3b",
    "neutral": "#6b7280",
    "light": "#e8eef2",
    "dark": "#263238",
}
CAT_COLORS = ["#2f5d7c", "#c46a3d", "#5b8c5a", "#8a6d3b", "#7b5ea7", "#b04a6a", "#4a7c8a"]


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, format="png")
    plt.close(fig)
    print(f"saved: {path}")
    return path
