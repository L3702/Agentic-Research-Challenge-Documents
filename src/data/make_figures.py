"""
make_figures.py — 生成论文配图
=============================
输出（docs/figures/）
    fig1_tone_series.png   情绪走势（对应 3.1 节）
    fig2_scatter.png       情绪与振幅/绝对收益散点（5.2 节）
    fig3_quantile.png      三分位分组柱状图（5.2 节）
    fig4_break.png         2014 前后系数对比（5.3 节）
    fig5_robust.png        稳健性系数（5.5 节）
"""

from __future__ import annotations

import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)

# ---- 中文字体 ----
CANDIDATES = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
              r"C:\Windows\Fonts\simsun.ttc"]
FONT = next((p for p in CANDIDATES if os.path.exists(p)), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    CN = font_manager.FontProperties(fname=FONT).get_name()
else:
    CN = "DejaVu Sans"
plt.rcParams["font.family"] = CN
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.edgecolor"] = "#444444"
plt.rcParams["axes.linewidth"] = 0.8
GREY, ACCENT = "#4d4d4d", "#1a1a1a"

# ---- 数据 ----
s = pd.read_csv("data/processed/sentiment.csv")
mk = pd.read_csv("data/processed/market_events.csv")
df = s.merge(mk, on="period")
df["abs1"] = df.r16_m1_0.abs()

spec = importlib.util.spec_from_file_location("h1", "src/data/run_h1.py")
h1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h1)


def cell(d, y, xs, var, lag=4):
    r = h1.fit(d, y, xs, lag)
    if r is None:
        return None
    row = r[r["var"] == var].iloc[0]
    se = abs(row.coef / row.t) if row.t != 0 else np.nan
    return row.coef * 10000, se * 10000, row.t


# ============ 图 1 情绪走势 ============
fig, ax = plt.subplots(figsize=(7.2, 2.9))
x = pd.PeriodIndex(df.period, freq="Q").to_timestamp()
ax.plot(x, df.tone_P, color=ACCENT, lw=1.3, label="政策指引段（tone_P）")
ax.plot(x, df.tone_R, color=GREY, lw=1.0, ls="--", label="宏观回顾段（tone_R）")
ax.axvline(pd.Timestamp("2014-01-01"), color="#999999", ls=":", lw=1.0)
ax.text(pd.Timestamp("2014-04-01"), 0.83, "2014 断点", fontsize=8, color="#666666")
for yr in ("2008-10-01", "2015-07-01", "2020-01-01"):
    ax.axvspan(pd.Timestamp(yr), pd.Timestamp(yr) + pd.DateOffset(months=9),
               color="#dddddd", alpha=0.5, lw=0)
ax.set_ylabel("情绪值", fontsize=9)
ax.set_ylim(0.25, 0.90)
ax.tick_params(labelsize=8)
ax.legend(fontsize=8, frameon=False, loc="lower right", ncol=2)
ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in (2001, 2005, 2010, 2015, 2020, 2026)])
ax.set_xticklabels(["2001", "2005", "2010", "2015", "2020", "2026"])
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.savefig(f"{OUT}/fig1_tone_series.png")
plt.close(fig)

# ============ 图 2 散点 ============
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, ycol, ylab, yscale in [
        (axes[0], "rng16_evt", "事件日日内振幅（%）", 100),
        (axes[1], "abs1", "事件日绝对收益（%）", 100)]:
    d = df.dropna(subset=["z_tone_P", ycol])
    ax.scatter(d.z_tone_P, d[ycol] * yscale, s=22, facecolors="none",
               edgecolors="#555555", linewidths=0.7, alpha=0.8)
    b, a = np.polyfit(d.z_tone_P, d[ycol] * yscale, 1)
    grid = np.linspace(d.z_tone_P.min(), d.z_tone_P.max(), 50)
    pred = a + b * grid
    resid = d[ycol] * yscale - (a + b * d.z_tone_P)
    se = np.sqrt((resid ** 2).sum() / (len(d) - 2)) * np.sqrt(
        1 / len(d) + (grid - d.z_tone_P.mean()) ** 2 / ((d.z_tone_P - d.z_tone_P.mean()) ** 2).sum())
    ax.plot(grid, pred, color=ACCENT, lw=1.2)
    ax.fill_between(grid, pred - 1.96 * se, pred + 1.96 * se, color="#dddddd", alpha=0.6, lw=0)
    ax.set_xlabel("政策指引段情绪（标准化）", fontsize=9)
    ax.set_ylabel(ylab, fontsize=9)
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_title("(1) 日内振幅", fontsize=9, loc="left")
axes[1].set_title("(2) 绝对收益", fontsize=9, loc="left")
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_scatter.png")
plt.close(fig)

# ============ 图 3 三分位 ============
d = df.dropna(subset=["z_tone_P", "rng16_evt"]).copy()
d["grp"] = pd.qcut(d.z_tone_P, 3, labels=["低", "中", "高"])
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
for ax, col, ylab, sc in [(axes[0], "rng16_evt", "平均日内振幅（bp）", 10000),
                          (axes[1], "r16_1d", "平均一日收益（bp）", 10000)]:
    g = d.groupby("grp", observed=True)[col]
    m, se, n = g.mean() * sc, g.sem() * sc, g.count()
    ax.bar(range(3), m.values, yerr=1.96 * se.values, capsize=3,
           color=["#c9c9c9", "#8f8f8f", "#3d3d3d"], edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{i}\n(n={k})" for i, k in zip(m.index, n)], fontsize=8)
    ax.set_ylabel(ylab, fontsize=9)
    ax.axhline(0, color="black", lw=0.6)
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_title("(1) 低情绪组振幅最高", fontsize=9, loc="left")
axes[1].set_title("(2) 各组收益无显著差异", fontsize=9, loc="left")
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_quantile.png")
plt.close(fig)

# ============ 图 4 断点前后 ============
dd = df.copy()
dd["year"] = dd.period.str[:4].astype(int)
pre_d, post_d = dd[dd.year < 2014], dd[dd.year >= 2014]
rows = []
for y, lab in [("r16_1d", "一日收益"), ("r16_5d", "五日收益"),
               ("rng16_evt", "日内振幅"), ("abs1", "绝对收益")]:
    b1 = cell(pre_d, y, ["z_tone_P"], "z_tone_P")
    b2 = cell(post_d, y, ["z_tone_P"], "z_tone_P")
    if b1 and b2:
        rows.append((lab, b1[0], b1[1], b2[0], b2[1]))

fig, ax = plt.subplots(figsize=(6.4, 3.0))
ypos = np.arange(len(rows))
for i, (lab, b1, s1, b2, s2) in enumerate(rows):
    ax.errorbar(b1, i + 0.15, xerr=1.96 * s1, fmt="o", color=ACCENT, ms=4,
                capsize=3, lw=0.9, label="2014 年以前" if i == 0 else None)
    ax.errorbar(b2, i - 0.15, xerr=1.96 * s2, fmt="s", color="#8a8a8a", ms=4,
                capsize=3, lw=0.9, label="2014 年及以后" if i == 0 else None)
ax.axvline(0, color="black", lw=0.7, ls="--")
ax.set_yticks(ypos)
ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.set_xlabel("情绪对市场反应的影响（基点）", fontsize=9)
ax.tick_params(labelsize=8)
ax.legend(fontsize=8, frameon=False, loc="upper left")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_break.png")
plt.close(fig)

# ============ 图 5 稳健性 ============
alt = pd.read_csv("data/processed/sentiment_alt.csv")[["period", "z_tone_P"]].rename(
    columns={"z_tone_P": "zP_alt"})
pc = pd.read_csv("data/processed/policy_controls.csv")
base = df.copy()
base["year"] = base.period.str[:4].astype(int)
base["post14"] = (base.year >= 2014).astype(float)
base["zx"] = base.z_tone_P * base.post14
base = base.merge(alt, on="period", how="left").merge(pc, on="period", how="left")

settings = [
    ("基准", base, "z_tone_P", "rng16_evt", "zx", ["z_tone_P", "zx", "post14"], "r16_1d"),
    ("28 行业", base, "z_tone_P", "rng28_evt", "zx", ["z_tone_P", "zx", "post14"], "r28_1d"),
    ("替代词典", base, "zP_alt", "rng16_evt", "zx_alt", ["zP_alt", "zx_alt", "post14"], "r16_1d"),
    ("剔除 2015", base[base.year != 2015], "z_tone_P", "rng16_evt", "zx", ["z_tone_P", "zx", "post14"], "r16_1d"),
    ("剔除 2008–09", base[~base.year.isin([2008, 2009])], "z_tone_P", "rng16_evt", "zx", ["z_tone_P", "zx", "post14"], "r16_1d"),
    ("控制政策操作", base, "z_tone_P", "rng16_evt", "zx",
     ["z_tone_P", "zx", "post14", "rrr_cut", "lpr_cut", "omo_cut"], "r16_1d"),
]

res = []
for name, d, zp, ycol, zx, xs, yret in settings:
    d = d.copy()
    if "zP_alt" in d and zp == "zP_alt":
        d["zx_alt"] = d.zP_alt * d.post14
    b = cell(d, ycol, [zp], zp)
    c = cell(d, yret, xs, zx)
    if b and c:
        res.append((name, b[0], b[1], c[0], c[1]))

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
ypos = np.arange(len(res))
for ax, idx, title in [(axes[0], 1, "(1) 振幅方程"), (axes[1], 3, "(2) 断点交互项")]:
    vals = [r[idx] for r in res]
    errs = [1.96 * r[idx + 1] for r in res]
    ax.errorbar(vals, ypos, xerr=errs, fmt="o", color=ACCENT, ms=4, capsize=3, lw=0.9)
    ax.axvline(0, color="black", lw=0.7, ls="--")
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel("基点", fontsize=9)
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_yticks(ypos)
axes[0].set_yticklabels([r[0] for r in res], fontsize=8.5)
axes[0].invert_yaxis()
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_robust.png")
plt.close(fig)

print("[完成] 5 张图写入", OUT)
for f in sorted(os.listdir(OUT)):
    print("   ", f, os.path.getsize(os.path.join(OUT, f)) // 1024, "KB")
