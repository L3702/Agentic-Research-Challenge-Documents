"""Prepare manuscript figures without changing the archived source images."""

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
SOURCE = ROOT.parent / "图表" / "figures"
TOP_CROPS = {
    "fig01_research_framework.png": 100,
    "fig14_event_study_car.png": 65,
    "fig09_risk_return.png": 95,
}
BOTTOM_CROPS = {"fig14_event_study_car.png": 52}
manifest = []
for name, top in TOP_CROPS.items():
    path = SOURCE / name
    image = Image.open(path)
    bottom = BOTTOM_CROPS.get(name, 0)
    image.crop((0, top, image.width, image.height - bottom)).save(OUT / name)
    manifest.append({"input": str(path.relative_to(ROOT.parent)),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "output": "figures/" + name, "removed_top_pixels": top,
                     "removed_bottom_pixels": bottom})

plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 10,
                     "axes.unicode_minus": False, "pdf.fonttype": 42})

datafile = ROOT.parent / "experiments/E1/sentiment_dictionary_full.csv"
with datafile.open(encoding="utf-8-sig", newline="") as stream:
    rows = sorted(
        (row for row in csv.DictReader(stream) if row["dictionary"] == "ntusd_simplified"),
        key=lambda row: row["publication_date"],
    )
dates = [datetime.fromisoformat(row["publication_date"]) for row in rows]
scores = np.array([float(row["sentiment_score"]) for row in rows])
coverage = [float(row["pos_neg_coverage"]) for row in rows]
# Match the paper's 0.88 fraction of the 164 mm text width.
with plt.rc_context({"font.size": 8}):
    figure, (tone, hits) = plt.subplots(
        2, 1, figsize=(144.32 / 25.4, 82 / 25.4), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1]}, layout="constrained",
    )
    tone.axhline(0, color="#6b7280", linestyle="--", linewidth=0.7)
    tone.plot(dates, scores, color="#2f5d7c", linewidth=1.2)
    tone.fill_between(dates, scores, 0, where=scores >= 0,
                      color="#2f5d7c", alpha=0.25, interpolate=True)
    tone.fill_between(dates, scores, 0, where=scores < 0,
                      color="#c46a3d", alpha=0.25, interpolate=True)
    tone.set_ylabel("词频情绪得分（NTUSD 简体）")
    hits.bar(dates, coverage, width=20, color="#5b8c5a", alpha=0.85)
    hits.set_ylabel("正负词\n覆盖率")
    hits.set_xlabel("报告发布日期")
    hits.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    hits.xaxis.set_major_locator(mdates.YearLocator(2))
    hits.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    hits.margins(x=0.035)
    for axis in (tone, hits):
        axis.grid(color="#e3e3e3", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    figure.align_ylabels((tone, hits))
    figure.savefig(OUT / "fig13_sentiment.png", dpi=500)
    plt.close(figure)
manifest.append({"input": str(datafile.relative_to(ROOT.parent)),
                 "sha256": hashlib.sha256(datafile.read_bytes()).hexdigest(),
                 "output": "figures/fig13_sentiment.png",
                 "columns": ["publication_date", "sentiment_score", "pos_neg_coverage"],
                 "filter": {"dictionary": "ntusd_simplified"}, "rows": len(rows)})

datafile = ROOT.parent / "experiments/E5/e5_heterogeneity.csv"
with datafile.open(encoding="utf-8-sig", newline="") as stream:
    rows = sorted(csv.DictReader(stream), key=lambda row: float(row["sens_main"]))
beta = np.array([float(r["sens_main"]) for r in rows])
low = np.array([float(r["sens_main_ci_low"]) for r in rows])
high = np.array([float(r["sens_main_ci_high"]) for r in rows])
figure, axis = plt.subplots(figsize=(6.2, 7.0), layout="constrained")
y = np.arange(len(rows))
axis.barh(y, beta, height=0.62,
          color=np.where(beta >= 0, "#ad674b", "#38657d"))
axis.errorbar(beta, y, xerr=[beta - low, high - beta], fmt="none",
              ecolor="#707880", elinewidth=0.8, capsize=2)
axis.set_yticks(y, [r["name"] for r in rows])
axis.tick_params(axis="y", length=0, labelsize=9.5)
axis.set_xlabel("行业 CAR 对报告情绪得分的回归斜率")
axis.axvline(0, color="#7a7a7a", linestyle="--", linewidth=0.7)
axis.grid(axis="x", color="#e3e3e3", linewidth=0.5)
axis.set_axisbelow(True)
axis.margins(y=0.015)
axis.spines[["top", "right", "left"]].set_visible(False)
figure.savefig(OUT / "fig19_heterogeneity.pdf", bbox_inches="tight")
plt.close(figure)
manifest.append({"input": str(datafile.relative_to(ROOT.parent)),
                 "sha256": hashlib.sha256(datafile.read_bytes()).hexdigest(),
                 "output": "figures/fig19_heterogeneity.pdf",
                 "columns": ["name", "sens_main", "sens_main_ci_low", "sens_main_ci_high"]})

datafile = ROOT.parent / "experiments/E7/e7_portfolio.csv"
with datafile.open(encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))
dates = [datetime(2016, 3, 31)] + [datetime.fromisoformat(r["target_date"]) for r in rows]
figure, axis = plt.subplots(figsize=(8.0, 4.2), layout="constrained")
series = [
    ("long_short_net", "文本多空（净成本）", "#38657d", "-"),
    ("baseline_long_short_net", "无文本多空（净成本）", "#ad674b", "-"),
    ("industry_ew_return", "同池行业等权", "#57865c", "--"),
]
for column, label, color, style in series:
    values = np.array([float(row[column]) for row in rows])
    nav = np.r_[1000.0, 1000.0 * np.cumprod(1.0 + values)]
    axis.plot(dates, nav, label=label, color=color, linestyle=style, linewidth=1.4)
axis.set_ylabel("累计净值（期初 1000）")
axis.set_xlabel("持有期末")
axis.xaxis.set_major_locator(mdates.YearLocator(2))
axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axis.legend(frameon=False, loc="upper left", fontsize=9.5)
axis.axhline(1000, color="#999999", linewidth=0.6, linestyle=":")
axis.grid(color="#e3e3e3", linewidth=0.5)
axis.spines[["top", "right"]].set_visible(False)
figure.savefig(OUT / "fig18_portfolio_equity.pdf", bbox_inches="tight")
plt.close(figure)
manifest.append({"input": str(datafile.relative_to(ROOT.parent)),
                 "sha256": hashlib.sha256(datafile.read_bytes()).hexdigest(),
                 "output": "figures/fig18_portfolio_equity.pdf",
                 "columns": [item[0] for item in series]})
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print("Prepared three title-cropped figures and redrew the E1, E5, and E7 figures.")
