"""E8 robustness experiment: reproducible event and prediction comparisons.

All downloaded pages, intermediate observations and runtime caches live in
_temp/e8. Final results and the reproducible source live in experiments/E8.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "E8"
CACHE = ROOT / "_temp" / "e8"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(CACHE / "pycache"))
sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from PIL import Image
from statsmodels.stats.multitest import multipletests


SOURCES = [
    {
        "source_id": "statsmodels_hac",
        "title": "statsmodels cov_hac: Newey-West covariance",
        "url": "https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html",
        "role": "HAC inference and the lag parameter; observations must be chronologically ordered",
        "terms": ["Newey", "nlags", "spaced time periods"],
    },
    {
        "source_id": "statsmodels_multipletests",
        "title": "statsmodels multipletests: Holm and Benjamini-Hochberg",
        "url": "https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html",
        "role": "Retain raw p; report blockwise BH and dependence-robust Holm adjustments",
        "terms": ["holm", "fdr_bh", "positively correlated"],
    },
    {
        "source_id": "scipy_welch",
        "title": "SciPy ttest_ind: independent samples and Welch test",
        "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html",
        "role": "High-minus-low differences use event means and unequal-variance inference",
        "terms": ["equal_var", "Welch", "independent"],
    },
    {
        "source_id": "sklearn_time_split",
        "title": "scikit-learn TimeSeriesSplit: time-ordered validation",
        "url": "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html",
        "role": "Chronological expanding evaluation; E8 additionally purges unrealized labels",
        "terms": ["time-ordered", "future data", "gap"],
    },
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def fetch_sources(refresh: bool = False) -> pd.DataFrame:
    """Cache primary methodological documentation, without refreshing market data."""
    import requests
    from bs4 import BeautifulSoup

    target = CACHE / "sources"
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in SOURCES:
        row = {k: v for k, v in spec.items() if k != "terms"}
        path = target / (spec["source_id"] + ".html")
        meta = path.with_suffix(".json")
        try:
            if refresh or not path.exists():
                r = requests.get(spec["url"], timeout=35)
                r.raise_for_status()
                path.write_bytes(r.content)
                meta.write_text(json.dumps({"retrieved_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                                            "http_status": r.status_code,
                                            "resolved_url": r.url}, ensure_ascii=False, indent=2), encoding="utf-8")
            data = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
            soup = BeautifulSoup(path.read_bytes(), "html.parser")
            content = soup.find("main") or soup
            text = " ".join(content.stripped_strings)
            matches = {word: word.lower() in text.lower() for word in spec["terms"]}
            row.update(data)
            row.update(status="verified" if all(matches.values()) else "partial_term_match",
                       local_cache=str(path.relative_to(ROOT)), sha256=digest(path),
                       matched_terms=json.dumps(matches), failure_reason="")
            # Store the whole documentation as a cache; the report paraphrases only.
            path.with_suffix(".txt").write_text(text, encoding="utf-8")
        except Exception as exc:
            row.update(status="failed", failure_reason=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return pd.DataFrame(rows)


def manifest() -> pd.DataFrame:
    paths = [
        "EXPERIMENT_PLAN.md", "METHOD_PLAN.md",
        "experiments/E1/mpr_index_full.csv", "experiments/E1/sentiment_dictionary_full.csv",
        "data/external/sw_industries/sw_industry_daily.csv",
        "data/external/benchmark/sh000985_中证全指.csv",
        "data/external/benchmark/sh000300_沪深300.csv",
        "_temp/e0_market_raw/benchmark_sh000300.csv",
        "_temp/e4/e4_panel.csv", "_temp/e4/industry_quarterly.csv",
        "experiments/E3/e3_event_results.csv", "experiments/E3/e3_car_by_sentiment.csv",
        "experiments/E3/run_e3.py", "experiments/E4/run_e4.py",
        "experiments/E4/e4_fmb_results.csv", "experiments/E4/e4_oos_metrics.csv",
        "experiments/E5/e5_metrics.json", "experiments/E6/e6_metrics.json",
        "experiments/E7/e7_metrics.json", "experiments/E7/logs/e7_portfolio.md",
    ]
    rows = []
    for name in paths:
        p = ROOT / name
        rows.append(dict(path=name, exists=p.exists(), bytes=p.stat().st_size if p.exists() else 0,
                         sha256=digest(p) if p.exists() else "",
                         role="context_only" if "/E5/" in name or "/E6/" in name or "/E7/" in name else "input_or_reproduction_reference"))
    return pd.DataFrame(rows)


def adjust_tests(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["p_adjustment_family"] = "event_all_settings"
    pred = out["block"].eq("prediction")
    strict = out.get("timing_protocol", pd.Series("", index=out.index)).eq("strict_post_publication")
    out.loc[pred & ~strict, "p_adjustment_family"] = "prediction_legacy_all_settings"
    out.loc[pred & strict, "p_adjustment_family"] = "prediction_strict_all_settings"
    for col in ["p_value_bh", "p_value_holm", "mean_car_p_bh", "mean_car_p_holm"]:
        out[col] = np.nan
    for _, group in out.groupby("p_adjustment_family", sort=False):
        for pcol, prefix in [("p_value", "p_value"), ("mean_car_p", "mean_car_p")]:
            if pcol not in out:
                continue
            valid = pd.to_numeric(group[pcol], errors="coerce").dropna()
            if valid.empty:
                continue
            if not valid.between(0, 1).all():
                raise AssertionError(f"Invalid p values: {pcol}")
            out.loc[valid.index, prefix + "_bh"] = multipletests(valid.to_numpy(), method="fdr_bh")[1]
            out.loc[valid.index, prefix + "_holm"] = multipletests(valid.to_numpy(), method="holm")[1]
    out["raw_significant_5pct"] = out["p_value"] < 0.05
    out["bh_significant_5pct"] = out["p_value_bh"] < 0.05
    out["holm_significant_5pct"] = out["p_value_holm"] < 0.05
    # Compare each model with its own matched reference, never a differently
    # covered full-sample estimate. Unknown comparisons remain missing.
    for col in ["coefficient_sign_consistent", "raw_significance_consistent", "ic_direction_consistent", "oos_r2_direction_consistent"]:
        out[col] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out["paired_effect_difference"] = np.nan
    out["ic_change_vs_paired_reference"] = np.nan
    out["oos_r2_change_vs_paired_reference"] = np.nan
    for idx, row in out.iterrows():
        event_row = row["block"] == "event"
        effect = row.get("paired_variant_estimate", np.nan) if event_row else row.get("estimate", np.nan)
        reference = row.get("paired_baseline_same_measure_estimate", np.nan) if event_row else row.get("paired_baseline_estimate", np.nan)
        p = row.get("paired_variant_p", np.nan) if event_row else row.get("p_value", np.nan)
        rp = row.get("paired_baseline_same_measure_p", np.nan) if event_row else row.get("paired_p_value", np.nan)
        if pd.notna(effect) and pd.notna(reference):
            out.loc[idx, "coefficient_sign_consistent"] = bool(np.sign(effect) == np.sign(reference))
            out.loc[idx, "paired_effect_difference"] = effect - reference
        if pd.notna(p) and pd.notna(rp):
            out.loc[idx, "raw_significance_consistent"] = bool((p < 0.05) == (rp < 0.05))
        if not event_row:
            for metric, flag, delta in [("ic_mean", "ic_direction_consistent", "ic_change_vs_paired_reference"),
                                        ("oos_r2", "oos_r2_direction_consistent", "oos_r2_change_vs_paired_reference")]:
                value, ref = row.get(metric, np.nan), row.get("paired_" + metric, np.nan)
                if pd.notna(value) and pd.notna(ref):
                    out.loc[idx, flag] = bool(np.sign(value) == np.sign(ref))
                    out.loc[idx, delta] = value - ref
    return out


def fmt(value: Any, decimals: int = 4) -> str:
    try:
        return f"{float(value):.{decimals}f}" if np.isfinite(float(value)) else "NA"
    except (ValueError, TypeError):
        return "NA"


def table(frame: pd.DataFrame, columns: list[tuple[str, str]], numeric: int = 5) -> list[str]:
    lines = ["| " + " | ".join(label for _, label in columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for _, row in frame.iterrows():
        values = []
        for key, _ in columns:
            value = row.get(key, "NA")
            if isinstance(value, (float, np.floating)):
                count_column = key.startswith("n_") or key in ["bytes", "records", "failure_count"]
                value = fmt(value, 0 if count_column else numeric)
            values.append(str(value).replace("|", "/").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def collect_failures(sources: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for block, files in [("event", ["failures.csv", "setting_failures.csv", "sentiment_exclusions.csv"]), ("prediction", ["failures.csv"])]:
        for name in files:
            p = CACHE / block / name
            if not p.exists() or p.stat().st_size <= 3:
                continue
            try:
                f = pd.read_csv(p, dtype={"issue": str, "code": str})
            except pd.errors.EmptyDataError:
                continue
            f.insert(0, "block", block)
            f.insert(1, "source_file", str(p.relative_to(ROOT)))
            frames.append(f)
    failed_sources = sources[sources["status"].eq("failed")].copy()
    if len(failed_sources):
        failed_sources["block"] = "source"
        failed_sources["reason"] = failed_sources["failure_reason"]
        frames.append(failed_sources)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["block", "reason"])


def make_plot(summary: pd.DataFrame) -> None:
    """A compact, four-panel research figure with explicit effect units."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlesize": 11,
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.10, h_pad=0.10, hspace=0.10, wspace=0.12)
    blue, red, gray, green = "#277DA8", "#B3484F", "#818A91", "#2F846C"
    for ax in axes.flat:
        ax.set_facecolor("#FAFBFC")
        ax.grid(axis="x", color="#DDE2E6", linewidth=0.6)
        ax.set_axisbelow(True)
    ev = summary[summary["block"].eq("event")].copy()
    # Derive the four event-window rows from declared parameters, not row order.
    windows = ev[ev["family"].isin(["reference", "event_window"])]
    if len(windows) != 4:
        windows = ev[ev["setting_id"].isin(["event_main", "event_window_1", "event_window_3", "event_window_10"])]
    windows = windows.sort_values("event_window")
    ax = axes[0, 0]
    for y, (_, row) in enumerate(windows.iterrows()):
        x, lo, hi = [float(row[c]) * 100 for c in ["estimate", "ci_low", "ci_high"]]
        ax.plot([lo, hi], [y, y], color=blue, lw=1.5)
        ax.scatter(x, y, color=blue, s=35, zorder=3)
        ax.annotate(f" p={fmt(row['p_value'], 3)}", (hi, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    ax.axvline(0, color=gray, lw=0.8)
    ax.set_yticks(range(len(windows)), [f"[0,+{int(w)}]" for w in windows["event_window"]])
    ax.invert_yaxis()
    ax.margins(x=0.28, y=0.25)
    ax.set_title("A  Event CAR: high minus low sentiment")
    ax.set_xlabel("CAR difference (percentage points), Welch 95% CI")

    ax = axes[0, 1]
    select = ev[ev["family"].isin(["baseline", "reference", "dictionary", "benchmark", "estimation_window", "normal_return_model", "model", "purge"])]
    select = select[~select["setting_id"].duplicated()]
    if len(select) > 12:
        select = select.iloc[:12]
    labels = []
    for y, (_, row) in enumerate(select.iterrows()):
        labels.append(str(row["setting_id"]).removeprefix("event_"))
        x = float(row.get("paired_variant_estimate", row["estimate"])) * 100
        base = float(row.get("paired_baseline_same_measure_estimate", row.get("paired_baseline_estimate", np.nan))) * 100
        ax.plot([base, x], [y, y], color="#BEC7CD", lw=1)
        ax.scatter(base, y, facecolors="white", edgecolors=gray, s=30, zorder=3)
        ax.scatter(x, y, color=red if row["status"] == "limited" else blue, s=27, zorder=4)
    ax.axvline(0, color=gray, lw=0.8)
    ax.set_yticks(range(len(select)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("B  One-setting changes on common samples")
    ax.set_xlabel("High-minus-low CAR (pp); hollow = paired reference")

    pred = summary[summary["block"].eq("prediction")]
    ax = axes[1, 0]
    ids = ["prediction_main", "prediction_dlut", "prediction_mpr_domain", "prediction_combo", "prediction_lag2", "prediction_tercile", "prediction_winsor"]
    old = pred.set_index("setting_id")
    labels = ["NTUSD", "DLUT", "Domain", "Combination", "Lag 2", "Terciles", "Winsor 1%"]
    for y, ident in enumerate(ids):
        for shift, name, color in [(-0.13, ident, gray), (0.13, ident.replace("prediction_", "prediction_strict_", 1), green)]:
            if name in old.index:
                x = float(old.loc[name, "incremental_oos_r2"]) * 100
                if np.isfinite(x):
                    ax.scatter(x, y + shift, color=color, s=28, label=("Legacy timing" if color == gray else "Strict timing") if y == 0 else None)
    ax.axvline(0, color=red, lw=0.8)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_title("C  Test-set incremental OOS R-squared")
    ax.set_xlabel("Text minus refitted no-text model (percentage points)")
    ax.legend(loc="best", fontsize=8, frameon=False)

    ax = axes[1, 1]
    overlap = ev[ev["family"].isin(["baseline", "reference", "estimation_window", "purge"])]
    overlap = overlap.drop_duplicates("setting_id")
    if len(overlap) > 8:
        overlap = overlap.iloc[:8]
    for y, (_, row) in enumerate(overlap.iterrows()):
        share = float(row.get("prior_event_overlap_share", np.nan)) * 100
        ax.barh(y, share, color=green if share == 0 else blue, height=0.55)
        ax.text(min(share+1.5, 101.5), y, f"{share:.1f}%", va="center", fontsize=8)
    ax.set_yticks(range(len(overlap)), [str(x).removeprefix("event_") for x in overlap["setting_id"]], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 119)
    ax.set_title("D  Previous MPR in estimation observations")
    ax.set_xlabel("Industry-event cells containing previous event day (%)")
    fig.suptitle("E8 Robustness checks | MPR sentiment and industry returns", fontsize=15, weight="bold")
    fig.savefig(OUT / "e8_robustness_plots.png", dpi=100)
    plt.close(fig)
    with Image.open(OUT / "e8_robustness_plots.png") as im:
        if max(im.size) > 1400 or im.width * im.height > 5_242_880:
            raise AssertionError(f"Figure dimensions exceed preview budget: {im.size}")


def write_log(summary: pd.DataFrame, details: dict[str, Any], metrics: dict[str, Any], sources: pd.DataFrame,
              input_manifest: pd.DataFrame, failures: pd.DataFrame) -> None:
    ev = summary[summary["block"].eq("event")]
    pr = summary[summary["block"].eq("prediction")]
    lines = ["# E8 稳健性检验", "", f"运行时间：{metrics['run_at']}。代码版本：`{metrics['git_commit']}`。", "",
             "## 1. 结论与解释边界", "",
             "本次结果不能支持“更积极的 MPR 情绪稳定提高行业收益或预测能力”。事件研究的高低组差异对事件窗、词典和正常收益模型敏感；原始显著结果须同时查看多重检验校正。严格修正预测收益与训练标签时点后，主词典仍没有稳定的样本外增量。未拒绝零假设不等于证明不存在经济效应。", "",
             "- E3 没有确认唯一主事件窗；本实验以 [0,+5] 为展示/配对锚点，并精确复现全部四窗。该选择不是新增预注册，也不按显著性挑主结果。",
             "- E4 原预测目标为报告期下一季度，该季度已在报告发布前开始。旧口径仅作复现和敏感性诊断，不能作为严格的前瞻预测证据。",
             "- 严格口径改用 MPR 发布后的首个完整季度，并重新估计模型，只允许预测时已经实现的历史标签进入训练。目标与训练成熟时间一起修正，因此单列为新的时点修正参考设定。",
             "- 以上均为关联和预测检验，不构成因果结论或可交易策略。E5 的行业差异、E6 的 LLM 结论和 E7 的组合表现不因 E8 自动升级；LLM Prompt/模型敏感性属于 E9。", "",
             "## 2. 基准与单项变化规则", "",
             "事件参考：NTUSD 简体词典、沪深300、市场模型、[-120,-1] 估计窗、[0,+5] 事件窗、全部可用行业、原始 CAR、沿用 E3 全103期三分位。每个事件先对行业 CAR 等权平均，再在事件层面检验；不用行业行数冒充独立时间样本数。", "",
             "预测参考：截距 + mom1_z + vol_q_z + beta_120_z + 4 个行业板块虚拟变量；文本模型额外加入 trailing-z 情绪 × 标准化行业情绪暴露。每季截面 OLS，扩展窗平均历史系数，最低 8 个已拟合季度，测试目标季度固定 2021Q1–2026Q3。共同情绪本身被截距吸收，不能直接用于行业排序。", "",
             "除显式时点修正外，每个设定改变一个研究维度，其他设定继承对应参考。lag2 是整套预先可得信息向量再滞后一报告季度（情绪交互和三项连续控制一起平移，行业板块不变）；不使用 E4 与 mom1 重复的 mom2 冒充真实 lag2。Winsor 是异常值处理规则这一维度，包含明确指定的特征/拟合标签处理。", "",
             "共同样本：事件按 issue×code 交集重算变体和参考；不同基准不能把 2002 起的沪深300样本与 2011 起的中证全指样本直接相减。预测在共同面板行重新拟合无文本及 NTUSD 参考，再按共同预测键计分。`estimate` 为各设定全可用样本结果，公平比较使用 `paired_*`，不要混用。", "",
             "## 3. 输入、数据范围与复现", ""]
    lines += table(input_manifest[["path", "bytes", "role"]], [("path", "文件"), ("bytes", "字节"), ("role", "用途")])
    lines += ["", "完整 SHA256 见 `e8_input_manifest.csv`。行情复用已冻结的数据，不为某个设定临时切换下载版本；互联网只核验方法文档。", "",
              "中证全指文件只有 2011 年起的行情；不拼接其他指数补造早期历史。不同交易日历及缺失行业单元均明示剔除。当前行业分类历史回溯、幸存者偏差及非实时数据修订风险仍然存在。", "",
              "### 3.1 模块审计记录", "", "```json", json.dumps(details, ensure_ascii=False, indent=2), "```", "",
              "## 4. 推断、缩尾与多重检验", "",
              "- 事件高减低：事件均值上的 Welch 不等方差检验和 95% CI；连续得分：事件 CAR 对标准化情绪的 HC3 回归。连续斜率和高低差的单位不同，不能比较数值大小。Welch 假设两组事件独立，时间相关仍是局限；连续 HC3 也不等于 HAC。",
              "- 预测 FMB：季度系数均值的 Newey-West/Bartlett lag=2 推断；扩展历史系数预测。稀疏子样本中的“lag”按可用季度观测序列计算，非严格等间隔的日历 HAC。",
              "- OOS R² = 1−SSE/每季横截面实现收益离差平方和；这个分母是事后诊断基准，不是可实施预测。额外报告与同样本重新拟合的无文本预测比较。IC 为季度截面 Spearman 相关均值。",
              "- 事件缩尾：行业 CAR 合并分布 1%/99% 截断，属于描述性全样本敏感性，不作 OOS 宣称。预测缩尾：当季截面特征截断，拟合标签阈值只从该预测原点之前已实现标签估计；测试实际收益保持原始值。",
              "- 原始 p 全保留。主效应分别在本报告定义的 event、prediction legacy、prediction strict 三个汇总块内对全部有限 p 做 BH 与 Holm；全样本均值 CAR 是另一个明确标记的次要检验族。参考行、低功效行均计入，未事后删除不显著设定。检验族没有单独预注册。",
              "- 重叠稳健性设定相关性强。BH 是探索性 FDR 描述，任意依赖下不保证控制；Holm 用于不依赖独立性的更保守判断。全部设定是事后稳健性分析，不能改写为独立验证样本或预注册发现。", "",
              "## 5. 事件研究完整结果", ""]
    lines += table(ev, [("setting_id", "设定"), ("status", "状态"), ("n_periods", "期数"), ("n_high", "高"), ("n_low", "低"),
                        ("estimate", "效应"), ("p_value", "原始p"), ("p_value_bh", "BH p"), ("p_value_holm", "Holm p"),
                        ("paired_baseline_same_measure_estimate", "配对参考"), ("paired_variant_estimate", "配对变体"),
                        ("prior_event_overlap_share", "上一期日重叠占比")])
    lines += ["", "后期子样本仍沿用 E3 全样本分组，不为了增加显著性重新划分。高/低任一组少于 5 期标记 limited；这不是失败归零，更不能据不显著写成稳定无效。全样本分组只用于描述性事件比较，不能拿去构建历史实时信号。", "",
              "## 6. 预测完整结果", ""]
    lines += table(pr, [("setting_id", "设定"), ("status", "状态"), ("n_periods", "拟合期数"), ("n_oos_periods", "测试期数"),
                        ("estimate", "FMB均值"), ("p_value", "NW p"), ("p_value_bh", "BH p"), ("p_value_holm", "Holm p"),
                        ("ic_mean", "IC"), ("oos_r2", "OOS R²"), ("incremental_oos_r2", "相对无文本ΔR²"), ("mae", "MAE")])
    lines += ["", "早期子样本不覆盖 2021 年后的固定测试期，OOS 指标为 NA 并标 limited；仍报告其可计算 FMB 系数。不把空结果填零。三分组中组编码为0，使共同情绪交互列为0时截面秩不足，该季显式记录未拟合；三角色均限定到共同可识别季度拟合与训练。其结果是尾部情绪条件估计，标记 limited，不代表完整三组模型得到全样本验证，也不能只比较剩余季度的显著性。", "",
              "## 7. 稳健与脆弱的判断", "",
              f"- 事件主效应：{len(ev)} 个设定，原始 p<0.05 共 {int(ev['raw_significant_5pct'].sum())} 个；BH 后 {int(ev['bh_significant_5pct'].sum())} 个；Holm 后 {int(ev['holm_significant_5pct'].sum())} 个。低功效/失败设定保留在表内。",
              "- 在市场模型内，NTUSD 的四个事件窗，以及 [0,+5] 下三种词典和等权组合的高减低 CAR 点估计均为负；该负向点估计方向较一致，显著性不稳健。这是单项变化结果，并未运行全部词典×窗口组合。改用均值调整模型后差值转正且不显著，故负向关联也不能跨正常收益模型概括。只在个别短窗出现的显著关联属于脆弱、探索性证据。",
              "- 估计窗 90/120/180/250 日包含上一期事件，不能据长窗间相似就宣称排除污染。30日和净化估计观测是更直接的诊断；净化仍不能控制所有同期宏观新闻。",
              "- 词典选择可改变 FMB 系数方向；旧口径单一词典的显著系数不等于严格时点下可预测。严格主设定和各变体需同时查看IC、相对无文本ΔR²和MAE，不能只选系数p。",
              "- 旧口径等权词典组合的 IC、OOS R²、MAE 点估计略优于无文本参考，但时点修正后其增量 R²转负。该孤立改善不能升级为稳定预测证据。",
              "- 可保留的结论强度：当前数据和模型未给出跨词典、事件窗和时点协议一致的稳定正向证据。局部显著性、子样本无显著性与主结论均按有限样本描述，禁止推广为政策因果或交易收益保证。", "",
              "## 8. 失败、缺失与每设定记录", "",
              f"总计保留 {len(failures)} 条底层失败/低功效记录，见 `e8_failures.csv`。数据不足、缺失日期、秩不足、训练期不足与原点时间约束分别保存，没有用0替换失败收益。`code=ALL` 代表一个请求期对全部31行业失败；不要把原始记录数等同于展开后的行业单元数。", ""]
    lines += ["`ok` 只表示该设定计算成功且达到本报告最低样本条件，不表示效应显著或结论稳健；`limited` 表示样本功效、测试覆盖或条件识别受限；`failed` 才表示未能估计核心结果。", ""]
    lines += table(summary, [("setting_id", "设定"), ("status", "状态"), ("failure_reason", "推断/测试限制"),
                            ("n_failure_records", "事件底层失败记录"), ("failure_count", "预测拟合/预测失败记录")])
    if len(failures) and "reason" in failures:
        counts = failures.groupby(["block", "reason"], dropna=False).size().reset_index(name="records")
        lines += ["", "### 8.1 失败原因汇总", ""] + table(counts, [("block", "模块"), ("reason", "原因"), ("records", "记录数")])
    lines += ["", "## 9. 互联网方法核验", "",
              "以下是官方软件/方法文档。它们支持计算选择，不是本项目实证结果的外部证据；数据结论完全来自上述冻结输入。原始网页、抓取时间和哈希保存在 `_temp/e8/sources/`，来源索引见 `e8_sources.csv`。", ""]
    for row in sources.itertuples():
        lines += [f"- {row.title}：{row.url}；状态 `{row.status}`。用途：{row.role}。"]
    lines += ["", "## 10. 复现、环境与验收", "", "```powershell", "python -B experiments/E8/run_e8.py",
              "python -B -m unittest discover -s experiments/E8 -p 'test_e8*.py' -v", "```", "",
              "仅重建图表和总日志（沿用本次缓存结果）：`python -B experiments/E8/run_e8.py --from-cache`。",
              "若 E4 输入缓存缺失，先按 E4 原脚本重建并核对输入 manifest；E8 不静默下载替代数据或覆盖既有实验结果。", "",
              "完整运行元数据见 `e8_metrics.json`；所有下载页、中间面板、系数、预测和 matplotlib/Python 缓存在 `_temp/e8/`。", "",
              "```json", json.dumps(metrics, ensure_ascii=False, indent=2), "```", ""]
    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "logs" / "e8_robustness.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--refresh-sources", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    before = manifest()
    if args.from_cache:
        event = pd.read_csv(CACHE / "event" / "summary.csv")
        prediction = pd.read_csv(CACHE / "prediction" / "summary.csv")
        details = {name: json.loads((CACHE / name / "details.json").read_text(encoding="utf-8")) for name in ["event", "prediction"]}
    else:
        from e8_event import run_event
        from e8_prediction import run_prediction
        event, ed = run_event()
        prediction, pd_ = run_prediction()
        details = dict(event=ed, prediction=pd_)
    summary = adjust_tests(pd.concat([event, prediction], ignore_index=True))
    if summary["setting_id"].duplicated().any():
        raise AssertionError("Duplicate setting identifiers")
    sources = fetch_sources(args.refresh_sources)
    failures = collect_failures(sources)
    after = manifest()
    if not before.equals(after):
        raise AssertionError("An input changed during E8; rerun with a stable snapshot")
    metrics = dict(run_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                   as_of_date="2026-11-19", cache_reuse=args.from_cache,
                   git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   python=sys.version.split()[0], platform=platform.platform(), pandas=pd.__version__, numpy=np.__version__,
                   scipy=scipy.__version__, matplotlib=matplotlib.__version__, seed=20261119,
                   n_settings=len(summary), status_counts=summary["status"].value_counts().to_dict(),
                   failure_records=len(failures), sources_verified=int(sources["status"].eq("verified").sum()),
                   raw_significant=int(summary["raw_significant_5pct"].sum()), bh_significant=int(summary["bh_significant_5pct"].sum()),
                   holm_significant=int(summary["holm_significant_5pct"].sum()),
                   script_sha256={p.name: digest(p) for p in sorted(OUT.glob("*.py"))},
                   no_existing_input_modified=True,
                   primary_effect_p_families=summary.groupby("p_adjustment_family")["p_value"].count().to_dict())
    summary.to_csv(OUT / "e8_robustness_summary.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(OUT / "e8_failures.csv", index=False, encoding="utf-8-sig")
    sources.to_csv(OUT / "e8_sources.csv", index=False, encoding="utf-8-sig")
    before.to_csv(OUT / "e8_input_manifest.csv", index=False, encoding="utf-8-sig")
    (OUT / "e8_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    make_plot(summary)
    write_log(summary, details, metrics, sources, before, failures)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("E8 finished; results: experiments/E8; cache: _temp/e8")
    return 1 if summary["status"].eq("failed").any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
