# -*- coding: utf-8 -*-
"""从真实实验输出构造 Fig 13-20 的绘图输入（写入 图表/results/）。

只读 experiments/ 下的实验产物与 _temp/e4 的预测缓存，不修改任何实验数据；
所有筛选、重命名与聚合口径都在此集中登记，便于复核。

输出（图表/results/）：
  sentiment_dictionary_full.csv   Fig 13  E1 主词典情绪时间序列
  e3_car_by_sentiment.csv         Fig 14  E3 市场模型 高/中/低组 CAR
  e3_event_results.csv            Fig 15  E3 市场模型 行业 × 事件窗 CAR
  e4_coefficient_series.csv       Fig 16  E4 主词典 FMB 情绪系数时序
  e4_oos_metrics.csv              Fig 17  由 E4 逐期预测计算的 IC/ICIR/OOS R²
  e7_portfolio.csv                Fig 18  E7 多空组合收益
  e5_heterogeneity.csv            Fig 19  E5 行业敏感度
  e8_robustness_summary.csv       Fig 20  E8 稳健性设定（含 block 列）

复现：python 图表/plotting_scripts/build_experiment_inputs.py
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from _style import PROJECT_ROOT

EXP_DIR = os.path.join(PROJECT_ROOT, "experiments")
TEMP_DIR = os.path.join(PROJECT_ROOT, "_temp")
OUT_DIR = os.path.join(PROJECT_ROOT, "图表", "results")

PRIMARY_DICT = "ntusd_simplified"          # E1/E3/E4/E5/E8 主词典
PRIMARY_EVENT_MODEL = "market"             # E3 主正常收益模型
GROUP_CN = {"high": "高", "mid": "中", "low": "低"}
OOS_WINDOWS = ("validation", "test")

# E8 设定 ID → 中文标签（仅用于展示，不改变数值口径）
SETTING_CN = {
    "event_main": "主设定（NTUSD 简体，[0,+5]）",
    "event_dictionary_dlut": "词典：DLUT",
    "event_dictionary_mpr_domain": "词典：MPR 领域",
    "event_dictionary_combo_ntusd_dlut": "词典：NTUSD + DLUT 组合",
    "event_window_1": "事件窗 [0,+1]",
    "event_window_3": "事件窗 [0,+3]",
    "event_window_10": "事件窗 [0,+10]",
    "event_pre2008": "子样本：2008 年前",
    "event_post2008": "子样本：2008 年起",
    "event_pre2015": "子样本：2015 年前",
    "event_post2015": "子样本：2015 年起",
    "event_pre2020": "子样本：2020 年前",
    "event_post2020": "子样本：2020 年起",
    "event_benchmark_allshare": "基准：中证全指",
    "event_winsor_1pct": "行业 CAR 缩尾 1%",
    "event_estimation_30": "估计窗 30 日",
    "event_estimation_60": "估计窗 60 日",
    "event_estimation_90": "估计窗 90 日",
    "event_estimation_180": "估计窗 180 日",
    "event_estimation_250": "估计窗 250 日",
    "event_estimation_purged": "估计窗：剔除前期 MPR 窗",
    "event_continuous": "情绪形式：连续 HC3 斜率",
    "event_mean_adjusted": "正常收益：均值调整",
    "prediction_main": "主设定（NTUSD 简体）",
    "prediction_dlut": "词典：DLUT",
    "prediction_mpr_domain": "词典：MPR 领域",
    "prediction_combo": "词典：等权组合",
    "prediction_lag2": "信息滞后 2 期",
    "prediction_tercile": "情绪形式：历史三分位",
    "prediction_winsor": "缩尾 1%",
    "prediction_before2008": "子样本：2008 年前",
    "prediction_after2008": "子样本：2008 年后",
    "prediction_before2015": "子样本：2015 年前",
    "prediction_after2015": "子样本：2015 年后",
    "prediction_before2020": "子样本：2020 年前",
    "prediction_after2020": "子样本：2020 年后",
}
for _k in list(SETTING_CN):
    if _k.startswith("prediction_"):
        SETTING_CN[_k.replace("prediction_", "prediction_strict_", 1)] = "严格口径：" + SETTING_CN[_k]


def _q_to_date(q):
    """'2002Q2' -> 2002-04-01（季初），用于时间轴。"""
    s = str(q).strip()
    year, quarter = s.split("Q")
    return pd.Timestamp(int(year), (int(quarter) - 1) * 3 + 1, 1)


def _write(df, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"wrote: {path}  rows={len(df)}")
    return path


def build_fig13():
    """E1 主词典情绪得分与覆盖率（按发布日期排序）。"""
    src = os.path.join(EXP_DIR, "E1", "sentiment_dictionary_full.csv")
    df = pd.read_csv(src)
    df = df[df["dictionary"] == PRIMARY_DICT].copy()
    out = pd.DataFrame({
        "period": pd.to_datetime(df["publication_date"]),
        "sentiment": df["sentiment_score"],
        "coverage": df["pos_neg_coverage"],
    }).sort_values("period").reset_index(drop=True)
    return _write(out, "sentiment_dictionary_full.csv")


def build_fig14():
    """E3 市场模型：高/中/低情绪组在各事件窗的 CAR 与 Bootstrap CI。"""
    src = os.path.join(EXP_DIR, "E3", "e3_car_by_sentiment.csv")
    df = pd.read_csv(src)
    df = df[(df["model"] == PRIMARY_EVENT_MODEL) & (df["group"].isin(GROUP_CN))].copy()
    out = pd.DataFrame({
        "window": df["event_window"],
        "group": df["group"].map(GROUP_CN),
        "car": df["mean_car"],
        "t": df["t_stat"],
        "ci_low": df["boot_ci_low"],
        "ci_high": df["boot_ci_high"],
    })
    out["_g"] = out["group"].map({"高": 0, "中": 1, "低": 2})
    out = out.sort_values(["window", "_g"]).drop(columns="_g").reset_index(drop=True)
    return _write(out, "e3_car_by_sentiment.csv")


def build_fig15():
    """E3 市场模型：行业 × 事件窗 CAR 面板（热力图取行业-窗均值）。"""
    src = os.path.join(EXP_DIR, "E3", "e3_event_results.csv")
    df = pd.read_csv(src)
    df = df[df["model"] == PRIMARY_EVENT_MODEL].copy()
    out = pd.DataFrame({
        "industry": df["name"],
        "window": df["event_window"],
        "car": df["car"],
    }).dropna(subset=["car"])
    return _write(out, "e3_event_results.csv")


def build_fig16():
    """E4 主词典 FMB 情绪暴露交互系数的逐期时序。"""
    src = os.path.join(EXP_DIR, "E4", "e4_coefficient_series.csv")
    df = pd.read_csv(src)
    coef = f"sent_x_exposure_{PRIMARY_DICT}"
    df = df[(df["model"] == "sentiment") & (df["dictionary"] == PRIMARY_DICT)
            & (df["coefficient"] == coef)].copy()
    out = pd.DataFrame({
        "period": df["target_quarter"].map(_q_to_date),
        "beta": df["estimate"],
        "se": df["std_error"],
    }).dropna(subset=["beta", "se"]).sort_values("period").reset_index(drop=True)
    return _write(out, "e4_coefficient_series.csv")


def build_fig17():
    """由 E4 逐期样本外预测计算 IC / 滚动 ICIR / 逐期 OOS R²。

    IC：每个目标季截面 Spearman 秩相关（预测 vs 实际）。
    OOS R²：1 - SSE/SST，SST 以该季截面均值为基准（与 E4 聚合口径同定义，
    但 E4 的汇总值为全样本合并计算，图 17 为逐季值，两者不可直接互换）。
    ICIR：5 期滚动 IC 均值 / IC 标准差（min_periods=3）。
    """
    src = os.path.join(TEMP_DIR, "e4", "e4_oos_predictions.csv")
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"缺少 {src}；请先运行 experiments/E4/run_e4.py 生成逐期预测缓存。")
    df = pd.read_csv(src)
    df = df[(df["model"] == "sentiment") & (df["dictionary"] == PRIMARY_DICT)].copy()
    rows = []
    for q, g in df.groupby("target_quarter"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ic = stats.spearmanr(g["predicted"], g["actual"]).statistic
        sse = float(((g["actual"] - g["predicted"]) ** 2).sum())
        sst = float(((g["actual"] - g["actual"].mean()) ** 2).sum())
        rows.append({
            "period": _q_to_date(q),
            "ic": float(ic) if np.isfinite(ic) else np.nan,
            "oos_r2": 1.0 - sse / sst if sst > 0 else np.nan,
            "n": int(len(g)),
        })
    out = pd.DataFrame(rows).sort_values("period").reset_index(drop=True)
    roll = out["ic"].rolling(5, min_periods=3)
    out["icir"] = roll.mean() / roll.std()
    out = out.drop(columns=["n"])
    return _write(out, "e4_oos_metrics.csv")


def build_fig18():
    """E7 多空组合逐季收益（验证期 + 测试期连成一条样本外序列）。"""
    src = os.path.join(EXP_DIR, "E7", "e7_portfolio.csv")
    df = pd.read_csv(src)
    df = df[df["split"].isin(OOS_WINDOWS)].copy()
    out = pd.DataFrame({
        "period": df["target_quarter"].map(_q_to_date),
        "long": df["long_basket_return"],
        "short": df["short_basket_return"],
        "ls": df["long_short_net"],
        "benchmark": df["industry_ew_return"],
    }).sort_values("period").reset_index(drop=True)
    return _write(out, "e7_portfolio.csv")


def build_fig19():
    """E5 行业敏感度主口径（市场模型 [0,+1]，ntusd_simplified）。"""
    src = os.path.join(EXP_DIR, "E5", "e5_heterogeneity.csv")
    df = pd.read_csv(src)
    out = pd.DataFrame({
        "industry": df["name"],
        "beta": df["sens_main"],
        "ci_low": df["sens_main_ci_low"],
        "ci_high": df["sens_main_ci_high"],
    }).dropna(subset=["beta"]).sort_values("beta").reset_index(drop=True)
    return _write(out, "e5_heterogeneity.csv")


def build_fig20():
    """E8 稳健性设定汇总（保留 block 列，绘图脚本默认取 event 块）。"""
    src = os.path.join(EXP_DIR, "E8", "e8_robustness_summary.csv")
    df = pd.read_csv(src)
    out = pd.DataFrame({
        "block": df["block"],
        "setting": df["setting_id"].map(SETTING_CN).fillna(df["setting_id"]),
        "beta": df["estimate"],
        "ci_low": df["ci_low"],
        "ci_high": df["ci_high"],
    })
    limited = df["status"].astype(str).eq("limited")
    out.loc[limited, "setting"] = out.loc[limited, "setting"] + "（限制）"
    out = out.dropna(subset=["beta", "ci_low", "ci_high"]).reset_index(drop=True)
    return _write(out, "e8_robustness_summary.csv")


def main():
    build_fig13()
    build_fig14()
    build_fig15()
    build_fig16()
    build_fig17()
    build_fig18()
    build_fig19()
    build_fig20()
    print(f"\n全部输入已写入：{OUT_DIR}")


if __name__ == "__main__":
    main()
