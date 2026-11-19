#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E7 高低情绪多空组合实验。

研究口径在运行前固定：
1. 信号使用 E4 的下一季度样本外预测值；主信号为 ntusd_simplified 词典情绪
   经滚动行业情绪暴露得到的 sent_exposure 预测。共同 MPR 情绪本身在季度
   截面上没有行业差异，因此不能直接对共同分数排序。
2. 每个目标季度按预测值排序，做多最高 5 个行业、做空最低 5 个行业，两腿
   均为 1/5 等权。多头总权重 +1、空头总权重 -1，组合毛敞口为 200%。
3. 交易成本预设为单边 10bp；同时报告换手、毛收益、净收益与成本。
4. 对照包括无文本预测多空组合、沪深 300、同一行业池等权组合，以及固定
   随机种子下的随机 5/5 分组分布。
5. 用 CH3（Liu-Stambaugh-Yuan 中国三因子）做 MKT、SMB、VMG 稳健性回归。

本实验是研究组合，不视为可直接执行的交易策略。
"""

from __future__ import annotations

import hashlib
import json
import platform
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
E7_DIR = ROOT / "experiments" / "E7"
LOG_DIR = E7_DIR / "logs"
TEMP_DIR = ROOT / "_temp" / "E7"

E4_PRED_PATH = ROOT / "_temp" / "e4" / "e4_oos_predictions.csv"
MPR_INDEX_PATH = ROOT / "experiments" / "E1" / "mpr_index_full.csv"
INDUSTRY_QUARTER_PATH = ROOT / "_temp" / "e4" / "industry_quarterly.csv"
BENCH_PATH = ROOT / "data" / "external" / "benchmark" / "benchmark_daily.csv"
BENCH_CACHE_PATH = ROOT / "_temp" / "e0_market_raw" / "benchmark_sh000300.csv"

FACTOR_PATH = TEMP_DIR / "CH3_factors_daily_202608.xlsx"
FACTOR_URL = "https://en.mingshiim.com/static/CH3_factors_daily_202608.xlsx"
FACTOR_DOC_PATH = TEMP_DIR / "size_value_china_appendix_2_rev.pdf"
FACTOR_DOC_URL = "https://faculty.wharton.upenn.edu/wp-content/uploads/2016/11/size_value_china_appendix_2_rev.pdf"

MAIN_DICT = "ntusd_simplified"
N_LONG = 5
N_SHORT = 5
N_RANDOM = 1000
COST_BPS = 10.0
SEED = 20261119
NW_LAG = 1

LOG_LINES: list[str] = []


def log(line: str = "") -> None:
    LOG_LINES.append(line)
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_factor_file() -> None:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    if FACTOR_PATH.exists() and FACTOR_PATH.stat().st_size > 0:
        return
    log(f"[data] downloading CH3 factors from {FACTOR_URL}")
    urllib.request.urlretrieve(FACTOR_URL, FACTOR_PATH)


def quarter_start(label: str) -> pd.Timestamp:
    year, quarter = label.split("Q")
    return pd.Timestamp(year=int(year), month=(int(quarter) - 1) * 3 + 1, day=1)


def quarter_end(label: str) -> pd.Timestamp:
    return quarter_start(label) + pd.offsets.QuarterEnd(startingMonth=3)


def quarter_label(date: pd.Timestamp) -> str:
    return f"{date.year}Q{date.quarter}"


def load_predictions() -> pd.DataFrame:
    predictions = pd.read_csv(E4_PRED_PATH, dtype={"code": str})
    sentiment = predictions[
        (predictions["model"] == "sentiment")
        & (predictions["dictionary"] == MAIN_DICT)
        & (predictions["spec"] == "sent_exposure")
    ].copy()
    baseline = predictions[
        (predictions["model"] == "baseline_notext")
        & (predictions["spec"] == "baseline_notext")
    ][["target_quarter", "code", "predicted"]].rename(
        columns={"predicted": "baseline_predicted"}
    )
    sentiment = sentiment.merge(baseline, on=["target_quarter", "code"], how="left")
    sentiment = sentiment.rename(
        columns={
            "target_quarter": "e4_target_quarter",
            "actual": "e4_actual_log_return",
        }
    )
    sentiment = sentiment.dropna(subset=["predicted", "baseline_predicted"]).copy()

    mpr = pd.read_csv(MPR_INDEX_PATH, dtype={"issue": str})
    mpr["publication_date"] = pd.to_datetime(mpr["publication_date"])
    sentiment = sentiment.merge(
        mpr[["issue", "publication_date", "report_period_end"]],
        on="issue",
        how="left",
        validate="many_to_one",
    )
    if sentiment["publication_date"].isna().any():
        raise AssertionError("部分 E4 信号无法匹配 MPR 发布日期")

    # MPR 报告通常在下季度开始后才发布。E4 的“报告期下一季度”目标在发布时
    # 已经走完一部分，直接交易会产生不可观察的前瞻窗口。E7 主口径改为发布后
    # 第一个完整季度，并仅使用 E4 的横截面排序；E4 原目标收益保留作审计字段。
    publication_period = sentiment["publication_date"].dt.to_period("Q")
    sentiment["publication_quarter"] = publication_period.astype(str)
    sentiment["target_quarter"] = (publication_period + 1).astype(str)

    industry_q = pd.read_csv(INDUSTRY_QUARTER_PATH, dtype={"code": str})
    industry_q["last_date"] = pd.to_datetime(industry_q["last_date"])
    industry_q["quarter_end"] = pd.to_datetime(industry_q["quarter_end"])
    industry_q = industry_q.rename(
        columns={
            "quarter": "target_quarter",
            "name": "actual_name",
            "log_return": "actual",
            "n_days": "actual_n_days",
        }
    )
    industry_q = industry_q[
        ["code", "target_quarter", "actual_name", "actual", "actual_n_days", "last_date", "quarter_end"]
    ]
    sentiment = sentiment.merge(industry_q, on=["code", "target_quarter"], how="left")
    complete = (
        sentiment["actual"].notna()
        & (sentiment["actual_n_days"] >= 40)
        & (sentiment["last_date"] >= sentiment["quarter_end"] - pd.Timedelta(days=7))
    )
    sentiment = sentiment[complete].copy()
    sentiment["name"] = sentiment["actual_name"].fillna(sentiment["name"])
    sentiment["actual_simple"] = np.expm1(sentiment["actual"].astype(float))
    sentiment["target_date"] = sentiment["target_quarter"].map(quarter_end)
    sentiment["target_start"] = sentiment["target_quarter"].map(quarter_start)
    sentiment["availability_lag_days"] = (
        sentiment["target_start"] - sentiment["publication_date"]
    ).dt.days
    if (sentiment["availability_lag_days"] <= 0).any():
        raise AssertionError("目标季度开始前未获得全部信号，存在时点错误")
    return sentiment.sort_values(["target_quarter", "code"]).reset_index(drop=True)


def load_benchmark() -> pd.DataFrame:
    path = BENCH_CACHE_PATH if BENCH_CACHE_PATH.exists() else BENCH_PATH
    bench = pd.read_csv(path, dtype={"code": str})
    bench["date"] = pd.to_datetime(bench["date"])
    bench = bench[(bench["code"] == "sh000300")].sort_values("date").copy()
    bench["quarter"] = bench["date"].map(quarter_label)
    quarterly = (
        bench.groupby("quarter", sort=True)
        .agg(
            first_trade_date=("date", "min"),
            last_trade_date=("date", "max"),
            close=("close", "last"),
            n_days=("date", "size"),
        )
        .reset_index()
    )
    quarterly["market_return"] = quarterly["close"].pct_change()
    return quarterly


def load_factor_daily() -> pd.DataFrame:
    ensure_factor_file()
    factors = pd.read_excel(FACTOR_PATH, sheet_name="Returnseries")
    date_text = factors["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    factors["date"] = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
    factors = factors.dropna(subset=["date"]).copy()
    factors = factors.rename(columns={"mktrf": "MKT_RF", "SMB": "SMB", "VMG": "VMG"})
    for column in ["rf_dly", "MKT_RF", "SMB", "VMG"]:
        factors[column] = pd.to_numeric(factors[column], errors="coerce")
    factors = factors.dropna(subset=["rf_dly", "MKT_RF", "SMB", "VMG"])
    factors["quarter"] = factors["date"].map(quarter_label)
    return factors.sort_values("date").reset_index(drop=True)


def aggregate_factors(daily: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for q, frame in daily.groupby("quarter", sort=True):
        row: dict[str, Any] = {"quarter": q, "factor_n_days": int(len(frame))}
        for column, output in [
            ("rf_dly", "rf_return"),
            ("MKT_RF", "MKT_RF"),
            ("SMB", "SMB"),
            ("VMG", "VMG"),
        ]:
            row[output] = float(np.prod(1.0 + frame[column].to_numpy(dtype=float)) - 1.0)
        rows.append(row)
    quarterly = pd.DataFrame(rows)
    quarterly = quarterly.merge(
        benchmark[["quarter", "n_days"]].rename(columns={"n_days": "benchmark_n_days"}),
        on="quarter",
        how="left",
    )
    quarterly["factor_complete"] = (
        quarterly["benchmark_n_days"].notna()
        & (quarterly["factor_n_days"] >= np.ceil(0.8 * quarterly["benchmark_n_days"]))
    )
    return quarterly


def build_weights(selected_codes: Iterable[str], k: int, direction: int) -> dict[str, float]:
    return {code: direction / k for code in list(selected_codes)[:k]}


def portfolio_weights(frame: pd.DataFrame, signal_column: str) -> dict[str, float]:
    ranked = frame.sort_values([signal_column, "code"], ascending=[False, True])
    long_codes = ranked.head(N_LONG)["code"].tolist()
    short_codes = ranked.tail(N_SHORT)["code"].tolist()
    if set(long_codes) & set(short_codes):
        raise AssertionError("多空行业出现重叠")
    weights = build_weights(long_codes, N_LONG, 1)
    weights.update(build_weights(short_codes, N_SHORT, -1))
    return weights


def traded_notional(current: dict[str, float], previous: dict[str, float]) -> float:
    codes = set(current) | set(previous)
    return float(sum(abs(current.get(code, 0.0) - previous.get(code, 0.0)) for code in codes))


def random_long_short(
    actual: np.ndarray,
    observed: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    n = len(actual)
    draws = np.empty(N_RANDOM, dtype=float)
    for draw in range(N_RANDOM):
        order = rng.permutation(n)
        long_mean = float(actual[order[:N_LONG]].mean())
        short_mean = float(actual[order[-N_SHORT:]].mean())
        draws[draw] = long_mean - short_mean
    p_hi = float((1 + np.sum(draws >= observed)) / (N_RANDOM + 1))
    return {
        "random_ls_mean": float(np.mean(draws)),
        "random_ls_p05": float(np.quantile(draws, 0.05)),
        "random_ls_p95": float(np.quantile(draws, 0.95)),
        "random_quantile": float(stats.percentileofscore(draws, observed, kind="rank") / 100.0),
        "random_p_one_sided": p_hi,
    }


def build_portfolios(
    predictions: pd.DataFrame,
    benchmark: pd.DataFrame,
    factors_q: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    previous_main: dict[str, float] = {}
    previous_baseline: dict[str, float] = {}
    portfolio_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []

    for target_q, frame in predictions.groupby("target_quarter", sort=True):
        frame = frame.copy()
        frame["rank_sentiment"] = frame["predicted"].rank(ascending=False, method="first").astype(int)
        frame["rank_baseline"] = frame["baseline_predicted"].rank(ascending=False, method="first").astype(int)
        for row in frame.itertuples(index=False):
            signal_rows.append(
                {
                    "target_quarter": target_q,
                    "issue": row.issue,
                    "publication_date": row.publication_date,
                    "publication_quarter": row.publication_quarter,
                    "e4_target_quarter": row.e4_target_quarter,
                    "target_date": row.target_date,
                    "availability_lag_days": row.availability_lag_days,
                    "code": row.code,
                    "name": row.name,
                    "split": row.split,
                    "sentiment_prediction": row.predicted,
                    "baseline_prediction": row.baseline_predicted,
                    "actual_log_return": row.actual,
                    "actual_simple_return": row.actual_simple,
                    "sentiment_rank": row.rank_sentiment,
                    "baseline_rank": row.rank_baseline,
                }
            )

        main_weights = portfolio_weights(frame, "predicted")
        baseline_weights = portfolio_weights(frame, "baseline_predicted")
        main_traded = traded_notional(main_weights, previous_main)
        baseline_traded = traded_notional(baseline_weights, previous_baseline)
        main_cost = main_traded * COST_BPS / 10000.0
        baseline_cost = baseline_traded * COST_BPS / 10000.0

        actual_by_code = frame.set_index("code")["actual_simple"].astype(float)
        names_by_code = frame.set_index("code")["name"].to_dict()
        long_codes = [code for code, weight in main_weights.items() if weight > 0]
        short_codes = [code for code, weight in main_weights.items() if weight < 0]
        baseline_long_codes = [code for code, weight in baseline_weights.items() if weight > 0]
        baseline_short_codes = [code for code, weight in baseline_weights.items() if weight < 0]

        long_return = float(sum(main_weights[c] * actual_by_code[c] for c in long_codes))
        short_basket_return = float(
            sum((-main_weights[c]) * actual_by_code[c] for c in short_codes)
        )
        long_short_gross = long_return - short_basket_return
        long_short_net = long_short_gross - main_cost

        baseline_long_return = float(
            sum(baseline_weights[c] * actual_by_code[c] for c in baseline_long_codes)
        )
        baseline_short_basket_return = float(
            sum((-baseline_weights[c]) * actual_by_code[c] for c in baseline_short_codes)
        )
        baseline_ls_gross = baseline_long_return - baseline_short_basket_return
        baseline_ls_net = baseline_ls_gross - baseline_cost

        random_stats = random_long_short(
            frame["actual_simple"].to_numpy(dtype=float),
            long_short_gross,
            rng,
        )

        for model, weights in [("sentiment_ntusd", main_weights), ("baseline_notext", baseline_weights)]:
            for code, weight in weights.items():
                weight_rows.append(
                    {
                        "model": model,
                        "target_quarter": target_q,
                        "code": code,
                        "name": names_by_code.get(code, ""),
                        "weight": weight,
                    }
                )

        portfolio_rows.append(
            {
                "target_quarter": target_q,
                "issue": frame["issue"].iloc[0],
                "split": frame["split"].iloc[0],
                "publication_date": frame["publication_date"].iloc[0],
                "publication_quarter": frame["publication_quarter"].iloc[0],
                "e4_target_quarter": frame["e4_target_quarter"].iloc[0],
                "target_date": frame["target_date"].iloc[0],
                "availability_lag_days": int(frame["availability_lag_days"].iloc[0]),
                "n_industries": int(len(frame)),
                "long_codes": ";".join(long_codes),
                "long_names": ";".join(names_by_code.get(code, "") for code in long_codes),
                "short_codes": ";".join(short_codes),
                "short_names": ";".join(names_by_code.get(code, "") for code in short_codes),
                "long_basket_return": long_return,
                "short_basket_return": short_basket_return,
                "short_contribution": -short_basket_return,
                "long_short_gross": long_short_gross,
                "one_way_turnover": main_traded / 2.0,
                "traded_notional": main_traded,
                "transaction_cost": main_cost,
                "long_short_net": long_short_net,
                "industry_ew_return": float(frame["actual_simple"].mean()),
                "baseline_long_return": baseline_long_return,
                "baseline_short_basket_return": baseline_short_basket_return,
                "baseline_long_short_gross": baseline_ls_gross,
                "baseline_traded_notional": baseline_traded,
                "baseline_transaction_cost": baseline_cost,
                "baseline_long_short_net": baseline_ls_net,
                **random_stats,
            }
        )
        previous_main = main_weights
        previous_baseline = baseline_weights

    portfolio = pd.DataFrame(portfolio_rows)
    portfolio = portfolio.merge(
        benchmark[["quarter", "market_return", "first_trade_date", "last_trade_date"]]
        .rename(columns={"quarter": "target_quarter"}),
        on="target_quarter",
        how="left",
    )
    portfolio = portfolio.merge(
        factors_q.rename(columns={"quarter": "target_quarter"}),
        on="target_quarter",
        how="left",
    )
    portfolio["long_short_net_excess"] = portfolio["long_short_net"] - portfolio["rf_return"]
    portfolio["long_short_gross_excess"] = portfolio["long_short_gross"] - portfolio["rf_return"]
    portfolio["baseline_net_excess"] = portfolio["baseline_long_short_net"] - portfolio["rf_return"]
    portfolio["market_excess"] = portfolio["market_return"] - portfolio["rf_return"]
    portfolio = portfolio.sort_values("target_quarter").reset_index(drop=True)

    if (portfolio["n_industries"] < N_LONG + N_SHORT).any():
        raise AssertionError("至少一个季度的行业数不足以形成不重叠 5/5 分组")
    check_cols = ["long_short_gross", "long_short_net", "industry_ew_return", "market_return"]
    if portfolio[check_cols].isna().any().any():
        raise AssertionError("组合收益存在缺失")

    signals = pd.DataFrame(signal_rows).sort_values(["target_quarter", "sentiment_rank"])
    weights = pd.DataFrame(weight_rows).sort_values(
        ["model", "target_quarter", "weight"], ascending=[True, True, False]
    )
    return portfolio, signals, weights


def performance_metrics(
    values: pd.Series,
    rf: pd.Series,
    turnover: pd.Series | None = None,
    periods_per_year: int = 4,
) -> dict[str, float | int]:
    frame = pd.concat([values.rename("return"), rf.rename("rf")], axis=1).dropna()
    returns = frame["return"].to_numpy(dtype=float)
    excess = (frame["return"] - frame["rf"]).to_numpy(dtype=float)
    n = int(len(frame))
    if n == 0:
        return {"n_periods": 0}
    wealth = np.cumprod(1.0 + returns)
    cumulative = float(wealth[-1] - 1.0)
    annual_return = float((wealth[-1] ** (periods_per_year / n)) - 1.0)
    annual_vol = float(np.std(returns, ddof=1) * np.sqrt(periods_per_year)) if n > 1 else float("nan")
    excess_std = np.std(excess, ddof=1) if n > 1 else float("nan")
    sharpe = (
        float(np.mean(excess) / excess_std * np.sqrt(periods_per_year))
        if n > 1 and excess_std > 0
        else float("nan")
    )
    path = np.concatenate([[1.0], wealth])
    running_max = np.maximum.accumulate(path)
    max_drawdown = float(np.min(path / running_max - 1.0))
    result: dict[str, float | int] = {
        "n_periods": n,
        "cumulative_return": cumulative,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "hit_rate": float(np.mean(returns > 0)),
        "average_period_return": float(np.mean(returns)),
    }
    if turnover is not None:
        aligned = turnover.reindex(values.index).dropna()
        result["average_turnover"] = float(aligned.mean()) if len(aligned) else float("nan")
    return result


def factor_regression(
    frame: pd.DataFrame,
    dependent: str,
    model_name: str,
    sample_name: str,
) -> dict[str, Any]:
    columns = [dependent, "MKT_RF", "SMB", "VMG"]
    reg = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(reg) < 8:
        raise AssertionError(f"{model_name}/{sample_name} 因子回归样本不足")
    y = reg[dependent].to_numpy(dtype=float)
    X = sm.add_constant(reg[["MKT_RF", "SMB", "VMG"]].to_numpy(dtype=float), has_constant="add")
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAG})
    alpha_q = float(fit.params[0])
    alpha_annual = float((1.0 + alpha_q) ** 4 - 1.0) if alpha_q > -1.0 else float("nan")
    return {
        "model": model_name,
        "sample": sample_name,
        "dependent": dependent,
        "start_quarter": reg.index.min(),
        "end_quarter": reg.index.max(),
        "n_obs": int(len(reg)),
        "alpha_quarterly": alpha_q,
        "alpha_annualized": alpha_annual,
        "alpha_se": float(fit.bse[0]),
        "alpha_t": float(fit.tvalues[0]),
        "alpha_p": float(fit.pvalues[0]),
        "MKT_RF_coef": float(fit.params[1]),
        "MKT_RF_t": float(fit.tvalues[1]),
        "SMB_coef": float(fit.params[2]),
        "SMB_t": float(fit.tvalues[2]),
        "VMG_coef": float(fit.params[3]),
        "VMG_t": float(fit.tvalues[3]),
        "r2": float(fit.rsquared),
        "adj_r2": float(fit.rsquared_adj),
        "nw_lag": NW_LAG,
        "factor_source": "CH3 daily factors, Liu-Stambaugh-Yuan (2019)",
        "factor_url": FACTOR_URL,
        "notes": "Quarterly factors compound daily returns; alpha uses HAC(1) standard errors.",
    }


def build_metric_table(portfolio: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = portfolio.set_index("target_quarter")
    series_map = {
        "sentiment_long_short_net": "情绪多空（净成本）",
        "sentiment_long_short_gross": "情绪多空（毛收益）",
        "sentiment_long_only": "情绪多头",
        "sentiment_short_contribution": "情绪空头收益贡献",
        "market": "基准：沪深300",
        "industry_ew": "基准：行业等权",
        "baseline_long_short_net": "无文本多空（净成本）",
    }
    column_map = {
        "sentiment_long_short_net": "long_short_net",
        "sentiment_long_short_gross": "long_short_gross",
        "sentiment_long_only": "long_basket_return",
        "sentiment_short_contribution": "short_contribution",
        "market": "market_return",
        "industry_ew": "industry_ew_return",
        "baseline_long_short_net": "baseline_long_short_net",
    }
    samples = {
        "full_oos": (portfolio["target_quarter"].min(), portfolio["target_quarter"].max()),
        "validation": ("2016Q2", "2020Q4"),
        "test": ("2021Q1", "2026Q3"),
    }
    rows: list[dict[str, Any]] = []
    for sample, (start, end) in samples.items():
        sample_index = indexed.index[(indexed.index >= start) & (indexed.index <= end)]
        subset = indexed.loc[sample_index]
        for series, label in series_map.items():
            turnover = subset["one_way_turnover"] if series in {
                "sentiment_long_short_net", "sentiment_long_short_gross"
            } else None
            metrics = performance_metrics(subset[column_map[series]], subset["rf_return"], turnover)
            rows.append({"sample": sample, "series": label, **metrics})
    metrics = pd.DataFrame(rows)

    regression_rows: list[dict[str, Any]] = []
    complete = portfolio[portfolio["factor_complete"]].set_index("target_quarter")
    regression_models = {
        "sentiment_long_short_net": "long_short_net_excess",
        "sentiment_long_short_gross": "long_short_gross_excess",
        "baseline_long_short_net": "baseline_net_excess",
    }
    for sample, (start, end) in samples.items():
        subset = complete[(complete.index >= start) & (complete.index <= end)]
        if len(subset) < 8:
            continue
        for model, dependent in regression_models.items():
            regression_rows.append(factor_regression(subset, dependent, model, sample))
    return metrics, pd.DataFrame(regression_rows)


def plot_equity_curve(portfolio: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    frame = portfolio.sort_values("target_quarter").copy()
    dates = pd.to_datetime(frame["target_date"]).reset_index(drop=True)
    curves = {
        "情绪多空（净成本）": frame["long_short_net"],
        "无文本多空（净成本）": frame["baseline_long_short_net"],
        "情绪多头": frame["long_basket_return"],
        "沪深300": frame["market_return"],
        "行业等权": frame["industry_ew_return"],
    }
    colors = {
        "情绪多空（净成本）": "#6F4AA8",
        "无文本多空（净成本）": "#B07A3E",
        "情绪多头": "#2F7D68",
        "沪深300": "#3E6C8E",
        "行业等权": "#8A8A8A",
    }
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    x = pd.concat([
        pd.Series([dates.iloc[0] - pd.offsets.QuarterEnd()]),
        dates,
    ]).reset_index(drop=True)
    for label, returns in curves.items():
        wealth = np.concatenate([[1.0], np.cumprod(1.0 + returns.to_numpy(dtype=float))])
        ax.plot(x, wealth, label=label, color=colors[label], linewidth=2.0 if "多空" in label else 1.5)
    boundary = pd.Timestamp("2021-01-01")
    ax.axvline(boundary, color="#9C9C9C", linestyle="--", linewidth=1.0)
    ax.text(boundary, ax.get_ylim()[1], " 测试期", va="top", ha="left", fontsize=9, color="#666666")
    ax.set_title("E7 情绪多空组合累计净值（2016Q2–2026Q3）", fontsize=16, pad=14)
    ax.set_ylabel("累计净值（起点 = 1）")
    ax.set_xlabel("目标收益季度")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.01,
        0.01,
        f"注：研究组合，5/5 等权，单边成本 {COST_BPS:.0f}bp；按 MPR 发布后首个完整季度持有。",
        fontsize=8.5,
        color="#666666",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fmt_pct(value: float | int) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value) * 100:.2f}%"


def fmt_num(value: float | int, digits: int = 2) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def markdown_table(metrics: pd.DataFrame, regressions: pd.DataFrame) -> list[str]:
    full = metrics[metrics["sample"] == "full_oos"].copy()
    alpha_map = regressions[regressions["sample"] == "full_oos"].set_index("model")
    model_series_map = {
        "情绪多空（净成本）": "sentiment_long_short_net",
        "情绪多空（毛收益）": "sentiment_long_short_gross",
        "无文本多空（净成本）": "baseline_long_short_net",
    }
    lines = [
        "| 组合/基准 | 年化收益 | 年化波动 | 夏普 | 最大回撤 | 胜率 | 平均单边换手 | 年化alpha | alpha t |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in full.itertuples(index=False):
        model = model_series_map.get(row.series)
        alpha = alpha_map.loc[model, "alpha_annualized"] if model in alpha_map.index else np.nan
        alpha_t = alpha_map.loc[model, "alpha_t"] if model in alpha_map.index else np.nan
        turnover = getattr(row, "average_turnover", np.nan)
        lines.append(
            f"| {row.series} | {fmt_pct(row.annual_return)} | {fmt_pct(row.annual_vol)} | "
            f"{fmt_num(row.sharpe)} | {fmt_pct(row.max_drawdown)} | {fmt_pct(row.hit_rate)} | "
            f"{fmt_pct(turnover)} | {fmt_pct(alpha)} | {fmt_num(alpha_t)} |"
        )
    return lines


def write_log(
    portfolio: pd.DataFrame,
    signals: pd.DataFrame,
    metrics: pd.DataFrame,
    regressions: pd.DataFrame,
) -> str:
    def row_for(sample: str, series: str) -> pd.Series:
        return metrics[(metrics["sample"] == sample) & (metrics["series"] == series)].iloc[0]

    main_full = row_for("full_oos", "情绪多空（净成本）")
    baseline_full = row_for("full_oos", "无文本多空（净成本）")
    market_full = row_for("full_oos", "基准：沪深300")
    ew_full = row_for("full_oos", "基准：行业等权")
    main_alpha = regressions[
        (regressions["sample"] == "full_oos") & (regressions["model"] == "sentiment_long_short_net")
    ].iloc[0]
    alpha_supported = bool(main_alpha["alpha_annualized"] > 0 and main_alpha["alpha_t"] >= 1.96)
    if alpha_supported:
        conclusion = (
            "因子调整后主情绪多空组合 alpha 为正且达到双侧 5% 显著性。"
            "这表示当前样本中存在统计证据，但仍需结合子样本稳定性和交易成本解释。"
        )
    else:
        conclusion = (
            "主情绪多空组合没有得到稳定的因子调整超额收益。"
            f"全样本年化净收益为 {fmt_pct(main_full['annual_return'])}，"
            f"年化 alpha 为 {fmt_pct(main_alpha['alpha_annualized'])}（t={main_alpha['alpha_t']:.2f}）。"
            "结果与 E4 测试集 IC 偏弱、预测模型未优于无文本基准一致，不能包装为策略成果。"
        )
    random_mean = portfolio["random_ls_mean"].mean()
    main_mean = portfolio["long_short_gross"].mean()
    outside_high = float(np.mean(portfolio["long_short_gross"] > portfolio["random_ls_p95"]))
    outside_low = float(np.mean(portfolio["long_short_gross"] < portfolio["random_ls_p05"]))

    lines: list[str] = []
    add = lines.append
    add("# E7 高低情绪多空组合实验日志")
    add("")
    add("## 1. 实验目的与解释边界")
    add("")
    add("- 检验 MPR 词典情绪通过行业样本外预测排序后，是否具有可解释的行业配置信息。")
    add("- 使用 E4 的样本外预测排序，并把持有期修正为报告发布后的第一个完整季度，避免使用发布前收益。")
    add("- 组合是研究组合，不代表可直接执行的 A 股行业多空策略；行业指数卖空、融券可得性和冲击成本均未建模。")
    add("")
    add("## 2. 预注册构建规则")
    add("")
    add(f"- 主信号：E4 `sentiment / {MAIN_DICT} / sent_exposure` 的样本外横截面排序值。")
    add("- 共同 MPR 情绪在单个季度内对所有行业相同，不能直接形成行业横截面排序；E7 排序的是共同情绪经行业滚动情绪暴露转换后的预测值。")
    add("- 时点修正：全部 E4 目标季度均在 MPR 发布前开始。E7 不直接采用 E4 的“报告期下一季度”收益，而把每个报告信号用于其发布后的第一个完整季度；仅使用预测排序，不使用预测幅度。E4 原目标季度保留在数据表中供审计。")
    add(f"- 每个目标季度：预测值最高 {N_LONG} 个行业等权做多，最低 {N_SHORT} 个行业等权做空；两腿不重叠。")
    add("- 多头总权重 +100%，空头总权重 -100%，组合毛敞口 200%。权重只在季度初形成，持有至季度末。")
    add(f"- 单边交易成本：{COST_BPS:.0f}bp。季度交易成本 = 交易名义额 × {COST_BPS:.0f}bp；交易名义额按当前与上期权重向量绝对变动之和计算。")
    add(f"- 随机分组：每季度随机 5/5 分组 {N_RANDOM} 次，随机种子 {SEED}；仅作为排序是否特殊的描述性对照。")
    add("- 基准：沪深 300；行业等权基准使用与预测模型相同的当季行业池。无文本 E4 预测用于判断文本情绪是否增加配置信息。")
    add("")
    add("## 3. 数据与样本")
    add("")
    add(f"- E4 样本外预测：`{E4_PRED_PATH.relative_to(ROOT)}`，与发布后完整季度收益对齐后共 {len(signals)} 个行业-季度观测、{portfolio['target_quarter'].nunique()} 个目标季度。")
    add("- 行业季度收益：`_temp/e4/industry_quarterly.csv`；仅纳入每季至少 40 个交易日且末交易日距季末不超过 7 天的观测。")
    add(f"- 行业数范围：{int(portfolio['n_industries'].min())}–{int(portfolio['n_industries'].max())}。")
    add(f"- 主样本：{portfolio['target_quarter'].min()} 至 {portfolio['target_quarter'].max()}；验证期 2016Q2–2020Q4，测试期 2021Q1–2026Q3。")
    add(f"- 市场基准：`{(BENCH_CACHE_PATH if BENCH_CACHE_PATH.exists() else BENCH_PATH).relative_to(ROOT)}`。")
    add(f"- 中国三因子：`{FACTOR_PATH.relative_to(ROOT)}`，CH3（Liu, Stambaugh & Yuan, 2019, *Size and Value in China*）；{FACTOR_URL}。")
    add(f"- 因子说明：`{FACTOR_DOC_PATH.relative_to(ROOT)}`；{FACTOR_DOC_URL}。")
    add(f"- 日因子复利汇总至季度；文件截至 2026-08-31，2026Q2 仅有 43/60 个交易日、2026Q3 仅有 44/65 个交易日，两季被排除出因子回归。完整因子回归季度数：{int(portfolio['factor_complete'].sum())}。")
    add("")
    add("## 4. 数据完整性")
    add("")
    add(f"- 至少每季度行业数：{int(portfolio['n_industries'].min())}，满足不重叠 5/5 分组。")
    add(f"- 长、空两腿重叠检查：通过；每季度均恰有 {N_LONG} 个多头和 {N_SHORT} 个空头行业。")
    add(f"- 发布日期与形成日检查：通过；信号发布日期至目标季度开始的间隔为 {int(portfolio['availability_lag_days'].min())}–{int(portfolio['availability_lag_days'].max())} 天，未使用发布前收益。")
    add(f"- 组合收益缺失：{int(portfolio[['long_short_gross', 'long_short_net', 'market_return', 'industry_ew_return']].isna().sum().sum())}。")
    add(f"- 首季初始化单边换手：{fmt_pct(portfolio.iloc[0]['one_way_turnover'])}；之后随行业重新排序变化。")
    add("")
    add("## 5. 绩效结果")
    add("")
    add(f"- 主多空全样本：累计净收益 {fmt_pct(main_full['cumulative_return'])}，年化收益 {fmt_pct(main_full['annual_return'])}，年化波动 {fmt_pct(main_full['annual_vol'])}，夏普 {main_full['sharpe']:.2f}，最大回撤 {fmt_pct(main_full['max_drawdown'])}，季度胜率 {fmt_pct(main_full['hit_rate'])}。")
    add(f"- 无文本多空：年化收益 {fmt_pct(baseline_full['annual_return'])}，夏普 {baseline_full['sharpe']:.2f}，最大回撤 {fmt_pct(baseline_full['max_drawdown'])}。")
    add(f"- 沪深300：年化收益 {fmt_pct(market_full['annual_return'])}，夏普 {market_full['sharpe']:.2f}；同行业池等权：年化收益 {fmt_pct(ew_full['annual_return'])}，夏普 {ew_full['sharpe']:.2f}。")
    add(f"- 随机 5/5 分组的季度多空毛收益均值平均为 {fmt_pct(random_mean)}；主组合季度均值 {fmt_pct(main_mean)}。主组合高于随机 95 分位的季度占 {fmt_pct(outside_high)}，低于 5 分位的季度占 {fmt_pct(outside_low)}。")
    add("")
    add("### 全样本绩效与因子 alpha")
    add("")
    lines.extend(markdown_table(metrics, regressions))
    add("")
    add("`年化alpha` 为季度 CH3 回归截距的四季度复利换算；标准误采用 Newey-West HAC(1)。")
    add("")
    add("### 子样本稳定性")
    add("")
    add("| 样本 | 组合 | 年化收益 | 年化波动 | 夏普 | 最大回撤 | 胜率 |")
    add("|---|---|---:|---:|---:|---:|---:|")
    for sample in ["validation", "test"]:
        for series in ["情绪多空（净成本）", "无文本多空（净成本）", "基准：沪深300", "基准：行业等权"]:
            row = row_for(sample, series)
            add(
                f"| {sample} | {series} | {fmt_pct(row['annual_return'])} | {fmt_pct(row['annual_vol'])} | "
                f"{fmt_num(row['sharpe'])} | {fmt_pct(row['max_drawdown'])} | {fmt_pct(row['hit_rate'])} |"
            )
    add("")
    add("## 6. 因子稳健性回归")
    add("")
    add("| 样本 | 模型 | N | 年化alpha | alpha t | MKT beta | SMB beta | VMG beta | R² |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in regressions.itertuples(index=False):
        add(
            f"| {row.sample} | {row.model} | {row.n_obs} | {fmt_pct(row.alpha_annualized)} | {row.alpha_t:.2f} | "
            f"{row.MKT_RF_coef:.3f} | {row.SMB_coef:.3f} | {row.VMG_coef:.3f} | {row.r2:.3f} |"
        )
    add("")
    add("## 7. 结论")
    add("")
    add(conclusion)
    add("")
    add("## 8. 局限")
    add("")
    add(f"- 样本外期间仅 {portfolio['target_quarter'].nunique()} 个季度，且 2016Q2 起才有发布后完整季度，统计功效有限。")
    add("- 使用申万行业指数，未模拟个股卖空成本、涨跌停、融券限制和行业指数可交易性。")
    add("- 交易成本固定为单边 10bp，是研究假设；记录换手与成本不使组合成为可执行策略。")
    add("- E4 预测的原目标是报告期下一季度；E7 为消除发布时点泄漏而改变评估持有期，只使用排序值。两者收益幅度不可直接互相替代。")
    add("- CH3 因子本地文件在 2026Q2 和 2026Q3 覆盖不完整，因子回归止于 2026Q1。")
    add("- E4 表明共同情绪变量与横截面截距共线；E7 排序依赖行业滚动情绪暴露，若暴露估计不稳定，组合排序也会不稳定。")
    add("")
    add("## 9. 输出与复现")
    add("")
    add("- 组合季度表：`experiments/E7/e7_portfolio.csv`。")
    add("- 因子回归：`experiments/E7/e7_factor_regression.csv`。")
    add("- 累计净值图：`experiments/E7/e7_equity_curve.png`。")
    add("- 缓存信号、权重与因子季度数据：`_temp/E7/`。")
    add("```powershell")
    add("python experiments/E7/run_e7.py")
    add("```")
    add(f"- 脚本 SHA256：`{sha256(Path(__file__))}`。")
    add(f"- 运行环境：Python {platform.python_version()}；pandas {pd.__version__}；numpy {np.__version__}；statsmodels {sm.__version__}。")
    return "\n".join(lines) + "\n"


def main() -> None:
    E7_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    log("[1/5] loading predictions, benchmark and factors")
    predictions = load_predictions()
    benchmark = load_benchmark()
    factor_daily = load_factor_daily()
    factors_q = aggregate_factors(factor_daily, benchmark)
    log(
        f"[data] predictions={len(predictions)}, quarters={predictions['target_quarter'].nunique()}, "
        f"industries={predictions['code'].nunique()}"
    )
    log(
        f"[data] factors={len(factor_daily)}, range {factor_daily['date'].min().date()}.."
        f"{factor_daily['date'].max().date()}, complete quarters={int(factors_q['factor_complete'].sum())}"
    )

    log("[2/5] building pre-specified portfolios")
    portfolio, signals, weights = build_portfolios(predictions, benchmark, factors_q)
    log(
        f"[portfolio] quarters={len(portfolio)}, range "
        f"{portfolio['target_quarter'].min()}..{portfolio['target_quarter'].max()}"
    )

    log("[3/5] computing metrics and factor regressions")
    metrics, regressions = build_metric_table(portfolio)
    main_alpha = regressions[
        (regressions["sample"] == "full_oos") & (regressions["model"] == "sentiment_long_short_net")
    ].iloc[0]
    log(
        f"[alpha] annualized={main_alpha['alpha_annualized']:.4f}, "
        f"t={main_alpha['alpha_t']:.2f}, N={int(main_alpha['n_obs'])}"
    )

    log("[4/5] writing tables and caches")
    portfolio.to_csv(E7_DIR / "e7_portfolio.csv", index=False, encoding="utf-8-sig")
    regressions.to_csv(E7_DIR / "e7_factor_regression.csv", index=False, encoding="utf-8-sig")
    signals.to_csv(TEMP_DIR / "e7_signal_panel.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(TEMP_DIR / "e7_weights.csv", index=False, encoding="utf-8-sig")
    factors_q.to_csv(TEMP_DIR / "e7_factor_quarterly.csv", index=False, encoding="utf-8-sig")

    log("[5/5] plotting equity curve and writing log")
    plot_equity_curve(portfolio, E7_DIR / "e7_equity_curve.png")
    log_text = write_log(portfolio, signals, metrics, regressions)
    (LOG_DIR / "e7_portfolio.md").write_text(log_text, encoding="utf-8")

    main_metrics = metrics[
        (metrics["sample"] == "full_oos") & (metrics["series"] == "情绪多空（净成本）")
    ].iloc[0]
    summary = {
        "target_quarters": int(len(portfolio)),
        "start": portfolio["target_quarter"].min(),
        "end": portfolio["target_quarter"].max(),
        "signal_rows": int(len(signals)),
        "factor_complete_quarters": int(portfolio["factor_complete"].sum()),
        "main_net_annual_return": float(main_metrics["annual_return"]),
        "main_alpha_annualized": float(main_alpha["alpha_annualized"]),
        "main_alpha_t": float(main_alpha["alpha_t"]),
        "script_sha256": sha256(Path(__file__)),
    }
    (E7_DIR / "e7_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"[done] {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
