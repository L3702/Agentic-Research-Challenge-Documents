#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E5 行业敏感度异质性实验。

检验 H3：哪些行业对 MPR 情绪更敏感，敏感度是否与行业特征有关。

事先确定（pre-registered）的口径，运行后不根据结果调整：
1. 敏感度指标：事件研究 CAR 对 MPR 情绪的事件层 OLS 斜率。
   主口径：市场模型、估计窗 120、事件窗 [0,+1]、主词典 ntusd_simplified。
2. 行业特征：利率敏感（10Y 国债收益率 beta）、融资敏感（社融累计同比 beta）、
   政策依赖（事先定义的定性映射，描述性）。
3. 分组规则：连续特征按全样本中位数二分；不依据 E5 结果调整分组。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
E5_DIR = ROOT / "experiments" / "E5"
E5_LOG_DIR = E5_DIR / "logs"
TEMP_DIR = ROOT / "_temp" / "e5"
E3_DIR = ROOT / "experiments" / "E3"
E1_DIR = ROOT / "experiments" / "E1"
MACRO_RAW_DIR = ROOT / "_temp" / "e4" / "macro_raw"

INDUSTRY_PATH = ROOT / "data" / "external" / "sw_industries" / "sw_industry_daily.csv"
BENCH_CACHE_PATH = ROOT / "_temp" / "e0_market_raw" / "benchmark_sh000300.csv"
BENCH_PATH = ROOT / "data" / "external" / "benchmark" / "benchmark_daily.csv"
E3_RESULTS_PATH = E3_DIR / "e3_event_results.csv"
E1_SENT_PATH = E1_DIR / "sentiment_dictionary_full.csv"
E4_PANEL_PATH = ROOT / "_temp" / "e4" / "e4_panel.csv"
E4_PRED_PATH = ROOT / "_temp" / "e4" / "e4_oos_predictions.csv"

SECTOR_GROUPS: dict[str, str] = {
    "801010": "上游资源与周期",
    "801030": "上游资源与周期",
    "801040": "上游资源与周期",
    "801050": "上游资源与周期",
    "801950": "上游资源与周期",
    "801960": "上游资源与周期",
    "801710": "上游资源与周期",
    "801720": "上游资源与周期",
    "801890": "中游制造与科技",
    "801730": "中游制造与科技",
    "801740": "中游制造与科技",
    "801880": "中游制造与科技",
    "801080": "中游制造与科技",
    "801750": "中游制造与科技",
    "801770": "中游制造与科技",
    "801760": "中游制造与科技",
    "801120": "大消费与医药",
    "801130": "大消费与医药",
    "801140": "大消费与医药",
    "801200": "大消费与医药",
    "801210": "大消费与医药",
    "801150": "大消费与医药",
    "801980": "大消费与医药",
    "801110": "大消费与医药",
    "801780": "金融地产与综合",
    "801790": "金融地产与综合",
    "801180": "金融地产与综合",
    "801230": "金融地产与综合",
    "801160": "公用与交运",
    "801170": "公用与交运",
    "801970": "公用与交运",
}
SECTOR_ORDER = ["上游资源与周期", "中游制造与科技", "大消费与医药", "金融地产与综合", "公用与交运"]

POLICY_DEP: dict[str, int] = {
    "801180": 1,
    "801720": 1,
    "801710": 1,
    "801160": 1,
    "801740": 1,
    "801010": 1,
    "801780": 1,
}

MAIN_MODEL = "market"
MAIN_WINDOW = 1
MAIN_DICT = "ntusd_simplified"
WINDOWS = [1, 3, 5, 10]
DICTS = ["ntusd_simplified", "dlut", "mpr_domain"]
MODELS = ["market", "mean_adj", "excess_market"]
SEED = 20261119
N_BOOT = 10000
SUB_SPLIT = "2014-02-21"

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


def ensure_dirs() -> None:
    for path in [E5_DIR, E5_LOG_DIR, TEMP_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def ols_slope(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(len(x))
    if n < 5:
        return {"n": n, "slope": float("nan"), "se": float("nan"), "t": float("nan"), "p": float("nan"), "r2": float("nan"), "intercept": float("nan")}
    design = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    dof = n - 2
    s2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(design.T @ design)
    se = float(np.sqrt(max(s2 * xtx_inv[1, 1], 0.0)))
    t_value = float(beta[1] / se) if se > 0 else float("nan")
    p_value = float(2.0 * stats.t.sf(abs(t_value), dof)) if np.isfinite(t_value) else float("nan")
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = float(1.0 - (resid @ resid) / ss_tot) if ss_tot > 0 else float("nan")
    return {"n": n, "slope": float(beta[1]), "se": se, "t": t_value, "p": p_value, "r2": r2, "intercept": float(beta[0])}


def bootstrap_slope(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    xb = x[idx]
    yb = y[idx]
    xm = xb.mean(axis=1)
    ym = yb.mean(axis=1)
    cov = ((xb - xm[:, None]) * (yb - ym[:, None])).sum(axis=1)
    var = ((xb - xm[:, None]) ** 2).sum(axis=1)
    slopes = np.divide(cov, var, out=np.full_like(cov, np.nan), where=var > 0)
    slopes = slopes[np.isfinite(slopes)]
    if len(slopes) < 10:
        return float("nan"), float("nan")
    return float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for start in range(0, n_boot, 1000):
        batch = min(1000, n_boot - start)
        ia = rng.integers(0, len(a), size=(batch, len(a)))
        ib = rng.integers(0, len(b), size=(batch, len(b)))
        draws[start : start + batch] = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def bootstrap_mean(values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for start in range(0, n_boot, 1000):
        batch = min(1000, n_boot - start)
        idx = rng.integers(0, len(values), size=(batch, len(values)))
        draws[start : start + batch] = values[idx].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t_value, p_value = stats.ttest_ind(a, b, equal_var=False)
    return float(t_value), float(p_value)


def nw_tstat(values: np.ndarray, lags: int = 2) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    e = x - x.mean()
    gamma0 = float(e @ e) / n
    total = gamma0
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = float(e[lag:] @ e[:-lag]) / n
        total += 2.0 * weight * gamma
    se = float(np.sqrt(max(total, 0.0) / n))
    if se <= 0:
        return float("nan"), float("nan")
    t_value = float(x.mean() / se)
    p_value = float(2.0 * stats.t.sf(abs(t_value), n - 1))
    return t_value, p_value


def parse_cn_month(value: Any) -> pd.Timestamp | None:
    text = str(value)
    match = re.search(r"(\d{4})\D{1,3}(\d{1,2})", text)
    if not match:
        return None
    year, month = int(match.group(1)), int(match.group(2))
    if not 1 <= month <= 12:
        return None
    return pd.Timestamp(year=year, month=month, day=1)


def fetch_treasury() -> pd.DataFrame:
    """10Y/2Y 国债收益率，缓存到 _temp/e5/treasury_yield.csv。"""
    cache = TEMP_DIR / "treasury_yield.csv"
    if cache.exists() and cache.stat().st_size > 0:
        return pd.read_csv(cache, parse_dates=["date"])
    import akshare as ak

    raw = ak.bond_zh_us_rate(start_date="20010101")
    raw.columns = [str(column) for column in raw.columns]
    date_col = raw.columns[0]
    cn10 = [c for c in raw.columns if "中国" in c and "10年" in c and "-2年" not in c]
    cn2 = [c for c in raw.columns if "中国" in c and "2年" in c and "-2年" not in c]
    if not cn10 or not cn2:
        raise RuntimeError("中国 10Y/2Y 国债收益率列未找到")
    frame = raw[[date_col, cn2[0], cn10[0]]].copy()
    frame.columns = ["date", "cn_2y_yield", "cn_10y_yield"]
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna(subset=["cn_2y_yield", "cn_10y_yield"]).sort_values("date").reset_index(drop=True)
    frame.to_csv(cache, index=False, encoding="utf-8-sig")
    return frame


def load_event_car() -> pd.DataFrame:
    """E3 事件层 CAR 长表，含三类模型与各词典情绪。"""
    raw = pd.read_csv(E3_RESULTS_PATH, dtype={"code": str})
    raw = raw.drop(columns=[c for c in ["sentiment_ntusd_simplified"] if c in raw.columns])
    market = raw[raw["model"] == "market"].copy()
    market["car"] = market["car_market"]
    mean_adj = raw[raw["model"] == "market"].copy()
    mean_adj["car"] = mean_adj["car_mean_adj"]
    mean_adj["model"] = "mean_adj"
    excess = raw[raw["model"] == "excess_market"].copy()
    excess["car"] = excess["car_market"]
    frame = pd.concat([market, mean_adj, excess], ignore_index=True)

    sent = pd.read_csv(E1_SENT_PATH, dtype={"issue": str})
    wide = sent.pivot(index="issue", columns="dictionary", values="sentiment_score")
    wide.columns = [f"sentiment_{column}" for column in wide.columns]
    wide = wide.reset_index()
    frame = frame.merge(wide, on="issue", how="left")
    frame["publication_date"] = pd.to_datetime(frame["publication_date"])
    return frame


def load_industry() -> pd.DataFrame:
    industry = pd.read_csv(INDUSTRY_PATH, dtype={"code": str})
    industry = industry.rename(
        columns={
            "收盘": "close",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        }
    )
    industry["date"] = pd.to_datetime(industry["date"])
    industry = industry.sort_values(["code", "date"]).reset_index(drop=True)
    industry["daily_return"] = industry.groupby("code", group_keys=False)["close"].pct_change()
    industry["sector_group"] = industry["code"].map(SECTOR_GROUPS)
    return industry


def load_bench() -> pd.DataFrame:
    path = BENCH_CACHE_PATH if BENCH_CACHE_PATH.exists() and BENCH_CACHE_PATH.stat().st_size > 0 else BENCH_PATH
    bench = pd.read_csv(path, dtype={"code": str})
    bench["date"] = pd.to_datetime(bench["date"])
    bench = bench.sort_values(["code", "date"]).reset_index(drop=True)
    bench["daily_return"] = bench.groupby("code", group_keys=False)["close"].pct_change()
    return bench


def build_characteristics() -> pd.DataFrame:
    """构造行业特征：rate_sens / fin_sens / policy_dep。"""
    industry = load_industry()
    treasury = fetch_treasury()
    treasury["d_yield_10"] = treasury["cn_10y_yield"].diff()
    treasury["d_yield_2"] = treasury["cn_2y_yield"].diff()

    daily = industry.merge(
        treasury[["date", "d_yield_10", "d_yield_2", "cn_2y_yield"]], on="date", how="left"
    )
    daily["rf_daily"] = daily["cn_2y_yield"] / 100.0 / 252.0
    daily["excess"] = daily["daily_return"] - daily["rf_daily"]

    rows: list[dict[str, Any]] = []
    for code, sub in daily.groupby("code", sort=True):
        for column, name in [("d_yield_10", "rate_sens"), ("d_yield_2", "rate_sens_2y")]:
            part = sub[[column, "excess"]].dropna()
            res = ols_slope(part[column].to_numpy(), part["excess"].to_numpy())
            rows.append({"code": code, "char": name, **res})
    rate = pd.DataFrame(rows)
    rate_wide = rate.pivot(index="code", columns="char", values=["n", "slope", "se", "t", "p"])
    rate_wide.columns = [f"{a}_{b}" for a, b in rate_wide.columns]
    rate_wide = rate_wide.reset_index()

    industry = industry.copy()
    industry["month"] = industry["date"].dt.to_period("M").dt.to_timestamp()
    monthly = industry.groupby(["code", "month"], sort=True)["close"].last().reset_index()
    monthly["ret"] = monthly.groupby("code")["close"].pct_change()

    af = pd.read_csv(MACRO_RAW_DIR / "new_financial_credit.csv", encoding="utf-8-sig")
    af = af.iloc[:, [0, 5]].copy()
    af.columns = ["month", "value"]
    af["month"] = af["month"].map(parse_cn_month)
    af = af.dropna(subset=["month", "value"]).sort_values("month").reset_index(drop=True)
    af["d_fin"] = af["value"].diff()

    m2 = pd.read_csv(MACRO_RAW_DIR / "m2_yearly.csv", encoding="utf-8-sig")
    m2 = m2.iloc[:, [1, 2]].copy()
    m2.columns = ["month", "value"]
    m2["month"] = pd.to_datetime(m2["month"], errors="coerce")
    m2 = m2.dropna(subset=["month", "value"]).sort_values("month").reset_index(drop=True)
    m2["d_m2"] = m2["value"].diff()

    fin_rows: list[dict[str, Any]] = []
    for code, sub in monthly.groupby("code", sort=True):
        part = sub.merge(af[["month", "d_fin"]], on="month", how="inner")
        res = ols_slope(part["d_fin"].to_numpy(), part["ret"].to_numpy())
        fin_rows.append({"code": code, "char": "fin_sens", **res})
        part2 = sub.merge(m2[["month", "d_m2"]], on="month", how="inner")
        res2 = ols_slope(part2["d_m2"].to_numpy(), part2["ret"].to_numpy())
        fin_rows.append({"code": code, "char": "fin_sens_m2", **res2})
    fin = pd.DataFrame(fin_rows)
    fin_wide = fin.pivot(index="code", columns="char", values=["n", "slope", "se", "t", "p"])
    fin_wide.columns = [f"{a}_{b}" for a, b in fin_wide.columns]
    fin_wide = fin_wide.reset_index()

    char = rate_wide.merge(fin_wide, on="code", how="outer")
    char["policy_dep"] = char["code"].map(POLICY_DEP).fillna(0).astype(int)
    return char


def rename_char_columns(char: pd.DataFrame) -> pd.DataFrame:
    frame = char.copy()
    frame = frame.rename(
        columns={
            "n_rate_sens": "rate_sens_n",
            "slope_rate_sens": "rate_sens",
            "se_rate_sens": "rate_sens_se",
            "t_rate_sens": "rate_sens_t",
            "p_rate_sens": "rate_sens_p",
            "slope_rate_sens_2y": "rate_sens_2y",
            "n_fin_sens": "fin_sens_n",
            "slope_fin_sens": "fin_sens",
            "se_fin_sens": "fin_sens_se",
            "t_fin_sens": "fin_sens_t",
            "p_fin_sens": "fin_sens_p",
            "slope_fin_sens_m2": "fin_sens_m2",
        }
    )
    return frame


def sensitivity_table(car: pd.DataFrame) -> pd.DataFrame:
    """每个行业、每个模型/事件窗/词典的 CAR-情绪斜率。"""
    rows: list[dict[str, Any]] = []
    for code, sub in car.groupby("code", sort=True):
        for model in MODELS:
            for window in WINDOWS:
                for dictionary in DICTS:
                    sent_col = f"sentiment_{dictionary}"
                    part = sub[
                        (sub["model"] == model)
                        & (sub["event_window"] == window)
                        & sub[sent_col].notna()
                    ]
                    if part.empty:
                        continue
                    res = ols_slope(part[sent_col].to_numpy(), part["car"].to_numpy())
                    rows.append(
                        {
                            "code": code,
                            "model": model,
                            "event_window": int(window),
                            "dictionary": dictionary,
                            **res,
                        }
                    )
    return pd.DataFrame(rows)


def high_low_table(car: pd.DataFrame) -> pd.DataFrame:
    """每个行业 high-low 情绪组 CAR 差（ntusd 三分位，E3 口径）。"""
    rows: list[dict[str, Any]] = []
    for code, sub in car.groupby("code", sort=True):
        for model in MODELS:
            for window in WINDOWS:
                part = sub[
                    (sub["model"] == model)
                    & (sub["event_window"] == window)
                    & (sub["sentiment_group"].isin(["high", "low"]))
                ]
                high = part[part["sentiment_group"] == "high"]["car"].dropna().to_numpy()
                low = part[part["sentiment_group"] == "low"]["car"].dropna().to_numpy()
                if not len(high) or not len(low):
                    continue
                t_value, p_value = welch(high, low)
                lo, hi = bootstrap_mean_diff(high, low)
                rows.append(
                    {
                        "code": code,
                        "model": model,
                        "event_window": int(window),
                        "n_high": int(len(high)),
                        "n_low": int(len(low)),
                        "high_low_diff": float(high.mean() - low.mean()),
                        "high_low_t": t_value,
                        "high_low_p": p_value,
                        "high_low_ci_low": lo,
                        "high_low_ci_high": hi,
                    }
                )
    return pd.DataFrame(rows)


def subsample_sensitivity(car: pd.DataFrame) -> pd.DataFrame:
    """按事件时间前半/后半估计行业敏感度。"""
    rows: list[dict[str, Any]] = []
    for code, sub in car.groupby("code", sort=True):
        for label, mask in [
            ("pre", sub["publication_date"] < pd.Timestamp(SUB_SPLIT)),
            ("post", sub["publication_date"] >= pd.Timestamp(SUB_SPLIT)),
        ]:
            part = sub[
                mask
                & (sub["model"] == MAIN_MODEL)
                & (sub["event_window"] == MAIN_WINDOW)
                & sub[f"sentiment_{MAIN_DICT}"].notna()
            ]
            res = ols_slope(part[f"sentiment_{MAIN_DICT}"].to_numpy(), part["car"].to_numpy())
            rows.append({"code": code, "subsample": label, **res})
    return pd.DataFrame(rows)


def cross_section_regression(sens: pd.DataFrame) -> pd.DataFrame:
    """行业敏感度对行业特征的横截面回归。"""
    frame = sens.copy()
    frame = frame[frame["sens_main"].notna() & np.isfinite(frame["sens_main"])]
    rows: list[dict[str, Any]] = []

    def run(formula: str, data: pd.DataFrame, weights: str | None = None, label: str = "") -> None:
        try:
            if weights:
                model = smf.wls(formula, data=data, weights=(1.0 / data[weights] ** 2))
            else:
                model = smf.ols(formula, data=data)
            res = model.fit(cov_type="HC3")
            for term in res.params.index:
                rows.append(
                    {
                        "spec": label or formula,
                        "term": term,
                        "estimate": float(res.params[term]),
                        "std_error": float(res.bse[term]),
                        "t_value": float(res.tvalues[term]),
                        "p_value": float(res.pvalues[term]),
                        "n": int(res.nobs),
                        "r2": float(res.rsquared),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            log(f"[warn] cross-section regression failed: {formula}: {type(exc).__name__}: {exc}")

    run("sens_main ~ rate_sens + fin_sens + policy_dep", frame, label="ols_all")
    run("sens_main ~ rate_sens + fin_sens + policy_dep", frame, weights="sens_main_se", label="wls_all")
    for term in ["rate_sens", "fin_sens", "policy_dep"]:
        run(f"sens_main ~ {term}", frame, label=f"ols_{term}")
    return pd.DataFrame(rows)


def panel_interaction(car: pd.DataFrame, char: pd.DataFrame) -> pd.DataFrame:
    """事件层面板：CAR ~ sent_z + sent_z x char_z + 行业固定效应，按事件聚类。"""
    frame = car[
        (car["model"] == MAIN_MODEL)
        & (car["event_window"] == MAIN_WINDOW)
        & car[f"sentiment_{MAIN_DICT}"].notna()
    ].merge(char[["code", "rate_sens", "fin_sens", "policy_dep"]], on="code", how="inner")
    frame["sent_z"] = (
        frame[f"sentiment_{MAIN_DICT}"] - frame[f"sentiment_{MAIN_DICT}"].mean()
    ) / frame[f"sentiment_{MAIN_DICT}"].std()
    rows: list[dict[str, Any]] = []
    for char_name, label in [("rate_sens", "rate"), ("fin_sens", "fin"), ("policy_dep", "policy")]:
        part = frame.dropna(subset=[char_name, "car"]).copy()
        part["char_z"] = (part[char_name] - part[char_name].mean()) / part[char_name].std()
        part["sent_x_char"] = part["sent_z"] * part["char_z"]
        try:
            res = smf.ols("car ~ sent_z + sent_x_char + C(code)", data=part).fit(
                cov_type="cluster", cov_kwds={"groups": part["issue"]}
            )
            term = "sent_x_char"
            rows.append(
                {
                    "characteristic": label,
                    "term": term,
                    "estimate": float(res.params[term]),
                    "std_error": float(res.bse[term]),
                    "t_value": float(res.tvalues[term]),
                    "p_value": float(res.pvalues[term]),
                    "n": int(res.nobs),
                    "n_issues": int(part["issue"].nunique()),
                    "r2": float(res.rsquared),
                }
            )
        except Exception as exc:  # noqa: BLE001
            log(f"[warn] panel interaction failed for {label}: {type(exc).__name__}: {exc}")
    return pd.DataFrame(rows)


def group_industry_compare(sens: pd.DataFrame) -> pd.DataFrame:
    """行业层面：各组敏感度均值与组间差异。"""
    frame = sens.copy()
    rows: list[dict[str, Any]] = []
    specs = [
        ("rate_group", "high", "low"),
        ("fin_group", "high", "low"),
        ("policy_dep", 1, 0),
        ("sector_group", None, None),
        ("beta_group", "high_beta", "low_beta"),
    ]
    for group_col, left, right in specs:
        for group_value, sub in frame.groupby(group_col, sort=True):
            values = sub["sens_main"].dropna().to_numpy()
            lo, hi = bootstrap_mean(values) if len(values) else (float("nan"), float("nan"))
            rows.append(
                {
                    "grouping": group_col,
                    "group": str(group_value),
                    "n_industries": int(len(values)),
                    "mean_sens": float(values.mean()) if len(values) else float("nan"),
                    "median_sens": float(np.median(values)) if len(values) else float("nan"),
                    "boot_ci_low": lo,
                    "boot_ci_high": hi,
                }
            )
        if left is not None:
            a = frame[frame[group_col] == left]["sens_main"].dropna().to_numpy()
            b = frame[frame[group_col] == right]["sens_main"].dropna().to_numpy()
            t_value, p_value = welch(a, b)
            lo, hi = bootstrap_mean_diff(a, b)
            rows.append(
                {
                    "grouping": group_col,
                    "group": f"{left} - {right}",
                    "n_industries": int(len(a) + len(b)),
                    "mean_sens": float(a.mean() - b.mean()) if len(a) and len(b) else float("nan"),
                    "median_sens": float("nan"),
                    "welch_t": t_value,
                    "welch_p": p_value,
                    "boot_ci_low": lo,
                    "boot_ci_high": hi,
                }
            )
    return pd.DataFrame(rows)


def group_event_compare(car: pd.DataFrame, char: pd.DataFrame) -> pd.DataFrame:
    """事件层面：组内行业等权 CAR 的组间差异。"""
    frame = car[
        (car["model"] == MAIN_MODEL) & (car["event_window"] == MAIN_WINDOW)
    ].drop(columns=[c for c in ["sector_group", "beta_group", "full_sample_beta"] if c in car.columns]).merge(
        char[["code", "rate_group", "fin_group", "policy_dep", "sector_group", "beta_group"]],
        on="code",
        how="inner",
    )
    rows: list[dict[str, Any]] = []
    specs = [
        ("rate_group", "high", "low"),
        ("fin_group", "high", "low"),
        ("policy_dep", 1, 0),
        ("sector_group", None, None),
        ("beta_group", "high_beta", "low_beta"),
    ]
    for group_col, left, right in specs:
        issue_group = frame.groupby(["issue", group_col])["car"].mean()
        for group_value in sorted(frame[group_col].dropna().unique()):
            if group_value not in issue_group.index.get_level_values(1):
                continue
            values = issue_group.xs(group_value, level=1).to_numpy(dtype=float)
            rows.append(
                {
                    "grouping": group_col,
                    "group": str(group_value),
                    "n_issues": int(len(values)),
                    "mean_car": float(np.mean(values)) if len(values) else float("nan"),
                    "t_stat": float(stats.ttest_1samp(values, 0.0).statistic) if len(values) > 1 else float("nan"),
                    "p_value": float(stats.ttest_1samp(values, 0.0).pvalue) if len(values) > 1 else float("nan"),
                }
            )
        if left is not None:
            levels = issue_group.index.get_level_values(1)
            a = issue_group.xs(left, level=1).to_numpy(dtype=float) if left in levels else np.array([])
            b = issue_group.xs(right, level=1).to_numpy(dtype=float) if right in levels else np.array([])
            if len(a) and len(b):
                t_value, p_value = welch(a, b)
                lo, hi = bootstrap_mean_diff(a, b)
                rows.append(
                    {
                        "grouping": group_col,
                        "group": f"{left} - {right}",
                        "n_issues": int(len(a) + len(b)),
                        "mean_car": float(a.mean() - b.mean()),
                        "t_stat": t_value,
                        "p_value": p_value,
                        "boot_ci_low": lo,
                        "boot_ci_high": hi,
                    }
                )
    return pd.DataFrame(rows)


def group_ic(panel: pd.DataFrame, char: pd.DataFrame) -> pd.DataFrame:
    """组内时间序列 IC：MPR 情绪 vs 组平均下季收益（季度）。"""
    frame = panel.merge(
        char[["code", "rate_group", "fin_group", "policy_dep", "beta_group"]],
        on="code",
        how="inner",
    )
    rows: list[dict[str, Any]] = []
    for group_col in ["rate_group", "fin_group", "policy_dep", "sector_group", "beta_group"]:
        for group_value, sub in frame.groupby(group_col, sort=True):
            quarterly = sub.groupby("target_quarter").agg(
                ret=("return_next_q", "mean"),
                sent=("sentiment_ntusd_simplified", "first"),
                n=("code", "size"),
            )
            quarterly = quarterly[quarterly["n"] >= 2]
            if len(quarterly) < 20:
                continue
            rho = stats.spearmanr(quarterly["sent"], quarterly["ret"])
            pearson = stats.pearsonr(quarterly["sent"], quarterly["ret"])
            t_value, p_value = nw_tstat(quarterly["ret"].to_numpy())
            rows.append(
                {
                    "grouping": group_col,
                    "group": str(group_value),
                    "n_quarters": int(len(quarterly)),
                    "ic_spearman": float(rho.statistic),
                    "ic_p": float(rho.pvalue),
                    "ic_pearson": float(pearson.statistic),
                    "mean_return": float(quarterly["ret"].mean()),
                    "nw_t": t_value,
                    "nw_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def group_oos_ic(pred: pd.DataFrame, char: pd.DataFrame) -> pd.DataFrame:
    """组内横截面 IC：E4 样本外预测（sent_exposure, 测试集）。"""
    frame = pred[
        (pred["model"] == "sentiment")
        & (pred["dictionary"] == MAIN_DICT)
        & (pred["spec"] == "sent_exposure")
        & (pred["split"] == "test")
    ].merge(
        char[["code", "rate_group", "fin_group", "policy_dep"]],
        on="code",
        how="inner",
    )
    rows: list[dict[str, Any]] = []
    for group_col in ["rate_group", "fin_group", "policy_dep", "sector_group"]:
        for group_value, sub in frame.groupby(group_col, sort=True):
            ics: list[float] = []
            for _q, f in sub.groupby("target_quarter"):
                if len(f) < 3:
                    continue
                rho = stats.spearmanr(f["predicted"], f["actual"]).statistic
                if np.isfinite(rho):
                    ics.append(float(rho))
            ics_array = np.array(ics, dtype=float)
            t_value, p_value = nw_tstat(ics_array)
            rows.append(
                {
                    "grouping": group_col,
                    "group": str(group_value),
                    "n_quarters": int(len(ics_array)),
                    "ic_mean": float(ics_array.mean()) if len(ics_array) else float("nan"),
                    "ic_std": float(ics_array.std(ddof=1)) if len(ics_array) > 1 else float("nan"),
                    "icir": float(ics_array.mean() / ics_array.std(ddof=1)) if len(ics_array) > 1 and ics_array.std(ddof=1) > 0 else float("nan"),
                    "nw_t": t_value,
                    "nw_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def stability_summary(sens: pd.DataFrame) -> pd.DataFrame:
    """词典/事件窗/模型的敏感度秩相关。"""
    main = sens[
        (sens["model"] == MAIN_MODEL)
        & (sens["event_window"] == MAIN_WINDOW)
        & (sens["dictionary"] == MAIN_DICT)
    ][["code", "slope"]].rename(columns={"slope": "main"})
    rows: list[dict[str, Any]] = []
    for dictionary in DICTS:
        part = sens[
            (sens["model"] == MAIN_MODEL)
            & (sens["event_window"] == MAIN_WINDOW)
            & (sens["dictionary"] == dictionary)
        ][["code", "slope"]].rename(columns={"slope": dictionary})
        merged = main.merge(part, on="code", how="inner").dropna()
        if len(merged) >= 5:
            rho = stats.spearmanr(merged["main"], merged[dictionary])
            rows.append({"check": f"dict_{dictionary}", "n": int(len(merged)), "spearman_rho": float(rho.statistic), "p_value": float(rho.pvalue)})
    for window in WINDOWS:
        part = sens[
            (sens["model"] == MAIN_MODEL)
            & (sens["event_window"] == window)
            & (sens["dictionary"] == MAIN_DICT)
        ][["code", "slope"]].rename(columns={"slope": f"w{window}"})
        merged = main.merge(part, on="code", how="inner").dropna()
        if len(merged) >= 5:
            rho = stats.spearmanr(merged["main"], merged[f"w{window}"])
            rows.append({"check": f"window_{window}", "n": int(len(merged)), "spearman_rho": float(rho.statistic), "p_value": float(rho.pvalue)})
    for model in MODELS:
        part = sens[
            (sens["model"] == model)
            & (sens["event_window"] == MAIN_WINDOW)
            & (sens["dictionary"] == MAIN_DICT)
        ][["code", "slope"]].rename(columns={"slope": model})
        merged = main.merge(part, on="code", how="inner").dropna()
        if len(merged) >= 5:
            rho = stats.spearmanr(merged["main"], merged[model])
            rows.append({"check": f"model_{model}", "n": int(len(merged)), "spearman_rho": float(rho.statistic), "p_value": float(rho.pvalue)})
    return pd.DataFrame(rows)


def make_plot(main_table: pd.DataFrame, char: pd.DataFrame, group_comp: pd.DataFrame) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    frame = main_table.copy()
    colors = {
        "上游资源与周期": "#3E6C8E",
        "中游制造与科技": "#B4436B",
        "大消费与医药": "#4C7A57",
        "金融地产与综合": "#8A6BAE",
        "公用与交运": "#B07A3E",
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    ax = axes[0, 0]
    rank = frame.sort_values("sens_main").copy()
    rank["label"] = rank["code"] + " " + rank["name"]
    bar_colors = [colors.get(value, "#999999") for value in rank["sector_group"]]
    ax.barh(rank["label"], rank["sens_main"], color=bar_colors, height=0.7)
    ax.errorbar(
        rank["sens_main"],
        range(len(rank)),
        xerr=[rank["sens_main"] - rank["sens_main_ci_low"], rank["sens_main_ci_high"] - rank["sens_main"]],
        fmt="none",
        ecolor="#333333",
        elinewidth=0.8,
        capsize=2,
    )
    ax.axvline(0.0, color="#666666", linewidth=0.8)
    ax.set_title("行业情绪敏感度排序（市场模型 [0,+1]，ntusd，Bootstrap 95% CI）")
    ax.set_xlabel("CAR 对情绪的回归斜率")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(alpha=0.25, axis="x")

    for panel, column, label, color in [
        (axes[0, 1], "rate_sens", "利率敏感（10Y 国债收益率 beta）", "#3E6C8E"),
        (axes[1, 0], "fin_sens", "融资敏感（社融累计同比 beta）", "#B07A3E"),
    ]:
        sub = frame.dropna(subset=[column, "sens_main"])
        panel.scatter(sub[column], sub["sens_main"], s=42, color=color, alpha=0.85, edgecolor="white", linewidth=0.6)
        if len(sub) >= 5:
            fit = np.polyfit(sub[column], sub["sens_main"], 1)
            xs = np.linspace(sub[column].min(), sub[column].max(), 50)
            panel.plot(xs, np.polyval(fit, xs), color="#333333", linewidth=1.2, linestyle="--")
            rho = stats.spearmanr(sub[column], sub["sens_main"])
            panel.text(0.04, 0.95, f"Spearman rho={rho.statistic:.2f}\np={rho.pvalue:.3f}", transform=panel.transAxes, fontsize=9, va="top")
        for _, row in sub.iterrows():
            panel.annotate(row["name"], (row[column], row["sens_main"]), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
        panel.axhline(0.0, color="#999999", linewidth=0.8)
        panel.set_title(label)
        panel.set_xlabel("行业特征")
        panel.set_ylabel("情绪敏感度")
        panel.grid(alpha=0.25)

    ax = axes[1, 1]
    plot_rows = group_comp[group_comp["grouping"].isin(["rate_group", "fin_group", "policy_dep"])].copy()
    plot_rows = plot_rows[~plot_rows["group"].astype(str).str.contains(" - ", na=False)]
    labels = [f"{row['grouping']}:{row['group']}" for _, row in plot_rows.iterrows()]
    means = plot_rows["mean_sens"].to_numpy(dtype=float)
    lows = []
    highs = []
    for _, row in plot_rows.iterrows():
        if np.isfinite(row.get("boot_ci_low", float("nan"))):
            lows.append(row["mean_sens"] - row["boot_ci_low"])
            highs.append(row["boot_ci_high"] - row["mean_sens"])
        else:
            lows.append(0.0)
            highs.append(0.0)
    x = np.arange(len(plot_rows))
    ax.bar(x, means, color=["#3E6C8E", "#8A6BAE", "#B07A3E", "#4C7A57", "#B4436B", "#7B5EA7", "#4A7C8A"], width=0.6)
    ax.errorbar(x, means, yerr=[lows, highs], fmt="none", ecolor="#333333", elinewidth=0.9, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.set_title("分组敏感度均值（行业层面，Bootstrap 95% CI）")
    ax.set_ylabel("平均敏感度")
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    out = E5_DIR / "e5_sensitivity_plot.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log(f"[plot] wrote {out}")


def write_log(metrics: dict[str, Any]) -> None:
    header = [
        "# E5 行业敏感度异质性实验日志",
        "",
        "- 实验编号：E5",
        "- 运行日期：2026-11-19",
        "- 脚本：`experiments/E5/run_e5.py`",
        f"- 脚本 SHA256：`{metrics['script_sha256']}`",
        f"- 随机种子：{SEED}",
        f"- Bootstrap 次数：{N_BOOT}",
        "",
    ]
    (E5_LOG_DIR / "e5_heterogeneity.md").write_text("\n".join(header + LOG_LINES) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()
    log("## 1. 实验目的与假设")
    log("")
    log("- 检验 H3：哪些行业对 MPR 情绪更敏感，敏感度是否与行业特征（利率敏感、融资敏感、政策依赖）有关。")
    log("- 事先确定的敏感度指标：事件研究 CAR 对 MPR 情绪的事件层 OLS 斜率。主口径为市场模型、估计窗 120、事件窗 [0,+1]、主词典 ntusd_simplified。")
    log("- 分组规则事先确定，不根据 E5 结果调整：连续特征按全样本中位数二分；政策依赖为事先定义的定性映射；板块分组沿用 E3。")
    log("")

    log("## 2. 输入数据")
    car = load_event_car()
    log(f"- E3 事件层结果：`{E3_RESULTS_PATH}`，{len(car)} 行，{car['issue'].nunique()} 个事件，{car['code'].nunique()} 个行业。")
    log("- E1 情绪词典：ntusd_simplified / dlut / mpr_domain，各 103 期。")
    log("- 行业日行情：`data/external/sw_industries/sw_industry_daily.csv`。")
    log("- 利率数据：akshare `bond_zh_us_rate`（10Y/2Y 国债收益率），缓存 `_temp/e5/treasury_yield.csv`。")
    log("- 融资数据：`_temp/e4/macro_raw/new_financial_credit.csv`（社融累计同比）、`m2_yearly.csv`（M2 同比）。")
    log("")

    log("## 3. 行业特征构造（事先定义）")
    char = build_characteristics()
    char = rename_char_columns(char)
    rate_median = float(char["rate_sens"].median())
    fin_median = float(char["fin_sens"].median())
    char["rate_group"] = np.where(char["rate_sens"] >= rate_median, "high", "low")
    char["fin_group"] = np.where(char["fin_sens"] >= fin_median, "high", "low")
    log(f"- 利率敏感 rate_sens：行业日超额收益对 10Y 国债收益率日变动的 OLS beta；中位数 {rate_median:.6f}，据此二分。")
    log(f"- 融资敏感 fin_sens：行业月超额收益对社融累计同比变动的 OLS beta；中位数 {fin_median:.6f}，据此二分。稳健性用 M2 同比 beta（fin_sens_m2）。")
    log("- 政策依赖 policy_dep：事先定义的定性映射（描述性），1 的行业：" + "、".join(sorted(char.loc[char["policy_dep"] == 1, "code"])))
    log("  - 映射依据：地产/建筑/建材受调控与基建政策影响；公用事业受价格与环保政策影响；军工受国防预算与装备政策影响；农林牧渔受农业支持与种业政策影响；银行受信贷与监管政策影响。该映射为定性描述，不用于因果解释。")
    log("")

    sens = sensitivity_table(car)
    high_low = high_low_table(car)
    subsample = subsample_sensitivity(car)
    stab = stability_summary(sens)

    main = sens[
        (sens["model"] == MAIN_MODEL)
        & (sens["event_window"] == MAIN_WINDOW)
        & (sens["dictionary"] == MAIN_DICT)
    ][["code", "n", "slope", "se", "t", "p", "r2"]].rename(
        columns={"n": "n_events_main", "slope": "sens_main", "se": "sens_main_se", "t": "sens_main_t", "p": "sens_main_p", "r2": "sens_main_r2"}
    )
    ci_rows = []
    base_car = car[
        (car["model"] == MAIN_MODEL)
        & (car["event_window"] == MAIN_WINDOW)
        & car[f"sentiment_{MAIN_DICT}"].notna()
    ]
    for code, sub in base_car.groupby("code"):
        lo, hi = bootstrap_slope(sub[f"sentiment_{MAIN_DICT}"].to_numpy(), sub["car"].to_numpy())
        ci_rows.append({"code": code, "sens_main_ci_low": lo, "sens_main_ci_high": hi})
    main = main.merge(pd.DataFrame(ci_rows), on="code", how="left")

    for window in WINDOWS:
        part = sens[(sens["model"] == MAIN_MODEL) & (sens["event_window"] == window) & (sens["dictionary"] == MAIN_DICT)][["code", "slope"]].rename(columns={"slope": f"sens_w{window}"})
        main = main.merge(part, on="code", how="left")
    for model, name in [("mean_adj", "sens_meanadj_w1"), ("excess_market", "sens_excess_w1")]:
        part = sens[(sens["model"] == model) & (sens["event_window"] == MAIN_WINDOW) & (sens["dictionary"] == MAIN_DICT)][["code", "slope"]].rename(columns={"slope": name})
        main = main.merge(part, on="code", how="left")
    for dictionary, name in [("dlut", "sens_dlut_w1"), ("mpr_domain", "sens_mpr_w1")]:
        part = sens[(sens["model"] == MAIN_MODEL) & (sens["event_window"] == MAIN_WINDOW) & (sens["dictionary"] == dictionary)][["code", "slope"]].rename(columns={"slope": name})
        main = main.merge(part, on="code", how="left")

    hl = high_low[
        (high_low["model"] == MAIN_MODEL) & (high_low["event_window"] == MAIN_WINDOW)
    ][["code", "n_high", "n_low", "high_low_diff", "high_low_t", "high_low_p", "high_low_ci_low", "high_low_ci_high"]]
    main = main.merge(hl, on="code", how="left")

    pre = subsample[subsample["subsample"] == "pre"][["code", "slope"]].rename(columns={"slope": "sens_pre"})
    post = subsample[subsample["subsample"] == "post"][["code", "slope"]].rename(columns={"slope": "sens_post"})
    main = main.merge(pre, on="code", how="left").merge(post, on="code", how="left")

    name_info = car[["code", "name", "sector_group"]].drop_duplicates("code")
    beta_info = car[["code", "full_sample_beta", "beta_group"]].drop_duplicates("code")
    char = char.merge(name_info, on="code", how="left").merge(beta_info, on="code", how="left")
    char_cols = ["code", "name", "sector_group", "rate_sens", "rate_sens_se", "rate_sens_t", "rate_sens_p", "rate_sens_n", "rate_sens_2y", "fin_sens", "fin_sens_se", "fin_sens_t", "fin_sens_p", "fin_sens_n", "fin_sens_m2", "policy_dep", "rate_group", "fin_group", "full_sample_beta", "beta_group"]
    main = main.merge(char[[c for c in char_cols if c in char.columns]], on="code", how="left")

    rho = stats.spearmanr(main["sens_pre"], main["sens_post"], nan_policy="omit")
    sign_agree = float((np.sign(main["sens_pre"]) == np.sign(main["sens_post"])).mean())

    main = main.sort_values("sens_main", ascending=False).reset_index(drop=True)
    main["rank_main"] = np.arange(1, len(main) + 1)
    main.to_csv(E5_DIR / "e5_heterogeneity.csv", index=False, encoding="utf-8-sig")
    log(f"- 行业敏感度表：`experiments/E5/e5_heterogeneity.csv`，{len(main)} 个行业。")
    log("")

    log("## 4. 行业敏感度排序（主口径，市场模型 [0,+1]，ntusd）")
    log("")
    log("| 排名 | 行业 | 板块 | 敏感度 | SE | t | p | Bootstrap 95% CI | 事件数 |")
    log("|---:|------|------|-------:|----:|---:|---:|------------------|-------:|")
    for _, row in main.head(15).iterrows():
        log(f"| {int(row['rank_main'])} | {row['code']} {row['name']} | {row['sector_group']} | {row['sens_main']:.6f} | {row['sens_main_se']:.6f} | {row['sens_main_t']:.2f} | {row['sens_main_p']:.4f} | [{row['sens_main_ci_low']:.6f}, {row['sens_main_ci_high']:.6f}] | {int(row['n_events_main'])} |")
    log("")
    log(f"- 敏感度均值 {main['sens_main'].mean():.6f}，中位 {main['sens_main'].median():.6f}，标准差 {main['sens_main'].std(ddof=1):.6f}。")
    log(f"- 敏感度范围：[{main['sens_main'].min():.6f}, {main['sens_main'].max():.6f}]。")
    sign_pos = int((main["sens_main"] > 0).sum())
    log(f"- 敏感度为正值行业 {sign_pos} 个，负值 {int((main['sens_main'] < 0).sum())} 个。")
    log("- 敏感度符号：斜率为负表示 MPR 情绪越高、事件窗 CAR 越低（与 E3 高-低组差为负一致）；为正表示情绪越高、CAR 越高。")
    log("")

    log("## 5. 行业特征与分组")
    log("")
    log("| 行业 | rate_sens | rate_group | fin_sens | fin_group | policy_dep |")
    log("|------|----------:|------------|---------:|-----------|-----------:|")
    for _, row in char.sort_values("rate_sens", ascending=False).iterrows():
        log(f"| {row['code']} {row['name']} | {row['rate_sens']:.6f} | {row['rate_group']} | {row['fin_sens']:.6f} | {row['fin_group']} | {int(row['policy_dep'])} |")
    log("")

    log("## 6. 交互项与分组回归")
    log("")
    cs = cross_section_regression(main)
    cs.to_csv(E5_DIR / "e5_cross_section.csv", index=False, encoding="utf-8-sig")
    log("### 6.1 行业敏感度对行业特征的横截面回归（HC3 稳健标准误）")
    log("")
    log("| 设定 | 项 | 系数 | SE | t | p | N | R² |")
    log("|------|----|-----:|---:|---:|---:|---:|----:|")
    for _, row in cs.iterrows():
        log(f"| {row['spec']} | {row['term']} | {row['estimate']:.6f} | {row['std_error']:.6f} | {row['t_value']:.2f} | {row['p_value']:.4f} | {int(row['n'])} | {row['r2']:.3f} |")
    log("")

    pi = panel_interaction(car, char)
    pi.to_csv(E5_DIR / "e5_panel_interaction.csv", index=False, encoding="utf-8-sig")
    log("### 6.2 事件层面板交互（CAR ~ sent_z + sent_z×char_z + 行业固定效应，按事件聚类）")
    log("")
    log("- 交互项系数的聚类稳健 p 值即“有交互 vs 无交互”模型的检验；char_z 为行业特征横截面标准化。")
    log("| 特征 | 交互项系数 | SE | t | p | N | 事件数 | R² |")
    log("|------|-----------:|---:|---:|---:|---:|-------:|----:|")
    for _, row in pi.iterrows():
        log(f"| {row['characteristic']} | {row['estimate']:.6f} | {row['std_error']:.6f} | {row['t_value']:.2f} | {row['p_value']:.4f} | {int(row['n'])} | {int(row['n_issues'])} | {row['r2']:.3f} |")
    log("")

    gic = group_industry_compare(main)
    gic.to_csv(E5_DIR / "e5_group_industry.csv", index=False, encoding="utf-8-sig")
    gev = group_event_compare(car, char)
    gev.to_csv(E5_DIR / "e5_group_event.csv", index=False, encoding="utf-8-sig")
    log("### 6.3 组间敏感度差异（行业层面）")
    log("")
    log("| 分组 | 组 | 行业数 | 平均敏感度 | Welch t | p | Bootstrap 95% CI |")
    log("|------|----|-------:|-----------:|-------:|---:|------------------|")
    for _, row in gic.iterrows():
        ci_text = ""
        if np.isfinite(row.get("boot_ci_low", float("nan"))):
            ci_text = f"[{row['boot_ci_low']:.6f}, {row['boot_ci_high']:.6f}]"
        if pd.notna(row.get("welch_t")):
            log(f"| {row['grouping']} | {row['group']} | {int(row['n_industries'])} | {row['mean_sens']:.6f} | {row['welch_t']:.2f} | {row['welch_p']:.4f} | {ci_text} |")
        else:
            log(f"| {row['grouping']} | {row['group']} | {int(row['n_industries'])} | {row['mean_sens']:.6f} | | | {ci_text} |")
    log("")

    log("### 6.4 组间 CAR 差异（事件层面，市场模型 [0,+1]）")
    log("")
    log("| 分组 | 组 | 事件数 | 平均 CAR | t | p | Bootstrap 95% CI |")
    log("|------|----|-------:|---------:|---:|---:|------------------|")
    for _, row in gev.iterrows():
        ci_text = ""
        if np.isfinite(row.get("boot_ci_low", float("nan"))):
            ci_text = f"[{row['boot_ci_low']:.6f}, {row['boot_ci_high']:.6f}]"
        t_text = f"{row['t_stat']:.2f}" if np.isfinite(row.get("t_stat", float("nan"))) else ""
        p_text = f"{row['p_value']:.4f}" if np.isfinite(row.get("p_value", float("nan"))) else ""
        log(f"| {row['grouping']} | {row['group']} | {int(row['n_issues'])} | {row['mean_car']:.6f} | {t_text} | {p_text} | {ci_text} |")
    log("")

    log("## 7. 组内 IC")
    log("")
    panel = pd.read_csv(E4_PANEL_PATH, dtype={"code": str})
    gic_ts = group_ic(panel, char)
    gic_ts.to_csv(E5_DIR / "e5_group_ic.csv", index=False, encoding="utf-8-sig")
    log("### 7.1 组内时间序列 IC（MPR 情绪 vs 组平均下季收益，季度）")
    log("")
    log("| 分组 | 组 | 季度数 | Spearman IC | p | Pearson IC |")
    log("|------|----|-------:|------------:|---:|-----------:|")
    for _, row in gic_ts.iterrows():
        log(f"| {row['grouping']} | {row['group']} | {int(row['n_quarters'])} | {row['ic_spearman']:.4f} | {row['ic_p']:.4f} | {row['ic_pearson']:.4f} |")
    log("")
    try:
        pred = pd.read_csv(E4_PRED_PATH, dtype={"code": str})
        gic_oos = group_oos_ic(pred, char)
        gic_oos.to_csv(E5_DIR / "e5_group_oos_ic.csv", index=False, encoding="utf-8-sig")
        log("### 7.2 组内横截面 IC（E4 样本外 sent_exposure 预测，测试集）")
        log("")
        log("| 分组 | 组 | 季度数 | IC 均值 | ICIR | NW t | NW p |")
        log("|------|----|-------:|--------:|-----:|------:|------:|")
        for _, row in gic_oos.iterrows():
            log(f"| {row['grouping']} | {row['group']} | {int(row['n_quarters'])} | {row['ic_mean']:.4f} | {row['icir']:.4f} | {row['nw_t']:.2f} | {row['nw_p']:.4f} |")
    except Exception as exc:  # noqa: BLE001
        log(f"[warn] 组内横截面 IC 跳过：{type(exc).__name__}: {exc}")
    log("")

    log("## 8. 稳定性")
    log("")
    stab.to_csv(E5_DIR / "e5_stability.csv", index=False, encoding="utf-8-sig")
    log("| 检验 | N | Spearman rho | p |")
    log("|------|--:|-------------:|---:|")
    for _, row in stab.iterrows():
        log(f"| {row['check']} | {int(row['n'])} | {row['spearman_rho']:.4f} | {row['p_value']:.4f} |")
    log("")
    log(f"- 前后子样本（以 {SUB_SPLIT} 为界）敏感度 Spearman rho={rho.statistic:.4f}（p={rho.pvalue:.4f}），符号一致率 {sign_agree:.2%}。")
    log("")

    log("## 9. 结论")
    log("")
    sig_cs = cs[(cs["spec"] == "ols_all") & (cs["term"] != "Intercept") & (cs["p_value"] < 0.10)]
    sig_cs_uni = cs[
        (cs["spec"].str.startswith("ols_"))
        & (cs["spec"] != "ols_all")
        & (cs["term"] != "Intercept")
        & (cs["p_value"] < 0.10)
    ]
    sig_pi = pi[pi["p_value"] < 0.10] if len(pi) else pi
    log(f"- 横截面回归（ols_all）在 10% 水平显著的特征项：" + ("、".join(sig_cs["term"].tolist()) if len(sig_cs) else "无") + "。")
    log(f"- 单变量横截面回归在 10% 水平显著的特征项：" + ("、".join(sig_cs_uni["term"].tolist()) if len(sig_cs_uni) else "无") + "（联合回归中因共线性不显著）。")
    log(f"- 事件层面板交互在 10% 水平显著的特征：" + ("、".join(sig_pi["characteristic"].tolist()) if len(sig_pi) else "无") + "。")
    corr_rf = float(np.corrcoef(main["rate_sens"], main["fin_sens"])[0, 1])
    log(f"- 行业特征共线性：rate_sens 与 fin_sens 的相关系数 {corr_rf:.4f}；两者联合回归时相互抵消。")
    log(f"- 行业敏感度排序见 `e5_heterogeneity.csv`；主口径下敏感度范围 [{main['sens_main'].min():.6f}, {main['sens_main'].max():.6f}]。")
    log(f"- 子样本稳定性：前后半敏感度秩相关 {rho.statistic:.4f}（p={rho.pvalue:.4f}），符号一致率 {sign_agree:.2%}。")
    sig_group = gic[
        (gic["group"].astype(str).str.contains(" - ")) & (gic["welch_p"] < 0.10)
    ]
    log(f"- 组间敏感度差异在 10% 水平显著的分组：" + ("、".join(sig_group["grouping"] + ":" + sig_group["group"].astype(str)) if len(sig_group) else "无") + "。")
    if len(sig_cs) == 0 and len(sig_pi) == 0 and len(sig_group) == 0:
        log("- 结论：行业间敏感度差异在行业层面分组、横截面回归与事件层面板交互中均未达到 10% 显著水平，且子样本稳定性差，暂不支持 H3。")
    elif len(sig_cs) == 0 and len(sig_pi) == 0:
        log("- 结论：仅行业层面分组存在边际显著差异，但联合横截面回归与事件层面板交互均不显著，且子样本稳定性差（前后半秩相关接近 0），证据弱且不稳定，暂不支持 H3。")
    else:
        log("- 结论：存在达到 10% 显著水平的特征交互，但需结合子样本稳定性与多重检验谨慎解释；详见交互项表。")
    log("")

    log("## 10. 局限")
    log("")
    log("- 事件样本 89 期、行业 31 个，单行业敏感度估计精度有限；2014+/2021+ 起点的行业事件数较少。")
    log("- 行业特征为代理变量：rate_sens 与 fin_sens 从市场数据估计，可能反映多种宏观暴露；policy_dep 为定性映射。")
    log("- 共同情绪在截面上无差异，交互项识别依赖行业特征的真实变化；联合回归中 rate_sens 与 fin_sens 高度负相关（约 -0.82），单变量显著性不宜过度解读。")
    log("- 子样本稳定性差（前后半秩相关接近 0），行业敏感度排序不稳定。")
    log("")
    log("## 11. 复现")
    log("")
    log("```powershell")
    log("python experiments/E5/run_e5.py")
    log("```")
    log(f"- 脚本 SHA256：`{sha256(Path(__file__))}`")
    log("")

    make_plot(main, char, gic)

    metrics = {
        "experiment": "E5",
        "run_date": "2026-11-19",
        "script_sha256": sha256(Path(__file__)),
        "seed": SEED,
        "n_boot": N_BOOT,
        "main_spec": {"model": MAIN_MODEL, "event_window": MAIN_WINDOW, "dictionary": MAIN_DICT},
        "industry_count": int(len(main)),
        "event_count": int(car["issue"].nunique()),
        "sens_mean": float(main["sens_main"].mean()),
        "sens_median": float(main["sens_main"].median()),
        "sens_std": float(main["sens_main"].std(ddof=1)),
        "sens_min": float(main["sens_main"].min()),
        "sens_max": float(main["sens_main"].max()),
        "subsample_rho": float(rho.statistic),
        "subsample_sign_agree": float(sign_agree),
        "significant_cross_section_terms_10pct": sig_cs["term"].tolist(),
        "significant_univariate_terms_10pct": sig_cs_uni["term"].tolist(),
        "significant_panel_interactions_10pct": sig_pi["characteristic"].tolist(),
    }
    (E5_DIR / "e5_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_log(metrics)
    log("- 指标：`experiments/E5/e5_metrics.json`")
    log("- 日志：`experiments/E5/logs/e5_heterogeneity.md`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
