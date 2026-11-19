# -*- coding: utf-8 -*-
"""E2 event-study baseline for the MPR sentiment project."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
E2_DIR = ROOT / "experiments" / "E2"
LOG_DIR = E2_DIR / "logs"
TEMP_DIR = ROOT / "_temp" / "e2"

INDUSTRY_PATH = ROOT / "data" / "external" / "sw_industries" / "sw_industry_daily.csv"
BENCH_PATH = ROOT / "data" / "external" / "benchmark" / "benchmark_daily.csv"
BENCH_CACHE_PATH = TEMP_DIR.parent / "e0_market_raw" / "benchmark_sh000300.csv"
MPR_INDEX_PATH = ROOT / "experiments" / "E1" / "mpr_index_full.csv"
SENTIMENT_PATH = ROOT / "experiments" / "E1" / "sentiment_dictionary_full.csv"

BENCH_CODE = "sh000300"
MAIN_EST = 120
EST_WINDOWS = [90, 120, 180, 250]
EVENT_WINDOWS = [1, 3, 5, 10]
DICTS = ["ntusd_simplified", "dlut", "mpr_domain"]
MAIN_DICT = "ntusd_simplified"
SEED = 20261119
N_BOOT = 10_000

RNG = np.random.default_rng(SEED)

POLLUTION_EVENTS: list[dict[str, str]] = [
    {"name": "9/11 恐怖袭击与全球市场动荡", "start": "2001-09-11", "end": "2001-09-21", "source": "公开历史资料"},
    {"name": "国务院暂停国有股减持", "start": "2002-06-24", "end": "2002-06-28", "source": "公开历史资料"},
    {"name": "伊拉克战争与 SARS 疫情", "start": "2003-03-20", "end": "2003-04-30", "source": "公开历史资料"},
    {"name": "宏观调控与 A 股持续下跌", "start": "2004-04-27", "end": "2004-05-14", "source": "公开历史资料"},
    {"name": "股权分置改革启动", "start": "2005-04-29", "end": "2005-06-06", "source": "公开历史资料"},
    {"name": "央行连续上调准备金率、加息预期", "start": "2006-06-01", "end": "2006-07-14", "source": "公开历史资料"},
    {"name": "证券交易印花税上调（530 大跌）", "start": "2007-05-30", "end": "2007-06-05", "source": "公开历史资料"},
    {"name": "美国次贷危机发酵", "start": "2007-08-01", "end": "2007-10-31", "source": "公开历史资料"},
    {"name": "全球股市大跌", "start": "2008-01-22", "end": "2008-02-05", "source": "公开历史资料"},
    {"name": "雷曼兄弟破产与全球金融危机", "start": "2008-09-15", "end": "2008-10-31", "source": "公开历史资料"},
    {"name": "四万亿刺激计划宣布", "start": "2008-11-05", "end": "2008-11-14", "source": "公开报道"},
    {"name": "加息周期与通胀治理", "start": "2010-10-19", "end": "2010-12-27", "source": "公开历史资料"},
    {"name": "美债评级下调与欧债危机", "start": "2011-08-05", "end": "2011-11-30", "source": "公开历史资料"},
    {"name": "钱荒", "start": "2013-06-19", "end": "2013-07-05", "source": "公开历史资料"},
    {"name": "A 股股灾", "start": "2015-06-12", "end": "2015-07-09", "source": "公开历史资料"},
    {"name": "811 汇改与全球市场动荡", "start": "2015-08-11", "end": "2015-08-26", "source": "公开历史资料"},
    {"name": "熔断机制触发", "start": "2016-01-04", "end": "2016-01-07", "source": "公开历史资料"},
    {"name": "中美贸易摩擦升级", "start": "2018-03-22", "end": "2018-12-31", "source": "公开历史资料"},
    {"name": "中美谈判破裂、加税", "start": "2019-05-05", "end": "2019-06-10", "source": "公开历史资料"},
    {"name": "新冠疫情冲击", "start": "2020-01-20", "end": "2020-03-31", "source": "公开历史资料"},
    {"name": "核心资产抱团瓦解、美债利率上行", "start": "2021-02-18", "end": "2021-03-15", "source": "公开历史资料"},
    {"name": "教培双减政策", "start": "2021-07-23", "end": "2021-08-06", "source": "公开历史资料"},
    {"name": "上海封控与疫情冲击", "start": "2022-03-01", "end": "2022-04-26", "source": "公开历史资料"},
    {"name": "防疫优化与地产三支箭", "start": "2022-11-11", "end": "2022-12-31", "source": "公开历史资料"},
    {"name": "小微盘流动性危机", "start": "2024-01-22", "end": "2024-02-05", "source": "公开历史资料"},
    {"name": "924 一揽子政策与急涨", "start": "2024-09-24", "end": "2024-10-31", "source": "Bing 核验（2024-09-24 国新办发布会）"},
    {"name": "美国对等关税冲击", "start": "2025-04-03", "end": "2025-04-11", "source": "公开报道"},
    {"name": "中美日内瓦经贸会谈联合声明", "start": "2025-05-12", "end": "2025-05-15", "source": "公开报道"},
    {"name": "反内卷/统一大市场政策", "start": "2025-07-01", "end": "2025-07-07", "source": "公开报道"},
    {"name": "深圳 APEC 峰会", "start": "2026-11-18", "end": "2026-11-19", "source": "Bing 核验"},
]


def ensure_dirs() -> None:
    for path in [E2_DIR, LOG_DIR, TEMP_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def log(line: str = "") -> None:
    print(line)
    with (TEMP_DIR / "e2_run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prices() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[dict[str, Any]]]:
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
    industry["date"] = pd.to_datetime(industry["date"]).dt.strftime("%Y-%m-%d")
    industry = industry.sort_values(["code", "date"]).reset_index(drop=True)
    industry["daily_return"] = industry.groupby("code", group_keys=False)["close"].pct_change()

    # Prefer the E0-verified cache that starts 2002-01-04; the checked-in
    # benchmark_daily.csv starts 2005-01-04 and would drop 2002-2004 events.
    if BENCH_CACHE_PATH.exists() and BENCH_CACHE_PATH.stat().st_size > 0:
        bench = pd.read_csv(BENCH_CACHE_PATH, dtype={"code": str})
    else:
        bench = pd.read_csv(BENCH_PATH, dtype={"code": str})
    bench["date"] = pd.to_datetime(bench["date"]).dt.strftime("%Y-%m-%d")
    bench = bench.sort_values(["code", "date"]).reset_index(drop=True)
    bench["daily_return"] = bench.groupby("code", group_keys=False)["close"].pct_change()

    calendar = sorted(bench.loc[bench["code"] == BENCH_CODE, "date"].unique().tolist())

    asset_rows: list[dict[str, Any]] = []
    for code, name in industry[["code", "name"]].drop_duplicates().itertuples(index=False):
        frame = industry[industry["code"] == code]
        asset_rows.append(
            {
                "code": code,
                "name": name,
                "rows": len(frame),
                "start": frame["date"].min(),
                "end": frame["date"].max(),
            }
        )
    bench_frame = bench[bench["code"] == BENCH_CODE]
    asset_rows.append(
        {
            "code": BENCH_CODE,
            "name": "沪深300",
            "rows": len(bench_frame),
            "start": bench_frame["date"].min(),
            "end": bench_frame["date"].max(),
        }
    )
    return industry, bench, calendar, asset_rows


def fetch_rf() -> pd.DataFrame | None:
    cache = TEMP_DIR / "rf_rate.csv"
    if cache.exists() and cache.stat().st_size > 0:
        return pd.read_csv(cache)
    try:
        import akshare as ak

        raw = ak.bond_zh_us_rate(start_date="20010101")
        raw.columns = [str(column) for column in raw.columns]
        date_col = raw.columns[0]
        cn2y = [column for column in raw.columns if "中国" in column and "2年" in column]
        if not cn2y:
            raise RuntimeError("China 2Y yield column not found")
        frame = raw[[date_col, cn2y[0]]].copy()
        frame.columns = ["date", "cn_2y_yield"]
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        frame = frame.dropna(subset=["cn_2y_yield"]).sort_values("date").reset_index(drop=True)
        frame["rf_daily"] = frame["cn_2y_yield"] / 100.0 / 252.0
        frame.to_csv(cache, index=False, encoding="utf-8-sig")
        return frame
    except Exception as exc:  # noqa: BLE001
        log(f"[warn] risk-free rate fetch failed: {type(exc).__name__}: {exc}")
        return None


def load_events() -> pd.DataFrame:
    mpr = pd.read_csv(MPR_INDEX_PATH, dtype={"issue": str})
    mpr["publication_date"] = pd.to_datetime(mpr["publication_date"]).dt.strftime("%Y-%m-%d")
    sent = pd.read_csv(SENTIMENT_PATH, dtype={"issue": str})
    sent = sent[["issue", "dictionary", "sentiment_score", "weighted_score"]]
    wide = sent.pivot(index="issue", columns="dictionary", values="sentiment_score")
    wide.columns = [f"sentiment_{column}" for column in wide.columns]
    wide = wide.reset_index()
    return mpr.merge(wide, on="issue", how="left")


def align_events(events: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    cal_index = {date: idx for idx, date in enumerate(calendar)}
    rows: list[dict[str, Any]] = []
    for _, row in events.iterrows():
        pub = row["publication_date"]
        if pub in cal_index:
            event_trading_date = pub
            alignment = "publication_date_is_trading_day"
            date_match = True
        elif pub < calendar[0]:
            # The benchmark history starts after this publication date; mapping
            # it to the first benchmark trading day would fabricate an event
            # date, so the issue is dropped instead.
            event_trading_date = None
            alignment = "event_date_before_benchmark_start"
            date_match = False
        else:
            future = [date for date in calendar if date >= pub]
            if not future:
                event_trading_date = None
                alignment = "no_trading_day_after_publication"
                date_match = False
            else:
                event_trading_date = future[0]
                alignment = "mapped_to_next_trading_day"
                date_match = False
        rows.append(
            {
                "issue": row["issue"],
                "report_period": row["report_period"],
                "publication_date": pub,
                "event_date": pub,
                "event_trading_date": event_trading_date,
                "alignment_status": alignment,
                "date_match": date_match,
                "sentiment_ntusd_simplified": row.get("sentiment_ntusd_simplified"),
                "sentiment_dlut": row.get("sentiment_dlut"),
                "sentiment_mpr_domain": row.get("sentiment_mpr_domain"),
            }
        )
    aligned = pd.DataFrame(rows)
    aligned["event_idx"] = aligned["event_trading_date"].map(cal_index)
    return aligned


def assign_groups(events: pd.DataFrame, dictionary: str) -> pd.DataFrame:
    column = f"sentiment_{dictionary}"
    frame = events.copy()
    valid = frame[column].notna()
    ordered = frame.loc[valid, ["issue", column]].sort_values(column)
    n = len(ordered)
    labels: dict[str, str] = {}
    for idx, (_, row) in enumerate(ordered.iterrows()):
        labels[row["issue"]] = (
            "low" if idx < n / 3 else ("mid" if idx < 2 * n / 3 else "high")
        )
    frame["sentiment_group"] = frame["issue"].map(labels)
    return frame


def pollution_flags(events: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    pollution = pd.DataFrame(POLLUTION_EVENTS)
    pollution["start"] = pd.to_datetime(pollution["start"])
    pollution["end"] = pd.to_datetime(pollution["end"])
    pollution.to_csv(TEMP_DIR / "pollution_events.csv", index=False, encoding="utf-8-sig")

    cal_dt = pd.to_datetime(pd.Series(calendar))

    def flag_row(row: pd.Series) -> tuple[bool, str]:
        if pd.isna(row["event_idx"]):
            return False, ""
        idx = int(row["event_idx"])
        end_idx = min(idx + max(EVENT_WINDOWS), len(calendar) - 1)
        window_start = cal_dt.iloc[idx]
        window_end = cal_dt.iloc[end_idx]
        hits = pollution[
            (pollution["start"] <= window_end) & (pollution["end"] >= window_start)
        ]
        return (len(hits) > 0, ";".join(hits["name"].tolist()))

    flags = events.apply(flag_row, axis=1)
    events = events.copy()
    events["polluted"] = [item[0] for item in flags]
    events["pollution_hits"] = [item[1] for item in flags]
    return events


def build_observations(
    industry: pd.DataFrame,
    bench: pd.DataFrame,
    calendar: list[str],
    events: pd.DataFrame,
    rf: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    bench_ret = (
        bench[bench["code"] == BENCH_CODE]
        .set_index("date")["daily_return"]
        .sort_index()
    )
    rf_daily = None
    if rf is not None and "rf_daily" in rf.columns:
        rf_daily = rf.set_index("date")["rf_daily"].sort_index()

    series: dict[str, pd.DataFrame] = {}
    for code, name in industry[["code", "name"]].drop_duplicates().itertuples(index=False):
        frame = industry[industry["code"] == code].copy()
        frame = frame.set_index("date").sort_index()
        frame["name"] = name
        series[code] = frame

    issues = events.sort_values("event_idx", na_position="last").reset_index(drop=True)
    prior_event_dates: dict[str, str | None] = {}
    previous_pub: str | None = None
    for _, row in issues.iterrows():
        prior_event_dates[row["issue"]] = previous_pub
        previous_pub = row["publication_date"]

    obs_rows: list[dict[str, Any]] = []
    skip_rows: list[dict[str, Any]] = []

    for _, issue_row in issues.iterrows():
        if pd.isna(issue_row["event_idx"]):
            skip_rows.append(
                {
                    "issue": issue_row["issue"],
                    "code": None,
                    "reason": "event_date_before_benchmark_start_or_no_calendar",
                }
            )
            continue
        t_idx = int(issue_row["event_idx"])
        prior_pub = prior_event_dates.get(issue_row["issue"])
        for code, frame in series.items():
            for est_len in EST_WINDOWS:
                est_start_idx = t_idx - est_len
                if est_start_idx < 0:
                    skip_rows.append(
                        {
                            "issue": issue_row["issue"],
                            "code": code,
                            "reason": f"estimation_window_{est_len}_exceeds_history",
                        }
                    )
                    continue
                est_dates = calendar[est_start_idx:t_idx]
                if not all(date in frame.index for date in est_dates):
                    skip_rows.append(
                        {
                            "issue": issue_row["issue"],
                            "code": code,
                            "reason": f"estimation_window_{est_len}_missing_industry_dates",
                        }
                    )
                    continue
                est_ret = frame.loc[est_dates, "daily_return"].astype(float).to_numpy()
                mkt_ret = bench_ret.reindex(est_dates).astype(float).to_numpy()
                if np.isnan(est_ret).any() or np.isnan(mkt_ret).any():
                    skip_rows.append(
                        {
                            "issue": issue_row["issue"],
                            "code": code,
                            "reason": f"estimation_window_{est_len}_nan_returns",
                        }
                    )
                    continue

                beta, alpha = np.polyfit(mkt_ret, est_ret, 1)
                contains_prior = bool(
                    prior_pub is not None and est_dates[0] <= prior_pub <= est_dates[-1]
                )

                for window in EVENT_WINDOWS:
                    end_idx = t_idx + window
                    if end_idx >= len(calendar):
                        skip_rows.append(
                            {
                                "issue": issue_row["issue"],
                                "code": code,
                                "reason": f"event_window_0_{window}_exceeds_calendar",
                            }
                        )
                        continue
                    event_dates = calendar[t_idx : end_idx + 1]
                    if not all(date in frame.index for date in event_dates):
                        skip_rows.append(
                            {
                                "issue": issue_row["issue"],
                                "code": code,
                                "reason": f"event_window_0_{window}_missing_industry_dates",
                            }
                        )
                        continue
                    ri = frame.loc[event_dates, "daily_return"].astype(float).to_numpy()
                    rm = bench_ret.reindex(event_dates).astype(float).to_numpy()
                    if np.isnan(ri).any() or np.isnan(rm).any():
                        skip_rows.append(
                            {
                                "issue": issue_row["issue"],
                                "code": code,
                                "reason": f"event_window_0_{window}_nan_returns",
                            }
                        )
                        continue

                    ar_market = ri - (alpha + beta * rm)
                    mu = float(np.mean(est_ret))
                    ar_mean = ri - mu
                    obs_rows.append(
                        {
                            "issue": issue_row["issue"],
                            "publication_date": issue_row["publication_date"],
                            "event_trading_date": issue_row["event_trading_date"],
                            "code": code,
                            "name": frame["name"].iloc[0],
                            "estimation_window": est_len,
                            "event_window": window,
                            "model": "market",
                            "alpha": float(alpha),
                            "beta": float(beta),
                            "mean_est_return": mu,
                            "car_market": float(np.sum(ar_market)),
                            "car_mean_adj": float(np.sum(ar_mean)),
                            "ar_market": ";".join(f"{value:.8f}" for value in ar_market),
                            "ar_mean_adj": ";".join(f"{value:.8f}" for value in ar_mean),
                            "est_window_contains_prior_event": contains_prior,
                            "polluted": bool(issue_row["polluted"]),
                            "pollution_hits": issue_row["pollution_hits"],
                            "sentiment_ntusd_simplified": issue_row["sentiment_ntusd_simplified"],
                            "sentiment_dlut": issue_row["sentiment_dlut"],
                            "sentiment_mpr_domain": issue_row["sentiment_mpr_domain"],
                            "sentiment_group": issue_row["sentiment_group"],
                        }
                    )

            if rf_daily is not None:
                est_dates = calendar[max(0, t_idx - MAIN_EST) : t_idx]
                if len(est_dates) == MAIN_EST and all(date in frame.index for date in est_dates):
                    rf_est = rf_daily.reindex(est_dates)
                    if rf_est.notna().all():
                        ri_ex = (
                            frame.loc[est_dates, "daily_return"].astype(float)
                            - rf_est.astype(float)
                        ).to_numpy()
                        rm_ex = (
                            bench_ret.reindex(est_dates).astype(float)
                            - rf_est.astype(float)
                        ).to_numpy()
                        beta_ex, alpha_ex = np.polyfit(rm_ex, ri_ex, 1)
                        for window in EVENT_WINDOWS:
                            end_idx = t_idx + window
                            if end_idx >= len(calendar):
                                continue
                            event_dates = calendar[t_idx : end_idx + 1]
                            if not all(date in frame.index for date in event_dates):
                                continue
                            rf_ev = rf_daily.reindex(event_dates)
                            if rf_ev.isna().any():
                                continue
                            ri = frame.loc[event_dates, "daily_return"].astype(float).to_numpy()
                            rm = bench_ret.reindex(event_dates).astype(float).to_numpy()
                            expected = alpha_ex + beta_ex * (
                                rm - rf_ev.astype(float).to_numpy()
                            )
                            ar = ri - rf_ev.astype(float).to_numpy() - expected
                            obs_rows.append(
                                {
                                    "issue": issue_row["issue"],
                                    "publication_date": issue_row["publication_date"],
                                    "event_trading_date": issue_row["event_trading_date"],
                                    "code": code,
                                    "name": frame["name"].iloc[0],
                                    "estimation_window": MAIN_EST,
                                    "event_window": window,
                                    "model": "excess_market",
                                    "alpha": float(alpha_ex),
                                    "beta": float(beta_ex),
                                    "mean_est_return": float(np.mean(ri_ex)),
                                    "car_market": float(np.sum(ar)),
                                    "car_mean_adj": np.nan,
                                    "ar_market": ";".join(f"{value:.8f}" for value in ar),
                                    "ar_mean_adj": "",
                                    "est_window_contains_prior_event": bool(
                                        prior_pub is not None
                                        and est_dates[0] <= prior_pub <= est_dates[-1]
                                    ),
                                    "polluted": bool(issue_row["polluted"]),
                                    "pollution_hits": issue_row["pollution_hits"],
                                    "sentiment_ntusd_simplified": issue_row["sentiment_ntusd_simplified"],
                                    "sentiment_dlut": issue_row["sentiment_dlut"],
                                    "sentiment_mpr_domain": issue_row["sentiment_mpr_domain"],
                                    "sentiment_group": issue_row["sentiment_group"],
                                }
                            )

    observations = pd.DataFrame(obs_rows)

    period_frames: list[pd.DataFrame] = []
    main_obs = observations[observations["model"] != "excess_market"]
    for model_column, model_name in [
        ("car_market", "market"),
        ("car_mean_adj", "mean_adj"),
    ]:
        if not len(main_obs):
            continue
        grouped = (
            main_obs.dropna(subset=[model_column])
            .groupby(["issue", "estimation_window", "event_window"], as_index=False)
            .agg(
                period_car=(model_column, "mean"),
                n_industries=(model_column, "size"),
                period_car_std=(model_column, "std"),
            )
        )
        grouped["model"] = model_name
        period_frames.append(grouped)
    ex_obs = observations[observations["model"] == "excess_market"]
    if len(ex_obs):
        grouped = (
            ex_obs.dropna(subset=["car_market"])
            .groupby(["issue", "estimation_window", "event_window"], as_index=False)
            .agg(
                period_car=("car_market", "mean"),
                n_industries=("car_market", "size"),
                period_car_std=("car_market", "std"),
            )
        )
        grouped["model"] = "excess_market"
        period_frames.append(grouped)
    period_car = (
        pd.concat(period_frames, ignore_index=True) if period_frames else pd.DataFrame()
    )
    return observations, period_car, skip_rows


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    n = len(values)
    sample_means = np.empty(n_boot, dtype=float)
    for start in range(0, n_boot, 1000):
        batch = min(1000, n_boot - start)
        idx = RNG.integers(0, n, size=(batch, n))
        sample_means[start : start + batch] = values[idx].mean(axis=1)
    return float(np.percentile(sample_means, 2.5)), float(np.percentile(sample_means, 97.5))


def summarize(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = int(len(values))
    if n == 0:
        return {
            "n_periods": 0,
            "mean_car": float("nan"),
            "median_car": float("nan"),
            "std_car": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "sign_pos": 0,
            "sign_neg": 0,
            "sign_p_value": float("nan"),
            "boot_ci_low": float("nan"),
            "boot_ci_high": float("nan"),
        }
    mean = float(np.mean(values))
    median = float(np.median(values))
    std = float(np.std(values, ddof=1)) if n > 1 else float("nan")
    if n > 1:
        t_stat, p_value = stats.ttest_1samp(values, 0.0)
        t_stat, p_value = float(t_stat), float(p_value)
    else:
        t_stat, p_value = float("nan"), float("nan")
    sign_pos = int(np.sum(values > 0))
    sign_neg = int(np.sum(values < 0))
    sign_p = (
        float(stats.binomtest(sign_pos, sign_pos + sign_neg, 0.5).pvalue)
        if (sign_pos + sign_neg) > 0
        else float("nan")
    )
    lo, hi = bootstrap_ci(values)
    return {
        "n_periods": n,
        "mean_car": mean,
        "median_car": median,
        "std_car": std,
        "t_stat": t_stat,
        "p_value": p_value,
        "sign_pos": sign_pos,
        "sign_neg": sign_neg,
        "sign_p_value": sign_p,
        "boot_ci_low": lo,
        "boot_ci_high": hi,
    }


def group_diff(values_a: np.ndarray, values_b: np.ndarray) -> dict[str, Any]:
    """Difference of two independent issue-level groups: Welch t + bootstrap.

    A sign test is not reported here because the high and low groups contain
    different issues and cannot be paired; the sign test is reported for the
    single-group CAR means in `summarize`.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    diff = float(a.mean() - b.mean()) if len(a) and len(b) else float("nan")
    if len(a) > 1 and len(b) > 1:
        t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
        t_stat, p_value = float(t_stat), float(p_value)
    else:
        t_stat, p_value = float("nan"), float("nan")
    n = int(len(a) + len(b))
    if len(a) and len(b):
        boot = np.array(
            [
                a[RNG.integers(0, len(a), len(a))].mean()
                - b[RNG.integers(0, len(b), len(b))].mean()
                for _ in range(N_BOOT)
            ]
        )
        lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    else:
        lo, hi = float("nan"), float("nan")
    return {
        "n_periods": n,
        "mean_car": diff,
        "median_car": float("nan"),
        "std_car": float("nan"),
        "t_stat": t_stat,
        "p_value": p_value,
        "sign_pos": 0,
        "sign_neg": 0,
        "sign_p_value": float("nan"),
        "boot_ci_low": lo,
        "boot_ci_high": hi,
    }


def build_summary_rows(
    period_car: pd.DataFrame,
    observations: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not len(period_car):
        return pd.DataFrame()

    meta = events.set_index("issue")[
        [
            "publication_date",
            "polluted",
            "pollution_hits",
            "sentiment_ntusd_simplified",
            "sentiment_dlut",
            "sentiment_mpr_domain",
            "sentiment_group",
        ]
    ].to_dict(orient="index")

    def add_rows(
        frame: pd.DataFrame,
        model: str,
        dictionary: str,
        robustness: str,
        pollution_excluded: bool,
    ) -> None:
        if not len(frame):
            return
        if dictionary == MAIN_DICT:
            group_labels = {
                issue: meta.get(issue, {}).get("sentiment_group")
                for issue in frame["issue"].unique()
            }
        else:
            scores = {
                issue: meta.get(issue, {}).get(f"sentiment_{dictionary}")
                for issue in frame["issue"].unique()
            }
            valid = {k: v for k, v in scores.items() if pd.notna(v)}
            ordered = sorted(valid, key=lambda k: valid[k])
            n = len(ordered)
            group_labels = {
                issue: (
                    "low" if idx < n / 3 else ("mid" if idx < 2 * n / 3 else "high")
                )
                for idx, issue in enumerate(ordered)
            }
        if pollution_excluded:
            frame = frame[
                ~frame["issue"].map(lambda issue: meta.get(issue, {}).get("polluted", False))
            ]
        for (est, window), sub in frame.groupby(["estimation_window", "event_window"]):
            issue_groups: dict[str, list[float]] = {}
            for issue, value in zip(sub["issue"], sub["period_car"]):
                if pd.isna(value):
                    continue
                group = group_labels.get(issue)
                issue_groups.setdefault("all", []).append(float(value))
                if group in {"high", "mid", "low"}:
                    issue_groups.setdefault(group, []).append(float(value))
            n_ind = float(sub["n_industries"].mean()) if "n_industries" in sub else float("nan")
            for group in ["all", "high", "mid", "low", "neutral"]:
                values = np.array(
                    issue_groups.get("mid" if group == "neutral" else group, []),
                    dtype=float,
                )
                if not len(values):
                    continue
                stats_row = summarize(values)
                rows.append(
                    {
                        "robustness": robustness,
                        "model": model,
                        "sentiment_dictionary": dictionary,
                        "estimation_window": int(est),
                        "event_window": int(window),
                        "group": group,
                        "n_periods": stats_row["n_periods"],
                        "n_industries_mean": n_ind,
                        "mean_car": stats_row["mean_car"],
                        "median_car": stats_row["median_car"],
                        "std_car": stats_row["std_car"],
                        "t_stat": stats_row["t_stat"],
                        "p_value": stats_row["p_value"],
                        "sign_pos": stats_row["sign_pos"],
                        "sign_neg": stats_row["sign_neg"],
                        "sign_p_value": stats_row["sign_p_value"],
                        "boot_ci_low": stats_row["boot_ci_low"],
                        "boot_ci_high": stats_row["boot_ci_high"],
                        "pollution_excluded": pollution_excluded,
                    }
                )
            high = np.array(issue_groups.get("high", []), dtype=float)
            low = np.array(issue_groups.get("low", []), dtype=float)
            if len(high) and len(low):
                diff = group_diff(high, low)
                rows.append(
                    {
                        "robustness": robustness,
                        "model": model,
                        "sentiment_dictionary": dictionary,
                        "estimation_window": int(est),
                        "event_window": int(window),
                        "group": "high_minus_low",
                        "n_periods": diff["n_periods"],
                        "n_industries_mean": n_ind,
                        "mean_car": diff["mean_car"],
                        "median_car": diff["median_car"],
                        "std_car": diff["std_car"],
                        "t_stat": diff["t_stat"],
                        "p_value": diff["p_value"],
                        "sign_pos": diff["sign_pos"],
                        "sign_neg": diff["sign_neg"],
                        "sign_p_value": diff["sign_p_value"],
                        "boot_ci_low": diff["boot_ci_low"],
                        "boot_ci_high": diff["boot_ci_high"],
                        "pollution_excluded": pollution_excluded,
                    }
                )

    market = period_car[period_car["model"] == "market"]
    mean_adj = period_car[period_car["model"] == "mean_adj"]
    excess = period_car[period_car["model"] == "excess_market"]

    add_rows(market[market["estimation_window"] == MAIN_EST], "market", MAIN_DICT, "base", False)
    add_rows(mean_adj[mean_adj["estimation_window"] == MAIN_EST], "mean_adj", MAIN_DICT, "base", False)
    for est_len in [90, 180, 250]:
        add_rows(
            market[market["estimation_window"] == est_len],
            "market",
            MAIN_DICT,
            f"est{est_len}",
            False,
        )
    add_rows(market[market["estimation_window"] == MAIN_EST], "market", MAIN_DICT, "excl_pollution", True)
    add_rows(mean_adj[mean_adj["estimation_window"] == MAIN_EST], "mean_adj", MAIN_DICT, "excl_pollution", True)
    for dictionary in ["dlut", "mpr_domain"]:
        add_rows(
            market[market["estimation_window"] == MAIN_EST],
            "market",
            dictionary,
            f"dict_{dictionary}",
            False,
        )

    long16 = {
        "801010", "801030", "801040", "801050", "801080", "801110", "801120",
        "801130", "801140", "801150", "801160", "801170", "801180", "801200",
        "801210", "801230",
    }
    long_obs = observations[
        (observations["model"] != "excess_market") & (observations["code"].isin(long16))
    ]
    if len(long_obs):
        long_market = (
            long_obs.dropna(subset=["car_market"])
            .groupby(["issue", "estimation_window", "event_window"], as_index=False)
            .agg(
                period_car=("car_market", "mean"),
                n_industries=("car_market", "size"),
            )
        )
        long_market["model"] = "market"
        add_rows(
            long_market[long_market["estimation_window"] == MAIN_EST],
            "market",
            MAIN_DICT,
            "long16",
            False,
        )
    if len(excess):
        add_rows(excess, "excess_market", MAIN_DICT, "excess_rf", False)
    return pd.DataFrame(rows)


def make_plot(summary: pd.DataFrame) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    base = summary[summary["robustness"] == "base"]
    market = base[base["model"] == "market"]
    windows = sorted(market["event_window"].unique())
    labels = [f"[0,+{window}]" for window in windows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    group_styles = {
        "all": ("全样本", "#4C5B6B", "o"),
        "high": ("高情绪", "#B4436B", "s"),
        "mid": ("中情绪(中性)", "#4C7A57", "^"),
        "low": ("低情绪", "#3E6C8E", "D"),
    }
    for group, (label, color, marker) in group_styles.items():
        sub = market[market["group"] == group].set_index("event_window").reindex(windows)
        means = sub["mean_car"].to_numpy(dtype=float)
        lo = sub["boot_ci_low"].to_numpy(dtype=float)
        hi = sub["boot_ci_high"].to_numpy(dtype=float)
        ax.errorbar(
            range(len(windows)),
            means,
            yerr=[means - lo, hi - means],
            label=label,
            color=color,
            marker=marker,
            capsize=3,
            linewidth=1.6,
        )
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(labels)
    ax.set_title("市场模型：情绪分组的事件窗 CAR 均值（Bootstrap 95% CI）")
    ax.set_ylabel("CAR 均值")
    ax.set_xlabel("事件窗")
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    ax = axes[1]
    for model, label, color in [
        ("market", "市场模型", "#3E6C8E"),
        ("mean_adj", "均值调整模型", "#B4436B"),
    ]:
        sub = base[(base["model"] == model) & (base["group"] == "all")]
        sub = sub.set_index("event_window").reindex(windows)
        means = sub["mean_car"].to_numpy(dtype=float)
        lo = sub["boot_ci_low"].to_numpy(dtype=float)
        hi = sub["boot_ci_high"].to_numpy(dtype=float)
        ax.errorbar(
            range(len(windows)),
            means,
            yerr=[means - lo, hi - means],
            label=label,
            color=color,
            marker="o",
            capsize=3,
            linewidth=1.6,
        )
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(labels)
    ax.set_title("全样本：市场模型 vs 均值调整模型")
    ax.set_ylabel("CAR 均值")
    ax.set_xlabel("事件窗")
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = E2_DIR / "e2_car_plot.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log(f"[plot] wrote {out}")


def write_log(
    environment: dict[str, str],
    asset_rows: list[dict[str, Any]],
    events: pd.DataFrame,
    observations: pd.DataFrame,
    period_car: pd.DataFrame,
    summary: pd.DataFrame,
    skip_rows: list[dict[str, Any]],
    rf: pd.DataFrame | None,
    metrics: dict[str, Any],
) -> None:
    lines: list[str] = []
    add = lines.append
    add("# E2 事件研究 Baseline 实验日志")
    add("")
    add("- 实验编号：E2")
    add("- 运行日期：2026-11-19")
    add("- 脚本：`experiments/E2/run_e2.py`")
    add(f"- 随机种子：{SEED}")
    add(f"- Bootstrap 次数：{N_BOOT}")
    add(f"- Python：{environment['python']}")
    add(
        f"- pandas：{environment['pandas']}；numpy：{environment['numpy']}；"
        f"scipy：{environment['scipy']}；matplotlib：{environment['matplotlib']}"
    )
    add("")
    add("## 1. 输入数据")
    add("")
    add("| 数据 | 文件 | 行数 |")
    add("|------|------|------|")
    add(f"| 行业指数 | `data/external/sw_industries/sw_industry_daily.csv` | {metrics['industry_rows']} |")
    add(f"| 市场基准 | `data/external/benchmark/benchmark_daily.csv` | {metrics['benchmark_rows']} |")
    add(f"| MPR 期次 | `experiments/E1/mpr_index_full.csv` | {metrics['mpr_issues']} |")
    add(f"| 情绪得分 | `experiments/E1/sentiment_dictionary_full.csv` | {metrics['sentiment_rows']} |")
    add(
        "| 无风险利率 | `_temp/e2/rf_rate.csv`（akshare `bond_zh_us_rate`，"
        f"中国 2 年期国债收益率） | {metrics['rf_rows']} |"
    )
    add(f"| 市场基准来源 | `{metrics['benchmark_source']}` | - |")
    add("")
    add("行业资产（前 5 条与末 3 条）：")
    add("")
    add("| 代码 | 名称 | 行数 | 起始 | 结束 |")
    add("|------|------|------|------|------|")
    for row in asset_rows[:5] + asset_rows[-3:]:
        add(f"| {row['code']} | {row['name']} | {row['rows']} | {row['start']} | {row['end']} |")
    add("")
    add("## 2. 事件对齐")
    add("")
    add("- 事件日定义：MPR 发布日期 `publication_date`。")
    add("- 事件交易日对齐状态分布：")
    add("")
    add("| 对齐状态 | 期数 |")
    add("|----------|------|")
    for status, count in events["alignment_status"].value_counts().items():
        add(f"| {status} | {count} |")
    add("")
    add(
        f"- 事件日早于基准起点（2002-01-04）或无后续交易日而被剔除的期次："
        f"{metrics['dropped_event_issues']}"
    )
    add(
        f"- 估计窗/事件窗数据不足而剔除的 (期, 行业, 窗) 单元：{metrics['skipped_cells']}"
        "（明细见 `_temp/e2/e2_skip_cells.csv`）"
    )
    add(f"- 主样本可用期数（市场模型，估计窗 120）：{metrics['main_periods']}")
    add("")
    add("## 3. 方法")
    add("")
    add("- 估计窗：事件前 120 个交易日（t-120 至 t-1）；稳健性另用 90 / 180 / 250。")
    add("- 事件窗：[0,+1]、[0,+3]、[0,+5]、[0,+10]。")
    add("- 市场模型：`R_i = alpha + beta * R_m + eps`，估计窗 OLS，正常收益 `alpha + beta * R_m`。")
    add("- 均值调整模型：正常收益 = 估计窗行业收益均值。")
    add("- AR = 实际收益 - 正常收益；CAR = 事件窗 AR 之和。")
    add(
        "- 横截面均值：每个事件期对可用行业 CAR 等权平均，得到期层面 CAR；"
        "t 值、符号检验、Bootstrap CI 均在期层面计算。"
    )
    add(
        "- 分组：NTUSD 简体得分三分位（高/中/低），中组同时作为“中性情绪组”；"
        "DLUT 与自制词典用于稳健性。"
    )
    add("- 组间差异：高组期均值 - 低组期均值，Welch t 检验 + Bootstrap CI。")
    add(
        "- 无风险利率：用于补充的“超额市场模型”（行业与市场收益均减 2 年期国债日收益）；"
        "主模型不依赖无风险利率。"
    )
    add("")
    add("## 4. 污染事件检查")
    add("")
    add(
        f"- 污染事件清单：`_temp/e2/pollution_events.csv`（{len(POLLUTION_EVENTS)} 条，"
        "公开历史资料/公开报道整理，部分条目经 Bing 核验）。"
    )
    add("- 检查窗口：每个 MPR 事件的最宽事件窗 [0,+10]。")
    add(f"- 被标记为污染的事件期数：{int(events['polluted'].sum())} / {len(events)}。")
    add("")
    add("| 事件期 | 发布日 | 污染命中 |")
    add("|--------|--------|----------|")
    for _, row in events[events["polluted"]].iterrows():
        add(f"| {row['issue']} | {row['publication_date']} | {row['pollution_hits']} |")
    add("")
    add("## 5. 主结果（市场模型，估计窗 120）")
    add("")
    main = summary[(summary["robustness"] == "base") & (summary["model"] == "market")]
    add("| 事件窗 | 组 | 期数 | CAR 均值 | 中位 | 标准差 | t | p | 符号检验 p | Bootstrap 95% CI |")
    add("|--------|----|------|----------|------|--------|---|-----|-------------|------------------|")
    for _, row in main.iterrows():
        add(
            f"| [0,+{int(row['event_window'])}] | {row['group']} | {int(row['n_periods'])} | "
            f"{row['mean_car']:.6f} | {row['median_car']:.6f} | {row['std_car']:.6f} | "
            f"{row['t_stat']:.3f} | {row['p_value']:.4f} | {row['sign_p_value']:.4f} | "
            f"[{row['boot_ci_low']:.6f}, {row['boot_ci_high']:.6f}] |"
        )
    add("")
    add("## 6. 对照与稳健性")
    add("")
    add("| 稳健性 | 模型 | 事件窗 | 组 | 期数 | CAR 均值 | t | p | Bootstrap 95% CI |")
    add("|--------|------|--------|----|------|----------|---|-----|------------------|")
    for _, row in summary[summary["robustness"] != "base"].iterrows():
        add(
            f"| {row['robustness']} | {row['model']} | [0,+{int(row['event_window'])}] | {row['group']} | "
            f"{int(row['n_periods'])} | {row['mean_car']:.6f} | {row['t_stat']:.3f} | "
            f"{row['p_value']:.4f} | [{row['boot_ci_low']:.6f}, {row['boot_ci_high']:.6f}] |"
        )
    add("")
    add("## 7. 估计窗与上一期事件的重叠检查")
    add("")
    if len(observations):
        contains = observations[observations["estimation_window"] == MAIN_EST][
            "est_window_contains_prior_event"
        ]
        add(
            f"- 主估计窗（120 日）内包含上一期 MPR 发布日的 (期, 行业) 单元占比："
            f"{contains.mean():.2%}。"
        )
        add("- 该重叠是季度 MPR 间隔较短所致，已用 90 / 180 / 250 日估计窗做稳健性（见上表）。")
    add("")
    add("## 8. 限制与说明")
    add("")
    add("- 基准沪深300 自 2002-01-04 起，2001 年部分 MPR 期次无法进入市场模型样本。")
    add(
        "- 申万一级行业指数起点不同：16 个行业自 1999-12-30，14 个行业自 2014-02-21"
        "（其中 3 个自 2021-12-13）；早期事件横截面行业数较少，`long16` 稳健性给出长样本行业结果。"
    )
    add(
        "- 估计窗要求窗口内每个交易日都有行业数据；缺失或不足会剔除该 (期, 行业, 窗) 单元，"
        "明细见 `_temp/e2/e2_skip_cells.csv`。"
    )
    add("- 污染清单基于公开历史资料整理，可能不全；剔除污染事件的结果作为稳健性，不与主结果混用。")
    add("- 若全样本或分组 CAR 不显著，本日志如实记录，不做因果解释。")
    add("")
    add("## 9. 完成条件核对")
    add("")
    add("- 代码可复现：`python experiments/E2/run_e2.py`，随机种子与脚本哈希已记录。")
    add("- 事件窗与估计窗不重叠：估计窗为 t-120 至 t-1，事件窗为 t 至 t+k。")
    add("- 显著性与置信区间：主表含 t 值、p 值、符号检验 p、Bootstrap 95% CI。")
    add("- 分组比较：高/中/低三组及高-低差值均已输出。")
    add(f"- 全样本与分组 CAR 显著性：{metrics['significance_note']}")
    add("")

    out = LOG_DIR / "e2_event_baseline.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    log(f"[log] wrote {out}")


def main() -> int:
    ensure_dirs()
    (TEMP_DIR / "e2_run.log").write_text("", encoding="utf-8")
    log("E2 event-study baseline start")

    environment = {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }

    industry, bench, calendar, asset_rows = load_prices()
    log(
        f"[data] industry rows={len(industry)}, benchmark rows={len(bench)}, "
        f"calendar days={len(calendar)}"
    )
    rf = fetch_rf()
    if rf is not None:
        log(f"[data] risk-free rows={len(rf)}, range {rf['date'].min()}..{rf['date'].max()}")
    else:
        log("[data] risk-free rate unavailable; excess-return robustness skipped")

    events = load_events()
    events = align_events(events, calendar)
    events = assign_groups(events, MAIN_DICT)
    events = pollution_flags(events, calendar)
    events.to_csv(TEMP_DIR / "e2_events_aligned.csv", index=False, encoding="utf-8-sig")
    log(
        f"[events] issues={len(events)}, aligned={int(events['event_idx'].notna().sum())}, "
        f"polluted={int(events['polluted'].sum())}"
    )

    observations, period_car, skip_rows = build_observations(industry, bench, calendar, events, rf)
    observations.to_csv(TEMP_DIR / "e2_observations.csv", index=False, encoding="utf-8-sig")
    period_car.to_csv(TEMP_DIR / "e2_period_car.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(skip_rows).to_csv(TEMP_DIR / "e2_skip_cells.csv", index=False, encoding="utf-8-sig")
    log(
        f"[obs] observation rows={len(observations)}, period rows={len(period_car)}, "
        f"skipped cells={len(skip_rows)}"
    )

    summary = build_summary_rows(period_car, observations, events)
    summary.to_csv(E2_DIR / "e2_event_baseline.csv", index=False, encoding="utf-8-sig")
    summary[summary["robustness"] != "base"].to_csv(
        TEMP_DIR / "e2_robustness.csv", index=False, encoding="utf-8-sig"
    )
    log(f"[summary] rows={len(summary)}")

    if len(summary):
        make_plot(summary)

    def sig_note(model: str, group: str) -> str:
        sub = summary[
            (summary["robustness"] == "base")
            & (summary["model"] == model)
            & (summary["group"] == group)
        ]
        hits = [
            f"[0,+{int(row.event_window)}] p={row.p_value:.4f}"
            for row in sub.itertuples()
            if row.p_value < 0.05
        ]
        return "、".join(hits) if hits else "无"

    base_market = summary[
        (summary["robustness"] == "base")
        & (summary["model"] == "market")
        & (summary["group"] == "all")
    ]
    main_periods = int(base_market["n_periods"].max()) if len(base_market) else 0
    significance_note = (
        "全样本市场模型显著事件窗："
        + sig_note("market", "all")
        + "；低情绪组显著事件窗："
        + sig_note("market", "low")
        + "；高-低差值显著事件窗："
        + sig_note("market", "high_minus_low")
        + "；均值调整模型全样本显著事件窗："
        + sig_note("mean_adj", "all")
        + "。其余事件窗/组均不显著，如实记录，不做因果解释。"
    )

    metrics = {
        "industry_rows": int(len(industry)),
        "benchmark_rows": int(len(bench)),
        "benchmark_source": (
            "workspace_cache:_temp/e0_market_raw/benchmark_sh000300.csv"
            if BENCH_CACHE_PATH.exists() and BENCH_CACHE_PATH.stat().st_size > 0
            else "workspace:data/external/benchmark/benchmark_daily.csv"
        ),
        "mpr_issues": int(len(events)),
        "sentiment_rows": int(len(pd.read_csv(SENTIMENT_PATH))),
        "rf_rows": int(len(rf)) if rf is not None else 0,
        "dropped_event_issues": int(events["event_idx"].isna().sum()),
        "skipped_cells": int(len(skip_rows)),
        "observation_rows": int(len(observations)),
        "period_rows": int(len(period_car)),
        "summary_rows": int(len(summary)),
        "polluted_issues": int(events["polluted"].sum()),
        "main_periods": main_periods,
        "significance_note": significance_note,
        "script_sha256": sha256(Path(__file__)),
        "seed": SEED,
        "n_boot": N_BOOT,
    }
    (E2_DIR / "e2_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    write_log(
        environment,
        asset_rows,
        events,
        observations,
        period_car,
        summary,
        skip_rows,
        rf,
        metrics,
    )
    log("E2 done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
