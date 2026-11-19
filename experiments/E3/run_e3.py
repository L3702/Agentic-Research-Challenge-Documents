# -*- coding: utf-8 -*-
"""E3 formal event-study experiment for the MPR sentiment project.

Estimation pipeline mirrors experiments/E2/run_e2.py (market model /
mean-adjusted / excess market model, estimation window t-120..t-1) and
extends it with:
  * the full 31 Shenwan level-1 industries,
  * pre-registered sentiment terciles and two pre-registered industry
    groupings (sector mapping and full-sample market beta median split),
  * event-window consistency checks,
  * full-sample vs subsample comparisons.

Outputs land in experiments/E3, cache files in _temp/e3.
"""

from __future__ import annotations

import hashlib
import json
import shutil
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
E3_DIR = ROOT / "experiments" / "E3"
LOG_DIR = E3_DIR / "logs"
TEMP_DIR = ROOT / "_temp" / "e3"

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

# Pre-registered descriptive industry grouping by chain position / board
# attribute. The mapping is fixed in code and is NOT chosen from event-study
# results. 31 industries, 8 + 8 + 8 + 4 + 3.
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

LONG16 = {
    "801010", "801030", "801040", "801050", "801080", "801110", "801120",
    "801130", "801140", "801150", "801160", "801170", "801180", "801200",
    "801210", "801230",
}

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
    for path in [E3_DIR, LOG_DIR, TEMP_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def log(line: str = "") -> None:
    print(line)
    with (TEMP_DIR / "e3_run.log").open("a", encoding="utf-8") as handle:
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
    industry["sector_group"] = industry["code"].map(SECTOR_GROUPS)

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
                "sector_group": SECTOR_GROUPS[code],
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
            "sector_group": "benchmark",
        }
    )
    return industry, bench, calendar, asset_rows


def fetch_rf() -> pd.DataFrame | None:
    cache = TEMP_DIR / "rf_rate.csv"
    if cache.exists() and cache.stat().st_size > 0:
        return pd.read_csv(cache)
    reused = TEMP_DIR.parent / "e2" / "rf_rate.csv"
    if reused.exists() and reused.stat().st_size > 0:
        shutil.copyfile(reused, cache)
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


def full_sample_betas(industry: pd.DataFrame, bench: pd.DataFrame) -> dict[str, float]:
    """Market beta of each industry over the full sample (no event CAR used)."""
    bench_ret = (
        bench[bench["code"] == BENCH_CODE]
        .set_index("date")["daily_return"]
        .sort_index()
    )
    out: dict[str, float] = {}
    for code, frame in industry.groupby("code"):
        merged = pd.DataFrame({"ri": frame.set_index("date")["daily_return"]}).join(
            bench_ret.rename("rm"), how="inner"
        ).dropna()
        if len(merged) < 250:
            out[code] = float("nan")
            continue
        out[code] = float(np.polyfit(merged["rm"].to_numpy(), merged["ri"].to_numpy(), 1)[0])
    return out


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
                            "report_period": issue_row["report_period"],
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
                            "n_days": int(len(event_dates)),
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
                                    "report_period": issue_row["report_period"],
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
                                    "n_days": int(len(event_dates)),
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


def stats_row(values: np.ndarray, n_industries: float) -> dict[str, Any]:
    row = summarize(values)
    row["n_industries_mean"] = float(n_industries)
    return row


def sentiment_summary(
    period_car: pd.DataFrame,
    events: pd.DataFrame,
    estimation_window: int = MAIN_EST,
) -> pd.DataFrame:
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
    rows: list[dict[str, Any]] = []
    if not len(period_car):
        return pd.DataFrame()
    for model in ["market", "mean_adj", "excess_market"]:
        frame = period_car[
            (period_car["model"] == model)
            & (period_car["estimation_window"] == estimation_window)
        ]
        if not len(frame):
            continue
        for window, sub in frame.groupby("event_window"):
            issue_groups: dict[str, list[float]] = {}
            for issue, value in zip(sub["issue"], sub["period_car"]):
                if pd.isna(value):
                    continue
                group = meta.get(issue, {}).get("sentiment_group")
                issue_groups.setdefault("all", []).append(float(value))
                if group in {"high", "mid", "low"}:
                    issue_groups.setdefault(group, []).append(float(value))
            n_ind = float(sub["n_industries"].mean())
            for group in ["all", "high", "mid", "low", "neutral"]:
                values = np.array(
                    issue_groups.get("mid" if group == "neutral" else group, []),
                    dtype=float,
                )
                if not len(values):
                    continue
                row = stats_row(values, n_ind)
                rows.append({"model": model, "event_window": int(window), "group": group, **row})
            high = np.array(issue_groups.get("high", []), dtype=float)
            low = np.array(issue_groups.get("low", []), dtype=float)
            if len(high) and len(low):
                diff = group_diff(high, low)
                rows.append(
                    {"model": model, "event_window": int(window), "group": "high_minus_low", **diff}
                )
    return pd.DataFrame(rows)


def car_long(observations: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model, column in [
        ("market", "car_market"),
        ("mean_adj", "car_mean_adj"),
        ("excess_market", "car_market"),
    ]:
        # E2 stores the mean-adjusted CAR as a column on the market-model
        # rows; only the excess-return model has its own row type.
        source_model = "market" if model == "mean_adj" else model
        sub = observations[observations["model"] == source_model].dropna(subset=[column]).copy()
        if not len(sub):
            continue
        sub["car"] = sub[column]
        sub["model"] = model
        frames.append(sub[["issue", "code", "name", "model", "estimation_window", "event_window", "car", "polluted", "sentiment_group"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def industry_car(car: pd.DataFrame, asset_rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not len(car):
        return pd.DataFrame()
    info = pd.DataFrame(asset_rows)
    info = info[info["sector_group"] != "benchmark"][["code", "name", "sector_group"]]
    frame = car.merge(info, on=["code", "name"], how="left")
    # Keep the main estimation window so the industry ranking is comparable
    # with the headline results; other estimation windows stay in the
    # robustness table.
    frame = frame[frame["estimation_window"] == MAIN_EST]
    rows: list[dict[str, Any]] = []
    for (code, name, sector, model, window), sub in frame.groupby(
        ["code", "name", "sector_group", "model", "event_window"]
    ):
        values = sub["car"].to_numpy(dtype=float)
        row = stats_row(values, float("nan"))
        rows.append(
            {
                "code": code,
                "name": name,
                "sector_group": sector,
                "model": model,
                "estimation_window": MAIN_EST,
                "event_window": int(window),
                **row,
            }
        )
    return pd.DataFrame(rows)


def industry_group_summary(
    car: pd.DataFrame,
    asset_rows: list[dict[str, Any]],
    beta_group: dict[str, str],
    estimation_window: int = MAIN_EST,
) -> pd.DataFrame:
    """Issue-level industry-group CAR means, then across-issue stats."""
    if not len(car):
        return pd.DataFrame()
    info = pd.DataFrame(asset_rows)
    info = info[info["sector_group"] != "benchmark"][["code", "name", "sector_group"]]
    frame = car.merge(info, on=["code", "name"], how="left")
    frame["beta_group"] = frame["code"].map(beta_group)
    frame = frame[frame["estimation_window"] == estimation_window]

    rows: list[dict[str, Any]] = []
    for group_col in ["sector_group", "beta_group"]:
        for model in ["market", "mean_adj", "excess_market"]:
            sub = frame[frame["model"] == model]
            if not len(sub):
                continue
            for window, win_sub in sub.groupby("event_window"):
                issue_group = win_sub.groupby(["issue", group_col])["car"].mean()
                for group_value, issue_values in issue_group.groupby(level=1):
                    values = issue_values.to_numpy(dtype=float)
                    row = stats_row(values, float("nan"))
                    rows.append(
                        {
                            "grouping": group_col,
                            "group": str(group_value),
                            "model": model,
                            "event_window": int(window),
                            **row,
                        }
                    )
                pairs: list[tuple[str, str]] = []
                if group_col == "sector_group":
                    pairs = [
                        ("上游资源与周期", "大消费与医药"),
                        ("上游资源与周期", "公用与交运"),
                        ("中游制造与科技", "大消费与医药"),
                    ]
                else:
                    pairs = [("high_beta", "low_beta")]
                for left, right in pairs:
                    levels = issue_group.index.get_level_values(1)
                    a = issue_group.xs(left, level=1).to_numpy(dtype=float) if left in levels else np.array([])
                    b = issue_group.xs(right, level=1).to_numpy(dtype=float) if right in levels else np.array([])
                    if len(a) and len(b):
                        diff = group_diff(a, b)
                        rows.append(
                            {
                                "grouping": group_col,
                                "group": f"{left} - {right}",
                                "model": model,
                                "event_window": int(window),
                                **diff,
                            }
                        )
    return pd.DataFrame(rows)


def window_consistency(period_car: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Sign agreement and rank correlation of period CAR across windows."""
    meta = events.set_index("issue")["sentiment_group"].to_dict()
    rows: list[dict[str, Any]] = []
    for model in ["market", "mean_adj"]:
        frame = period_car[
            (period_car["model"] == model)
            & (period_car["estimation_window"] == MAIN_EST)
        ]
        if not len(frame):
            continue
        for group in ["all", "high", "low"]:
            if group == "all":
                sub = frame
            else:
                issues = {issue for issue, g in meta.items() if g == group}
                sub = frame[frame["issue"].isin(issues)]
            pivot = sub.pivot_table(index="issue", columns="event_window", values="period_car")
            pivot = pivot.dropna()
            for left, right in [(1, 3), (3, 5), (5, 10), (1, 10)]:
                if left not in pivot.columns or right not in pivot.columns:
                    continue
                pair = pivot[[left, right]].dropna()
                if len(pair) < 5:
                    continue
                sign_agree = float((np.sign(pair[left]) == np.sign(pair[right])).mean())
                rho = float(stats.spearmanr(pair[left], pair[right]).statistic)
                rows.append(
                    {
                        "model": model,
                        "group": group,
                        "window_left": left,
                        "window_right": right,
                        "n_periods": int(len(pair)),
                        "sign_agreement": sign_agree,
                        "spearman_rho": rho,
                    }
                )
    return pd.DataFrame(rows)


def subsample_summary(
    period_car: pd.DataFrame,
    events: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Full sample vs subsamples: base / excl_pollution / long16 / post2014."""
    meta = events.set_index("issue")[
        ["publication_date", "polluted", "sentiment_group"]
    ].to_dict(orient="index")
    long16_obs = observations[
        (observations["model"] != "excess_market")
        & (observations["code"].isin(LONG16))
    ]
    long16_frames: list[pd.DataFrame] = []
    for column, model_name in [("car_market", "market"), ("car_mean_adj", "mean_adj")]:
        grouped = (
            long16_obs.dropna(subset=[column])
            .groupby(["issue", "estimation_window", "event_window"], as_index=False)
            .agg(period_car=(column, "mean"), n_industries=(column, "size"))
        )
        grouped["model"] = model_name
        long16_frames.append(grouped)
    long16_period = pd.concat(long16_frames, ignore_index=True)

    post_issues = {
        issue
        for issue, value in meta.items()
        if str(value.get("publication_date")) >= "2014-02-21"
    }

    rows: list[dict[str, Any]] = []
    for label, frame in [
        ("base", period_car),
        ("excl_pollution", period_car[~period_car["issue"].map(lambda i: meta.get(i, {}).get("polluted", False))]),
        ("long16", long16_period),
        ("post2014", period_car[period_car["issue"].isin(post_issues)]),
    ]:
        for model in ["market", "mean_adj"]:
            sub = frame[
                (frame["model"] == model) & (frame["estimation_window"] == MAIN_EST)
            ]
            if not len(sub):
                continue
            for window, win_sub in sub.groupby("event_window"):
                issue_groups: dict[str, list[float]] = {}
                for issue, value in zip(win_sub["issue"], win_sub["period_car"]):
                    if pd.isna(value):
                        continue
                    group = meta.get(issue, {}).get("sentiment_group")
                    issue_groups.setdefault("all", []).append(float(value))
                    if group in {"high", "low"}:
                        issue_groups.setdefault(group, []).append(float(value))
                for group in ["all", "high", "low"]:
                    values = np.array(issue_groups.get(group, []), dtype=float)
                    if not len(values):
                        continue
                    row = summarize(values)
                    rows.append(
                        {
                            "subsample": label,
                            "model": model,
                            "event_window": int(window),
                            "group": group,
                            **row,
                        }
                    )
                high = np.array(issue_groups.get("high", []), dtype=float)
                low = np.array(issue_groups.get("low", []), dtype=float)
                if len(high) and len(low):
                    diff = group_diff(high, low)
                    rows.append(
                        {
                            "subsample": label,
                            "model": model,
                            "event_window": int(window),
                            "group": "high_minus_low",
                            **diff,
                        }
                    )
    return pd.DataFrame(rows)


def robustness_summary(
    period_car: pd.DataFrame,
    observations: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Estimation-window, pollution, dictionary and excess-return robustness."""
    meta = events.set_index("issue")[
        ["polluted", "sentiment_ntusd_simplified", "sentiment_dlut", "sentiment_mpr_domain", "sentiment_group"]
    ].to_dict(orient="index")

    def labels_for(dictionary: str) -> dict[str, str]:
        if dictionary == MAIN_DICT:
            return {issue: value.get("sentiment_group") for issue, value in meta.items()}
        scores = {
            issue: value.get(f"sentiment_{dictionary}") for issue, value in meta.items()
        }
        valid = {k: v for k, v in scores.items() if pd.notna(v)}
        ordered = sorted(valid, key=lambda k: valid[k])
        n = len(ordered)
        return {
            issue: ("low" if idx < n / 3 else ("mid" if idx < 2 * n / 3 else "high"))
            for idx, issue in enumerate(ordered)
        }

    rows: list[dict[str, Any]] = []

    def emit(frame: pd.DataFrame, model: str, label: str, dictionary: str, pollution_excluded: bool) -> None:
        if not len(frame):
            return
        group_labels = labels_for(dictionary)
        frame = frame.copy()
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
                if group in {"high", "low"}:
                    issue_groups.setdefault(group, []).append(float(value))
            for group in ["all", "high", "low"]:
                values = np.array(issue_groups.get(group, []), dtype=float)
                if not len(values):
                    continue
                row = summarize(values)
                rows.append(
                    {
                        "robustness": label,
                        "model": model,
                        "dictionary": dictionary,
                        "estimation_window": int(est),
                        "event_window": int(window),
                        "group": group,
                        "pollution_excluded": pollution_excluded,
                        **row,
                    }
                )
            high = np.array(issue_groups.get("high", []), dtype=float)
            low = np.array(issue_groups.get("low", []), dtype=float)
            if len(high) and len(low):
                diff = group_diff(high, low)
                rows.append(
                    {
                        "robustness": label,
                        "model": model,
                        "dictionary": dictionary,
                        "estimation_window": int(est),
                        "event_window": int(window),
                        "group": "high_minus_low",
                        "pollution_excluded": pollution_excluded,
                        **diff,
                    }
                )

    market = period_car[period_car["model"] == "market"]
    mean_adj = period_car[period_car["model"] == "mean_adj"]
    excess = period_car[period_car["model"] == "excess_market"]
    for est_len in [90, 180, 250]:
        emit(market[market["estimation_window"] == est_len], "market", f"est{est_len}", MAIN_DICT, False)
    emit(market[market["estimation_window"] == MAIN_EST], "market", "excl_pollution", MAIN_DICT, True)
    emit(mean_adj[mean_adj["estimation_window"] == MAIN_EST], "mean_adj", "excl_pollution", MAIN_DICT, True)
    for dictionary in ["dlut", "mpr_domain"]:
        emit(market[market["estimation_window"] == MAIN_EST], "market", f"dict_{dictionary}", dictionary, False)
    if len(excess):
        emit(excess, "excess_market", "excess_rf", MAIN_DICT, False)
    return pd.DataFrame(rows)


def make_plot(
    sentiment: pd.DataFrame,
    sector: pd.DataFrame,
    beta: pd.DataFrame,
    industry: pd.DataFrame,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    base = sentiment[sentiment["model"] == "market"]
    windows = sorted(base["event_window"].unique())
    labels = [f"[0,+{window}]" for window in windows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    group_styles = {
        "all": ("全样本", "#4C5B6B", "o"),
        "high": ("高情绪", "#B4436B", "s"),
        "mid": ("中情绪", "#4C7A57", "^"),
        "low": ("低情绪", "#3E6C8E", "D"),
    }
    for group, (label, color, marker) in group_styles.items():
        sub = base[base["group"] == group].set_index("event_window").reindex(windows)
        means = sub["mean_car"].to_numpy(dtype=float)
        lo = sub["boot_ci_low"].to_numpy(dtype=float)
        hi = sub["boot_ci_high"].to_numpy(dtype=float)
        ax.errorbar(range(len(windows)), means, yerr=[means - lo, hi - means], label=label, color=color, marker=marker, capsize=3, linewidth=1.6)
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(labels)
    ax.set_title("市场模型：情绪分组的事件窗 CAR（Bootstrap 95% CI）")
    ax.set_ylabel("CAR 均值")
    ax.set_xlabel("事件窗")
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    diff = base[base["group"] == "high_minus_low"].set_index("event_window").reindex(windows)
    means = diff["mean_car"].to_numpy(dtype=float)
    lo = diff["boot_ci_low"].to_numpy(dtype=float)
    hi = diff["boot_ci_high"].to_numpy(dtype=float)
    ax.errorbar(range(len(windows)), means, yerr=[means - lo, hi - means], color="#8A4B6B", marker="o", capsize=3, linewidth=1.6)
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(labels)
    ax.set_title("高情绪 - 低情绪：事件窗 CAR 差值（Bootstrap 95% CI）")
    ax.set_ylabel("CAR 差值")
    ax.set_xlabel("事件窗")
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    sector_market = sector[
        (sector["grouping"] == "sector_group") & (sector["model"] == "market") & (~sector["group"].str.contains(" - "))
    ]
    colors = ["#3E6C8E", "#B4436B", "#4C7A57", "#8A6BAE", "#B07A3E"]
    for color, group in zip(colors, SECTOR_ORDER):
        sub = sector_market[sector_market["group"] == group].set_index("event_window").reindex(windows)
        means = sub["mean_car"].to_numpy(dtype=float)
        lo = sub["boot_ci_low"].to_numpy(dtype=float)
        hi = sub["boot_ci_high"].to_numpy(dtype=float)
        ax.errorbar(range(len(windows)), means, yerr=[means - lo, hi - means], label=group, color=color, marker="o", capsize=3, linewidth=1.4)
    beta_market = beta[
        (beta["grouping"] == "beta_group") & (beta["model"] == "market") & (beta["group"].isin(["high_beta", "low_beta"]))
    ]
    for group, color in [("high_beta", "#8A4B6B"), ("low_beta", "#3E6C8E")]:
        sub = beta_market[beta_market["group"] == group].set_index("event_window").reindex(windows)
        means = sub["mean_car"].to_numpy(dtype=float)
        ax.plot(range(len(windows)), means, label=f"{group}(全样本beta中位分组)", color=color, marker="x", linestyle="--", linewidth=1.3)
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(labels)
    ax.set_title("行业市场模型 CAR：板块分组与 beta 分组（描述性）")
    ax.set_ylabel("CAR 均值")
    ax.set_xlabel("事件窗")
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 1]
    ind = industry[
        (industry["model"] == "market") & (industry["event_window"] == 5)
    ].copy()
    ind["label"] = ind["code"] + " " + ind["name"]
    ind = ind.sort_values("mean_car")
    color_map = dict(zip(SECTOR_ORDER, colors))
    bar_colors = [color_map.get(value, "#999999") for value in ind["sector_group"]]
    ax.barh(ind["label"], ind["mean_car"], color=bar_colors, height=0.7)
    ax.axvline(0.0, color="#666666", linewidth=0.8)
    ax.set_title("行业 [0,+5] 平均 CAR（市场模型，估计窗 120，描述性）")
    ax.set_xlabel("平均 CAR")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    out = E3_DIR / "e3_car_plot.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log(f"[plot] wrote {out}")


def write_log(
    environment: dict[str, str],
    asset_rows: list[dict[str, Any]],
    events: pd.DataFrame,
    observations: pd.DataFrame,
    period_car: pd.DataFrame,
    sentiment: pd.DataFrame,
    sector: pd.DataFrame,
    beta: pd.DataFrame,
    industry: pd.DataFrame,
    consistency: pd.DataFrame,
    subsample: pd.DataFrame,
    robustness: pd.DataFrame,
    skip_rows: list[dict[str, Any]],
    rf: pd.DataFrame | None,
    beta_full: dict[str, float],
    beta_group: dict[str, str],
    metrics: dict[str, Any],
) -> None:
    lines: list[str] = []
    add = lines.append
    add("# E3 正式事件研究实验日志")
    add("")
    add("- 实验编号：E3")
    add("- 运行日期：2026-11-19")
    add("- 脚本：`experiments/E3/run_e3.py`")
    add(f"- 随机种子：{SEED}")
    add(f"- Bootstrap 次数：{N_BOOT}")
    add(f"- Python：{environment['python']}")
    add(
        f"- pandas：{environment['pandas']}；numpy：{environment['numpy']}；"
        f"scipy：{environment['scipy']}；matplotlib：{environment['matplotlib']}"
    )
    add("")
    add("## 1. 实验目的与假设")
    add("")
    add("- 检验 H2/H3 的事件研究部分：MPR 情绪高低组在发布后行业收益上是否存在差异，差异是否集中在特定行业。")
    add("- 分组规则事先确定：按 NTUSD 简体情绪得分全样本三分位分为低/中/高组，中组同时作为中性组；不根据事件研究结果调整分组。")
    add("- 行业分组为事先固定的描述性分组：板块映射表（5 组）与全样本市场 beta 中位数分组（高/低 beta），均不依据事件窗 CAR 挑选。")
    add("")
    add("## 2. 输入数据")
    add("")
    add("| 数据 | 文件 | 行数 |")
    add("|------|------|------|")
    add(f"| 行业指数（申万一级 31 个） | `data/external/sw_industries/sw_industry_daily.csv` | {metrics['industry_rows']} |")
    add(f"| 市场基准 | `data/external/benchmark/benchmark_daily.csv` | {metrics['benchmark_rows']} |")
    add(f"| MPR 期次 | `experiments/E1/mpr_index_full.csv` | {metrics['mpr_issues']} |")
    add(f"| 情绪得分 | `experiments/E1/sentiment_dictionary_full.csv` | {metrics['sentiment_rows']} |")
    add(f"| 无风险利率 | `_temp/e3/rf_rate.csv`（akshare `bond_zh_us_rate`，中国 2 年期国债收益率） | {metrics['rf_rows']} |")
    add(f"| 市场基准来源 | `{metrics['benchmark_source']}` | - |")
    add("")
    add("行业资产（31 个）：")
    add("")
    add("| 代码 | 名称 | 行数 | 起始 | 结束 | 板块分组 | 全样本 beta | beta 分组 |")
    add("|------|------|------|------|------|----------|-------------|----------|")
    for row in asset_rows:
        if row["sector_group"] == "benchmark":
            continue
        code = row["code"]
        add(
            f"| {code} | {row['name']} | {row['rows']} | {row['start']} | {row['end']} | "
            f"{row['sector_group']} | {beta_full.get(code, float('nan')):.4f} | {beta_group.get(code, '')} |"
        )
    add("")
    add(f"- beta 分组中位数：{np.median([value for value in beta_full.values() if not np.isnan(value)]):.4f}。")
    add("")
    add("## 3. 事件对齐")
    add("")
    add("- 事件日定义：MPR 发布日期 `publication_date`。")
    add("- 事件交易日对齐状态分布：")
    add("")
    add("| 对齐状态 | 期数 |")
    add("|----------|------|")
    for status, count in events["alignment_status"].value_counts().items():
        add(f"| {status} | {count} |")
    add("")
    add(f"- 因事件日早于基准起点（2002-01-04）或无后续交易日而剔除的期次：{metrics['dropped_event_issues']}")
    add(f"- 估计窗/事件窗数据不足而剔除的 (期, 行业, 窗) 单元：{metrics['skipped_cells']}（明细见 `_temp/e3/e3_skip_cells.csv`）")
    add(f"- 主样本可用期数（市场模型，估计窗 120）：{metrics['main_periods']}")
    add("")
    add("## 4. 方法")
    add("")
    add("- 估计窗：事件前 120 个交易日（t-120 至 t-1）；稳健性另用 90 / 180 / 250。")
    add("- 事件窗：[0,+1]、[0,+3]、[0,+5]、[0,+10]。")
    add("- 市场模型：`R_i = alpha + beta * R_m + eps`，估计窗 OLS，正常收益 `alpha + beta * R_m`；同一事件的所有行业共用同一估计窗长度。")
    add("- 均值调整模型：正常收益 = 估计窗行业收益均值。")
    add("- 超额市场模型：行业与市场收益均减 2 年期国债日收益（无风险利率补充模型）。")
    add("- AR = 实际收益 - 正常收益；CAR = 事件窗 AR 之和。")
    add("- 期层面 CAR：每个事件期对可用行业 CAR 等权平均；t 值、符号检验、Bootstrap CI 均在期层面计算。")
    add("- 行业组 CAR：先对组内行业按期等权平均，再对期做统计；组间差异用 Welch t + Bootstrap CI。")
    add("- 组间差异：高组期均值 - 低组期均值，Welch t 检验 + Bootstrap CI。")
    add("- 污染检查：每个事件最宽事件窗 [0,+10] 与污染清单区间重叠即标记。")
    add("")
    add("## 5. 污染事件检查")
    add("")
    add(
        f"- 污染事件清单：`_temp/e3/pollution_events.csv`（{len(POLLUTION_EVENTS)} 条，"
        "公开历史资料/公开报道整理，部分条目经 Bing 核验；与 E2 清单一致）。"
    )
    add(f"- 被标记为污染的事件期数：{int(events['polluted'].sum())} / {len(events)}。")
    add("")
    add("| 事件期 | 发布日 | 污染命中 |")
    add("|--------|--------|----------|")
    for _, row in events[events["polluted"]].iterrows():
        add(f"| {row['issue']} | {row['publication_date']} | {row['pollution_hits']} |")
    add("")
    add("## 6. 主结果：CAR 按情绪分组（市场模型与均值调整模型，估计窗 120）")
    add("")
    add("| 模型 | 事件窗 | 组 | 期数 | CAR 均值 | 中位 | 标准差 | t | p | 符号检验 p | Bootstrap 95% CI |")
    add("|------|--------|----|------|----------|------|--------|---|-----|-------------|------------------|")
    for _, row in sentiment.iterrows():
        add(
            f"| {row['model']} | [0,+{int(row['event_window'])}] | {row['group']} | {int(row['n_periods'])} | "
            f"{row['mean_car']:.6f} | {row['median_car']:.6f} | {row['std_car']:.6f} | "
            f"{row['t_stat']:.3f} | {row['p_value']:.4f} | {row['sign_p_value']:.4f} | "
            f"[{row['boot_ci_low']:.6f}, {row['boot_ci_high']:.6f}] |"
        )
    add("")
    add("## 7. 行业组间差异（估计窗 120）")
    add("")
    add("### 7.1 板块分组")
    add("")
    add("| 模型 | 事件窗 | 组 | 期数 | CAR 均值 | t | p | Bootstrap 95% CI |")
    add("|------|--------|----|------|----------|---|-----|------------------|")
    for _, row in sector[
        (sector["grouping"] == "sector_group")
        & (sector["model"] == "market")
    ].iterrows():
        add(
            f"| {row['model']} | [0,+{int(row['event_window'])}] | {row['group']} | {int(row['n_periods'])} | "
            f"{row['mean_car']:.6f} | {row['t_stat']:.3f} | {row['p_value']:.4f} | "
            f"[{row['boot_ci_low']:.6f}, {row['boot_ci_high']:.6f}] |"
        )
    add("")
    add("### 7.2 beta 分组")
    add("")
    add("| 模型 | 事件窗 | 组 | 期数 | CAR 均值 | t | p | Bootstrap 95% CI |")
    add("|------|--------|----|------|----------|---|-----|------------------|")
    for _, row in beta[
        (beta["grouping"] == "beta_group")
        & (beta["model"] == "market")
    ].iterrows():
        add(
            f"| {row['model']} | [0,+{int(row['event_window'])}] | {row['group']} | {int(row['n_periods'])} | "
            f"{row['mean_car']:.6f} | {row['t_stat']:.3f} | {row['p_value']:.4f} | "
            f"[{row['boot_ci_low']:.6f}, {row['boot_ci_high']:.6f}] |"
        )
    add("")
    add("### 7.3 行业平均 CAR 排序（市场模型，[0,+5]，描述性）")
    add("")
    add("| 行业 | 板块分组 | 平均 CAR | 期数 | t | p |")
    add("|------|----------|----------|------|---|-----|")
    top = industry[(industry["model"] == "market") & (industry["event_window"] == 5)].sort_values(
        "mean_car", ascending=False
    )
    for _, row in top.iterrows():
        add(
            f"| {row['code']} {row['name']} | {row['sector_group']} | {row['mean_car']:.6f} | "
            f"{int(row['n_periods'])} | {row['t_stat']:.3f} | {row['p_value']:.4f} |"
        )
    add("")
    add("- 行业排序仅为描述性结果，用于提出后续可检验的分组假设，不构成因果结论。")
    add("")
    add("## 8. 事件窗一致性")
    add("")
    add("| 模型 | 组 | 窗 1 | 窗 2 | 期数 | 符号一致率 | Spearman rho |")
    add("|------|----|------|------|------|------------|--------------|")
    for _, row in consistency.iterrows():
        add(
            f"| {row['model']} | {row['group']} | [0,+{int(row['window_left'])}] | "
            f"[0,+{int(row['window_right'])}] | {int(row['n_periods'])} | "
            f"{row['sign_agreement']:.3f} | {row['spearman_rho']:.3f} |"
        )
    add("")
    add("- 一致性表格用于判断结果是否对事件窗敏感；若符号一致率低或 rho 接近 0，在结论中明确说明。")
    add("")
    add("## 9. 子样本对比")
    add("")
    add("| 子样本 | 模型 | 事件窗 | 组 | 期数 | CAR 均值 | t | p | Bootstrap 95% CI |")
    add("|--------|------|--------|----|------|----------|---|-----|------------------|")
    for _, row in subsample.iterrows():
        add(
            f"| {row['subsample']} | {row['model']} | [0,+{int(row['event_window'])}] | {row['group']} | "
            f"{int(row['n_periods'])} | {row['mean_car']:.6f} | {row['t_stat']:.3f} | "
            f"{row['p_value']:.4f} | [{row['boot_ci_low']:.6f}, {row['boot_ci_high']:.6f}] |"
        )
    add("")
    add("## 10. 稳健性")
    add("")
    add("| 稳健性 | 模型 | 词典 | 估计窗 | 事件窗 | 组 | 期数 | CAR 均值 | t | p | Bootstrap 95% CI |")
    add("|--------|------|------|--------|--------|----|------|----------|---|-----|------------------|")
    for _, row in robustness.iterrows():
        add(
            f"| {row['robustness']} | {row['model']} | {row['dictionary']} | {int(row['estimation_window'])} | "
            f"[0,+{int(row['event_window'])}] | {row['group']} | {int(row['n_periods'])} | "
            f"{row['mean_car']:.6f} | {row['t_stat']:.3f} | {row['p_value']:.4f} | "
            f"[{row['boot_ci_low']:.6f}, {row['boot_ci_high']:.6f}] |"
        )
    add("")
    add("## 11. 估计窗与上一期事件的重叠检查")
    add("")
    if len(observations):
        contains = observations[observations["estimation_window"] == MAIN_EST][
            "est_window_contains_prior_event"
        ]
        add(
            f"- 主估计窗（120 日）内包含上一期 MPR 发布日的 (期, 行业) 单元占比：{contains.mean():.2%}。"
        )
        add("- 该重叠是季度 MPR 间隔较短所致，已用 90 / 180 / 250 日估计窗做稳健性（见上表）。")
    add("")
    add("## 12. 限制与说明")
    add("")
    add("- 基准沪深300 自 2002-01-04 起，2001 年部分 MPR 期次无法进入市场模型样本。")
    add("- 申万一级行业指数起点不同：16 个行业自 1999-12-30，14 个行业自 2014-02-21（其中 3 个自 2021-12-13）；早期事件横截面行业数较少，`long16` 子样本给出长样本行业结果。")
    add("- 估计窗要求窗口内每个交易日都有行业数据；缺失或不足会剔除该 (期, 行业, 窗) 单元，明细见 `_temp/e3/e3_skip_cells.csv`。")
    add("- `post2014` 子样本低情绪组仅 6 期（三分位按全样本 103 期划分），检验力低，仅作参考。")
    add("- 污染清单基于公开历史资料整理，可能不全；剔除污染事件的结果作为稳健性，不与主结果混用。")
    add("- 行业分组（板块映射、全样本 beta 中位数）为描述性分组，行业组间差异不写成因果结论。")
    add("- 显著与不显著结果均如实记录；若结果对事件窗敏感，以事件窗一致性表格为准并明确说明。")
    add("")
    add("## 13. 完成条件核对")
    add("")
    add("- 代码可复现：`python experiments/E3/run_e3.py`，随机种子与脚本哈希已记录。")
    add("- 全样本与 31 个行业：主结果覆盖 31 个申万一级行业；早期事件行业数较少已在明细中体现。")
    add("- 分组规则事先确定：NTUSD 简体三分位，脚本内固定，未按结果调整。")
    add("- 同一事件所有行业共用市场模型估计窗：估计窗长度 120，对同事件所有行业一致。")
    add("- 污染检查：已报告命中清单与剔除污染后的稳健性。")
    add(f"- 全样本与分组 CAR 显著性：{metrics['significance_note']}")
    add("- 显著与不显著均报告；行业分组结果为描述性，不作因果解释；事件窗敏感性已单列一致性表。")
    add("")

    out = LOG_DIR / "e3_event_study.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    log(f"[log] wrote {out}")


def main() -> int:
    ensure_dirs()
    (TEMP_DIR / "e3_run.log").write_text("", encoding="utf-8")
    log("E3 formal event-study start")

    environment = {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }

    industry, bench, calendar, asset_rows = load_prices()
    log(
        f"[data] industry rows={len(industry)}, industries={industry['code'].nunique()}, "
        f"benchmark rows={len(bench)}, calendar days={len(calendar)}"
    )
    rf = fetch_rf()
    if rf is not None:
        log(f"[data] risk-free rows={len(rf)}, range {rf['date'].min()}..{rf['date'].max()}")
    else:
        log("[data] risk-free rate unavailable; excess-return robustness skipped")

    beta_full = full_sample_betas(industry, bench)
    beta_values = np.array([value for value in beta_full.values() if not np.isnan(value)])
    beta_median = float(np.median(beta_values))
    beta_group = {
        code: ("high_beta" if value >= beta_median else "low_beta")
        for code, value in beta_full.items()
        if not np.isnan(value)
    }
    log(f"[data] full-sample beta median={beta_median:.4f}, high={sum(1 for v in beta_group.values() if v == 'high_beta')}, low={sum(1 for v in beta_group.values() if v == 'low_beta')}")

    events = load_events()
    events = align_events(events, calendar)
    events = assign_groups(events, MAIN_DICT)
    events = pollution_flags(events, calendar)
    events.to_csv(TEMP_DIR / "e3_events_aligned.csv", index=False, encoding="utf-8-sig")
    log(
        f"[events] issues={len(events)}, aligned={int(events['event_idx'].notna().sum())}, "
        f"polluted={int(events['polluted'].sum())}"
    )

    observations, period_car, skip_rows = build_observations(industry, bench, calendar, events, rf)
    observations.to_csv(TEMP_DIR / "e3_observations.csv", index=False, encoding="utf-8-sig")
    period_car.to_csv(TEMP_DIR / "e3_period_car.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(skip_rows).to_csv(TEMP_DIR / "e3_skip_cells.csv", index=False, encoding="utf-8-sig")
    log(
        f"[obs] observation rows={len(observations)}, period rows={len(period_car)}, "
        f"skipped cells={len(skip_rows)}"
    )

    sentiment = sentiment_summary(period_car, events)
    sentiment.to_csv(E3_DIR / "e3_car_by_sentiment.csv", index=False, encoding="utf-8-sig")
    log(f"[summary] sentiment rows={len(sentiment)}")

    car = car_long(observations)
    industry_df = industry_car(car, asset_rows)
    industry_df.to_csv(E3_DIR / "e3_industry_car.csv", index=False, encoding="utf-8-sig")
    sector_df = industry_group_summary(car, asset_rows, beta_group)
    sector_df.to_csv(E3_DIR / "e3_industry_group_car.csv", index=False, encoding="utf-8-sig")
    log(f"[summary] industry rows={len(industry_df)}, industry-group rows={len(sector_df)}")

    consistency = window_consistency(period_car, events)
    subsample = subsample_summary(period_car, events, observations)
    robustness = robustness_summary(period_car, observations, events)
    robustness.to_csv(E3_DIR / "e3_robustness.csv", index=False, encoding="utf-8-sig")
    consistency.to_csv(TEMP_DIR / "e3_window_consistency.csv", index=False, encoding="utf-8-sig")
    subsample.to_csv(TEMP_DIR / "e3_subsample.csv", index=False, encoding="utf-8-sig")

    # Event-level result table: main estimation window only, one row per
    # (issue, industry, model, event_window).
    event_results = observations[observations["estimation_window"] == MAIN_EST].copy()
    event_results["sector_group"] = event_results["code"].map(SECTOR_GROUPS)
    event_results["beta_group"] = event_results["code"].map(beta_group)
    event_results["full_sample_beta"] = event_results["code"].map(beta_full)
    event_results["car"] = np.where(
        event_results["model"] == "mean_adj",
        event_results["car_mean_adj"],
        event_results["car_market"],
    )
    event_out = event_results[
        [
            "issue", "report_period", "publication_date", "event_trading_date",
            "code", "name", "sector_group", "beta_group", "full_sample_beta",
            "sentiment_group", "sentiment_ntusd_simplified", "polluted", "pollution_hits",
            "estimation_window", "event_window", "model", "alpha", "beta",
            "car_market", "car_mean_adj", "car", "n_days",
        ]
    ].sort_values(["issue", "code", "model", "event_window"])
    event_out.to_csv(E3_DIR / "e3_event_results.csv", index=False, encoding="utf-8-sig")
    log(f"[output] e3_event_results.csv rows={len(event_out)}")

    sector_market = sector_df[sector_df["grouping"] == "sector_group"]
    beta_out = sector_df[sector_df["grouping"] == "beta_group"]
    make_plot(sentiment, sector_market, beta_out, industry_df)

    def sig_note(model: str, group: str) -> str:
        sub = sentiment[
            (sentiment["model"] == model) & (sentiment["group"] == group)
        ]
        hits = [
            f"[0,+{int(row.event_window)}] p={row.p_value:.4f}"
            for row in sub.itertuples()
            if row.p_value < 0.05
        ]
        return "、".join(hits) if hits else "无"

    base_market = sentiment[
        (sentiment["model"] == "market") & (sentiment["group"] == "all")
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
        + "；超额市场模型全样本显著事件窗："
        + sig_note("excess_market", "all")
        + "。其余事件窗/组均不显著，如实记录，不做因果解释。"
    )

    metrics = {
        "industry_rows": int(len(industry)),
        "industry_count": int(industry["code"].nunique()),
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
        "event_result_rows": int(len(event_out)),
        "sentiment_summary_rows": int(len(sentiment)),
        "industry_rows_summary": int(len(industry_df)),
        "industry_group_rows": int(len(sector_df)),
        "polluted_issues": int(events["polluted"].sum()),
        "main_periods": main_periods,
        "beta_median": beta_median,
        "significance_note": significance_note,
        "script_sha256": sha256(Path(__file__)),
        "seed": SEED,
        "n_boot": N_BOOT,
    }
    (E3_DIR / "e3_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    write_log(
        environment,
        asset_rows,
        events,
        observations,
        period_car,
        sentiment,
        sector_market,
        beta_out,
        industry_df,
        consistency,
        subsample,
        robustness,
        skip_rows,
        rf,
        beta_full,
        beta_group,
        metrics,
    )
    log("E3 done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
