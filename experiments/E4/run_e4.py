"""E4 cross-sectional prediction and Fama-MacBeth experiment.

The experiment tests whether MPR dictionary sentiment adds cross-sectional
predictive information for next-quarter Shenwan level-1 industry returns.

Identification note
-------------------
MPR sentiment is one common quarterly score.  In a cross-sectional
regression with an intercept, that common score is collinear with the
intercept and has no cross-sectional identifying variation.  The main
specification therefore uses pre-registered industry-sector interactions
(sentiment x sector) plus lagged industry characteristics.  A panel
robustness section reports industry-fixed-effects and time-fixed-effects
results and makes the common-sentiment absorption explicit.

Outputs land in experiments/E4.  Cache files land in _temp/e4.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
E4_DIR = ROOT / "experiments" / "E4"
LOG_DIR = E4_DIR / "logs"
TEMP_DIR = ROOT / "_temp" / "e4"
MACRO_RAW_DIR = TEMP_DIR / "macro_raw"

INDUSTRY_PATH = ROOT / "data" / "external" / "sw_industries" / "sw_industry_daily.csv"
BENCH_PATH = ROOT / "data" / "external" / "benchmark" / "sh000300_沪深300.csv"
BENCH_CACHE_PATH = ROOT / "_temp" / "e0_market_raw" / "benchmark_sh000300.csv"
MPR_INDEX_PATH = ROOT / "experiments" / "E1" / "mpr_index_full.csv"
SENTIMENT_PATH = ROOT / "experiments" / "E1" / "sentiment_dictionary_full.csv"

DICTS = ["ntusd_simplified", "dlut", "mpr_domain"]
PRIMARY_DICT = "ntusd_simplified"
SEED = 20261119
NW_LAG = 2
MIN_DAYS_QUARTER = 40
MIN_DAYS_BETA = 60
BETA_WINDOW = 120
VOL_WINDOW = 60

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


def ensure_dirs() -> None:
    for path in [E4_DIR, LOG_DIR, TEMP_DIR, MACRO_RAW_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def log(line: str = "") -> None:
    with (TEMP_DIR / "e4_run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
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


def ensure_macro_raw() -> None:
    """Use cached akshare macro files, or fetch the small required set."""
    required = [
        "macro_china_cpi.csv",
        "macro_china_money_supply.csv",
        "macro_china_pmi.csv",
        "macro_china_gdp.csv",
        "macro_china_society_electricity.csv",
        "macro_china_shibor_all.csv",
    ]
    if all((MACRO_RAW_DIR / name).exists() for name in required):
        return
    import akshare as ak

    specs = {
        "macro_china_cpi.csv": "macro_china_cpi",
        "macro_china_money_supply.csv": "macro_china_money_supply",
        "macro_china_pmi.csv": "macro_china_pmi",
        "macro_china_gdp.csv": "macro_china_gdp",
        "macro_china_society_electricity.csv": "macro_china_society_electricity",
        "macro_china_shibor_all.csv": "macro_china_shibor_all",
    }
    for filename, func_name in specs.items():
        target = MACRO_RAW_DIR / filename
        if target.exists() and target.stat().st_size > 0:
            continue
        frame = getattr(ak, func_name)()
        frame.to_csv(target, index=False, encoding="utf-8-sig")
        log(f"[macro] fetched {func_name} rows={len(frame)}")


def parse_cn_month(value: Any) -> pd.Timestamp | None:
    text = str(value)
    match = re.search(r"(\d{4})\D{1,3}(\d{1,2})", text)
    if not match:
        return None
    year, month = int(match.group(1)), int(match.group(2))
    if not 1 <= month <= 12:
        return None
    return pd.Timestamp(year=year, month=month, day=1)


def parse_cn_quarter(value: Any) -> pd.Timestamp | None:
    text = str(value)
    match = re.search(r"(\d{4})\D{1,3}(\d{1,2})(?:\D{1,3}(\d{1,2}))?", text)
    if not match:
        return None
    year = int(match.group(1))
    first = int(match.group(2))
    last = int(match.group(3)) if match.group(3) else first
    month = max(3, min(12, last * 3))
    if first == 1 and last == 3:
        month = 3
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def parse_dotted_month(value: Any) -> pd.Timestamp | None:
    text = str(value).strip()
    match = re.search(r"(\d{4})\.(\d{1,2})(?:\.\d+)?", text)
    if not match:
        return None
    year, month = int(match.group(1)), int(match.group(2))
    if not 1 <= month <= 12:
        return None
    return pd.Timestamp(year=year, month=month, day=1)


def asof_monthly(
    series: pd.DataFrame,
    value_date: pd.Timestamp,
    release_day: int | None = None,
) -> float | None:
    """Latest statistical-month value whose release date is <= value_date.

    `series["date"]` is the first day of the statistical month.  When
    `release_day` is given, the release date is that day in the following
    month.  When it is None, the value is treated as released at month end.
    """
    frame = series.copy()
    if release_day is None:
        frame["release_date"] = frame["date"] + pd.offsets.MonthEnd(0)
    else:
        frame["release_date"] = (
            frame["date"]
            + pd.offsets.MonthBegin(1)
            + pd.Timedelta(days=release_day - 1)
        )
    eligible = frame[frame["release_date"] <= value_date]
    if eligible.empty:
        return None
    return float(eligible.sort_values("date").iloc[-1]["value"])


def asof_daily(series: pd.DataFrame, value_date: pd.Timestamp) -> float | None:
    eligible = series[series["date"] <= value_date]
    if eligible.empty:
        return None
    return float(eligible.sort_values("date").iloc[-1]["value"])


def load_macro() -> pd.DataFrame:
    """Build issue-level macro controls available at MPR publication dates."""
    ensure_macro_raw()

    cpi = pd.read_csv(MACRO_RAW_DIR / "macro_china_cpi.csv", encoding="utf-8-sig")
    cpi = cpi.iloc[:, [0, 2]].copy()
    cpi.columns = ["month", "value"]
    cpi["date"] = cpi["month"].map(parse_cn_month)
    cpi = cpi.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)

    m2 = pd.read_csv(MACRO_RAW_DIR / "macro_china_money_supply.csv", encoding="utf-8-sig")
    m2 = m2.iloc[:, [0, 2]].copy()
    m2.columns = ["month", "value"]
    m2["date"] = m2["month"].map(parse_cn_month)
    m2 = m2.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)

    pmi = pd.read_csv(MACRO_RAW_DIR / "macro_china_pmi.csv", encoding="utf-8-sig")
    pmi = pmi.iloc[:, [0, 1]].copy()
    pmi.columns = ["month", "value"]
    pmi["date"] = pmi["month"].map(parse_cn_month)
    pmi = pmi.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)

    gdp = pd.read_csv(MACRO_RAW_DIR / "macro_china_gdp.csv", encoding="utf-8-sig")
    gdp = gdp.iloc[:, [0, 2]].copy()
    gdp.columns = ["quarter", "value"]
    gdp["date"] = gdp["quarter"].map(parse_cn_quarter)
    gdp = gdp.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)

    elec = pd.read_csv(MACRO_RAW_DIR / "macro_china_society_electricity.csv", encoding="utf-8-sig")
    elec = elec.iloc[:, [0, 2]].copy()
    elec.columns = ["month", "value"]
    elec["date"] = elec["month"].map(parse_dotted_month)
    elec = elec.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)

    shibor = pd.read_csv(MACRO_RAW_DIR / "macro_china_shibor_all.csv", encoding="utf-8-sig")
    shibor = shibor.iloc[:, [0, 9]].copy()
    shibor.columns = ["date", "value"]
    shibor["date"] = pd.to_datetime(shibor["date"], errors="coerce")
    shibor = shibor.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)

    mpr = pd.read_csv(MPR_INDEX_PATH, dtype={"issue": str})
    mpr["publication_date"] = pd.to_datetime(mpr["publication_date"])

    rows: list[dict[str, Any]] = []
    for _, issue in mpr.iterrows():
        pub = issue["publication_date"]
        row: dict[str, Any] = {"issue": issue["issue"], "publication_date": pub}
        row["cpi_yoy"] = asof_monthly(cpi, pub, release_day=9)
        row["m2_yoy"] = asof_monthly(m2, pub, release_day=15)
        row["pmi"] = asof_monthly(pmi, pub, release_day=None)
        row["gdp_yoy"] = asof_daily(gdp, pub)
        row["electricity_yoy"] = asof_monthly(elec, pub, release_day=15)
        row["shibor_3m"] = asof_daily(shibor, pub)

        cpi_3m = asof_monthly(cpi, pub - pd.offsets.MonthBegin(3), release_day=9)
        m2_3m = asof_monthly(m2, pub - pd.offsets.MonthBegin(3), release_day=15)
        pmi_3m = asof_monthly(pmi, pub - pd.offsets.MonthBegin(3), release_day=None)
        elec_3m = asof_monthly(elec, pub - pd.offsets.MonthBegin(3), release_day=15)
        shibor_3m_prior = asof_daily(shibor, pub - pd.offsets.MonthBegin(3))

        row["cpi_chg"] = row["cpi_yoy"] - cpi_3m if row["cpi_yoy"] is not None and cpi_3m is not None else None
        row["m2_chg"] = row["m2_yoy"] - m2_3m if row["m2_yoy"] is not None and m2_3m is not None else None
        row["pmi_chg"] = row["pmi"] - pmi_3m if row["pmi"] is not None and pmi_3m is not None else None
        row["electricity_chg"] = (
            row["electricity_yoy"] - elec_3m
            if row["electricity_yoy"] is not None and elec_3m is not None
            else None
        )
        row["shibor_chg"] = (
            row["shibor_3m"] - shibor_3m_prior
            if row["shibor_3m"] is not None and shibor_3m_prior is not None
            else None
        )
        rows.append(row)

    macro = pd.DataFrame(rows).sort_values("publication_date").reset_index(drop=True)
    change_cols = ["cpi_chg", "m2_chg", "pmi_chg", "electricity_chg", "shibor_chg"]
    macro["macro_pc1"] = macro[change_cols].mean(axis=1, skipna=True)
    macro["macro_components_available"] = macro[change_cols].notna().sum(axis=1)

    # Trailing z-scores avoid using future cross-time means.  The current
    # observation is included because it is known at the forecast origin.
    for column in change_cols + ["macro_pc1"]:
        values = macro[column].astype(float)
        rolling = values.expanding(min_periods=8).mean()
        rolling_std = values.expanding(min_periods=8).std(ddof=1)
        macro[f"{column}_z"] = (values - rolling) / rolling_std

    macro.to_csv(TEMP_DIR / "macro_issue_level.csv", index=False, encoding="utf-8-sig")
    return macro


def load_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
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

    bench_path = BENCH_CACHE_PATH if BENCH_CACHE_PATH.exists() and BENCH_CACHE_PATH.stat().st_size > 0 else BENCH_PATH
    bench = pd.read_csv(bench_path, dtype={"code": str})
    bench["date"] = pd.to_datetime(bench["date"])
    bench = bench.sort_values(["code", "date"]).reset_index(drop=True)
    bench["daily_return"] = bench.groupby("code", group_keys=False)["close"].pct_change()
    return industry, bench


def quarter_label(date: pd.Timestamp) -> str:
    return f"{date.year}Q{date.quarter}"


def next_quarter_label(label: str) -> str:
    year, q = label.split("Q")
    year, q = int(year), int(q)
    if q == 4:
        return f"{year + 1}Q1"
    return f"{year}Q{q + 1}"


def quarter_end(label: str) -> pd.Timestamp:
    year, q = label.split("Q")
    return pd.Timestamp(year=int(year), month=int(q) * 3, day=1) + pd.offsets.MonthEnd(0)


def build_industry_quarterly(industry: pd.DataFrame) -> pd.DataFrame:
    frame = industry.copy()
    frame["quarter"] = frame["date"].dt.year.astype(str) + "Q" + frame["date"].dt.quarter.astype(str)
    quarterly = (
        frame.groupby(["code", "quarter"], sort=True)
        .agg(
            name=("name", "last"),
            sector_group=("sector_group", "last"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            close=("close", "last"),
            n_days=("date", "size"),
            daily_ret_std=("daily_return", "std"),
        )
        .reset_index()
    )
    quarterly["log_return"] = quarterly.groupby("code")["close"].transform(lambda s: np.log(s).diff())
    quarterly["quarter_end"] = quarterly["quarter"].map(quarter_end)
    quarterly.to_csv(TEMP_DIR / "industry_quarterly.csv", index=False, encoding="utf-8-sig")
    return quarterly


def trailing_beta(
    industry_daily: pd.DataFrame,
    bench_daily: pd.DataFrame,
    code: str,
    end_date: pd.Timestamp,
) -> float | None:
    ind = industry_daily[(industry_daily["code"] == code) & (industry_daily["date"] <= end_date)]
    mkt = bench_daily[(bench_daily["code"] == "sh000300") & (bench_daily["date"] <= end_date)]
    merged = ind[["date", "daily_return"]].merge(
        mkt[["date", "daily_return"]], on="date", how="inner", suffixes=("_i", "_m")
    ).dropna()
    if len(merged) < MIN_DAYS_BETA:
        return None
    merged = merged.tail(BETA_WINDOW)
    ri = merged["daily_return_i"].to_numpy(dtype=float)
    rm = merged["daily_return_m"].to_numpy(dtype=float)
    variance = float(np.var(rm, ddof=1))
    if variance <= 0:
        return None
    return float(np.cov(ri, rm, ddof=1)[0, 1] / variance)


def trailing_vol(
    industry_daily: pd.DataFrame,
    code: str,
    end_date: pd.Timestamp,
) -> float | None:
    frame = industry_daily[
        (industry_daily["code"] == code) & (industry_daily["date"] <= end_date)
    ].tail(VOL_WINDOW)
    if len(frame) < MIN_DAYS_BETA:
        return None
    return float(frame["daily_return"].std(ddof=1) * np.sqrt(252))


def build_panel(industry: pd.DataFrame, bench: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    quarterly = build_industry_quarterly(industry)
    mpr = pd.read_csv(MPR_INDEX_PATH, dtype={"issue": str})
    mpr["publication_date"] = pd.to_datetime(mpr["publication_date"])
    sent = pd.read_csv(SENTIMENT_PATH, dtype={"issue": str})
    sent = sent[["issue", "dictionary", "sentiment_score", "weighted_score"]]
    wide = sent.pivot(index="issue", columns="dictionary", values="sentiment_score")
    wide.columns = [f"sentiment_{column}" for column in wide.columns]
    wide = wide.reset_index()
    events = mpr.merge(wide, on="issue", how="left")
    events = events.merge(
        macro.drop(columns=["publication_date"]), on="issue", how="left"
    )
    events["target_quarter"] = events["report_period"].map(next_quarter_label)

    industry_daily = industry[["code", "date", "daily_return"]].copy()
    bench_daily = bench[["code", "date", "daily_return"]].copy()

    # Rolling industry exposure to the common MPR sentiment factor.  The
    # exposure is estimated only from quarters up to the report period, so it
    # is available at the MPR publication date.  A common factor interaction
    # is not identified in a single cross-section when the exposure main
    # effect is also included; the FMB sentiment model therefore uses this
    # exposure signal without adding the exposure main effect.
    returns_matrix = quarterly.pivot(index="quarter", columns="code", values="log_return").sort_index()
    sentiment_by_q = events.set_index("report_period")[
        [f"sentiment_{dictionary}" for dictionary in DICTS]
    ]
    exposure: dict[tuple[str, str, str], float | None] = {}
    for issue, report_q in events[["issue", "report_period"]].drop_duplicates().itertuples(index=False):
        for code in returns_matrix.columns:
            r = returns_matrix[code].dropna()
            r = r.loc[:report_q]
            if len(r) < 8:
                for dictionary in DICTS:
                    exposure[(issue, code, dictionary)] = None
                continue
            for dictionary in DICTS:
                s = sentiment_by_q[f"sentiment_{dictionary}"].reindex(r.index).astype(float)
                aligned = pd.DataFrame({"r": r.to_numpy(), "s": s.to_numpy()}, index=r.index).dropna().tail(20)
                if len(aligned) < 8 or float(aligned["s"].var(ddof=1)) <= 0:
                    exposure[(issue, code, dictionary)] = None
                    continue
                cov = float(np.cov(aligned["r"].to_numpy(), aligned["s"].to_numpy(), ddof=1)[0, 1])
                var = float(np.var(aligned["s"].to_numpy(), ddof=1))
                exposure[(issue, code, dictionary)] = cov / var if var > 0 else None

    quarterly_by_code = {code: frame.set_index("quarter") for code, frame in quarterly.groupby("code")}
    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        target_q = event["target_quarter"]
        report_q = event["report_period"]
        prior_q_obj = pd.Period(report_q, freq="Q") - 1
        prior_q = f"{prior_q_obj.year}Q{prior_q_obj.quarter}"
        pub = event["publication_date"]
        for code, frame in quarterly_by_code.items():
            if target_q not in frame.index or report_q not in frame.index:
                continue
            target = frame.loc[target_q]
            report = frame.loc[report_q]
            if not isinstance(target, pd.Series) or not isinstance(report, pd.Series):
                continue
            if int(target["n_days"]) < MIN_DAYS_QUARTER or int(report["n_days"]) < MIN_DAYS_QUARTER:
                continue
            if target["last_date"] < target["quarter_end"] - pd.Timedelta(days=7):
                continue
            mom1 = float(report["log_return"]) if pd.notna(report["log_return"]) else None
            vol_q = (
                float(report["daily_ret_std"] * np.sqrt(252))
                if pd.notna(report["daily_ret_std"])
                else None
            )
            mom2 = None
            if prior_q in frame.index:
                prior = frame.loc[prior_q]
                if isinstance(prior, pd.Series) and pd.notna(prior["close"]) and pd.notna(report["close"]):
                    mom2 = float(np.log(report["close"] / prior["close"]))
            beta = trailing_beta(industry_daily, bench_daily, code, pub)
            vol60 = trailing_vol(industry_daily, code, pub)
            row: dict[str, Any] = {
                "issue": event["issue"],
                "report_period": report_q,
                "target_quarter": target_q,
                "publication_date": pub,
                "code": code,
                "name": target["name"],
                "sector_group": target["sector_group"],
                "return_next_q": float(target["log_return"]),
                "mom1": mom1,
                "mom2": mom2,
                "vol_q": vol_q,
                "vol_60": vol60,
                "beta_120": beta,
                "n_days_target": int(target["n_days"]),
            }
            for dictionary in DICTS:
                row[f"sentiment_{dictionary}"] = event.get(f"sentiment_{dictionary}")
                row[f"sentiment_{dictionary}_raw"] = event.get(f"sentiment_{dictionary}")
                row[f"sent_beta_{dictionary}"] = exposure.get((event["issue"], code, dictionary))
            for column in [
                "cpi_yoy", "m2_yoy", "pmi", "gdp_yoy", "electricity_yoy", "shibor_3m",
                "cpi_chg", "m2_chg", "pmi_chg", "electricity_chg", "shibor_chg",
                "macro_pc1", "macro_pc1_z", "macro_components_available",
            ]:
                row[column] = event.get(column)
            rows.append(row)

    panel = pd.DataFrame(rows)
    panel["target_date"] = pd.to_datetime(panel["target_quarter"].map(quarter_end))
    panel = panel.sort_values(["target_quarter", "code"]).reset_index(drop=True)

    # Cross-sectional z-scores for continuous industry characteristics.
    for column in ["mom1", "mom2", "vol_q", "vol_60", "beta_120"]:
        group_mean = panel.groupby("target_quarter")[column].transform("mean")
        group_std = panel.groupby("target_quarter")[column].transform("std")
        panel[f"{column}_z"] = (panel[column] - group_mean) / group_std
    for dictionary in DICTS:
        column = f"sent_beta_{dictionary}"
        group_mean = panel.groupby("target_quarter")[column].transform("mean")
        group_std = panel.groupby("target_quarter")[column].transform("std")
        panel[f"{column}_z"] = (panel[column] - group_mean) / group_std

    # Trailing z-scores for sentiment and macro are computed on issue-level
    # observations, not on the duplicated industry panel, because both are
    # common within a quarter.
    issue_level = (
        panel.drop_duplicates("issue")[["issue", "publication_date"] + [f"sentiment_{d}" for d in DICTS] + ["macro_pc1"]]
        .sort_values("publication_date")
        .reset_index(drop=True)
    )
    for dictionary in DICTS:
        column = f"sentiment_{dictionary}"
        values = issue_level[column].astype(float)
        rolling_mean = values.expanding(min_periods=8).mean()
        rolling_std = values.expanding(min_periods=8).std(ddof=1)
        issue_level[f"{column}_z"] = (values - rolling_mean) / rolling_std
    macro_values = issue_level["macro_pc1"].astype(float)
    macro_mean = macro_values.expanding(min_periods=8).mean()
    macro_std = macro_values.expanding(min_periods=8).std(ddof=1)
    issue_level["macro_pc1_trail_z"] = (macro_values - macro_mean) / macro_std
    z_map = issue_level.set_index("issue")[
        [f"sentiment_{d}_z" for d in DICTS] + ["macro_pc1_trail_z"]
    ]
    panel = panel.join(z_map, on="issue", how="left")

    # Sector dummies and interactions.
    for sector in SECTOR_ORDER[1:]:
        panel[f"sector_{SECTOR_ORDER.index(sector)}"] = (
            panel["sector_group"] == sector
        ).astype(float)
    for dictionary in DICTS:
        sent_z = panel[f"sentiment_{dictionary}_z"]
        for sector in SECTOR_ORDER[1:]:
            panel[f"sent_x_{sector}_{dictionary}"] = sent_z * panel[f"sector_{SECTOR_ORDER.index(sector)}"]
        panel[f"sent_x_mom1_{dictionary}"] = sent_z * panel["mom1_z"]
        panel[f"sent_x_vol_{dictionary}"] = sent_z * panel["vol_q_z"]
        panel[f"sent_x_beta_{dictionary}"] = sent_z * panel["beta_120_z"]
        panel[f"sent_x_exposure_{dictionary}"] = sent_z * panel[f"sent_beta_{dictionary}_z"]
    panel["macro_x_beta"] = panel["macro_pc1_trail_z"] * panel["beta_120_z"]
    for sector in SECTOR_ORDER[1:]:
        panel[f"macro_x_sector_{SECTOR_ORDER.index(sector)}"] = (
            panel["macro_pc1_trail_z"] * panel[f"sector_{SECTOR_ORDER.index(sector)}"]
        )

    panel.to_csv(TEMP_DIR / "e4_panel.csv", index=False, encoding="utf-8-sig")
    return panel


def design_columns(spec: str, dictionary: str | None) -> list[str]:
    base = [
        "mom1_z",
        "vol_q_z",
        "beta_120_z",
        "sector_1",
        "sector_2",
        "sector_3",
        "sector_4",
    ]
    if spec == "baseline_notext":
        return base
    if dictionary is None:
        raise ValueError("sentiment spec requires dictionary")
    if spec == "sent_exposure":
        return base + [f"sent_x_exposure_{dictionary}"]
    raise ValueError(f"unknown spec: {spec}")


SPECS: list[tuple[str, str | None]] = [("baseline_notext", None)]
for _dict in DICTS:
    for _spec in ["sent_exposure"]:
        SPECS.append((_spec, _dict))


def cross_section_ols(
    panel: pd.DataFrame,
    spec: str,
    dictionary: str | None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run one OLS per target quarter and return coefficient rows."""
    columns = design_columns(spec, dictionary)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for target_q, frame in panel.groupby("target_quarter", sort=True):
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["return_next_q"] + columns)
        n = len(frame)
        k = len(columns) + 1
        if n < k + 2:
            skipped.append({"target_quarter": target_q, "reason": f"n_{n}_lt_k_{k}+2"})
            continue
        X = np.column_stack([np.ones(n)] + [frame[column].to_numpy(dtype=float) for column in columns])
        y = frame["return_next_q"].to_numpy(dtype=float)
        beta, _residuals, rank, _sv = np.linalg.lstsq(X, y, rcond=None)
        if rank < k:
            skipped.append({"target_quarter": target_q, "reason": f"rank_{rank}_lt_k_{k}"})
            continue
        resid = y - X @ beta
        dof = n - rank
        sigma2 = float(resid @ resid / dof)
        XtX_inv = np.linalg.pinv(X.T @ X)
        white_meat = X.T @ (X * (resid ** 2)[:, None])
        cov = XtX_inv @ white_meat @ XtX_inv
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        tss = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(resid @ resid) / tss if tss > 0 else float("nan")
        names = ["Intercept"] + columns
        for idx, name in enumerate(names):
            estimate = float(beta[idx])
            std_error = float(se[idx])
            t_value = estimate / std_error if std_error > 0 else float("nan")
            p_value = float(2 * stats.norm.sf(abs(t_value))) if np.isfinite(t_value) else float("nan")
            rows.append(
                {
                    "model": "baseline_notext" if spec == "baseline_notext" else "sentiment",
                    "dictionary": dictionary or "none",
                    "spec": spec,
                    "target_quarter": target_q,
                    "issue": frame["issue"].iloc[0],
                    "coefficient": name,
                    "estimate": estimate,
                    "std_error": std_error,
                    "t_value": t_value,
                    "p_value": p_value,
                    "n_industries": n,
                    "r2": r2,
                    "rank": int(rank),
                    "n_vars": k,
                }
            )
    coefficients = pd.DataFrame(rows)
    return coefficients, skipped


def nw_tstat(values: np.ndarray, lag: int = NW_LAG) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return float("nan"), float("nan")
    mu = float(np.mean(values))
    errors = values - mu
    gamma0 = float(np.mean(errors ** 2))
    variance = gamma0
    for ell in range(1, lag + 1):
        if ell >= len(errors):
            break
        weight = 1.0 - ell / (lag + 1.0)
        gamma = float(np.mean(errors[ell:] * errors[:-ell]))
        variance += 2.0 * weight * gamma
    variance /= len(values)
    if variance <= 0:
        return float("nan"), float("nan")
    t_value = mu / np.sqrt(variance)
    p_value = float(2 * stats.norm.sf(abs(t_value)))
    return float(t_value), p_value


def summarize_fmb(coefficients: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["model", "dictionary", "spec", "coefficient"]
    for keys, frame in coefficients.groupby(group_cols, sort=False):
        values = frame["estimate"].to_numpy(dtype=float)
        n = len(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n > 1 else float("nan")
        median = float(np.median(values))
        if n >= 10:
            lower, upper = np.percentile(values, [5, 95])
            winsorized_mean = float(np.mean(np.clip(values, lower, upper)))
        else:
            winsorized_mean = mean
        t_fmb = mean / (std / np.sqrt(n)) if n > 1 and std > 0 else float("nan")
        p_fmb = float(2 * stats.norm.sf(abs(t_fmb))) if np.isfinite(t_fmb) else float("nan")
        t_nw, p_nw = nw_tstat(values)
        hit_rate = float(np.mean(np.sign(values) == np.sign(mean))) if mean != 0 else float("nan")
        rows.append(
            {
                "model": keys[0],
                "dictionary": keys[1],
                "spec": keys[2],
                "coefficient": keys[3],
                "is_sentiment": int(keys[3].startswith("sent_x_")),
                "n_periods": n,
                "mean": mean,
                "median": median,
                "winsorized_mean": winsorized_mean,
                "std": std,
                "t_fmb": t_fmb,
                "p_fmb": p_fmb,
                "t_nw": t_nw,
                "p_nw": p_nw,
                "hit_rate": hit_rate,
                "mean_r2": float(frame["r2"].mean()),
                "mean_n": float(frame["n_industries"].mean()),
                "first_period": frame["target_quarter"].min(),
                "last_period": frame["target_quarter"].max(),
            }
        )
    return pd.DataFrame(rows)


def predict_oos(
    panel: pd.DataFrame,
    coefficients: pd.DataFrame,
    spec: str,
    dictionary: str | None,
    split: str,
    start_q: str,
    end_q: str,
    min_train_periods: int = 8,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = design_columns(spec, dictionary)
    coef_rows = coefficients[coefficients["spec"] == spec]
    if dictionary is not None:
        coef_rows = coef_rows[coef_rows["dictionary"] == dictionary]
    coef_rows = coef_rows[coef_rows["target_quarter"] < end_q]
    coef_means = coef_rows.groupby("coefficient")["estimate"].mean().to_dict()
    expected = ["Intercept"] + columns
    if any(name not in coef_means for name in expected):
        return pd.DataFrame(), {"skip_reason": "missing_coefficients"}

    forecast_rows: list[dict[str, Any]] = []
    test_panel = panel[(panel["target_quarter"] >= start_q) & (panel["target_quarter"] <= end_q)].copy()
    for target_q, frame in test_panel.groupby("target_quarter", sort=True):
        prior = coef_rows[coef_rows["target_quarter"] < target_q]
        if prior["target_quarter"].nunique() < min_train_periods:
            continue
        prior_means = prior.groupby("coefficient")["estimate"].mean().to_dict()
        if any(name not in prior_means for name in expected):
            continue
        clean = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["return_next_q"] + columns)
        if len(clean) < 8:
            continue
        X = np.column_stack([np.ones(len(clean))] + [clean[column].to_numpy(dtype=float) for column in columns])
        beta = np.array([prior_means[name] for name in expected], dtype=float)
        prediction = X @ beta
        for pos, (idx, row) in enumerate(clean.iterrows()):
            forecast_rows.append(
                {
                    "model": "baseline_notext" if spec == "baseline_notext" else "sentiment",
                    "dictionary": dictionary or "none",
                    "spec": spec,
                    "split": split,
                    "target_quarter": target_q,
                    "issue": row["issue"],
                    "code": row["code"],
                    "name": row["name"],
                    "sector_group": row["sector_group"],
                    "actual": row["return_next_q"],
                    "predicted": float(prediction[pos]),
                }
            )
    predictions = pd.DataFrame(forecast_rows)
    if predictions.empty:
        return predictions, {"skip_reason": "no_forecasts"}

    metrics = oos_metrics(predictions)
    metrics.update(
        {
            "model": "baseline_notext" if spec == "baseline_notext" else "sentiment",
            "dictionary": dictionary or "none",
            "spec": spec,
            "split": split,
            "start_period": start_q,
            "end_period": end_q,
            "n_periods": int(predictions["target_quarter"].nunique()),
        }
    )
    return predictions, metrics


def oos_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    y = predictions["actual"].to_numpy(dtype=float)
    yhat = predictions["predicted"].to_numpy(dtype=float)
    error = y - yhat
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    sse = float(np.sum(error ** 2))
    cs_mean = predictions.groupby("target_quarter")["actual"].transform("mean")
    sst_cs = float(np.sum((y - cs_mean.to_numpy(dtype=float)) ** 2))
    sst_overall = float(np.sum((y - y.mean()) ** 2))
    oos_r2 = 1.0 - sse / sst_cs if sst_cs > 0 else float("nan")
    oos_r2_overall = 1.0 - sse / sst_overall if sst_overall > 0 else float("nan")

    ic_values: list[float] = []
    pearson_values: list[float] = []
    for _q, frame in predictions.groupby("target_quarter", sort=True):
        if len(frame) < 3:
            continue
        rank_ic = stats.spearmanr(frame["predicted"], frame["actual"]).statistic
        pearson_ic = stats.pearsonr(frame["predicted"], frame["actual"]).statistic
        if np.isfinite(rank_ic):
            ic_values.append(float(rank_ic))
        if np.isfinite(pearson_ic):
            pearson_values.append(float(pearson_ic))
    ic = np.array(ic_values, dtype=float)
    ic_mean = float(np.mean(ic)) if len(ic) else float("nan")
    ic_std = float(np.std(ic, ddof=1)) if len(ic) > 1 else float("nan")
    icir = ic_mean / ic_std if ic_std and ic_std > 0 else float("nan")
    ic_t, ic_p = nw_tstat(ic)
    return {
        "n_obs": int(len(predictions)),
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": icir,
        "ic_t": ic_t,
        "ic_p": ic_p,
        "pearson_ic_mean": float(np.mean(pearson_values)) if pearson_values else float("nan"),
        "oos_r2": oos_r2,
        "oos_r2_overall": oos_r2_overall,
        "mae": mae,
        "rmse": rmse,
    }


def paired_mae_comparison(
    baseline: pd.DataFrame,
    model: pd.DataFrame,
) -> tuple[float, float]:
    base = baseline.groupby("target_quarter").apply(
        lambda frame: np.mean(np.abs(frame["actual"] - frame["predicted"])), include_groups=False
    )
    mod = model.groupby("target_quarter").apply(
        lambda frame: np.mean(np.abs(frame["actual"] - frame["predicted"])), include_groups=False
    )
    common = base.index.intersection(mod.index)
    if len(common) < 3:
        return float("nan"), float("nan")
    diff = base.loc[common].to_numpy(dtype=float) - mod.loc[common].to_numpy(dtype=float)
    t_value, p_value = nw_tstat(diff)
    return float(t_value), float(p_value)


def panel_robustness(panel: pd.DataFrame) -> pd.DataFrame:
    """Industry-FE panel with and without time FE.

    With time FE, a common sentiment score is absorbed.  The interaction
    terms remain identified and are the relevant robustness check.
    """
    import statsmodels.formula.api as smf

    rows: list[dict[str, Any]] = []
    specs = [
        ("panel_industry_fe", "return_next_q ~ sentiment_z + mom1_z + vol_q_z + beta_120_z + macro_pc1 + C(code)", "code"),
        ("panel_time_fe", "return_next_q ~ mom1_z + vol_q_z + beta_120_z + C(code) + C(issue)", "code"),
        ("panel_time_fe_interact", "return_next_q ~ sent_x_mom1 + sent_x_vol + sent_x_beta + mom1_z + vol_q_z + beta_120_z + C(code) + C(issue)", "code"),
    ]
    for dictionary in DICTS:
        frame = panel.copy()
        frame = frame.rename(
            columns={
                f"sentiment_{dictionary}_z": "sentiment_z",
                f"sent_x_mom1_{dictionary}": "sent_x_mom1",
                f"sent_x_vol_{dictionary}": "sent_x_vol",
                f"sent_x_beta_{dictionary}": "sent_x_beta",
            }
        )
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["return_next_q", "sentiment_z", "mom1_z", "vol_q_z", "beta_120_z", "macro_pc1"]
        )
        for spec_name, formula, cluster in specs:
            try:
                model = smf.ols(formula, data=frame).fit(
                    cov_type="cluster", cov_kwds={"groups": frame[cluster]}
                )
                params = model.params
                bse = model.bse
                for term in params.index:
                    if term not in ["sentiment_z", "sent_x_mom1", "sent_x_vol", "sent_x_beta"]:
                        continue
                    rows.append(
                        {
                            "dictionary": dictionary,
                            "spec": spec_name,
                            "term": term,
                            "estimate": float(params[term]),
                            "std_error": float(bse[term]),
                            "t_value": float(params[term] / bse[term]) if bse[term] > 0 else float("nan"),
                            "p_value": float(2 * stats.norm.sf(abs(params[term] / bse[term]))) if bse[term] > 0 else float("nan"),
                            "n_obs": int(model.nobs),
                            "r2": float(model.rsquared),
                            "cluster": cluster,
                        }
                    )
                if spec_name == "panel_time_fe":
                    rows.append(
                        {
                            "dictionary": dictionary,
                            "spec": spec_name,
                            "term": "sentiment_z",
                            "estimate": float("nan"),
                            "std_error": float("nan"),
                            "t_value": float("nan"),
                            "p_value": float("nan"),
                            "n_obs": int(model.nobs),
                            "r2": float(model.rsquared),
                            "cluster": "absorbed_by_time_fe",
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "dictionary": dictionary,
                        "spec": spec_name,
                        "term": "ERROR",
                        "estimate": float("nan"),
                        "std_error": float("nan"),
                        "t_value": float("nan"),
                        "p_value": float("nan"),
                        "n_obs": 0,
                        "r2": float("nan"),
                        "cluster": f"{type(exc).__name__}: {exc}",
                    }
                )
    return pd.DataFrame(rows)


def write_log(
    environment: dict[str, Any],
    panel: pd.DataFrame,
    fmb_results: pd.DataFrame,
    oos_metrics: pd.DataFrame,
    panel_results: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    lines: list[str] = []

    def add(line: str = "") -> None:
        lines.append(line)

    add("# E4 横截面预测与 Fama-MacBeth 实验日志")
    add()
    add("## 1. 实验目的与假设")
    add()
    add("- 检验 H2 的预测部分：MPR 情绪是否对下一期行业收益有横截面预测力，是否优于无文本 Baseline。")
    add("- 输入：行业季度收益、MPR 词典情绪、市场收益、滞后行业收益、波动率、宏观控制变量。")
    add("- 主词典：`ntusd_simplified`；`dlut` 与 `mpr_domain` 为稳健性对照。")
    add()
    add("## 2. 识别口径")
    add()
    add("- MPR 情绪是季度层面共同变量。若横截面回归含截距，共同情绪与截距完全共线，不能识别“情绪高低行业”的横截面差异。")
    add("- 共同宏观变量同样与截距共线，不能直接放入单期横截面回归。宏观控制改放在行业面板稳健性中，并另报共同宏观暴露诊断。")
    add("- FMB 主模型使用滚动估计的行业情绪暴露 `sent_beta`：仅用报告期及以前季度估计行业收益对 MPR 情绪的敏感度，再与当季 `sentiment_z` 相乘。该信号在截面上有行业差异，且不作为主效应进入 Baseline。")
    add("- 无文本 Baseline 使用滞后行业收益、波动率、滚动市场 beta、行业板块；情绪模型在同一 Baseline 上增加 `sentiment_z × sent_beta_z`。")
    add("- 面板稳健性分别报告行业固定效应与时间固定效应；时间固定效应会吸收共同情绪，但情绪与行业特征的交互项仍可识别。")
    add()
    add("## 3. 数据与样本")
    add()
    add(f"- 行业日行情：`data/external/sw_industries/sw_industry_daily.csv`，{metrics['industry_rows']} 行，{metrics['industry_count']} 个申万一级行业。")
    add(f"- 市场基准：{metrics['benchmark_source']}。")
    add(f"- MPR 指数与情绪：`experiments/E1/mpr_index_full.csv`、`experiments/E1/sentiment_dictionary_full.csv`，共 {metrics['mpr_issues']} 期。")
    add(f"- 宏观数据：akshare 公开接口，缓存于 `_temp/e4/macro_raw/`；按 MPR 发布日“最近可得值”对齐。")
    add(f"- 目标收益：MPR 报告期下一季度行业对数收益；要求目标季和报告期季均至少 {MIN_DAYS_QUARTER} 个交易日。")
    add(f"- 预测样本：{metrics['panel_first_quarter']} 至 {metrics['panel_last_quarter']}，共 {metrics['n_target_quarters']} 个目标季；行业截面数范围 {metrics['min_n_industries']} 至 {metrics['max_n_industries']}。")
    add(f"- 时间划分：训练 {metrics['train_start']}–{metrics['train_end']}，验证 {metrics['val_start']}–{metrics['val_end']}，测试 {metrics['test_start']}–{metrics['test_end']}。")
    add()
    add("## 4. 模型设定")
    add()
    add("- `baseline_notext`：截距 + `mom1_z` + `vol_q_z` + `beta_120_z` + 行业板块虚拟变量。")
    add("- `sent_exposure`：Baseline + `sentiment_z × sent_beta_z`。`sent_beta_z` 是滚动 20 个季度估计的行业收益对 MPR 情绪的暴露，再做截面标准化。")
    add("- 面板稳健性另报 `sentiment_z × mom1_z`、`sentiment_z × vol_q_z`、`sentiment_z × beta_120_z`，用于检验情绪是否调节行业特征斜率。")
    add("- 所有连续行业特征在季度截面上标准化；情绪与宏观指数使用时间Trailing z-score，避免跨期尺度漂移和前瞻。")
    add()
    add("## 5. Fama-MacBeth 结果")
    add()
    add("下表只列情绪相关系数；完整结果见 `e4_fmb_results.csv` 与 `e4_coefficient_series.csv`。")
    add()
    sentiment_summary = fmb_results[fmb_results["is_sentiment"] == 1].copy()
    if not sentiment_summary.empty:
        add("| 词典 | 设定 | 系数 | 均值 | 中位数 | 缩尾均值 | FMB t | NW t | 显著期数占比 | 平均R² |")
        add("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in sentiment_summary.itertuples():
            add(
                f"| {row.dictionary} | {row.spec} | {row.coefficient} | {row.mean:.5f} | "
                f"{row.median:.5f} | {row.winsorized_mean:.5f} | "
                f"{row.t_fmb:.2f} | {row.t_nw:.2f} | {row.hit_rate:.2f} | {row.mean_r2:.3f} |"
            )
    else:
        add("- 无情绪系数结果。")
    add()
    add("## 6. 样本外比较")
    add()
    add("- IC 为预测值与实际收益的横截面 Spearman 秩相关；ICIR = IC 均值 / IC 标准差。")
    add("- OOS R² 以每个目标季截面均值为基准；MAE/RMSE 为行业-季度面板误差。")
    add("- 样本外预测使用截至前一期的 expanding-window FMB 系数均值，不使用测试期信息。")
    add("- 情绪模型可能因滚动情绪暴露缺失而少若干行；相对 Baseline 的改进均按两者共同样本重算 Baseline 指标。")
    add()
    add("| 样本 | 模型 | 词典 | 设定 | 期数 | IC | ICIR | OOS R² | MAE | 相对Baseline MAE |")
    add("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in oos_metrics.itertuples():
        improvement = "—" if row.dictionary == "none" else f"{row.mae_improvement_pct:.2f}%"
        add(
            f"| {row.split} | {row.model} | {row.dictionary} | {row.spec} | {row.n_periods} | "
            f"{row.ic_mean:.4f} | {row.icir:.3f} | {row.oos_r2:.4f} | {row.mae:.5f} | {improvement} |"
        )
    add()
    add("## 7. 面板稳健性")
    add()
    if not panel_results.empty:
        add("| 词典 | 设定 | 项 | 系数 | 聚类t | N |")
        add("|---|---|---|---:|---:|---:|")
        for row in panel_results.itertuples():
            add(
                f"| {row.dictionary} | {row.spec} | {row.term} | {row.estimate:.5f} | "
                f"{row.t_value:.2f} | {row.n_obs} |"
            )
    else:
        add("- 无面板结果。")
    add()
    add("## 8. 结论")
    add()
    add(metrics["conclusion"])
    add()
    add("## 9. 局限")
    add()
    add("- 季度样本仅约 100 期，横截面约 16–31 个行业，统计功效有限。")
    add("- 共同情绪不能直接解释行业横截面差异；本报告只检验其通过行业暴露交互产生预测信息的可能。")
    add("- 宏观接口发布日期与统计月份口径存在滞后；本实验采用保守的发布日可得值。")
    add("- 行业指数起点不同，早期截面行业数较少；FMB 结果受早期小截面影响。")
    add("- `mpr_domain` 在早期小截面存在个别极端系数；表中同时给出中位数和 5% 缩尾均值，结论不依赖单一均值口径。")
    add()
    add("## 10. 复现")
    add()
    add("```powershell")
    add("python experiments/E4/run_e4.py")
    add("```")
    add()
    add(f"- 脚本 SHA256：`{metrics['script_sha256']}`")
    add(f"- 运行环境：{environment}")
    add()

    (LOG_DIR / "e4_fmb.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ensure_dirs()
    (TEMP_DIR / "e4_run.log").write_text("", encoding="utf-8")
    log("E4 cross-sectional prediction start")

    environment = {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": stats.__name__ + ":" + getattr(stats, "__version__", "unknown"),
    }

    industry, bench = load_prices()
    log(f"[data] industry rows={len(industry)}, benchmark rows={len(bench)}")
    macro = load_macro()
    log(f"[data] macro issue rows={len(macro)}, range {macro['publication_date'].min().date()}..{macro['publication_date'].max().date()}")
    panel = build_panel(industry, bench, macro)
    log(
        f"[panel] rows={len(panel)}, target quarters={panel['target_quarter'].nunique()}, "
        f"industries={panel['code'].nunique()}, range {panel['target_quarter'].min()}..{panel['target_quarter'].max()}"
    )

    # Drop target quarters that are not complete.  The 2026Q3 MPR points to
    # 2026Q4, which is not complete at run time; the panel already requires a
    # full quarter, but keep an explicit guard for reproducibility.
    panel = panel[panel["target_quarter"] <= "2026Q3"].copy()

    coefficient_frames: list[pd.DataFrame] = []
    skip_rows: list[dict[str, Any]] = []
    for spec, dictionary in SPECS:
        coefficients, skipped = cross_section_ols(panel, spec, dictionary)
        coefficient_frames.append(coefficients)
        for item in skipped:
            item.update({"spec": spec, "dictionary": dictionary or "none"})
            skip_rows.append(item)
        log(
            f"[fmb] {spec}/{dictionary or 'none'}: coefficient rows={len(coefficients)}, "
            f"periods={coefficients['target_quarter'].nunique() if not coefficients.empty else 0}, skipped={len(skipped)}"
        )
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    coefficients.to_csv(E4_DIR / "e4_coefficient_series.csv", index=False, encoding="utf-8-sig")
    fmb_results = summarize_fmb(coefficients)
    fmb_results.to_csv(E4_DIR / "e4_fmb_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(skip_rows).to_csv(TEMP_DIR / "e4_fmb_skipped.csv", index=False, encoding="utf-8-sig")
    log(f"[output] e4_coefficient_series.csv rows={len(coefficients)}")
    log(f"[output] e4_fmb_results.csv rows={len(fmb_results)}")

    train_start, train_end = "2001Q2", "2015Q4"
    val_start, val_end = "2016Q1", "2020Q4"
    test_start, test_end = "2021Q1", "2026Q3"

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    baseline_predictions: dict[str, pd.DataFrame] = {}
    for split, start_q, end_q in [
        ("validation", val_start, val_end),
        ("test", test_start, test_end),
    ]:
        baseline_pred, baseline_metrics = predict_oos(
            panel, coefficients, "baseline_notext", None, split, start_q, end_q
        )
        if baseline_pred.empty:
            log(f"[oos] baseline {split} skipped: {baseline_metrics}")
            continue
        baseline_predictions[split] = baseline_pred
        metric_rows.append(baseline_metrics)
        prediction_frames.append(baseline_pred)
        log(
            f"[oos] baseline {split}: periods={baseline_metrics['n_periods']}, "
            f"IC={baseline_metrics['ic_mean']:.4f}, OOS_R2={baseline_metrics['oos_r2']:.4f}, MAE={baseline_metrics['mae']:.5f}"
        )
        for dictionary in DICTS:
            for spec in ["sent_exposure"]:
                pred, metrics = predict_oos(
                    panel, coefficients, spec, dictionary, split, start_q, end_q
                )
                if pred.empty:
                    log(f"[oos] {spec}/{dictionary} {split} skipped: {metrics}")
                    continue
                # Use a common-sample baseline for paired improvement metrics.
                sent_keys = pred[["target_quarter", "code"]].drop_duplicates()
                baseline_pred_paired = baseline_pred.merge(sent_keys, on=["target_quarter", "code"], how="inner")
                baseline_metrics_paired = oos_metrics(baseline_pred_paired) if not baseline_pred_paired.empty else {}
                paired_t, paired_p = paired_mae_comparison(baseline_pred, pred)
                metrics.update(
                    {
                        "paired_mae_t": paired_t,
                        "paired_mae_p": paired_p,
                        "baseline_model": "baseline_notext",
                        "baseline_mae": baseline_metrics_paired.get("mae", float("nan")),
                        "baseline_ic": baseline_metrics_paired.get("ic_mean", float("nan")),
                        "baseline_icir": baseline_metrics_paired.get("icir", float("nan")),
                        "baseline_oos_r2": baseline_metrics_paired.get("oos_r2", float("nan")),
                        "baseline_n_obs_paired": baseline_metrics_paired.get("n_obs", float("nan")),
                        "mae_improvement_pct": (
                            (baseline_metrics_paired.get("mae", float("nan")) - metrics["mae"])
                            / abs(baseline_metrics_paired.get("mae", float("nan"))) * 100.0
                            if baseline_metrics_paired.get("mae")
                            else float("nan")
                        ),
                        "ic_improvement": metrics["ic_mean"] - baseline_metrics_paired.get("ic_mean", float("nan")),
                        "icir_improvement": metrics["icir"] - baseline_metrics_paired.get("icir", float("nan")),
                        "r2_improvement_pp": (
                            metrics["oos_r2"] - baseline_metrics_paired.get("oos_r2", float("nan"))
                        ) * 100.0,
                    }
                )
                metric_rows.append(metrics)
                prediction_frames.append(pred)
                log(
                    f"[oos] {spec}/{dictionary} {split}: periods={metrics['n_periods']}, "
                    f"IC={metrics['ic_mean']:.4f}, OOS_R2={metrics['oos_r2']:.4f}, "
                    f"MAE={metrics['mae']:.5f}, improve={metrics['mae_improvement_pct']:.2f}%"
                )

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    if not predictions.empty:
        predictions.to_csv(TEMP_DIR / "e4_oos_predictions.csv", index=False, encoding="utf-8-sig")
    oos_metrics_df = pd.DataFrame(metric_rows)
    if not oos_metrics_df.empty:
        columns = [
            "model", "dictionary", "spec", "split", "start_period", "end_period", "n_periods", "n_obs",
            "ic_mean", "ic_std", "icir", "ic_t", "ic_p", "pearson_ic_mean", "oos_r2", "oos_r2_overall",
            "mae", "rmse", "paired_mae_t", "paired_mae_p", "baseline_model", "baseline_mae",
            "baseline_ic", "baseline_icir", "baseline_oos_r2", "baseline_n_obs_paired", "mae_improvement_pct",
            "ic_improvement", "icir_improvement", "r2_improvement_pp",
        ]
        oos_metrics_df = oos_metrics_df.reindex(columns=columns)
        oos_metrics_df.to_csv(E4_DIR / "e4_oos_metrics.csv", index=False, encoding="utf-8-sig")
        log(f"[output] e4_oos_metrics.csv rows={len(oos_metrics_df)}")

    panel_results = panel_robustness(panel)
    panel_results.to_csv(E4_DIR / "e4_panel_robustness.csv", index=False, encoding="utf-8-sig")
    log(f"[output] e4_panel_robustness.csv rows={len(panel_results)}")

    # Conclusion logic: the primary test is the primary dictionary's
    # sentiment-exposure specification on the test split.  A positive result
    # requires higher IC, higher OOS R2, and lower MAE than baseline.
    primary_oos = oos_metrics_df[
        (oos_metrics_df["split"] == "test")
        & (oos_metrics_df["dictionary"] == PRIMARY_DICT)
        & (oos_metrics_df["spec"] == "sent_exposure")
    ]
    primary_fmb = fmb_results[
        (fmb_results["dictionary"] == PRIMARY_DICT)
        & (fmb_results["spec"] == "sent_exposure")
        & (fmb_results["is_sentiment"] == 1)
    ]
    if primary_oos.empty:
        conclusion = "样本外主比较未能生成，H2 的预测部分无法判定。"
    else:
        row = primary_oos.iloc[0]
        ic_better = row["ic_mean"] > row["baseline_ic"]
        mae_better = row["mae"] < row["baseline_mae"]
        r2_better = row["oos_r2"] > row["baseline_oos_r2"]
        fmb_positive_share = float((primary_fmb["mean"] > 0).mean()) if not primary_fmb.empty else float("nan")
        if ic_better and mae_better and r2_better:
            conclusion = (
                f"主词典 `{PRIMARY_DICT}` 的 `sent_exposure` 在测试集上 IC={row['ic_mean']:.4f}、"
                f"OOS R²={row['oos_r2']:.4f}、MAE={row['mae']:.5f}，三项均优于无文本 Baseline，"
                "H2 的预测部分获得有限支持；但情绪暴露信号由滚动估计生成，结论应视为弱证据。"
            )
        else:
            conclusion = (
                f"主词典 `{PRIMARY_DICT}` 的 `sent_exposure` 在测试集上未同时优于无文本 Baseline："
                f"IC={row['ic_mean']:.4f} vs {row['baseline_ic']:.4f}，"
                f"OOS R²={row['oos_r2']:.4f} vs {row['baseline_oos_r2']:.4f}，"
                f"MAE={row['mae']:.5f} vs {row['baseline_mae']:.5f}。"
                "因此 H2 的预测部分未获支持；共同情绪不能直接识别行业横截面差异，滚动情绪暴露设定也未提供稳定增量。"
            )

    metrics = {
        "environment": environment,
        "industry_rows": int(len(industry)),
        "industry_count": int(industry["code"].nunique()),
        "benchmark_source": (
            "workspace_cache:_temp/e0_market_raw/benchmark_sh000300.csv"
            if BENCH_CACHE_PATH.exists() and BENCH_CACHE_PATH.stat().st_size > 0
            else "workspace:data/external/benchmark/sh000300_沪深300.csv"
        ),
        "mpr_issues": int(len(pd.read_csv(MPR_INDEX_PATH))),
        "panel_rows": int(len(panel)),
        "n_target_quarters": int(panel["target_quarter"].nunique()),
        "panel_first_quarter": panel["target_quarter"].min(),
        "panel_last_quarter": panel["target_quarter"].max(),
        "min_n_industries": int(panel.groupby("target_quarter")["code"].size().min()),
        "max_n_industries": int(panel.groupby("target_quarter")["code"].size().max()),
        "train_start": train_start,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": test_end,
        "fmb_result_rows": int(len(fmb_results)),
        "coefficient_rows": int(len(coefficients)),
        "oos_metric_rows": int(len(oos_metrics_df)),
        "panel_robustness_rows": int(len(panel_results)),
        "fmb_skipped_cells": int(len(skip_rows)),
        "conclusion": conclusion,
        "script_sha256": sha256(Path(__file__)),
        "seed": SEED,
        "nw_lag": NW_LAG,
    }
    (E4_DIR / "e4_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_log(environment, panel, fmb_results, oos_metrics_df, panel_results, metrics)
    log(f"[output] logs/e4_fmb.md")
    log(f"[done] {conclusion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
