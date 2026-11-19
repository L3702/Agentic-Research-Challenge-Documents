"""Event-study robustness with event-level inference and paired samples.

All reusable inputs are read only. Intermediate data are written beneath
``_temp/e8/event``. The standalone entry point is useful for independent QA;
``run_event`` is also called by the E8 report builder.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "_temp" / "e8" / "event"
PRIMARY_DICTIONARY = "ntusd_simplified"
DICTIONARIES = [PRIMARY_DICTIONARY, "dlut", "mpr_domain"]


@dataclass(frozen=True)
class EventSetting:
    setting_id: str
    family: str = "baseline"
    changed_parameter: str = "none"
    setting_value: str = "main"
    dictionary: str = PRIMARY_DICTIONARY
    benchmark: str = "sh000300"
    estimation_window: int = 120
    event_window: int = 5
    sample_start: str | None = None
    sample_end: str | None = None
    model: str = "market"
    purge: bool = False
    winsor: float = 0.0
    measure: str = "terciles"


def default_settings() -> list[EventSetting]:
    main = EventSetting("event_main")
    out = [main]
    for dictionary in ["dlut", "mpr_domain", "combo_ntusd_dlut"]:
        out.append(replace(main, setting_id=f"event_dictionary_{dictionary}",
                           family="dictionary", changed_parameter="dictionary",
                           setting_value=dictionary, dictionary=dictionary))
    for window in [1, 3, 10]:
        out.append(replace(main, setting_id=f"event_window_{window}",
                           family="event_window", changed_parameter="event_window",
                           setting_value=f"[0,+{window}]", event_window=window))
    for year in [2008, 2015, 2020]:
        out.append(replace(main, setting_id=f"event_pre{year}", family="subsample",
                           changed_parameter="publication_date_sample",
                           setting_value=f"before_{year}", sample_end=f"{year}-01-01"))
        out.append(replace(main, setting_id=f"event_post{year}", family="subsample",
                           changed_parameter="publication_date_sample",
                           setting_value=f"from_{year}", sample_start=f"{year}-01-01"))
    out.append(replace(main, setting_id="event_benchmark_allshare", family="benchmark",
                       changed_parameter="benchmark", setting_value="CSI_All_Share_000985",
                       benchmark="sh000985"))
    out.append(replace(main, setting_id="event_winsor_1pct", family="outliers",
                       changed_parameter="industry_car_winsor", setting_value="1pct_99pct",
                       winsor=0.01))
    for length in [30, 60, 90, 180, 250]:
        out.append(replace(main, setting_id=f"event_estimation_{length}",
                           family="estimation_window", changed_parameter="estimation_window",
                           setting_value=str(length), estimation_window=length))
    out.append(replace(main, setting_id="event_estimation_purged", family="estimation_window",
                       changed_parameter="purge_prior_mpr_windows", setting_value="remove_all_prior_[0,+10]",
                       purge=True))
    out.append(replace(main, setting_id="event_continuous", family="sentiment_form",
                       changed_parameter="sentiment_form", setting_value="continuous_HC3_slope",
                       measure="continuous"))
    out.append(replace(main, setting_id="event_mean_adjusted", family="normal_return_model",
                       changed_parameter="normal_return_model", setting_value="mean_adjusted",
                       model="mean_adj"))
    return out


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, str]]:
    mpr = pd.read_csv(ROOT / "experiments/E1/mpr_index_full.csv", dtype={"issue": str})
    mpr["publication_date"] = pd.to_datetime(mpr["publication_date"])
    sent = pd.read_csv(ROOT / "experiments/E1/sentiment_dictionary_full.csv", dtype={"issue": str})
    if sent.duplicated(["issue", "dictionary"]).any() or mpr["issue"].duplicated().any():
        raise ValueError("duplicate sentiment/issue input keys")
    wide = sent.pivot(index="issue", columns="dictionary", values="sentiment_score")
    events = mpr[["issue", "report_period", "publication_date"]].merge(wide, on="issue", validate="one_to_one")
    events = events.sort_values("publication_date").reset_index(drop=True)
    # Include only currently known scores in this composite. Early observations
    # without eight reports do not receive a composite score.
    trailing_z = []
    for dictionary in [PRIMARY_DICTIONARY, "dlut"]:
        x = events[dictionary].astype(float)
        trailing_z.append((x - x.expanding(min_periods=8).mean()) / x.expanding(min_periods=8).std())
    events["combo_ntusd_dlut"] = (trailing_z[0] + trailing_z[1]) / 2.0
    for dictionary in DICTIONARIES + ["combo_ntusd_dlut"]:
        events[f"group_{dictionary}"] = tercile_labels(events, dictionary)

    industry = pd.read_csv(ROOT / "data/external/sw_industries/sw_industry_daily.csv", dtype={"code": str})
    industry = industry.rename(columns={"收盘": "close"})
    industry["date"] = pd.to_datetime(industry["date"])
    industry = industry.sort_values(["code", "date"]).reset_index(drop=True)
    if industry.duplicated(["code", "date"]).any():
        raise ValueError("duplicate industry-date input keys")
    industry["daily_return"] = industry.groupby("code")["close"].pct_change(fill_method=None)
    benchmark_paths = {
        "sh000300": ROOT / "_temp/e0_market_raw/benchmark_sh000300.csv",
        "sh000985": ROOT / "data/external/benchmark/sh000985_中证全指.csv",
    }
    if not benchmark_paths["sh000300"].exists():
        benchmark_paths["sh000300"] = ROOT / "data/external/benchmark/sh000300_沪深300.csv"
    benchmarks: dict[str, pd.DataFrame] = {}
    sources = {}
    for code, path in benchmark_paths.items():
        raw = pd.read_csv(path, dtype={"code": str})
        raw = raw.loc[raw["code"] == code].copy()
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.sort_values("date").reset_index(drop=True)
        if raw["date"].duplicated().any() or raw.empty:
            raise ValueError(f"invalid benchmark input {code}")
        raw["daily_return"] = raw["close"].pct_change(fill_method=None)
        benchmarks[code] = raw
        sources[code] = str(path.relative_to(ROOT)).replace("\\", "/")
    return events, industry, benchmarks, sources


def tercile_labels(events: pd.DataFrame, score_column: str) -> pd.Series:
    """Reproduce E3's deterministic full-corpus rank thirds, including ties."""
    valid = events[score_column].notna()
    ordered = events.loc[valid, ["issue", score_column]].sort_values(score_column)
    n = len(ordered)
    mapping = {
        issue: "low" if i < n / 3 else ("mid" if i < 2 * n / 3 else "high")
        for i, issue in enumerate(ordered["issue"])
    }
    return events["issue"].map(mapping)


def align_event_indices(dates: pd.DatetimeIndex, publications: pd.Series) -> np.ndarray:
    positions = dates.searchsorted(pd.DatetimeIndex(publications), side="left").astype(int)
    positions[(pd.DatetimeIndex(publications) < dates[0]) | (positions >= len(dates))] = -1
    return positions


def prior_event_mask(calendar: pd.DatetimeIndex, publications: pd.Series, width: int = 10) -> np.ndarray:
    """Mark actual event trading days through +width on this calendar."""
    mask = np.zeros(len(calendar), dtype=bool)
    for pos in align_event_indices(calendar, publications):
        if pos >= 0:
            mask[pos : min(pos + width + 1, len(calendar))] = True
    return mask


def compute_observations(
    events: pd.DataFrame,
    industry: pd.DataFrame,
    benchmarks: dict[str, pd.DataFrame],
    settings: list[EventSetting],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate each unique return specification once, with strict coverage."""
    observations = []
    failures = []
    unique_specs = sorted({(s.benchmark, s.estimation_window, s.event_window, s.purge, s.model) for s in settings})
    assets = sorted(industry["code"].unique())
    daily = industry.pivot(index="date", columns="code", values="daily_return").sort_index()
    names = industry.drop_duplicates("code").set_index("code")["name"].to_dict()
    for benchmark, est_len, window, purge, model in unique_specs:
        spec_id = f"{benchmark}:{est_len}:{window}:{int(purge)}:{model}"
        market = benchmarks[benchmark]
        calendar = pd.DatetimeIndex(market["date"])
        rm = market["daily_return"].to_numpy(dtype=float)
        ri = daily.reindex(index=calendar, columns=assets).to_numpy(dtype=float)
        positions = align_event_indices(calendar, events["publication_date"])
        historical_event_days = prior_event_mask(calendar, events["publication_date"])
        for event_number, event in events.iterrows():
            event_pos = int(positions[event_number])
            common_failure = None
            if event_pos < 0:
                common_failure = "publication_outside_benchmark_calendar"
            elif event_pos < est_len:
                common_failure = "insufficient_benchmark_estimation_history"
            elif event_pos + window >= len(calendar):
                common_failure = "incomplete_event_window"
            if common_failure:
                failures.append({"return_spec_id": spec_id, "issue": event["issue"], "code": "ALL", "reason": common_failure})
                continue
            candidates = np.arange(event_pos - est_len, event_pos)
            selected = candidates[~historical_event_days[candidates]] if purge else candidates
            if len(selected) < (30 if purge else est_len):
                failures.append({"return_spec_id": spec_id, "issue": event["issue"], "code": "ALL", "reason": "fewer_than_30_estimation_days_after_purge"})
                continue
            ev_indices = np.arange(event_pos, event_pos + window + 1)
            assert selected.max() < event_pos
            assert not purge or not historical_event_days[selected].any()
            rm_est = rm[selected]
            rm_event = rm[ev_indices]
            if not np.isfinite(rm_est).all() or not np.isfinite(rm_event).all():
                failures.append({"return_spec_id": spec_id, "issue": event["issue"], "code": "ALL", "reason": "missing_benchmark_returns"})
                continue
            prior_position = int(positions[event_number - 1]) if event_number else -1
            # An earlier MPR outside this benchmark's calendar cannot be an
            # included estimate-day observation.
            overlap = prior_position >= 0 and bool(np.isin(prior_position, selected))
            overlap_window = bool(historical_event_days[selected].any())
            for col, code in enumerate(assets):
                y_est = ri[selected, col]
                y_event = ri[ev_indices, col]
                if not np.isfinite(y_est).all():
                    reason = "missing_industry_estimation_returns"
                elif not np.isfinite(y_event).all():
                    reason = "missing_industry_event_returns"
                else:
                    reason = ""
                if reason:
                    failures.append({"return_spec_id": spec_id, "issue": event["issue"], "code": code, "reason": reason})
                    continue
                if model == "market":
                    beta, alpha = np.polyfit(rm_est, y_est, 1)
                    expected = alpha + beta * rm_event
                elif model == "mean_adj":
                    alpha, beta = float(y_est.mean()), 0.0
                    expected = np.full(len(y_event), alpha)
                else:
                    raise ValueError(f"unknown model {model}")
                observations.append({
                    "return_spec_id": spec_id, "issue": event["issue"], "code": code,
                    "name": names[code], "publication_date": event["publication_date"],
                    "event_trading_date": calendar[event_pos], "benchmark": benchmark,
                    "estimation_window": est_len, "event_window": window, "purge": purge,
                    "model": model, "car": float((y_event - expected).sum()),
                    "alpha": float(alpha), "beta": float(beta),
                    "estimation_start": calendar[selected[0]],
                    "estimation_end": calendar[selected[-1]], "n_estimation_days": len(selected),
                    "n_purged_days": len(candidates) - len(selected),
                    "prior_event_in_estimation": overlap,
                    "any_mpr_window_in_estimation": overlap_window,
                })
    return pd.DataFrame(observations), pd.DataFrame(failures)


def welch_difference(high: np.ndarray, low: np.ndarray) -> dict[str, float]:
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    high, low = high[np.isfinite(high)], low[np.isfinite(low)]
    empty = {k: float("nan") for k in ["estimate", "std_error", "t_stat", "p_value", "ci_low", "ci_high"]}
    if len(high) == 0 or len(low) == 0:
        return empty
    estimate = float(high.mean() - low.mean())
    if min(len(high), len(low)) < 2:
        return {**empty, "estimate": estimate}
    a = np.var(high, ddof=1) / len(high)
    b = np.var(low, ddof=1) / len(low)
    se = float(np.sqrt(a + b))
    if se <= 0:
        return {**empty, "estimate": estimate, "std_error": se}
    df = float((a + b) ** 2 / (a * a / (len(high) - 1) + b * b / (len(low) - 1)))
    t_stat = estimate / se
    critical = float(stats.t.ppf(0.975, df))
    return {"estimate": estimate, "std_error": se, "t_stat": t_stat,
            "p_value": float(2 * stats.t.sf(abs(t_stat), df)),
            "ci_low": estimate - critical * se, "ci_high": estimate + critical * se}


def continuous_slope(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """HC3 OLS slope; x is standardized within the descriptive event sample."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    empty = {k: float("nan") for k in ["estimate", "std_error", "t_stat", "p_value", "ci_low", "ci_high"]}
    if len(x) < 5 or np.std(x, ddof=1) <= 0:
        return empty
    z = (x - x.mean()) / x.std(ddof=1)
    design = np.column_stack([np.ones(len(x)), z])
    bread = np.linalg.inv(design.T @ design)
    beta = bread @ design.T @ y
    residual = y - design @ beta
    leverage = np.sum((design @ bread) * design, axis=1)
    hc3_residual = residual / (1.0 - leverage)
    covariance = bread @ (design.T @ (design * (hc3_residual ** 2)[:, None])) @ bread
    se = float(np.sqrt(max(covariance[1, 1], 0.0)))
    t_stat = float(beta[1] / se) if se > 0 else float("nan")
    critical = float(stats.t.ppf(0.975, len(y) - 2))
    return {"estimate": float(beta[1]), "std_error": se, "t_stat": t_stat,
            "p_value": float(2 * stats.t.sf(abs(t_stat), len(y) - 2)),
            "ci_low": float(beta[1] - critical * se), "ci_high": float(beta[1] + critical * se)}


def select_observations(observations: pd.DataFrame, setting: EventSetting) -> pd.DataFrame:
    mask = ((observations["benchmark"] == setting.benchmark)
            & (observations["estimation_window"] == setting.estimation_window)
            & (observations["event_window"] == setting.event_window)
            & (observations["purge"] == setting.purge)
            & (observations["model"] == setting.model))
    if setting.sample_start:
        mask &= observations["publication_date"] >= pd.Timestamp(setting.sample_start)
    if setting.sample_end:
        mask &= observations["publication_date"] < pd.Timestamp(setting.sample_end)
    out = observations.loc[mask].copy()
    if setting.winsor and not out.empty:
        lo, hi = out["car"].quantile([setting.winsor, 1.0 - setting.winsor])
        out["car"] = out["car"].clip(lo, hi)
    return out


def evaluate_sample(observations: pd.DataFrame, events: pd.DataFrame, setting: EventSetting) -> tuple[dict[str, Any], pd.DataFrame]:
    empty_stats = {k: float("nan") for k in ["estimate", "std_error", "t_stat", "p_value", "ci_low", "ci_high"]}
    if observations.empty:
        return {**empty_stats, "status": "failed", "failure_reason": "no_valid_event_industry_observations",
                "n_obs": 0, "n_periods": 0, "n_high": 0, "n_mid": 0, "n_low": 0}, pd.DataFrame()
    periods = observations.groupby("issue", as_index=False).agg(
        period_car=("car", "mean"), n_industries=("code", "size"),
        prior_event_overlap=("prior_event_in_estimation", "mean"),
        any_prior_window_overlap=("any_mpr_window_in_estimation", "mean"))
    periods = periods.merge(events[["issue", "publication_date", setting.dictionary,
                                   f"group_{setting.dictionary}"]], on="issue", validate="one_to_one")
    periods = periods.rename(columns={setting.dictionary: "sentiment_score",
                                      f"group_{setting.dictionary}": "sentiment_group"})
    periods = periods.dropna(subset=["sentiment_score", "sentiment_group"]).sort_values("publication_date")
    high = periods.loc[periods["sentiment_group"] == "high", "period_car"].to_numpy()
    low = periods.loc[periods["sentiment_group"] == "low", "period_car"].to_numpy()
    diff = welch_difference(high, low)
    continuous = continuous_slope(periods["sentiment_score"].to_numpy(), periods["period_car"].to_numpy())
    core = continuous if setting.measure == "continuous" else diff
    status = "ok"
    reason = ""
    if not np.isfinite(core["estimate"]) or not np.isfinite(core["p_value"]):
        status, reason = "failed", "unidentified_effect_or_missing_high_low_group"
    elif setting.measure != "continuous" and min(len(high), len(low)) < 5:
        status, reason = "limited", "fewer_than_5_events_in_high_or_low_group"
    elif len(periods) < 20:
        status, reason = "limited", "fewer_than_20_events"
    used_issues = set(periods["issue"])
    used_observations = observations[observations["issue"].isin(used_issues)]
    all_car = periods["period_car"].to_numpy(dtype=float)
    mean_t, mean_p = stats.ttest_1samp(all_car, 0) if len(all_car) > 1 else (np.nan, np.nan)
    result = {
        **core, "status": status, "failure_reason": reason,
        "n_obs": len(used_observations), "n_periods": len(periods),
        "n_high": len(high), "n_mid": int((periods["sentiment_group"] == "mid").sum()), "n_low": len(low),
        "mean_car": float(periods["period_car"].mean()), "mean_car_t": float(mean_t), "mean_car_p": float(mean_p),
        "car_high": float(high.mean()) if len(high) else np.nan,
        "car_mid": float(periods.loc[periods["sentiment_group"] == "mid", "period_car"].mean()),
        "car_low": float(low.mean()) if len(low) else np.nan,
        "car_high_minus_low": diff["estimate"], "car_high_minus_low_p": diff["p_value"],
        "continuous_slope": continuous["estimate"], "continuous_p": continuous["p_value"],
        "prior_event_overlap_share": float(used_observations["prior_event_in_estimation"].mean()),
        "any_prior_window_overlap_share": float(used_observations["any_mpr_window_in_estimation"].mean()),
        "n_estimation_days_mean": float(used_observations["n_estimation_days"].mean()),
        "n_purged_days_mean": float(used_observations["n_purged_days"].mean()),
        "first_publication": str(periods["publication_date"].min().date()) if len(periods) else "",
        "last_publication": str(periods["publication_date"].max().date()) if len(periods) else "",
    }
    return result, periods


def common_observations(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["issue", "code"]
    common = left[keys].merge(right[keys], on=keys, validate="one_to_one")
    return (left.merge(common, on=keys, validate="one_to_one"),
            right.merge(common, on=keys, validate="one_to_one"))


def validate_e3(observations: pd.DataFrame) -> dict[str, Any]:
    """Independently compare every available E3 main-window industry CAR."""
    source = ROOT / "experiments/E3/e3_event_results.csv"
    if not source.exists():
        return {"status": "unavailable", "reason": "E3 result file absent"}
    e3 = pd.read_csv(source, dtype={"issue": str, "code": str})
    e3 = e3[(e3["model"] == "market") & (e3["estimation_window"] == 120)]
    current = observations[(observations["benchmark"] == "sh000300")
                           & (observations["estimation_window"] == 120)
                           & (~observations["purge"]) & (observations["model"] == "market")]
    keys = ["issue", "code", "event_window"]
    check = current[keys + ["car"]].merge(e3[keys + ["car"]], on=keys, suffixes=("_e8", "_e3"), validate="one_to_one", how="outer", indicator=True)
    missing = int((check["_merge"] != "both").sum())
    max_diff = float((check["car_e8"] - check["car_e3"]).abs().max())
    if missing or max_diff > 1e-10:
        raise AssertionError(f"E3 event reproduction failed: missing={missing}, max_diff={max_diff}")
    return {"status": "passed", "n_industry_event_windows": len(check), "max_absolute_car_difference": max_diff}


def run_event() -> tuple[pd.DataFrame, dict[str, Any]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    settings = default_settings()
    events, industry, benchmarks, sources = load_inputs()
    observations, failures = compute_observations(events, industry, benchmarks, settings)
    reproduction = validate_e3(observations)
    main_setting = settings[0]
    main = select_observations(observations, main_setting)
    summaries, period_frames, setting_failures, sentiment_exclusions = [], [], [], []
    for setting in settings:
        selected = select_observations(observations, setting)
        result, periods = evaluate_sample(selected, events, setting)
        # Composite sentiment's initial missing values must not enter either
        # side of a paired comparison.
        valid_issues = set(events.loc[events[setting.dictionary].notna(), "issue"])
        excluded_scores = selected.loc[~selected["issue"].isin(valid_issues), ["issue", "code"]].copy()
        for issue, code in excluded_scores.itertuples(index=False):
            sentiment_exclusions.append({"setting_id": setting.setting_id, "issue": issue,
                                         "code": code, "reason": "sentiment_unavailable_or_trailing_z_warmup"})
        selected = selected[selected["issue"].isin(valid_issues)]
        paired_variant, paired_main = common_observations(selected, main)
        variant_result, _ = evaluate_sample(paired_variant, events, setting)
        # For continuous-vs-tercile comparisons, report both baseline scales:
        # the conventional group contrast and the baseline continuous slope.
        baseline_result, _ = evaluate_sample(paired_main, events, main_setting)
        paired_scale = replace(main_setting, measure=setting.measure)
        baseline_same_measure, _ = evaluate_sample(paired_main, events, paired_scale)
        notes = ["exploratory robustness; event-level Welch inference; E3 full-corpus terciles retained; [0,+5] is a display anchor, not a new preregistered primary window"]
        if setting.dictionary == "combo_ntusd_dlut":
            notes.append("mean of NTUSD/DLUT expanding z scores, current and prior reports only, min 8; descriptive full-corpus composite terciles")
        if setting.purge:
            notes.append("candidate window stays [-120,-1]; remove every MPR [0,+10] trading day; min 30 retained days; no backfill")
        if setting.winsor:
            notes.append("descriptive pooled industry-CAR clipping at 1%/99%; no observations removed")
        if setting.measure == "continuous":
            notes.append("estimate is HC3 event-CAR slope per 1 sample SD of sentiment; unlike group contrast, units differ")
        if setting.benchmark == "sh000985":
            notes.append("CSI All Share starts 2011; use paired_variant/baseline fields for benchmark comparison")
        summary = {
            "block": "event", **asdict(setting), "comparison_id": "event_main",
            "metric_name": "event_CAR_continuous_slope" if setting.measure == "continuous" else "event_CAR_high_minus_low",
            "effect_unit": "decimal_CAR_per_sample_SD_sentiment" if setting.measure == "continuous" else "decimal_CAR_high_minus_low",
            "coefficient": "continuous_sentiment_slope" if setting.measure == "continuous" else "car_high_minus_low",
            **result,
            "paired_n_obs": len(paired_variant),
            "paired_n_periods": variant_result["n_periods"],
            "paired_status": variant_result["status"],
            "paired_failure_reason": variant_result["failure_reason"],
            "paired_variant_estimate": variant_result["estimate"],
            "paired_variant_p": variant_result["p_value"],
            "paired_variant_ci_low": variant_result["ci_low"],
            "paired_variant_ci_high": variant_result["ci_high"],
            "paired_variant_mean_car": variant_result.get("mean_car", np.nan),
            "paired_baseline_estimate": baseline_result["estimate"],
            "paired_baseline_p": baseline_result["p_value"],
            "paired_baseline_ci_low": baseline_result["ci_low"],
            "paired_baseline_ci_high": baseline_result["ci_high"],
            "paired_baseline_mean_car": baseline_result.get("mean_car", np.nan),
            "paired_baseline_same_measure_estimate": baseline_same_measure["estimate"],
            "paired_baseline_same_measure_p": baseline_same_measure["p_value"],
            "paired_sign_consistent": (bool(np.sign(variant_result["estimate"]) == np.sign(baseline_same_measure["estimate"]))
                                       if np.isfinite(variant_result["estimate"]) and np.isfinite(baseline_same_measure["estimate"]) else None),
            "notes": "; ".join(notes),
        }
        spec_id = f"{setting.benchmark}:{setting.estimation_window}:{setting.event_window}:{int(setting.purge)}:{setting.model}"
        relevant_failures = failures[failures["return_spec_id"] == spec_id].copy()
        relevant_failures = relevant_failures.merge(events[["issue", "publication_date"]], on="issue", how="left")
        if setting.sample_start:
            relevant_failures = relevant_failures[relevant_failures["publication_date"] >= pd.Timestamp(setting.sample_start)]
        if setting.sample_end:
            relevant_failures = relevant_failures[relevant_failures["publication_date"] < pd.Timestamp(setting.sample_end)]
        summary["n_failure_records"] = len(relevant_failures)
        summary["n_failed_industry_cells"] = int(np.where(relevant_failures["code"] == "ALL", industry["code"].nunique(), 1).sum())
        summary["n_missing_sentiment_cells"] = len(excluded_scores)
        summary["failure_reasons"] = "; ".join(f"{reason}={count}" for reason, count in relevant_failures["reason"].value_counts().items())
        summaries.append(summary)
        if len(periods):
            periods.insert(0, "setting_id", setting.setting_id)
            period_frames.append(periods)
        if result["status"] != "ok":
            setting_failures.append({"setting_id": setting.setting_id, "stage": "inference", "reason": result["failure_reason"], "status": result["status"]})
        if variant_result["status"] != "ok":
            setting_failures.append({"setting_id": setting.setting_id, "stage": "paired_inference", "reason": variant_result["failure_reason"], "status": variant_result["status"]})
    summary_df = pd.DataFrame(summaries)
    periods_df = pd.concat(period_frames, ignore_index=True)
    observations.to_csv(CACHE / "observations.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(CACHE / "failures.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(setting_failures, columns=["setting_id", "stage", "reason", "status"]).to_csv(CACHE / "setting_failures.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sentiment_exclusions, columns=["setting_id", "issue", "code", "reason"]).to_csv(CACHE / "sentiment_exclusions.csv", index=False, encoding="utf-8-sig")
    periods_df.to_csv(CACHE / "period_stats.csv", index=False, encoding="utf-8-sig")
    events.to_csv(CACHE / "event_sentiment.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(CACHE / "summary.csv", index=False, encoding="utf-8-sig")
    details = {
        "n_settings": len(settings), "n_mpr_reports": len(events),
        "n_industries": int(industry["code"].nunique()),
        "n_return_observations": len(observations),
        "n_failed_return_records": len(failures),
        "n_failed_return_cells": int(np.where(failures["code"] == "ALL", industry["code"].nunique(), 1).sum()),
        "n_setting_inference_warnings": len(setting_failures),
        "n_missing_sentiment_cells": len(sentiment_exclusions),
        "status_counts": summary_df["status"].value_counts().to_dict(),
        "benchmark_sources": sources,
        "benchmark_coverage": {code: {"start": str(data["date"].min().date()), "end": str(data["date"].max().date()), "n_rows": len(data)} for code, data in benchmarks.items()},
        "e3_reproduction": reproduction,
        "group_definition": "E3 full-corpus score ranking: low ranks < N/3, middle < 2N/3, remaining high; all 103 reports for single dictionaries",
        "subsample_definition": "MPR publication date < Jan 1 of cut year versus >= Jan 1; original full-corpus labels retained",
        "paired_definition": "exact common issue-code observations; average industries inside each issue, then recompute both event-level contrasts",
        "continuous_definition": "HC3 OLS of event-mean CAR on sentiment standardized by analytic-sample SD; descriptive, no OOS claim",
        "display_anchor": "E3 reports all four event windows; [0,+5] is only the E8 comparison/display anchor, not a newly preregistered primary outcome",
        "failure_count_definition": "failure records with code ALL cover every requested industry; n_failed_return_cells expands ALL by the 31-industry universe",
        "failure_file": "_temp/e8/event/failures.csv",
        "setting_failure_file": "_temp/e8/event/setting_failures.csv",
        "sentiment_exclusion_file": "_temp/e8/event/sentiment_exclusions.csv",
        "prior_overlap_note": "trading-date alignment, unlike E3 publication-calendar-date check; purged removes all prior [0,+10] event days",
    }
    (CACHE / "details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_df, details


if __name__ == "__main__":
    summary, details = run_event()
    print(summary[["setting_id", "status", "n_periods", "n_high", "n_low", "estimate", "p_value", "paired_variant_estimate", "paired_baseline_estimate", "prior_event_overlap_share"]].to_string(index=False))
    print(json.dumps(details, ensure_ascii=False, indent=2))
