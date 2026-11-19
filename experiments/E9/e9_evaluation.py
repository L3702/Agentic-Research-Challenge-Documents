"""Rebuild E6-compatible features and evaluate exploratory E9 score variants."""

from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "E4"))
sys.path.insert(0, str(ROOT / "experiments" / "E6"))
import run_e4 as e4
import run_e6 as e6


SPLITS = {
    "validation": ("2016Q1", "2020Q4"),
    "test": ("2021Q1", "2026Q3"),
}
FEATURES = [
    "sentiment_llm",
    "sent_beta_llm",
    "sent_beta_llm_z",
    "sentiment_llm_z",
    "sent_x_exposure_llm",
]


@lru_cache(maxsize=1)
def _source_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(
        ROOT / "_temp" / "e6" / "e6_panel.csv",
        dtype={"issue": str, "code": str},
    )
    quarterly = pd.read_csv(
        ROOT / "_temp" / "e6" / "industry_quarterly.csv",
        dtype={"code": str},
    )
    mpr = pd.read_csv(
        ROOT / "experiments" / "E1" / "mpr_index_full.csv",
        dtype={"issue": str},
    )
    return panel, quarterly, mpr


def rebuild_variant_panel(scores: pd.DataFrame) -> pd.DataFrame:
    """Recompute every feature used by E6's LLM model without stale CSV caches."""
    if not {"issue", "llm_sentiment"}.issubset(scores.columns):
        raise ValueError("Scores require issue and llm_sentiment columns")
    if scores["issue"].duplicated().any():
        raise ValueError("Duplicate issue in score table")
    original_panel, quarterly, mpr = _source_frames()
    panel = original_panel.copy()
    score_series = pd.to_numeric(
        scores.assign(issue=scores["issue"].astype(str))
        .set_index("issue")["llm_sentiment"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    score_by_quarter = mpr.set_index("report_period")["issue"].map(score_series)
    returns_matrix = quarterly.pivot(
        index="quarter", columns="code", values="log_return"
    ).sort_index()
    exposure: dict[tuple[str, str], float] = {}
    for issue, report_quarter in mpr[["issue", "report_period"]].itertuples(index=False):
        for code in returns_matrix.columns:
            returns = returns_matrix[code].dropna().loc[:report_quarter]
            if len(returns) < 8:
                exposure[(issue, code)] = np.nan
                continue
            aligned = pd.DataFrame(
                {"r": returns, "s": score_by_quarter.reindex(returns.index)}
            ).dropna().tail(20)
            if len(aligned) < 8 or float(aligned["s"].var(ddof=1)) <= 0:
                exposure[(issue, code)] = np.nan
                continue
            covariance = np.cov(aligned["r"], aligned["s"], ddof=1)[0, 1]
            variance = np.var(aligned["s"], ddof=1)
            exposure[(issue, code)] = float(covariance / variance)
    panel["sentiment_llm"] = panel["issue"].map(score_series)
    panel["sentiment_llm_raw"] = panel["sentiment_llm"]
    panel["sent_beta_llm"] = [
        exposure.get((issue, code), np.nan)
        for issue, code in panel[["issue", "code"]].itertuples(index=False)
    ]
    group = panel.groupby("target_quarter")["sent_beta_llm"]
    panel["sent_beta_llm_z"] = (
        panel["sent_beta_llm"] - group.transform("mean")
    ) / group.transform("std")
    issue_level = (
        panel.drop_duplicates("issue")
        .sort_values("publication_date")[["issue", "sentiment_llm"]]
        .copy()
    )
    sentiment = issue_level["sentiment_llm"]
    issue_level["sentiment_llm_z"] = (
        sentiment - sentiment.expanding(min_periods=8).mean()
    ) / sentiment.expanding(min_periods=8).std(ddof=1)
    panel["sentiment_llm_z"] = panel["issue"].map(
        issue_level.set_index("issue")["sentiment_llm_z"]
    )
    panel["sent_x_exposure_llm"] = (
        panel["sentiment_llm_z"] * panel["sent_beta_llm_z"]
    )
    return panel


def _evaluate_panel(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    coefficients, _skipped = e6.cross_section_ols(
        panel, "sent_exposure", ["llm"], "llm", "llm"
    )
    predictions_by_split = {split: pd.DataFrame() for split in SPLITS}
    if coefficients.empty:
        return predictions_by_split
    for split, (start_quarter, end_quarter) in SPLITS.items():
        predictions, _metrics = e6.predict_oos(
            panel,
            coefficients,
            "sent_exposure",
            ["llm"],
            "llm",
            "llm",
            split,
            start_quarter,
            end_quarter,
        )
        predictions_by_split[split] = predictions
    return predictions_by_split


def evaluate_scores(scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return native validation/test forecasts; absent forecasts remain empty."""
    return _evaluate_panel(rebuild_variant_panel(scores))


def metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute inherited E6 metrics, retaining NA for unavailable forecasts."""
    if frame.empty:
        return {
            "n_obs": 0,
            "n_quarters": 0,
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "icir": np.nan,
            "ic_t": np.nan,
            "ic_p": np.nan,
            "pearson_ic_mean": np.nan,
            "oos_r2": np.nan,
            "oos_r2_overall": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
        }
    result = e4.oos_metrics(frame)
    result["n_quarters"] = int(frame["target_quarter"].nunique())
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_e6_reproduction(baseline_scores: pd.DataFrame) -> dict[str, Any]:
    """Verify baseline features/forecasts/metrics and report inherited timing."""
    original_panel, _quarterly, _mpr = _source_frames()
    rebuilt = rebuild_variant_panel(baseline_scores)
    feature_errors = {}
    for column in FEATURES:
        original_values = original_panel[column].to_numpy(dtype=float)
        rebuilt_values = rebuilt[column].to_numpy(dtype=float)
        np.testing.assert_allclose(
            original_values, rebuilt_values, atol=1e-10, rtol=0, equal_nan=True
        )
        feature_errors[column] = float(
            np.nanmax(np.abs(original_values - rebuilt_values))
        )
    forecasts = _evaluate_panel(rebuilt)
    cached_path = ROOT / "_temp" / "e6" / "e6_oos_predictions.csv"
    published_path = ROOT / "experiments" / "E6" / "e6_llm_oos_metrics.csv"
    cached = pd.read_csv(cached_path, dtype={"code": str})
    published = pd.read_csv(published_path)
    split_evidence = {}
    for split, frame in forecasts.items():
        reference = cached[(cached["model"] == "llm") & (cached["split"] == split)]
        paired = reference.merge(
            frame,
            on=["target_quarter", "code"],
            suffixes=("_old", "_new"),
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        if not paired["_merge"].eq("both").all():
            raise AssertionError(f"Baseline prediction keys differ for {split}")
        np.testing.assert_allclose(
            paired["predicted_old"], paired["predicted_new"], atol=1e-10, rtol=0
        )
        expected = published[
            (published["model"] == "llm") & (published["split"] == split)
        ].iloc[0]
        calculated = metrics(frame)
        for key in ["n_obs", "ic_mean", "oos_r2", "mae", "rmse"]:
            np.testing.assert_allclose(
                calculated[key], expected[key], atol=1e-10, rtol=0
            )
        split_evidence[split] = {
            "n_obs": calculated["n_obs"],
            "n_quarters": calculated["n_quarters"],
            "ic_mean": calculated["ic_mean"],
            "oos_r2": calculated["oos_r2"],
            "mae": calculated["mae"],
            "prediction_max_abs_error": float(
                np.max(np.abs(paired["predicted_old"] - paired["predicted_new"]))
            ),
        }
    events = original_panel.drop_duplicates("issue")[[
        "issue", "target_quarter", "publication_date"
    ]].copy()
    publication = pd.to_datetime(events["publication_date"])
    target_periods = pd.PeriodIndex(events["target_quarter"], freq="Q")
    start_dates = pd.Series(target_periods.start_time, index=events.index)
    end_dates = pd.Series(target_periods.end_time, index=events.index)
    inside_target = (publication > start_dates) & (publication < end_dates)
    days_after_start = (publication - start_dates).dt.days
    coefficients, _skipped = e6.cross_section_ols(
        rebuilt, "sent_exposure", ["llm"], "llm", "llm"
    )
    coefficient_quarters = coefficients["target_quarter"].drop_duplicates()
    return {
        "passed": True,
        "absolute_tolerance": 1e-10,
        "feature_max_abs_errors": feature_errors,
        "splits": split_evidence,
        "source_sha256": {
            "e6_panel": _sha256(ROOT / "_temp" / "e6" / "e6_panel.csv"),
            "e6_quarterly_returns": _sha256(
                ROOT / "_temp" / "e6" / "industry_quarterly.csv"
            ),
            "e6_predictions": _sha256(cached_path),
            "e6_published_metrics": _sha256(published_path),
        },
        "timing": {
            "report_issues": int(len(events)),
            "publications_inside_target_quarter": int(inside_target.sum()),
            "days_after_target_start_min": int(days_after_start.min()),
            "days_after_target_start_max": int(days_after_start.max()),
            "days_after_target_start_mean": float(days_after_start.mean()),
            "prior_coefficient_quarters_before_2016Q1": int(
                coefficient_quarters.lt("2016Q1").sum()
            ),
            "prior_coefficient_quarters_before_2021Q1": int(
                coefficient_quarters.lt("2021Q1").sum()
            ),
            "prior_coefficient_quarters_before_2026Q3": int(
                coefficient_quarters.lt("2026Q3").sum()
            ),
            "interpretation": "E6-compatible retrospective exploratory comparison; target quarterly returns include pre-publication days.",
            "coefficient_updates": "Expanding quarterly coefficients, including completed validation/test quarters before each forecast.",
            "r2_reference": "Realized cross-sectional mean within each evaluated target quarter; not an implementable historical-mean forecast.",
        },
    }
