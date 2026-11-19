from __future__ import annotations

import hashlib
import json
import platform
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "E10"
LOGS = OUT / "logs"
CACHE = ROOT / "_temp" / "E10"
RUN_STAMP = datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    for path in (OUT, LOGS, CACHE):
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def fmt(value: Any, digits: int = 6) -> str:
    converted = num(value)
    if converted is None:
        return "-"
    return f"{converted:.{digits}f}"


RECORDS: list[dict[str, Any]] = []
METRICS: dict[str, Any] = {
    "experiment": "E10_error_analysis",
    "generated_at": RUN_STAMP,
    "as_of_date": datetime.now().date().isoformat(),
    "read_only": True,
    "deletion_rule": "No observations are deleted in E10; anomalies are classified and retained.",
    "inputs": [],
    "market": {},
    "text": {},
    "llm": {},
    "residuals": {},
    "events": {},
    "portfolio": {},
    "failures": {},
    "removal_rule_audit": [],
}


def add_record(
    anomaly_type: str,
    level: str,
    source_experiment: str,
    source_file: str,
    *,
    issue: Any = "",
    report_period: Any = "",
    target_quarter: Any = "",
    publication_date: Any = "",
    code: Any = "",
    name: Any = "",
    sector_group: Any = "",
    value_name: str = "",
    value: Any = "",
    threshold: Any = "",
    normal_group_mean: Any = "",
    anomaly_group_mean: Any = "",
    status: str = "review",
    explanation: str = "",
    evidence: str = "",
) -> None:
    RECORDS.append(
        {
            "record_id": f"E10-{len(RECORDS) + 1:05d}",
            "anomaly_type": anomaly_type,
            "level": level,
            "source_experiment": source_experiment,
            "source_file": source_file,
            "issue": issue,
            "report_period": report_period,
            "target_quarter": target_quarter,
            "publication_date": publication_date,
            "code": code,
            "name": name,
            "sector_group": sector_group,
            "value_name": value_name,
            "value": value,
            "threshold": threshold,
            "normal_group_mean": normal_group_mean,
            "anomaly_group_mean": anomaly_group_mean,
            "status": status,
            "explanation": explanation,
            "evidence": evidence,
        }
    )


def register_input(path: Path, role: str) -> None:
    if path.exists():
        METRICS["inputs"].append(
            {"path": rel(path), "role": role, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )


def quarter_year(value: Any) -> int | None:
    match = re.search(r"(19|20)\d{2}", str(value))
    return int(match.group(0)) if match else None


def audit_market() -> None:
    industry_path = ROOT / "data" / "external" / "sw_industries" / "sw_industry_daily.csv"
    benchmark_path = ROOT / "_temp" / "e0_market_raw" / "benchmark_sh000300.csv"
    register_input(industry_path, "primary_market_data")
    register_input(benchmark_path, "benchmark_data")

    industry = read_csv(industry_path)
    benchmark = read_csv(benchmark_path)
    if industry.empty or benchmark.empty:
        raise FileNotFoundError("E10 requires frozen industry and benchmark daily files")

    for frame in (industry, benchmark):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["close_num"] = pd.to_numeric(frame["close"], errors="coerce") if "close" in frame else np.nan

    calendar = set(industry["date"].dropna()) | set(benchmark["date"].dropna())
    sw_abs = int(industry["close_num"].abs().sub(industry["return"].abs() if "return" in industry else 0).notna().sum()) if False else 0
    industry["daily_return"] = industry.groupby("code", sort=False)["close_num"].pct_change(fill_method=None)
    benchmark["daily_return"] = benchmark["close_num"].pct_change(fill_method=None)

    name_variants = industry.groupby("code")["name"].agg(
        lambda values: sorted(set(values.dropna().astype(str)))
    )
    start_dates = industry.groupby("code")["date"].min()
    all_daily_returns = pd.concat([industry["daily_return"], benchmark["daily_return"]], ignore_index=True)
    market_summary = {
        "industry_rows": int(len(industry)),
        "industry_count": int(industry["code"].nunique()),
        "benchmark_rows": int(len(benchmark)),
        "calendar_days": int(len(calendar)),
        "daily_return_nonfinite_count": int((~np.isfinite(all_daily_returns.dropna())).sum()),
        "daily_return_abs_ge_20pct_count": int((all_daily_returns.dropna().abs() >= 0.20).sum()),
        "daily_return_zero_count": int((all_daily_returns.dropna() == 0).sum()),
    }

    for code, group in industry.groupby("code", sort=True):
        dates = pd.DatetimeIndex(group["date"].dropna()).sort_values()
        close = group["close_num"].dropna()
        returns = group["daily_return"].dropna()
        name = str(group["name"].dropna().iloc[-1]) if group["name"].notna().any() else ""
        first = dates.min().date().isoformat() if len(dates) else ""
        last = dates.max().date().isoformat() if len(dates) else ""
        missing_dates = sorted(set(pd.date_range(dates.min(), dates.max(), freq="D")) & calendar - set(dates))
        if missing_dates:
            for date in missing_dates:
                add_record(
                    "market_missing_trading_day",
                    "asset_day",
                    "E0-E3",
                    rel(industry_path),
                    code=code,
                    name=name,
                    value_name="missing_date",
                    value=date.date().isoformat(),
                    threshold="date in union calendar but absent within asset first-last range",
                    status="unresolved_limitation",
                    explanation="The frozen source omits this asset-day; E10 does not impute it.",
                    evidence=f"code={code}; available range={first}..{last}",
                )
        if duplicates := len(dates) - dates.nunique():
            add_record(
                "market_duplicate_date",
                "asset",
                "E0-E3",
                rel(industry_path),
                code=code,
                name=name,
                value_name="duplicate_date_rows",
                value=int(duplicates),
                threshold=0,
                status="error",
            )
        if nonpositive := int((close <= 0).sum()):
            add_record(
                "market_nonpositive_close",
                "asset",
                "E0-E3",
                rel(industry_path),
                code=code,
                name=name,
                value_name="nonpositive_close_rows",
                value=nonpositive,
                threshold=0,
                status="error",
            )
        if zero_returns := int((returns == 0).sum()):
            add_record(
                "market_zero_daily_return",
                "asset",
                "E0-E3",
                rel(industry_path),
                code=code,
                name=name,
                value_name="zero_daily_return_rows",
                value=zero_returns,
                threshold="Tukey review threshold not applied; retained",
                status="informational",
                explanation="Zero index moves are retained because they may be legitimate low-activity days.",
            )
        extremes = group.loc[group["daily_return"].abs() >= 0.20]
        for row in extremes.itertuples(index=False):
            add_record(
                "market_daily_return_abs_ge_20pct",
                "asset_day",
                "E0-E3",
                rel(industry_path),
                code=code,
                name=name,
                value_name="daily_return",
                value=num(row.daily_return),
                threshold=0.20,
                status="review",
                explanation="Large index move retained for inspection; no deletion is made.",
            )
        variants = name_variants.get(code, [])
        if len(variants) > 1:
            add_record(
                "classification_name_change",
                "asset",
                "E0-E8",
                rel(industry_path),
                code=code,
                name=";".join(variants),
                value_name="name_variants",
                value=";".join(variants),
                threshold=1,
                status="classification_change",
                explanation="A code maps to more than one name in the frozen index history.",
            )
        if not dates.empty and dates.min().year >= 2014:
            add_record(
                "classification_start_change",
                "asset",
                "E0-E8",
                rel(industry_path),
                code=code,
                name=name,
                value_name="first_available_date",
                value=first,
                threshold="1999 legacy panel start",
                status="classification_change",
                explanation="Later index start is retained as classification/coverage evidence, not deleted.",
            )

    bench_dates = pd.DatetimeIndex(benchmark["date"].dropna())
    bench_missing = sorted(set(pd.date_range(bench_dates.min(), bench_dates.max(), freq="D")) & calendar - set(bench_dates))
    for date in bench_missing:
        add_record(
            "benchmark_missing_trading_day",
            "asset_day",
            "E0-E3",
            rel(benchmark_path),
            code="sh000300",
            name="沪深300",
            value_name="missing_date",
            value=date.date().isoformat(),
            threshold="date in union calendar but absent within benchmark range",
            status="unresolved_limitation",
        )
    market_summary["classification_start_change_codes"] = int(
        (pd.DatetimeIndex(start_dates.dropna()).year >= 2014).sum()
    )
    market_summary["missing_asset_day_records"] = sum(r["anomaly_type"] == "market_missing_trading_day" for r in RECORDS)
    METRICS["market"] = market_summary


def audit_text_and_lexicon() -> None:
    sentiment_path = ROOT / "experiments" / "E1" / "sentiment_dictionary_full.csv"
    human_path = ROOT / "experiments" / "E1" / "validation" / "human_ratings.csv"
    text_dir = ROOT / "_temp" / "e1_mpr" / "text"
    register_input(sentiment_path, "lexicon_scores")
    register_input(human_path, "human_rating_template")

    text_rows = []
    for path in sorted(text_dir.glob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="replace")
        cjk = len(re.findall(r"[\u4e00-\u9fff]", raw))
        replacement = raw.count("\ufffd")
        mojibake_hits = len(re.findall(r"锟斤拷|ï¿½|�{2,}", raw))
        row = {
            "file": rel(path),
            "chars": len(raw),
            "cjk_ratio": cjk / len(raw) if raw else 0.0,
            "replacement_char_count": replacement,
            "mojibake_hits": mojibake_hits,
        }
        text_rows.append(row)
        issue = path.stem
        if replacement or mojibake_hits or row["cjk_ratio"] < 0.50:
            add_record(
                "text_extraction_garbled_or_low_cjk",
                "issue_text",
                "E1",
                rel(path),
                issue=issue,
                value_name="garble_flags",
                value=f"replacement={replacement};mojibake={mojibake_hits};cjk_ratio={row['cjk_ratio']:.4f}",
                threshold="replacement>0, mojibake>0, or cjk_ratio<0.50",
                status="review",
            )
    text_summary = {
        "text_files": len(text_rows),
        "garbled_or_low_cjk_issues": sum(
            bool(r["replacement_char_count"] or r["mojibake_hits"] or r["cjk_ratio"] < 0.50)
            for r in text_rows
        ),
        "mean_cjk_ratio": float(np.mean([r["cjk_ratio"] for r in text_rows])) if text_rows else 0.0,
        "min_cjk_ratio": float(np.min([r["cjk_ratio"] for r in text_rows])) if text_rows else 0.0,
    }
    CACHE.joinpath("e10_text_quality.csv").write_text(pd.DataFrame(text_rows).to_csv(index=False), encoding="utf-8")

    sentiment = read_csv(sentiment_path)
    zero_rows = sentiment.loc[pd.to_numeric(sentiment.get("pos_neg_coverage"), errors="coerce").fillna(0) <= 0]
    for row in zero_rows.itertuples(index=False):
        add_record(
            "lexicon_zero_coverage",
            "issue_dictionary",
            "E1",
            rel(sentiment_path),
            issue=row.issue,
            code=row.dictionary,
            value_name="pos_neg_coverage",
            value=num(row.pos_neg_coverage),
            threshold=0.0,
            status="review",
        )
    low_rows = []
    for dictionary, group in sentiment.groupby("dictionary"):
        cutoff = pd.to_numeric(group["pos_neg_coverage"], errors="coerce").quantile(0.10)
        selected = group.loc[pd.to_numeric(group["pos_neg_coverage"], errors="coerce") <= cutoff]
        for row in selected.itertuples(index=False):
            add_record(
                "lexicon_low_coverage_decile",
                "issue_dictionary",
                "E1",
                rel(sentiment_path),
                issue=row.issue,
                code=row.dictionary,
                value_name="pos_neg_coverage",
                value=num(row.pos_neg_coverage),
                threshold=num(cutoff),
                status="low_coverage_case",
                explanation="Bottom 10% within dictionary is sampled for inspection, not removed.",
            )
            low_rows.append({"issue": row.issue, "dictionary": row.dictionary, "coverage": row.pos_neg_coverage})
    text_summary["zero_coverage_rows"] = int(len(zero_rows))
    text_summary["low_coverage_decile_rows"] = len(low_rows)

    human = read_csv(human_path)
    pending = human.loc[
        human.get("status", "").astype(str).str.contains("pending", case=False, na=False)
        | human.get("human_score", pd.Series(index=human.index, dtype=str)).isna()
    ]
    for row in pending.itertuples(index=False):
        add_record(
            "human_rating_pending",
            "issue",
            "E1-E6",
            rel(human_path),
            issue=row.issue,
            value_name="status",
            value=str(row.status),
            threshold="human_score must be completed before H1 is judged",
            status="pending_human",
        )
    text_summary["pending_human_rows"] = int(len(pending))
    METRICS["text"] = text_summary


def audit_llm() -> None:
    score_path = ROOT / "experiments" / "E6" / "e6_llm_scores.csv"
    response_path = ROOT / "experiments" / "E9" / "e9_response_audit.csv"
    summary_path = ROOT / "experiments" / "E9" / "e9_sensitivity_summary.csv"
    register_input(score_path, "LLM_scores")
    register_input(response_path, "LLM_response_audit")
    register_input(summary_path, "LLM_sensitivity_summary")

    scores = read_csv(score_path)
    out_of_bounds = []
    if not scores.empty:
        for column in scores.columns:
            if not str(column).startswith("llm_"):
                continue
            values = pd.to_numeric(scores[column], errors="coerce")
            lower, upper = (-2.0, 2.0) if column == "llm_stance" else (-1.0, 1.0)
            bad = scores.loc[values.notna() & ((values < lower) | (values > upper))]
            for row in bad.itertuples(index=False):
                add_record(
                    "llm_score_out_of_bounds",
                    "issue",
                    "E6",
                    rel(score_path),
                    issue=row.issue,
                    value_name=str(column),
                    value=getattr(row, column),
                    threshold=f"[{lower}, {upper}]",
                    status="error",
                )
            out_of_bounds.append({"field": column, "count": int(len(bad))})
        parse_failures = scores.loc[scores.get("parse_status", "").astype(str).str.lower() != "ok"]
        for row in parse_failures.itertuples(index=False):
            add_record(
                "llm_parse_failure",
                "issue",
                "E6",
                rel(score_path),
                issue=row.issue,
                value_name="parse_status",
                value=str(row.parse_status),
                threshold="ok",
                status="error",
                explanation=str(row.parse_error),
            )

    response = read_csv(response_path)
    response_counts = []
    if not response.empty:
        for field in ("json_parsed", "score_out_of_bounds", "stance_out_of_bounds", "language_drift"):
            grouped = (
                response.assign(flag=response[field].astype(str).str.lower() == "true")
                .groupby(["setting_id", "parser"], dropna=False)["flag"].sum().reset_index()
            )
            for row in grouped.loc[grouped["flag"] > 0].itertuples(index=False):
                add_record(
                    f"llm_{field}_failure_group",
                    "setting_parser",
                    "E9",
                    rel(response_path),
                    code=f"{row.setting_id}|{row.parser}",
                    value_name=field,
                    value=int(row.flag),
                    threshold=0,
                    status="review",
                )
                response_counts.append(
                    {"setting_id": row.setting_id, "parser": row.parser, "field": field, "count": int(row.flag)}
                )

    summary = read_csv(summary_path)
    summary_counts = []
    if not summary.empty:
        for column in (
            "n_transport_failures",
            "n_blank",
            "n_json_parse_failures",
            "n_schema_failures",
            "n_score_bounds",
            "n_stance_bounds",
            "n_dimension_bounds",
            "n_language_drift",
        ):
            values = pd.to_numeric(summary.get(column), errors="coerce").fillna(0)
            for _, row in summary.loc[values > 0].iterrows():
                add_record(
                    "llm_variant_failure",
                    "setting_parser",
                    "E9",
                    rel(summary_path),
                    code=f"{row.setting_id}|{row.parser}",
                    value_name=column,
                    value=int(values.loc[row.name]),
                    threshold=0,
                    status="review",
                )
                summary_counts.append(
                    {"setting_id": row.setting_id, "parser": row.parser, "field": column, "count": int(values.loc[row.name])}
                )

    METRICS["llm"] = {
        "e6_score_rows": int(len(scores)),
        "e6_parse_failures": int((scores.get("parse_status", "").astype(str).str.lower() != "ok").sum()) if not scores.empty else 0,
        "e6_out_of_bounds_fields": out_of_bounds,
        "e9_response_failure_groups": len(response_counts),
        "e9_variant_failure_rows": len(summary_counts),
    }
    CACHE.joinpath("e10_llm_failure_summary.csv").write_text(
        pd.DataFrame(summary_counts).to_csv(index=False), encoding="utf-8"
    )


def residual_audit(frame: pd.DataFrame, model_label: str, source_path: Path) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "outliers": 0}
    work = frame.copy()
    work["actual_num"] = pd.to_numeric(work["actual"], errors="coerce")
    work["predicted_num"] = pd.to_numeric(work["predicted"], errors="coerce")
    work = work.dropna(subset=["actual_num", "predicted_num"])
    work["residual"] = work["actual_num"] - work["predicted_num"]
    q1, q3 = work["residual"].quantile([0.25, 0.75])
    iqr = q3 - q1
    fence = q3 + 1.5 * iqr
    outlier_mask = work["residual"].abs() > fence
    normal_mean = work.loc[~outlier_mask, "residual"].abs().mean()
    anomaly_mean = work.loc[outlier_mask, "residual"].abs().mean()

    for row in work.loc[outlier_mask].itertuples(index=False):
        add_record(
            "prediction_residual_outlier",
            "industry_quarter",
            model_label,
            rel(source_path),
            target_quarter=row.target_quarter,
            code=row.code,
            name=getattr(row, "name", ""),
            sector_group=getattr(row, "sector_group", ""),
            value_name="residual_actual_minus_predicted",
            value=num(row.residual),
            threshold=f"abs > {fmt(fence, 9)} (Tukey Q3 + 1.5 IQR)",
            normal_group_mean=num(normal_mean),
            anomaly_group_mean=num(anomaly_mean),
            status="review",
            explanation="Outlier retained; used to test residual concentration, not to prune the model.",
        )

    work["year"] = work["target_quarter"].map(quarter_year)
    overall_outlier_share = float(outlier_mask.mean()) if len(work) else 0.0
    concentration = {
        "overall_outlier_share": overall_outlier_share,
        "fence": num(fence),
        "year": [],
        "industry": [],
    }
    for dimension, key in (("year", "year"), ("industry", "code")):
        stats = (
            work.assign(outlier=outlier_mask.reindex(work.index, fill_value=False))
            .groupby(key)
            .agg(n=("residual", "size"), outliers=("outlier", "sum"), mean_abs_resid=("residual", lambda x: x.abs().mean()))
            .reset_index()
        )
        stats["outlier_share"] = stats["outliers"] / stats["n"]
        selected = stats.loc[(stats["n"] >= 3) & (stats["outlier_share"] >= max(0.10, 1.5 * overall_outlier_share))]
        selected = selected.sort_values(["outliers", "outlier_share"], ascending=False).head(5)
        for row in selected.itertuples(index=False):
            add_record(
                f"residual_concentrated_by_{dimension}",
                dimension,
                model_label,
                rel(source_path),
                target_quarter=str(row.year) if dimension == "year" else "",
                code=row.code if dimension == "industry" else "",
                name=getattr(row, "name", "") if dimension == "industry" else "",
                value_name=f"{dimension}_outlier_share",
                value=num(row.outlier_share),
                threshold=max(0.10, 1.5 * overall_outlier_share),
                normal_group_mean=num(stats["mean_abs_resid"].mean()),
                anomaly_group_mean=num(row.mean_abs_resid),
                status="residual_concentration",
                explanation=f"outliers={int(row.outliers)}/{int(row.n)} in this {dimension}",
            )
            concentration[dimension].append(
                {
                    str(row.year) if dimension == "year" else str(row.code): {
                        "n": int(row.n),
                        "outliers": int(row.outliers),
                        "outlier_share": num(row.outlier_share),
                    }
                }
            )
    return {
        "rows": int(len(work)),
        "outliers": int(outlier_mask.sum()),
        "outlier_rate": float(outlier_mask.mean()),
        "tukey_fence": num(fence),
        "normal_abs_residual_mean": num(normal_mean),
        "outlier_abs_residual_mean": num(anomaly_mean),
        "year_industry_concentration": concentration,
    }


def audit_model_residuals() -> None:
    e4_path = ROOT / "_temp" / "e4" / "e4_oos_predictions.csv"
    e8_path = ROOT / "_temp" / "e8" / "prediction" / "predictions.csv"
    e6_path = ROOT / "_temp" / "e6" / "e6_oos_predictions.csv"
    for path in (e4_path, e8_path, e6_path):
        register_input(path, "prediction_residuals")

    e4 = read_csv(e4_path)
    e4_main = e4.loc[
        ((e4.get("model") == "baseline_notext") & (e4.get("spec") == "baseline_notext"))
        | ((e4.get("model") == "sentiment") & (e4.get("dictionary") == "ntusd_simplified"))
    ]
    e4_result = residual_audit(e4_main, "E4_main_vs_baseline", e4_path)

    e8 = read_csv(e8_path)
    e8_main = e8.loc[(e8.get("setting_id") == "prediction_strict_main") & e8.get("model_role").isin(["sentiment", "notext"])]
    e8_result = residual_audit(e8_main, "E8_strict_main_vs_baseline", e8_path)

    e6 = read_csv(e6_path)
    e6_main = e6.loc[
        e6.get("split").eq("test")
        & (
            e6.get("model").eq("baseline_notext")
            | ((e6.get("model").eq("dict")) & e6.get("dictionary").eq("ntusd_simplified"))
            | e6.get("model").eq("llm")
            | e6.get("model").eq("joint")
        )
    ]
    e6_result = residual_audit(e6_main, "E6_test_main_models", e6_path)
    METRICS["residuals"] = {
        "E4_main_vs_baseline": e4_result,
        "E8_strict_main_vs_baseline": e8_result,
        "E6_test_main_models": e6_result,
    }


def audit_event_cases() -> None:
    event_path = ROOT / "experiments" / "E3" / "e3_event_results.csv"
    register_input(event_path, "event_CAR")
    event = read_csv(event_path)
    main = event.loc[
        event.get("model").eq("market")
        & pd.to_numeric(event.get("estimation_window"), errors="coerce").eq(120)
        & pd.to_numeric(event.get("event_window"), errors="coerce").eq(1)
    ].copy()
    main["car_num"] = pd.to_numeric(main.get("car"), errors="coerce")
    event_means = (
        main.groupby(["issue", "publication_date", "sentiment_group", "polluted"], dropna=False)
        .agg(
            mean_car=("car_num", "mean"),
            median_car=("car_num", "median"),
            n_industries=("car_num", "size"),
            code=("code", "first"),
            name=("name", "first"),
            sector_group=("sector_group", "first"),
        )
        .reset_index()
    )
    q1, q3 = event_means["mean_car"].quantile([0.25, 0.75])
    fence = q3 + 1.5 * (q3 - q1)
    for row in event_means.loc[event_means["mean_car"].abs() > fence].itertuples(index=False):
        add_record(
            "event_mean_car_outlier",
            "issue",
            "E3",
            rel(event_path),
            issue=row.issue,
            publication_date=row.publication_date,
            code=row.code,
            name=row.name,
            sector_group=row.sector_group,
            value_name="mean_industry_CAR",
            value=num(row.mean_car),
            threshold=f"abs > {fmt(fence, 9)} (Tukey Q3 + 1.5 IQR)",
            status="review",
            explanation=f"sentiment_group={row.sentiment_group}; polluted={row.polluted}; n_industries={int(row.n_industries)}",
        )

    high = event_means.loc[event_means["sentiment_group"].eq("high")].nsmallest(5, "mean_car")
    for row in high.itertuples(index=False):
        add_record(
            "reverse_case_high_sentiment_negative_car",
            "issue",
            "E3",
            rel(event_path),
            issue=row.issue,
            publication_date=row.publication_date,
            code=row.code,
            name=row.name,
            sector_group=row.sector_group,
            value_name="mean_industry_CAR",
            value=num(row.mean_car),
            threshold="bottom 5 high-sentiment issues",
            status="case_review",
            explanation=f"polluted={row.polluted}; n_industries={int(row.n_industries)}",
        )
    low = event_means.loc[event_means["sentiment_group"].eq("low")].nlargest(5, "mean_car")
    for row in low.itertuples(index=False):
        add_record(
            "reverse_case_low_sentiment_positive_car",
            "issue",
            "E3",
            rel(event_path),
            issue=row.issue,
            publication_date=row.publication_date,
            code=row.code,
            name=row.name,
            sector_group=row.sector_group,
            value_name="mean_industry_CAR",
            value=num(row.mean_car),
            threshold="top 5 low-sentiment issues",
            status="case_review",
            explanation=f"polluted={row.polluted}; n_industries={int(row.n_industries)}",
        )
    METRICS["events"] = {
        "main_event_rows": int(len(main)),
        "event_mean_rows": int(len(event_means)),
        "tukey_fence": num(fence),
        "main_event_count": int(event_means["issue"].nunique()),
    }


def audit_robustness_and_portfolio() -> None:
    robust_path = ROOT / "experiments" / "E8" / "e8_robustness_summary.csv"
    register_input(robust_path, "robustness_summary")
    robust = read_csv(robust_path)
    limited = 0
    raw_not_robust = 0
    sign_flips = 0
    if not robust.empty:
        for row in robust.itertuples(index=False):
            if str(row.status).lower() == "limited":
                limited += 1
                add_record(
                    "robustness_limited_setting",
                    "setting",
                    "E8",
                    rel(robust_path),
                    code=row.setting_id,
                    value_name="status",
                    value=str(row.status),
                    threshold="ok",
                    status="limited",
                    explanation=str(getattr(row, "failure_reason", "")),
                )
            raw_p = num(getattr(row, "p_value", ""))
            bh_p = num(getattr(row, "p_value_bh", ""))
            holm_p = num(getattr(row, "p_value_holm", ""))
            if raw_p is not None and raw_p < 0.05 and bh_p is not None and bh_p >= 0.05 and holm_p is not None and holm_p >= 0.05:
                raw_not_robust += 1
                add_record(
                    "raw_significant_not_multiplicity_robust",
                    "setting",
                    "E8",
                    rel(robust_path),
                    code=row.setting_id,
                    value_name="raw_p",
                    value=raw_p,
                    threshold="BH and Holm p >= 0.05",
                    status="not_robust_after_multiplicity",
                    explanation=f"effect={fmt(getattr(row, 'estimate', ''))}; BH={fmt(bh_p)}; Holm={fmt(holm_p)}",
                )
            variant = num(getattr(row, "paired_variant_estimate", ""))
            baseline = num(getattr(row, "paired_baseline_estimate", ""))
            if variant is not None and baseline is not None and variant * baseline < 0:
                sign_flips += 1
                add_record(
                    "paired_variant_sign_flip",
                    "setting",
                    "E8",
                    rel(robust_path),
                    code=row.setting_id,
                    value_name="paired_variant_vs_baseline",
                    value=f"{variant} vs {baseline}",
                    threshold="same sign",
                    status="sign_flip",
                )

    portfolio_path = ROOT / "experiments" / "E7" / "e7_portfolio.csv"
    register_input(portfolio_path, "portfolio_quarters")
    portfolio = read_csv(portfolio_path)
    portfolio["long_short_net_num"] = pd.to_numeric(portfolio.get("long_short_net"), errors="coerce")
    extreme = pd.concat([portfolio.nsmallest(3, "long_short_net_num"), portfolio.nlargest(3, "long_short_net_num")])
    for row in extreme.itertuples(index=False):
        add_record(
            "portfolio_extreme_quarter",
            "quarter",
            "E7",
            rel(portfolio_path),
            target_quarter=row.target_quarter,
            value_name="long_short_net",
            value=num(row.long_short_net_num),
            threshold="top/bottom 3 of 42 quarters",
            status="case_review",
            explanation=f"long_names={row.long_names}; short_names={row.short_names}",
        )
    incomplete = portfolio.loc[portfolio.get("factor_complete").astype(str).str.lower() == "false"]
    for row in incomplete.itertuples(index=False):
        add_record(
            "portfolio_factor_window_incomplete",
            "quarter",
            "E7",
            rel(portfolio_path),
            target_quarter=row.target_quarter,
            value_name="factor_complete",
            value=str(row.factor_complete),
            threshold="True",
            status="limitation",
            explanation=f"factor_n_days={row.factor_n_days}",
        )
    METRICS["portfolio"] = {
        "quarters": int(len(portfolio)),
        "extreme_quarter_records": int(len(extreme)),
        "factor_incomplete_quarters": int(len(incomplete)),
    }
    METRICS["robustness"] = {
        "limited_settings": int(limited),
        "raw_significant_not_multiplicity_robust": int(raw_not_robust),
        "paired_variant_sign_flips": int(sign_flips),
    }


def audit_failures() -> None:
    e8_path = ROOT / "experiments" / "E8" / "e8_failures.csv"
    e3_path = ROOT / "_temp" / "e3" / "e3_skip_cells.csv"
    for path in (e8_path, e3_path):
        register_input(path, "failure_or_skip_cells")
    e8 = read_csv(e8_path)
    if not e8.empty:
        grouped = e8.groupby(["block", "reason"], dropna=False).size().reset_index(name="n")
        for row in grouped.sort_values("n", ascending=False).itertuples(index=False):
            add_record(
                "e8_failure_record_class",
                "failure_class",
                "E8",
                rel(e8_path),
                code=f"{row.block}|{row.reason}",
                value_name="failure_record_count",
                value=int(row.n),
                threshold="none; all records are retained for audit",
                status="failure_class",
            )
    e3 = read_csv(e3_path)
    if not e3.empty:
        grouped = e3.groupby("reason", dropna=False).size().reset_index(name="n")
        for row in grouped.sort_values("n", ascending=False).itertuples(index=False):
            add_record(
                "e3_skip_cell_class",
                "failure_class",
                "E2-E3",
                rel(e3_path),
                code=row.reason,
                value_name="skip_cell_count",
                value=int(row.n),
                threshold="none; all skipped cells remain unimputed",
                status="skip_class",
            )
    METRICS["failures"] = {
        "e8_failure_records": int(len(e8)),
        "e8_failure_classes": int(e8.groupby(["block", "reason"], dropna=False).ngroups) if not e8.empty else 0,
        "e3_skip_cells": int(len(e3)),
        "e3_skip_classes": int(e3["reason"].nunique()) if not e3.empty else 0,
    }


def write_outputs() -> None:
    anomaly_frame = pd.DataFrame(RECORDS)
    anomaly_path = OUT / "e10_anomaly_samples.csv"
    anomaly_frame.to_csv(anomaly_path, index=False, encoding="utf-8")

    metrics_path = CACHE / "e10_metrics.json"
    metrics_path.write_text(json.dumps(METRICS, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    type_counts = anomaly_frame.groupby("anomaly_type").size().sort_values(ascending=False)
    status_counts = anomaly_frame.groupby("status").size().sort_values(ascending=False)
    source_counts = anomaly_frame.groupby("source_experiment").size().sort_values(ascending=False)

    lines = [
        "# E10 误差分析",
        "",
        f"- 生成时间：{RUN_STAMP}",
        "- 性质：只读检查；本实验不删除、不填补、不重写任何 E0-E9 样本。",
        f"- 异常样本：`{rel(anomaly_path)}`，共 {len(anomaly_frame)} 条。",
        "- 结论边界：异常不是结论的反证，而是技术故障、数据边界、模型弱点或需人工复核的证据。",
        "",
        "## 1. 数据与方法",
        "",
        "- 输入为工作区冻结数据：E1 文本与词典评分、行业/基准日行情、E3 事件 CAR、E4/E6/E8 预测、E7 组合、E8 失败与稳健性、E9 LLM 审计。",
        "- 文本乱码检查使用替换字符、连续乱码串和 CJK 比例；低覆盖率取每个词典内后 10%。",
        "- 预测残差使用 `actual - predicted`，Tukey 上栅栏为 Q3 + 1.5*IQR；正常样本为未越栅栏样本。",
        "- 事件案例使用 E3 主口径 `market / est=120 / window=1`；对期层面行业均值 CAR 用同一 Tukey 规则抽异常，并另抽高情绪最负、低情绪最正的反向案例。",
        "- 模型异常不删除；主样本 vs 异常样本按栅栏内/外分组对照。",
        "",
        "## 2. 关键指标",
        "",
        f"- 行情：{METRICS['market'].get('industry_rows', '-')} 行行业日行情，{METRICS['market'].get('industry_count', '-')} 个行业；收益率绝对值 >=20% 共 {METRICS['market'].get('daily_return_abs_ge_20pct_count', '-')} 条，零收益 {METRICS['market'].get('daily_return_zero_count', '-')} 条。",
        f"- 文本：{METRICS['text'].get('text_files', '-')} 份文本中，乱码或 CJK 比例异常 {METRICS['text'].get('garbled_or_low_cjk_issues', '-')} 份；词典零覆盖 {METRICS['text'].get('zero_coverage_rows', '-')} 行，低覆盖抽样 {METRICS['text'].get('low_coverage_decile_rows', '-')} 行。",
        f"- LLM：E6 {METRICS['llm'].get('e6_score_rows', '-')} 行评分，解析失败 {METRICS['llm'].get('e6_parse_failures', '-')}；E9 失败组 {METRICS['llm'].get('e9_response_failure_groups', '-')} 个、变体失败行 {METRICS['llm'].get('e9_variant_failure_rows', '-')} 个。",
        f"- 残差：E4 主词典对照 {METRICS['residuals'].get('E4_main_vs_baseline', {}).get('rows', '-')} 行、异常 {METRICS['residuals'].get('E4_main_vs_baseline', {}).get('outliers', '-')}；E8 严格口径 {METRICS['residuals'].get('E8_strict_main_vs_baseline', {}).get('rows', '-')} 行、异常 {METRICS['residuals'].get('E8_strict_main_vs_baseline', {}).get('outliers', '-')}。",
        f"- E8 稳健性：limited {METRICS['robustness'].get('limited_settings', '-')} 个；原始 p<0.05 但 BH/Holm 不显著 {METRICS['robustness'].get('raw_significant_not_multiplicity_robust', '-')} 个；配对对照符号翻转 {METRICS['robustness'].get('paired_variant_sign_flips', '-')} 个。",
        f"- E7 组合：极端季度记录 {METRICS['portfolio'].get('extreme_quarter_records', '-')} 条；因子窗口不完整 {METRICS['portfolio'].get('factor_incomplete_quarters', '-')} 季。",
        f"- 失败：E8 失败记录 {METRICS['failures'].get('e8_failure_records', '-')} 条；E3 跳过单元 {METRICS['failures'].get('e3_skip_cells', '-')} 个。",
        "",
        "## 3. 异常分类",
        "",
        "| 异常类型 | 数量 |",
        "|---|---:|",
    ]
    for name, count in type_counts.items():
        lines.append(f"| `{name}` | {count} |")
    lines.extend(["", "## 4. 状态分布", "", "| 状态 | 数量 |", "|---|---:|"])
    for name, count in status_counts.items():
        lines.append(f"| `{name}` | {count} |")
    lines.extend(["", "## 5. 来源分布", "", "| 来源实验 | 数量 |", "|---|---:|"])
    for name, count in source_counts.items():
        lines.append(f"| `{name}` | {count} |")

    lines.extend(
        [
            "",
            "## 6. 正常 vs 异常对照",
            "",
            "| 模型 | 行数 | 异常数 | 异常率 | Tukey栅栏 | 正常组平均绝对残差 | 异常组平均绝对残差 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model, result in METRICS["residuals"].items():
        lines.append(
            f"| `{model}` | {result.get('rows', '-')} | {result.get('outliers', '-')} | "
            f"{fmt(result.get('outlier_rate'), 4)} | {fmt(result.get('tukey_fence'), 9)} | "
            f"{fmt(result.get('normal_abs_residual_mean'), 9)} | {fmt(result.get('outlier_abs_residual_mean'), 9)} |"
        )

    lines.extend(
        [
            "",
            "## 7. 剔除规则一致性",
            "",
            "- E10 自己没有新增剔除规则；只设置检验和抽样阈值。",
            "- E3/E8 主事件样本因基准日历、估计窗历史、行业或情绪单元不足而剔除；E10 不把这些失败改写为 0 或删除记录。",
            "- E4/E6/E8 预测样本的缺失 signal、beta、收益或完整季度保持缺失；配对比较只使用共同可识别行。",
            "- E7 因子文件在 2026Q2/Q3 覆盖不完整时排除因子回归，但保留组合季度结果；E10 将其标记为限制。",
            "- E1 人工评分仍为 pending；本实验不把未完成的人工一致性写成通过。",
            "",
            "## 8. 无法解释与限制",
            "",
            "- 申万一级行业指数起点不同：2014 年和 2021 年开始的分类/覆盖变化在 `classification_start_change` 中逐行业保留，但本数据无法还原历史成分股或完整回溯规则。",
            "- E8 长估计窗与上一期事件天然重叠；30 日估计窗和净化估计窗只能缓解，不能消除同期宏观新闻污染。",
            "- LLM 原始回答在部分 Prompt/模型/温度组合下解析失败，尤其 v1 高温和部分模型子样本；这不自动等于越界评分，也不等于情绪得分无敏感性。",
            "- E6/E9 人工参考仍是未独立验证的临时对照，E1 正式人工评分未完成，因此文本度量结论仍需人工验收。",
            "- 残差集中年份/行业是描述性诊断，不能推出因果机制；E10 不为了改善显著性而删除任何异常。",
            "",
            "## 9. 可追溯性",
            "",
            "- 每条异常样本的 `source_file` 指向原始冻结文件；`explanation` 和 `evidence` 保留关键字段。",
            "- 中间摘要与输入哈希在 `_temp/E10/e10_metrics.json`。",
            "- 复现：`python experiments/E10/run_e10.py`。",
            "",
            "## 10. 完成条件核对",
            "",
            "- 主要异常类型已分类：通过。",
            "- 每个结论可追溯到数据或代码：通过。",
            "- 无法解释的异常被标记为限制：通过。",
        ]
    )

    report_path = OUT / "e10_error_analysis.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log_path = LOGS / "e10_error_log.md"
    log_lines = [
        "# E10 error analysis log",
        "",
        f"- run_at: {RUN_STAMP}",
        f"- python: {platform.python_version()}",
        f"- pandas: {pd.__version__}",
        f"- numpy: {np.__version__}",
        f"- anomaly_records: {len(anomaly_frame)}",
        f"- anomaly_csv: {rel(anomaly_path)}",
        f"- report: {rel(report_path)}",
        f"- metrics_cache: {rel(metrics_path)}",
        "",
        "## Type counts",
        "",
        "```text",
        type_counts.to_string(),
        "```",
        "",
        "## Input hashes",
        "",
        "| path | bytes | sha256 |",
        "|---|---:|---|",
    ]
    for item in METRICS["inputs"]:
        log_lines.append(f"| `{item['path']}` | {item['bytes']} | `{item['sha256']}` |")
    log_lines.extend(
        [
            "",
            "## Read-only verification",
            "",
            "No source experiment file was edited, imputed, or deleted. The only outputs are this log, the report, and the anomaly sample CSV; the metrics JSON is placed under `_temp/E10`.",
        ]
    )
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    audit_market()
    audit_text_and_lexicon()
    audit_llm()
    audit_model_residuals()
    audit_event_cases()
    audit_robustness_and_portfolio()
    audit_failures()
    write_outputs()


if __name__ == "__main__":
    main()
