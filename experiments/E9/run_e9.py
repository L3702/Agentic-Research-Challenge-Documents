"""Recompute exploratory E9 sensitivity results from frozen responses, offline."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import scipy
from scipy import stats

from e9_core import PARSER_RULES, error_status, language_check, parse_output, validate_object
from e9_evaluation import assert_e6_reproduction, evaluate_scores, metrics


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "E9"
CACHE = ROOT / "_temp" / "e9"
SNAPSHOT = CACHE / "source_snapshot"
MAIN_PARSER = "e6_lenient"
SETTINGS = {
    "baseline_e6": (None, "v1_e6.txt", "qwen3.8-27b", 0.0, True, None),
    "prompt_v1_legacy": ("v1_base", "v1_legacy_e9.txt", "qwen3.8-27b", 0.0, True, "baseline_e6"),
    "prompt_v2_minimal": ("v2_minimal", "v2_minimal_legacy_e9.txt", "qwen3.8-27b", 0.0, False, "baseline_e6"),
    "prompt_v1_legacy_temp07": ("v1_temp07", "v1_legacy_e9.txt", "qwen3.8-27b", 0.7, True, "prompt_v1_legacy"),
    "prompt_v2_minimal_temp07": ("v2_temp07", "v2_minimal_legacy_e9.txt", "qwen3.8-27b", 0.7, False, "prompt_v2_minimal"),
    "model_glm53_subset": ("model_glm53", "v2_minimal_legacy_e9.txt", "glm-5.3", 0.0, False, "prompt_v2_minimal"),
    "model_gpt4o_subset": ("model_gpt4o", "v2_minimal_legacy_e9.txt", "gpt-4o", 0.0, False, "prompt_v2_minimal"),
    "model_deepseek_subset": ("model_deepseek", "v2_minimal_legacy_e9.txt", "deepseek-v4-flash", 0.0, False, "prompt_v2_minimal"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_json(path: Path, value) -> None:
    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(value) for value in item]
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (float, np.floating)):
            return float(item) if math.isfinite(item) else None
        return item

    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig", float_format="%.12g", na_rep="")


def load_records() -> tuple[dict, list[dict]]:
    inventory = json.loads((SNAPSHOT / "inventory.json").read_text(encoding="utf-8"))
    hashes = {Path(row["source"]).name: row["sha256"] for row in inventory}
    result, used = {}, []
    for setting, (prefix, _, model, temperature, _, _) in SETTINGS.items():
        folder = SNAPSHOT / ("e6_llm_raw" if prefix is None else "e9_llm_raw")
        paths = sorted(folder.glob("*.json" if prefix is None else f"{prefix}__*.json"))
        records = {}
        for path in paths:
            digest = sha256(path)
            if digest != hashes[path.name]:
                raise ValueError(f"Frozen input changed: {relative(path)}")
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            issue = record["issue"]
            if issue in records:
                raise ValueError(f"Duplicate issue {setting}/{issue}")
            assert record["model"] == model
            assert float(record["temperature"]) == temperature
            records[issue] = record
            used.append({"setting_id": setting, "issue": issue, "source_path": relative(path), "sha256": digest})
        expected = 24 if setting.startswith("model_") else 103
        assert len(records) == expected, (setting, len(records), expected)
        result[setting] = records
    return result, used


def audit_responses(records: dict) -> pd.DataFrame:
    rows = []
    for setting, collection in records.items():
        require_dimensions = SETTINGS[setting][4]
        for issue, record in collection.items():
            text = record.get("raw_output_text") or ""
            transport = error_status(record)
            usage = record.get("usage") or {}
            input_tokens = usage.get("input_tokens", record.get("input_tokens"))
            output_tokens = usage.get("output_tokens", record.get("output_tokens"))
            metered = not transport and isinstance(input_tokens, (float, int)) and isinstance(output_tokens, (float, int)) and input_tokens + output_tokens > 0
            for parser in PARSER_RULES:
                obj, parse_error, repaired = parse_output(text, parser) if not transport else (None, None, False)
                validation = validate_object(obj, require_dimensions) if obj is not None else validate_object(None, require_dimensions)
                language = language_check(obj)
                valid = not transport and obj is not None and validation["schema_valid"]
                status = transport or ("valid" if valid else "blank_output" if not text.strip() else "json_failure" if obj is None else "schema_failure")
                rows.append({
                    "setting_id": setting, "parser": parser, "issue": issue,
                    "model_alias": record["model"], "temperature": record["temperature"],
                    "timestamp": record.get("timestamp"), "response_status": status,
                    "transport_failure": bool(transport), "blank_output": not transport and not text.strip(),
                    "json_parsed": obj is not None, "json_parse_failure": not transport and obj is None,
                    "schema_valid": bool(valid), "brace_repaired": repaired,
                    "parse_error": parse_error, "schema_error": validation["schema_error"] if obj is not None else "",
                    "score_out_of_bounds": validation["score_out_of_bounds"],
                    "stance_out_of_bounds": validation["stance_out_of_bounds"],
                    "dimension_out_of_bounds": validation["dimension_out_of_bounds"],
                    "llm_sentiment": validation["score"] if valid else None,
                    "llm_stance": validation["stance"] if valid else None,
                    **language,
                    "input_tokens": input_tokens if metered else None,
                    "output_tokens": output_tokens if metered else None,
                    "usage_metered": metered,
                    "latency_sec": record.get("latency_sec") if not transport else None,
                    "max_output_tokens": record.get("max_output_tokens"),
                    "output_at_cap": bool(metered and output_tokens >= record.get("max_output_tokens", math.inf)),
                    "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
                    "raw_output_chars": len(text),
                })
    return pd.DataFrame(rows)


def safe_correlation(x, y, method: str) -> float:
    if len(x) < 3 or pd.Series(x).nunique() < 2 or pd.Series(y).nunique() < 2:
        return np.nan
    return float((stats.pearsonr if method == "pearson" else stats.spearmanr)(x, y).statistic)


def score_comparison(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setting in SETTINGS:
        comparators = list(dict.fromkeys(["baseline_e6", SETTINGS[setting][5] or "baseline_e6"]))
        for parser in PARSER_RULES:
            current = scores[(scores.setting_id == setting) & (scores.parser == parser)]
            for comparator in comparators:
                reference = scores[(scores.setting_id == comparator) & (scores.parser == MAIN_PARSER)]
                pair = current.merge(reference, on="issue", suffixes=("_variant", "_reference"))
                rows.append({
                    "setting_id": setting, "parser": parser, "comparator": comparator, "n_pairs": len(pair),
                    "sentiment_pearson": safe_correlation(pair.llm_sentiment_variant, pair.llm_sentiment_reference, "pearson"),
                    "sentiment_spearman": safe_correlation(pair.llm_sentiment_variant, pair.llm_sentiment_reference, "spearman"),
                    "sentiment_mae": (pair.llm_sentiment_variant - pair.llm_sentiment_reference).abs().mean(),
                    "stance_exact_agreement": (pair.llm_stance_variant == pair.llm_stance_reference).mean() if len(pair) else np.nan,
                })
    return pd.DataFrame(rows)


def weighted_kappa(x: pd.Series, y: pd.Series) -> float:
    if not len(x):
        return np.nan
    observed = np.mean((np.asarray(x) - np.asarray(y)) ** 2)
    expected = np.mean((np.asarray(x)[:, None] - np.asarray(y)[None, :]) ** 2)
    return float(1 - observed / expected) if expected > 0 else np.nan


def reference_agreement(scores: pd.DataFrame) -> pd.DataFrame:
    reference = pd.read_csv(ROOT / "_temp" / "e6_human_spotcheck.csv")
    assert len(reference) == 15 and reference["issue"].is_unique
    rows = []
    for setting in SETTINGS:
        for parser in PARSER_RULES:
            current = scores[(scores.setting_id == setting) & (scores.parser == parser)]
            joined = current.merge(reference, on="issue", how="inner")
            for cohort in ("all", "pilot", "validation"):
                pair = joined if cohort == "all" else joined[joined.sample_type == cohort]
                for measure in ("sentiment_x2", "stance"):
                    automatic = pair.llm_sentiment * 2 if measure == "sentiment_x2" else pair.llm_stance
                    rows.append({
                        "setting_id": setting, "parser": parser, "reference_kind": "e6_provisional_reference_unverified",
                        "verified_human": False, "annotation_cohort": cohort, "measure": measure,
                        "n_pairs": len(pair), "mae": (automatic - pair.human_score).abs().mean(),
                        "pearson": safe_correlation(automatic, pair.human_score, "pearson"),
                        "spearman": safe_correlation(automatic, pair.human_score, "spearman"),
                        "exact_agreement": (automatic == pair.human_score).mean() if len(pair) and measure == "stance" else np.nan,
                        "quadratic_weighted_kappa": weighted_kappa(automatic, pair.human_score) if measure == "stance" else np.nan,
                    })
    return pd.DataFrame(rows)


def evaluate_all(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    memo, results, all_predictions = {}, [], []

    def evaluate(frame):
        key = tuple(frame[["issue", "llm_sentiment"]].sort_values("issue").itertuples(index=False, name=None))
        if key not in memo:
            memo[key] = evaluate_scores(frame[["issue", "llm_sentiment"]])
        return memo[key]

    def add_rows(setting, parser, scope, comparator, outputs, references=None, training_n=None):
        for split, predictions in outputs.items():
            reference = references[split] if references is not None else None
            if reference is not None:
                if predictions.empty or reference.empty:
                    paired, ref_paired = predictions.iloc[:0], reference.iloc[:0]
                else:
                    keys = predictions[["target_quarter", "code"]].merge(reference[["target_quarter", "code"]], on=["target_quarter", "code"])
                    paired = predictions.merge(keys, on=["target_quarter", "code"])
                    ref_paired = reference.merge(keys, on=["target_quarter", "code"])
                    assert paired[["target_quarter", "code"]].duplicated().sum() == 0
                current_metrics, reference_metrics = metrics(paired), metrics(ref_paired)
            else:
                paired, current_metrics, reference_metrics = predictions, metrics(predictions), {}
            row = {"setting_id": setting, "parser": parser, "scope": scope, "comparator": comparator,
                   "split": split, "training_score_issues": training_n, "status": "available" if len(paired) else "no_forecasts",
                   "evaluation_kind": "e6_compatible_retrospective_expanding", **current_metrics}
            for key, value in reference_metrics.items():
                row[f"reference_{key}"] = value
            for key in ("ic_mean", "oos_r2", "mae"):
                if reference_metrics:
                    row[f"delta_{key}"] = current_metrics[key] - reference_metrics[key]
            results.append(row)
            if scope == "native" and len(predictions):
                all_predictions.append(predictions.assign(setting_id=setting, parser=parser))

    baseline_scores = scores[(scores.setting_id == "baseline_e6") & (scores.parser == MAIN_PARSER)]
    baseline_predictions = evaluate(baseline_scores)
    for setting in SETTINGS:
        for parser in PARSER_RULES:
            current = scores[(scores.setting_id == setting) & (scores.parser == parser)]
            outputs = evaluate(current)
            add_rows(setting, parser, "native", "", outputs, training_n=len(current))
            add_rows(setting, parser, "common_predictions_e6", "baseline_e6", outputs, baseline_predictions, len(current))
            comparator = SETTINGS[setting][5]
            if comparator:
                reference = scores[(scores.setting_id == comparator) & (scores.parser == MAIN_PARSER)]
                common_issues = set(current.issue) & set(reference.issue)
                current_controlled = current[current.issue.isin(common_issues)]
                reference_controlled = reference[reference.issue.isin(common_issues)]
                controlled, ref_controlled = evaluate(current_controlled), evaluate(reference_controlled)
                add_rows(setting, parser, "common_score_history", comparator, controlled, ref_controlled, len(common_issues))
    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    return pd.DataFrame(results), predictions


def make_summary(audit, comparisons, agreement, oos) -> pd.DataFrame:
    rows = []
    baseline_audit = audit[(audit.setting_id == "baseline_e6") & (audit.parser == MAIN_PARSER)]
    for (setting, parser), frame in audit.groupby(["setting_id", "parser"], sort=False):
        returned = frame[~frame.transport_failure]
        comparator = SETTINGS[setting][5] or "baseline_e6"
        comparison = comparisons[(comparisons.setting_id == setting) & (comparisons.parser == parser) & (comparisons.comparator == "baseline_e6")].iloc[0]
        controlled = comparisons[(comparisons.setting_id == setting) & (comparisons.parser == parser) & (comparisons.comparator == comparator)].iloc[0]
        row = {
            "setting_id": setting, "parser": parser, "analysis_status": "exploratory",
            "model_alias": SETTINGS[setting][2], "immutable_model_version": None,
            "prompt_file": f"prompts/{SETTINGS[setting][1]}", "temperature": SETTINGS[setting][3],
            "controlled_comparator": comparator, "n_intended": len(frame), "n_valid": int(frame.schema_valid.sum()),
            "n_transport_failures": int(frame.transport_failure.sum()), "n_http429": int((frame.response_status == "http_429").sum()),
            "n_returned": len(returned), "n_blank": int(frame.blank_output.sum()),
            "n_json_parse_failures": int(frame.json_parse_failure.sum()),
            "json_parse_failure_rate_returned": frame.json_parse_failure.sum() / len(returned) if len(returned) else np.nan,
            "total_unusable_rate_intended": 1 - frame.schema_valid.sum() / len(frame),
            "n_schema_failures": int((frame.response_status == "schema_failure").sum()),
            "n_score_bounds": int(frame.score_out_of_bounds.sum()), "n_stance_bounds": int(frame.stance_out_of_bounds.sum()),
            "n_dimension_bounds": int(frame.dimension_out_of_bounds.sum()), "n_brace_repaired": int(frame.brace_repaired.sum()),
            "n_language_applicable": int(frame.language_applicable.sum()), "n_language_drift": int(frame.language_drift.sum()),
            "language_drift_rate_applicable": frame.language_drift.sum() / frame.language_applicable.sum() if frame.language_applicable.sum() else np.nan,
            "n_metered": int(frame.usage_metered.sum()), "observed_input_tokens": frame.input_tokens.sum(min_count=1),
            "observed_output_tokens": frame.output_tokens.sum(min_count=1), "observed_latency_sum_sec": frame.latency_sec.sum(min_count=1),
            "n_output_at_cap": int(frame.output_at_cap.sum()), "monetary_cost": None,
            "verified_human_n": 0, "verified_human_agreement": None, "verified_human_status": "pending_independent_team_labels",
            "score_pairs_e6": comparison.n_pairs, "score_pearson_e6": comparison.sentiment_pearson,
            "score_spearman_e6": comparison.sentiment_spearman, "score_mae_e6": comparison.sentiment_mae,
            "score_pairs_controlled": controlled.n_pairs, "score_pearson_controlled": controlled.sentiment_pearson,
        }
        token_pairs = frame[frame.usage_metered].merge(baseline_audit[baseline_audit.usage_metered], on="issue", suffixes=("_variant", "_reference"))
        row["cost_token_pairs_e6"] = len(token_pairs)
        for token in ("input_tokens", "output_tokens"):
            reference_tokens = token_pairs[f"{token}_reference"].sum()
            row[f"paired_{token}_change_e6"] = token_pairs[f"{token}_variant"].sum() / reference_tokens - 1 if reference_tokens else np.nan
        for measure in ("sentiment_x2", "stance"):
            ref = agreement[(agreement.setting_id == setting) & (agreement.parser == parser) & (agreement.annotation_cohort == "all") & (agreement.measure == measure)].iloc[0]
            row[f"provisional_{measure}_n"] = ref.n_pairs
            row[f"provisional_{measure}_mae"] = ref.mae
            row[f"provisional_{measure}_spearman"] = ref.spearman
        for split in ("validation", "test"):
            for scope, label in (("native", "native"), ("common_predictions_e6", "paired_e6"), ("common_score_history", "controlled")):
                selected = oos[(oos.setting_id == setting) & (oos.parser == parser) & (oos.split == split) & (oos.scope == scope)]
                if not selected.empty:
                    for key in ("n_obs", "n_quarters", "ic_mean", "oos_r2", "mae"):
                        row[f"{split}_{label}_{key}"] = selected.iloc[0][key]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    records, used_sources = load_records()
    audit = audit_responses(records)
    scores = audit.loc[audit.schema_valid, ["setting_id", "parser", "issue", "llm_sentiment", "llm_stance"]].copy()
    assert not scores.duplicated(["setting_id", "parser", "issue"]).any()
    baseline = scores[(scores.setting_id == "baseline_e6") & (scores.parser == MAIN_PARSER)]
    validation = assert_e6_reproduction(baseline[["issue", "llm_sentiment"]])
    comparisons = score_comparison(scores)
    agreement = reference_agreement(scores)
    print("Input and parser checks complete; evaluating retrospective OOS...", flush=True)
    oos, predictions = evaluate_all(scores)
    summary = make_summary(audit, comparisons, agreement, oos)
    for name, frame in [
        ("e9_response_audit.csv", audit), ("e9_scores.csv", scores),
        ("e9_score_comparisons.csv", comparisons), ("e9_reference_agreement.csv", agreement),
        ("e9_oos_metrics.csv", oos), ("e9_oos_predictions.csv", predictions),
        ("e9_sensitivity_summary.csv", summary), ("e9_raw_inventory.csv", pd.DataFrame(used_sources)),
    ]:
        write_csv(name, frame)
    published_baseline = pd.read_csv(ROOT / "experiments" / "E6" / "e6_llm_scores.csv")
    joined = baseline.merge(published_baseline, on="issue", suffixes=("_new", "_old"))
    np.testing.assert_allclose(joined.llm_sentiment_new, joined.llm_sentiment_old, atol=1e-12)
    formal_human = pd.read_csv(ROOT / "experiments" / "E1" / "validation" / "human_ratings.csv")
    validation.update({
        "analysis_status": "exploratory", "all_source_hashes_verified": True,
        "selected_raw_records": len(used_sources), "snapshot_inventory_files": 594,
        "parser_policies": list(PARSER_RULES), "baseline_scores_match_e6": True,
        "formal_human_status_counts": formal_human.status.value_counts().to_dict(),
        "true_human_agreement_available": False, "immutable_model_versions_available": False,
        "transport_failures_not_model_sensitivity": True,
        "no_score_imputation": True, "no_fallback_nontext_metrics": True,
        "scientific_completion": "partial_exploratory_human_and_clean_temperature_oos_pending",
    })
    write_json(OUT / "e9_validation.json", validation)
    settings_manifest = []
    for setting, (_, prompt, model, temperature, dimensions, comparator) in SETTINGS.items():
        collection = list(records[setting].values())
        settings_manifest.append({
            "setting_id": setting, "prompt_path": f"prompts/{prompt}",
            "prompt_utf8_sha256": sha256(OUT / "prompts" / prompt), "model_alias_requested": model,
            "immutable_backend_revision": None, "temperature_requested": temperature,
            "require_dimensions": dimensions, "controlled_comparator": comparator,
            "actual_intended_issues": sorted(records[setting]),
            "max_output_tokens_counts": dict(Counter(record.get("max_output_tokens") for record in collection)),
            "recorded_timestamp_start": min(record["timestamp"] for record in collection),
            "recorded_timestamp_end": max(record["timestamp"] for record in collection),
        })
    provenance_files = [
        ROOT / "experiments/E6/e6_llm_scores.csv", ROOT / "experiments/E6/e6_llm_oos_metrics.csv",
        ROOT / "experiments/E1/mpr_index_full.csv", ROOT / "experiments/E1/validation/human_ratings.csv",
        ROOT / "_temp/e6_human_spotcheck.csv", ROOT / "_temp/e6_human_spotcheck_method.md",
        ROOT / "_temp/e6/e6_panel.csv", ROOT / "_temp/e6/industry_quarterly.csv",
        ROOT / "_temp/e6/e6_oos_predictions.csv", ROOT / "experiments/E4/run_e4.py",
        ROOT / "experiments/E6/run_e6.py", ROOT / "experiments/E6/run_e6_score.py",
        SNAPSHOT / "inventory.json",
    ] + sorted(OUT.glob("*.py"))
    manifest = {
        "analysis_status": "exploratory", "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "runner_mode": "offline_frozen_individual_responses_only", "settings": settings_manifest,
        "input_and_code_hashes": [{"path": relative(path), "sha256": sha256(path)} for path in provenance_files],
        "source_selection": "individual per-issue files, never progress snapshots; progress conflicts retained for audit",
        "known_progress_conflicts": {"v1_base": 20, "v1_temp07": 23, "v2_minimal": 52},
        "source_snapshot_root": relative(SNAPSHOT), "source_inventory": "e9_raw_inventory.csv",
        "internet_sources": json.loads((CACHE / "web" / "sources.json").read_text(encoding="utf-8")),
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "validation_period": ["2016Q1", "2020Q4"], "test_period": ["2021Q1", "2026Q3"],
        "evaluation_kind": "E6-compatible retrospective expanding FMB; not a tradable post-publication forecast",
        "human_reference_kind": "unverified existing E6 provisional reference, not independent team human labels",
        "token_cost_scope": "retained individual responses only; discarded attempts and billing unavailable; zero placeholders missing",
        "historical_prompt_caveats": ["E9 v1 has lexical transcription errors compared with E6 v1", "historical doubled braces preserved", "E9 request wrapper adds a final newline compared with E6"],
    }
    write_json(OUT / "e9_prompt_model_manifest.json", manifest)
    print(summary[summary.parser == MAIN_PARSER][["setting_id", "n_valid", "n_transport_failures", "test_native_n_obs", "test_native_ic_mean", "test_native_oos_r2"]].to_string(index=False), flush=True)
    print("E9 offline outputs written; all baseline reproduction assertions passed.", flush=True)


if __name__ == "__main__":
    main()
