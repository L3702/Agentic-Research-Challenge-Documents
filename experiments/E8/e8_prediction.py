"""E8 prediction sensitivity and a separately identified timing correction.

Legacy rows reproduce the E4 report-quarter target, whose return has already
started at publication.  Strict rows refit both models for the first complete
quarter after publication and purge labels not realized at the forecast origin.
All files written by this module are in _temp/e8/prediction.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "_temp" / "e8" / "prediction"
PANEL_PATH = ROOT / "_temp" / "e4" / "e4_panel.csv"
QUARTER_PATH = ROOT / "_temp" / "e4" / "industry_quarterly.csv"
BASE = ["mom1_z", "vol_q_z", "beta_120_z", "sector_1", "sector_2", "sector_3", "sector_4"]
PRIMARY = "sent_x_exposure_ntusd_simplified"
KEY = ["issue", "code"]
ASOF = pd.Timestamp("2026-11-19")
TEST_START, TEST_END = "2021Q1", "2026Q3"


def nw_mean(values: Any, lag: int = 2) -> dict[str, float]:
    """E4-compatible Newey-West mean inference (Bartlett lag 2)."""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {k: np.nan for k in ["estimate", "std_error", "t_stat", "p_value", "ci_low", "ci_high"]}
    mean = float(a.mean())
    if len(a) < 3:
        return dict(estimate=mean, std_error=np.nan, t_stat=np.nan, p_value=np.nan, ci_low=np.nan, ci_high=np.nan)
    e = a - mean
    v = float(np.mean(e**2))
    for ell in range(1, min(lag + 1, len(a))):
        v += 2 * (1 - ell / (lag + 1)) * float(np.mean(e[ell:] * e[:-ell]))
    se = float(np.sqrt(max(v / len(a), 0)))
    t = mean / se if se > 0 else np.nan
    return dict(estimate=mean, std_error=se, t_stat=t, p_value=float(2 * stats.norm.sf(abs(t))),
                ci_low=mean - 1.959963984540054 * se, ci_high=mean + 1.959963984540054 * se)


def load_panel() -> pd.DataFrame:
    p = pd.read_csv(PANEL_PATH, dtype={"code": str, "issue": str})
    p["publication_date"] = pd.to_datetime(p["publication_date"])
    p["label_end"] = pd.PeriodIndex(p["target_quarter"], freq="Q").to_timestamp(how="end").normalize()
    p["label_start"] = pd.PeriodIndex(p["target_quarter"], freq="Q").to_timestamp(how="start")
    p["original_target_quarter"] = p["target_quarter"]
    p["actual_evaluation"] = p["return_next_q"]
    p["signal"] = p[PRIMARY]
    p["primary_signal"] = p[PRIMARY]
    p["source_issue"] = p["issue"]
    p["source_publication_date"] = p["publication_date"]
    if p.duplicated(KEY).any():
        raise AssertionError("Duplicate issue-code in E4 panel")
    return p.sort_values(["target_quarter", "code"]).reset_index(drop=True)


def strict_target(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    q = pd.read_csv(QUARTER_PATH, dtype={"code": str})
    q = q.rename(columns={"quarter": "strict_quarter", "log_return": "strict_return", "n_days": "strict_n_days",
                          "last_date": "strict_last_date", "quarter_end": "strict_end"})
    q["strict_last_date"] = pd.to_datetime(q["strict_last_date"])
    q["strict_end"] = pd.to_datetime(q["strict_end"])
    p["target_quarter"] = (p["publication_date"].dt.to_period("Q") + 1).astype(str)
    p = p.merge(q[["code", "strict_quarter", "strict_return", "strict_n_days", "strict_last_date", "strict_end"]],
                left_on=["code", "target_quarter"], right_on=["code", "strict_quarter"], how="left", validate="one_to_one")
    valid = (p["strict_return"].notna() & (p["strict_n_days"] >= 40)
             & (p["strict_last_date"] >= p["strict_end"] - pd.Timedelta(days=7)) & (p["strict_end"] < ASOF))
    p = p.loc[valid].copy()
    p["return_next_q"] = p["strict_return"]
    p["actual_evaluation"] = p["strict_return"]
    p["label_end"] = p["strict_end"]
    p["label_start"] = pd.PeriodIndex(p["target_quarter"], freq="Q").to_timestamp()
    if not (p["publication_date"] < p["label_start"]).all():
        raise AssertionError("Strict target starts before signal is available")
    return p


def true_quarter_lag(panel: pd.DataFrame) -> pd.DataFrame:
    """Move the complete predictor vector back one issue, never mom2 aliasing."""
    p = panel.copy()
    p["previous_issue"] = (pd.PeriodIndex(p["issue"], freq="Q") - 1).astype(str)
    cols = BASE[:3] + ["signal", "primary_signal", "source_issue", "source_publication_date"]
    old = p[KEY + cols].copy().rename(columns={"issue": "previous_issue", **{c: f"lagged_{c}" for c in cols}})
    p = p.merge(old, on=["previous_issue", "code"], how="left", validate="many_to_one")
    for c in cols:
        p[c] = p[f"lagged_{c}"]
    return p.drop(columns=[f"lagged_{c}" for c in cols])


def historical_terciles(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    issue = p[["issue", "publication_date", "sentiment_ntusd_simplified"]].drop_duplicates("issue").sort_values("publication_date")
    out: dict[str, float] = {}
    thresholds: dict[str, tuple[float, float]] = {}
    for i, row in enumerate(issue.itertuples(index=False)):
        prior = issue.iloc[:i]["sentiment_ntusd_simplified"].dropna()
        if len(prior) < 8:
            out[row.issue] = np.nan
            thresholds[row.issue] = (np.nan, np.nan)
            continue
        lo, hi = prior.quantile([1 / 3, 2 / 3])
        out[row.issue] = -1.0 if row.sentiment_ntusd_simplified < lo else (1.0 if row.sentiment_ntusd_simplified > hi else 0.0)
        thresholds[row.issue] = (float(lo), float(hi))
    p["ordinal_sentiment"] = p["issue"].map(out)
    p["tercile_low_threshold"] = p["issue"].map(lambda s: thresholds[s][0])
    p["tercile_high_threshold"] = p["issue"].map(lambda s: thresholds[s][1])
    p["signal"] = p["ordinal_sentiment"] * p["sent_beta_ntusd_simplified_z"]
    return p


def winsor_past(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    for col in BASE[:3] + ["signal"]:
        p[col] = p.groupby("target_quarter")[col].transform(lambda s: s.clip(s.quantile(.01), s.quantile(.99)))
    # Original realized outcomes are retained for OOS scoring. Fitting labels
    # are clipped only against outcomes completed before that issue's origin.
    for _, f in p.groupby("target_quarter", sort=True):
        origin = f["publication_date"].iloc[0]
        old = p.loc[p["label_end"] < origin, "actual_evaluation"].dropna()
        if len(old) < 100:
            continue
        lo, hi = old.quantile([.01, .99])
        p.loc[f.index, "return_next_q"] = f["return_next_q"].clip(lo, hi)
    return p


def fit_cross_sections(panel: pd.DataFrame, signal_col: str | None, setting: str,
                       role: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    cols = BASE + ([signal_col] if signal_col else [])
    expected = ["Intercept"] + cols
    rows, failures = [], []
    for period, f in panel.groupby("target_quarter", sort=True):
        f = f.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + ["return_next_q", "actual_evaluation"])
        k = len(expected)
        if len(f) < k + 2:
            failures.append(dict(setting_id=setting, model_role=role, target_quarter=period, phase="fit", reason=f"n_{len(f)}_lt_{k}+2"))
            continue
        x = np.column_stack([np.ones(len(f)), f[cols].to_numpy(float)])
        y = f["return_next_q"].to_numpy(float)
        b, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        if rank < k:
            failures.append(dict(setting_id=setting, model_role=role, target_quarter=period, phase="fit", reason=f"rank_{rank}_lt_{k}"))
            continue
        row = dict(target_quarter=period, issue=f["issue"].iloc[0], label_end=f["label_end"].max(),
                   n_obs=len(f), setting_id=setting, model_role=role)
        row.update(zip(expected, b.tolist()))
        rows.append(row)
    return pd.DataFrame(rows), failures


def metrics(pred: pd.DataFrame, col: str = "predicted") -> dict[str, float]:
    if pred.empty:
        return dict(ic_mean=np.nan, oos_r2=np.nan, mae=np.nan, rmse=np.nan, ic_p=np.nan, n_oos_periods=0, n_oos_obs=0)
    e = pred["actual"] - pred[col]
    denom = float(((pred["actual"] - pred.groupby("target_quarter")["actual"].transform("mean"))**2).sum())
    ics = []
    for _, f in pred.groupby("target_quarter", sort=True):
        if len(f) >= 3 and f[col].nunique() > 1 and f["actual"].nunique() > 1:
            ics.append(float(stats.spearmanr(f[col], f["actual"]).statistic))
    return dict(ic_mean=float(np.mean(ics)) if ics else np.nan,
                oos_r2=1-float((e**2).sum())/denom if denom > 0 else np.nan,
                mae=float(e.abs().mean()), rmse=float(np.sqrt((e**2).mean())), ic_p=nw_mean(ics)["p_value"],
                n_oos_periods=int(pred["target_quarter"].nunique()), n_oos_obs=len(pred))


def forecast(panel: pd.DataFrame, coefficients: pd.DataFrame, signal_col: str | None,
             strict: bool, setting: str, role: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    cols = BASE + ([signal_col] if signal_col else [])
    names = ["Intercept"] + cols
    rows, failed = [], []
    if coefficients.empty:
        return pd.DataFrame(), [dict(setting_id=setting, model_role=role, phase="forecast", reason="no_fitted_quarters")]
    c = coefficients.copy()
    c["label_end"] = pd.to_datetime(c["label_end"])
    test = panel.loc[panel["target_quarter"].between(TEST_START, TEST_END)]
    for period, f in test.groupby("target_quarter", sort=True):
        origin = f["publication_date"].iloc[0]
        prior = c.loc[(c["label_end"] < origin) if strict else (c["target_quarter"] < period)]
        if len(prior) < 8:
            failed.append(dict(setting_id=setting, model_role=role, target_quarter=period, phase="forecast", reason=f"train_periods_{len(prior)}_lt_8"))
            continue
        f = f.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + ["actual_evaluation"])
        if len(f) < 8:
            failed.append(dict(setting_id=setting, model_role=role, target_quarter=period, phase="forecast", reason=f"test_industries_{len(f)}_lt_8"))
            continue
        pred = np.column_stack([np.ones(len(f)), f[cols].to_numpy(float)]) @ prior[names].mean().to_numpy(float)
        train_hash = hashlib.sha256("|".join(prior["target_quarter"].tolist()).encode("ascii")).hexdigest()
        for (_, r), val in zip(f.iterrows(), pred):
            rows.append(dict(setting_id=setting, model_role=role, target_quarter=period, issue=r["issue"], code=r["code"],
                             actual=float(r["actual_evaluation"]), predicted=float(val), publication_date=origin,
                             target_start=r["label_start"], target_end=r["label_end"],
                             max_train_label_end=prior["label_end"].max(), n_train_periods=len(prior),
                             train_quarter_set_sha256=train_hash, strict_timing=strict))
    return pd.DataFrame(rows), failed


def evaluate(panel: pd.DataFrame, setting: str, strict: bool = False,
             primary_panel: pd.DataFrame | None = None) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    cols = BASE + ["signal", "primary_signal", "return_next_q", "actual_evaluation"]
    original = panel.replace([np.inf, -np.inf], np.nan)
    eligible = original[cols].notna().all(axis=1)
    p = original.loc[eligible].copy()
    failures, all_coefs, all_pred = [], [], []
    for _, r in original.loc[~eligible].iterrows():
        missing = ",".join(c for c in cols if pd.isna(r[c]))
        failures.append(dict(setting_id=setting, issue=r["issue"], code=r["code"], target_quarter=r["target_quarter"],
                             phase="eligibility", reason=f"missing_or_nonfinite:{missing}"))
    for period in sorted(set(original["target_quarter"]) - set(p["target_quarter"])):
        failures.append(dict(setting_id=setting, target_quarter=period, phase="fit", reason="no_complete_predictor_outcome_rows"))
    if primary_panel is not None:
        ref_cols = BASE[:3] + ["primary_signal", "return_next_q"]
        ref = primary_panel[KEY + ref_cols].rename(columns={c: f"ref_{c}" for c in ref_cols})
        p = p.merge(ref, on=KEY, how="inner", validate="one_to_one")
        p = p.dropna(subset=[f"ref_{c}" for c in BASE[:3]+["primary_signal"]])
    main_result: dict[str, Any] = {}
    fitted = []
    for role, sig in [("sentiment", "signal"), ("notext", None), ("paired_primary", "primary_signal")]:
        pp = p.copy()
        if role == "paired_primary" and primary_panel is not None:
            for c in BASE[:3]+["primary_signal", "return_next_q"]:
                pp[c] = pp[f"ref_{c}"]
        co, fail = fit_cross_sections(pp, sig, setting, role)
        failures.extend(fail)
        main_result[f"raw_fit_periods_{role}"] = len(co)
        fitted.append((role, sig, pp, co))
    period_sets = [set(co["target_quarter"]) if not co.empty else set() for _, _, _, co in fitted]
    common_fit = set.intersection(*period_sets)
    main_result["common_identifiable_periods"] = len(common_fit)
    for role, sig, pp, co in fitted:
        if not co.empty:
            dropped = co.loc[~co["target_quarter"].isin(common_fit)]
            for period in dropped["target_quarter"]:
                failures.append(dict(setting_id=setting, model_role=role, target_quarter=period,
                                     phase="common_fit", reason="excluded_to_match_all_three_roles_identifiable_quarters"))
            co = co.loc[co["target_quarter"].isin(common_fit)].copy()
        all_coefs.append(co)
        pr, fail = forecast(pp, co, sig, strict, setting, role)
        failures.extend(fail)
        all_pred.append(pr)
        mm = metrics(pr)
        if role == "sentiment":
            main_result.update(nw_mean(co["signal"] if not co.empty else []))
            main_result.update(mm)
            main_result.update(n_periods=len(co), n_obs=int(co["n_obs"].sum()) if not co.empty else 0,
                               first_period=co["target_quarter"].min() if not co.empty else "",
                               last_period=co["target_quarter"].max() if not co.empty else "")
        elif role == "notext":
            main_result.update({f"baseline_{k}": v for k, v in mm.items()})
        else:
            main_result.update({f"paired_{k}": v for k, v in mm.items()})
            main_result.update({f"paired_{k}": v for k, v in nw_mean(co["primary_signal"] if not co.empty else []).items()})
    # All roles use the same identifiable training quarters and score on the
    # same forecast keys. No unpaired fallback is allowed for missing forecasts.
    if all(len(pr) for pr in all_pred):
        keys = ["target_quarter", "code"]
        common = all_pred[0][keys]
        for pr in all_pred[1:]:
            common = common.merge(pr[keys], on=keys, how="inner", validate="one_to_one")
        scored = [pr.merge(common, on=keys, how="inner", validate="one_to_one") for pr in all_pred]
        for role, pr in zip(["", "baseline_", "paired_"], scored):
            main_result.update({f"{role}{k}": v for k, v in metrics(pr).items()})
        main_result["oos_pairing_status"] = "paired"
    else:
        for prefix in ["", "baseline_", "paired_"]:
            main_result.update({f"{prefix}{k}": v for k, v in metrics(pd.DataFrame()).items()})
        main_result["oos_pairing_status"] = "unavailable_one_or_more_roles_have_no_forecasts"
    main_result["incremental_oos_r2"] = main_result["oos_r2"] - main_result["baseline_oos_r2"]
    main_result["incremental_ic"] = main_result["ic_mean"] - main_result["baseline_ic_mean"]
    main_result["baseline_ic"] = main_result["baseline_ic_mean"]
    main_result["baseline_mae"] = main_result["baseline_mae"]
    main_result["paired_baseline_estimate"] = main_result.get("paired_estimate", np.nan)
    main_result["paired_baseline_p_value"] = main_result.get("paired_p_value", np.nan)
    main_result["failure_count"] = len(failures)
    return main_result, pd.concat(all_coefs, ignore_index=True), pd.concat(all_pred, ignore_index=True), failures


def configurations(base: pd.DataFrame) -> list[tuple[dict[str, Any], pd.DataFrame]]:
    cfgs: list[tuple[dict[str, Any], pd.DataFrame]] = []

    def add(ident: str, family: str, value: str, p: pd.DataFrame, strict: bool = False, notes: str = "") -> None:
        cfgs.append((dict(block="prediction", setting_id=ident, family=family, changed_parameter=family,
                          setting_value=value, comparison_id="prediction_strict_main" if strict else "prediction_main",
                          strict_timing=strict, metric_name="FMB sentiment-exposure coefficient mean",
                          coef_name="sentiment_exposure_interaction", timing_protocol="strict_post_publication" if strict else "legacy_e4_overlap",
                          inference_time_basis="observed_fitted_quarter_sequence_not_calendar",
                          notes=notes), p))

    for strict in [False, True]:
        p = strict_target(base) if strict else base.copy()
        pre = "prediction_strict_" if strict else "prediction_"
        timing = ("First complete quarter after publication; both models refitted; labels purged by realization date. " if strict else
                  "E4 legacy report-quarter+1 target overlaps pre-publication returns; descriptive sensitivity only. ")
        add(pre+"main", "reference", "ntusd_simplified", p, strict, timing)
        for dictionary in ["dlut", "mpr_domain"]:
            pp = p.copy()
            pp["signal"] = pp[f"sent_x_exposure_{dictionary}"]
            add(pre+dictionary, "dictionary", dictionary, pp, strict, timing)
        pp = p.copy()
        pp["signal"] = pp[[PRIMARY, "sent_x_exposure_dlut"]].mean(axis=1, skipna=False)
        add(pre+"combo", "dictionary", "equal_ntusd_dlut_interactions", pp, strict,
            timing+"Equal average of each dictionary's trailing-z sentiment times its own standardized exposure; no fitted combination weight.")
        add(pre+"lag2", "lag", "two_quarter_information_lag", true_quarter_lag(p), strict,
            timing+"One additional exact report-quarter shift of sentiment exposure and three continuous controls; sector indicators fixed.")
        add(pre+"tercile", "sentiment_form", "past_tercile_ordinal", historical_terciles(p), strict,
            timing+"Prior issues only determine cutoffs; ordinal -1/0/+1 times industry exposure. Coefficients are identified only in the conditional high/low tails; mid-group zero periods excluded from all three training sets. Not a full-sample replacement. NW2 is over the observed fitted-quarter sequence, not elapsed calendar quarters; coefficient magnitude has different units.")
        add(pre+"winsor", "outliers", "one_percent", winsor_past(p), strict,
            timing+"Continuous predictor cross-sections clipped at 1/99%; fitting outcomes clipped using labels completed before each origin; OOS actuals never clipped.")
        for year in [2008, 2015, 2020]:
            for side in ["before", "after"]:
                mask = p["publication_date"].dt.year < year if side == "before" else p["publication_date"].dt.year >= year
                add(pre+f"{side}{year}", "subsample", f"{side}_{year}", p.loc[mask].copy(), strict,
                    timing+f"Publication year cutoff {year}-01-01; {side}; training and FMB restricted to that subsample.")
    return cfgs


def run_prediction() -> tuple[pd.DataFrame, dict[str, Any]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    base = load_panel()
    strict_ref = strict_target(base)
    rows, coefs, predictions, failures = [], [], [], []
    for cfg, pp in configurations(base):
        ref = strict_ref if cfg["strict_timing"] else base
        try:
            result, co, pr, fail = evaluate(pp, cfg["setting_id"], cfg["strict_timing"], ref)
            failures.extend(fail)
            coefs.append(co)
            predictions.append(pr)
            status = "ok" if result["n_periods"] >= 8 else "limited"
            reason = "" if status == "ok" else "fewer_than_eight_fitted_quarters"
            if not result["n_oos_periods"]:
                status, reason = "limited", "no_eligible_2021Q1_2026Q3_test_periods"
            if not result["n_periods"]:
                status, reason = "failed", "no_full_rank_cross_section"
            elif cfg["family"] == "sentiment_form":
                status, reason = "limited", "ordinal_middle_group_unidentified_conditional_tails_only"
            if reason:
                failures.append(dict(setting_id=cfg["setting_id"], phase="setting", reason=reason))
            rows.append({**cfg, **result, "status": status, "failure_reason": reason})
            print(f"[prediction] {cfg['setting_id']}: T={result['n_periods']}, OOS={result['n_oos_periods']}, status={status}")
        except Exception as exc:
            rows.append({**cfg, "status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}", "n_periods": 0, "n_obs": 0})
            failures.append(dict(setting_id=cfg["setting_id"], phase="setting", reason=f"{type(exc).__name__}: {exc}"))
    summary = pd.DataFrame(rows)
    summary.to_csv(CACHE / "summary.csv", index=False, encoding="utf-8-sig")
    coef = pd.concat(coefs, ignore_index=True) if coefs else pd.DataFrame()
    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    coef.to_csv(CACHE / "coefficients.csv", index=False, encoding="utf-8-sig")
    pred.to_csv(CACHE / "predictions.csv", index=False, encoding="utf-8-sig")
    rejected = base.merge(strict_ref[KEY], on=KEY, how="left", indicator=True)
    rejected = rejected.loc[rejected["_merge"] == "left_only", KEY + ["publication_date"]].copy()
    rejected["setting_id"] = "prediction_strict_main"
    rejected["phase"] = "strict_label_availability"
    rejected["reason"] = "first_complete_postpublication_quarter_unavailable_or_incomplete_asof_2026_11_19"
    failures.extend(rejected.to_dict("records"))
    pd.DataFrame(failures).to_csv(CACHE / "failures.csv", index=False, encoding="utf-8-sig")
    audit = base[KEY+["publication_date", "label_start", "label_end", "target_quarter"]].copy()
    audit["label_started_before_publication"] = audit["label_start"] <= audit["publication_date"]
    audit.to_csv(CACHE / "legacy_timing_audit.csv", index=False, encoding="utf-8-sig")
    strict_predictions = pred.loc[pred["strict_timing"].eq(True)] if not pred.empty else pred
    if not strict_predictions.empty:
        assert (pd.to_datetime(strict_predictions["max_train_label_end"]) < pd.to_datetime(strict_predictions["publication_date"])).all()
        assert (pd.to_datetime(strict_predictions["publication_date"]) < pd.to_datetime(strict_predictions["target_start"])).all()
    # Independent stored E4 coefficients provide a regression check without
    # importing its stateful script or rewriting its outputs.
    e4 = pd.read_csv(ROOT / "experiments" / "E4" / "e4_fmb_results.csv")
    e4_main = e4.loc[(e4["dictionary"] == "ntusd_simplified") & (e4["coefficient"] == PRIMARY)].iloc[0]
    own = summary.set_index("setting_id").loc["prediction_main"]
    replicate_delta = float(own["estimate"] - e4_main["mean"])
    if abs(replicate_delta) > 1e-9:
        raise AssertionError(f"E4 main coefficient reproduction drift: {replicate_delta}")
    e4_oos = pd.read_csv(ROOT / "experiments" / "E4" / "e4_oos_metrics.csv")
    e4_oos = e4_oos.loc[(e4_oos["dictionary"] == "ntusd_simplified") & (e4_oos["split"] == "test")].iloc[0]
    oos_deltas = {k: float(own[k] - e4_oos[k]) for k in ["ic_mean", "oos_r2", "mae"]}
    if any(abs(v) > 1e-9 for v in oos_deltas.values()):
        raise AssertionError(f"E4 sentiment OOS reproduction drift: {oos_deltas}")
    details = dict(input_panel=str(PANEL_PATH.relative_to(ROOT)), input_quarterly=str(QUARTER_PATH.relative_to(ROOT)),
                   cache=str(CACHE.relative_to(ROOT)), n_settings=len(summary),
                   legacy_prepublication_rows=int(audit["label_started_before_publication"].sum()),
                   legacy_rows=len(audit), legacy_issues=int(audit["issue"].nunique()),
                   strict_rows=len(strict_ref), strict_issues=int(strict_ref["issue"].nunique()),
                   strict_forecast_rows=len(strict_predictions), strict_time_violations=0,
                   e4_main_coefficient_delta=replicate_delta, e4_main_oos_deltas=oos_deltas,
                   failure_records=len(failures), strict_incomplete_target_rows=len(rejected),
                   oos_r2_definition="1-SSE/SST around realized quarterly cross-sectional mean; diagnostic denominator is ex-post, not a feasible forecast",
                   limitations=["Legacy target overlaps returns before publication and is not valid prospective OOS evidence.",
                                "Strict model changes label timing and realization-aware training together; it is a separate timing-corrected reference.",
                                "Industry composition and historical index backfill remain inherited data limitations.",
                                "Subsample rows exclude the cutoff year from before and include it in after; short or absent OOS is explicit."])
    (CACHE / "details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary, details


if __name__ == "__main__":
    result, detail = run_prediction()
    print(result[["setting_id", "status", "estimate", "p_value", "ic_mean", "oos_r2", "incremental_oos_r2"]].to_string(index=False))
    print(json.dumps(detail, ensure_ascii=False, indent=2))
