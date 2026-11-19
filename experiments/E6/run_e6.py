"""E6: LLM vs dictionary sentiment comparison.

Experiment goal
---------------
Test H4: does an LLM sentiment score contain incremental information over the
dictionary score for cross-sectional industry return prediction?

Fixed protocol
--------------
* Same MPR texts, same industry return panel, same splits as E4:
  train 2001Q2-2015Q4, validation 2016Q1-2020Q4, test 2021Q1-2026Q3.
* The LLM model / prompt version / temperature are fixed by the scoring
  script that produces ``e6_llm_scores.csv`` and are recorded in that file.
* Panel construction reuses ``experiments/E4/run_e4.py`` so the dictionary
  models are numerically identical to E4.  The only extension is one more
  sentiment column, ``llm``, built from the LLM scores.
* Models: ``baseline_notext``, ``dict`` (ntusd_simplified primary; dlut and
  mpr_domain as robustness), ``llm``, and ``joint`` (dict + llm).  Every
  sentiment model uses the ``sent_exposure`` specification: base controls
  plus ``sentiment_z * sent_beta_z``.

Inputs
------
* ``experiments/E6/e6_llm_scores.csv`` (produced by the LLM scoring script)
* ``experiments/E1/sentiment_dictionary_full.csv``
* ``experiments/E1/mpr_index_full.csv``
* ``data/external/...`` (loaded through E4 helpers)

Outputs
-------
* ``experiments/E6/e6_llm_scores.csv`` (normalised copy)
* ``experiments/E6/e6_llm_vs_dict.csv``
* ``experiments/E6/e6_llm_vs_dict_summary.csv``
* ``experiments/E6/e6_llm_oos_metrics.csv``
* ``experiments/E6/e6_llm_incremental_regression.csv``
* ``experiments/E6/e6_llm_human_consistency.csv`` (when human spot-check exists)
* ``experiments/E6/logs/e6_llm_metrics.md``
* ``experiments/E6/e6_metrics.json``

Reproduce
---------
``python experiments/E6/run_e6.py``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
E6_DIR = ROOT / "experiments" / "E6"
LOG_DIR = E6_DIR / "logs"
TEMP_DIR = ROOT / "_temp" / "e6"

E1_DIR = ROOT / "experiments" / "E1"
MPR_INDEX_PATH = E1_DIR / "mpr_index_full.csv"
E1_SENTIMENT_PATH = E1_DIR / "sentiment_dictionary_full.csv"
HUMAN_SPOTCHECK_CANDIDATES = [
    ROOT / "_temp" / "e6_human_spotcheck.csv",
    TEMP_DIR / "e6_human_spotcheck.csv",
]

DICTS = ["ntusd_simplified", "dlut", "mpr_domain"]
PRIMARY_DICT = "ntusd_simplified"
SEED = 20261119
NW_LAG = 2

TRAIN_START, TRAIN_END = "2001Q2", "2015Q4"
VAL_START, VAL_END = "2016Q1", "2020Q4"
TEST_START, TEST_END = "2021Q1", "2026Q3"

BASE_COLUMNS = ["mom1_z", "vol_q_z", "beta_120_z", "sector_1", "sector_2", "sector_3", "sector_4"]

LLM_DIMENSION_COLUMNS = [
    "llm_policy_stance",
    "llm_economic_judgment",
    "llm_risk_tone",
    "llm_liquidity_credit",
    "llm_policy_outlook",
]

# name, spec, dictionaries, model label, dictionary label
MODELS: list[dict[str, Any]] = [
    {"name": "baseline_notext", "spec": "baseline_notext", "dictionaries": None,
     "model": "baseline_notext", "dictionary": "none"},
    {"name": "dict_ntusd_simplified", "spec": "sent_exposure", "dictionaries": ["ntusd_simplified"],
     "model": "dict", "dictionary": "ntusd_simplified"},
    {"name": "dict_dlut", "spec": "sent_exposure", "dictionaries": ["dlut"],
     "model": "dict", "dictionary": "dlut"},
    {"name": "dict_mpr_domain", "spec": "sent_exposure", "dictionaries": ["mpr_domain"],
     "model": "dict", "dictionary": "mpr_domain"},
    {"name": "llm", "spec": "sent_exposure", "dictionaries": ["llm"],
     "model": "llm", "dictionary": "llm"},
    {"name": "joint_ntusd_llm", "spec": "sent_exposure", "dictionaries": ["ntusd_simplified", "llm"],
     "model": "joint", "dictionary": "ntusd_simplified+llm"},
]


def ensure_dirs() -> None:
    for path in [E6_DIR, LOG_DIR, TEMP_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def log(line: str = "") -> None:
    with (TEMP_DIR / "e6_run.log").open("a", encoding="utf-8") as handle:
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


def load_llm_scores(path: Path) -> pd.DataFrame:
    """Read and normalise the LLM score file produced by the scoring script."""
    if not path.exists():
        raise FileNotFoundError(
            f"LLM 得分文件不存在：{path}\n"
            "请先运行 LLM 评分脚本生成 e6_llm_scores.csv。"
        )
    frame = pd.read_csv(path, dtype={"issue": str})
    required = ["issue", "llm_sentiment"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"e6_llm_scores.csv 缺少必需列：{missing}")
    if frame["issue"].duplicated().any():
        raise ValueError("e6_llm_scores.csv 中 issue 有重复")

    mpr = pd.read_csv(MPR_INDEX_PATH, dtype={"issue": str})
    mpr = mpr[["issue", "report_period", "publication_date", "text_chars"]].drop_duplicates("issue")
    for column in ["report_period", "publication_date", "text_chars"]:
        if column not in frame.columns:
            frame = frame.merge(mpr[["issue", column]], on="issue", how="left")

    dict_scores = pd.read_csv(E1_SENTIMENT_PATH, dtype={"issue": str})
    dict_wide = dict_scores.pivot(index="issue", columns="dictionary", values="sentiment_score")
    dict_wide.columns = [f"dict_{column}" for column in dict_wide.columns]
    dict_wide = dict_wide.reset_index()
    for column in dict_wide.columns:
        if column == "issue" or column in frame.columns:
            continue
        frame = frame.merge(dict_wide[["issue", column]], on="issue", how="left")

    for column, default in [
        ("llm_stance", 0),
        ("model", "unknown"),
        ("prompt_version", "unknown"),
        ("temperature", 0.0),
        ("latency_sec", 0.0),
        ("input_tokens", 0),
        ("output_tokens", 0),
        ("raw_json_path", ""),
        ("parse_status", "unknown"),
    ]:
        if column not in frame.columns:
            frame[column] = default
    for column in LLM_DIMENSION_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan

    frame = frame.sort_values("issue").reset_index(drop=True)
    return frame


def write_combined_sentiment(llm_scores: pd.DataFrame, target: Path) -> pd.DataFrame:
    """Append the LLM score as one more dictionary column for E4's build_panel."""
    dict_scores = pd.read_csv(E1_SENTIMENT_PATH, dtype={"issue": str})
    dict_scores = dict_scores[["issue", "dictionary", "sentiment_score", "weighted_score"]]
    llm_rows = pd.DataFrame(
        {
            "issue": llm_scores["issue"],
            "dictionary": "llm",
            "sentiment_score": llm_scores["llm_sentiment"].astype(float),
            "weighted_score": llm_scores["llm_sentiment"].astype(float),
        }
    )
    combined = pd.concat([dict_scores, llm_rows], ignore_index=True)
    combined.to_csv(target, index=False, encoding="utf-8-sig")
    return combined


def build_e6_panel(llm_scores: pd.DataFrame) -> pd.DataFrame:
    """Reuse E4's panel builder with one extra sentiment column."""
    sys.path.insert(0, str(ROOT / "experiments" / "E4"))
    import run_e4 as e4  # noqa: E402

    e4.TEMP_DIR = TEMP_DIR
    e4.MACRO_RAW_DIR = ROOT / "_temp" / "e4" / "macro_raw"
    e4.SENTIMENT_PATH = TEMP_DIR / "sentiment_dict_with_llm.csv"
    e4.DICTS = DICTS + ["llm"]

    industry, bench = e4.load_prices()
    macro = e4.load_macro()
    panel = e4.build_panel(industry, bench, macro)
    panel = panel[panel["target_quarter"] <= "2026Q3"].copy()
    panel.to_csv(TEMP_DIR / "e6_panel.csv", index=False, encoding="utf-8-sig")

    required = [
        "sentiment_llm",
        "sentiment_llm_z",
        "sent_beta_llm",
        "sent_beta_llm_z",
        "sent_x_exposure_llm",
    ]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise RuntimeError(f"E6 面板缺少 LLM 情绪列：{missing}")
    return panel


def design_columns(spec: str, dictionaries: list[str] | None) -> list[str]:
    if spec == "baseline_notext":
        return list(BASE_COLUMNS)
    if dictionaries is None:
        raise ValueError("sentiment spec requires dictionaries")
    if spec == "sent_exposure":
        return list(BASE_COLUMNS) + [f"sent_x_exposure_{name}" for name in dictionaries]
    raise ValueError(f"unknown spec: {spec}")


def cross_section_ols(
    panel: pd.DataFrame,
    spec: str,
    dictionaries: list[str] | None,
    model_label: str,
    dictionary_label: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """One OLS per target quarter, identical to E4's implementation."""
    columns = design_columns(spec, dictionaries)
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
                    "model": model_label,
                    "dictionary": dictionary_label,
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
    return pd.DataFrame(rows), skipped


def predict_oos(
    panel: pd.DataFrame,
    coefficients: pd.DataFrame,
    spec: str,
    dictionaries: list[str] | None,
    model_label: str,
    dictionary_label: str,
    split: str,
    start_q: str,
    end_q: str,
    min_train_periods: int = 8,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Expanding-window FMB prediction, identical to E4's implementation."""
    columns = design_columns(spec, dictionaries)
    coef_rows = coefficients[coefficients["spec"] == spec]
    coef_rows = coef_rows[coef_rows["dictionary"] == dictionary_label]
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
                    "model": model_label,
                    "dictionary": dictionary_label,
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

    import run_e4 as e4

    metrics = e4.oos_metrics(predictions)
    metrics.update(
        {
            "model": model_label,
            "dictionary": dictionary_label,
            "spec": spec,
            "split": split,
            "start_period": start_q,
            "end_period": end_q,
            "n_periods": int(predictions["target_quarter"].nunique()),
        }
    )
    return predictions, metrics


def paired_summary(reference: pd.DataFrame, model: pd.DataFrame, e4: Any) -> dict[str, Any]:
    """Paired MAE comparison against a reference model on the common sample."""
    keys = model[["target_quarter", "code"]].drop_duplicates()
    reference_paired = reference.merge(keys, on=["target_quarter", "code"], how="inner")
    reference_metrics = e4.oos_metrics(reference_paired) if not reference_paired.empty else {}
    paired_t, paired_p = e4.paired_mae_comparison(reference, model)
    return {
        "reference_n_obs_paired": reference_metrics.get("n_obs", float("nan")),
        "reference_mae": reference_metrics.get("mae", float("nan")),
        "reference_ic": reference_metrics.get("ic_mean", float("nan")),
        "reference_icir": reference_metrics.get("icir", float("nan")),
        "reference_oos_r2": reference_metrics.get("oos_r2", float("nan")),
        "paired_mae_t": paired_t,
        "paired_mae_p": paired_p,
    }


def fmb_summary(coefficients: pd.DataFrame, e4: Any) -> pd.DataFrame:
    return e4.summarize_fmb(coefficients)


def sentiment_correlations(llm_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-issue long table and per-dictionary correlation summary."""
    long_rows: list[dict[str, Any]] = []
    for row in llm_scores.itertuples(index=False):
        for dictionary in DICTS:
            dict_value = getattr(row, f"dict_{dictionary}", np.nan)
            long_rows.append(
                {
                    "issue": row.issue,
                    "dictionary": dictionary,
                    "llm_sentiment": float(row.llm_sentiment),
                    "dict_sentiment": float(dict_value),
                    "difference": float(row.llm_sentiment) - float(dict_value),
                    "abs_difference": abs(float(row.llm_sentiment) - float(dict_value)),
                }
            )
    long_table = pd.DataFrame(long_rows)

    summary_rows: list[dict[str, Any]] = []
    for dictionary in DICTS:
        frame = long_table[long_table["dictionary"] == dictionary].dropna(
            subset=["llm_sentiment", "dict_sentiment"]
        )
        if len(frame) < 3:
            continue
        pearson = float(stats.pearsonr(frame["llm_sentiment"], frame["dict_sentiment"]).statistic)
        spearman = float(stats.spearmanr(frame["llm_sentiment"], frame["dict_sentiment"]).statistic)
        mae = float(np.mean(np.abs(frame["difference"])))
        rmse = float(np.sqrt(np.mean(frame["difference"] ** 2)))
        summary_rows.append(
            {
                "dictionary": dictionary,
                "n": int(len(frame)),
                "pearson_r": pearson,
                "spearman_rho": spearman,
                "mae": mae,
                "rmse": rmse,
                "mean_difference": float(frame["difference"].mean()),
                "llm_mean": float(frame["llm_sentiment"].mean()),
                "dict_mean": float(frame["dict_sentiment"].mean()),
            }
        )
    return long_table, pd.DataFrame(summary_rows)


def human_consistency(llm_scores: pd.DataFrame) -> pd.DataFrame:
    """Compare LLM and dictionary scores with the pre-registered human sample."""
    human_path = next((path for path in HUMAN_SPOTCHECK_CANDIDATES if path.exists()), None)
    if human_path is None:
        return pd.DataFrame()
    human = pd.read_csv(human_path, dtype={"issue": str})
    human = human.dropna(subset=["human_score"])
    if human.empty:
        return pd.DataFrame()
    merged = human.merge(llm_scores, on="issue", how="left")
    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        human_score = float(row.human_score)
        llm_stance = float(row.llm_stance) if pd.notna(row.llm_stance) else float("nan")
        llm_scaled = float(row.llm_sentiment) * 2.0
        common = {
            "issue": row.issue,
            "human_score": human_score,
            "rater_id": row.rater_id,
            "confidence_1_5": row.confidence_1_5,
            "dictionary": "llm",
            "dict_sentiment_scaled": np.nan,
            "dict_difference": np.nan,
        }
        rows.append(
            {
                **common,
                "method": "llm_stance",
                "source": "llm",
                "score": llm_stance,
                "difference": llm_stance - human_score,
            }
        )
        rows.append(
            {
                **common,
                "method": "llm_sentiment_x2",
                "source": "llm",
                "score": llm_scaled,
                "difference": llm_scaled - human_score,
            }
        )
        for dictionary in DICTS:
            dict_value = float(getattr(row, f"dict_{dictionary}"))
            rows.append(
                {
                    **common,
                    "method": "dict_x2",
                    "source": dictionary,
                    "score": dict_value * 2.0,
                    "difference": dict_value * 2.0 - human_score,
                    "dict_sentiment_scaled": dict_value * 2.0,
                    "dict_difference": dict_value * 2.0 - human_score,
                    "dictionary": dictionary,
                }
            )
    return pd.DataFrame(rows)


def write_metrics_log(
    environment: dict[str, Any],
    llm_scores: pd.DataFrame,
    correlations: pd.DataFrame,
    oos_metrics: pd.DataFrame,
    incremental: pd.DataFrame,
    human_table: pd.DataFrame,
    metrics: dict[str, Any],
    out_dir: Path,
) -> None:
    lines: list[str] = []

    def add(line: str = "") -> None:
        lines.append(line)

    add("# E6 LLM 与词典情绪比较实验日志")
    add()
    add("## 1. 实验目的与假设")
    add()
    add("- 检验 H4：在相同样本、相同划分、相同控制变量下，LLM 情绪得分是否比词典法有更多增量信息。")
    add("- 对照：词典模型、LLM 模型、词典 + LLM 联合模型。")
    add()
    add("## 2. 固定口径")
    add()
    add(f"- LLM 模型：`{metrics.get('llm_model', 'unknown')}`；Prompt 版本：`{metrics.get('prompt_version', 'unknown')}`；温度：`{metrics.get('temperature', 'unknown')}`。")
    add(f"- 面板与划分完全复用 E4：训练 {TRAIN_START}–{TRAIN_END}，验证 {VAL_START}–{VAL_END}，测试 {TEST_START}–{TEST_END}。")
    add("- 情绪模型设定：Baseline 控制 + `sentiment_z × sent_beta_z`；联合模型同时放入词典与 LLM 的情绪暴露交互项。")
    add("- LLM 得分文件：`experiments/E6/e6_llm_scores.csv`；原始 JSON 与成本见 `logs/e6_llm_prompt_and_cost.md`。")
    add()
    add("## 3. LLM 与词典的相关性")
    add()
    if not correlations.empty:
        add("| 词典 | N | Pearson r | Spearman rho | MAE | RMSE | LLM 均值 | 词典均值 |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in correlations.itertuples(index=False):
            add(
                f"| {row.dictionary} | {row.n} | {row.pearson_r:.4f} | {row.spearman_rho:.4f} | "
                f"{row.mae:.4f} | {row.rmse:.4f} | {row.llm_mean:.4f} | {row.dict_mean:.4f} |"
            )
    else:
        add("- 无相关性结果。")
    add()
    add("## 4. 样本外比较")
    add()
    add("| 样本 | 模型 | 词典 | 期数 | N | IC | ICIR | OOS R² | MAE | vs Baseline MAE | vs 词典 MAE |")
    add("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in oos_metrics.itertuples(index=False):
        base_imp = "—" if row.dictionary == "none" else f"{row.mae_improvement_pct:.2f}%"
        dict_imp = "—" if pd.isna(getattr(row, "dict_mae_improvement_pct", np.nan)) else f"{row.dict_mae_improvement_pct:.2f}%"
        add(
            f"| {row.split} | {row.model} | {row.dictionary} | {row.n_periods} | {row.n_obs} | "
            f"{row.ic_mean:.4f} | {row.icir:.3f} | {row.oos_r2:.4f} | {row.mae:.5f} | {base_imp} | {dict_imp} |"
        )
    add()
    add("## 5. 增量回归（联合模型）")
    add()
    if not incremental.empty:
        summary = incremental[incremental["row_type"] == "fmb_summary"]
        sentiment_rows = summary[summary["coefficient"].str.startswith("sent_x_")]
        add("| 系数 | 均值 | FMB t | FMB p | NW t | NW p | 期数 |")
        add("|---|---:|---:|---:|---:|---:|---:|")
        for row in sentiment_rows.itertuples(index=False):
            add(
                f"| {row.coefficient} | {row.mean:.5f} | {row.t_fmb:.2f} | {row.p_fmb:.4f} | "
                f"{row.t_nw:.2f} | {row.p_nw:.4f} | {row.n_periods} |"
            )
    else:
        add("- 无增量回归结果。")
    add()
    add("## 6. 人工抽检一致性")
    add()
    if not human_table.empty:
        add("| 方法 | 来源 | N | Pearson r | Spearman rho | MAE | 绝对误差≤1 占比 |")
        add("|---|---|---:|---:|---:|---:|---:|")
        for (method, source), frame in human_table.groupby(["method", "source"], sort=False):
            frame = frame.dropna(subset=["score", "human_score"])
            if len(frame) < 3:
                continue
            pearson = float(stats.pearsonr(frame["score"], frame["human_score"]).statistic)
            spearman = float(stats.spearmanr(frame["score"], frame["human_score"]).statistic)
            mae = float(np.mean(np.abs(frame["difference"])))
            within1 = float(np.mean(np.abs(frame["difference"]) <= 1.0))
            add(f"| {method} | {source} | {len(frame)} | {pearson:.4f} | {spearman:.4f} | {mae:.4f} | {within1:.2%} |")
    else:
        add("- 未找到人工抽检文件，LLM-人工一致性未计算。")
    add()
    add("## 7. 结论")
    add()
    add(metrics["conclusion"])
    add()
    add("## 8. 局限")
    add()
    add("- LLM 模型、Prompt 与温度固定，但模型提供方后续可能更新底层权重；本报告记录模型名与调用时间。")
    add("- 人工抽检为单评分人预注册样本，未达到 E1 指南的双人独立评分要求。")
    add("- 季度样本约 100 期，横截面约 16–31 个行业，增量检验统计功效有限。")
    add()
    add("## 9. 复现")
    add()
    add("```powershell")
    add("python experiments/E6/run_e6.py")
    add("```")
    add()
    add(f"- 脚本 SHA256：`{metrics['script_sha256']}`")
    add(f"- 运行环境：{environment}")
    add()

    log_dir = Path(out_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "e6_llm_metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="E6 LLM vs dictionary sentiment comparison")
    parser.add_argument("--llm-scores", default=str(E6_DIR / "e6_llm_scores.csv"))
    parser.add_argument("--out-dir", default=str(E6_DIR))
    args = parser.parse_args()

    ensure_dirs()
    (TEMP_DIR / "e6_run.log").write_text("", encoding="utf-8")
    log("E6 LLM vs dictionary sentiment comparison start")

    environment = {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": stats.__name__ + ":" + getattr(stats, "__version__", "unknown"),
    }

    llm_path = Path(args.llm_scores)
    try:
        llm_scores = load_llm_scores(llm_path)
    except FileNotFoundError as exc:
        log(f"[error] {exc}")
        return 2
    except ValueError as exc:
        log(f"[error] {exc}")
        return 2

    log(f"[data] LLM scores rows={len(llm_scores)}, issues={llm_scores['issue'].nunique()}")
    log(f"[data] model={llm_scores['model'].iloc[0]}, prompt={llm_scores['prompt_version'].iloc[0]}, temperature={llm_scores['temperature'].iloc[0]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm_scores.to_csv(out_dir / "e6_llm_scores.csv", index=False, encoding="utf-8-sig")

    combined = write_combined_sentiment(llm_scores, TEMP_DIR / "sentiment_dict_with_llm.csv")
    log(f"[data] combined sentiment rows={len(combined)}")

    panel = build_e6_panel(llm_scores)
    log(
        f"[panel] rows={len(panel)}, target quarters={panel['target_quarter'].nunique()}, "
        f"industries={panel['code'].nunique()}, range {panel['target_quarter'].min()}..{panel['target_quarter'].max()}"
    )

    sys.path.insert(0, str(ROOT / "experiments" / "E4"))
    import run_e4 as e4

    coefficient_frames: list[pd.DataFrame] = []
    skip_rows: list[dict[str, Any]] = []
    for model in MODELS:
        coefficients, skipped = cross_section_ols(
            panel,
            model["spec"],
            model["dictionaries"],
            model["model"],
            model["dictionary"],
        )
        coefficient_frames.append(coefficients)
        for item in skipped:
            item.update({"model": model["model"], "dictionary": model["dictionary"], "spec": model["spec"]})
            skip_rows.append(item)
        log(
            f"[fmb] {model['name']}: coefficient rows={len(coefficients)}, "
            f"periods={coefficients['target_quarter'].nunique() if not coefficients.empty else 0}, skipped={len(skipped)}"
        )
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    coefficients.to_csv(TEMP_DIR / "e6_coefficient_series.csv", index=False, encoding="utf-8-sig")

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    predictions_by_model: dict[str, pd.DataFrame] = {}
    for split, start_q, end_q in [
        ("validation", VAL_START, VAL_END),
        ("test", TEST_START, TEST_END),
    ]:
        for model in MODELS:
            pred, metrics = predict_oos(
                panel,
                coefficients,
                model["spec"],
                model["dictionaries"],
                model["model"],
                model["dictionary"],
                split,
                start_q,
                end_q,
            )
            if pred.empty:
                log(f"[oos] {model['name']} {split} skipped: {metrics}")
                continue
            predictions_by_model.setdefault(model["name"], {})[split] = pred
            prediction_frames.append(pred)

            baseline_pred = predictions_by_model.get("baseline_notext", {}).get(split)
            if baseline_pred is not None:
                paired = paired_summary(baseline_pred, pred, e4)
                metrics.update(
                    {
                        "baseline_model": "baseline_notext",
                        "baseline_mae": paired["reference_mae"],
                        "baseline_ic": paired["reference_ic"],
                        "baseline_icir": paired["reference_icir"],
                        "baseline_oos_r2": paired["reference_oos_r2"],
                        "baseline_n_obs_paired": paired["reference_n_obs_paired"],
                        "paired_mae_t": paired["paired_mae_t"],
                        "paired_mae_p": paired["paired_mae_p"],
                        "mae_improvement_pct": (
                            (paired["reference_mae"] - metrics["mae"]) / abs(paired["reference_mae"]) * 100.0
                            if paired["reference_mae"]
                            else float("nan")
                        ),
                        "ic_improvement": metrics["ic_mean"] - paired["reference_ic"],
                        "icir_improvement": metrics["icir"] - paired["reference_icir"],
                        "r2_improvement_pp": (metrics["oos_r2"] - paired["reference_oos_r2"]) * 100.0,
                    }
                )

            dict_pred = predictions_by_model.get("dict_ntusd_simplified", {}).get(split)
            if dict_pred is not None and model["name"] in {"llm", "joint_ntusd_llm"}:
                dict_paired = paired_summary(dict_pred, pred, e4)
                metrics.update(
                    {
                        "dict_model": "dict",
                        "dict_dictionary": PRIMARY_DICT,
                        "dict_mae": dict_paired["reference_mae"],
                        "dict_ic": dict_paired["reference_ic"],
                        "dict_icir": dict_paired["reference_icir"],
                        "dict_oos_r2": dict_paired["reference_oos_r2"],
                        "dict_n_obs_paired": dict_paired["reference_n_obs_paired"],
                        "dict_mae_improvement_pct": (
                            (dict_paired["reference_mae"] - metrics["mae"]) / abs(dict_paired["reference_mae"]) * 100.0
                            if dict_paired["reference_mae"]
                            else float("nan")
                        ),
                        "dict_ic_improvement": metrics["ic_mean"] - dict_paired["reference_ic"],
                        "dict_icir_improvement": metrics["icir"] - dict_paired["reference_icir"],
                        "dict_r2_improvement_pp": (metrics["oos_r2"] - dict_paired["reference_oos_r2"]) * 100.0,
                    }
                )

            metric_rows.append(metrics)
            log(
                f"[oos] {model['name']} {split}: periods={metrics['n_periods']}, "
                f"IC={metrics['ic_mean']:.4f}, OOS_R2={metrics['oos_r2']:.4f}, MAE={metrics['mae']:.5f}"
            )

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    if not predictions.empty:
        predictions.to_csv(TEMP_DIR / "e6_oos_predictions.csv", index=False, encoding="utf-8-sig")

    oos_metrics_df = pd.DataFrame(metric_rows)
    if not oos_metrics_df.empty:
        columns = [
            "model", "dictionary", "spec", "split", "start_period", "end_period", "n_periods", "n_obs",
            "ic_mean", "ic_std", "icir", "ic_t", "ic_p", "pearson_ic_mean", "oos_r2", "oos_r2_overall",
            "mae", "rmse", "paired_mae_t", "paired_mae_p",
            "baseline_model", "baseline_mae", "baseline_ic", "baseline_icir", "baseline_oos_r2",
            "baseline_n_obs_paired", "mae_improvement_pct", "ic_improvement", "icir_improvement",
            "r2_improvement_pp",
            "dict_model", "dict_dictionary", "dict_mae", "dict_ic", "dict_icir", "dict_oos_r2",
            "dict_n_obs_paired", "dict_mae_improvement_pct", "dict_ic_improvement",
            "dict_icir_improvement", "dict_r2_improvement_pp",
            "llm_coef", "llm_t", "llm_p", "llm_t_fmb", "llm_p_fmb",
            "dict_coef", "dict_t", "dict_p", "dict_t_fmb", "dict_p_fmb",
        ]
        oos_metrics_df = oos_metrics_df.reindex(columns=columns)
        oos_metrics_df.to_csv(out_dir / "e6_llm_oos_metrics.csv", index=False, encoding="utf-8-sig")
        log(f"[output] e6_llm_oos_metrics.csv rows={len(oos_metrics_df)}")

    # Incremental regression output for the joint model.
    joint_coefficients = coefficients[
        (coefficients["model"] == "joint") & (coefficients["spec"] == "sent_exposure")
    ].copy()
    joint_fmb = fmb_summary(joint_coefficients, e4)
    joint_fmb = joint_fmb.copy()
    joint_fmb["row_type"] = "fmb_summary"
    joint_quarterly = joint_coefficients.copy()
    joint_quarterly["row_type"] = "quarterly"
    summary_columns = [
        "row_type", "target_quarter", "coefficient", "estimate", "std_error", "t_value", "p_value",
        "n_industries", "r2", "n_periods", "mean", "median", "winsorized_mean", "std",
        "t_fmb", "p_fmb", "t_nw", "p_nw", "hit_rate", "mean_r2", "mean_n", "first_period", "last_period",
    ]
    incremental = pd.concat([joint_quarterly, joint_fmb], ignore_index=True).reindex(columns=summary_columns)
    incremental.to_csv(out_dir / "e6_llm_incremental_regression.csv", index=False, encoding="utf-8-sig")

    # Attach joint FMB sentiment coefficients to the joint OOS row.
    if not oos_metrics_df.empty and not joint_fmb.empty:
        def _find(coefficient: str) -> pd.Series:
            return joint_fmb[joint_fmb["coefficient"] == coefficient].iloc[0]

        llm_row = _find("sent_x_exposure_llm")
        dict_row = _find(f"sent_x_exposure_{PRIMARY_DICT}")
        mask = (oos_metrics_df["model"] == "joint") & (oos_metrics_df["spec"] == "sent_exposure")
        for column, value in [
            ("llm_coef", llm_row["mean"]), ("llm_t", llm_row["t_nw"]), ("llm_p", llm_row["p_nw"]),
            ("llm_t_fmb", llm_row["t_fmb"]), ("llm_p_fmb", llm_row["p_fmb"]),
            ("dict_coef", dict_row["mean"]), ("dict_t", dict_row["t_nw"]), ("dict_p", dict_row["p_nw"]),
            ("dict_t_fmb", dict_row["t_fmb"]), ("dict_p_fmb", dict_row["p_fmb"]),
        ]:
            oos_metrics_df.loc[mask, column] = value
        oos_metrics_df.to_csv(out_dir / "e6_llm_oos_metrics.csv", index=False, encoding="utf-8-sig")

    long_table, correlations = sentiment_correlations(llm_scores)
    long_table.to_csv(out_dir / "e6_llm_vs_dict.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(out_dir / "e6_llm_vs_dict_summary.csv", index=False, encoding="utf-8-sig")
    log(f"[output] e6_llm_vs_dict.csv rows={len(long_table)}")

    human_table = human_consistency(llm_scores)
    if not human_table.empty:
        human_table.to_csv(out_dir / "e6_llm_human_consistency.csv", index=False, encoding="utf-8-sig")
        log(f"[output] e6_llm_human_consistency.csv rows={len(human_table)}")

    # Conclusion logic: H4 is supported only if the LLM model beats the
    # dictionary model on the test split and the joint model keeps a
    # significant LLM coefficient.
    test_rows = oos_metrics_df[oos_metrics_df["split"] == "test"]
    llm_test = test_rows[test_rows["model"] == "llm"]
    dict_test = test_rows[test_rows["dictionary"] == PRIMARY_DICT]
    joint_test = test_rows[test_rows["model"] == "joint"]
    if llm_test.empty or dict_test.empty or joint_test.empty:
        conclusion = "样本外主比较未能完整生成，H4 无法判定。"
    else:
        llm_row = llm_test.iloc[0]
        dict_row = dict_test.iloc[0]
        joint_row = joint_test.iloc[0]
        llm_better = (
            llm_row["ic_mean"] > dict_row["ic_mean"]
            and llm_row["oos_r2"] > dict_row["oos_r2"]
            and llm_row["mae"] < dict_row["mae"]
        )
        llm_coef = float(joint_row.get("llm_coef", float("nan")))
        llm_p = float(joint_row.get("llm_p", float("nan")))
        incremental_significant = np.isfinite(llm_coef) and np.isfinite(llm_p) and llm_p < 0.05
        if llm_better and incremental_significant:
            conclusion = (
                f"测试集上 LLM 模型 IC={llm_row['ic_mean']:.4f}、OOS R²={llm_row['oos_r2']:.4f}、"
                f"MAE={llm_row['mae']:.5f}，优于词典模型 IC={dict_row['ic_mean']:.4f}、"
                f"OOS R²={dict_row['oos_r2']:.4f}、MAE={dict_row['mae']:.5f}；"
                f"联合模型中 LLM 系数={llm_coef:.5f}（NW p={llm_p:.4f}）显著。H4 获得有限支持。"
            )
        else:
            conclusion = (
                f"测试集上 LLM 模型 IC={llm_row['ic_mean']:.4f}、OOS R²={llm_row['oos_r2']:.4f}、"
                f"MAE={llm_row['mae']:.5f}，词典模型 IC={dict_row['ic_mean']:.4f}、"
                f"OOS R²={dict_row['oos_r2']:.4f}、MAE={dict_row['mae']:.5f}；"
                f"联合模型中 LLM 系数={llm_coef:.5f}（NW p={llm_p:.4f}）。"
                "LLM 未同时优于词典且增量系数不显著，H4 未获支持。"
            )

    metrics = {
        "environment": environment,
        "llm_model": str(llm_scores["model"].iloc[0]),
        "prompt_version": str(llm_scores["prompt_version"].iloc[0]),
        "temperature": float(llm_scores["temperature"].iloc[0]),
        "llm_issues": int(len(llm_scores)),
        "panel_rows": int(len(panel)),
        "n_target_quarters": int(panel["target_quarter"].nunique()),
        "panel_first_quarter": panel["target_quarter"].min(),
        "panel_last_quarter": panel["target_quarter"].max(),
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "val_start": VAL_START,
        "val_end": VAL_END,
        "test_start": TEST_START,
        "test_end": TEST_END,
        "oos_metric_rows": int(len(oos_metrics_df)),
        "coefficient_rows": int(len(coefficients)),
        "fmb_skipped_cells": int(len(skip_rows)),
        "conclusion": conclusion,
        "script_sha256": sha256(Path(__file__)),
        "seed": SEED,
        "nw_lag": NW_LAG,
    }
    (out_dir / "e6_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_metrics_log(environment, llm_scores, correlations, oos_metrics_df, incremental, human_table, metrics, out_dir)
    log(f"[output] logs/e6_llm_metrics.md")
    log(f"[done] {conclusion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
