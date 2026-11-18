# -*- coding: utf-8 -*-
"""E1 experiment runner: dictionary sentiment scores and validation outputs."""

from __future__ import annotations

import datetime as dt
import json
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from e1_lib import (
    DICTIONARY_NAMES,
    E0_SAMPLE_ISSUES,
    E1_DIR,
    LOG_DIR,
    PILOT_N,
    ROOT,
    SEED,
    VALIDATION_DIR,
    VALIDATION_FRACTION,
    add_dictionary_words_to_jieba,
    build_scores,
    ensure_dirs,
    fetch_mpr_corpus,
    load_dictionaries,
    load_stopwords,
    sha256_file,
)


def safe_corr(
    x: list[float], y: list[float], method: str
) -> tuple[float | None, float | None, int]:
    x_series = pd.Series(x, dtype="float64").dropna().reset_index(drop=True)
    y_series = pd.Series(y, dtype="float64").dropna().reset_index(drop=True)
    length = min(len(x_series), len(y_series))
    x_values = x_series.iloc[:length].to_numpy()
    y_values = y_series.iloc[:length].to_numpy()
    if length < 3 or len(pd.unique(x_values)) < 2 or len(pd.unique(y_values)) < 2:
        return None, None, int(length)
    if method == "pearson":
        result = pearsonr(x_values, y_values)
    elif method == "spearman":
        result = spearmanr(x_values, y_values)
    elif method == "kendall":
        result = kendalltau(x_values, y_values)
    else:
        raise ValueError(method)
    return float(result.statistic), float(result.pvalue), int(length)


def sign_label(value: float, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def compute_dictionary_consistency(full_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs = [
        ("ntusd_simplified", "dlut"),
        ("ntusd_simplified", "mpr_domain"),
        ("dlut", "mpr_domain"),
    ]
    for left, right in pairs:
        left_frame = full_scores[full_scores["dictionary"].eq(left)].set_index("issue")
        right_frame = full_scores[full_scores["dictionary"].eq(right)].set_index("issue")
        common = left_frame.index.intersection(right_frame.index)
        left_scores = left_frame.loc[common, "sentiment_score"]
        right_scores = right_frame.loc[common, "sentiment_score"]
        for method in ["pearson", "spearman", "kendall"]:
            corr, pvalue, n = safe_corr(
                left_scores.tolist(), right_scores.tolist(), method
            )
            rows.append(
                {
                    "dictionary_1": left,
                    "dictionary_2": right,
                    "metric": method,
                    "n": n,
                    "correlation": corr,
                    "pvalue": pvalue,
                }
            )
        left_signs = left_scores.apply(sign_label)
        right_signs = right_scores.apply(sign_label)
        nonzero = left_signs.ne(0) & right_signs.ne(0)
        rows.append(
            {
                "dictionary_1": left,
                "dictionary_2": right,
                "metric": "sign_agreement_nonzero",
                "n": int(nonzero.sum()),
                "correlation": (
                    float((left_signs.loc[nonzero] == right_signs.loc[nonzero]).mean())
                    if nonzero.any()
                    else None
                ),
                "pvalue": None,
            }
        )
        rows.append(
            {
                "dictionary_1": left,
                "dictionary_2": right,
                "metric": "sign_agreement_all",
                "n": int(len(common)),
                "correlation": float((left_signs == right_signs).mean()),
                "pvalue": None,
            }
        )
    return pd.DataFrame(rows)


def select_validation_issues(full_index: pd.DataFrame) -> tuple[list[str], list[str]]:
    issues = full_index["issue"].tolist()
    rng = random.Random(SEED)
    pilot_count = min(PILOT_N, len(issues))
    pilot = rng.sample(issues, pilot_count)
    remaining = [issue for issue in issues if issue not in set(pilot)]
    validation_count = max(1, int(round(len(issues) * VALIDATION_FRACTION)))
    validation_count = min(validation_count, len(remaining))
    validation = rng.sample(remaining, validation_count)
    return sorted(pilot), sorted(validation)


def write_human_ratings_template(
    full_index: pd.DataFrame,
    pilot: list[str],
    validation: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_type, issues in [("pilot", pilot), ("validation", validation)]:
        for issue in issues:
            row = full_index[full_index["issue"].eq(issue)].iloc[0]
            rows.append(
                {
                    "issue": issue,
                    "publication_date": row["publication_date"],
                    "sample_type": sample_type,
                    "rater_id": "",
                    "human_score": "",
                    "confidence_1_5": "",
                    "status": "pending_human",
                    "notes": "",
                }
            )
    return pd.DataFrame(rows)


def write_human_guideline() -> None:
    guideline = """# E1 人工评分规则（提前写明）

## 评分单位

以一期《中国货币政策执行报告》全文为一个评分单位。评分对象是报告整体传递的货币政策立场、经济金融形势判断与政策取向，而不是个别句子或标题。

## 评分标尺

| 分值 | 含义 | 判断要点 |
|------|------|----------|
| 2 | 明显积极 | 总体强调增长改善、风险可控、政策积极支持，负面表述很少 |
| 1 | 偏积极 | 积极表述多于负面表述，但保留一定审慎或风险提示 |
| 0 | 中性 | 正负表述大致平衡，或主要是客观描述、数据陈述 |
| -1 | 偏消极 | 负面表述多于积极表述，或强调压力、风险与约束 |
| -2 | 明显消极 | 总体强调下行压力、风险隐患、疲软或政策收紧，积极表述很少 |

## 评分规则

1. 先通读目录、各部分小结、结论与政策展望，再回看正文关键段落。
2. 关注政策立场、经济形势判断、风险表述、流动性与信贷表述、政策取向。
3. 数字、表格、技术性说明本身不计为情绪；只有在作者明确给出判断时才计入。
4. 同一报告中正面与负面表述同时存在时，按总体倾向和篇幅权重综合判断。
5. 评分时不得查看自动词典得分，避免锚定。
6. 置信度使用 1-5 分：1 表示非常不确定，5 表示非常确定。
7. 试评 5 期后，由团队统一校准口径，再完成正式抽检。

## 重复评分

- 同一期报告至少由两名成员独立评分；若无法安排两人，同一评分人间隔至少 24 小时重复评分一次。
- 记录差异；若分差绝对值大于 1，先讨论校准，再提交。

## 文件填写

在 `validation/human_ratings.csv` 中填写 `rater_id`、`human_score`、`confidence_1_5` 和 `notes`。`status` 保持 `pending_human` 或改为 `completed`。填写完成后重跑 `run_e1.py`，脚本会读取已完成的人工评分并计算人工-自动一致性。
"""
    (VALIDATION_DIR / "human_rating_guideline.md").write_text(
        guideline, encoding="utf-8"
    )


def load_completed_human_ratings() -> pd.DataFrame:
    path = VALIDATION_DIR / "human_ratings.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    if "human_score" not in frame.columns:
        return pd.DataFrame()
    frame = frame[frame["human_score"].astype(str).str.strip().ne("")]
    frame["human_score"] = pd.to_numeric(frame["human_score"], errors="coerce")
    frame = frame[frame["human_score"].notna()]
    return frame


def compute_human_consistency(
    full_scores: pd.DataFrame, human_ratings: pd.DataFrame
) -> pd.DataFrame:
    if human_ratings.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    human = human_ratings.groupby("issue", as_index=False)["human_score"].mean()
    for dictionary in DICTIONARY_NAMES:
        scores = full_scores[full_scores["dictionary"].eq(dictionary)].set_index("issue")
        merged = human.merge(
            scores[["sentiment_score"]],
            left_on="issue",
            right_index=True,
            how="inner",
        )
        for method in ["pearson", "spearman", "kendall"]:
            corr, pvalue, n = safe_corr(
                merged["human_score"].tolist(),
                merged["sentiment_score"].tolist(),
                method,
            )
            rows.append(
                {
                    "dictionary": dictionary,
                    "metric": method,
                    "n": n,
                    "correlation": corr,
                    "pvalue": pvalue,
                }
            )
        rows.append(
            {
                "dictionary": dictionary,
                "metric": "sign_agreement_nonzero",
                "n": int(len(merged)),
                "correlation": (
                    float(
                        (
                            merged["human_score"].apply(sign_label)
                            == merged["sentiment_score"].apply(sign_label)
                        ).mean()
                    )
                    if not merged.empty
                    else None
                ),
                "pvalue": None,
            }
        )
    return pd.DataFrame(rows)


def coverage_by_year(scores: pd.DataFrame) -> pd.DataFrame:
    frame = scores.copy()
    frame["year"] = frame["issue"].str[:4]
    rows = []
    for (year, dictionary), group in frame.groupby(["year", "dictionary"]):
        rows.append(
            {
                "year": year,
                "dictionary": dictionary,
                "reports": len(group),
                "mean_pos_coverage": group["pos_coverage"].mean(),
                "mean_neg_coverage": group["neg_coverage"].mean(),
                "mean_pos_neg_coverage": group["pos_neg_coverage"].mean(),
                "min_pos_neg_coverage": group["pos_neg_coverage"].min(),
                "max_pos_neg_coverage": group["pos_neg_coverage"].max(),
                "zero_coverage_reports": int(group["pos_neg_coverage"].eq(0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["year", "dictionary"]).reset_index(drop=True)


def anomalous_texts(
    scores: pd.DataFrame, full_index: pd.DataFrame
) -> pd.DataFrame:
    zero = scores[scores["pos_neg_coverage"].eq(0)][
        ["issue", "dictionary", "text_chars", "tokens", "pos_neg_coverage"]
    ].copy()
    short = full_index[~full_index["min_chars_pass"]][
        ["issue", "text_chars", "text_status", "download_status", "pages"]
    ].copy()
    failed = full_index[
        full_index["text_status"].astype(str).str.startswith("fail")
        | full_index["download_status"].astype(str).str.startswith("fail")
    ][["issue", "text_chars", "text_status", "download_status", "pages"]].copy()
    zero["anomaly_type"] = "zero_dictionary_coverage"
    short["anomaly_type"] = "short_text"
    failed["anomaly_type"] = "fetch_or_extract_failure"
    combined = pd.concat([zero, short, failed], ignore_index=True, sort=False)
    return combined.drop_duplicates(subset=["issue", "anomaly_type"]).sort_values(
        ["issue", "anomaly_type"]
    )


def write_log(
    full_index: pd.DataFrame,
    full_scores: pd.DataFrame,
    sample_scores: pd.DataFrame,
    dictionaries: dict[str, dict[str, Any]],
    consistency: pd.DataFrame,
    human_consistency: pd.DataFrame,
    human_ratings: pd.DataFrame,
    coverage: pd.DataFrame,
    anomalies: pd.DataFrame,
    unmatched: pd.DataFrame,
    pilot: list[str],
    validation: list[str],
    environment: dict[str, str],
) -> dict[str, Any]:
    lines = [
        "# E1 词典情绪度量与人工抽检 Baseline 实验日志",
        "",
        "- 实验编号：E1",
        f"- 运行日期：{dt.date.today().isoformat()}",
        "- 脚本：`experiments/E1/run_e1.py`",
        f"- Python：{environment['python']}",
        f"- pandas：{environment['pandas']}",
        f"- jieba：{environment['jieba']}",
        f"- pdfplumber：{environment['pdfplumber']}",
        f"- requests：{environment['requests']}",
        f"- 随机种子：{SEED}",
        f"- 人工抽检比例：{VALIDATION_FRACTION:.0%}",
        f"- 试评期数：{PILOT_N}",
        "- 对照设置：NTUSD 简体版 vs 大连理工情感词汇本体 vs 自制 MPR 领域词典；自动评分 vs 人工评分（待完成）。",
        "",
        "## 1. 样本与抓取",
        "",
        f"- 全样本期数：{len(full_index)}",
        f"- 全样本区间：{full_index['issue'].min()} 至 {full_index['issue'].max()}",
        f"- PDF 下载成功：{int(full_index['download_status'].eq('ok').sum())}/{len(full_index)}",
        f"- 文本提取成功或缓存：{int(full_index['text_status'].isin(['ok', 'cached']).sum())}/{len(full_index)}",
        f"- 文本字符数 > 10000：{int(full_index['min_chars_pass'].sum())}/{len(full_index)}",
        f"- E0 样本期数：{len(sample_scores['issue'].drop_duplicates())}",
        "",
        "### 全样本清单",
        "",
        "| 期次 | 发布日期 | 页数 | 文本字符数 | PDF 状态 | 文本状态 | >10000 |",
        "|------|----------|------|------------|----------|----------|---------|",
    ]
    for _, row in full_index.iterrows():
        lines.append(
            f"| {row['issue']} | {row['publication_date']} | {row['pages']} | "
            f"{row['text_chars']} | {row['download_status']} | "
            f"{row['text_status']} | {row['min_chars_pass']} |"
        )

    lines += [
        "",
        "## 2. 词典与预处理",
        "",
        "- 分词工具：jieba，固定版本；词典词条已加入 jieba 自定义词典。",
        "- 停用词表：stopwords-iso/stopwords-zh（MIT），另加报告标题、部分编号等无情绪含义词。",
        "- 文本预处理：按行读取，分词后去除停用词、纯标点、纯数字、纯英文和非中文字符词。",
        "",
        "| 词典 | 来源与许可 | 正向词条 | 负向词条 | 备注 |",
        "|------|------------|----------|----------|------|",
    ]
    for name in DICTIONARY_NAMES:
        metadata = dictionaries[name]["metadata"]
        lines.append(
            f"| {name} | {metadata['source']} | "
            f"{len(dictionaries[name]['positive'])} | "
            f"{len(dictionaries[name]['negative'])} | "
            f"去除歧义/重叠 {metadata.get('ambiguous_removed', 0)} 条 |"
        )

    lines += [
        "",
        "## 3. 评分指标",
        "",
        "- 情绪得分：`(正词频 - 负词频) / (正词频 + 负词频)`；无命中时为 0。",
        "- 加权得分：DLUT 与自制词典使用词条权重；NTUSD 为等权。",
        "- 正词覆盖率：`正词频 / 总词数`；负词覆盖率：`负词频 / 总词数`。",
        "- 正负词覆盖度：`(正词频 + 负词频) / 总词数`。",
        "",
        "### 全样本得分摘要（按词典）",
        "",
        "| 词典 | 期数 | 平均情绪得分 | 中位情绪得分 | 平均正覆盖率 | 平均负覆盖率 | 平均正负覆盖度 | 零覆盖期数 |",
        "|------|------|--------------|--------------|--------------|--------------|----------------|------------|",
    ]
    for name in DICTIONARY_NAMES:
        group = full_scores[full_scores["dictionary"].eq(name)]
        lines.append(
            f"| {name} | {len(group)} | {group['sentiment_score'].mean():.6f} | "
            f"{group['sentiment_score'].median():.6f} | "
            f"{group['pos_coverage'].mean():.6f} | "
            f"{group['neg_coverage'].mean():.6f} | "
            f"{group['pos_neg_coverage'].mean():.6f} | "
            f"{int(group['pos_neg_coverage'].eq(0).sum())} |"
        )

    lines += [
        "",
        "### 每年覆盖率摘要（全样本）",
        "",
        "| 年份 | 词典 | 期数 | 平均正覆盖率 | 平均负覆盖率 | 平均正负覆盖度 | 零覆盖期数 |",
        "|------|------|------|--------------|--------------|----------------|------------|",
    ]
    for _, row in coverage.iterrows():
        lines.append(
            f"| {row['year']} | {row['dictionary']} | {row['reports']} | "
            f"{row['mean_pos_coverage']:.6f} | {row['mean_neg_coverage']:.6f} | "
            f"{row['mean_pos_neg_coverage']:.6f} | {row['zero_coverage_reports']} |"
        )

    lines += [
        "",
        "## 4. 词典间一致性",
        "",
        "| 词典 1 | 词典 2 | 指标 | 样本数 | 相关系数 | p 值 |",
        "|--------|--------|------|--------|----------|------|",
    ]
    for _, row in consistency.iterrows():
        corr = "" if pd.isna(row["correlation"]) else f"{row['correlation']:.6f}"
        pvalue = "" if pd.isna(row["pvalue"]) else f"{row['pvalue']:.3g}"
        lines.append(
            f"| {row['dictionary_1']} | {row['dictionary_2']} | "
            f"{row['metric']} | {row['n']} | {corr} | {pvalue} |"
        )

    positive_score_counts = {
        name: int(full_scores[full_scores["dictionary"].eq(name)]["sentiment_score"].gt(0).sum())
        for name in DICTIONARY_NAMES
    }
    lines += [
        "",
        f"- 注意：全样本中三套词典的情绪得分均为正（正分期数："
        f"{', '.join(f'{k}={v}/{len(full_index)}' for k, v in positive_score_counts.items())}）。"
        "因此符号一致性为 1.0 主要反映 MPR 文本整体偏积极与词典词表覆盖结构，不能单独作为强一致性证据；应以 Pearson、Spearman、Kendall 相关系数为主。",
    ]

    lines += [
        "",
        "## 5. 人工抽检",
        "",
        f"- 试评期次：{', '.join(pilot)}",
        f"- 正式抽检期次：{', '.join(validation)}",
        f"- 已完成人工评分期数：{len(human_ratings)}",
        "- 评分规则：见 `validation/human_rating_guideline.md`。",
        "",
    ]
    if human_ratings.empty:
        lines += [
            "**人工评分状态：未完成。** `validation/human_ratings.csv` 已生成模板，`human_score` 字段待团队成员填写。",
            "在人工评分完成前，不计算人工-自动相关系数，也不判定 H1 通过。",
            "",
        ]
    else:
        lines += [
            "| 词典 | 指标 | 样本数 | 相关系数 | p 值 |",
            "|------|------|--------|----------|------|",
        ]
        for _, row in human_consistency.iterrows():
            corr = "" if pd.isna(row["correlation"]) else f"{row['correlation']:.6f}"
            pvalue = "" if pd.isna(row["pvalue"]) else f"{row['pvalue']:.3g}"
            lines.append(
                f"| {row['dictionary']} | {row['metric']} | {row['n']} | "
                f"{corr} | {pvalue} |"
            )
        lines.append("")

    lines += [
        "## 6. 缺失词与异常文本",
        "",
        f"- 未命中任何词典的高频词已写入 `validation/unmatched_terms.csv`，共 {len(unmatched)} 个不同词。",
        f"- 异常文本清单已写入 `validation/anomalous_texts.csv`，共 {len(anomalies)} 条记录。",
        "",
    ]
    if not unmatched.empty:
        lines += [
            "### 未命中高频词前 30 个",
            "",
            "| 词 | 频次 |",
            "|----|------|",
        ]
        for _, row in unmatched.head(30).iterrows():
            lines.append(f"| {row['term']} | {row['count']} |")
        lines.append("")
    if not anomalies.empty:
        lines += ["### 异常文本摘要", ""]
        for _, row in anomalies.iterrows():
            lines.append(
                f"- {row['issue']}：{row['anomaly_type']}，"
                f"text_chars={row.get('text_chars', '')}，"
                f"text_status={row.get('text_status', '')}"
            )
        lines.append("")

    lines += [
        "## 7. 重复评分稳定性",
        "",
        "- 词典评分为确定性计算；同一文本重复运行两次的得分差异为 0。",
        "- 人工重复评分稳定性将在人工评分完成后计算，指标为同一期两次评分的组内相关系数或分差均值。",
        "",
        "## 8. 完成条件核对",
        "",
        f"- 至少一套词典完成全样本评分：通过（{len(DICTIONARY_NAMES)} 套词典，{len(full_scores)} 行）。",
        f"- 人工抽检样本量足够：样本已确定（试评 {len(pilot)} 期 + 正式 {len(validation)} 期），人工评分{'已完成' if not human_ratings.empty else '未完成'}。",
        f"- 一致性结果、缺失词和异常文本均已记录：词典间一致性已记录；人工一致性{'已记录' if not human_ratings.empty else '待人工评分'}；缺失词与异常文本已记录。",
        "- 具体一致性阈值由团队和指导教师确认，本实验不预设为研究结论。",
        "",
    ]
    if human_ratings.empty:
        lines += [
            "**H1 状态：未判定。** 原因：人工评分未完成，且一致性阈值尚未由团队和指导教师确认。",
            "在人工评分完成前，不把词典间一致性或自动评分结果表述为 H1 通过。",
            "",
        ]
    else:
        lines += [
            "**H1 状态：待阈值确认。** 人工-自动相关性已计算；是否通过需按团队和指导教师确认的阈值判断。",
            "",
        ]

    lines += [
        "## 9. 输出文件",
        "",
        "- `sentiment_dictionary_sample.csv`：E0 五期样本在三套词典下的评分。",
        "- `sentiment_dictionary_full.csv`：全样本期次在三套词典下的评分。",
        "- `validation/human_ratings.csv`：试评与正式抽检人工评分表。",
        "- `validation/human_rating_guideline.md`：人工评分规则。",
        "- `dictionaries/mpr_domain_dictionary.csv`：自制 MPR 领域词典。",
        "- `dictionaries/dictionary_sources.md`：词典、停用词来源与许可。",
        "- `validation/dictionary_consistency.csv`：词典间相关性与符号一致性。",
        "- `validation/coverage_by_year.csv`：逐年度覆盖率。",
        "- `validation/unmatched_terms.csv`：未命中任何词典的高频词。",
        "- `validation/anomalous_texts.csv`：异常文本清单。",
        "- `mpr_index_full.csv`：全样本元数据。",
        "- `e1_metrics.json`：关键指标汇总。",
        "",
    ]

    (LOG_DIR / "e1_sentiment_validation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    metrics = {
        "mpr_issues": len(full_index),
        "mpr_pdf_success": int(full_index["download_status"].eq("ok").sum()),
        "mpr_text_success_or_cached": int(
            full_index["text_status"].isin(["ok", "cached"]).sum()
        ),
        "mpr_text_min_chars_pass": int(full_index["min_chars_pass"].sum()),
        "dictionaries": len(DICTIONARY_NAMES),
        "dictionary_scores_rows": len(full_scores),
        "validation_sample_reports": len(pilot) + len(validation),
        "human_ratings_completed": len(human_ratings),
        "unmatched_terms": len(unmatched),
        "anomaly_records": len(anomalies),
        "seed": SEED,
        "script_sha256": sha256_file(Path(__file__)),
    }
    (E1_DIR / "e1_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ensure_dirs()

    import pdfplumber as _pdfplumber
    import requests as _requests

    environment = {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "jieba": __import__("jieba").__version__,
        "pdfplumber": _pdfplumber.__version__,
        "requests": _requests.__version__,
    }

    full_index = fetch_mpr_corpus()
    full_index.to_csv(
        E1_DIR / "mpr_index_full.csv", index=False, encoding="utf-8-sig"
    )

    dictionaries = load_dictionaries()
    stopwords = load_stopwords()
    add_dictionary_words_to_jieba(dictionaries)

    full_scores, unmatched = build_scores(full_index, dictionaries, stopwords)
    full_scores.to_csv(
        E1_DIR / "sentiment_dictionary_full.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sample_issues = [
        issue for issue in E0_SAMPLE_ISSUES if issue in set(full_index["issue"])
    ]
    sample_scores = full_scores[full_scores["issue"].isin(sample_issues)].copy()
    sample_scores.to_csv(
        E1_DIR / "sentiment_dictionary_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pilot, validation = select_validation_issues(full_index)
    human_ratings_path = VALIDATION_DIR / "human_ratings.csv"
    if human_ratings_path.exists():
        human_ratings_template = pd.read_csv(
            human_ratings_path, encoding="utf-8-sig", dtype=str
        ).fillna("")
        expected = {
            issue
            for issue in pilot + validation
        }
        existing = set(human_ratings_template.get("issue", []).astype(str))
        missing = sorted(expected - existing)
        if missing:
            missing_rows = write_human_ratings_template(
                full_index, [issue for issue in pilot if issue in missing], []
            )
            missing_rows = missing_rows[
                missing_rows["issue"].isin(set(missing))
            ]
            human_ratings_template = pd.concat(
                [human_ratings_template, missing_rows], ignore_index=True
            )
    else:
        human_ratings_template = write_human_ratings_template(
            full_index, pilot, validation
        )
    human_ratings_template.to_csv(
        human_ratings_path,
        index=False,
        encoding="utf-8-sig",
    )
    write_human_guideline()

    consistency = compute_dictionary_consistency(full_scores)
    consistency.to_csv(
        VALIDATION_DIR / "dictionary_consistency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    human_ratings = load_completed_human_ratings()
    human_consistency = compute_human_consistency(full_scores, human_ratings)
    if not human_consistency.empty:
        human_consistency.to_csv(
            VALIDATION_DIR / "human_consistency.csv",
            index=False,
            encoding="utf-8-sig",
        )

    coverage = coverage_by_year(full_scores)
    coverage.to_csv(
        VALIDATION_DIR / "coverage_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )

    anomalies = anomalous_texts(full_scores, full_index)
    anomalies.to_csv(
        VALIDATION_DIR / "anomalous_texts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    unmatched.to_csv(
        VALIDATION_DIR / "unmatched_terms.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = write_log(
        full_index,
        full_scores,
        sample_scores,
        dictionaries,
        consistency,
        human_consistency,
        human_ratings,
        coverage,
        anomalies,
        unmatched,
        pilot,
        validation,
        environment,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
