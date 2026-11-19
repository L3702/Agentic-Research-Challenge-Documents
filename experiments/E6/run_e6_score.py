"""E6 LLM scoring: fixed-model, fixed-prompt, fixed-temperature batch scoring.

This script scores every MPR issue with one LLM model through the local
Codex++ Responses gateway.  The prompt version (v1), model name and
temperature are written into every raw record and into the run log so the
scoring step is reproducible.

Usage:
    python experiments/E6/run_e6_score.py --model qwen3.8-27b --concurrency 3

Raw per-issue responses are cached under _temp/e6_llm_raw/.  Re-running the
script skips issues whose cached response parsed successfully, so an
interrupted batch can be resumed without duplicate API calls.

Outputs:
    experiments/E6/e6_llm_scores.csv
    experiments/E6/logs/e6_llm_prompt_and_cost.md
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
E6_DIR = ROOT / "experiments" / "E6"
LOG_DIR = E6_DIR / "logs"
RAW_DIR = ROOT / "_temp" / "e6_llm_raw"
CONFIG_PATH = Path(r"C:\CodexData\config.toml")
MPR_INDEX_PATH = ROOT / "experiments" / "E1" / "mpr_index_full.csv"
SENTIMENT_PATH = ROOT / "experiments" / "E1" / "sentiment_dictionary_full.csv"
TEXT_DIR = ROOT / "_temp" / "e1_mpr" / "text"

PROMPT_VERSION = "v1"
PROMPT_V1 = """你是一位宏观经济与货币政策分析专家。请阅读以下《中国货币政策执行报告》（季度：{issue}）的全文，评估该报告整体传递的货币政策立场与经济金融情绪。

请严格按以下 JSON 结构输出，不要输出任何解释、标题或 Markdown 代码块：
{{
  "sentiment_score": 0.00,
  "stance": 0,
  "dimensions": {{
    "policy_stance": 0.00,
    "economic_judgment": 0.00,
    "risk_tone": 0.00,
    "liquidity_credit": 0.00,
    "policy_outlook": 0.00
 }},
  "key_evidence": ["原文片段1", "原文片段2"],
  "reasoning": "一句话中文说明"
}}

评分口径：
- sentiment_score：连续分值，范围 [-1, 1]。+1 表示明显积极/宽松，-1 表示明显消极/收紧，0 为中性。
- stance：整数，取值 {{-2, -1, 0, 1, 2}}，分别对应明显消极、偏消极、中性、偏积极、明显积极。
- dimensions 各项均在 [-1, 1]，数值越高越积极/宽松：
  - policy_stance：货币政策立场（宽松/积极 vs 收紧/审慎）
  - economic_judgment：对经济与金融形势的判断（积极 vs 消极）
  - risk_tone：风险表述（风险可控/乐观 vs 风险隐患突出）
  - liquidity_credit：流动性与信贷（充裕/扩张 vs 收紧/收缩）
  - policy_outlook：未来政策取向（偏宽松/积极 vs 偏收紧/审慎）
- key_evidence：最多 5 条支撑评分的原文片段，每条不超过 40 字。
- reasoning：一句话说明总体判断依据。

判断要求：
1. 以报告整体立场为准，不以个别句子或标题下结论。
2. 数字、表格、技术性说明本身不计为情绪，只有作者明确给出判断时才计入。
3. 正负表述同时存在时，按总体倾向与篇幅权重综合判断。
"""


def ensure_dirs() -> None:
    for path in [E6_DIR, LOG_DIR, RAW_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_gateway_config() -> tuple[str, str]:
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    base_match = re.search(r'base_url\s*=\s*"([^"]+)"', raw)
    token_match = re.search(r'experimental_bearer_token\s*=\s*"([^"]+)"', raw)
    if not base_match or not token_match:
        raise RuntimeError(f"gateway config not found in {CONFIG_PATH}")
    return base_match.group(1).rstrip("/"), token_match.group(1)


def build_input(issue: str, text: str) -> str:
    prompt = PROMPT_V1.replace("{issue}", issue)
    return (
        prompt
        + "\n\n===== 报告全文开始 =====\n"
        + text
        + "\n===== 报告全文结束 ====="
    )


def extract_output_text(response: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    return "".join(chunks)


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    candidate = text.strip()
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
    candidate = re.sub(r"\s*```$", "", candidate).strip()
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj, None
        return None, "parsed JSON is not an object"
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict):
                    return obj, None
            except json.JSONDecodeError:
                pass
        return None, f"json decode error: {exc}"


def call_once(
    base: str,
    token: str,
    model: str,
    issue: str,
    text: str,
    temperature: float,
    max_output_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    session = requests.Session()
    body = {
        "model": model,
        "input": build_input(issue, text),
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    started = time.time()
    response = session.post(
        f"{base}/responses",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        timeout=timeout,
    )
    latency = time.time() - started
    response.raise_for_status()
    payload = response.json()
    usage = payload.get("usage", {}) or {}
    raw_text = extract_output_text(payload)
    parsed, parse_error = parse_json_object(raw_text)
    return {
        "issue": issue,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "timestamp": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "latency_sec": round(latency, 3),
        "usage": usage,
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "raw_output_text": raw_text,
        "parsed": parsed,
        "parse_error": parse_error,
    }


def score_issue(
    base: str,
    token: str,
    model: str,
    issue: str,
    text: str,
    temperature: float,
    max_output_tokens: int,
    timeout: int,
    max_retries: int,
) -> dict[str, Any]:
    cache_path = RAW_DIR / f"{issue}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("parse_error") is None and cached.get("parsed") is not None:
            cached["cache_hit"] = True
            return cached

    last_error = "unknown"
    retries = 0
    for attempt in range(max_retries + 1):
        try:
            record = call_once(
                base, token, model, issue, text, temperature, max_output_tokens, timeout
            )
            retries = attempt
            record["retries"] = retries
            cache_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return record
        except Exception as exc:  # noqa: BLE001 - network APIs raise many types
            last_error = f"{type(exc).__name__}: {exc}"
            retries = attempt + 1
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 8))

    record = {
        "issue": issue,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "timestamp": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "latency_sec": None,
        "usage": {},
        "input_tokens": 0,
        "output_tokens": 0,
        "raw_output_text": "",
        "parsed": None,
        "parse_error": f"call failed after {max_retries} retries: {last_error}",
        "retries": retries,
    }
    cache_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=4000)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    ensure_dirs()
    base, token = load_gateway_config()

    index = pd.read_csv(MPR_INDEX_PATH, dtype={"issue": str})
    sentiment = pd.read_csv(SENTIMENT_PATH, dtype={"issue": str})
    dict_wide = sentiment.pivot(index="issue", columns="dictionary", values="sentiment_score")
    dict_wide = dict_wide.rename(
        columns={name: f"dict_{name}" for name in dict_wide.columns}
    ).reset_index()

    session = requests.Session()
    records: list[dict[str, Any]] = []
    total = len(index)
    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {}
        for row in index.itertuples(index=False):
            text_path = TEXT_DIR / f"{row.issue}.txt"
            if not text_path.exists():
                records.append(
                    {
                        "issue": row.issue,
                        "model": args.model,
                        "prompt_version": PROMPT_VERSION,
                        "temperature": args.temperature,
                        "timestamp": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
                        "latency_sec": None,
                        "usage": {},
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "raw_output_text": "",
                        "parsed": None,
                        "parse_error": f"text file missing: {text_path}",
                        "retries": 0,
                    }
                )
                done += 1
                continue
            text = text_path.read_text(encoding="utf-8", errors="replace")
            futures[
                pool.submit(
                    score_issue,
                    base,
                    token,
                    args.model,
                    row.issue,
                    text,
                    args.temperature,
                    args.max_output_tokens,
                    args.timeout,
                    args.max_retries,
                )
            ] = row.issue
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            done += 1
            status = "ok" if record.get("parsed") is not None else "FAIL"
            print(f"[{done}/{total}] {record['issue']} {status} {record.get('latency_sec')}s")

    raw = pd.DataFrame(records)
    raw = raw.sort_values("issue").reset_index(drop=True)
    raw["raw_json_path"] = raw["issue"].map(lambda issue: str(RAW_DIR / f"{issue}.json"))
    parsed = raw["parsed"].map(lambda value: value if isinstance(value, dict) else {})
    scores = pd.DataFrame(
        {
            "llm_sentiment": parsed.map(lambda d: d.get("sentiment_score")),
            "llm_stance": parsed.map(lambda d: d.get("stance")),
            "llm_policy_stance": parsed.map(lambda d: (d.get("dimensions") or {}).get("policy_stance")),
            "llm_economic_judgment": parsed.map(
                lambda d: (d.get("dimensions") or {}).get("economic_judgment")
            ),
            "llm_risk_tone": parsed.map(lambda d: (d.get("dimensions") or {}).get("risk_tone")),
            "llm_liquidity_credit": parsed.map(
                lambda d: (d.get("dimensions") or {}).get("liquidity_credit")
            ),
            "llm_policy_outlook": parsed.map(
                lambda d: (d.get("dimensions") or {}).get("policy_outlook")
            ),
        }
    )
    scores.index = raw["issue"]
    output = index[["issue", "report_period", "publication_date", "text_chars"]].merge(
        scores, left_on="issue", right_index=True, how="left"
    )
    raw_summary = raw[
        [
            "issue",
            "model",
            "prompt_version",
            "temperature",
            "max_output_tokens",
            "latency_sec",
            "input_tokens",
            "output_tokens",
            "retries",
            "raw_json_path",
            "parse_error",
        ]
    ]
    output = output.merge(raw_summary, on="issue", how="left")
    output = output.merge(dict_wide, on="issue", how="left")
    columns = [
        "issue",
        "report_period",
        "publication_date",
        "text_chars",
        "llm_sentiment",
        "llm_stance",
        "llm_policy_stance",
        "llm_economic_judgment",
        "llm_risk_tone",
        "llm_liquidity_credit",
        "llm_policy_outlook",
        "dict_ntusd_simplified",
        "dict_dlut",
        "dict_mpr_domain",
        "model",
        "prompt_version",
        "temperature",
        "max_output_tokens",
        "latency_sec",
        "input_tokens",
        "output_tokens",
        "retries",
        "raw_json_path",
        "parse_error",
    ]
    output = output.reindex(columns=columns)
    output["parse_status"] = output["parse_error"].map(
        lambda value: "ok" if pd.isna(value) else "fail"
    )
    scores_path = E6_DIR / "e6_llm_scores.csv"
    output.to_csv(scores_path, index=False, encoding="utf-8-sig")

    ok = int((output["parse_status"] == "ok").sum())
    failed = total - ok
    total_input = int(output["input_tokens"].fillna(0).sum())
    total_output = int(output["output_tokens"].fillna(0).sum())
    total_latency = float(output["latency_sec"].dropna().sum())
    total_retries = int(raw["retries"].fillna(0).sum())
    caps_used = sorted(
        {int(value) for value in output["max_output_tokens"].dropna().unique()}
    )

    log_lines = [
        "# E6 LLM 评分：Prompt、模型与成本记录",
        "",
        f"- 模型：`{args.model}`（固定，来自 Codex++ 网关，端砚上游）",
        f"- Prompt 版本：`{PROMPT_VERSION}`（固定，见下方全文）",
        f"- 温度：`{args.temperature}`",
        f"- 最大输出 tokens：`{args.max_output_tokens}`",
        f"- 本批次实际使用过的输出上限：`{caps_used}`（主批 4000；输出截断期次用 6000 重试；另有 3 期在更早的 1500 上限批次中已成功并命中缓存）",
        f"- 并发：`{args.concurrency}`；单期超时：`{args.timeout}s`；最大重试：`{args.max_retries}`",
        f"- 期数：{total}；解析成功：{ok}；失败：{failed}",
        f"- 累计输入 tokens：{total_input}；累计输出 tokens：{total_output}",
        f"- 累计调用耗时（串行等价）：{total_latency:.1f}s；累计重试次数：{total_retries}",
        "- 直接 API 费用：0（使用本机 Codex++ 网关订阅，网关不单独计费）；成本以 token 量与耗时记录",
        "",
        "## 复现命令",
        "",
        "```powershell",
        f"python experiments/E6/run_e6_score.py --model {args.model} --concurrency {args.concurrency}",
        "```",
        "",
        "## Prompt v1 全文",
        "",
        "```text",
        PROMPT_V1.rstrip(),
        "```",
        "",
        "> 注：Prompt v1 模板中的 JSON 示例使用了双大括号（固定版本的历史遗留），模型仍按单大括号 schema 输出；该版本已固定，未因测试结果修改。",
        "",
        "## 失败记录",
        "",
    ]
    failures = output[output["parse_status"] != "ok"]
    if failures.empty:
        log_lines.append("- 无")
    else:
        for row in failures.itertuples(index=False):
            log_lines.append(f"- {row.issue}: {row.parse_error}")
    (LOG_DIR / "e6_llm_prompt_and_cost.md").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )

    print(f"[output] {scores_path} rows={len(output)} ok={ok} failed={failed}")
    print(
        f"[cost] input_tokens={total_input} output_tokens={total_output} "
        f"latency_sum={total_latency:.1f}s retries={total_retries}"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
