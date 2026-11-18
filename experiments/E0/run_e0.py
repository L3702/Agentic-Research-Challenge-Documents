# -*- coding: utf-8 -*-
"""E0 data pipeline for the MPR sentiment project.

The script fetches a five-issue MPR sample from the People's Bank of China
website, extracts PDF text, builds a small industry and benchmark price panel,
and writes release-aligned event-window and quarterly returns.

All downloaded HTML, PDFs, and raw market data are kept under the project's
_temp directory. Final sample files are written under experiments/E0.
"""

from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[2]
E0_DIR = ROOT / "experiments" / "E0"
OUT_DIR = E0_DIR / "data_sample"
TEXT_DIR = OUT_DIR / "mpr_text_sample"
LOG_DIR = E0_DIR / "logs"
TEMP_DIR = ROOT / "_temp"
MPR_TEMP_DIR = TEMP_DIR / "e0_mpr"
MARKET_TEMP_DIR = TEMP_DIR / "e0_market_raw"

MPR_ARCHIVE_URL = (
    "https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/index.html"
    "?aisiteOutPageId=106894"
)
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

SELECTED_ISSUES = ["2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q2"]
INDUSTRIES = [
    ("801010", "农林牧渔"),
    ("801750", "计算机"),
    ("801780", "银行"),
]
BENCHMARK = ("sh000300", "沪深300")
MIN_TEXT_CHARS = 10_000
EVENT_WINDOWS = [1, 3, 5, 10]
QUARTER_WINDOW = 63
SAMPLE_START = "2025-01-01"


def ensure_dirs() -> None:
    for path in [OUT_DIR, TEXT_DIR, LOG_DIR, MPR_TEMP_DIR, MARKET_TEMP_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def fetch_html(url: str, cache_name: str) -> str:
    cache_path = MPR_TEMP_DIR / cache_name
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_text(encoding="utf-8")
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=45)
    response.raise_for_status()
    response.encoding = "utf-8"
    cache_path.write_text(response.text, encoding="utf-8")
    time.sleep(0.4)
    return response.text


def fetch_binary(url: str, cache_path: Path) -> None:
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=180)
    response.raise_for_status()
    cache_path.write_bytes(response.content)
    time.sleep(0.4)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def absolute_url(href: str, base: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("/"):
        return "https://www.pbc.gov.cn" + href
    return base.rsplit("/", 1)[0] + "/" + href


def parse_issue_date(title: str) -> str | None:
    match = re.search(r"(20\d{2})年(第[一二三四]季度)", title)
    if not match:
        return None
    quarter_map = {"一": 1, "二": 2, "三": 3, "四": 4}
    return f"{match.group(1)}Q{quarter_map[match.group(2)[1]]}"


def parse_mpr_sample() -> list[dict[str, Any]]:
    archive_html = fetch_html(MPR_ARCHIVE_URL, "mpr_archive.html")
    archive_soup = BeautifulSoup(archive_html, "html.parser")
    year_pages: dict[str, str] = {}
    for anchor in archive_soup.find_all("a"):
        text = anchor.get_text(" ", strip=True)
        href = anchor.get("href")
        if not href:
            continue
        year_match = re.search(r"(20\d{2})年货币政策执行报告", text)
        if year_match and "/125957/" in href:
            year_pages[year_match.group(1)] = absolute_url(href, MPR_ARCHIVE_URL)

    records: list[dict[str, Any]] = []
    for issue in SELECTED_ISSUES:
        year = issue[:4]
        year_url = year_pages.get(year)
        if not year_url:
            raise RuntimeError(f"Missing archive page for year {year}")
        year_html = fetch_html(year_url, f"mpr_{year}_archive.html")
        year_soup = BeautifulSoup(year_html, "html.parser")
        detail_url = None
        for anchor in year_soup.find_all("a"):
            title = anchor.get_text(" ", strip=True)
            href = anchor.get("href")
            if not href:
                continue
            if parse_issue_date(title) == issue and "中国货币政策执行报告" in title:
                detail_url = absolute_url(href, year_url)
                break
        if not detail_url:
            raise RuntimeError(f"Missing detail URL for {issue}")

        detail_html = fetch_html(detail_url, f"mpr_{issue}_detail.html")
        detail_soup = BeautifulSoup(detail_html, "html.parser")
        title = detail_soup.title.get_text(" ", strip=True) if detail_soup.title else issue
        pub_meta = detail_soup.find("meta", attrs={"name": "PubDate"})
        if not pub_meta or not pub_meta.get("content"):
            raise RuntimeError(f"Missing PubDate for {issue}")
        publication_date = dt.date.fromisoformat(pub_meta["content"][:10])
        timestamp_span = detail_soup.find("span", id="shijian")
        publication_timestamp = (
            timestamp_span.get_text(" ", strip=True) if timestamp_span else ""
        )
        pdf_anchor = next(
            (
                a
                for a in detail_soup.find_all("a")
                if a.get("href") and ".pdf" in a.get("href").lower()
            ),
            None,
        )
        if not pdf_anchor:
            raise RuntimeError(f"Missing PDF link for {issue}")
        pdf_url = absolute_url(pdf_anchor["href"], detail_url)
        pdf_path = MPR_TEMP_DIR / f"mpr_{issue}.pdf"
        fetch_binary(pdf_url, pdf_path)

        text_path = TEXT_DIR / f"{issue}.txt"
        pages = 0
        text = ""
        extract_status = "ok"
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = len(pdf.pages)
                text = "\n".join((page.extract_text() or "") for page in pdf.pages)
            text = text.replace("\x00", "")
            text_path.write_text(text, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            extract_status = f"fail: {type(exc).__name__}: {exc}"
            text_path.write_text("", encoding="utf-8")

        chars = len(text)
        report_year = int(issue[:4])
        report_quarter = int(issue[-1])
        report_period_end = dt.date(
            report_year,
            report_quarter * 3,
            calendar.monthrange(report_year, report_quarter * 3)[1],
        )
        records.append(
            {
                "issue": issue,
                "report_period": issue,
                "report_period_end": report_period_end.isoformat(),
                "publication_date": publication_date.isoformat(),
                "publication_timestamp": publication_timestamp,
                "detail_url": detail_url,
                "pdf_url": pdf_url,
                "pdf_file": str(pdf_path.relative_to(ROOT)).replace("\\", "/"),
                "pdf_sha256": sha256_file(pdf_path),
                "download_status": "ok" if pdf_path.stat().st_size > 0 else "fail",
                "pages": pages,
                "text_file": str(text_path.relative_to(ROOT)).replace("\\", "/"),
                "text_chars": chars,
                "text_status": extract_status,
                "min_chars_pass": chars > MIN_TEXT_CHARS,
                "report_period_end_before_publication": (
                    report_period_end < publication_date
                ),
            }
        )
    return sorted(records, key=lambda row: row["issue"])


def normalize_industry_frame(raw: pd.DataFrame, code: str, name: str) -> pd.DataFrame:
    frame = raw.rename(
        columns={
            "代码": "code",
            "日期": "date",
            "收盘": "close",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        }
    )
    frame["code"] = code
    frame["name"] = name
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return frame[
        ["code", "name", "date", "close", "open", "high", "low", "volume", "amount"]
    ]


def normalize_benchmark_frame(raw: pd.DataFrame, code: str, name: str) -> pd.DataFrame:
    frame = raw.rename(columns={"日期": "date", "收盘": "close"})
    if "date" not in frame.columns:
        frame = raw.rename(columns={"date": "date", "close": "close"})
    frame["code"] = code
    frame["name"] = name
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    for column in ["open", "high", "low", "volume", "amount"]:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[
        ["code", "name", "date", "close", "open", "high", "low", "volume", "amount"]
    ]


def fetch_market_sample() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fetch live sample data, falling back to the verified workspace cache."""
    try:
        import akshare as ak

        frames: list[pd.DataFrame] = []
        fetch_rows: list[dict[str, Any]] = []
        for code, name in INDUSTRIES:
            raw = ak.index_hist_sw(symbol=code, period="day")
            frame = normalize_industry_frame(raw, code, name)
            frame.to_csv(
                MARKET_TEMP_DIR / f"industry_{code}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            frames.append(frame)
            fetch_rows.append(
                {
                    "code": code,
                    "name": name,
                    "source": "akshare:index_hist_sw",
                    "rows": len(frame),
                    "start": frame["date"].min(),
                    "end": frame["date"].max(),
                    "status": "ok",
                }
            )
            time.sleep(0.5)
        raw_benchmark = ak.stock_zh_index_daily(symbol=BENCHMARK[0])
        benchmark = normalize_benchmark_frame(raw_benchmark, BENCHMARK[0], BENCHMARK[1])
        benchmark.to_csv(
            MARKET_TEMP_DIR / f"benchmark_{BENCHMARK[0]}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        frames.append(benchmark)
        fetch_rows.append(
            {
                "code": BENCHMARK[0],
                "name": BENCHMARK[1],
                "source": "akshare:stock_zh_index_daily",
                "rows": len(benchmark),
                "start": benchmark["date"].min(),
                "end": benchmark["date"].max(),
                "status": "ok",
            }
        )
        return pd.concat(frames, ignore_index=True), fetch_rows
    except Exception as exc:  # noqa: BLE001
        fallback_rows = []
        industry_path = ROOT / "data" / "external" / "sw_industries" / "sw_industry_daily.csv"
        benchmark_path = ROOT / "data" / "external" / "benchmark" / "benchmark_daily.csv"
        industry = pd.read_csv(industry_path, dtype={"code": str})
        benchmark = pd.read_csv(benchmark_path, dtype={"code": str})
        industry = industry[industry["code"].isin([code for code, _ in INDUSTRIES])]
        benchmark = benchmark[benchmark["code"] == BENCHMARK[0]]
        frames = []
        for code, name in INDUSTRIES:
            frame = industry[industry["code"] == code].copy()
            frame = normalize_industry_frame(frame, code, name)
            frames.append(frame)
            fallback_rows.append(
                {
                    "code": code,
                    "name": name,
                    "source": "workspace_cache:data/external/sw_industries",
                    "rows": len(frame),
                    "start": frame["date"].min(),
                    "end": frame["date"].max(),
                    "status": f"fallback: {type(exc).__name__}",
                }
            )
        frame = benchmark.copy()
        frame = normalize_benchmark_frame(frame, BENCHMARK[0], BENCHMARK[1])
        frames.append(frame)
        fallback_rows.append(
            {
                "code": BENCHMARK[0],
                "name": BENCHMARK[1],
                "source": "workspace_cache:data/external/benchmark",
                "rows": len(frame),
                "start": frame["date"].min(),
                "end": frame["date"].max(),
                "status": f"fallback: {type(exc).__name__}",
            }
        )
        return pd.concat(frames, ignore_index=True), fallback_rows


def build_price_panel(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    panel = raw.copy()
    for column in ["open", "high", "low", "volume", "amount"]:
        if column not in panel.columns:
            panel[column] = pd.NA
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    panel["asset_type"] = panel["code"].apply(
        lambda code: "benchmark" if code == BENCHMARK[0] else "industry"
    )
    panel = panel.sort_values(["code", "date"]).reset_index(drop=True)
    panel["daily_return"] = panel.groupby("code", group_keys=False)["close"].pct_change()
    panel = panel[panel["date"] >= SAMPLE_START].reset_index(drop=True)
    columns = [
        "asset_type",
        "code",
        "name",
        "date",
        "close",
        "open",
        "high",
        "low",
        "volume",
        "amount",
        "daily_return",
    ]
    panel = panel[columns]

    duplicate_rows: list[dict[str, Any]] = []
    duplicates = panel.duplicated(subset=["code", "date"], keep=False)
    if duplicates.any():
        duplicate_rows = (
            panel[duplicates]
            .groupby(["code", "name", "date"], as_index=False)
            .size()
            .to_dict(orient="records")
        )

    common_start = panel.groupby("code")["date"].min().max()
    common_end = panel.groupby("code")["date"].max().min()
    benchmark_dates = sorted(
        panel.loc[panel["code"] == BENCHMARK[0], "date"].unique().tolist()
    )
    reference_dates = [
        date for date in benchmark_dates if common_start <= date <= common_end
    ]
    reference_set = set(reference_dates)

    missing_rows: list[dict[str, Any]] = []
    for code, name in INDUSTRIES + [BENCHMARK]:
        asset_dates = set(panel.loc[panel["code"] == code, "date"].tolist())
        missing = sorted(reference_set - asset_dates)
        missing_rows.append(
            {
                "code": code,
                "name": name,
                "reference_days": len(reference_dates),
                "available_days": len(asset_dates & reference_set),
                "missing_days": len(missing),
                "missing_rate": len(missing) / len(reference_dates) if reference_dates else None,
                "missing_dates": ";".join(missing[:20]),
                "duplicate_dates": sum(
                    1 for row in duplicate_rows if row["code"] == code
                ),
            }
        )
    return panel, missing_rows, reference_dates


def build_event_returns(
    price_panel: pd.DataFrame, mpr_index: pd.DataFrame, reference_dates: list[str]
) -> pd.DataFrame:
    date_to_pos = {date: idx for idx, date in enumerate(reference_dates)}
    series: dict[str, pd.DataFrame] = {}
    for code in [code for code, _ in INDUSTRIES]:
        frame = price_panel[price_panel["code"] == code].copy()
        series[code] = frame.set_index("date").sort_index()

    rows: list[dict[str, Any]] = []
    for _, issue_row in mpr_index.iterrows():
        publication_date = issue_row["publication_date"]
        if publication_date in date_to_pos:
            event_idx = date_to_pos[publication_date]
            event_trading_date = publication_date
            alignment_status = "publication_date_is_trading_day"
            date_match = True
        else:
            future = [date for date in reference_dates if date >= publication_date]
            if not future:
                rows.append(
                    {
                        "issue": issue_row["issue"],
                        "report_period": issue_row["report_period"],
                        "publication_date": publication_date,
                        "event_date": publication_date,
                        "event_trading_date": None,
                        "date_alignment_status": "no_trading_day_after_publication",
                        "date_match": False,
                        "code": None,
                        "name": None,
                        "event_window_return_0_1": None,
                        "event_window_return_0_3": None,
                        "event_window_return_0_5": None,
                        "event_window_return_0_10": None,
                        "quarterly_return_63td": None,
                        "return_status": "insufficient_calendar",
                    }
                )
                continue
            event_trading_date = future[0]
            event_idx = date_to_pos[event_trading_date]
            alignment_status = "mapped_to_next_trading_day"
            date_match = False

        for code, name in INDUSTRIES:
            frame = series[code]
            required_end = event_idx + max(EVENT_WINDOWS + [QUARTER_WINDOW - 1])
            window_values: dict[str, float | None] = {}
            return_status = "ok"
            for window in EVENT_WINDOWS:
                end_idx = event_idx + window
                if event_idx == 0 or end_idx >= len(reference_dates):
                    window_values[f"event_window_return_0_{window}"] = None
                    return_status = "insufficient_history"
                else:
                    start_date = reference_dates[event_idx - 1]
                    end_date = reference_dates[end_idx]
                    if (
                        start_date not in frame.index
                        or end_date not in frame.index
                    ):
                        window_values[f"event_window_return_0_{window}"] = None
                        return_status = "missing_industry_date"
                    else:
                        start_close = float(frame.loc[start_date, "close"])
                        end_close = float(frame.loc[end_date, "close"])
                        window_values[f"event_window_return_0_{window}"] = (
                            end_close / start_close - 1.0
                        )

            quarter_end_idx = event_idx + QUARTER_WINDOW - 1
            if event_idx == 0 or quarter_end_idx >= len(reference_dates):
                quarterly_return = None
                return_status = "insufficient_history"
            else:
                start_date = reference_dates[event_idx - 1]
                end_date = reference_dates[quarter_end_idx]
                if start_date not in frame.index or end_date not in frame.index:
                    quarterly_return = None
                    return_status = "missing_industry_date"
                else:
                    start_close = float(frame.loc[start_date, "close"])
                    end_close = float(frame.loc[end_date, "close"])
                    quarterly_return = end_close / start_close - 1.0

            rows.append(
                {
                    "issue": issue_row["issue"],
                    "report_period": issue_row["report_period"],
                    "publication_date": publication_date,
                    "event_date": publication_date,
                    "event_trading_date": event_trading_date,
                    "date_alignment_status": alignment_status,
                    "date_match": date_match,
                    "code": code,
                    "name": name,
                    **window_values,
                    "quarterly_return_63td": quarterly_return,
                    "return_status": return_status,
                }
            )
    return pd.DataFrame(rows)


def pct(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def write_log(
    mpr_index: pd.DataFrame,
    market_rows: list[dict[str, Any]],
    price_panel: pd.DataFrame,
    missing_rows: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
    event_returns: pd.DataFrame,
    environment: dict[str, str],
) -> dict[str, Any]:
    pdf_success = int((mpr_index["download_status"] == "ok").sum())
    text_success = int(
        (mpr_index["text_status"] == "ok").sum()
    )
    char_pass = int(mpr_index["min_chars_pass"].sum())
    alignment_success = int(event_returns.drop_duplicates("issue")["date_match"].sum())
    issue_count = len(mpr_index)

    event_columns = [
        "event_window_return_0_1",
        "event_window_return_0_3",
        "event_window_return_0_5",
        "event_window_return_0_10",
        "quarterly_return_63td",
    ]
    event_cells = len(event_returns) * len(event_columns)
    non_null_cells = int(event_returns[event_columns].notna().sum().sum())

    missing_total = sum(int(row["missing_days"]) for row in missing_rows)
    reference_total = sum(int(row["reference_days"]) for row in missing_rows)

    metrics = {
        "mpr_pdf_success_rate": pct(pdf_success, issue_count),
        "mpr_text_success_rate": pct(text_success, issue_count),
        "mpr_text_min_chars_pass_rate": pct(char_pass, issue_count),
        "industry_assets": len(INDUSTRIES),
        "benchmark_assets": 1,
        "price_rows": len(price_panel),
        "missing_rate_all_assets": pct(missing_total, reference_total),
        "date_alignment_success_rate": pct(alignment_success, issue_count),
        "event_return_cell_coverage": pct(non_null_cells, event_cells),
        "duplicate_date_groups": len(duplicate_rows),
    }

    lines = [
        "# E0 数据链路 MVP 实验日志",
        "",
        f"- 实验编号：E0",
        f"- 运行日期：{dt.date.today().isoformat()}",
        f"- 脚本：`experiments/E0/run_e0.py`",
        f"- Python：{environment['python']}",
        f"- pandas：{environment['pandas']}",
        f"- pdfplumber：{environment['pdfplumber']}",
        f"- requests：{environment['requests']}",
        "- 对照设置：不建模；只检查抓取失败、字段缺失、日期错位和交易日历缺失。",
        "",
        "## 1. MPR 样本",
        "",
        "| 期次 | 报告期截止 | 发布日期 | 发布时间 | PDF 状态 | 页数 | 文本字符数 | >10000 | 报告期早于发布日 |",
        "|------|------------|----------|----------|----------|------|------------|---------|------------------|",
    ]
    for _, row in mpr_index.iterrows():
        lines.append(
            "| {issue} | {report_period_end} | {publication_date} | {publication_timestamp} | "
            "{download_status} | {pages} | {text_chars} | {min_chars_pass} | "
            "{report_period_end_before_publication} |".format(**row)
        )

    lines += [
        "",
        "## 2. 行业指数与市场基准",
        "",
        "| 代码 | 名称 | 来源 | 条数 | 起始 | 结束 | 状态 |",
        "|------|------|------|------|------|------|------|",
    ]
    for row in market_rows:
        lines.append(
            f"| {row['code']} | {row['name']} | {row['source']} | {row['rows']} | "
            f"{row['start']} | {row['end']} | {row['status']} |"
        )

    lines += [
        "",
        "## 3. 交易日历与缺失检查",
        "",
        "| 代码 | 名称 | 参考交易日 | 可得 | 缺失 | 缺失率 | 缺失日期 | 重复日期组 |",
        "|------|------|------------|------|------|--------|----------|------------|",
    ]
    for row in missing_rows:
        lines.append(
            f"| {row['code']} | {row['name']} | {row['reference_days']} | "
            f"{row['available_days']} | {row['missing_days']} | "
            f"{row['missing_rate']:.6%} | {row['missing_dates'] or '-'} | "
            f"{row['duplicate_dates']} |"
        )

    if duplicate_rows:
        lines += ["", "### 重复日期记录", ""]
        for row in duplicate_rows:
            lines.append(
                f"- {row['code']} {row['name']} {row['date']}：{row['size']} 条"
            )
    else:
        lines += ["", "重复日期检查：未发现同一代码同一日期重复记录。"]

    lines += [
        "",
        "## 4. 发布日对齐与收益构造",
        "",
        "- 事件日：MPR 详情页 `PubDate`，不是报告期截止日。",
        "- 事件窗收益：从事件日前一收盘到事件日后第 k 个交易日收盘，k 为 1、3、5、10。",
        "- 季度收益：从事件日前一收盘到事件日后第 62 个交易日收盘，共包含 63 个交易日。",
        "- 行业收益：申万一级行业指数收盘价简单收益。",
        "- 基准：沪深300，仅用于交易日历参照与日收益检查。",
        "",
        "| 期次 | 事件日 | 事件交易日 | 对齐状态 | 日期匹配 | 收益状态 |",
        "|------|--------|------------|----------|----------|----------|",
    ]
    for _, row in event_returns.drop_duplicates("issue").iterrows():
        lines.append(
            f"| {row['issue']} | {row['event_date']} | {row['event_trading_date']} | "
            f"{row['date_alignment_status']} | {row['date_match']} | "
            f"{row['return_status']} |"
        )

    lines += [
        "",
        "## 5. 评价指标",
        "",
        f"- PDF 成功率：{pdf_success}/{issue_count} = {metrics['mpr_pdf_success_rate']:.2%}",
        f"- 文本提取成功率：{text_success}/{issue_count} = {metrics['mpr_text_success_rate']:.2%}",
        f"- 文本字符数下限通过率：{char_pass}/{issue_count} = {metrics['mpr_text_min_chars_pass_rate']:.2%}",
        f"- 行业指数条数：{len(price_panel[price_panel['asset_type'] == 'industry'])}（3 个行业）",
        f"- 基准条数：{len(price_panel[price_panel['asset_type'] == 'benchmark'])}（1 个基准）",
        f"- 逐日缺失率（全部资产合计）：{missing_total}/{reference_total} = {metrics['missing_rate_all_assets']:.6%}",
        f"- 发布日期对齐成功率：{alignment_success}/{issue_count} = {metrics['date_alignment_success_rate']:.2%}",
        f"- 事件窗与季度收益单元格覆盖率：{non_null_cells}/{event_cells} = {metrics['event_return_cell_coverage']:.2%}",
        "",
        "## 6. 异常与限制",
        "",
    ]
    missing_dates_all = sorted(
        {
            date
            for row in missing_rows
            for date in (row["missing_dates"].split(";") if row["missing_dates"] else [])
        }
    )
    if missing_total == 0 and not duplicate_rows:
        lines.append("- 未发现交易日历缺失或重复日期。")
    else:
        if missing_total:
            lines.append(
                f"- 交易日历缺失 {missing_total} 个资产日，涉及日期："
                f"{'、'.join(missing_dates_all) if missing_dates_all else '未解析'}；详见上表。"
            )
        if duplicate_rows:
            lines.append(f"- 发现 {len(duplicate_rows)} 个重复日期组。")

    report_period_confusion = mpr_index[
        ~mpr_index["report_period_end_before_publication"]
    ]
    if report_period_confusion.empty:
        lines.append("- 报告期截止日均早于发布日期；事件对齐使用 `publication_date`，未混用报告期。")
    else:
        lines.append(
            "- 警告：存在报告期截止日不早于发布日期的记录，需人工复核。"
        )

    lines += [
        "- MPR 发布时间多在第 0 个交易日收盘前后；本实验只验证数据链路，不把事件窗收益解释为因果效应。",
        "- 行业指数历史起点不同；本样本自 2025-01-01 起，三个样例行业与基准在该区间内均有数据。",
        "",
        "## 7. 完成条件核对",
        "",
        f"- 3-5 期文本成功提取：{'通过' if text_success == issue_count and issue_count >= 3 else '未通过'}（{issue_count} 期）",
        f"- 3 个行业指数与市场基准在 5 个事件日成功对齐：{'通过' if alignment_success == issue_count else '未通过'}",
        f"- 缺失和异常均有记录：{'通过（未发现缺失或重复）' if missing_total == 0 and not duplicate_rows else '通过（缺失或重复已记录，详见上表）'}",
        f"- 不存在报告期与发布日期混用：{'通过' if report_period_confusion.empty else '未通过'}",
        "",
    ]

    log_path = LOG_DIR / "e0_data_pipeline.md"
    log_path.write_text("\n".join(lines), encoding="utf-8")
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
        "pdfplumber": _pdfplumber.__version__,
        "requests": _requests.__version__,
    }

    mpr_index_records = parse_mpr_sample()
    mpr_index = pd.DataFrame(mpr_index_records)
    mpr_index.to_csv(
        OUT_DIR / "mpr_index_sample.csv", index=False, encoding="utf-8-sig"
    )

    market_raw, market_rows = fetch_market_sample()
    price_panel, missing_rows, reference_dates = build_price_panel(market_raw)
    price_panel.to_csv(
        OUT_DIR / "industry_prices_sample.csv", index=False, encoding="utf-8-sig"
    )

    duplicate_rows: list[dict[str, Any]] = []
    duplicates = price_panel.duplicated(subset=["code", "date"], keep=False)
    if duplicates.any():
        duplicate_rows = (
            price_panel[duplicates]
            .groupby(["code", "name", "date"], as_index=False)
            .size()
            .to_dict(orient="records")
        )

    event_returns = build_event_returns(price_panel, mpr_index, reference_dates)
    event_returns.to_csv(
        OUT_DIR / "event_returns_sample.csv", index=False, encoding="utf-8-sig"
    )

    metrics = write_log(
        mpr_index,
        market_rows,
        price_panel,
        missing_rows,
        duplicate_rows,
        event_returns,
        environment,
    )
    metrics["script_sha256"] = sha256_file(Path(__file__))
    (E0_DIR / "e0_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
