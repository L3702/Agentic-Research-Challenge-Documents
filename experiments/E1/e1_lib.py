# -*- coding: utf-8 -*-
"""Shared helpers for the E1 sentiment experiment."""

from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import re
import time
from pathlib import Path
from typing import Any

import jieba
import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[2]
E1_DIR = ROOT / "experiments" / "E1"
DICT_DIR = E1_DIR / "dictionaries"
VALIDATION_DIR = E1_DIR / "validation"
LOG_DIR = E1_DIR / "logs"
TEMP_DIR = ROOT / "_temp"
MPR_TEMP_DIR = TEMP_DIR / "e1_mpr"
PDF_TEMP_DIR = MPR_TEMP_DIR / "pdf"
TEXT_TEMP_DIR = MPR_TEMP_DIR / "text"
DETAIL_TEMP_DIR = MPR_TEMP_DIR / "detail"
DICT_TEMP_DIR = TEMP_DIR / "e1_dictionaries"
E0_SAMPLE_ISSUES = ["2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q2"]

MPR_ARCHIVE_URL = (
    "https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/index.html"
)
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}
MIN_TEXT_CHARS = 10_000
SEED = 20261118
VALIDATION_FRACTION = 0.10
PILOT_N = 5
DICTIONARY_NAMES = ["ntusd_simplified", "dlut", "mpr_domain"]


def ensure_dirs() -> None:
    for path in [
        E1_DIR,
        DICT_DIR,
        VALIDATION_DIR,
        LOG_DIR,
        MPR_TEMP_DIR,
        PDF_TEMP_DIR,
        TEXT_TEMP_DIR,
        DETAIL_TEMP_DIR,
        DICT_TEMP_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def make_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def fetch_html(session: requests.Session, url: str, cache_path: Path) -> str:
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_text(encoding="utf-8")
    response = session.get(url, headers=REQUEST_HEADERS, timeout=45)
    response.raise_for_status()
    response.encoding = "utf-8"
    cache_path.write_text(response.text, encoding="utf-8")
    time.sleep(0.25)
    return response.text


def fetch_binary(session: requests.Session, url: str, cache_path: Path) -> None:
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return
    response = session.get(url, headers=REQUEST_HEADERS, timeout=180)
    response.raise_for_status()
    cache_path.write_bytes(response.content)
    time.sleep(0.25)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def absolute_url(href: str, base: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return "https://www.pbc.gov.cn" + href
    return base.rsplit("/", 1)[0] + "/" + href


def parse_archive_years(html: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    years: dict[str, str] = {}
    for anchor in soup.find_all("a"):
        text = anchor.get_text(" ", strip=True)
        href = anchor.get("href")
        if not href:
            continue
        match = re.search(r"([0-9]{4})年货币政策执行报告", text)
        if match and "/125957/" in href:
            years[match.group(1)] = absolute_url(href, MPR_ARCHIVE_URL)
    return dict(sorted(years.items()))


def parse_year_issues(html: str, year_url: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    quarter_map = {"一": 1, "二": 2, "三": 3, "四": 4}
    issues: dict[str, str] = {}
    for anchor in soup.find_all("a"):
        text = anchor.get_text(" ", strip=True)
        href = anchor.get("href")
        if not href:
            continue
        match = re.search(
            r"([0-9]{4})年\s*第([一二三四])季度\s*中国货币政策执行报告", text
        )
        if match:
            issue = f"{match.group(1)}Q{quarter_map[match.group(2)]}"
            issues[issue] = absolute_url(href, year_url)
    return dict(sorted(issues.items()))


def parse_detail_page(html: str, detail_url: str) -> tuple[str, str, str]:
    soup = BeautifulSoup(html, "html.parser")
    meta = soup.find("meta", attrs={"name": "PubDate"})
    if not meta or not meta.get("content"):
        raise RuntimeError(f"Missing PubDate in detail page: {detail_url}")
    publication_date = str(meta["content"])[:10]
    timestamp_span = soup.find("span", id="shijian")
    publication_timestamp = (
        timestamp_span.get_text(" ", strip=True) if timestamp_span else ""
    )
    pdf_anchor = next(
        (
            anchor
            for anchor in soup.find_all("a")
            if anchor.get("href") and ".pdf" in anchor["href"].lower()
        ),
        None,
    )
    if not pdf_anchor:
        raise RuntimeError(f"Missing PDF link in detail page: {detail_url}")
    pdf_url = absolute_url(pdf_anchor["href"], detail_url)
    return publication_date, publication_timestamp, pdf_url


def extract_pdf_text(pdf_path: Path) -> tuple[str, int, str]:
    pages = 0
    text = ""
    status = "ok"
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = len(pdf.pages)
            text = "\n".join((page.extract_text() or "") for page in pdf.pages)
        text = text.replace("\x00", "")
    except Exception as exc:  # noqa: BLE001
        status = f"fail: {type(exc).__name__}: {exc}"
    return text, pages, status


def fetch_mpr_corpus() -> pd.DataFrame:
    session = make_session()
    archive_html = fetch_html(
        session, MPR_ARCHIVE_URL, MPR_TEMP_DIR / "mpr_archive.html"
    )
    years = parse_archive_years(archive_html)
    all_issues: dict[str, str] = {}
    for year, year_url in years.items():
        year_html = fetch_html(
            session, year_url, MPR_TEMP_DIR / f"year_{year}.html"
        )
        all_issues.update(parse_year_issues(year_html, year_url))

    records: list[dict[str, Any]] = []
    for issue, detail_url in all_issues.items():
        detail_cache = DETAIL_TEMP_DIR / f"{issue}.html"
        detail_html = fetch_html(session, detail_url, detail_cache)
        publication_date = ""
        publication_timestamp = ""
        pdf_url = ""
        download_status = "pending"
        pdf_path = PDF_TEMP_DIR / f"{issue}.pdf"
        text_path = TEXT_TEMP_DIR / f"{issue}.txt"
        pages = 0
        text_chars = 0
        text_status = "pending"
        try:
            publication_date, publication_timestamp, pdf_url = parse_detail_page(
                detail_html, detail_url
            )
            fetch_binary(session, pdf_url, pdf_path)
            download_status = "ok" if pdf_path.stat().st_size > 0 else "fail"
            if text_path.exists() and text_path.stat().st_size > 0:
                text = text_path.read_text(encoding="utf-8")
                text_chars = len(text)
                text_status = "cached"
                with pdfplumber.open(pdf_path) as pdf:
                    pages = len(pdf.pages)
            else:
                text, pages, text_status = extract_pdf_text(pdf_path)
                text_path.write_text(text, encoding="utf-8")
                text_chars = len(text)
        except Exception as exc:  # noqa: BLE001
            download_status = f"fail: {type(exc).__name__}: {exc}"
            text_status = f"fail: {type(exc).__name__}: {exc}"

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
                "publication_date": publication_date,
                "publication_timestamp": publication_timestamp,
                "detail_url": detail_url,
                "pdf_url": pdf_url,
                "pdf_file": str(pdf_path.relative_to(ROOT)).replace("\\", "/"),
                "pdf_sha256": sha256_file(pdf_path) if pdf_path.exists() else "",
                "download_status": download_status,
                "pages": pages,
                "text_file": str(text_path.relative_to(ROOT)).replace("\\", "/"),
                "text_chars": text_chars,
                "text_status": text_status,
                "min_chars_pass": text_chars > MIN_TEXT_CHARS,
            }
        )
        print(
            f"fetched {issue}: pages={pages}, chars={text_chars}, "
            f"status={text_status}",
            flush=True,
        )
    return pd.DataFrame(records).sort_values("issue").reset_index(drop=True)


def load_ntusd() -> tuple[dict[str, int], dict[str, int], dict[str, Any]]:
    positive_path = DICT_TEMP_DIR / "ntusd_simplified_positive.txt"
    negative_path = DICT_TEMP_DIR / "ntusd_simplified_negative.txt"
    positive_raw = positive_path.read_bytes().decode("utf-16").splitlines()
    negative_raw = negative_path.read_bytes().decode("utf-16").splitlines()
    positive = {word.strip(): 1 for word in positive_raw if word.strip()}
    negative = {word.strip(): 1 for word in negative_raw if word.strip()}
    ambiguous = sorted(set(positive) & set(negative))
    ambiguous_set = set(ambiguous)
    positive = {
        word: weight
        for word, weight in positive.items()
        if word not in ambiguous_set
    }
    negative = {
        word: weight
        for word, weight in negative.items()
        if word not in ambiguous_set
    }
    metadata = {
        "source": "NTUSD 简体中文情感词典（ntunlplab/NTUSD，MIT；简体文件来自公开汇总仓库）",
        "positive_raw": len(positive_raw),
        "negative_raw": len(negative_raw),
        "ambiguous_removed": len(ambiguous),
        "positive": len(positive),
        "negative": len(negative),
    }
    return positive, negative, metadata


def load_dlut() -> tuple[dict[str, int], dict[str, int], dict[str, Any]]:
    path = DICT_TEMP_DIR / "dlut_emotion.csv"
    frame = pd.read_csv(
        path,
        encoding="utf-8",
        on_bad_lines="skip",
        engine="python",
    )
    frame.columns = [column.strip() for column in frame.columns]
    word_col = frame.columns[0]
    category_col = "情感分类"
    polarity_col = "极性"
    intensity_col = "强度"
    frame = frame[[word_col, category_col, polarity_col, intensity_col]].copy()
    frame[word_col] = frame[word_col].astype(str).str.strip()
    frame = frame[frame[word_col].ne("") & frame[word_col].ne("nan")]
    frame[polarity_col] = pd.to_numeric(frame[polarity_col], errors="coerce")
    frame[intensity_col] = pd.to_numeric(frame[intensity_col], errors="coerce")
    grouped = frame.groupby(word_col, dropna=False).agg(
        polarities=(
            polarity_col,
            lambda values: set(values.dropna().astype(int)),
        ),
        max_intensity=(intensity_col, "max"),
    )
    positive: dict[str, int] = {}
    negative: dict[str, int] = {}
    ambiguous: list[str] = []
    neutral: list[str] = []
    for word, row in grouped.iterrows():
        polarities = row["polarities"]
        intensity = int(row["max_intensity"] if pd.notna(row["max_intensity"]) else 1)
        if polarities == {1}:
            positive[word] = intensity
        elif polarities == {2}:
            negative[word] = intensity
        elif polarities == {0} or not polarities:
            neutral.append(word)
        elif 1 in polarities and 2 in polarities:
            ambiguous.append(word)
        elif 1 in polarities:
            positive[word] = intensity
        elif 2 in polarities:
            negative[word] = intensity
    metadata = {
        "source": "大连理工大学情感词汇本体（学术研究许可）",
        "raw_rows": len(frame),
        "unique_words": len(grouped),
        "ambiguous_removed": len(ambiguous),
        "neutral_removed": len(neutral),
        "positive": len(positive),
        "negative": len(negative),
    }
    return positive, negative, metadata


def load_mpr_domain() -> tuple[dict[str, int], dict[str, int], dict[str, Any]]:
    path = DICT_DIR / "mpr_domain_dictionary.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["word"] = frame["word"].astype(str).str.strip()
    frame = frame[frame["word"].ne("")]
    positive = {
        row.word: int(row.weight)
        for row in frame.itertuples()
        if row.polarity == "positive"
    }
    negative = {
        row.word: int(row.weight)
        for row in frame.itertuples()
        if row.polarity == "negative"
    }
    ambiguous = sorted(set(positive) & set(negative))
    ambiguous_set = set(ambiguous)
    positive = {
        word: weight
        for word, weight in positive.items()
        if word not in ambiguous_set
    }
    negative = {
        word: weight
        for word, weight in negative.items()
        if word not in ambiguous_set
    }
    metadata = {
        "source": "自制 MPR 宏观金融领域词典 v1",
        "raw_rows": len(frame),
        "ambiguous_removed": len(ambiguous),
        "positive": len(positive),
        "negative": len(negative),
    }
    return positive, negative, metadata


def load_dictionaries() -> dict[str, dict[str, Any]]:
    loaders = {
        "ntusd_simplified": load_ntusd,
        "dlut": load_dlut,
        "mpr_domain": load_mpr_domain,
    }
    dictionaries: dict[str, dict[str, Any]] = {}
    for name in DICTIONARY_NAMES:
        positive, negative, metadata = loaders[name]()
        dictionaries[name] = {
            "positive": positive,
            "negative": negative,
            "metadata": metadata,
        }
    return dictionaries


def load_stopwords() -> set[str]:
    path = DICT_TEMP_DIR / "stopwords_zh.txt"
    stopwords = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    stopwords.update(
        {
            "中国人民银行",
            "货币政策执行报告",
            "中国货币政策执行报告",
            "货币政策",
            "执行报告",
            "第一部分",
            "第二部分",
            "第三部分",
            "第四部分",
            "第五部分",
            "目",
            "录",
        }
    )
    return stopwords


def add_dictionary_words_to_jieba(
    dictionaries: dict[str, dict[str, Any]],
) -> None:
    words: set[str] = set()
    for dictionary in dictionaries.values():
        words.update(dictionary["positive"])
        words.update(dictionary["negative"])
    for word in sorted(words):
        jieba.add_word(word, freq=1_000_000)


def tokenize_text(text: str, stopwords: set[str]) -> list[str]:
    tokens: list[str] = []
    for token in jieba.cut(text, cut_all=False):
        token = token.strip()
        if not token or token in stopwords:
            continue
        if not re.search(r"[一-鿿]", token):
            continue
        tokens.append(token)
    return tokens


def score_tokens(
    tokens: list[str], dictionary: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    positive = dictionary["positive"]
    negative = dictionary["negative"]
    token_counts: dict[str, int] = {}
    for token in tokens:
        token_counts[token] = token_counts.get(token, 0) + 1
    pos_counts = {
        word: token_counts[word] for word in positive if word in token_counts
    }
    neg_counts = {
        word: token_counts[word] for word in negative if word in token_counts
    }
    pos_count = sum(pos_counts.values())
    neg_count = sum(neg_counts.values())
    pos_weighted = sum(
        positive[word] * count for word, count in pos_counts.items()
    )
    neg_weighted = sum(
        negative[word] * count for word, count in neg_counts.items()
    )
    total_tokens = len(tokens)
    pos_neg_count = pos_count + neg_count
    sentiment_score = (
        (pos_count - neg_count) / pos_neg_count if pos_neg_count else 0.0
    )
    weighted_score = (
        (pos_weighted - neg_weighted) / (pos_weighted + neg_weighted)
        if (pos_weighted + neg_weighted)
        else 0.0
    )
    top_positive = sorted(
        pos_counts.items(), key=lambda item: (-item[1], item[0])
    )[:10]
    top_negative = sorted(
        neg_counts.items(), key=lambda item: (-item[1], item[0])
    )[:10]
    return {
        "tokens": total_tokens,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "pos_weighted": pos_weighted,
        "neg_weighted": neg_weighted,
        "pos_unique_hits": len(pos_counts),
        "neg_unique_hits": len(neg_counts),
        "pos_rate": pos_count / total_tokens if total_tokens else 0.0,
        "neg_rate": neg_count / total_tokens if total_tokens else 0.0,
        "pos_coverage": pos_count / total_tokens if total_tokens else 0.0,
        "neg_coverage": neg_count / total_tokens if total_tokens else 0.0,
        "pos_neg_coverage": pos_neg_count / total_tokens if total_tokens else 0.0,
        "sentiment_score": sentiment_score,
        "weighted_score": weighted_score,
        "sentiment_score_per_1k": sentiment_score * 1000,
        "top_positive_terms": ";".join(
            f"{word}:{count}" for word, count in top_positive
        ),
        "top_negative_terms": ";".join(
            f"{word}:{count}" for word, count in top_negative
        ),
    }


def build_scores(
    full_index: pd.DataFrame,
    dictionaries: dict[str, dict[str, Any]],
    stopwords: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    unmatched_counts: dict[str, int] = {}
    dictionary_words = {
        word
        for dictionary in dictionaries.values()
        for word in list(dictionary["positive"]) + list(dictionary["negative"])
    }
    for _, record in full_index.iterrows():
        text_path = ROOT / record["text_file"]
        text = (
            text_path.read_text(encoding="utf-8")
            if text_path.exists()
            else ""
        )
        tokens = tokenize_text(text, stopwords)
        for token in tokens:
            if token not in dictionary_words:
                unmatched_counts[token] = unmatched_counts.get(token, 0) + 1
        for dictionary_name in DICTIONARY_NAMES:
            score = score_tokens(tokens, dictionaries[dictionary_name])
            rows.append(
                {
                    "issue": record["issue"],
                    "report_period": record["report_period"],
                    "publication_date": record["publication_date"],
                    "text_chars": record["text_chars"],
                    "text_status": record["text_status"],
                    "min_chars_pass": record["min_chars_pass"],
                    "dictionary": dictionary_name,
                    "dictionary_version": dictionaries[dictionary_name]["metadata"]["source"],
                    "dictionary_positive_size": len(
                        dictionaries[dictionary_name]["positive"]
                    ),
                    "dictionary_negative_size": len(
                        dictionaries[dictionary_name]["negative"]
                    ),
                    **score,
                }
            )
    unmatched = pd.DataFrame(
        [
            {"term": term, "count": count}
            for term, count in sorted(
                unmatched_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        columns=["term", "count"],
    )
    return pd.DataFrame(rows), unmatched
