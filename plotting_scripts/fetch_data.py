# -*- coding: utf-8 -*-
"""
抓取论文图表所需的真实公开数据。

数据来源：
  - 申万一级行业指数日行情：akshare index_hist_sw
  - 市场基准（沪深300 / 中证全指）：akshare stock_zh_index_daily_em

输出：
  - data/external/sw_industries/sw_industry_daily.csv   （长表：行业+日期+OHLCV）
  - data/external/benchmark/benchmark_daily.csv         （长表：基准+日期+收盘）
  - data/external/sw_industries/sw_industry_codes.csv   （行业代码与名称）
  - logs/data_fetch_log.md                              （抓取台账）

说明：本脚本只获取公开行情数据，不包含任何 MPR 文本或词典数据。
"""

import os
import time
import datetime

import akshare as ak
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "external")
IND_DIR = os.path.join(DATA_DIR, "sw_industries")
BM_DIR = os.path.join(DATA_DIR, "benchmark")
LOG_DIR = os.path.join(ROOT, "logs")

INDUSTRIES = [
    ("801010", "农林牧渔"), ("801030", "基础化工"), ("801040", "钢铁"),
    ("801050", "有色金属"), ("801080", "电子"), ("801110", "家用电器"),
    ("801120", "食品饮料"), ("801130", "纺织服饰"), ("801140", "轻工制造"),
    ("801150", "医药生物"), ("801160", "公用事业"), ("801170", "交通运输"),
    ("801180", "房地产"), ("801200", "商贸零售"), ("801210", "社会服务"),
    ("801230", "综合"), ("801710", "建筑材料"), ("801720", "建筑装饰"),
    ("801730", "电力设备"), ("801740", "国防军工"), ("801750", "计算机"),
    ("801760", "传媒"), ("801770", "通信"), ("801780", "银行"),
    ("801790", "非银金融"), ("801880", "汽车"), ("801890", "机械设备"),
    ("801950", "煤炭"), ("801960", "石油石化"), ("801970", "环保"),
    ("801980", "美容护理"),
]

BENCHMARKS = [("sh000300", "沪深300"), ("sh000985", "中证全指")]


def fetch_industries():
    os.makedirs(IND_DIR, exist_ok=True)
    frames = []
    rows = []
    for code, name in INDUSTRIES:
        ok = False
        for attempt in range(3):
            try:
                df = ak.index_hist_sw(symbol=code, period="day")
                df = df.rename(columns={"代码": "code", "日期": "date"})
                df["code"] = code
                df["name"] = name
                df = df[["code", "name", "date", "收盘", "开盘", "最高", "最低", "成交量", "成交额"]]
                df.to_csv(os.path.join(IND_DIR, f"sw_{code}_{name}.csv"), index=False, encoding="utf-8-sig")
                frames.append(df)
                rows.append((code, name, len(df), str(df["date"].min()), str(df["date"].max()), "ok"))
                ok = True
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    rows.append((code, name, 0, "", "", f"fail: {e}"))
                else:
                    time.sleep(2)
        time.sleep(0.6)
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(os.path.join(IND_DIR, "sw_industry_daily.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(rows, columns=["code", "name", "rows", "start", "end", "status"]).to_csv(
        os.path.join(IND_DIR, "fetch_status.csv"), index=False, encoding="utf-8-sig"
    )
    return rows


def fetch_benchmarks():
    os.makedirs(BM_DIR, exist_ok=True)
    frames = []
    rows = []
    for code, name in BENCHMARKS:
        try:
            df = ak.stock_zh_index_daily_em(symbol=code)
            df = df.rename(columns={"date": "date", "close": "close"})
            df["code"] = code
            df["name"] = name
            df = df[["code", "name", "date", "close"]]
            df.to_csv(os.path.join(BM_DIR, f"{code}_{name}.csv"), index=False, encoding="utf-8-sig")
            frames.append(df)
            rows.append((code, name, len(df), str(df["date"].min()), str(df["date"].max()), "ok"))
        except Exception as e:  # noqa: BLE001
            rows.append((code, name, 0, "", "", f"fail: {e}"))
        time.sleep(0.6)
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(os.path.join(BM_DIR, "benchmark_daily.csv"), index=False, encoding="utf-8-sig")
    return rows


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    t0 = datetime.datetime.now()
    ind_rows = fetch_industries()
    bm_rows = fetch_benchmarks()
    t1 = datetime.datetime.now()
    log = [
        "# 数据抓取台账",
        "",
        f"- 抓取时间：{t0:%Y-%m-%d %H:%M:%S}",
        f"- 耗时：{(t1 - t0).total_seconds():.1f} 秒",
        f"- 数据源：akshare {ak.__version__}",
        "",
        "## 申万一级行业指数",
        "",
        "| 代码 | 名称 | 行数 | 起始 | 结束 | 状态 |",
        "|------|------|------|------|------|------|",
    ]
    for r in ind_rows:
        log.append("| " + " | ".join(str(x) for x in r) + " |")
    log += [
        "",
        "## 市场基准",
        "",
        "| 代码 | 名称 | 行数 | 起始 | 结束 | 状态 |",
        "|------|------|------|------|------|------|",
    ]
    for r in bm_rows:
        log.append("| " + " | ".join(str(x) for x in r) + " |")
    with open(os.path.join(LOG_DIR, "data_fetch_log.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    ok = sum(1 for r in ind_rows if r[5] == "ok")
    print(f"industries ok: {ok}/{len(ind_rows)}; benchmarks ok: {sum(1 for r in bm_rows if r[5] == 'ok')}/{len(bm_rows)}")


if __name__ == "__main__":
    main()
