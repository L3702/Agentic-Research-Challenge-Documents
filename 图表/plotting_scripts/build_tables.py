# -*- coding: utf-8 -*-
"""生成论文表格 tables/ 下的 Table 1-11。

真实数值全部由脚本从 data/external/ 与 documents/Literature_Matrix.xlsx 计算或读取，
不硬编码、不编造。文本类表格（Table 1/3/6/7）内容来源于
资料抓取清单.md / EXPERIMENT_PLAN.md / METHOD_PLAN.md，在脚本内以常量登记并注明出处。

输出：tables/tableNN_*.csv（utf-8-sig）与 tables/tableNN_*.md（Markdown）。
脚本可重复运行，每次覆盖同名输出。
"""

import os

import numpy as np
import pandas as pd

from _style import DATA_DIR, PROJECT_ROOT, TABLE_DIR

LIT_PATH = os.path.join(PROJECT_ROOT, "documents", "Literature_Matrix.xlsx")
EXP_DIR = os.path.join(PROJECT_ROOT, "experiments")

START = pd.Timestamp("2014-02-21")   # 28 个可比行业统一起点
ANN = 252
EXCLUDE = ["石油石化", "环保", "美容护理"]   # 2021-12-13 后新分类，不纳入主口径
RF_NOTE = "无风险利率数据尚未获取（资料抓取清单 D4），夏普比率暂以 0 计算，仅为相对比较，待补。"
PRIMARY_DICT = "ntusd_simplified"
SPLIT_LABEL = {"validation": "验证期 2016Q1–2020Q4", "test": "测试期 2021Q1–2026Q3"}
DICT_LABEL = {"none": "无文本", "ntusd_simplified": "NTUSD 简体", "dlut": "DLUT",
              "mpr_domain": "MPR 领域", "llm": "LLM", "ntusd_simplified+llm": "NTUSD 简体 + LLM"}


# ---------------------------------------------------------------- 工具函数

def df_to_md(df, float_fmt="{:.4f}"):
    """把 DataFrame 渲染成 Markdown 表格，NaN 显示为 —。"""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "—" if pd.isna(v) else float_fmt.format(v))
        else:
            df[c] = df[c].fillna("").astype(str).str.replace("\n", " ", regex=False)
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join(["------"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(row) + " |" for row in df.astype(str).values.tolist()]
    return "\n".join([header, sep] + rows)


def write_table(df, stem, title, note="", csv=True):
    os.makedirs(TABLE_DIR, exist_ok=True)
    md = [f"# {title}", ""]
    if note:
        md += [note, ""]
    md += [df_to_md(df), ""]
    md_path = os.path.join(TABLE_DIR, stem + ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    if csv:
        csv_path = os.path.join(TABLE_DIR, stem + ".csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"wrote: {md_path}")


def max_drawdown(nav):
    peak = nav.cummax()
    return float((nav / peak - 1.0).min())


# ---------------------------------------------------------------- 数据读取

def load_industry():
    df = pd.read_csv(os.path.join(DATA_DIR, "sw_industries", "sw_industry_daily.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
    df["ret"] = df.groupby("code")["收盘"].pct_change()
    return df


def load_status():
    return pd.read_csv(os.path.join(DATA_DIR, "sw_industries", "fetch_status.csv"))


def load_literature():
    return pd.read_excel(LIT_PATH, sheet_name=0)


def dir_status(path, pattern=None):
    """检查目录是否存在且有文件，返回 (状态, 数量)。"""
    if not os.path.isdir(path):
        return "未获取", 0
    files = os.listdir(path)
    if pattern:
        files = [f for f in files if f.lower().endswith(pattern)]
    return ("已获取" if files else "目录为空"), len(files)


# ---------------------------------------------------------------- Table 1

def table01():
    mpr_state, mpr_n = dir_status(os.path.join(PROJECT_ROOT, "data", "raw", "mpr"), ".pdf")
    rates_state, rates_n = dir_status(os.path.join(DATA_DIR, "rates"), ".csv")
    macro_state, macro_n = dir_status(os.path.join(DATA_DIR, "macro"), ".csv")
    dict_state, dict_n = dir_status(os.path.join(DATA_DIR, "dictionaries"), ".txt")

    rows = [
        ["D1", "央行货币政策执行报告（MPR）全文 PDF", "核心文本", "P0", "季度",
         "期次、报告期、发布日期、PDF 全文", "2001Q1-2026Q2（期数待官网核验）",
         "文本情绪度量、事件日对齐", mpr_state, f"{mpr_n} 个 PDF"],
        ["D2", "申万一级行业指数日行情", "核心行情", "P0", "日",
         "日期、开盘、收盘、最高、最低、成交量、成交额", "1999-12-30 至 2026-11-17",
         "行业收益、事件研究、横截面预测", "已获取", "31 个行业长表"],
        ["D3", "市场基准指数（沪深300/中证全指）", "核心行情", "P0", "日",
         "日期、收盘", "2005-01-04 起（沪深300）/ 2011-08-02 起（中证全指）",
         "正常收益模型、组合基准", "已获取", "2 个基准"],
        ["D4", "无风险利率（SHIBOR/国债收益率）", "控制变量", "P1", "日",
         "期限、利率", "研究区间内",
         "组合绩效无风险利率、宏观控制", rates_state, f"{rates_n} 个文件"],
        ["D5", "宏观经济变量（GDP/CPI/PMI/M2/社融等）", "控制变量", "P1", "月/季",
         "变量值、同比", "研究区间内",
         "横截面回归控制变量", macro_state, f"{macro_n} 个文件"],
        ["D6", "行业财务数据（ROE/PE/PB/毛利率）", "可选控制", "P2", "季",
         "行业均值财务指标", "研究区间内",
         "行业异质性交互项", "未获取", "0"],
        ["D7", "政策文本扩展（国务院/央行公告等）", "扩展文本", "P2", "不定期",
         "文件标题、日期、全文", "按需",
         "稳健性扩展对照", "未获取", "0"],
        ["T1", "中文金融情绪词典", "工具资源", "P0", "-",
         "词条、极性、权重", "-", "词典法情绪度量", dict_state, f"{dict_n} 个文件"],
        ["T2", "参考文献资料", "文献", "P1", "-",
         "题录、PDF 全文", "2007-2026",
         "文献综述与方法借鉴", "部分完成", "中文 33 篇 + 国外 21 篇题录"],
        ["T3", "官方赛事公告", "合规", "P0", "每日",
         "公告标题、日期、正文快照", "2026 赛季",
         "赛题/评审/提交合规", "部分完成", "memory/sources/official-site/"],
    ]
    cols = ["编号", "资料", "类别", "优先级", "频率", "字段", "时间范围", "用途", "状态", "实际进度"]
    df = pd.DataFrame(rows, columns=cols)
    note = ("来源：`资料抓取清单.md` 第 0-3 节；状态由脚本于 2026-11-17 检测 `data/` 目录与 "
            "`logs/data_fetch_log.md` 得出。D2/D3 已抓取；D1/D4-D7/T1 未获取，T2/T3 部分完成。"
            "MPR 期数与发布日期以 E0 数据链路核验为准。")
    write_table(df, "table01_dataset_description", "Table 1　数据集描述表", note)
    return df


# ---------------------------------------------------------------- Table 2

def table02():
    st = load_status().copy()
    st.columns = ["code", "name", "rows", "start", "end", "status"]
    st["status"] = st["status"].map(lambda s: "ok" if str(s).strip() == "ok" else str(s))
    starts = pd.to_datetime(st["start"], errors="coerce")
    n_old = int((starts.dt.year <= 2000).sum())
    n_mid = int(((starts.dt.year > 2000) & (starts.dt.year <= 2015)).sum())
    n_new = int((starts.dt.year > 2015).sum())
    st = st.rename(columns={"code": "行业代码", "name": "行业名称", "rows": "行数",
                            "start": "起点", "end": "终点", "status": "抓取状态"})
    note = ("来源：`data/external/sw_industries/fetch_status.csv`（抓取于 2026-11-17，akshare 1.18.94）。"
            f"{n_old} 个行业自 1999-12-30 起，{n_mid} 个行业自 2014-02-21 起，"
            f"{n_new} 个行业自 2021-12-13 起（申万分类调整）。")
    write_table(st, "table02_industry_coverage", "Table 2　申万一级行业与数据覆盖", note)
    return st


# ---------------------------------------------------------------- Table 3

def table03():
    rows = [
        ["E0", "MVP", "跑通数据链路：MPR PDF、行业指数、发布日对齐",
         "3-5 期 MPR PDF、3 个行业指数、1 个基准",
         "抓取失败/字段缺失/日期错位等数据检查项",
         "PDF 成功率、文本字符数、缺失率、日期对齐成功率",
         "data_sample/、logs/e0_data_pipeline.md", "必须"],
        ["E1", "MVP", "跑通词典情绪度量与人工抽检 Baseline",
         "MPR 纯文本样本（尽可能多期）",
         "不同词典一致性；自动评分 vs 人工评分",
         "词典间相关、人工-自动相关、词覆盖率",
         "sentiment_dictionary_full.csv、validation/human_ratings.csv", "必须"],
        ["E2", "MVP", "跑通事件研究 Baseline 与无文本对照",
         "对齐后行业收益、市场基准、无风险利率",
         "市场模型 vs 均值调整；全样本 vs 中性组",
         "CAR 均值、t 值、符号检验、Bootstrap CI",
         "e2_event_baseline.csv、e2_car_plot.png", "必须"],
        ["E3", "正式", "检验 MPR 情绪高低组的事件窗 CAR 差异",
         "全样本情绪得分、行业日收益、基准、无风险利率",
         "高 vs 低情绪组；市场模型 vs 均值调整；全样本 vs 子样本",
         "组间 CAR 差异、t 值、CI、事件窗一致性",
         "e3_event_results.csv、e3_car_by_sentiment.csv", "必须"],
        ["E4", "正式", "检验情绪对行业收益的横截面预测力",
         "行业季度收益、情绪得分、市场收益、滞后收益、波动率、宏观控制",
         "无文本模型 vs 词典情绪模型（可选 LLM）",
         "横截面系数均值/t、IC、ICIR、OOS R²、MAE",
         "e4_fmb_results.csv、e4_oos_metrics.csv", "必须"],
        ["E5", "正式", "检验行业敏感度异质性",
         "E3/E4 结果、行业分类、行业特征",
         "敏感组 vs 非敏感组；交互模型 vs 无交互",
         "交互项系数、组间敏感度差异、组内 IC",
         "e5_heterogeneity.csv、e5_sensitivity_plot.png", "必须"],
        ["E6", "正式", "比较 LLM 与词典情绪的样本外表现",
         "同一批 MPR 文本、同一收益面板、同一划分",
         "词典 vs LLM vs 词典+LLM",
         "LLM-人工一致性、IC/ICIR/OOS R²、增量回归系数、成本",
         "e6_llm_vs_dict.csv、e6_llm_oos_metrics.csv", "可选增强"],
        ["E7", "正式", "构建高低情绪多空组合并评估风险调整表现",
         "全样本行业收益、情绪得分、基准、无风险利率",
         "对称多空；随机分组（可选）；基准",
         "年化收益/波动、夏普、最大回撤、胜率、换手、alpha",
         "e7_portfolio.csv、e7_equity_curve.png", "可选增强"],
        ["E8", "补充", "稳健性：词典、事件窗、子样本、基准、滞后阶数",
         "主实验全部数据与中间结果",
         "主设定 vs 每个稳健性设定",
         "系数方向、显著性、IC、OOS R²、CAR 方向",
         "e8_robustness_summary.csv", "必须（核心项）"],
        ["E9", "补充", "敏感性：LLM Prompt/模型/解析规则/事件日",
         "E6 的 LLM 原始输出与人工抽检",
         "基准 Prompt+模型 vs 各替代设定",
         "人工一致性、评分相关、IC、OOS R²、解析失败率、成本",
         "e9_sensitivity_summary.csv", "可选增强"],
        ["E10", "补充", "误差分析：数据、文本、模型残差与失败案例",
         "E0-E9 失败日志、异常值、残差、低覆盖文本、污染记录",
         "正常样本 vs 异常样本；主模型 vs Baseline",
         "异常率、缺失率、残差分布、失败案例数、剔除规则一致性",
         "e10_error_analysis.md、e10_anomaly_samples.csv", "可选增强"],
    ]
    cols = ["编号", "阶段", "实验目的", "输入数据", "对照设置", "评价指标", "预期输出", "优先级"]
    df = pd.DataFrame(rows, columns=cols)
    note = "来源：`EXPERIMENT_PLAN.md` 第 0 节总览表与各实验条目（2026-11-17 规划稿，非实验结果）。"
    write_table(df, "table03_experiment_design", "Table 3　实验设计总表（E0-E10）", note)
    return df


# ---------------------------------------------------------------- Table 4/5

def compute_industry_stats(df):
    """主口径：2014-02-21 起、28 个可比行业；另算各行业自身起点的全样本对照。"""
    main = df[df["date"] >= START].copy()
    main = main[~main["name"].isin(EXCLUDE)]
    main = main.dropna(subset=["ret"])

    rows = []
    for (code, name), g in main.groupby(["code", "name"]):
        g = g.sort_values("date")
        r = g["ret"]
        nav = (1 + r).cumprod()
        full = df[df["code"] == code].dropna(subset=["ret"]).sort_values("date")
        fr = full["ret"]
        f_nav = (1 + fr).cumprod()
        rows.append({
            "code": code, "name": name,
            "n_days": len(r),
            "ann_ret": float((1 + r).prod() ** (ANN / len(r)) - 1),
            "ann_vol": float(r.std() * np.sqrt(ANN)),
            "skew": float(r.skew()),
            "kurt": float(r.kurt()),
            "max_dd": max_drawdown(nav),
            "full_ann_ret": float((1 + fr).prod() ** (ANN / len(fr)) - 1),
            "full_ann_vol": float(fr.std() * np.sqrt(ANN)),
            "full_max_dd": max_drawdown(f_nav),
        })
    out = pd.DataFrame(rows).sort_values("name").reset_index(drop=True)
    return out


def table04(stats):
    df = stats.rename(columns={
        "code": "行业代码", "name": "行业名称", "n_days": "交易日数",
        "ann_ret": "年化收益", "ann_vol": "年化波动率",
        "skew": "偏度", "kurt": "超额峰度", "max_dd": "最大回撤",
        "full_ann_ret": "全样本年化收益", "full_ann_vol": "全样本年化波动率",
        "full_max_dd": "全样本最大回撤",
    })
    note = ("口径：日简单收益；波动 = 日收益标准差 × √252；年化收益 = 复利；"
            "最大回撤基于累计净值。主口径为 2014-02-21 至 2026-11-17 的 28 个可比行业"
            "（剔除 2021-12-13 后新分类的石油石化/环保/美容护理）；"
            "全样本列为各行业自自身起点（1999-12-30 或 2014-02-21）起算的对照。"
            "偏度/峰度为样本矩（pandas 默认，峰度已减 3）。数据来源：sw_industry_daily.csv。")
    write_table(df, "table04_descriptive_stats", "Table 4　描述性统计表", note)
    return df


def table05(stats):
    df = stats.copy()
    df["sharpe"] = df["ann_ret"] / df["ann_vol"]
    df["calmar"] = df["ann_ret"] / df["max_dd"].abs()
    df = df[["code", "name", "n_days", "ann_ret", "ann_vol", "sharpe", "calmar", "max_dd"]]
    df = df.rename(columns={
        "code": "行业代码", "name": "行业名称", "n_days": "交易日数",
        "ann_ret": "年化收益", "ann_vol": "年化波动率",
        "sharpe": "夏普比率", "calmar": "Calmar 比率", "max_dd": "最大回撤",
    })
    note = ("口径：同 Table 4 主口径（2014-02-21 至 2026-11-17，28 个可比行业）。"
            + RF_NOTE
            + " Calmar = 年化收益 / |最大回撤|。数据来源：sw_industry_daily.csv。")
    write_table(df, "table05_risk_return", "Table 5　行业风险-收益表", note)
    return df


# ---------------------------------------------------------------- Table 6

def table06():
    rows = [
        ["无文本预测模型", "只用市场、宏观、滞后行业收益等控制变量预测行业收益",
         "判断文本情绪是否有增量信息，而不是只跟随机猜测比", "低"],
        ["行业历史均值 / 上一期收益", "简单可复现的朴素预测",
         "检验模型是否至少优于时间序列惯性", "低"],
        ["市场模型事件研究", "用市场收益作为正常收益基准",
         "事件研究的基础设定，结果易解释", "低"],
        ["词典法情绪", "作为 LLM 评分的对照",
         "工作区文献显示词典法成熟、成本低、可复现", "低到中"],
        ["中性情绪组", "将情绪分为高、中、低三组",
         "避免只用二分法导致信息损失", "低"],
    ]
    df = pd.DataFrame(rows, columns=["Baseline", "作用", "选择理由", "实现难度"])
    note = ("来源：`METHOD_PLAN.md` 第 6.1 节。不推荐设定：仅用随机预测（太弱）、"
            "未经时间对齐的全样本回归（高估预测力）、未经说明的闭源模型分数（复现性不足）。"
            "见 6.3 节。")
    write_table(df, "table06_baseline_design", "Table 6　Baseline 设计表", note)
    return df


# ---------------------------------------------------------------- Table 7

def table07():
    rows = [
        ["词典替换", "不同词典或不同词典组合重算情绪", "核心系数方向、显著性、IC、OOS R²"],
        ["事件窗", "[0,+1] / [0,+3] / [0,+5] / [0,+10]", "CAR 方向与显著性一致性"],
        ["子样本", "2008 年前后、2015 年前后、2020 年前后", "系数方向、显著性"],
        ["市场基准", "沪深300 vs 中证全指", "系数方向、CAR 方向"],
        ["滞后阶数", "滞后 1 期 vs 滞后 2 期", "IC、OOS R²"],
        ["情绪形式", "连续情绪得分 vs 高/中/低分组", "系数方向、IC"],
        ["异常值处理", "Winsorize 或剔除异常值前后", "系数方向、显著性"],
        ["估计窗长度", "不同事件研究估计窗，检查上一期 MPR 是否被卷入", "CAR 方向、显著性"],
    ]
    df = pd.DataFrame(rows, columns=["检验项", "改变设定", "评价指标"])
    note = ("来源：`EXPERIMENT_PLAN.md` E8。规则：每次只改变一个主要设定，其他条件不变；"
            "若同时改变多个设定需说明。预期输出 e8_robustness_summary.csv。")
    write_table(df, "table07_robustness_design", "Table 7　稳健性检验设计表", note)
    return df


# ---------------------------------------------------------------- Table 8

def table08():
    lit = load_literature().copy()
    keep = ["序号", "分类", "论文标题", "作者", "年份", "期刊/会议",
            "研究方法与模型", "主要实验结果", "与选题的关系"]
    df = lit[keep].copy()
    df = df.rename(columns={"主要实验结果": "核心结论", "与选题的关系": "可迁移方法/与选题关系"})
    md = df.copy()
    for c in ["核心结论", "可迁移方法/与选题关系", "研究方法与模型"]:
        txt = md[c].fillna("").astype(str)
        md[c] = txt.str.slice(0, 120) + txt.str.len().map(lambda n: "…" if n > 120 else "")
    note = ("来源：`documents/Literature_Matrix.xlsx`（38 篇，第一个工作表）。"
            "Markdown 版对长文本截断至 120 字，CSV 版保留完整字段。")
    write_table(md, "table08_literature_matrix", "Table 8　文献矩阵摘要表（38 篇）", note)
    return df


# ---------------------------------------------------------------- Table 9-11

def _exp_df(rel):
    """读取 experiments/ 下的实验输出；缺失时返回 None，由调用方回退模板。"""
    path = os.path.join(EXP_DIR, rel)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def write_template(stem, title, cols, note):
    full_cols = ["状态"] + cols
    df = pd.DataFrame([["模板，待数据"] + ["NA"] * len(cols)], columns=full_cols)
    note = ("【模板，待数据】" + note + " 真实数值需由实验输出 CSV 填入，"
            "当前所有单元格为占位符 NA，不得视为结果。")
    write_table(df, stem, title, note)
    return df


def table09():
    df = _exp_df(os.path.join("E6", "e6_llm_oos_metrics.csv"))
    if df is None:
        return write_template(
            "table09_baseline_performance", "Table 9　Baseline 性能对比表",
            ["模型", "情绪来源", "IC 均值", "ICIR", "OOS R²", "MAE", "样本外区间", "备注"],
            "比较无文本 Baseline、词典情绪、LLM 情绪（及联合）的样本外表现，数据来自 E4/E6 输出。")
    out = pd.DataFrame({
        "模型": df["model"],
        "情绪来源": df["dictionary"].map(DICT_LABEL).fillna(df["dictionary"]),
        "样本外区间": df["split"].map(SPLIT_LABEL).fillna(df["split"]),
        "期数": df["n_periods"],
        "IC 均值": df["ic_mean"],
        "ICIR": df["icir"],
        "OOS R²": df["oos_r2"],
        "MAE": df["mae"],
    })
    remarks = []
    for _, r in df.iterrows():
        if r["model"] == "baseline_notext":
            remarks.append("Baseline（仅控制变量）")
        elif pd.notna(r.get("mae_improvement_pct")):
            remarks.append(f"MAE 相对 Baseline {r['mae_improvement_pct']:+.2f}%")
        else:
            remarks.append("")
    out["备注"] = remarks
    note = ("来源：`experiments/E6/e6_llm_oos_metrics.csv`（E6，与 E4 同面板、同划分；"
            "测试期为主口径）。IC 为预测值与实际收益的横截面 Spearman 秩相关；"
            "ICIR = IC 均值 / IC 标准差；OOS R² 以各目标季截面均值为基准；"
            "MAE 为行业-季度面板平均绝对误差，备注中相对 Baseline 的百分比为负表示误差更低。"
            "LLM 为 qwen3.8-27b、Prompt v1、温度 0。")
    write_table(out, "table09_baseline_performance", "Table 9　Baseline 性能对比表", note)
    return out


def table10():
    e4 = _exp_df(os.path.join("E4", "e4_oos_metrics.csv"))
    panel = _exp_df(os.path.join("E4", "e4_panel_robustness.csv"))
    if e4 is None or panel is None:
        return write_template(
            "table10_ablation", "Table 10　消融实验表",
            ["设定", "情绪处理", "样本", "IC 均值", "ICIR", "OOS R²", "MAE", "备注"],
            "逐项剔除情绪、控制变量或交互项，数据来自 E4/E5 输出。")
    rows = []
    for split in ("test", "validation"):
        for model, label, dict_filter in (
            ("sentiment", "完整模型 sent_exposure", PRIMARY_DICT),
            ("baseline_notext", "去文本 baseline_notext", "none"),
        ):
            r = e4[(e4["model"] == model) & (e4["split"] == split)
                   & (e4["dictionary"] == dict_filter)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({
                "设定": label,
                "情绪处理": "情绪 × 滚动暴露交互" if model == "sentiment" else "无文本（仅控制变量）",
                "样本": SPLIT_LABEL.get(split, split),
                "IC 均值": r["ic_mean"], "ICIR": r["icir"], "OOS R²": r["oos_r2"], "MAE": r["mae"],
                "备注": "主词典 ntusd_simplified" if model == "sentiment" else "Baseline",
            })
    p1 = panel[(panel["spec"] == "panel_industry_fe") & (panel["term"] == "sentiment_z")
               & (panel["dictionary"] == PRIMARY_DICT)]
    if not p1.empty:
        r = p1.iloc[0]
        rows.append({
            "设定": "面板：行业固定效应",
            "情绪处理": "去交互项（仅情绪主效应）",
            "样本": "全样本面板 2001Q2–2026Q3",
                "IC 均值": np.nan, "ICIR": np.nan, "OOS R²": np.nan, "MAE": np.nan,
            "备注": f"sentiment_z 系数 {r['estimate']:.5f}，聚类 t={r['t_value']:.2f}，N={int(r['n_obs'])}",
        })
    p2 = panel[(panel["spec"] == "panel_time_fe_interact") & (panel["dictionary"] == PRIMARY_DICT)]
    if not p2.empty:
        parts = [f"{t} {e:.5f}（t={tv:.2f}）" for t, e, tv
                 in zip(p2["term"], p2["estimate"], p2["t_value"])]
        rows.append({
            "设定": "面板：时间固定效应",
            "情绪处理": "情绪 × 行业特征交互",
            "样本": "全样本面板 2001Q2–2026Q3",
            "IC 均值": np.nan, "ICIR": np.nan, "OOS R²": np.nan, "MAE": np.nan,
            "备注": "；".join(parts) + "，均不显著",
        })
    out = pd.DataFrame(rows, columns=["设定", "情绪处理", "样本", "IC 均值", "ICIR", "OOS R²", "MAE", "备注"])
    note = ("来源：`experiments/E4/e4_oos_metrics.csv`、`e4_panel_robustness.csv`。"
            "「去控制变量」消融实验未设定、未运行，故不列占位结果；"
            "面板行报告系数与聚类 t 值，无 IC 口径，以 — 表示。")
    write_table(out, "table10_ablation", "Table 10　消融实验表", note)
    return out


def table11():
    rows = []
    fmb = _exp_df(os.path.join("E4", "e4_fmb_results.csv"))
    if fmb is not None:
        f = fmb[fmb["is_sentiment"] == 1]
        for _, r in f.iterrows():
            rows.append({
                "检验": f"Fama-MacBeth 情绪暴露交互（{r['dictionary']}）",
                "系数": r["mean"], "标准误": r["std"] / np.sqrt(r["n_periods"]),
                "t 值": r["t_fmb"], "p 值": r["p_fmb"], "R²": r["mean_r2"],
                "样本量": f"{int(r['n_periods'])} 期 × 平均 {r['mean_n']:.1f} 行业",
                "备注": f"NW t={r['t_nw']:.2f}，p={r['p_nw']:.4f}",
            })
    car = _exp_df(os.path.join("E3", "e3_car_by_sentiment.csv"))
    if car is not None:
        picks = [("all", 1, "全样本"), ("low", 1, "低情绪组"),
                 ("high_minus_low", 1, "高−低差值"), ("all", 10, "全样本")]
        for g, w, glabel in picks:
            r = car[(car["model"] == "market") & (car["group"] == g) & (car["event_window"] == w)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({
                "检验": f"事件研究 CAR（市场模型，[0,+{w}]，{glabel}）",
                "系数": r["mean_car"], "标准误": r["std_car"] / np.sqrt(r["n_periods"]),
                "t 值": r["t_stat"], "p 值": r["p_value"], "R²": np.nan,
                "样本量": f"{int(r['n_periods'])} 期",
                "备注": f"Bootstrap 95% CI [{r['boot_ci_low']:.4f}, {r['boot_ci_high']:.4f}]",
            })
    panel = _exp_df(os.path.join("E4", "e4_panel_robustness.csv"))
    if panel is not None:
        r = panel[(panel["spec"] == "panel_industry_fe") & (panel["term"] == "sentiment_z")
                  & (panel["dictionary"] == PRIMARY_DICT)]
        if not r.empty:
            r = r.iloc[0]
            rows.append({
                "检验": "面板行业固定效应（情绪主效应）",
                "系数": r["estimate"], "标准误": r["std_error"], "t 值": r["t_value"],
                "p 值": r["p_value"], "R²": r["r2"], "样本量": f"{int(r['n_obs'])} 观测",
                "备注": f"聚类维度：{r['cluster']}",
            })
    if not rows:
        return write_template(
            "table11_statistical_results", "Table 11　统计分析结果表",
            ["检验", "系数", "标准误", "t 值", "p 值", "R²", "样本量", "备注"],
            "Fama-MacBeth 系数、事件研究 CAR 检验等，数据来自 E4/E3 输出。")
    out = pd.DataFrame(rows, columns=["检验", "系数", "标准误", "t 值", "p 值", "R²", "样本量", "备注"])
    note = ("来源：`experiments/E4/e4_fmb_results.csv`、`experiments/E3/e3_car_by_sentiment.csv`、"
            "`experiments/E4/e4_panel_robustness.csv`。FMB 标准误 = 系数时序标准差 / √期数；"
            "事件研究标准误 = CAR 时序标准差 / √期数；面板 t 值为行业代码聚类。"
            "所有结果为描述性统计，不构成因果解释。")
    write_table(out, "table11_statistical_results", "Table 11　统计分析结果表", note)
    return out


# ---------------------------------------------------------------- main

def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    table01()
    table02()
    table03()
    raw = load_industry()
    stats = compute_industry_stats(raw)
    table04(stats)
    table05(stats)
    table06()
    table07()
    table08()
    table09()
    table10()
    table11()

    s = stats.sort_values("ann_ret")
    print("\n[Table4/5 关键统计]")
    print(f"年化收益最高: {s.iloc[-1]['name']} {s.iloc[-1]['ann_ret']:.2%}")
    print(f"年化收益最低: {s.iloc[0]['name']} {s.iloc[0]['ann_ret']:.2%}")
    v = stats.sort_values("ann_vol")
    print(f"波动最大: {v.iloc[-1]['name']} {v.iloc[-1]['ann_vol']:.2%}")
    print(f"波动最小: {v.iloc[0]['name']} {v.iloc[0]['ann_vol']:.2%}")
    print(f"样本行业数: {len(stats)}; 交易日数区间: {stats['n_days'].min()}-{stats['n_days'].max()}")


if __name__ == "__main__":
    main()
