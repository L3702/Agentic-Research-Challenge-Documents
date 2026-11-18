# Figure & Table Audit：图表与表格学术可用性审查

- 审查时间：2026-11-18
- 审查对象：`图表/figures/`（12 张已生成图）、`图表/figures_preview/`（12 张单图 + 2 张总览）、`图表/tables/`（Table 1-11）、`图表/plotting_scripts/`、`logs/figure_table_qc.md`、`FIGURE_TABLE_CATALOG.md`、`FIGURE_TABLE_PLAN.md`
- 审查方式：逐张打开 `figures_preview/` 原图目视检查；用独立脚本从 `data/external/` 与 `documents/Literature_Matrix.xlsx` 重算全部数值；逐行读绘图与建表脚本；实测脚本在当前工作区能否复现
- 结论口径：**图像目视合格 ≠ 数值合格**。凡未追溯到真实数据/实验输出的项目，一律标注"无法完成数值核验"

---

## 0. 核心结论（先读这段）

1. **已生成的 12 张图，数值本身基本可信**：Fig 5-7 的文献计数、Fig 8-11 的行情统计、Fig 12 的基准走势，经独立重算全部与原始数据一致。
2. **但存在 3 个 Critical 级问题**，会使论文读者或评审对样本结构、核查结论产生误判：
   - Fig 4 图例写"1999 起 15 个 / 2014-02-21 起 13 个"，实际是 **16 个 / 12 个**；Table 2 注释同错。
   - `logs/figure_table_qc.md` 的"对照基准"与 Table 4/5 不是同一计算口径（先过滤后算收益 vs 先算收益后过滤），16 个老行业的年化收益对不上，而报告却写"全部通过"。
   - 所有数据图与核查脚本（Fig 4/8/9/10/11/12、verify）在当前工作区**无法运行**：`_style.py` 等脚本的 `DATA_DIR` 指向不存在的 `图表/data/external`，实测均抛 `FileNotFoundError`。
3. **Fig 13-20 与 Table 9-11 没有输入数据**，无法数值核验；工作区已如实标注"待数据/模板"，处理正确，但**不能用于论文初稿**。
4. 论文正文当前不在工作区（未发现 manuscript/paper 文件），因此"正文数值与图表是否一致"一项**无法核验**。

---

## 1. 图表清单（Fig 1-20）

| 编号 | 标题 | 文件名 | 类型 | 对应 RQ/H | 数据文件 | 生成代码 | 论文中准备支持的主张 | 状态 |
|------|------|--------|------|-----------|----------|----------|------------------------|------|
| Fig 1 | 研究框架：MPR 文本情绪 × A 股行业收益 | fig01_research_framework.png | 概念框架图 | Q1-Q3 / H1-H4 | METHOD_PLAN.md | fig01_research_framework.py | 研究问题、数据、方法、假设的组织方式 | 已生成 |
| Fig 2 | 实验设计总览（E0-E10） | fig02_experiment_design.png | 流程/泳道图 | H1-H4 | EXPERIMENT_PLAN.md | fig02_experiment_design.py | 每个实验的目的、输入输出与优先级 | 已生成 |
| Fig 3 | MPR 发布日与行业收益的时间对齐 | fig03_timeline.png | 示意图 | Q2 方法基础 | 资料抓取清单.md、METHOD_PLAN.md | fig03_timeline.py | 事件日=发布日、事件窗与估计窗不重叠 | 已生成 |
| Fig 4 | 申万一级 31 行业日行情数据覆盖 | fig04_industry_coverage.png | 甘特/横条图 | Q2 样本 | sw_industry_daily.csv | fig04_industry_coverage.py | 31 行业起点分 1999/2014/2021 三类，新分类行业需单独说明 | 已生成，**图例计数错误** |
| Fig 5 | 文献分类分布 | fig05_literature_category.png | 条形图 | 研究现状 | Literature_Matrix.xlsx | fig05_literature_category.py | 7 类文献数量分布（n=38） | 已生成 |
| Fig 6 | 文献年份分布 | fig06_literature_year.png | 柱状图 | 研究现状 | Literature_Matrix.xlsx | fig06_literature_year.py | 领域研究热度随时间上升 | 已生成 |
| Fig 7 | 文献方法族分布 | fig07_literature_methods.png | 条形图 | 研究现状 | Literature_Matrix.xlsx | fig07_literature_methods.py | 方法工具构成（可多选关键词归类） | 已生成 |
| Fig 8 | 申万一级行业累计净值 | fig08_cumulative_nav.png | 折线（对数轴） | Q2/Q3 背景 | sw_industry_daily.csv | fig08_cumulative_nav.py | 2014 年后行业长期表现分化 | 已生成 |
| Fig 9 | 行业风险-收益分布 | fig09_risk_return.png | 散点图 | Q2/Q3 背景 | sw_industry_daily.csv | fig09_risk_return.py | 收益与风险的行业分布 | 已生成 |
| Fig 10 | 行业日收益相关性 | fig10_correlation.png | 热力图 | Q2/Q3 背景 | sw_industry_daily.csv | fig10_correlation.py | 行业间联动结构 | 已生成 |
| Fig 11 | 行业年化波动率排序 | fig11_volatility.png | 条形图 | Q2/Q3 背景 | sw_industry_daily.csv | fig11_volatility.py | 各行业风险水平差异 | 已生成 |
| Fig 12 | 市场基准走势 | fig12_benchmark.png | 折线图 | Q2 背景 | benchmark_daily.csv | fig12_benchmark.py | 研究区间内市场环境 | 已生成 |
| Fig 13 | MPR 情绪得分时间序列 | — | 折线图 | Q1 / H1 | sentiment_dictionary_full.csv（未获取） | plot_sentiment.py | 情绪可度量与演化 | **待数据** |
| Fig 14 | 事件研究 CAR 曲线 | — | 折线图 | Q2 / H2 | e3_car_by_sentiment.csv（未产生） | plot_event_study.py | 高/中/低情绪组事件窗 CAR 差异 | **待数据** |
| Fig 15 | 行业×事件窗 CAR 热力图 | — | 热力图 | Q3 / H3 | e3_event_results.csv（未产生） | plot_event_heatmap.py | 行业对情绪的敏感度差异 | **待数据** |
| Fig 16 | Fama-MacBeth 系数时序 | — | 折线图 | Q2 / H2 | e4_coefficient_series.csv（未产生） | plot_fmb.py | 情绪横截面系数稳定性 | **待数据** |
| Fig 17 | 样本外 IC 时序 | — | 柱/线图 | Q2 / H2 | e4_oos_metrics.csv（未产生） | plot_oos.py | 情绪预测力稳定性 | **待数据** |
| Fig 18 | 多空组合净值曲线 | — | 折线图 | Q2 / H2 | e7_portfolio.csv（未产生） | plot_portfolio.py | 情绪信息的组合含义 | **待数据** |
| Fig 19 | 行业敏感度排序 | — | 横向条形图 | Q3 / H3 | e5_heterogeneity.csv（未产生） | plot_heterogeneity.py | 行业异质性方向与幅度 | **待数据** |
| Fig 20 | 稳健性森林图 | — | 误差棒图 | Q2/Q3 稳健性 | e8_robustness_summary.csv（未产生） | plot_robustness.py | 结论在哪些设定下成立 | **待数据** |

## 2. 表格清单（Table 1-11）

| 编号 | 标题 | 文件名 | 对应 RQ/H | 数据文件 | 生成代码 | 论文中准备支持的主张 | 状态 |
|------|------|--------|-----------|----------|----------|------------------------|------|
| Table 1 | 数据集描述表 | table01_dataset_description | 全篇数据基础 | 资料抓取清单.md + data/ 实检 | build_tables.py | 数据来源、频率、范围、状态 | 已生成 |
| Table 2 | 申万一级行业与数据覆盖 | table02_industry_coverage | Q2 样本 | fetch_status.csv / sw_industry_daily.csv | build_tables.py | 31 行业代码、行数、起止 | 已生成，**注释计数错误** |
| Table 3 | 实验设计总表（E0-E10） | table03_experiment_design | H1-H4 | EXPERIMENT_PLAN.md | build_tables.py | 实验目的、输入、对照、指标、输出 | 已生成 |
| Table 4 | 描述性统计表 | table04_descriptive_stats | Q2/Q3 | sw_industry_daily.csv | build_tables.py | 28 行业收益/波动/偏度/峰度/回撤 | 已生成 |
| Table 5 | 行业风险-收益表 | table05_risk_return | Q2/Q3 | sw_industry_daily.csv | build_tables.py | 年化收益/波动/夏普/Calmar | 已生成 |
| Table 6 | Baseline 设计表 | table06_baseline_design | H2/H4 | METHOD_PLAN.md 第 6 节 | build_tables.py | Baseline 定义与选择理由 | 已生成 |
| Table 7 | 稳健性检验设计表 | table07_robustness_design | Q2/Q3 | EXPERIMENT_PLAN.md E8 | build_tables.py | 稳健性检验项与指标 | 已生成 |
| Table 8 | 文献矩阵摘要表（38 篇） | table08_literature_matrix | 研究现状 | Literature_Matrix.xlsx | build_tables.py | 文献核心结论与可迁移方法 | 已生成 |
| Table 9 | Baseline 性能对比表 | table09_baseline_performance_template | H2/H4 | E4/E6 输出（未产生） | build_tables.py | 无文本/词典/LLM 样本外对比 | **模板，全 NA** |
| Table 10 | 消融实验表 | table10_ablation_template | H2/H3 | E4/E5 输出（未产生） | build_tables.py | 去文本/去控制/去交互结果 | **模板，全 NA** |
| Table 11 | 统计分析结果表 | table11_statistical_results_template | H2/H3 | E4 输出（未产生） | build_tables.py | FMB 系数/t/R²/样本量 | **模板，全 NA** |

---

## 3. 实验结果图核验（Fig 4、8-12）

### 3.1 独立重算方法与口径

- 数据源：`data/external/sw_industries/sw_industry_daily.csv`（144,429 行，31 行业，1999-12-30 至 2026-11-17）；`data/external/benchmark/benchmark_daily.csv`。
- 主口径：`START = 2014-02-21`，剔除石油石化/环保/美容护理，28 个可比行业；年化 = 复利；波动 = 日标准差 × √252。
- 重算脚本：`图表/plotting_scripts/_audit_recompute.py`（本次审查新建，仅打印对照，不改输出）。

### 3.2 Fig 4 行业数据覆盖

- **数据可追溯**：是。行业起点、行数、终点均来自真实抓取文件，与 `fetch_status.csv` 一致（行数合计 144,429，含煤炭 3,087 行）。
- **图中数值可重算**：条形长度与条端标注可重算；但**图例文字错误**（见问题 C1）。
- **样本/分组/单位**：31 行业；起点实际分组为 1999-12-30（16 个）、2014-02-21（12 个）、2021-12-13（3 个）。
- **坐标轴/图例**：横轴延伸至 2028（数据止于 2026-11-17），右侧留白偏大；图例位于右下，**遮挡"综合 801230"的条端数字**。
- **不确定性**：不适用（描述性覆盖图）。
- **结论**：图本身信息正确，但图例计数与遮挡必须修复。

### 3.3 Fig 8 累计净值

- **数据可追溯**：是。脚本读取真实 CSV。
- **可重算**：是。高亮行业终点净值（2014-02-21=1000，含当日收益口径）重算：煤炭 3257.0、食品饮料 3175.4、银行 2208.0、计算机 1343.0、传媒 655.6，与图中 2026 年位置一致。
- **口径**：28 个可比行业，剔除 3 个 2021 年新分类行业，图注已说明。
- **问题**：纵轴对数刻度标签混排（`4×10²`、`6×10²`、`4×10³`、`6×10³` 与 500/750/1500/2000/3000/5000 并列），读感混乱；起点标注"2014-02-21 = 1000"存在歧义（实际含 2014-02-21 当日收益，1000 对应 2014-02-20 收盘）。
- **不确定性**：不适用（描述性净值）。
- **结论**：数值合格，需修刻度与起点注释（Minor）。

### 3.4 Fig 9 风险-收益散点

- **可重算**：是。抽查煤炭（35.3%, 10.1%）、电子（33.2%, 9.6%）、食品饮料（26.2%, 9.9%）、银行（21.4%, 6.7%）、传媒（31.3%, -3.4%）、商贸零售（26.2%, -5.4%）、轻工制造（24.1%, 0.03%），全部与重算一致。
- **样本**：n=28，图注已标注。
- **坐标轴**：横轴为波动率（非收益），不从零开始可接受；纵轴含 0 虚线，未截断收益轴。
- **问题**：标签"医药生物"与"农林牧渔"文字重叠；"国防军工"与"计算机"标签距离过近。
- **结论**：数值合格，需修标签碰撞（Minor）。

### 3.5 Fig 10 相关性热力图

- **可重算**：是。抽查煤炭-钢铁 0.6458、银行-非银 0.6770、食品饮料-家电 0.7119、计算机-电子 0.8432、传媒-计算机 0.8638，与图中数值一致；全矩阵范围 0.1857-1.00。
- **样本**：n=28，图注已标注。
- **坐标轴/图例**：色标 -1 到 1，标签为行业简称，清晰。
- **问题**：单元格数值字号 5.8，在压缩预览中偏小（原图 300dpi 下可读）；无其他问题。
- **结论**：合格，可保留。

### 3.6 Fig 11 年化波动率

- **可重算**：是。煤炭 35.3%、国防军工 34.4%、计算机 34.0%、银行 21.4% 等全部一致；中位数 27.5% 与重算一致（第 14/15 位 27.4% 与 27.7% 的均值）。
- **样本**：n=28；颜色按是否高于中位数区分，规则在图注中说明。
- **结论**：合格，可保留。

### 3.7 Fig 12 市场基准

- **可重算**：是。沪深300（2005-01-04 起，5311 条）、中证全指（2011-08-02 起，3713 条），归一化净值与图一致。
- **口径**：两条基准**各自起点归一**（脚本 `groupby("code")` 内 `s / s.iloc[0] * 1000`），并非同一基准日归一；标题"全样本起点归一"含糊。
- **问题**：事件标注文字（2018 贸易摩擦、2020 疫情、2024 政策转向）与 x 轴年份刻度重叠，影响阅读。
- **结论**：数值合格，需修标题与标注位置（Minor/Major）。

---

## 4. 统计表格核验（Table 1-11）

### 4.1 Table 1 数据集描述

- 内容来源为 `资料抓取清单.md` + 目录实检，D2/D3 已获取、D1/D4-D7/T1 未获取，与工作区实际一致。
- **问题 M6**：D1 时间范围写"2001Q1-2026Q2（102 期）"为确定值，但 `METHOD_PLAN.md` 5.5 明确记载"102 期 vs 约 90 份"口径冲突、待官网核验。把未核验口径写成确定值，属图文/文文不一致风险。

### 4.2 Table 2 行业覆盖
- 表格数据本身正确：31 行业、行数与实际文件逐行业一致（含煤炭 3,087）。
- **问题 C1 延伸**：表注硬编码"15 个行业自 1999-12-30 起，13 个行业自 2014-02-21 起"，实际 16/12。

### 4.3 Table 3 / 6 / 7 设计类表格
- Table 3 与 `EXPERIMENT_PLAN.md` 第 0 节逐条一致（E0-E10 目的、输入、对照、指标、输出、优先级）。
- Table 6 与 `METHOD_PLAN.md` 6.1 一致；Table 7 与 E8 一致。
- 无数字计算，未发现遗漏或重复列。

### 4.4 Table 4 / 5 描述统计与风险收益
- **数值可追溯**：28 行业全部数值可由 `sw_industry_daily.csv` 重算得到，且与"先算收益后过滤"口径完全一致（diff = 0）。
- **统计正确性**：偏度/峰度为 pandas 样本矩（峰度已减 3，表注已说明）；最大回撤基于累计净值；夏普 = 年化收益/年化波动（无风险利率取 0，表注已声明"待补"）；Calmar = 年化收益/|最大回撤|。
- **口径一致性**：主口径 28 行业；全样本对照列对 1999 起点行业有效，对 2014 起点行业退化为与主口径相同（可接受）。
- **问题 C2**：`logs/figure_table_qc.md` 的对照基准使用"先过滤后算收益"口径，与 Table 4/5 不一致（16 个老行业年化收益不同），但报告声称"全部通过"。
- **小数位数/单位**：统一为 4 位小数、比率不带百分号；表头统一。未发现缺失值误写为 0。

### 4.5 Table 8 文献矩阵摘要
- 38 行，与 `Literature_Matrix.xlsx` 逐字段核对：序号、标题、分类、年份全部一致；核心结论与可迁移方法无缺失。
- Markdown 版对长文本截断至 120 字（CSV 保留完整），表注已声明；论文引用时应使用 CSV 完整版。

### 4.6 Table 9 / 10 / 11 模板
- 三表当前所有单元格为占位符 `NA`，并明确标注"模板，待数据""不得视为结果"。
- **结论**：处理方式正确，但**不得进入论文初稿**，需等 E3/E4/E5/E6 输出后填入真实结果。

---

## 5. 概念图与方法框架图核验（Fig 1-3）

### 5.1 Fig 1 研究框架图
- 节点：Q1/Q2/Q3 → 数据与度量 → 方法与检验 → H1/H2/H3；底部方法对照 H4 与 Baseline、因果边界。
- 与 `METHOD_PLAN.md` 第 1 节（Q1-Q3）、第 2 节（H1-H4）、第 3 节（方法 A/B）、第 6 节（Baseline）一致；未出现方法中不存在的模块。
- 文字清晰、布局均衡，无逻辑错误。**合格**。

### 5.2 Fig 2 实验设计总览
- E0-E10 分组与优先级和 `EXPERIMENT_PLAN.md` 一致（E6/E7 标为"增强（可选）"，与计划"可选增强"一致）。
- **问题 m7**：预期输出列不完整：E2 缺 `e2_event_baseline.csv`，E3 缺 `e3_event_results.csv`/`e3_car_by_sentiment.csv`，E4 缺 `e4_oos_metrics.csv`/`e4_coefficient_series.csv`，E5 缺 `e5_heterogeneity.csv`，E7 缺 `e7_portfolio.csv`。
- 无逻辑错误。**合格，建议补全输出列**。

### 5.3 Fig 3 时间对齐示意
- 报告期截止 → 估计窗（前 120 交易日）→ 发布日 t=0 → 事件窗 [0,+1]…[0,+10] → 下一季度收益。
- 与 `METHOD_PLAN.md` 5.5、`EXPERIMENT_PLAN.md` E2 的事件日/估计窗设定一致；未出现错误逻辑关系。
- **合格**。可考虑在图中补一句"估计窗与事件窗不得重叠"（计划中有此要求，图中未画/未写）。

---

## 6. 问题分级清单

### Critical

**C1｜Fig 4 图例与 Table 2 注释的行业计数错误**
- 位置：`fig04_industry_coverage.py` 图例（"1999 起（15 个老行业）""2014-02-21 起（13 个分类调整行业）"）；`build_tables.py` `table02()` 表注；输出 `figures/fig04_industry_coverage.png`、`tables/table02_industry_coverage.md`。
- 证据：`sw_industry_daily.csv` 起点分组实际为 1999-12-30 共 **16** 个、2014-02-21 共 **12** 个、2021-12-13 共 3 个（`fetch_status.csv` 同）。
- 影响：读者会误判样本结构与可比区间，直接影响 Q2 样本描述的可信度。
- 建议：把图例与表注改为 16/12/3，或改为脚本动态计数（推荐），避免再次漂移。
- 是否重算/重绘：需要重新生成图例与表注（条形本身无需重算）。

**C2｜质量核查报告与交付表格口径不一致，"全部通过"结论不成立**
- 位置：`logs/figure_table_qc.md`；根因在 `verify_figures_tables.py` 的 `recompute()`（先按 `date >= START` 过滤，再 `pct_change`）与 `build_tables.py` 的 `compute_industry_stats()`（先全样本 `pct_change`，再过滤）口径不同。
- 证据：对 16 个 1999 起点行业，两口径年化收益不同，例如交通运输 1.9428%（qc 基准）vs 1.8065%（Table 4）；商贸零售 -5.2804% vs -5.4086%；钢铁 2.1810% vs 2.0629%。qc 报告中的数字与 Table 4 不符，却写"全部通过"。
- 影响：核查报告本身不可信，无法作为验收依据；后续实验若沿用不同口径，Fig 8-11 与 Table 4/5 之间会出现隐性不一致。
- 建议：统一为一种口径（推荐"先算收益后过滤"，与 Table 4/5、Fig 8-11 一致），重写 `verify_figures_tables.py` 的 `recompute()`，重跑并更新 qc 报告；在报告中显式写明过滤与收益计算顺序。
- 是否重算：需要重跑核查并重写报告。

**C3｜绘图与核查脚本在当前工作区不可复现**
- 位置：`_style.py`、`fetch_data.py`、`build_tables.py`、`build_catalog.py`、`verify_figures_tables.py` 的 `ROOT = dirname(dirname(__file__))` 指向 `图表/`，导致 `DATA_DIR = 图表/data/external`（不存在）。
- 证据：实测 `python 图表/plotting_scripts/fig08_cumulative_nav.py` 与 `verify_figures_tables.py` 均抛 `FileNotFoundError: ...图表\data\external\sw_industries\sw_industry_daily.csv`。`FIGURE_TABLE_CATALOG.md` 第 5 节的"复现方式"因此失效。
- 影响：评审或合作者按文档无法重跑；图表"可复现"声明不成立。
- 建议：把 ROOT 上移一级到项目根（`dirname` 三次），或把 `DATA_DIR` 显式指向 `项目根/data/external`；同步修正 `build_catalog.py` 的输出路径（当前会写到 `图表/FIGURE_TABLE_CATALOG.md`，而实际 catalog 在项目根）。
- 是否重算：需要修正路径后重跑全部脚本。

### Major

**M4｜Fig 5/6/7 注释与坐标轴标签重叠**
- 位置：三张图底部 `ax.text(0.01, -0.12, ...)` 与 x 轴标签（"文献数量（篇）""发表年份""命中文献数（篇，可多选）"）重叠。
- 证据：`figures_preview/fig05/06/07` 目视；脚本 `fig05/06/07_literature_*.py`。
- 影响：图注不可读，影响论文排版质量。
- 建议：把注释移到图内空白区，或改为 `fig.text` 并下移 xlabel/注释间距（如 `labelpad` 或 `fig.subplots_adjust(bottom=...)`）。
- 是否重绘：需要。

**M5｜Fig 4 图例遮挡条端数值**
- 位置：`fig04` 图例位于右下，压住"综合 801230"的"6,494 条"标签。
- 建议：图例移到左上空白区，或缩短横轴右边界。
- 是否重绘：需要（与 C1 一并处理）。

**M6｜Table 1 把未核验的 MPR 期数写成确定值**
- 位置：`table01_dataset_description.md` D1 行"2001Q1-2026Q2（102 期）"。
- 证据：`METHOD_PLAN.md` 5.5 记载"102 期 vs 约 90 份"口径冲突、待核验。
- 建议：改为"2001Q1-2026Q2（期数待官网核验）"，待 E0 完成后再锁定。
- 是否重算：否，改文字。

**M7｜Fig 2 预期输出列不完整**
- 位置：`fig02_experiment_design.py` 的 `GROUPS` 输出字段。
- 证据：对照 `EXPERIMENT_PLAN.md` 各实验"预期输出文件"。
- 建议：补全 E2/E3/E4/E5/E7 的 CSV 输出，或注明"仅列主要输出"。
- 是否重绘：需要（若补全）。

### Minor

- **m8** Fig 8 对数轴刻度标签混排（`4×10²`/`6×10²` 与 500/750 并列）：关闭次刻度标签或自定义刻度。
- **m9** Fig 8 起点注释"2014-02-21 = 1000"歧义：改为"2014-02-21 起，期初 1000（含当日收益）"或调整口径。
- **m10** Fig 9 标签碰撞："医药生物"与"农林牧渔"重叠；"国防军工"与"计算机"过近。用自动避让或手动偏移。
- **m11** Fig 12 事件标注与 x 轴刻度重叠：把标注上移或改为图例说明。
- **m12** Fig 12 标题"全样本起点归一"含糊：改为"各自样本起点归一（沪深300：2005-01-04；中证全指：2011-08-02）"。
- **m13** Fig 10 单元格字号 5.8 偏小：原图 300dpi 可读，若论文缩放需确认。
- **m14** Fig 7 关键词规则可能过宽（`VAR` 子串、`回归`）：图注已声明"可多选、关键词归类"，建议正文补充规则表或附录。
- **m15** Fig 4 横轴延伸至 2028，右侧留白偏大：收窄到 2027 或标注数据截止。

---

## 7. 无法完成数值核验的项目

| 项目 | 缺少的材料 | 说明 |
|------|------------|------|
| Fig 13 | MPR 全文文本、情绪词典/LLM 得分 | `sentiment_dictionary_full.csv` 未获取 |
| Fig 14 | E3 事件研究结果 | `e3_car_by_sentiment.csv` 未产生 |
| Fig 15 | E3/E5 行业×事件窗结果 | `e3_event_results.csv` 未产生 |
| Fig 16 | E4 FMB 系数时序 | `e4_coefficient_series.csv` 未产生 |
| Fig 17 | E4 样本外指标 | `e4_oos_metrics.csv` 未产生 |
| Fig 18 | E7 组合净值 | `e7_portfolio.csv` 未产生 |
| Fig 19 | E5 行业敏感度 | `e5_heterogeneity.csv` 未产生 |
| Fig 20 | E8 稳健性汇总 | `e8_robustness_summary.csv` 未产生 |
| Table 9-11 | E3/E4/E5/E6 实验输出 | 当前全为 NA 模板 |
| 论文正文一致性 | 论文正文文件 | 工作区未发现 manuscript/paper，无法核验"正文数值=图表=CSV" |

---

## 8. 最终结论（对应用户第七部分）

**可以保留（数值合格，修改排版/标注后可进初稿）**
- Fig 1、Fig 2、Fig 3：概念图，逻辑与 METHOD_PLAN/EXPERIMENT_PLAN 一致。
- Fig 5、Fig 6、Fig 7：文献计数经重算一致，需修底部注释重叠。
- Fig 8、Fig 9、Fig 10、Fig 11：行情统计经独立重算一致，需修刻度/标签/遮挡。
- Fig 12：基准数据真实，需修标题与事件标注。
- Table 3、6、7、8：内容/数值与来源一致，可保留。
- Table 4、5：数值正确，但需先解决 C2 口径冲突，否则不能作为"已核查"交付。

**必须重新绘制**
- Fig 4：图例计数错误 + 图例遮挡（C1、M5）。

**需要补充数据或统计分析**
- Fig 13-20 全部；Table 9-11 全部。缺 MPR 文本、词典、宏观控制变量与 E3/E4/E5/E6/E7/E8 实验输出。

**图表与论文主张是否一致**
- 无法核验正文（不存在）。但已发现两处文档层不一致：Table 1 的 102 期口径与 METHOD_PLAN 冲突（M6）；qc 报告与 Table 4/5 口径冲突（C2）。

**是否达到论文初稿使用要求**
- **描述统计与文献部分**：接近可用，但须先修复 C1/C2/C3 与 M4/M5，并明确口径注释。
- **实验结果部分**：不达到。核心假设 H1-H4 的图（Fig 13-20）与表（Table 9-11）均无数据，当前无法支撑任何研究结论。
- **总体判断**：图表体系的"骨架"（框架、设计、数据覆盖、描述统计）已搭好且数据真实；但"证据链"（文本情绪→收益）尚未建立，核查流程本身也有口径与路径缺陷。建议在修复 Critical 项后，把工作重心转向 E0-E2 数据链路。

---

*审查：紫璃 (Murasaki)　2026-11-18*
