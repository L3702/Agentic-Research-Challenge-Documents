# Figure & Table Plan：MPR 文本情绪 × A 股行业收益

**版本**：v1.0（2026-11-17）
**配套文件**：`FIGURE_TABLE_CATALOG.md`、`plotting_scripts/`、`figures/`、`tables/`
**依据材料**：`METHOD_PLAN.md`、`EXPERIMENT_PLAN.md`、`资料抓取清单.md`、`Project_Overview.md`、`研究现状总结.md`、`documents/Literature_Matrix.xlsx`、`data/external/` 真实行情

## 0. 编制原则

1. **可追溯**：每张图表在 `FIGURE_TABLE_CATALOG.md` 中登记数据来源、脚本路径与生成时间。
2. **不编造**：实验结果图必须由真实实验输出 CSV 生成。当前工作区尚无 MPR 文本、词典得分与事件研究结果，因此这类图只提供**可复现的绘图代码与数据接口**，不生成任何假数据图，也不在图上标注虚构指标。
3. **不误导**：不使用截断纵轴、选择性样本或事后调参来放大结论；坐标轴从零或明确标注断点。
4. **口径一致**：事件日 = MPR 发布日期；收益对齐以发布日为准；描述统计与实验使用同一份行情数据版本。

**当前数据状态（2026-11-20 核验）**

- 已获取：申万一级 31 行业日行情（1999-12-30 至 2026-11-17，144,429 条）、沪深300 与中证全指日行情；MPR 全文 103 期（E1）；三套词典情绪得分（NTUSD 简体 / DLUT / MPR 领域）；宏观控制变量（E4 缓存）；无风险利率（E3 缓存）。
- 实验输出：E0-E10 已运行，事件研究（E3）、横截面预测（E4）、行业异质性（E5）、LLM 比较（E6）、多空组合（E7）、稳健性（E8）等结果文件就绪。
- 因此：**描述统计类与实验结果类图表均已生成**（Fig 1-20、Table 1-11），实验图输入由 `plotting_scripts/build_experiment_inputs.py` 从真实实验输出构造。

## 1. 图表总览

状态说明：**已生成** = 工作区存在真实图片/表格文件；**待数据** = 脚本已就位但输入未就绪（当前无待数据项）。

| 编号 | 图表名称 | 回答的问题 | 数据来源 | 绘图方式 | 状态 |
|------|----------|------------|----------|----------|------|
| Fig 1 | 研究框架图 | 研究问题、数据、方法、假设如何组织 | METHOD_PLAN.md | Matplotlib 流程图 | 已生成 |
| Fig 2 | 实验设计总览（E0–E10） | 每个实验解决什么、输入输出、优先级 | EXPERIMENT_PLAN.md | Matplotlib 流程图 | 已生成 |
| Fig 3 | 数据管线与时间对齐示意 | MPR 发布日如何对齐行业收益与事件窗 | 资料抓取清单.md、METHOD_PLAN.md | Matplotlib 示意图 | 已生成 |
| Fig 4 | 申万一级行业数据覆盖 | 31 行业各自数据起点与长度 | data/external/sw_industries/ | Matplotlib 甘特/横条图 | 已生成 |
| Fig 5 | 文献分类分布 | 研究版图的 7 条线索文献数量 | Literature_Matrix.xlsx | Matplotlib 条形图 | 已生成 |
| Fig 6 | 文献年份分布 | 领域研究热度随时间变化 | Literature_Matrix.xlsx | Matplotlib 柱状图 | 已生成 |
| Fig 7 | 文献方法族分布 | 方法工具构成（文本度量/计量/ML） | Literature_Matrix.xlsx | Matplotlib 堆叠/条形图 | 已生成 |
| Fig 8 | 行业累计净值曲线 | 2014 年后行业长期表现差异 | sw_industry_daily.csv | Matplotlib 折线 | 已生成 |
| Fig 9 | 行业风险-收益散点 | 收益与风险的行业分布 | sw_industry_daily.csv | Matplotlib 散点 | 已生成 |
| Fig 10 | 行业相关性热力图 | 行业间联动结构 | sw_industry_daily.csv | Matplotlib 热力图 | 已生成 |
| Fig 11 | 行业年化波动率对比 | 各行业风险水平排序 | sw_industry_daily.csv | Matplotlib 条形图 | 已生成 |
| Fig 12 | 市场基准走势 | 研究区间内市场环境 | benchmark_daily.csv | Matplotlib 折线 | 已生成 |
| Fig 13 | MPR 情绪得分时间序列 | 情绪是否可度量、如何演化 | E1 主词典得分（NTUSD 简体） | Matplotlib 折线 | 已生成 |
| Fig 14 | 事件研究 CAR 曲线 | 发布后高/中/低情绪组 CAR 差异 | E3 市场模型输出 | Matplotlib 折线 | 已生成 |
| Fig 15 | 行业 × 事件窗 CAR 热力图 | 哪些行业对情绪更敏感 | E3 行业市场模型输出 | Matplotlib 热力图 | 已生成 |
| Fig 16 | Fama-MacBeth 系数时序 | 情绪横截面系数的稳定性 | E4 主词典系数序列 | Matplotlib 折线 | 已生成 |
| Fig 17 | 样本外 IC 时序 | 情绪预测力是否稳定 | E4 逐期预测（计算） | Matplotlib 折线 | 已生成 |
| Fig 18 | 多空组合净值曲线 | 情绪信息的组合含义 | E7 组合输出 | Matplotlib 折线 | 已生成 |
| Fig 19 | 行业敏感度排序 | 行业异质性方向与幅度 | E5 敏感度输出 | Matplotlib 横向条形 | 已生成 |
| Fig 20 | 稳健性森林图 | 结论在哪些设定下成立 | E8 稳健性输出 | Matplotlib 误差棒 | 已生成 |

## 2. 逐图说明

### Fig 1 研究框架图

- **问题**：全文的逻辑骨架是什么？
- **数据**：METHOD_PLAN.md 第 1 节（Q1–Q3）、第 2 节（H1–H4）、第 3 节（方法 A/B）。
- **绘图**：左起三层：研究问题 → 数据/度量 → 方法与检验 → 假设与边界。

### Fig 2 实验设计总览

- **问题**：E0–E10 各自做什么、产出什么、哪些是必须项？
- **数据**：EXPERIMENT_PLAN.md 第 0 节总览表。
- **绘图**：泳道式流程图，按 MVP / 正式 / 补充三阶段分组。

### Fig 3 数据管线与时间对齐

- **问题**：MPR 发布日、事件窗、季度收益如何对齐？为什么不能用报告期截止日？
- **数据**：资料抓取清单.md D1/D2/D3、METHOD_PLAN.md 5.5。
- **绘图**：时间轴示意：报告期 → 发布日（事件日 t=0）→ 事件窗 [0,+1]…[0,+10] → 下一季度收益。

### Fig 4 行业数据覆盖

- **问题**：31 行业是否在同一时间起点？哪些行业是新分类？
- **数据**：真实抓取的 sw_industry_daily.csv（每行业 min/max 日期）。
- **绘图**：横向条形（甘特式），按起点分组显示 1999 / 2014 / 2021 三类。
- **注意**：801960/801970/801980 仅从 2021-12-13 起，801710–801890 从 2014-02-21 起，需在图中标注，避免误读为完整面板。

### Fig 5–7 文献分析

- **数据**：documents/Literature_Matrix.xlsx（38 篇，分类/年份/方法/结果字段）。
- **Fig 5**：7 个分类的文献数量。
- **Fig 6**：年份分布（2007–2026），反映领域热度。
- **Fig 7**：方法族分布（词典法、主题模型、预训练、LLM、事件研究、面板回归、TVP-VAR、ML/DL 等），由「研究方法与模型」字段归类计数。
- **注意**：文献方法归类为人工规则匹配，规则在脚本中显式列出，可复核。

### Fig 8–11 描述统计（真实行情）

- **样本口径**：统一使用 **2014-02-21 至 2026-11-17** 的可比区间，剔除 2021 年后新分类的 3 个行业（801960/801970/801980）单独说明；同时给出全样本口径的统计作为对照。
- **Fig 8**：行业累计净值（2014=1000 起），展示长期分化。
- **Fig 9**：年化收益（纵轴）vs 年化波动（横轴），标注行业名。
- **Fig 10**：日收益相关系数热力图（28 个 2014 起点行业）。
- **Fig 11**：年化波动率排序条形图。

### Fig 12 市场基准

- **数据**：沪深300、中证全指日收盘（真实）。
- **绘图**：归一化净值曲线，标注关键宏观时段（2015 股灾、2018 贸易摩擦、2020 疫情、2024 后）。

### Fig 13–20 实验结果图（已生成）

每张图对应一个实验编号，输入文件命名遵循 EXPERIMENT_PLAN.md 的预期输出：

实际输入由 `plotting_scripts/build_experiment_inputs.py` 从 `experiments/` 的真实输出筛选、
重命名后写入 `图表/results/`，再交给下列脚本出图；筛选口径在该脚本内集中登记。

| 图 | 输入 CSV | 关键列 |
|----|----------|--------|
| Fig 13 | sentiment_dictionary_full.csv | period, sentiment, coverage |
| Fig 14 | e3_car_by_sentiment.csv | window, group, car, t, ci_low, ci_high |
| Fig 15 | e3_event_results.csv | industry, window, car |
| Fig 16 | e4_coefficient_series.csv | period, beta, se |
| Fig 17 | e4_oos_metrics.csv | period, ic, icir, oos_r2 |
| Fig 18 | e7_portfolio.csv | period, long, short, ls, benchmark |
| Fig 19 | e5_heterogeneity.csv | industry, beta, ci |
| Fig 20 | e8_robustness_summary.csv | setting, beta, ci_low, ci_high |

脚本已就位，输入真实 CSV 后运行即可出图；列名若与计划不同，脚本顶部有 COLUMN_MAP 可改。

## 3. 表格清单

| 编号 | 表格名称 | 内容 | 数据来源 | 状态 |
|------|----------|------|----------|------|
| Table 1 | 数据集描述表 | 数据源、频率、字段、时间范围、用途、状态 | 资料抓取清单.md + 实际抓取 | 已生成 |
| Table 2 | 申万一级行业与数据覆盖 | 31 行业代码、名称、行数、起点、终点 | sw_industry_daily.csv | 已生成 |
| Table 3 | 实验设计总表 | E0–E10 目的、输入、对照、指标、输出 | EXPERIMENT_PLAN.md | 已生成 |
| Table 4 | 描述性统计表 | 行业收益/波动/偏度/峰度/最大回撤 | 真实行情 | 已生成 |
| Table 5 | 行业风险收益表 | 年化收益、年化波动、夏普、Calmar | 真实行情 | 已生成 |
| Table 6 | Baseline 设计表 | Baseline 定义、作用、选择理由 | METHOD_PLAN.md 第 6 节 | 已生成 |
| Table 7 | 稳健性检验设计表 | 检验项、改变设定、评价指标 | EXPERIMENT_PLAN.md E8 | 已生成 |
| Table 8 | 文献矩阵摘要表 | 38 篇文献的核心结论与可迁移方法 | Literature_Matrix.xlsx | 已生成 |
| Table 9 | Baseline 性能对比表 | 无文本 vs 词典 vs LLM 的 IC/OOS R² 等 | E6 输出（E4 同口径） | 已生成 |
| Table 10 | 消融实验表 | 去文本/去交互的结果；去控制未运行 | E4 输出 | 已生成 |
| Table 11 | 统计分析结果表 | FMB 系数、t、R²、样本量、事件研究 CAR | E4/E3 输出 | 已生成 |

## 4. 质量检查规则（第四步）

1. **数值一致性**：图表中的每个数字必须能回溯到 data/external/ 或实验输出 CSV；绘图脚本只读这些文件，不硬编码数值。
2. **跨图一致性**：Fig 8–11 与 Table 4/5 使用同一收益口径（2014-02-21 起、日简单收益、行业代码一致）。
3. **研究问题对应**：每张图标注其回答的子问题编号（Q1–Q3 / H1–H4）。
4. **反误导检查**：不使用非零起点的收益轴（除非明确标注为对数轴或断点）；不隐藏不显著结果（待数据图的脚本会同时输出不显著组）；相关/预测结论不写成因果。
5. **可追溯检查**：FIGURE_TABLE_CATALOG.md 记录每个文件的脚本、输入、生成时间、校验结果。

## 5. 目录结构

```
Agentic_Research_Challenge/
├── FIGURE_TABLE_PLAN.md
├── FIGURE_TABLE_CATALOG.md
├── figures/
├── tables/
├── plotting_scripts/
│   ├── _style.py
│   ├── fetch_data.py
│   ├── fig01_research_framework.py
│   ├── fig02_experiment_design.py
│   ├── fig03_timeline.py
│   ├── fig04_industry_coverage.py
│   ├── fig05_literature_category.py
│   ├── fig06_literature_year.py
│   ├── fig07_literature_methods.py
│   ├── fig08_cumulative_nav.py
│   ├── fig09_risk_return.py
│   ├── fig10_correlation.py
│   ├── fig11_volatility.py
│   ├── fig12_benchmark.py
│   ├── plot_event_study.py
│   ├── plot_fmb.py
│   ├── plot_portfolio.py
│   ├── plot_heterogeneity.py
│   └── plot_robustness.py
├── data/external/
└── logs/
```

*编制者：紫璃 (Murasaki)　2026-11-17*
