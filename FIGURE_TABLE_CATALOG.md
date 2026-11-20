# Figure & Table Catalog：图表目录与数据来源

- 生成时间：2026-11-20 14:33
- 配套：`FIGURE_TABLE_PLAN.md`、`图表/figures/`、`图表/tables/`、`图表/plotting_scripts/`
- 状态口径：**已生成** = 工作区存在真实文件；**待数据** = 脚本已就位，输入 CSV 就绪后即可出图，本次不生成图片。

## 1. 图（Fig 1-20）

| 编号 | 文件 | 绘图脚本 | 输入数据 | 生成时间 | 状态 | 回答的问题 |
|------|------|----------|----------|----------|------|------------|
| Fig 1 | fig01_research_framework.png | fig01_research_framework.py | METHOD_PLAN.md | 2026-11-18 08:47 | 已生成 | 研究框架：Q1-Q3、H1-H4、方法 A/B 的组织方式 |
| Fig 2 | fig02_experiment_design.png | fig02_experiment_design.py | EXPERIMENT_PLAN.md | 2026-11-18 08:50 | 已生成 | E0-E10 实验设计总览（MVP/正式/补充） |
| Fig 3 | fig03_timeline.png | fig03_timeline.py | 资料抓取清单.md、METHOD_PLAN.md | 2026-11-18 08:47 | 已生成 | MPR 发布日与行业收益的事件窗对齐 |
| Fig 4 | fig04_industry_coverage.png | fig04_industry_coverage.py | data/external/sw_industries/sw_industry_daily.csv | 2026-11-18 08:47 | 已生成 | 31 行业数据起点与长度（1999/2014/2021 三类） |
| Fig 5 | fig05_literature_category.png | fig05_literature_category.py | documents/Literature_Matrix.xlsx | 2026-11-18 08:47 | 已生成 | 文献 7 分类数量分布 |
| Fig 6 | fig06_literature_year.png | fig06_literature_year.py | documents/Literature_Matrix.xlsx | 2026-11-18 08:47 | 已生成 | 文献年份分布（2007-2026） |
| Fig 7 | fig07_literature_methods.png | fig07_literature_methods.py | documents/Literature_Matrix.xlsx | 2026-11-18 08:47 | 已生成 | 文献方法族分布（词典/主题/预训练/LLM/计量/ML） |
| Fig 8 | fig08_cumulative_nav.png | fig08_cumulative_nav.py | data/external/sw_industries/sw_industry_daily.csv | 2026-11-18 08:53 | 已生成 | 28 行业累计净值（2014-02-21=1000，对数轴） |
| Fig 9 | fig09_risk_return.png | fig09_risk_return.py | data/external/sw_industries/sw_industry_daily.csv | 2026-11-18 08:47 | 已生成 | 行业年化收益 vs 年化波动 |
| Fig 10 | fig10_correlation.png | fig10_correlation.py | data/external/sw_industries/sw_industry_daily.csv | 2026-11-18 08:47 | 已生成 | 28 行业日收益相关系数热力图 |
| Fig 11 | fig11_volatility.png | fig11_volatility.py | data/external/sw_industries/sw_industry_daily.csv | 2026-11-18 08:47 | 已生成 | 行业年化波动率排序 |
| Fig 12 | fig12_benchmark.png | fig12_benchmark.py | data/external/benchmark/benchmark_daily.csv | 2026-11-18 08:47 | 已生成 | 沪深300 / 中证全指归一化走势 |
| Fig 13 | fig13_sentiment.png | plot_sentiment.py | experiments/E1/sentiment_dictionary_full.csv | 2026-11-20 06:33 | 已生成 | MPR 情绪得分时间序列与覆盖率 |
| Fig 14 | fig14_event_study_car.png | plot_event_study.py | experiments/E3/e3_car_by_sentiment.csv | 2026-11-20 06:33 | 已生成 | 高/中/低情绪组事件窗 CAR 曲线 |
| Fig 15 | fig15_event_heatmap.png | plot_event_heatmap.py | experiments/E3/e3_event_results.csv | 2026-11-20 06:33 | 已生成 | 行业 × 事件窗 CAR 热力图 |
| Fig 16 | fig16_fmb_coefficient.png | plot_fmb.py | experiments/E4/e4_coefficient_series.csv | 2026-11-20 06:33 | 已生成 | Fama-MacBeth 情绪系数时序 |
| Fig 17 | fig17_oos_metrics.png | plot_oos.py | _temp/e4/e4_oos_predictions.csv（逐期计算） | 2026-11-20 06:33 | 已生成 | 样本外 IC / ICIR / OOS R² |
| Fig 18 | fig18_portfolio_equity.png | plot_portfolio.py | experiments/E7/e7_portfolio.csv | 2026-11-20 06:33 | 已生成 | 多空组合累计净值 |
| Fig 19 | fig19_heterogeneity.png | plot_heterogeneity.py | experiments/E5/e5_heterogeneity.csv | 2026-11-20 06:33 | 已生成 | 行业敏感度排序与置信区间 |
| Fig 20 | fig20_robustness.png | plot_robustness.py | experiments/E8/e8_robustness_summary.csv | 2026-11-20 06:33 | 已生成 | 稳健性森林图（event 块） |

## 2. 表（Table 1-11）

| 编号 | 文件（.md/.csv） | 生成脚本 | 输入数据 | 生成时间 | 状态 | 内容 |
|------|------------------|----------|----------|----------|------|------|
| Table 1 | table01_dataset_description.md/.csv | build_tables.py | 资料抓取清单.md + data/ 实检 | 2026-11-20 06:33 | 已生成 | 数据集描述：来源、频率、字段、范围、用途、状态 |
| Table 2 | table02_industry_coverage.md/.csv | build_tables.py | data/external/sw_industries/ | 2026-11-20 06:33 | 已生成 | 31 行业代码、名称、行数、起止 |
| Table 3 | table03_experiment_design.md/.csv | build_tables.py | EXPERIMENT_PLAN.md | 2026-11-20 06:33 | 已生成 | E0-E10 目的、输入、对照、指标、输出、优先级 |
| Table 4 | table04_descriptive_stats.md/.csv | build_tables.py | data/external/sw_industries/sw_industry_daily.csv | 2026-11-20 06:33 | 已生成 | 28 行业收益/波动/偏度/峰度/最大回撤（含全样本对照） |
| Table 5 | table05_risk_return.md/.csv | build_tables.py | data/external/sw_industries/sw_industry_daily.csv | 2026-11-20 06:33 | 已生成 | 年化收益、年化波动、夏普、Calmar |
| Table 6 | table06_baseline_design.md/.csv | build_tables.py | METHOD_PLAN.md 第 6 节 | 2026-11-20 06:33 | 已生成 | Baseline 定义、作用、选择理由 |
| Table 7 | table07_robustness_design.md/.csv | build_tables.py | EXPERIMENT_PLAN.md E8 | 2026-11-20 06:33 | 已生成 | 稳健性检验项、改变设定、评价指标 |
| Table 8 | table08_literature_matrix.md/.csv | build_tables.py | documents/Literature_Matrix.xlsx | 2026-11-20 06:33 | 已生成 | 38 篇文献核心结论与可迁移方法摘要 |
| Table 9 | table09_baseline_performance.md/.csv | build_tables.py | experiments/E6/e6_llm_oos_metrics.csv | 2026-11-20 06:33 | 已生成 | Baseline 性能对比（无文本/词典/LLM/联合） |
| Table 10 | table10_ablation.md/.csv | build_tables.py | experiments/E4/e4_oos_metrics.csv、e4_panel_robustness.csv | 2026-11-20 06:33 | 已生成 | 消融实验（去文本/去交互；去控制未运行） |
| Table 11 | table11_statistical_results.md/.csv | build_tables.py | experiments/E4/e4_fmb_results.csv、experiments/E3/e3_car_by_sentiment.csv | 2026-11-20 06:33 | 已生成 | 统计分析结果（FMB 系数、事件研究 CAR、面板） |

## 3. 数据源

| 数据 | 路径 | 范围 | 用途 |
|------|------|------|------|
| 申万一级行业指数日行情 | data/external/sw_industries/sw_industry_daily.csv | 1999-12-30 至 2026-11-17，31 行业 | Fig 4/8/9/10/11、Table 2/4/5 |
| 市场基准日行情 | data/external/benchmark/benchmark_daily.csv | 见文件 | Fig 12 |
| 文献矩阵 | documents/Literature_Matrix.xlsx | 38 篇 | Fig 5/6/7、Table 8 |
| 抓取台账 | logs/data_fetch_log.md | 2026-11-17 | 数据状态核查 |
| MPR 情绪得分（E1 词典法） | experiments/E1/sentiment_dictionary_full.csv | 103 期，2001Q1–2026Q3 | Fig 13、Table 9-11 |
| 事件研究输出（E3） | experiments/E3/e3_car_by_sentiment.csv、e3_event_results.csv | 89 个主样本事件 | Fig 14/15、Table 11 |
| 横截面预测输出（E4） | experiments/E4/e4_coefficient_series.csv、e4_fmb_results.csv | 102 个目标季 | Fig 16/17、Table 10/11 |
| 多空组合输出（E7） | experiments/E7/e7_portfolio.csv | 42 个目标季 | Fig 18 |

论文修订稿的组合净值图使用 `experiments/E7/e7_portfolio.csv` 的完整 E7 输出，由 `paper_v2/prepare_figures.py` 重绘。`图表/results/e7_portfolio.csv` 是早期图表使用的 5 列历史绘图摘录，仅保留信号组合的分腿/净收益和同池等权收益。
| 行业敏感度输出（E5） | experiments/E5/e5_heterogeneity.csv | 31 个行业 | Fig 19 |
| 稳健性输出（E8） | experiments/E8/e8_robustness_summary.csv | 49 个设定 | Fig 20 |

## 4. 质量核查

- 核查脚本：`图表/plotting_scripts/verify_figures_tables.py`，报告：`图表/logs/figure_table_qc.md`。
- 口径：收益计算顺序为**全样本计算日收益后按 2014-02-21 过滤**；年化 = 复利；波动 = 日标准差 × √252；夏普按无风险利率 0 计算（待补，见表注）。
- 结果：Table 4/5 与独立重算一致（28 行业，抽检对照见 qc 报告）；fig08-11 脚本均引用真实行情 CSV；未发现截断收益轴或选择性展示。
- Fig 13-20 与 Table 9-11 已由真实实验输出生成，核查脚本确认无模板占位、无空白图。

## 5. 复现方式

```bash
python 图表/plotting_scripts/fetch_data.py          # 抓取公开行情（已执行）
python 图表/plotting_scripts/build_experiment_inputs.py  # 由实验输出构造 Fig 13-20 输入
python 图表/plotting_scripts/build_tables.py        # 生成 图表/tables/（Table 1-11）
python 图表/plotting_scripts/fig01_*.py ... fig12_*.py   # 生成 图表/figures/ 已交付图
python 图表/plotting_scripts/plot_sentiment.py ... plot_robustness.py  # 生成 Fig 13-20
python 图表/plotting_scripts/verify_figures_tables.py      # 一致性核查
python 图表/plotting_scripts/compress_figures.py    # 生成压缩预览与总览
```

## 6. 附录：Fig 7 方法族关键词规则

来源：`图表/plotting_scripts/fig07_literature_methods.py` 的 `RULES`。单篇文献可计入多个方法族，**可多选、非互斥，合计大于 38**。

| 方法族 | 关键词（正则表达式） |
|--------|----------------------|
| 词典法与文本特征 | `词典|词频|情感单元|关键措辞|Harvard|Loughran|TF-IDF|可读性|相似度|措辞|语调` |
| 主题模型 | `LDA|主题模型|HDMR|rSCHOLAR|SCHOLAR` |
| 预训练模型 | `BERT|FinBERT|RoBERTa|Word2Vec|预训练` |
| 大语言模型 | `LLM|大模型|大语言模型|DeepSeek|Qwen|ChatGPT|GPT|通义` |
| 事件研究 | `事件研究|事件窗|CAR|异常收益` |
| 面板与回归 | `面板|固定效应|回归|Fama-MacBeth|OLS|中介效应|调节效应` |
| 结构模型与VAR | `SVAR|TVP-VAR|TVP-FAVAR|VAR|DSGE|脉冲响应|方差分解` |
| 波动率模型 | `GARCH|EGARCH|DCC|MIDAS` |
| 机器学习 | `随机森林|XGBoost|LightGBM|SVM|支持向量|逻辑回归|机器学习` |
| 深度学习 | `LSTM|BiLSTM|Informer|TCN|图神经|强化学习|DDPG|PPO|深度学习` |

*编制：紫璃 (Murasaki)　2026-11-20 更新*
