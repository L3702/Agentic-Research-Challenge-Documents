# Table 3　实验设计总表（E0-E10）

来源：`EXPERIMENT_PLAN.md` 第 0 节总览表与各实验条目（2026-11-17 规划稿，非实验结果）。

| 编号 | 阶段 | 实验目的 | 输入数据 | 对照设置 | 评价指标 | 预期输出 | 优先级 |
|------|------|------|------|------|------|------|------|
| E0 | MVP | 跑通数据链路：MPR PDF、行业指数、发布日对齐 | 3-5 期 MPR PDF、3 个行业指数、1 个基准 | 抓取失败/字段缺失/日期错位等数据检查项 | PDF 成功率、文本字符数、缺失率、日期对齐成功率 | data_sample/、logs/e0_data_pipeline.md | 必须 |
| E1 | MVP | 跑通词典情绪度量与人工抽检 Baseline | MPR 纯文本样本（尽可能多期） | 不同词典一致性；自动评分 vs 人工评分 | 词典间相关、人工-自动相关、词覆盖率 | sentiment_dictionary_full.csv、validation/human_ratings.csv | 必须 |
| E2 | MVP | 跑通事件研究 Baseline 与无文本对照 | 对齐后行业收益、市场基准、无风险利率 | 市场模型 vs 均值调整；全样本 vs 中性组 | CAR 均值、t 值、符号检验、Bootstrap CI | e2_event_baseline.csv、e2_car_plot.png | 必须 |
| E3 | 正式 | 检验 MPR 情绪高低组的事件窗 CAR 差异 | 全样本情绪得分、行业日收益、基准、无风险利率 | 高 vs 低情绪组；市场模型 vs 均值调整；全样本 vs 子样本 | 组间 CAR 差异、t 值、CI、事件窗一致性 | e3_event_results.csv、e3_car_by_sentiment.csv | 必须 |
| E4 | 正式 | 检验情绪对行业收益的横截面预测力 | 行业季度收益、情绪得分、市场收益、滞后收益、波动率、宏观控制 | 无文本模型 vs 词典情绪模型（可选 LLM） | 横截面系数均值/t、IC、ICIR、OOS R²、MAE | e4_fmb_results.csv、e4_oos_metrics.csv | 必须 |
| E5 | 正式 | 检验行业敏感度异质性 | E3/E4 结果、行业分类、行业特征 | 敏感组 vs 非敏感组；交互模型 vs 无交互 | 交互项系数、组间敏感度差异、组内 IC | e5_heterogeneity.csv、e5_sensitivity_plot.png | 必须 |
| E6 | 正式 | 比较 LLM 与词典情绪的样本外表现 | 同一批 MPR 文本、同一收益面板、同一划分 | 词典 vs LLM vs 词典+LLM | LLM-人工一致性、IC/ICIR/OOS R²、增量回归系数、成本 | e6_llm_vs_dict.csv、e6_llm_oos_metrics.csv | 可选增强 |
| E7 | 正式 | 构建高低情绪多空组合并评估风险调整表现 | 全样本行业收益、情绪得分、基准、无风险利率 | 对称多空；随机分组（可选）；基准 | 年化收益/波动、夏普、最大回撤、胜率、换手、alpha | e7_portfolio.csv、e7_equity_curve.png | 可选增强 |
| E8 | 补充 | 稳健性：词典、事件窗、子样本、基准、滞后阶数 | 主实验全部数据与中间结果 | 主设定 vs 每个稳健性设定 | 系数方向、显著性、IC、OOS R²、CAR 方向 | e8_robustness_summary.csv | 必须（核心项） |
| E9 | 补充 | 敏感性：LLM Prompt/模型/解析规则/事件日 | E6 的 LLM 原始输出与人工抽检 | 基准 Prompt+模型 vs 各替代设定 | 人工一致性、评分相关、IC、OOS R²、解析失败率、成本 | e9_sensitivity_summary.csv | 可选增强 |
| E10 | 补充 | 误差分析：数据、文本、模型残差与失败案例 | E0-E9 失败日志、异常值、残差、低覆盖文本、污染记录 | 正常样本 vs 异常样本；主模型 vs Baseline | 异常率、缺失率、残差分布、失败案例数、剔除规则一致性 | e10_error_analysis.md、e10_anomaly_samples.csv | 可选增强 |
