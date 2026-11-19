# E10 误差分析

- 生成时间：2026-11-19T23:47:44+08:00
- 性质：只读检查；本实验不删除、不填补、不重写任何 E0-E9 样本。
- 异常样本：`experiments/E10/e10_anomaly_samples.csv`，共 1002 条。
- 结论边界：异常不是结论的反证，而是技术故障、数据边界、模型弱点或需人工复核的证据。

## 1. 数据与方法

- 输入为工作区冻结数据：E1 文本与词典评分、行业/基准日行情、E3 事件 CAR、E4/E6/E8 预测、E7 组合、E8 失败与稳健性、E9 LLM 审计。
- 文本乱码检查使用替换字符、连续乱码串和 CJK 比例；低覆盖率取每个词典内后 10%。
- 预测残差使用 `actual - predicted`，Tukey 上栅栏为 Q3 + 1.5*IQR；正常样本为未越栅栏样本。
- 事件案例使用 E3 主口径 `market / est=120 / window=1`；对期层面行业均值 CAR 用同一 Tukey 规则抽异常，并另抽高情绪最负、低情绪最正的反向案例。
- 模型异常不删除；主样本 vs 异常样本按栅栏内/外分组对照。

## 2. 关键指标

- 行情：144429 行行业日行情，31 个行业；收益率绝对值 >=20% 共 0 条，零收益 0 条。
- 文本：103 份文本中，乱码或 CJK 比例异常 0 份；词典零覆盖 0 行，低覆盖抽样 33 行。
- LLM：E6 103 行评分，解析失败 0；E9 失败组 18 个、变体失败行 29 个。
- 残差：E4 主词典对照 2483 行、异常 120；E8 严格口径 1348 行、异常 57。
- E8 稳健性：limited 10 个；原始 p<0.05 但 BH/Holm 不显著 3 个；配对对照符号翻转 1 个。
- E7 组合：极端季度记录 6 条；因子窗口不完整 2 季。
- 失败：E8 失败记录 19876 条；E3 跳过单元 5851 个。

## 3. 异常分类

| 异常类型 | 数量 |
|---|---:|
| `market_missing_trading_day` | 521 |
| `prediction_residual_outlier` | 287 |
| `lexicon_low_coverage_decile` | 33 |
| `llm_variant_failure` | 29 |
| `e8_failure_record_class` | 22 |
| `llm_json_parsed_failure_group` | 18 |
| `classification_start_change` | 15 |
| `human_rating_pending` | 15 |
| `residual_concentrated_by_industry` | 14 |
| `robustness_limited_setting` | 10 |
| `e3_skip_cell_class` | 9 |
| `portfolio_extreme_quarter` | 6 |
| `reverse_case_high_sentiment_negative_car` | 5 |
| `reverse_case_low_sentiment_positive_car` | 5 |
| `event_mean_car_outlier` | 4 |
| `residual_concentrated_by_year` | 3 |
| `raw_significant_not_multiplicity_robust` | 3 |
| `portfolio_factor_window_incomplete` | 2 |
| `paired_variant_sign_flip` | 1 |

## 4. 状态分布

| 状态 | 数量 |
|---|---:|
| `unresolved_limitation` | 521 |
| `review` | 338 |
| `low_coverage_case` | 33 |
| `failure_class` | 22 |
| `residual_concentration` | 17 |
| `case_review` | 16 |
| `pending_human` | 15 |
| `classification_change` | 15 |
| `limited` | 10 |
| `skip_class` | 9 |
| `not_robust_after_multiplicity` | 3 |
| `limitation` | 2 |
| `sign_flip` | 1 |

## 5. 来源分布

| 来源实验 | 数量 |
|---|---:|
| `E0-E3` | 521 |
| `E4_main_vs_baseline` | 126 |
| `E6_test_main_models` | 115 |
| `E8_strict_main_vs_baseline` | 63 |
| `E9` | 47 |
| `E8` | 36 |
| `E1` | 33 |
| `E0-E8` | 15 |
| `E1-E6` | 15 |
| `E3` | 14 |
| `E2-E3` | 9 |
| `E7` | 8 |

## 6. 正常 vs 异常对照

| 模型 | 行数 | 异常数 | 异常率 | Tukey栅栏 | 正常组平均绝对残差 | 异常组平均绝对残差 |
|---|---:|---:|---:|---:|---:|---:|
| `E4_main_vs_baseline` | 2483 | 120 | 0.0483 | 0.243154511 | 0.078559875 | 0.301249995 |
| `E8_strict_main_vs_baseline` | 1348 | 57 | 0.0423 | 0.261900114 | 0.080416766 | 0.327389018 |
| `E6_test_main_models` | 2729 | 110 | 0.0403 | 0.246273756 | 0.077895108 | 0.320158537 |

## 7. 剔除规则一致性

- E10 自己没有新增剔除规则；只设置检验和抽样阈值。
- E3/E8 主事件样本因基准日历、估计窗历史、行业或情绪单元不足而剔除；E10 不把这些失败改写为 0 或删除记录。
- E4/E6/E8 预测样本的缺失 signal、beta、收益或完整季度保持缺失；配对比较只使用共同可识别行。
- E7 因子文件在 2026Q2/Q3 覆盖不完整时排除因子回归，但保留组合季度结果；E10 将其标记为限制。
- E1 人工评分仍为 pending；本实验不把未完成的人工一致性写成通过。

## 8. 无法解释与限制

- 申万一级行业指数起点不同：2014 年和 2021 年开始的分类/覆盖变化在 `classification_start_change` 中逐行业保留，但本数据无法还原历史成分股或完整回溯规则。
- E8 长估计窗与上一期事件天然重叠；30 日估计窗和净化估计窗只能缓解，不能消除同期宏观新闻污染。
- LLM 原始回答在部分 Prompt/模型/温度组合下解析失败，尤其 v1 高温和部分模型子样本；这不自动等于越界评分，也不等于情绪得分无敏感性。
- E6/E9 人工参考仍是未独立验证的临时对照，E1 正式人工评分未完成，因此文本度量结论仍需人工验收。
- 残差集中年份/行业是描述性诊断，不能推出因果机制；E10 不为了改善显著性而删除任何异常。

## 9. 可追溯性

- 每条异常样本的 `source_file` 指向原始冻结文件；`explanation` 和 `evidence` 保留关键字段。
- 中间摘要与输入哈希在 `_temp/E10/e10_metrics.json`。
- 复现：`python experiments/E10/run_e10.py`。

## 10. 完成条件核对

- 主要异常类型已分类：通过。
- 每个结论可追溯到数据或代码：通过。
- 无法解释的异常被标记为限制：通过。
