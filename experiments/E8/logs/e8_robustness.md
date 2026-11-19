# E8 稳健性检验

运行时间：2026-11-19T21:10:23.184472+08:00。代码版本：`acbc6bc87f8683cb6d03885e4d3f24a150910a62`。

## 1. 结论与解释边界

本次结果不能支持“更积极的 MPR 情绪稳定提高行业收益或预测能力”。事件研究的高低组差异对事件窗、词典和正常收益模型敏感；原始显著结果须同时查看多重检验校正。严格修正预测收益与训练标签时点后，主词典仍没有稳定的样本外增量。未拒绝零假设不等于证明不存在经济效应。

- E3 没有确认唯一主事件窗；本实验以 [0,+5] 为展示/配对锚点，并精确复现全部四窗。该选择不是新增预注册，也不按显著性挑主结果。
- E4 原预测目标为报告期下一季度，该季度已在报告发布前开始。旧口径仅作复现和敏感性诊断，不能作为严格的前瞻预测证据。
- 严格口径改用 MPR 发布后的首个完整季度，并重新估计模型，只允许预测时已经实现的历史标签进入训练。目标与训练成熟时间一起修正，因此单列为新的时点修正参考设定。
- 以上均为关联和预测检验，不构成因果结论或可交易策略。E5 的行业差异、E6 的 LLM 结论和 E7 的组合表现不因 E8 自动升级；LLM Prompt/模型敏感性属于 E9。

## 2. 基准与单项变化规则

事件参考：NTUSD 简体词典、沪深300、市场模型、[-120,-1] 估计窗、[0,+5] 事件窗、全部可用行业、原始 CAR、沿用 E3 全103期三分位。每个事件先对行业 CAR 等权平均，再在事件层面检验；不用行业行数冒充独立时间样本数。

预测参考：截距 + mom1_z + vol_q_z + beta_120_z + 4 个行业板块虚拟变量；文本模型额外加入 trailing-z 情绪 × 标准化行业情绪暴露。每季截面 OLS，扩展窗平均历史系数，最低 8 个已拟合季度，测试目标季度固定 2021Q1–2026Q3。共同情绪本身被截距吸收，不能直接用于行业排序。

除显式时点修正外，每个设定改变一个研究维度，其他设定继承对应参考。lag2 是整套预先可得信息向量再滞后一报告季度（情绪交互和三项连续控制一起平移，行业板块不变）；不使用 E4 与 mom1 重复的 mom2 冒充真实 lag2。Winsor 是异常值处理规则这一维度，包含明确指定的特征/拟合标签处理。

共同样本：事件按 issue×code 交集重算变体和参考；不同基准不能把 2002 起的沪深300样本与 2011 起的中证全指样本直接相减。预测在共同面板行重新拟合无文本及 NTUSD 参考，再按共同预测键计分。`estimate` 为各设定全可用样本结果，公平比较使用 `paired_*`，不要混用。

## 3. 输入、数据范围与复现

| 文件 | 字节 | 用途 |
|---|---|---|
| EXPERIMENT_PLAN.md | 17928 | input_or_reproduction_reference |
| METHOD_PLAN.md | 30461 | input_or_reproduction_reference |
| experiments/E1/mpr_index_full.csv | 42854 | input_or_reproduction_reference |
| experiments/E1/sentiment_dictionary_full.csv | 159054 | input_or_reproduction_reference |
| data/external/sw_industries/sw_industry_daily.csv | 12634203 | input_or_reproduction_reference |
| data/external/benchmark/sh000985_中证全指.csv | 155601 | input_or_reproduction_reference |
| data/external/benchmark/sh000300_沪深300.csv | 206410 | input_or_reproduction_reference |
| _temp/e0_market_raw/benchmark_sh000300.csv | 468903 | input_or_reproduction_reference |
| _temp/e4/e4_panel.csv | 2469665 | input_or_reproduction_reference |
| _temp/e4/industry_quarterly.csv | 319557 | input_or_reproduction_reference |
| experiments/E3/e3_event_results.csv | 3389664 | input_or_reproduction_reference |
| experiments/E3/e3_car_by_sentiment.csv | 14283 | input_or_reproduction_reference |
| experiments/E3/run_e3.py | 73203 | input_or_reproduction_reference |
| experiments/E4/run_e4.py | 58169 | input_or_reproduction_reference |
| experiments/E4/e4_fmb_results.csv | 10205 | input_or_reproduction_reference |
| experiments/E4/e4_oos_metrics.csv | 3844 | input_or_reproduction_reference |
| experiments/E5/e5_metrics.json | 797 | context_only |
| experiments/E6/e6_metrics.json | 999 | context_only |
| experiments/E7/e7_metrics.json | 360 | context_only |
| experiments/E7/logs/e7_portfolio.md | 8228 | context_only |

完整 SHA256 见 `e8_input_manifest.csv`。行情复用已冻结的数据，不为某个设定临时切换下载版本；互联网只核验方法文档。

中证全指文件只有 2011 年起的行情；不拼接其他指数补造早期历史。不同交易日历及缺失行业单元均明示剔除。当前行业分类历史回溯、幸存者偏差及非实时数据修订风险仍然存在。

### 3.1 模块审计记录

```json
{
  "event": {
    "n_settings": 23,
    "n_mpr_reports": 103,
    "n_industries": 31,
    "n_return_observations": 20285,
    "n_failed_return_records": 15031,
    "n_failed_return_cells": 18031,
    "n_setting_inference_warnings": 4,
    "n_missing_sentiment_cells": 32,
    "status_counts": {
      "ok": 21,
      "limited": 2
    },
    "benchmark_sources": {
      "sh000300": "_temp/e0_market_raw/benchmark_sh000300.csv",
      "sh000985": "data/external/benchmark/sh000985_中证全指.csv"
    },
    "benchmark_coverage": {
      "sh000300": {
        "start": "2002-01-04",
        "end": "2026-11-18",
        "n_rows": 6033
      },
      "sh000985": {
        "start": "2011-08-02",
        "end": "2026-11-17",
        "n_rows": 3713
      }
    },
    "e3_reproduction": {
      "status": "passed",
      "n_industry_event_windows": 6920,
      "max_absolute_car_difference": 9.974659986866641e-17
    },
    "group_definition": "E3 full-corpus score ranking: low ranks < N/3, middle < 2N/3, remaining high; all 103 reports for single dictionaries",
    "subsample_definition": "MPR publication date < Jan 1 of cut year versus >= Jan 1; original full-corpus labels retained",
    "paired_definition": "exact common issue-code observations; average industries inside each issue, then recompute both event-level contrasts",
    "continuous_definition": "HC3 OLS of event-mean CAR on sentiment standardized by analytic-sample SD; descriptive, no OOS claim",
    "display_anchor": "E3 reports all four event windows; [0,+5] is only the E8 comparison/display anchor, not a newly preregistered primary outcome",
    "failure_count_definition": "failure records with code ALL cover every requested industry; n_failed_return_cells expands ALL by the 31-industry universe",
    "failure_file": "_temp/e8/event/failures.csv",
    "setting_failure_file": "_temp/e8/event/setting_failures.csv",
    "sentiment_exclusion_file": "_temp/e8/event/sentiment_exclusions.csv",
    "prior_overlap_note": "trading-date alignment, unlike E3 publication-calendar-date check; purged removes all prior [0,+10] event days"
  },
  "prediction": {
    "input_panel": "_temp\\e4\\e4_panel.csv",
    "input_quarterly": "_temp\\e4\\industry_quarterly.csv",
    "cache": "_temp\\e8\\prediction",
    "n_settings": 26,
    "legacy_prepublication_rows": 2274,
    "legacy_rows": 2274,
    "legacy_issues": 102,
    "strict_rows": 2243,
    "strict_issues": 101,
    "strict_forecast_rows": 19707,
    "strict_time_violations": 0,
    "e4_main_coefficient_delta": -1.491862189340054e-16,
    "e4_main_oos_deltas": {
      "ic_mean": -8.673617379884035e-17,
      "oos_r2": -8.881784197001252e-16,
      "mae": 4.163336342344337e-17
    },
    "failure_records": 4809,
    "strict_incomplete_target_rows": 31,
    "oos_r2_definition": "1-SSE/SST around realized quarterly cross-sectional mean; diagnostic denominator is ex-post, not a feasible forecast",
    "limitations": [
      "Legacy target overlaps returns before publication and is not valid prospective OOS evidence.",
      "Strict model changes label timing and realization-aware training together; it is a separate timing-corrected reference.",
      "Industry composition and historical index backfill remain inherited data limitations.",
      "Subsample rows exclude the cutoff year from before and include it in after; short or absent OOS is explicit."
    ]
  }
}
```

## 4. 推断、缩尾与多重检验

- 事件高减低：事件均值上的 Welch 不等方差检验和 95% CI；连续得分：事件 CAR 对标准化情绪的 HC3 回归。连续斜率和高低差的单位不同，不能比较数值大小。Welch 假设两组事件独立，时间相关仍是局限；连续 HC3 也不等于 HAC。
- 预测 FMB：季度系数均值的 Newey-West/Bartlett lag=2 推断；扩展历史系数预测。稀疏子样本中的“lag”按可用季度观测序列计算，非严格等间隔的日历 HAC。
- OOS R² = 1−SSE/每季横截面实现收益离差平方和；这个分母是事后诊断基准，不是可实施预测。额外报告与同样本重新拟合的无文本预测比较。IC 为季度截面 Spearman 相关均值。
- 事件缩尾：行业 CAR 合并分布 1%/99% 截断，属于描述性全样本敏感性，不作 OOS 宣称。预测缩尾：当季截面特征截断，拟合标签阈值只从该预测原点之前已实现标签估计；测试实际收益保持原始值。
- 原始 p 全保留。主效应分别在本报告定义的 event、prediction legacy、prediction strict 三个汇总块内对全部有限 p 做 BH 与 Holm；全样本均值 CAR 是另一个明确标记的次要检验族。参考行、低功效行均计入，未事后删除不显著设定。检验族没有单独预注册。
- 重叠稳健性设定相关性强。BH 是探索性 FDR 描述，任意依赖下不保证控制；Holm 用于不依赖独立性的更保守判断。全部设定是事后稳健性分析，不能改写为独立验证样本或预注册发现。

## 5. 事件研究完整结果

| 设定 | 状态 | 期数 | 高 | 低 | 效应 | 原始p | BH p | Holm p | 配对参考 | 配对变体 | 上一期日重叠占比 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| event_main | ok | 89 | 29 | 30 | -0.00584 | 0.25034 | 0.68930 | 1.00000 | -0.00584 | -0.00584 | 1.00000 |
| event_dictionary_dlut | ok | 89 | 27 | 30 | -0.00445 | 0.36663 | 0.68930 | 1.00000 | -0.00584 | -0.00445 | 1.00000 |
| event_dictionary_mpr_domain | ok | 89 | 28 | 34 | -0.00268 | 0.61970 | 0.73690 | 1.00000 | -0.00584 | -0.00268 | 1.00000 |
| event_dictionary_combo_ntusd_dlut | ok | 87 | 26 | 31 | -0.00346 | 0.48654 | 0.69940 | 1.00000 | -0.00603 | -0.00346 | 1.00000 |
| event_window_1 | ok | 89 | 29 | 30 | -0.00609 | 0.01133 | 0.26050 | 0.26050 | -0.00584 | -0.00609 | 1.00000 |
| event_window_3 | ok | 89 | 29 | 30 | -0.00262 | 0.57512 | 0.73690 | 1.00000 | -0.00584 | -0.00262 | 1.00000 |
| event_window_10 | ok | 89 | 29 | 30 | -0.00344 | 0.64078 | 0.73690 | 1.00000 | -0.00584 | -0.00344 | 1.00000 |
| event_pre2008 | ok | 22 | 6 | 5 | -0.00805 | 0.40464 | 0.68930 | 1.00000 | -0.00805 | -0.00805 | 1.00000 |
| event_post2008 | ok | 67 | 23 | 25 | -0.00538 | 0.35736 | 0.68930 | 1.00000 | -0.00538 | -0.00538 | 1.00000 |
| event_pre2015 | ok | 50 | 7 | 26 | -0.00257 | 0.77204 | 0.80714 | 1.00000 | -0.00257 | -0.00257 | 1.00000 |
| event_post2015 | limited | 39 | 22 | 4 | -0.01182 | 0.24283 | 0.68930 | 1.00000 | -0.01182 | -0.01182 | 1.00000 |
| event_pre2020 | ok | 66 | 10 | 28 | -0.00563 | 0.41957 | 0.68930 | 1.00000 | -0.00563 | -0.00563 | 1.00000 |
| event_post2020 | limited | 23 | 19 | 2 | -0.00077 | 0.83351 | 0.83351 | 1.00000 | -0.00077 | -0.00077 | 1.00000 |
| event_benchmark_allshare | ok | 51 | 23 | 11 | -0.00763 | 0.07944 | 0.68930 | 1.00000 | -0.01081 | -0.00763 | 1.00000 |
| event_winsor_1pct | ok | 89 | 29 | 30 | -0.00605 | 0.22147 | 0.68930 | 1.00000 | -0.00584 | -0.00605 | 1.00000 |
| event_estimation_30 | ok | 97 | 32 | 32 | -0.00222 | 0.67960 | 0.74432 | 1.00000 | -0.00584 | -0.00409 | 0.00000 |
| event_estimation_60 | ok | 94 | 31 | 32 | -0.00266 | 0.60242 | 0.73690 | 1.00000 | -0.00584 | -0.00425 | 0.48885 |
| event_estimation_90 | ok | 92 | 30 | 31 | -0.00492 | 0.34193 | 0.68930 | 1.00000 | -0.00584 | -0.00633 | 1.00000 |
| event_estimation_180 | ok | 85 | 27 | 28 | -0.00743 | 0.17664 | 0.68930 | 1.00000 | -0.00765 | -0.00743 | 1.00000 |
| event_estimation_250 | ok | 80 | 24 | 28 | -0.00683 | 0.23444 | 0.68930 | 1.00000 | -0.00729 | -0.00683 | 1.00000 |
| event_estimation_purged | ok | 89 | 29 | 30 | -0.00574 | 0.24676 | 0.68930 | 1.00000 | -0.00584 | -0.00584 | 0.00000 |
| event_continuous | ok | 89 | 29 | 30 | -0.00216 | 0.32770 | 0.68930 | 1.00000 | -0.00216 | -0.00216 | 1.00000 |
| event_mean_adjusted | ok | 89 | 29 | 30 | 0.00715 | 0.48500 | 0.69940 | 1.00000 | -0.00584 | 0.00715 | 1.00000 |

后期子样本仍沿用 E3 全样本分组，不为了增加显著性重新划分。高/低任一组少于 5 期标记 limited；这不是失败归零，更不能据不显著写成稳定无效。全样本分组只用于描述性事件比较，不能拿去构建历史实时信号。

## 6. 预测完整结果

| 设定 | 状态 | 拟合期数 | 测试期数 | FMB均值 | NW p | BH p | Holm p | IC | OOS R² | 相对无文本ΔR² | MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| prediction_main | ok | 95 | 23 | -0.01197 | 0.62001 | 0.73875 | 1.00000 | -0.01691 | -0.60415 | -0.03328 | 0.08734 |
| prediction_dlut | ok | 95 | 23 | 0.04654 | 0.03099 | 0.40291 | 0.40291 | -0.02737 | -1.20186 | -0.63099 | 0.10375 |
| prediction_mpr_domain | ok | 95 | 23 | -0.02030 | 0.61730 | 0.73875 | 1.00000 | -0.09704 | -0.61235 | -0.04149 | 0.08746 |
| prediction_combo | ok | 95 | 23 | -0.00322 | 0.86372 | 0.86372 | 1.00000 | 0.01756 | -0.56221 | 0.00865 | 0.08588 |
| prediction_lag2 | ok | 94 | 23 | -0.03304 | 0.18357 | 0.73875 | 1.00000 | -0.01037 | -0.90299 | -0.33500 | 0.09562 |
| prediction_tercile | limited | 67 | 23 | -0.00134 | 0.65371 | 0.73875 | 1.00000 | -0.08757 | -0.54899 | -0.00275 | 0.08467 |
| prediction_winsor | ok | 95 | 23 | -0.01056 | 0.68193 | 0.73875 | 1.00000 | -0.02400 | -0.58665 | -0.02104 | 0.08682 |
| prediction_before2008 | limited | 20 | 0 | 0.01663 | 0.68106 | 0.73875 | 1.00000 | NA | NA | NA | NA |
| prediction_after2008 | ok | 75 | 23 | -0.01959 | 0.48727 | 0.73875 | 1.00000 | -0.01802 | -0.63996 | -0.10000 | 0.08809 |
| prediction_before2015 | limited | 48 | 0 | -0.03336 | 0.45566 | 0.73875 | 1.00000 | NA | NA | NA | NA |
| prediction_after2015 | ok | 47 | 23 | 0.00988 | 0.37461 | 0.73875 | 1.00000 | -0.08072 | -0.62596 | -0.04782 | 0.08710 |
| prediction_before2020 | limited | 68 | 0 | -0.01528 | 0.64991 | 0.73875 | 1.00000 | NA | NA | NA | NA |
| prediction_after2020 | ok | 27 | 19 | -0.00363 | 0.38911 | 0.73875 | 1.00000 | -0.07784 | -0.77273 | -0.00846 | 0.08951 |
| prediction_strict_main | ok | 94 | 23 | -0.03304 | 0.18357 | 0.39328 | 1.00000 | -0.00743 | -0.90389 | -0.34104 | 0.09567 |
| prediction_strict_dlut | ok | 94 | 23 | -0.06470 | 0.26377 | 0.39328 | 1.00000 | -0.01354 | -1.68370 | -1.12085 | 0.11419 |
| prediction_strict_mpr_domain | ok | 94 | 23 | -0.03758 | 0.42511 | 0.50240 | 1.00000 | -0.03592 | -0.72769 | -0.16484 | 0.09199 |
| prediction_strict_combo | ok | 94 | 23 | -0.06282 | 0.07377 | 0.35962 | 0.88528 | 0.00150 | -1.40961 | -0.84676 | 0.10848 |
| prediction_strict_lag2 | ok | 93 | 23 | -0.04439 | 0.01153 | 0.14987 | 0.14987 | 0.02131 | -1.05014 | -0.48036 | 0.09948 |
| prediction_strict_tercile | limited | 66 | 23 | -0.00076 | 0.84565 | 0.84565 | 1.00000 | -0.01714 | -0.60907 | -0.00024 | 0.08779 |
| prediction_strict_winsor | ok | 94 | 23 | -0.02830 | 0.27227 | 0.39328 | 1.00000 | -0.00612 | -0.78607 | -0.23176 | 0.09267 |
| prediction_strict_before2008 | limited | 20 | 0 | -0.04226 | 0.08299 | 0.35962 | 0.91289 | NA | NA | NA | NA |
| prediction_strict_after2008 | ok | 74 | 23 | -0.03054 | 0.32027 | 0.41635 | 1.00000 | -0.01251 | -0.85023 | -0.31294 | 0.09426 |
| prediction_strict_before2015 | limited | 48 | 0 | -0.07014 | 0.12671 | 0.39328 | 1.00000 | NA | NA | NA | NA |
| prediction_strict_after2015 | ok | 46 | 23 | 0.00568 | 0.57837 | 0.62657 | 1.00000 | 0.03909 | -0.55554 | -0.02251 | 0.08487 |
| prediction_strict_before2020 | limited | 68 | 0 | -0.04158 | 0.22096 | 0.39328 | 1.00000 | NA | NA | NA | NA |
| prediction_strict_after2020 | ok | 26 | 17 | -0.01069 | 0.24233 | 0.39328 | 1.00000 | -0.00453 | -0.74108 | -0.08840 | 0.08999 |

早期子样本不覆盖 2021 年后的固定测试期，OOS 指标为 NA 并标 limited；仍报告其可计算 FMB 系数。不把空结果填零。三分组中组编码为0，使共同情绪交互列为0时截面秩不足，该季显式记录未拟合；三角色均限定到共同可识别季度拟合与训练。其结果是尾部情绪条件估计，标记 limited，不代表完整三组模型得到全样本验证，也不能只比较剩余季度的显著性。

## 7. 稳健与脆弱的判断

- 事件主效应：23 个设定，原始 p<0.05 共 1 个；BH 后 0 个；Holm 后 0 个。低功效/失败设定保留在表内。
- 在市场模型内，NTUSD 的四个事件窗，以及 [0,+5] 下三种词典和等权组合的高减低 CAR 点估计均为负；该负向点估计方向较一致，显著性不稳健。这是单项变化结果，并未运行全部词典×窗口组合。改用均值调整模型后差值转正且不显著，故负向关联也不能跨正常收益模型概括。只在个别短窗出现的显著关联属于脆弱、探索性证据。
- 估计窗 90/120/180/250 日包含上一期事件，不能据长窗间相似就宣称排除污染。30日和净化估计观测是更直接的诊断；净化仍不能控制所有同期宏观新闻。
- 词典选择可改变 FMB 系数方向；旧口径单一词典的显著系数不等于严格时点下可预测。严格主设定和各变体需同时查看IC、相对无文本ΔR²和MAE，不能只选系数p。
- 旧口径等权词典组合的 IC、OOS R²、MAE 点估计略优于无文本参考，但时点修正后其增量 R²转负。该孤立改善不能升级为稳定预测证据。
- 可保留的结论强度：当前数据和模型未给出跨词典、事件窗和时点协议一致的稳定正向证据。局部显著性、子样本无显著性与主结论均按有限样本描述，禁止推广为政策因果或交易收益保证。

## 8. 失败、缺失与每设定记录

总计保留 19876 条底层失败/低功效记录，见 `e8_failures.csv`。数据不足、缺失日期、秩不足、训练期不足与原点时间约束分别保存，没有用0替换失败收益。`code=ALL` 代表一个请求期对全部31行业失败；不要把原始记录数等同于展开后的行业单元数。

`ok` 只表示该设定计算成功且达到本报告最低样本条件，不表示效应显著或结论稳健；`limited` 表示样本功效、测试覆盖或条件识别受限；`failed` 才表示未能估计核心结果。

| 设定 | 状态 | 推断/测试限制 | 事件底层失败记录 | 预测拟合/预测失败记录 |
|---|---|---|---|---|
| event_main | ok |  | 1313 | NA |
| event_dictionary_dlut | ok |  | 1313 | NA |
| event_dictionary_mpr_domain | ok |  | 1313 | NA |
| event_dictionary_combo_ntusd_dlut | ok |  | 1313 | NA |
| event_window_1 | ok |  | 1313 | NA |
| event_window_3 | ok |  | 1313 | NA |
| event_window_10 | ok |  | 1283 | NA |
| event_pre2008 | ok |  | 335 | NA |
| event_post2008 | ok |  | 978 | NA |
| event_pre2015 | ok |  | 743 | NA |
| event_post2015 | limited | fewer_than_5_events_in_high_or_low_group | 570 | NA |
| event_pre2020 | ok |  | 1050 | NA |
| event_post2020 | limited | fewer_than_5_events_in_high_or_low_group | 263 | NA |
| event_benchmark_allshare | ok |  | 751 | NA |
| event_winsor_1pct | ok |  | 1313 | NA |
| event_estimation_30 | ok |  | 1083 | NA |
| event_estimation_60 | ok |  | 1144 | NA |
| event_estimation_90 | ok |  | 1242 | NA |
| event_estimation_180 | ok |  | 1428 | NA |
| event_estimation_250 | ok |  | 1547 | NA |
| event_estimation_purged | ok |  | 1301 | NA |
| event_continuous | ok |  | 1313 | NA |
| event_mean_adjusted | ok |  | 1313 | NA |
| prediction_main | ok |  | NA | 224 |
| prediction_dlut | ok |  | NA | 224 |
| prediction_mpr_domain | ok |  | NA | 224 |
| prediction_combo | ok |  | NA | 224 |
| prediction_lag2 | ok |  | NA | 256 |
| prediction_tercile | limited | ordinal_middle_group_unidentified_conditional_tails_only | NA | 322 |
| prediction_winsor | ok |  | NA | 224 |
| prediction_before2008 | limited | no_eligible_2021Q1_2026Q3_test_periods | NA | 119 |
| prediction_after2008 | ok |  | NA | 105 |
| prediction_before2015 | limited | no_eligible_2021Q1_2026Q3_test_periods | NA | 143 |
| prediction_after2015 | ok |  | NA | 81 |
| prediction_before2020 | limited | no_eligible_2021Q1_2026Q3_test_periods | NA | 203 |
| prediction_after2020 | ok |  | NA | 33 |
| prediction_strict_main | ok |  | NA | 224 |
| prediction_strict_dlut | ok |  | NA | 224 |
| prediction_strict_mpr_domain | ok |  | NA | 224 |
| prediction_strict_combo | ok |  | NA | 224 |
| prediction_strict_lag2 | ok |  | NA | 256 |
| prediction_strict_tercile | limited | ordinal_middle_group_unidentified_conditional_tails_only | NA | 322 |
| prediction_strict_winsor | ok |  | NA | 224 |
| prediction_strict_before2008 | limited | no_eligible_2021Q1_2026Q3_test_periods | NA | 119 |
| prediction_strict_after2008 | ok |  | NA | 105 |
| prediction_strict_before2015 | limited | no_eligible_2021Q1_2026Q3_test_periods | NA | 143 |
| prediction_strict_after2015 | ok |  | NA | 81 |
| prediction_strict_before2020 | limited | no_eligible_2021Q1_2026Q3_test_periods | NA | 203 |
| prediction_strict_after2020 | ok |  | NA | 39 |

### 8.1 失败原因汇总

| 模块 | 原因 | 记录数 |
|---|---|---|
| event | fewer_than_5_events_in_high_or_low_group | 4 |
| event | incomplete_event_window | 2 |
| event | insufficient_benchmark_estimation_history | 24 |
| event | missing_industry_estimation_returns | 14931 |
| event | publication_outside_benchmark_calendar | 74 |
| event | sentiment_unavailable_or_trailing_z_warmup | 32 |
| prediction | excluded_to_match_all_three_roles_identifiable_quarters | 108 |
| prediction | first_complete_postpublication_quarter_unavailable_or_incomplete_asof_2026_11_19 | 31 |
| prediction | missing_or_nonfinite:beta_120_z,signal,primary_signal | 1280 |
| prediction | missing_or_nonfinite:mom1_z,vol_q_z,beta_120_z,signal,primary_signal | 62 |
| prediction | missing_or_nonfinite:signal | 32 |
| prediction | missing_or_nonfinite:signal,primary_signal | 3060 |
| prediction | no_complete_predictor_outcome_rows | 144 |
| prediction | no_eligible_2021Q1_2026Q3_test_periods | 6 |
| prediction | ordinal_middle_group_unidentified_conditional_tails_only | 2 |
| prediction | rank_8_lt_9 | 54 |
| prediction | train_periods_2_lt_8 | 3 |
| prediction | train_periods_3_lt_8 | 3 |
| prediction | train_periods_4_lt_8 | 6 |
| prediction | train_periods_5_lt_8 | 6 |
| prediction | train_periods_6_lt_8 | 6 |
| prediction | train_periods_7_lt_8 | 6 |

## 9. 互联网方法核验

以下是官方软件/方法文档。它们支持计算选择，不是本项目实证结果的外部证据；数据结论完全来自上述冻结输入。原始网页、抓取时间和哈希保存在 `_temp/e8/sources/`，来源索引见 `e8_sources.csv`。

- statsmodels cov_hac: Newey-West covariance：https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html；状态 `verified`。用途：HAC inference and the lag parameter; observations must be chronologically ordered。
- statsmodels multipletests: Holm and Benjamini-Hochberg：https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html；状态 `verified`。用途：Retain raw p; report blockwise BH and dependence-robust Holm adjustments。
- SciPy ttest_ind: independent samples and Welch test：https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html；状态 `verified`。用途：High-minus-low differences use event means and unequal-variance inference。
- scikit-learn TimeSeriesSplit: time-ordered validation：https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html；状态 `verified`。用途：Chronological expanding evaluation; E8 additionally purges unrealized labels。

## 10. 复现、环境与验收

```powershell
python -B experiments/E8/run_e8.py
python -B -m unittest discover -s experiments/E8 -p 'test_e8*.py' -v
```

仅重建图表和总日志（沿用本次缓存结果）：`python -B experiments/E8/run_e8.py --from-cache`。
若 E4 输入缓存缺失，先按 E4 原脚本重建并核对输入 manifest；E8 不静默下载替代数据或覆盖既有实验结果。

完整运行元数据见 `e8_metrics.json`；所有下载页、中间面板、系数、预测和 matplotlib/Python 缓存在 `_temp/e8/`。

```json
{
  "run_at": "2026-11-19T21:10:23.184472+08:00",
  "as_of_date": "2026-11-19",
  "cache_reuse": false,
  "git_commit": "acbc6bc87f8683cb6d03885e4d3f24a150910a62",
  "python": "3.14.0",
  "platform": "Windows-11-10.0.26200-SP0",
  "pandas": "3.0.5",
  "numpy": "2.5.2",
  "scipy": "1.18.1",
  "matplotlib": "3.11.2",
  "seed": 20261119,
  "n_settings": 49,
  "status_counts": {
    "ok": 39,
    "limited": 10
  },
  "failure_records": 19876,
  "sources_verified": 4,
  "raw_significant": 3,
  "bh_significant": 0,
  "holm_significant": 0,
  "script_sha256": {
    "e8_event.py": "8fd1fb962a9970bee03d22663ea4d21fe96e46d04ca61f959a4f2cb8cdd7e4c0",
    "e8_prediction.py": "667ed4dea9c383c790afd725cb7d06014a5eb9612424048716d9f9ec5fcb5b16",
    "run_e8.py": "4c4bb712e7d69fd7b7ba73f3e6c78125605e4d37e5e671a5521512959b37035c",
    "test_e8_event.py": "9e3239612e92de2a4b243adeea2c01a0c5b87a82d46d744fb8239bef36011ed2",
    "test_e8_prediction.py": "f9e3f4f4a5137c4a094a4d87e8e6dd837db4cef1e9e7c6203f416a335a86bcb5",
    "test_e8_report.py": "4e65636d80dbfa1fb9ab7c6030868264b6fa0d4b7a57742e0f3dff13df1a0239"
  },
  "no_existing_input_modified": true,
  "primary_effect_p_families": {
    "event_all_settings": 23,
    "prediction_legacy_all_settings": 13,
    "prediction_strict_all_settings": 13
  }
}
```
