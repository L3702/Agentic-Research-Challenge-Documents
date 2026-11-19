# E4 横截面预测与 Fama-MacBeth 实验日志

## 1. 实验目的与假设

- 检验 H2 的预测部分：MPR 情绪是否对下一期行业收益有横截面预测力，是否优于无文本 Baseline。
- 输入：行业季度收益、MPR 词典情绪、市场收益、滞后行业收益、波动率、宏观控制变量。
- 主词典：`ntusd_simplified`；`dlut` 与 `mpr_domain` 为稳健性对照。

## 2. 识别口径

- MPR 情绪是季度层面共同变量。若横截面回归含截距，共同情绪与截距完全共线，不能识别“情绪高低行业”的横截面差异。
- 共同宏观变量同样与截距共线，不能直接放入单期横截面回归。宏观控制改放在行业面板稳健性中，并另报共同宏观暴露诊断。
- FMB 主模型使用滚动估计的行业情绪暴露 `sent_beta`：仅用报告期及以前季度估计行业收益对 MPR 情绪的敏感度，再与当季 `sentiment_z` 相乘。该信号在截面上有行业差异，且不作为主效应进入 Baseline。
- 无文本 Baseline 使用滞后行业收益、波动率、滚动市场 beta、行业板块；情绪模型在同一 Baseline 上增加 `sentiment_z × sent_beta_z`。
- 面板稳健性分别报告行业固定效应与时间固定效应；时间固定效应会吸收共同情绪，但情绪与行业特征的交互项仍可识别。

## 3. 数据与样本

- 行业日行情：`data/external/sw_industries/sw_industry_daily.csv`，144429 行，31 个申万一级行业。
- 市场基准：workspace_cache:_temp/e0_market_raw/benchmark_sh000300.csv。
- MPR 指数与情绪：`experiments/E1/mpr_index_full.csv`、`experiments/E1/sentiment_dictionary_full.csv`，共 103 期。
- 宏观数据：akshare 公开接口，缓存于 `_temp/e4/macro_raw/`；按 MPR 发布日“最近可得值”对齐。
- 目标收益：MPR 报告期下一季度行业对数收益；要求目标季和报告期季均至少 40 个交易日。
- 预测样本：2001Q2 至 2026Q3，共 102 个目标季；行业截面数范围 16 至 31。
- 时间划分：训练 2001Q2–2015Q4，验证 2016Q1–2020Q4，测试 2021Q1–2026Q3。

## 4. 模型设定

- `baseline_notext`：截距 + `mom1_z` + `vol_q_z` + `beta_120_z` + 行业板块虚拟变量。
- `sent_exposure`：Baseline + `sentiment_z × sent_beta_z`。`sent_beta_z` 是滚动 20 个季度估计的行业收益对 MPR 情绪的暴露，再做截面标准化。
- 面板稳健性另报 `sentiment_z × mom1_z`、`sentiment_z × vol_q_z`、`sentiment_z × beta_120_z`，用于检验情绪是否调节行业特征斜率。
- 所有连续行业特征在季度截面上标准化；情绪与宏观指数使用时间Trailing z-score，避免跨期尺度漂移和前瞻。

## 5. Fama-MacBeth 结果

下表只列情绪相关系数；完整结果见 `e4_fmb_results.csv` 与 `e4_coefficient_series.csv`。

| 词典 | 设定 | 系数 | 均值 | 中位数 | 缩尾均值 | FMB t | NW t | 显著期数占比 | 平均R² |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ntusd_simplified | sent_exposure | sent_x_exposure_ntusd_simplified | -0.01197 | -0.00104 | 0.00145 | -0.54 | -0.50 | 0.51 | 0.604 |
| dlut | sent_exposure | sent_x_exposure_dlut | 0.04654 | 0.00423 | 0.01583 | 1.95 | 2.16 | 0.55 | 0.600 |
| mpr_domain | sent_exposure | sent_x_exposure_mpr_domain | -0.02030 | 0.00018 | 0.00325 | -0.45 | -0.50 | 0.49 | 0.597 |

## 6. 样本外比较

- IC 为预测值与实际收益的横截面 Spearman 秩相关；ICIR = IC 均值 / IC 标准差。
- OOS R² 以每个目标季截面均值为基准；MAE/RMSE 为行业-季度面板误差。
- 样本外预测使用截至前一期的 expanding-window FMB 系数均值，不使用测试期信息。
- 情绪模型可能因滚动情绪暴露缺失而少若干行；相对 Baseline 的改进均按两者共同样本重算 Baseline 指标。

| 样本 | 模型 | 词典 | 设定 | 期数 | IC | ICIR | OOS R² | MAE | 相对Baseline MAE |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| validation | baseline_notext | none | baseline_notext | 20 | -0.0511 | -0.184 | -2.2843 | 0.09348 | — |
| validation | sentiment | ntusd_simplified | sent_exposure | 20 | -0.0129 | -0.051 | -2.1728 | 0.09265 | -1.77% |
| validation | sentiment | dlut | sent_exposure | 20 | -0.0354 | -0.125 | -3.2834 | 0.10888 | -19.60% |
| validation | sentiment | mpr_domain | sent_exposure | 20 | -0.0323 | -0.141 | -2.2574 | 0.09317 | -2.34% |
| test | baseline_notext | none | baseline_notext | 23 | -0.0369 | -0.113 | -0.5720 | 0.08530 | — |
| test | sentiment | ntusd_simplified | sent_exposure | 23 | -0.0169 | -0.050 | -0.6041 | 0.08734 | -2.05% |
| test | sentiment | dlut | sent_exposure | 23 | -0.0274 | -0.100 | -1.2019 | 0.10375 | -21.23% |
| test | sentiment | mpr_domain | sent_exposure | 23 | -0.0970 | -0.356 | -0.6124 | 0.08746 | -2.19% |

## 7. 面板稳健性

| 词典 | 设定 | 项 | 系数 | 聚类t | N |
|---|---|---|---:|---:|---:|
| ntusd_simplified | panel_industry_fe | sentiment_z | 0.00247 | 0.64 | 2082 |
| ntusd_simplified | panel_time_fe | sentiment_z | nan | nan | 2082 |
| ntusd_simplified | panel_time_fe_interact | sent_x_mom1 | 0.00204 | 0.77 | 2082 |
| ntusd_simplified | panel_time_fe_interact | sent_x_vol | -0.00159 | -0.39 | 2082 |
| ntusd_simplified | panel_time_fe_interact | sent_x_beta | 0.00640 | 1.50 | 2082 |
| dlut | panel_industry_fe | sentiment_z | 0.02064 | 6.99 | 2082 |
| dlut | panel_time_fe | sentiment_z | nan | nan | 2082 |
| dlut | panel_time_fe_interact | sent_x_mom1 | 0.00123 | 0.50 | 2082 |
| dlut | panel_time_fe_interact | sent_x_vol | -0.00426 | -1.02 | 2082 |
| dlut | panel_time_fe_interact | sent_x_beta | 0.00893 | 1.55 | 2082 |
| mpr_domain | panel_industry_fe | sentiment_z | -0.00239 | -0.92 | 2082 |
| mpr_domain | panel_time_fe | sentiment_z | nan | nan | 2082 |
| mpr_domain | panel_time_fe_interact | sent_x_mom1 | 0.00119 | 0.41 | 2082 |
| mpr_domain | panel_time_fe_interact | sent_x_vol | 0.00024 | 0.07 | 2082 |
| mpr_domain | panel_time_fe_interact | sent_x_beta | 0.00318 | 0.81 | 2082 |

## 8. 结论

主词典 `ntusd_simplified` 的 `sent_exposure` 在测试集上未同时优于无文本 Baseline：IC=-0.0169 vs -0.0384，OOS R²=-0.6041 vs -0.5621，MAE=0.08734 vs 0.08558。因此 H2 的预测部分未获支持；共同情绪不能直接识别行业横截面差异，滚动情绪暴露设定也未提供稳定增量。

## 9. 局限

- 季度样本仅约 100 期，横截面约 16–31 个行业，统计功效有限。
- 共同情绪不能直接解释行业横截面差异；本报告只检验其通过行业暴露交互产生预测信息的可能。
- 宏观接口发布日期与统计月份口径存在滞后；本实验采用保守的发布日可得值。
- 行业指数起点不同，早期截面行业数较少；FMB 结果受早期小截面影响。
- `mpr_domain` 在早期小截面存在个别极端系数；表中同时给出中位数和 5% 缩尾均值，结论不依赖单一均值口径。

## 10. 复现

```powershell
python experiments/E4/run_e4.py
```

- 脚本 SHA256：`7e11e45d4793e7133698ee91b1bb1febd590e24cfb545e8f71004cdc5f9651ed`
- 运行环境：{'python': '3.14.0', 'pandas': '3.0.5', 'numpy': '2.5.2', 'scipy': 'scipy.stats:unknown'}
