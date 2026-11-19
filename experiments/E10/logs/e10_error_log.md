# E10 error analysis log

- run_at: 2026-11-19T23:47:44+08:00
- python: 3.14.0
- pandas: 3.0.5
- numpy: 2.5.2
- anomaly_records: 1002
- anomaly_csv: experiments/E10/e10_anomaly_samples.csv
- report: experiments/E10/e10_error_analysis.md
- metrics_cache: _temp/E10/e10_metrics.json

## Type counts

```text
anomaly_type
market_missing_trading_day                  521
prediction_residual_outlier                 287
lexicon_low_coverage_decile                  33
llm_variant_failure                          29
e8_failure_record_class                      22
llm_json_parsed_failure_group                18
classification_start_change                  15
human_rating_pending                         15
residual_concentrated_by_industry            14
robustness_limited_setting                   10
e3_skip_cell_class                            9
portfolio_extreme_quarter                     6
reverse_case_high_sentiment_negative_car      5
reverse_case_low_sentiment_positive_car       5
event_mean_car_outlier                        4
residual_concentrated_by_year                 3
raw_significant_not_multiplicity_robust       3
portfolio_factor_window_incomplete            2
paired_variant_sign_flip                      1
```

## Input hashes

| path | bytes | sha256 |
|---|---:|---|
| `data/external/sw_industries/sw_industry_daily.csv` | 12634203 | `8f2d85989907d636eceaca13eb1a81a129f8ac7ed8ca290aec5b2269a22f4e70` |
| `_temp/e0_market_raw/benchmark_sh000300.csv` | 468903 | `425520ba9b29c5d2903806cefdc9fb33f29516b72ff1ae8a4e253454bc829309` |
| `experiments/E1/sentiment_dictionary_full.csv` | 159054 | `ad6d1e673c4bdc9530e0335276639463a7ff84b7a5517dccfd63bce5e8a67268` |
| `experiments/E1/validation/human_ratings.csv` | 783 | `6b45f078f27695d55a970d49a52ffb186e25aad7a8dc2f6b5784e998c423740f` |
| `experiments/E6/e6_llm_scores.csv` | 24307 | `f26afb6b098a9fb45b7b732dfbd1c9a91e7753230963db02978b37d80cc436a9` |
| `experiments/E9/e9_response_audit.csv` | 365002 | `ecf3028b99471f4853314b08079ffd9fbfbd02df836efd34d8fa96dc185a4b69` |
| `experiments/E9/e9_sensitivity_summary.csv` | 14054 | `d8d76ddac418daf2c359158b1da4c69fa8512f06e2ce8980534297372852f7eb` |
| `_temp/e4/e4_oos_predictions.csv` | 683186 | `0b24d052212b278c53ab88247735c849fdc838d7c0c8487c01998db29de731e6` |
| `_temp/e8/prediction/predictions.csv` | 8438675 | `d2cfd31250b7523bcdcf02bb9915ffd0dd6e232e7d1bd129d280a6e767810840` |
| `_temp/e6/e6_oos_predictions.csv` | 994413 | `6727e11791e87749f6d1a2a965f07f336dab89e1e279f5f062101fce8f481efa` |
| `experiments/E3/e3_event_results.csv` | 3389664 | `dc23203ace553fe330bf80db1c39adc3c38e4a239b6ee5e591fc6d99f7aa1088` |
| `experiments/E8/e8_robustness_summary.csv` | 67234 | `ea8c9233f62482ccdedf08f41a8e8d25f9da8c992d9d21b4e126c3a616829039` |
| `experiments/E7/e7_portfolio.csv` | 34855 | `fc94e727122d256d1d92dcbf4e2308ef64a324685a55cf049f4455293aeb409e` |
| `experiments/E8/e8_failures.csv` | 2482541 | `e18f5f125a41d630dd2f0317862f3b3fd4bf6ff9cea7e0363b527eed7820b372` |
| `_temp/e3/e3_skip_cells.csv` | 347390 | `3cbafc3956b53aaf02db29ed983462f9a4feb90d83a7a5299255c18039a9b994` |

## Read-only verification

No source experiment file was edited, imputed, or deleted. The only outputs are this log, the report, and the anomaly sample CSV; the metrics JSON is placed under `_temp/E10`.
