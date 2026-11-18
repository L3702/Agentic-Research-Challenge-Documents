# Figure & Table Fix List：修改清单（按优先级）

- 生成时间：2026-11-18
- 依据：`FIGURE_TABLE_AUDIT.md`
- 优先级定义：P0 = 必须改，否则不能进初稿；P1 = 影响可读性/可复现性，进初稿前改；P2 = 排版细节，投稿前改
- 原则：只改图表、脚本与注释，**不改动任何实验数据**；无法核验的项目继续标注"待数据"

---

## P0-1｜Fig 4 图例计数错误 + 图例遮挡（Critical C1 / Major M5）

- 文件：`图表/plotting_scripts/fig04_industry_coverage.py`、`图表/figures/fig04_industry_coverage.png`
- 问题：图例硬编码"1999 起（15 个老行业）""2014-02-21 起（13 个分类调整行业）"，实际为 16/12/3；图例压住"综合 801230"条端数字。
- 修改：
  1. 图例标签改为动态计数：`f"1999 起（{len(old)} 个老行业）"` 等，避免再次漂移。
  2. 图例位置从 `lower right` 改为 `upper left`（该区域为空白），或移到轴外。
  3. 横轴右边界从 2028-06-01 收到 2027-01-01，减少留白。
- 验证：重跑脚本，确认图例数字 = 16/12/3，条端数字不被遮挡。

## P0-2｜Table 2 表注计数错误（Critical C1 延伸）

- 文件：`图表/plotting_scripts/build_tables.py`（`table02()` 的 `note`）、`图表/tables/table02_industry_coverage.md`
- 问题：硬编码"15 个行业自 1999-12-30 起，13 个行业自 2014-02-21 起"，实际 16/12。
- 修改：动态拼接计数，或改为"16 个 / 12 个 / 3 个"。
- 验证：重跑 `build_tables.py`，检查表注。

## P0-3｜核查口径冲突与 qc 报告失效（Critical C2）

- 文件：`图表/plotting_scripts/verify_figures_tables.py`、`logs/figure_table_qc.md`
- 问题：`recompute()` 用"先过滤后算收益"，`build_tables.py` 用"先算收益后过滤"，16 个老行业年化收益不一致；qc 报告数字与 Table 4 不符却写"全部通过"。
- 修改：
  1. 统一为"先全样本 `pct_change`，再过滤 `date >= START`"（与 Table 4/5、Fig 8-11 一致）。
  2. 在 `recompute()` 注释与 qc 报告中写明口径顺序。
  3. 重跑 `verify_figures_tables.py`，更新 qc 报告；若仍有不一致，如实列出。
- 验证：qc 报告中的年化收益与 Table 4 完全一致（rtol=1e-6）。

## P0-4｜脚本 ROOT/DATA_DIR 路径错误，图表不可复现（Critical C3）

- 文件：`图表/plotting_scripts/_style.py`、`fetch_data.py`、`build_tables.py`、`build_catalog.py`、`verify_figures_tables.py`
- 问题：`ROOT = dirname(dirname(__file__))` = `图表/`，`DATA_DIR` 指向不存在的 `图表/data/external`；实测 Fig 8 与 verify 均 `FileNotFoundError`。`build_catalog.py` 还会把 catalog 写到 `图表/` 而非项目根。
- 修改：
  1. 把 ROOT 上移一级（`dirname` 三次）指向项目根，或在 `_style.py` 统一定义 `PROJECT_ROOT` 供各脚本 import。
  2. `TABLE_DIR`、`FIG_DIR`、`PREVIEW_DIR` 保持指向 `图表/` 下对应子目录。
  3. `build_catalog.py` 输出路径改回项目根 `FIGURE_TABLE_CATALOG.md`。
  4. 更新 `FIGURE_TABLE_CATALOG.md` 第 5 节的复现命令与路径。
- 验证：在项目根重跑 `fig08_cumulative_nav.py`、`verify_figures_tables.py`、`build_tables.py`、`build_catalog.py`，全部成功且输出路径正确。

## P0-5｜Table 1 的 MPR 期数写成确定值（Major M6）

- 文件：`图表/plotting_scripts/build_tables.py`（`table01()`）、`图表/tables/table01_dataset_description.md`
- 问题：D1 写"2001Q1-2026Q2（102 期）"，与 `METHOD_PLAN.md` 5.5 的口径冲突记载不符。
- 修改：改为"2001Q1-2026Q2（期数待官网核验）"，待 E0 锁定后再填确定值。
- 验证：表内不再出现未核验的确定期数。

---

## P1-1｜Fig 5/6/7 注释与轴标签重叠（Major M4）

- 文件：`fig05_literature_category.py`、`fig06_literature_year.py`、`fig07_literature_methods.py`
- 修改：把底部注释改为 `fig.text(0.01, 0.01, ...)` 并 `fig.subplots_adjust(bottom=0.16)`，或把 xlabel 与注释合并为一行图注。
- 验证：注释与轴标签无重叠。

## P1-2｜Fig 8 对数轴刻度与起点注释（Minor m8/m9）

- 文件：`fig08_cumulative_nav.py`
- 修改：
  1. 关闭次刻度标签（`ax.yaxis.set_minor_locator(NullLocator())` 或 `tick_params(labelsize)` 控制），只保留自定义主刻度。
  2. 图注改为"2014-02-21 起，期初 1000（含首日收益）"或调整收益口径使起点严格对应 2014-02-21。
- 验证：刻度只显示 500/750/1000/1500/2000/3000/5000。

## P1-3｜Fig 9 标签碰撞（Minor m10）

- 文件：`fig09_risk_return.py`
- 修改：对"医药生物/农林牧渔""国防军工/计算机"使用 `adjustText` 或手动 `xytext` 偏移。
- 验证：任意缩放时标签不重叠。

## P1-4｜Fig 12 标题与事件标注（Minor m11/m12）

- 文件：`fig12_benchmark.py`
- 修改：标题改为"市场基准走势（各自样本起点归一）"并在图注写明两基准起始日；事件标注上移/改为图例，避免与年份刻度重叠。
- 验证：标注不与刻度重叠。

## P1-5｜Fig 2 预期输出列不完整（Major M7）

- 文件：`fig02_experiment_design.py`
- 修改：补全 E2/E3/E4/E5/E7 的 CSV 输出，或统一注明"仅列主要输出"。
- 验证：与 `EXPERIMENT_PLAN.md` 预期输出一致。

---

## P2｜细节项

- m13 Fig 10 单元格字号：若论文缩放至半栏，确认 5.8pt 仍可读，必要时增大到 6.5。
- m14 Fig 7 关键词规则：在正文或附录列出完整规则表，说明"可多选、非互斥"。
- m15 Fig 4 横轴右界：随 P0-1 一并收窄。
- Table 8：论文引用时改用 CSV 完整版，避免 md 截断丢失信息。
- 全部图：投稿前统一确认字体（微软雅黑/等线）在目标期刊排版中可嵌入。

---

## 不需要修改（保持现状）

- Fig 1、Fig 3：逻辑与布局合格。
- Fig 10、Fig 11：数值与可读性合格（除 P2 细节）。
- Table 3、6、7、8：内容与来源一致。
- Table 9、10、11：模板标注清晰，保持"待数据"状态，**不得填入任何占位数值**。
- Fig 13-20：保持"待数据"，不得生成任何假数据图。

---

## 修改顺序建议

1. P0-4（路径）→ 2. P0-3（口径）→ 3. P0-1/P0-2（计数）→ 4. P0-5（期数）→ 5. P1 排版 → 6. P2 细节。
每完成一项，重跑受影响脚本并更新 `FIGURE_TABLE_CATALOG.md` 的生成时间与核查状态。

*编制：紫璃 (Murasaki)　2026-11-18*
