# WorkBuddy Fix Prompts：图表修复指令

- 生成时间：2026-11-18
- 使用方式：将每条 Prompt 原样交给 WorkBuddy 执行；执行后由 `FIGURE_TABLE_FIX_LIST.md` 的验证项逐条验收
- 全局约束（每条指令均适用）：
  - **不得修改 `data/` 下任何数据文件**，不得编造或填充实验数值。
  - 只修改绘图脚本、表格生成脚本及其输出，不改动研究结论。
  - 修改后必须重跑脚本，确认输出路径与数值正确。
  - 工作区根目录：`D:\Codex\Agentic_Research_Challenge`；图表目录：`图表\`。

---

## Prompt 1｜修复脚本 ROOT/DATA_DIR 路径（P0-4）

> 工作区 `D:\Codex\Agentic_Research_Challenge` 下的绘图与建表脚本当前无法运行：`图表\plotting_scripts\_style.py`、`fetch_data.py`、`build_tables.py`、`build_catalog.py`、`verify_figures_tables.py` 都用 `ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`，得到 `图表\`，于是 `DATA_DIR` 指向不存在的 `图表\data\external`。真实数据在项目根 `D:\Codex\Agentic_Research_Challenge\data\external\`。请：
> 1. 在 `_style.py` 中定义 `PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`（即项目根），并保持 `FIG_DIR = PROJECT_ROOT/图表/figures`、`TABLE_DIR = PROJECT_ROOT/图表/tables`、`DATA_DIR = PROJECT_ROOT/data/external`。
> 2. 让 `fetch_data.py`、`build_tables.py`、`build_catalog.py`、`verify_figures_tables.py` 统一从 `_style.py` import `PROJECT_ROOT`/`DATA_DIR`，不再各自用两级 dirname。
> 3. `build_catalog.py` 的输出路径改为项目根 `FIGURE_TABLE_CATALOG.md`。
> 4. 保持 `图表\tables\`、`图表\figures\` 输出位置不变。
> 5. 在项目根执行 `python 图表\plotting_scripts\fig08_cumulative_nav.py` 与 `python 图表\plotting_scripts\verify_figures_tables.py`，确认不再抛 `FileNotFoundError`，并报告输出路径。

---

## Prompt 2｜统一核查口径并更新 qc 报告（P0-3）

> `图表\plotting_scripts\verify_figures_tables.py` 的 `recompute()` 先按 `date >= "2014-02-21"` 过滤、再 `groupby("code")["收盘"].pct_change()`；而 `build_tables.py` 的 `compute_industry_stats()` 先全样本 `pct_change()`、再过滤。两者对 1999 起点的 16 个行业年化收益不同（例：交通运输 1.9428% vs 1.8065%），导致 `logs\figure_table_qc.md` 的对照基准与 `tables\table04_descriptive_stats.csv` 不一致，但报告写"全部通过"。请：
> 1. 把 `verify_figures_tables.py` 的 `recompute()` 改为与 `build_tables.py` 相同口径：先 `df.groupby("code")["收盘"].pct_change()`，再 `df = df[df["date"] >= START]`，再 `dropna`。
> 2. 在函数 docstring 与 qc 报告中显式写明："收益计算顺序：全样本计算日收益后按 2014-02-21 过滤"。
> 3. 重跑 `verify_figures_tables.py`，更新 `logs\figure_table_qc.md`；若仍有不一致，如实列出，不得写"全部通过"。
> 4. 报告需包含：重算行业数 28、抽检 5 个行业的年化收益与 Table 4 的对照表。

---

## Prompt 3｜修复 Fig 4 图例计数与遮挡（P0-1）

> `图表\plotting_scripts\fig04_industry_coverage.py` 的图例硬编码"1999 起（15 个老行业）""2014-02-21 起（13 个分类调整行业）"，但 `data\external\sw_industries\sw_industry_daily.csv` 实际为 1999-12-30 起 16 个、2014-02-21 起 12 个、2021-12-13 起 3 个。同时图例位于右下，遮挡"综合 801230"的条端数字。请：
> 1. 图例标签改为动态计数：按 `start.year <= 2000`、`<= 2015`、其余三组分别 `len()` 统计，拼接为"1999 起（N 个）""2014-02-21 起（N 个）""2021-12-13 起（N 个）"。
> 2. 图例位置改为 `upper left`（该区域为空白），或移到轴外右侧。
> 3. 横轴右边界从 `2028-06-01` 收到 `2027-01-01`。
> 4. 重跑脚本，确认图例数字为 16/12/3，且所有条端"N 条"标签完整可见。

---

## Prompt 4｜修复 Table 2 表注计数（P0-2）

> `图表\plotting_scripts\build_tables.py` 的 `table02()` 表注硬编码"15 个行业自 1999-12-30 起，13 个行业自 2014-02-21 起"，实际为 16/12。请：
> 1. 在 `table02()` 中按 `load_status()` 的 `start` 列动态统计三组数量，拼接表注。
> 2. 重跑 `build_tables.py`，确认 `图表\tables\table02_industry_coverage.md` 表注为 16/12/3。
> 3. 不要修改表格中的数据行。

---

## Prompt 5｜修正 Table 1 的 MPR 期数表述（P0-5）

> `图表\plotting_scripts\build_tables.py` 的 `table01()` 把 D1 时间范围写成"2001Q1-2026Q2（102 期）"，但 `METHOD_PLAN.md` 5.5 明确记载期数口径冲突（102 期 vs 约 90 份），尚未核验。请：
> 1. 把 D1 时间范围改为"2001Q1-2026Q2（期数待官网核验）"。
> 2. 在表注中补一句："MPR 期数与发布日期以 E0 数据链路核验为准。"
> 3. 重跑 `build_tables.py`，确认 Table 1 不再出现未核验的确定期数。

---

## Prompt 6｜修复 Fig 5/6/7 注释与轴标签重叠（P1-1）

> `fig05_literature_category.py`、`fig06_literature_year.py`、`fig07_literature_methods.py` 底部注释用 `ax.text(0.01, -0.12, ...)`，与 x 轴标签（"文献数量（篇）""发表年份""命中文献数（篇，可多选）"）重叠。请：
> 1. 把注释改为 `fig.text(0.01, 0.02, ...)`，并设置 `fig.subplots_adjust(bottom=0.16)`（或足够间距）。
> 2. 或将 x 轴标签与注释合并为一行图注，取消 xlabel。
> 3. 重跑三个脚本，确认注释与轴标签无重叠、不被裁剪。

---

## Prompt 7｜修复 Fig 8 对数轴刻度与起点注释（P1-2）

> `fig08_cumulative_nav.py` 的纵轴同时显示 `4×10²`、`6×10²`、`4×10³`、`6×10³` 等次刻度标签与自定义主刻度 500/750/1000/1500/2000/3000/5000，读感混乱；且"2014-02-21 = 1000"未说明含首日收益。请：
> 1. 只保留自定义主刻度：设置 `ax.yaxis.set_minor_locator(NullLocator())` 与 `ax.yaxis.set_minor_formatter(NullFormatter())`。
> 2. 图注改为"2014-02-21 起，期初 1000（含首日收益）"，或把收益口径改为不含 2014-02-21 当日收益。
> 3. 重跑脚本，确认纵轴只显示 500/750/1000/1500/2000/3000/5000。

---

## Prompt 8｜修复 Fig 9 标签碰撞（P1-3）

> `fig09_risk_return.py` 中"医药生物"与"农林牧渔"标签重叠，"国防军工"与"计算机"距离过近。请：
> 1. 引入 `adjustText`（`pip install adjustText`）自动避让，或对这两组标签手动设置相反的 `xytext` 偏移。
> 2. 重跑脚本，确认所有行业标签互不重叠、不越界。

---

## Prompt 9｜修复 Fig 12 标题与事件标注（P1-4）

> `fig12_benchmark.py` 标题"全样本起点归一"含糊（实际两条基准各自起点归一），且事件标注文字与 x 轴年份刻度重叠。请：
> 1. 标题改为"市场基准走势（各自样本起点归一）"。
> 2. 图注写明"沪深300：2005-01-04 起；中证全指：2011-08-02 起"。
> 3. 事件标注统一放在图顶部空白区（或改为图例），避免与年份刻度重叠。
> 4. 重跑脚本，确认标注清晰。

---

## Prompt 10｜补全 Fig 2 预期输出列（P1-5）

> `fig02_experiment_design.py` 的 `GROUPS` 中预期输出列不完整。请对照 `EXPERIMENT_PLAN.md` 补全：
> - E2：`e2_event_baseline.csv`、`e2_car_plot.png`
> - E3：`e3_event_results.csv`、`e3_car_by_sentiment.csv`、`e3_car_plot.png`
> - E4：`e4_fmb_results.csv`、`e4_oos_metrics.csv`、`e4_coefficient_series.csv`
> - E5：`e5_heterogeneity.csv`、`e5_sensitivity_plot.png`
> - E7：`e7_portfolio.csv`、`e7_equity_curve.png`
> 若单框空间不足，可缩小字号或改为两行。重跑脚本确认排版不重叠。

---

## Prompt 11｜Fig 10 字号与 Fig 7 规则说明（P2）

> 1. `fig10_correlation.py` 单元格字号由 5.8 调整为 6.5，确认半栏缩放后可读。
> 2. 在 `FIGURE_TABLE_CATALOG.md` 或论文附录登记 Fig 7 的关键词规则表（`fig07_literature_methods.py` 的 `RULES`），并注明"可多选、非互斥，合计大于 38"。

---

## 禁止事项（务必遵守）

- 不得修改 `data/external/`、`data/raw/`、`documents/` 中的任何原始文件。
- 不得给 Table 9/10/11 填入任何数值；不得生成 Fig 13-20 的假数据图。
- 不得把"待数据"项目改写成"已完成"。
- 不得为让图表好看而截断收益轴或选择性展示。

*编制：紫璃 (Murasaki)　2026-11-18*
