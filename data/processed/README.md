# data/processed 说明

## 唯一有效的数据文件

| 文件 | 行数 | 用途 |
|---|---|---|
| `ret_panel.csv` | 1400 = 28 行业 × 50 季度（2014Q2–2026Q3） | **分析用面板**，由 `src/data/build_returns.py` 生成 |
| `mpr_meta.csv` | 102 | MPR 报告元信息（期次、发布日、URL、页码、字数） |

## 已废弃（不要再用）

| 文件 | 状态 |
|---|---|
| `ret_ex.csv` | **旧版（37 行业 / 2809 行 / 字符串季度 / 含 1999Q4 零值）**。已复制归档到 `data/archive/ret_ex_v0_37ind.csv`，此处的副本因被编辑器占用未能删除，请关闭 VS Code / Excel 后手动删除：`Remove-Item "data\processed\ret_ex.csv"` |

## 变量定义

见 `docs/04_数据修复记录.md`。要点：

- `ret`：季度累计收益（小数）
- `mkt_ret`：本样本 28 行业等权平均（固定篮子）
- `ret_ex`：`ret − mkt_ret`，**样本内相对收益**，每季求和恒为 0
- `n_days`：该季交易日数（55–66，用于验证季度完整性）

## 重建方式

```bash
python src/data/build_returns.py   # 重建面板
python src/data/audit_panel.py     # 生成 docs/数据体检.md
python src/data/verify_panel.py    # 独立核查（与原始日线逐条对账）
```

## 事件表

| 文件 | 内容 |
|---|---|
| event_map.csv | 102 期 MPR 发布日 → 交易日窗口（t_m1/t0/t1/t5/t10，含长假标记） |
| 	ext_availability.csv | 102 份 MPR PDF 的文本可得性诊断（69 份可抽取，33 份缺失） |

## 注意

et_ex.csv 已于 2026-11-14 删除（旧版 2809 行 / 37 行业）。现在 et_ex 是 et_panel.csv 里的一列。