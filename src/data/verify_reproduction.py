"""
verify_reproduction.py — 一键校验：确认复现结果与论文一致
========================================================
用法：
    python src/data/verify_reproduction.py

做三件事：
    1. 检查关键数据文件是否存在、行数是否对
    2. 重新计算几个核心统计量，与论文报告值比对
    3. 读取 h1_coef.csv，核对论文表 5.2、5.3 的关键系数与 t 值

全部通过时打印 "全部一致"，否则列出偏差项。容差：
    系数 ±5%（相对）；t 值 ±0.15（绝对）；相关/计数 ±0.002。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROC = "data/processed"
DICT = "data/external"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def main() -> int:
    # ---------------- 1. 文件与行数 ----------------
    expects = {
        "mpr_meta.csv": 102,
        "sentiment.csv": 102,
        "market_events.csv": 102,
        "event_map.csv": 102,
        "ret_panel.csv": 1400,
        "text_availability.csv": 102,
    }
    for fn, n in expects.items():
        p = os.path.join(PROC, fn)
        if not os.path.exists(p):
            check(f"文件 {fn}", False, "不存在")
            continue
        rows = len(pd.read_csv(p))
        check(f"文件 {fn}", rows == n, f"{rows} 行（预期 {n}）")

    if not os.path.exists(os.path.join(PROC, "sentiment.csv")):
        print("缺少 sentiment.csv，无法继续")
        return 1

    # ---------------- 2. 核心统计量 ----------------
    s = pd.read_csv(os.path.join(PROC, "sentiment.csv"))
    check("tone_P 非缺失期数", int(s.tone_P.notna().sum()) == 99,
          f"{int(s.tone_P.notna().sum())} 期（预期 99）")

    corr = float(s.tone_R.corr(s.tone_P))
    check("R 段与 P 段情绪相关", abs(corr - 0.375) < 0.002, f"{corr:.3f}（预期 0.375）")

    check("tone_P 均值", abs(float(s.tone_P.mean()) - 0.632) < 0.002,
          f"{s.tone_P.mean():.3f}（预期 0.632）")

    m = pd.read_csv(os.path.join(PROC, "market_events.csv"))
    check("事件窗收益非缺失", int(m.r16_1d.notna().sum()) == 102,
          f"{int(m.r16_1d.notna().sum())} 期（预期 102）")
    check("事件日振幅均值", abs(float(m.rng16_evt.mean()) - 0.0209) < 0.0005,
          f"{m.rng16_evt.mean():.4f}（预期 0.0209）")

    # ---------------- 3. 回归系数 ----------------
    coef_path = os.path.join(PROC, "h1_coef.csv")
    if not os.path.exists(coef_path):
        check("h1_coef.csv", False, "不存在，请先运行 run_h1.py")
    else:
        c = pd.read_csv(coef_path)

        def get(model: str, var: str):
            row = c[(c.model == model) & (c["var"] == var)]
            return (float(row.coef.iloc[0]), float(row.t.iloc[0])) if len(row) else (np.nan, np.nan)

        specs = [
            ("A1 发布后一日", "z_tone_P", 6.3e-4, 0.25, "表 5.2/5.3 收益方向"),
            ("B3 日内振幅", "z_tone_P", -2.78e-3, -2.58, "表 5.2 波动主结果"),
            ("B4 日内振幅+前波动", "z_tone_P", -1.85e-3, -2.43, "表 5.2 控制后"),
            ("C1 一日·断点", "zP_x_post", -7.56e-3, -2.11, "表 5.3 断点交互"),
            ("C2 五日·断点", "zP_x_post", -1.814e-2, -2.65, "表 5.3 五日交互"),
        ]
        for model, var, exp_coef, exp_t, note in specs:
            coef, t = get(model, var)
            if not np.isfinite(coef):
                check(f"{note}", False, f"未找到模型 {model}")
                continue
            ok_coef = abs(coef - exp_coef) <= max(abs(exp_coef) * 0.05, 1e-5)
            ok_t = abs(t - exp_t) <= 0.15
            check(f"{note}：{model}", ok_coef and ok_t,
                  f"coef={coef*10000:+.1f}bp（预期 {exp_coef*10000:+.1f}），t={t:+.2f}（预期 {exp_t:+.2f}）")

    # ---------------- 输出 ----------------
    print("=" * 72)
    for name, ok, detail in results:
        print(f"{'[通过]' if ok else '[偏差]'} {name:<34} {detail}")
    print("=" * 72)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    if n_fail == 0:
        print(f"全部一致（{len(results)} 项检查通过）")
        return 0
    print(f"存在 {n_fail} 项偏差，请对照 docs/可复现说明.md 第 5 节排查")
    return 1


if __name__ == "__main__":
    sys.exit(main())
