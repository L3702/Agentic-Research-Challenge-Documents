"""
run_h1.py — H1：MPR 文本情绪对市场收益与波动的影响
=================================================
第一张结果表，分两个面板：

    A 收益方向：情绪（P 段）是否影响发布后的市场收益？
    B 波动   ：情绪是否降低发布日的市场波动？（日内振幅 / |收益|，控制事前波动）

安慰剂：单独用 R 段（宏观回顾）——若 R 段也显著，说明测的是整篇语调而非新信息。

被解释变量（来自 build_market_events.py）
    r{16}_1d      发布日 t0 → t1 收益（"发布后一日"）
    r{16}_5d      t0 → t5
    car{16}_m1_p1 [-1,+1] 累计收益
    abs_m1        |t_m1→t0| 事件日绝对收益（粗糙的波动代理）
    rng_evt       事件日日内振幅 (最高-最低)/前收盘，等权平均（更干净的波动代理）
    prevol        事前 60 个交易日市场收益标准差（控制波动聚集）

标准误：Newey-West HAC（滞后 4 期）。系数单位：基点（bp）。
不依赖 statsmodels / scipy，纯 numpy。

输出：docs/H1结果.md、data/processed/h1_coef.csv
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

SENT = "data/processed/sentiment.csv"
MKT = "data/processed/market_events.csv"
OUT_MD = "docs/H1结果.md"
OUT_CSV = "data/processed/h1_coef.csv"


def ols_hac(y: np.ndarray, X: np.ndarray, lag: int = 4):
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)   # pinv：共线时不报错，只给最小范数解
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        G = (X[l:] * resid[l:, None]).T @ (X[:-l] * resid[:-l, None])
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, se, r2


def p_value(t: float) -> float:
    return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fit(df: pd.DataFrame, y: str, xs: list[str], lag: int = 4):
    d = df[[y] + xs].dropna()
    if len(d) < 10:
        return None
    X = np.column_stack([np.ones(len(d))] + [d[x].values for x in xs])
    beta, se, r2 = ols_hac(d[y].values, X, lag)
    out = []
    for i, nm in enumerate(["const"] + xs):
        t = beta[i] / se[i] if se[i] > 0 else np.nan
        p = p_value(t) if np.isfinite(t) else np.nan
        out.append({"var": nm, "coef": beta[i], "se": se[i], "t": t,
                    "p": p, "star": stars(p), "N": len(d), "R2": r2, "depvar": y})
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentiment", default=SENT)
    ap.add_argument("--market", default=MKT)
    ap.add_argument("--suffix", default="16", choices=["16", "28"])
    ap.add_argument("--lag", type=int, default=4)
    ap.add_argument("--out-md", default=OUT_MD, help="结果 markdown 输出路径")
    ap.add_argument("--since", type=int, default=0, help="只用该年份及以后的样本")
    args = ap.parse_args()

    if not os.path.exists(args.sentiment):
        print(f"[错误] 找不到情绪文件：{args.sentiment}")
        print("解决办法（二选一）：")
        print("  1) 从仓库下载（最快）：")
        print('     Invoke-WebRequest -Uri "https://raw.githubusercontent.com/L3702/'
              'Agentic-Research-Challenge-Documents/develop/data/processed/sentiment.csv" '
              f'-OutFile "{args.sentiment}"')
        print("  2) 本地重新生成：")
        print('     python src/data/build_sentiment.py --dict "data\\external\\姜富伟词典.xlsx"')
        return 1
    if not os.path.exists(args.market):
        print(f"[错误] 找不到市场收益文件：{args.market}")
        print("请先运行：python src/data/build_market_events.py")
        return 1

    s = pd.read_csv(args.sentiment)
    m = pd.read_csv(args.market)
    df = s.merge(m, on="period", how="inner")
    if args.since:
        df = df[df["period"].str[:4].astype(int) >= args.since].copy()
    k = args.suffix
    df["abs_m1"] = df[f"r{k}_m1_0"].abs()
    df["abs_15"] = df[f"r{k}_1_5"].abs()
    # 制度断点：2014 年前后（2013 钱荒 / 利率市场化 / 2015 汇改；MPR 亦在 2015 年改版）
    df["post14"] = (df["period"].str[:4].astype(int) >= 2014).astype(float)
    df["zP_x_post"] = df["z_tone_P"] * df["post14"]

    print(f"[合并] 情绪 {len(s)} 期 + 市场 {len(m)} 期 → {len(df)} 期")
    print(f"[口径] {k} 行业；振幅列 rng{k}_evt；事前波动 prevol{k}")

    panels = {
        "A 收益方向": [
            ("A1 发布后一日", f"r{k}_1d", ["z_tone_P"]),
            ("A2 发布后5日", f"r{k}_5d", ["z_tone_P"]),
            ("A3 一日·双段", f"r{k}_1d", ["z_tone_P", "z_tone_R"]),
            ("A4 五日·双段", f"r{k}_5d", ["z_tone_P", "z_tone_R"]),
            ("A5 CAR[-1,+1]", f"car{k}_m1_p1", ["z_tone_P", "z_tone_R"]),
            ("A6 安慰剂:只用R", f"r{k}_1d", ["z_tone_R"]),
        ],
        "C 制度断点（2014 前后）": [
            ("C1 一日·断点", f"r{k}_1d", ["z_tone_P", "zP_x_post", "post14"]),
            ("C2 五日·断点", f"r{k}_5d", ["z_tone_P", "zP_x_post", "post14"]),
            ("C3 振幅·断点", f"rng{k}_evt", ["z_tone_P", "zP_x_post", "post14"]),
            ("C4 |收益|·断点", "abs_m1", ["z_tone_P", "zP_x_post", "post14"]),
        ],
        "B 市场波动": [
            ("B1 |事件日收益|", "abs_m1", ["z_tone_P"]),
            ("B2 |事件日收益|+前波动", "abs_m1", ["z_tone_P", f"prevol{k}"]),
            ("B3 日内振幅", f"rng{k}_evt", ["z_tone_P"]),
            ("B4 日内振幅+前波动", f"rng{k}_evt", ["z_tone_P", f"prevol{k}"]),
            ("B5 振幅+双段+前波动", f"rng{k}_evt", ["z_tone_P", "z_tone_R", f"prevol{k}"]),
            ("B6 5日振幅+前波动", f"rng{k}_5d", ["z_tone_P", f"prevol{k}"]),
        ],
    }

    L = ["# H1 结果：MPR 文本情绪与市场收益 / 波动", ""]
    L.append(f"- 市场口径：**{k} 个申万一级行业等权组合**")
    L.append(f"- 样本：{len(df)} 期，{df['period'].min()}–{df['period'].max()}")
    L.append(f"- 标准误：Newey-West HAC（滞后 {args.lag} 期）；系数单位：基点（bp）")
    L.append("- 情绪变量已标准化，系数含义＝情绪每高 1 个标准差对应的 bp 变化")
    L.append("")

    all_rows = []
    for panel, specs in panels.items():
        if panel.startswith("C") and df["post14"].nunique() < 2:
            print("[跳过] 制度断点面板（当前子样本内 post14 无变化）")
            continue
        L.append(f"## {panel}")
        L.append("")
        L.append("| 模型 | 被解释变量 | z_tone_P | z_tone_P×2014后 | z_tone_R | 其他控制 | N | R² |")
        L.append("|---|---|---|---|---|---|---|---|")
        for name, y, xs in specs:
            r = fit(df, y, xs, args.lag)
            if r is None:
                continue
            r["panel"] = panel
            r["model"] = name
            all_rows.append(r)
            def cell(v):
                row = r[r["var"] == v]
                if len(row) == 0 or not np.isfinite(row.t.iloc[0]):
                    return ""
                return f"{row.coef.iloc[0]*10000:+.1f}{row.star.iloc[0]} ({row.t.iloc[0]:+.2f})"
            ctrl = ", ".join(x for x in xs if not x.startswith("z_tone"))
            L.append(f"| {name} | {y} | {cell('z_tone_P')} | {cell('zP_x_post')} | "
                     f"{cell('z_tone_R')} | {ctrl or '—'} | {int(r.N.iloc[0])} | {r.R2.iloc[0]:.3f} |")
        L.append("")

    res = pd.concat(all_rows, ignore_index=True)
    res.to_csv(OUT_CSV, index=False, float_format="%.6f")

    # ---- 断点敏感性：换个断点年份，交互项还显著吗？ ----
    L.append("## D 断点敏感性（换不同断点年份，看交互项 t 值）")
    L.append("")
    L.append("| 断点年份 | 一日收益 交互项 | 五日收益 交互项 | 振幅 交互项 |")
    L.append("|---|---|---|---|")
    rows_break = []
    for yr in ((2008, 2010, 2012, 2014, 2016, 2018) if df["post14"].nunique() >= 2 else ()):
        cells, rec = [], {"break": yr}
        for yname, col in (("r1", f"r{k}_1d"), ("r5", f"r{k}_5d"), ("rng", f"rng{k}_evt")):
            d2 = df.copy()
            d2["b"] = (d2["period"].str[:4].astype(int) >= yr).astype(float)
            d2["zx"] = d2["z_tone_P"] * d2["b"]
            rr = fit(d2, col, ["z_tone_P", "zx", "b"], args.lag)
            if rr is None:
                cells.append("")
                continue
            row = rr[rr["var"] == "zx"].iloc[0]
            cells.append(f"{row.coef*10000:+.1f}{row.star} ({row.t:+.2f})")
            rec[yname] = (row.coef, row.t)
        rows_break.append(rec)
        L.append(f"| {yr} | {cells[0]} | {cells[1]} | {cells[2]} |")
    L.append("")
    L.append("> 若交互项只在某一个年份显著、换年份就消失，说明断点是挑出来的；"
             "若多个候选年份都显著，说明变化是渐进的。")
    L.append("")
    # ---- 影响点：去掉危机年份 / 极端值，结论还在吗？ ----
    L.append("## E 影响点检验（去掉极端年份或极端值）")
    L.append("")
    L.append("| 设定 | 一日收益 z_tone_P | 日内振幅 z_tone_P | 断点交互(一日) | 断点交互(五日) | 样本 |")
    L.append("|---|---|---|---|---|---|")

    def _cell(res, var="z_tone_P"):
        if res is None:
            return ""
        row = res[res["var"] == var]
        if len(row) == 0:
            return ""
        row = row.iloc[0]
        return f"{row.coef*10000:+.1f}{row.star} ({row.t:+.2f})"

    def run_pair(d: pd.DataFrame):
        r1 = fit(d, f"r{k}_1d", ["z_tone_P"], args.lag)
        rv = fit(d, f"rng{k}_evt", ["z_tone_P"], args.lag)
        d2 = d.copy()
        d2["b"] = (d2["period"].str[:4].astype(int) >= 2014).astype(float)
        d2["zx"] = d2["z_tone_P"] * d2["b"]
        rc1 = fit(d2, f"r{k}_1d", ["z_tone_P", "zx", "b"], args.lag)
        rc5 = fit(d2, f"r{k}_5d", ["z_tone_P", "zx", "b"], args.lag)
        n = int(r1.N.iloc[0]) if r1 is not None else 0
        return (_cell(r1), _cell(rv), _cell(rc1, "zx"), _cell(rc5, "zx"), n)

    settings = [("全样本", df)]
    settings.append(("剔除 2015 年", df[df["period"].str[:4] != "2015"]))
    settings.append(("剔除 2020 年", df[df["period"].str[:4] != "2020"]))
    settings.append(("剔除 2003Q1（分段异常）", df[df["period"] != "2003Q1"]))
    r = df[f"r{k}_1d"]
    thr = 3 * r.std()
    settings.append(("剔除 |收益|>3σ", df[r.abs() <= thr]))
    settings.append(("收益缩尾 1%/99%", df.assign(**{f"r{k}_1d": r.clip(r.quantile(0.01), r.quantile(0.99))})))
    for name, d in settings:
        a, v, c1, c5, n = run_pair(d)
        L.append(f"| {name} | {a} | {v} | {c1} | {c5} | {n} |")
    L.append("")
    L.append("> 若剔掉 2015（股灾）或 2020（疫情）后系数大幅缩水，说明结论依赖少数极端事件。")
    L.append("")
    # ---- 置换检验：打乱情绪序列 500 次，看真实 t 值有多极端 ----
    rng = np.random.default_rng(20261116)
    L.append("## F 置换检验（打乱情绪与事件的配对，500 次）")
    L.append("")
    L.append("| 模型 | 真实系数 | 真实 t | 置换 p 值 |")
    L.append("|---|---|---|---|")

    def perm_test(y_col, xs, target, n_perm=500):
        d = df[[y_col] + xs].dropna().reset_index(drop=True)
        base = fit(d.assign(**{target: d[target]}), y_col, xs, args.lag)
        if base is None:
            return None
        obs = base[base["var"] == target].iloc[0]
        cnt = 0
        vals = d[target].values.copy()
        for _ in range(n_perm):
            d2 = d.copy()
            d2[target] = rng.permutation(vals)
            r = fit(d2, y_col, xs, args.lag)
            if r is None:
                continue
            row = r[r["var"] == target].iloc[0]
            if abs(row.t) >= abs(obs.t):
                cnt += 1
        return obs, cnt / n_perm

    for label, y_col, xs, target in (
            ("B3 日内振幅", f"rng{k}_evt", ["z_tone_P"], "z_tone_P"),
            ("B5 振幅+双段", f"rng{k}_evt", ["z_tone_P", "z_tone_R", f"prevol{k}"], "z_tone_P"),
            ("C1 断点交互", f"r{k}_1d", ["z_tone_P", "zP_x_post", "post14"], "zP_x_post")):
        out = perm_test(y_col, xs, target)
        if out is None:
            continue
        obs, p = out
        L.append(f"| {label} | {obs.coef*10000:+.1f} | {obs.t:+.2f} | {p:.3f} |")
    L.append("")
    L.append("> 置换检验不依赖分布假设：把情绪序列与事件随机配对 500 次，"
             "看真实 t 值是否比绝大多数随机配对更极端。")
    L.append("")

    corr = df[[f"r{k}_1d", f"rng{k}_evt", "z_tone_P", "z_tone_R"]].corr().round(3)
    L.append("## 相关系数（供参考）")
    L.append("")
    L.append("```")
    L.append(corr.to_string())
    L.append("```")
    L.append("")
    L.append("> 显著性：\\* p<0.10，\\*\\* p<0.05，\\*\\*\\* p<0.01（正态近似）。")
    L.append("> 括号内为 HAC t 值。")
    with open(args.out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))

    # 控制台
    print("-" * 78)
    for panel, specs in panels.items():
        print(f"[{panel}]")
        for name, y, xs in specs:
            r = res[(res.model == name)]
            if len(r) == 0:
                continue
            parts = []
            for v in ("z_tone_P", "zP_x_post", "z_tone_R"):
                row = r[r["var"] == v]
                if len(row) and np.isfinite(row.t.iloc[0]):
                    parts.append(f"{v[-1]}={row.coef.iloc[0]*10000:+.1f}bp{row.star.iloc[0]}"
                                 f"(t={row.t.iloc[0]:+.2f})")
            print(f"  {name:<22} N={int(r.N.iloc[0]):<3} " + "  ".join(parts))
    print("-" * 78)
    print(f"[完成] {args.out_md} / {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
