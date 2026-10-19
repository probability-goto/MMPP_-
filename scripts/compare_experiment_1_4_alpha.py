#!/usr/bin/env python
"""実験 1・4 の alpha=0.1 (旧) と alpha=0.005 (新) の比較表を作る.

数値はすべて CSV から読む (モデルの再計算はしない).
    実験 1 旧: results/experiment_1_a0.1.csv
    実験 1 新: results/experiment_1.csv (既定値 alpha=0.005)
    実験 4 旧: results/experiment_1_4_alpha_comparison/experiment_4_{level}_a0.1.csv
               (alpha=0.1 を E_B 等の列付きで再計算したもの.
                4 指標は results/experiment_4_{level}_a0.1.csv と一致することを確かめる)
    実験 4 新: results/experiment_4_{level}.csv (既定値 alpha=0.005)

出力: results/experiment_1_4_alpha_comparison.md
    1. 実験 1: バースト水準 x rho の表
    2. 実験 4: バースト水準 x K x rho の表
    3. 機械的な判定 (rho に対する単調性, バースト水準の順序の入れ替わり,
       K に対する増減と K=500 / K=1000 の相対差)
    4. キャプション用の固定パラメータ

使用例:
    python scripts/compare_experiment_1_4_alpha.py
"""
import argparse
import csv
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

# (CSV 列名, 表示名)
METRICS: List[Tuple[str, str]] = [
    ("P_block_arrival_stable", "P_block"),
    ("E_W", "E[W]"),
    ("Cost", "Cost"),
    ("ERP", "ERP"),
    ("E_B", "E[B]"),
    ("E_S", "E[S]"),
    ("E_I", "E[I]"),
]
LEVELS = ["weak", "medium", "strong"]
LEVEL_PARAMS = {"weak": (0.3, 1.0), "medium": (0.6, 0.1), "strong": (0.9, 0.01)}
K_LEVELS = [100, 200, 500, 1000]
RHO_TARGETS_1 = [0.1, 0.3, 0.5, 0.7, 0.95]
RHO_TARGETS_4 = [0.3, 0.7, 0.95]
ALPHA_OLD, ALPHA_NEW = 0.1, 0.005

# 「変化なし」「同値」とみなす相対差 (数値誤差の範囲)
TIE_RTOL = 1e-9


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def series(rows, metric: str, **match) -> Tuple[np.ndarray, np.ndarray]:
    """match に一致する行を rho 順に並べ, (rho, 値) を返す."""
    sel = [r for r in rows if all(str(r[k]) == str(v) for k, v in match.items())]
    sel.sort(key=lambda r: float(r["rho"]))
    return (
        np.array([float(r["rho"]) for r in sel]),
        np.array([float(r[metric]) for r in sel]),
    )


def nearest_index(grid: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(grid - target)))


def fmt(v: float) -> str:
    return f"{v:.4e}" if (v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4)) else f"{v:.5g}"


def rel_diff(new: float, old: float) -> float:
    return (new - old) / abs(old) if old != 0 else float("nan")


def fmt_rel(r: float) -> str:
    return "nan" if np.isnan(r) else f"{100 * r:+.3g}%"


def _sign_steps(values: np.ndarray) -> np.ndarray:
    """隣り合う点の差の符号 (相対差 TIE_RTOL 以内は 0)."""
    d = np.diff(values)
    scale = np.maximum(np.abs(values[1:]), np.abs(values[:-1]))
    s = np.sign(d)
    s[np.abs(d) <= TIE_RTOL * scale] = 0
    return s


def classify_monotone(x: np.ndarray, v: np.ndarray) -> str:
    """rho に対する単調性を判定する. 非単調なら内部の極値の位置 (格子点) を返す."""
    s = _sign_steps(v)
    nz = s[s != 0]
    if len(nz) == 0:
        return "一定"
    if np.all(nz > 0):
        return "単調増加" + (" (一部で横ばい)" if np.any(s == 0) else "")
    if np.all(nz < 0):
        return "単調減少" + (" (一部で横ばい)" if np.any(s == 0) else "")
    # 符号 (0 を除く) が変わる位置を極値とする. 横ばい区間はその手前の点で代表する
    extrema = []
    nz_idx = np.nonzero(s)[0]
    for a, b in zip(nz_idx[:-1], nz_idx[1:]):
        if s[a] != s[b]:
            kind = "極大" if s[a] > 0 else "極小"
            extrema.append(f"{kind} rho={x[b]:.4g}")
    return "非単調 (" + ", ".join(extrema) + ")"


def ordering_label(vals: Dict[str, float]) -> str:
    """水準の値を小さい順に並べた文字列 (相対差 TIE_RTOL 以内は '=')."""
    items = sorted(vals.items(), key=lambda kv: kv[1])
    out = items[0][0]
    for (_, prev), (name, v) in zip(items[:-1], items[1:]):
        tie = abs(v - prev) <= TIE_RTOL * max(abs(v), abs(prev))
        out += (" = " if tie else " < ") + name
    return out


def ordering_intervals(x: np.ndarray, per_level: Dict[str, np.ndarray]) -> List[Tuple[float, float, str]]:
    """rho 格子上で順序が一定の区間 (rho_lo, rho_hi, 順序) の列を返す."""
    labels = [ordering_label({k: per_level[k][i] for k in per_level}) for i in range(len(x))]
    out = []
    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or labels[i] != labels[start]:
            out.append((x[start], x[i - 1], labels[start]))
            start = i
    return out


def classify_in_K(vals: Sequence[float]) -> str:
    """K を大きくしたときの増減 (K 水準の順に並べた値)."""
    s = _sign_steps(np.asarray(vals))
    nz = s[s != 0]
    if len(nz) == 0:
        return "不変"
    if np.all(nz > 0):
        return "増加"
    if np.all(nz < 0):
        return "減少"
    return "非単調"


def summarize_over_rho(x: np.ndarray, labels: List[str]) -> str:
    """rho ごとのラベルを, 同じラベルが続く区間にまとめた文字列にする."""
    parts = []
    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or labels[i] != labels[start]:
            lo, hi = x[start], x[i - 1]
            rng = f"rho={lo:.4g}" if start == i - 1 else f"rho∈[{lo:.4g}, {hi:.4g}]"
            parts.append(f"{labels[start]} ({rng})")
            start = i
    return "全 rho で" + labels[0] if len(parts) == 1 else "; ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--exp4-old-dir", default="results/experiment_1_4_alpha_comparison",
        help="alpha=0.1 の実験 4 を E_B 等の列付きで計算した CSV の置き場所",
    )
    parser.add_argument("--out", default="results/experiment_1_4_alpha_comparison.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rd = args.results_dir
    old_sfx = f"_a{ALPHA_OLD:g}"
    e1_old = read_csv(os.path.join(rd, f"experiment_1{old_sfx}.csv"))
    e1_new = read_csv(os.path.join(rd, "experiment_1.csv"))
    e4_old = {lv: read_csv(os.path.join(args.exp4_old_dir, f"experiment_4_{lv}{old_sfx}.csv")) for lv in LEVELS}
    e4_new = {lv: read_csv(os.path.join(rd, f"experiment_4_{lv}.csv")) for lv in LEVELS}
    e4_orig = {lv: read_csv(os.path.join(rd, f"experiment_4_{lv}{old_sfx}.csv")) for lv in LEVELS}

    L: List[str] = []
    w = L.append
    w(f"# 実験 1・4: alpha={ALPHA_OLD:g} と alpha={ALPHA_NEW:g} の比較")
    w("")
    w("`scripts/compare_experiment_1_4_alpha.py` が CSV から自動生成した表. 相対差は "
      f"(alpha={ALPHA_NEW:g} の値 − alpha={ALPHA_OLD:g} の値) / |alpha={ALPHA_OLD:g} の値|.")
    w("")
    w("入力 CSV:")
    w("")
    w(f"- 実験 1 (alpha={ALPHA_OLD:g}): `{rd}/experiment_1{old_sfx}.csv`")
    w(f"- 実験 1 (alpha={ALPHA_NEW:g}, 既定値): `{rd}/experiment_1.csv`")
    w(f"- 実験 4 (alpha={ALPHA_OLD:g}): `{args.exp4_old_dir}/experiment_4_{{level}}{old_sfx}.csv` "
      "(E[B], E[S], E[I] を得るため列を加えて再計算したもの)")
    w(f"- 実験 4 (alpha={ALPHA_NEW:g}, 既定値): `{rd}/experiment_4_{{level}}.csv`")
    w("")

    # 再計算した alpha=0.1 の実験 4 が旧 CSV と一致するかの確認
    max_rel = 0.0
    for lv in LEVELS:
        for K in K_LEVELS:
            for m in ("P_block_arrival_stable", "E_W", "Cost", "ERP"):
                xo, vo = series(e4_orig[lv], m, K=K)
                xr, vr = series(e4_old[lv], m, K=K)
                assert np.allclose(xo, xr, rtol=0, atol=1e-12)
                max_rel = max(max_rel, float(np.max(np.abs(vr - vo) / np.abs(vo))))
    w(f"確認: 再計算した実験 4 (alpha={ALPHA_OLD:g}) の 4 指標と `{rd}/experiment_4_{{level}}{old_sfx}.csv` "
      f"の最大相対差 = {max_rel:.3e}")
    w("")

    # ---- 検査列の集計 ----
    w("## 0. 各点の検査 (alpha=0.005)")
    w("")
    w("| CSV | 点数 | 違反 | max cons_err | max flow_rel_err | min min_pi |")
    w("|---|---|---|---|---|---|")
    for name, rows in [("experiment_1.csv", e1_new)] + [
        (f"experiment_4_{lv}.csv", e4_new[lv]) for lv in LEVELS
    ]:
        nviol = sum(r["check"] != "ok" for r in rows)
        w(f"| {name} | {len(rows)} | {nviol} | "
          f"{max(float(r['cons_err']) for r in rows):.3e} | "
          f"{max(float(r['flow_rel_err']) for r in rows):.3e} | "
          f"{min(float(r['min_pi']) for r in rows):.3e} |")
    w("")

    # ---- 1. 実験 1 の表 ----
    w("## 1. 実験 1 (K=200): 代表的な rho での値")
    w("")
    for lv in LEVELS:
        d, s = LEVEL_PARAMS[lv]
        w(f"### {lv} (delta={d}, sigma={s})")
        w("")
        w(f"| rho (格子点) | 指標 | alpha={ALPHA_OLD:g} | alpha={ALPHA_NEW:g} | 相対差 |")
        w("|---|---|---|---|---|")
        for target in RHO_TARGETS_1:
            for m, label in METRICS:
                xo, vo = series(e1_old, m, burst_name=lv)
                xn, vn = series(e1_new, m, burst_name=lv)
                i = nearest_index(xo, target)
                j = nearest_index(xn, target)
                assert abs(xo[i] - xn[j]) < 1e-12
                w(f"| {xo[i]:.4g} | {label} | {fmt(vo[i])} | {fmt(vn[j])} | {fmt_rel(rel_diff(vn[j], vo[i]))} |")
        w("")

    # ---- 2. 実験 4 の表 ----
    w("## 2. 実験 4: 代表的な rho での値")
    w("")
    for lv in LEVELS:
        d, s = LEVEL_PARAMS[lv]
        for K in K_LEVELS:
            w(f"### {lv} (delta={d}, sigma={s}), K={K}")
            w("")
            w(f"| rho (格子点) | 指標 | alpha={ALPHA_OLD:g} | alpha={ALPHA_NEW:g} | 相対差 |")
            w("|---|---|---|---|---|")
            for target in RHO_TARGETS_4:
                for m, label in METRICS:
                    xo, vo = series(e4_old[lv], m, K=K)
                    xn, vn = series(e4_new[lv], m, K=K)
                    i = nearest_index(xo, target)
                    j = nearest_index(xn, target)
                    assert abs(xo[i] - xn[j]) < 1e-12
                    w(f"| {xo[i]:.4g} | {label} | {fmt(vo[i])} | {fmt(vn[j])} | {fmt_rel(rel_diff(vn[j], vo[i]))} |")
            w("")

    # ---- 3. 判定 ----
    w("## 3. 機械的な判定")
    w("")
    w(f"判定の規則: 隣り合う格子点の相対差が {TIE_RTOL:g} 以下なら「横ばい / 同値」とみなす. "
      "極値の位置は格子点 (15 点) の rho で示す.")
    w("")

    w("### 3.1 実験 1: rho に対する単調性")
    w("")
    w(f"| 水準 | 指標 | alpha={ALPHA_OLD:g} | alpha={ALPHA_NEW:g} | 同じか |")
    w("|---|---|---|---|---|")
    for lv in LEVELS:
        for m, label in METRICS:
            co = classify_monotone(*series(e1_old, m, burst_name=lv))
            cn = classify_monotone(*series(e1_new, m, burst_name=lv))
            w(f"| {lv} | {label} | {co} | {cn} | {'同じ' if co == cn else '異なる'} |")
    w("")

    w("### 3.2 実験 1: バースト水準の間の大小の順序 (小さい順)")
    w("")
    w("rho 格子上で順序が一定の区間を並べる. 区間が 2 つ以上あれば, 隣り合う区間の境の格子点の間で順序が入れ替わる.")
    w("")
    w(f"| 指標 | alpha | 順序が一定の区間 | 入れ替わり |")
    w("|---|---|---|---|")
    for m, label in METRICS:
        for alpha, rows in [(ALPHA_OLD, e1_old), (ALPHA_NEW, e1_new)]:
            per = {}
            for lv in LEVELS:
                x, v = series(rows, m, burst_name=lv)
                per[lv] = v
            iv = ordering_intervals(x, per)
            text = "<br>".join(
                (f"rho={lo:.4g}" if lo == hi else f"rho∈[{lo:.4g}, {hi:.4g}]") + f": {lab}"
                for lo, hi, lab in iv
            )
            swaps = (
                "なし" if len(iv) == 1 else
                ", ".join(f"{a[1]:.4g}〜{b[0]:.4g}" for a, b in zip(iv[:-1], iv[1:]))
            )
            w(f"| {label} | {alpha:g} | {text} | {swaps} |")
    w("")

    w("### 3.3 実験 4: rho に対する単調性 (K 別)")
    w("")
    w(f"各セルは「alpha={ALPHA_OLD:g} → alpha={ALPHA_NEW:g}」. 同じなら 1 つだけ書く.")
    w("")
    w("| 水準 | 指標 | " + " | ".join(f"K={K}" for K in K_LEVELS) + " |")
    w("|---|---|" + "---|" * len(K_LEVELS))
    for lv in LEVELS:
        for m, label in METRICS:
            cells = []
            for K in K_LEVELS:
                co = classify_monotone(*series(e4_old[lv], m, K=K))
                cn = classify_monotone(*series(e4_new[lv], m, K=K))
                cells.append(co if co == cn else f"{co} → **{cn}**")
            w(f"| {lv} | {label} | " + " | ".join(cells) + " |")
    w("")

    w("### 3.4 実験 4: K を大きくしたときの増減 (K=100→200→500→1000)")
    w("")
    w("rho ごとに K 4 水準の値の並びを判定し, 同じ判定が続く rho の区間にまとめる.")
    w("")
    w(f"| 水準 | 指標 | alpha={ALPHA_OLD:g} | alpha={ALPHA_NEW:g} |")
    w("|---|---|---|---|")
    for lv in LEVELS:
        for m, label in METRICS:
            cells = []
            for rows in (e4_old[lv], e4_new[lv]):
                by_K = [series(rows, m, K=K) for K in K_LEVELS]
                x = by_K[0][0]
                labels = [classify_in_K([v[i] for _, v in by_K]) for i in range(len(x))]
                cells.append(summarize_over_rho(x, labels))
            w(f"| {lv} | {label} | {cells[0]} | {cells[1]} |")
    w("")

    w("### 3.5 実験 4: K=500 と K=1000 の相対差 |v(1000) − v(500)| / |v(1000)|")
    w("")
    w(f"| 水準 | 指標 | alpha={ALPHA_OLD:g}: rho 全体の最大 (rho) | alpha={ALPHA_OLD:g}: rho=0.95 "
      f"| alpha={ALPHA_NEW:g}: rho 全体の最大 (rho) | alpha={ALPHA_NEW:g}: rho=0.95 |")
    w("|---|---|---|---|---|---|")
    for lv in LEVELS:
        for m, label in METRICS:
            cells = []
            for rows in (e4_old[lv], e4_new[lv]):
                x, v500 = series(rows, m, K=500)
                _, v1000 = series(rows, m, K=1000)
                rel = np.abs(v1000 - v500) / np.abs(v1000)
                k = int(np.argmax(rel))
                cells += [f"{rel[k]:.3e} ({x[k]:.4g})", f"{rel[-1]:.3e}"]
            w(f"| {lv} | {label} | " + " | ".join(cells) + " |")
    w("")

    # ---- 4. キャプション用 ----
    w("## 4. キャプション用の固定パラメータ")
    w("")
    w("両実験で共通: c=20, b=5, 1/mu=1 (mu=1), 1/beta=200 (beta=0.005), "
      f"alpha={ALPHA_NEW:g} (1/alpha={1 / ALPHA_NEW:g}). 旧図は alpha={ALPHA_OLD:g} (1/alpha={1 / ALPHA_OLD:g}). "
      "rho = lambda_bar / (c b mu) は [0.1, 0.95] の線形 15 点. 定常分布は GTH 法.")
    w("")
    w("- バースト水準 (対称 2 位相 MMPP): " + ", ".join(
        f"{lv}: delta={LEVEL_PARAMS[lv][0]}, sigma={LEVEL_PARAMS[lv][1]}" for lv in LEVELS))
    w("- 実験 1: K=200")
    w("- 実験 4: K ∈ {" + ", ".join(map(str, K_LEVELS)) + "}")
    w("- 状態数 N (実験 1, K=200): " + str(sorted({int(r['N_states']) for r in e1_new})))
    w("- 状態数 N (実験 4): " + ", ".join(
        f"K={K}: {sorted({int(r['N_states']) for r in e4_new['weak'] if int(r['K']) == K})}"
        for K in K_LEVELS))
    w("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"比較表を保存しました: {args.out}")


if __name__ == "__main__":
    main()
