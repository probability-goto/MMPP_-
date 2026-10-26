#!/usr/bin/env python
"""比較の地図の改善の仕組み: Base と Predictive の最良点を, フェーズごとの Delayoff の観点で比べる.

results/compare_map/ の CSV と summary.csv を読み, マスごとに次を計算する.
    1. 最良点どうしの E[W], Cost, P_block の変化 (Predictive / Base).
    2. 準静的な近似 (各フェーズをその到着率のポアソン到着として解き, 時間平均を合わせる. 各フェーズの時間の
       割合は 1/2. フェーズが十分長いときの近似):
       - 閑散期の到着率 lambda_0 のポアソン到着で ERP を最小にする beta (beta_qs. 対数格子で探す).
       - Base の最良点の beta* と, Predictive の最良点の形 (閑散期は gamma beta, 繁忙期は n_target = c なら
         保護で Delayoff が止まるので 0, n_target = 0 なら beta) の準静的な ERP. その差を準静的な改善率とする.
         0 < n_target < c では保護が台数によって効いたり効かなかったりするので計算しない (nan).
       - Base の繁忙期の 1 周期あたりの起動回数 (繁忙期のポアソン到着での alpha E[S] を周期あたりに直したもの).
    3. 実際の系 (MMPP) で Base の beta を beta_qs にした点の E[W], Cost, ERP (新しく解く).
    4. 1 周期 (閑散期と繁忙期の 1 組. 単位時間あたり sigma/2 回) あたりの回数.
       Base: セットアップの完了 alpha E[S]. Predictive: セットアップの完了, 事前セットアップで起動した台数.
新しく解くのは 2. のポアソン到着の点と 3. の点だけ.

出力: results/compare_map/mechanism.csv, results/compare_map/mechanism.md, figures/compare_map_mechanism.{pdf,png}

使用例:
    python scripts/compare_map_mechanism.py
"""
import csv
import os
from typing import Dict, List

import numpy as np

from mmpp_experiments import design as d
from mmpp_experiments.csvio import read_rows
from mmpp_experiments.plotting import save_figure

try:
    import compare_map as cm
except ImportError:
    import scripts.compare_map as cm

OUT_CSV = os.path.join(cm.OUT_DIR, "mechanism.csv")
OUT_MD = os.path.join(cm.OUT_DIR, "mechanism.md")
FIG_STEM = "compare_map_mechanism"

FIELDS = [
    "rho_B", "r1", "sigma", "improvement",
    "base_beta", "base_E_W", "base_Cost", "base_P", "base_ERP",
    "pred_beta", "pred_n_target", "pred_gamma", "pred_E_W", "pred_Cost", "pred_P", "pred_ERP",
    "dEW", "dCost", "P_ratio",
    "beta_qs", "small_E_W", "small_Cost", "small_ERP",
    "qs_base_ERP", "qs_pred_ERP", "qs_improvement",
    "base_setups_per_cycle", "base_qs_burst_setups_per_cycle",
    "pred_setups_per_cycle", "pred_p1_launch_per_cycle",
]

QS_BETA_GRID = [float(x) for x in np.logspace(-6, -2, 33)]


def _f(r, k) -> float:
    return float(r[k])


def _find(rows: List[dict], series: str, beta: float, nt: int, g: float) -> dict:
    hit = [r for r in rows if r["series"] == series and _f(r, "beta") == beta
           and int(r["n_target"]) == nt and _f(r, "gamma") == g]
    if not hit:
        raise KeyError(f"{series} beta={beta} n_target={nt} gamma={g} の行がない")
    return hit[0]


def _phase(rho_B: float, fac: float, beta: float) -> dict:
    return d.solve_row("qs", d.rho_of(rho_B, cm.DELTA) * fac, 0.0, 1.0, alpha=cm.ALPHA, beta=beta,
                       r1="")


def quasi_static(rho_B: float, beta_quiet: float, beta_burst: float) -> Dict[str, float]:
    """閑散期と繁忙期をそれぞれポアソン到着として解き, 時間平均を合わせた E[W], Cost, ERP."""
    ph = [_phase(rho_B, 1 - cm.DELTA, beta_quiet), _phase(rho_B, 1 + cm.DELTA, beta_burst)]
    ew = sum(_f(r, "E_N") for r in ph) / sum(_f(r, "lambda_eff") for r in ph)
    cost = sum(_f(r, "Cost") for r in ph) / 2
    return dict(E_W=ew, Cost=cost, ERP=ew * cost, burst_E_S=_f(ph[1], "E_S"))


def quiet_optimal_beta(rho_B: float) -> float:
    """閑散期の到着率のポアソン到着で ERP を最小にする beta (QS_BETA_GRID の中)."""
    return min(QS_BETA_GRID, key=lambda b: _f(_phase(rho_B, 1 - cm.DELTA, b), "ERP"))


def analyze() -> List[dict]:
    summary = list(csv.DictReader(open(os.path.join(cm.OUT_DIR, "summary.csv"), encoding="utf-8")))
    cell_rows = {rb: read_rows(cm.csv_path(rb)) for rb in cm.RHO_B_LEVELS}
    beta_qs = {rb: quiet_optimal_beta(rb) for rb in cm.RHO_B_LEVELS}
    out = []
    for s in summary:
        rho_B, r1, sigma = _f(s, "rho_B"), _f(s, "r1"), _f(s, "sigma")
        rows = [r for r in cell_rows[rho_B] if _f(r, "r1") == r1]
        b = _find(rows, "Base", _f(s, "base_beta"), 0, 1.0)
        if s["pred_is_base"] == "True":
            p = _find(rows, "Base", _f(s, "pred_beta"), 0, 1.0)
        else:
            p = _find(rows, "Predictive", _f(s, "pred_beta"), int(s["pred_n_target"]),
                      _f(s, "pred_gamma"))
        cyc = sigma / 2
        nt, g, pb = int(p["n_target"]), _f(p, "gamma"), _f(p, "beta")
        small = d.solve_row("small", d.rho_of(rho_B, cm.DELTA), cm.DELTA, sigma, alpha=cm.ALPHA,
                            beta=beta_qs[rho_B])
        qb = quasi_static(rho_B, _f(b, "beta"), _f(b, "beta"))
        # 保護が繁忙期の Delayoff を常に止めるのは n_target = c のときだけ. 0 < n_target < c は表せない
        qp = (quasi_static(rho_B, g * pb, 1e-12 if nt > 0 else pb)
              if nt in (0, cm.BASELINE["c"]) else dict(ERP=float("nan")))
        row = dict(
            rho_B=rho_B, r1=r1, sigma=sigma, improvement=_f(s, "improvement"),
            base_beta=_f(b, "beta"), base_E_W=_f(b, "E_W"), base_Cost=_f(b, "Cost"),
            base_P=_f(b, "P_block_arrival_stable"), base_ERP=_f(b, "ERP"),
            pred_beta=pb, pred_n_target=nt, pred_gamma=g,
            pred_E_W=_f(p, "E_W"), pred_Cost=_f(p, "Cost"), pred_P=_f(p, "P_block_arrival_stable"),
            pred_ERP=_f(p, "ERP"),
            beta_qs=beta_qs[rho_B], small_E_W=_f(small, "E_W"), small_Cost=_f(small, "Cost"),
            small_ERP=_f(small, "ERP"),
            qs_base_ERP=qb["ERP"], qs_pred_ERP=qp["ERP"],
            qs_improvement=(qb["ERP"] - qp["ERP"]) / qb["ERP"],
            base_setups_per_cycle=cm.ALPHA * _f(b, "E_S") / cyc,
            base_qs_burst_setups_per_cycle=cm.ALPHA * qb["burst_E_S"] * 0.5 / cyc,
            pred_setups_per_cycle=(_f(p, "setup_completion_rate") / cyc if p["series"] != "Base"
                                   else cm.ALPHA * _f(p, "E_S") / cyc),
            pred_p1_launch_per_cycle=(_f(p, "p1_launch_rate") / cyc
                                      if p["p1_launch_rate"] not in ("", None) else 0.0),
        )
        row["dEW"] = row["pred_E_W"] / row["base_E_W"] - 1
        row["dCost"] = row["pred_Cost"] / row["base_Cost"] - 1
        row["P_ratio"] = row["pred_P"] / row["base_P"]
        out.append(row)
        print(f"rho_B={rho_B:g}, r1={r1:.3g} 完了")
    return out


def plot(rows: List[dict], fig_dir: str = cm.FIG_DIR) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = {0.6: "tab:green", 0.8: "tab:blue", 1.12: "tab:red"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for rho_B in cm.RHO_B_LEVELS:
        rr = sorted([r for r in rows if r["rho_B"] == rho_B], key=lambda r: r["r1"])
        xs = [r["r1"] for r in rr]
        c = color[rho_B]
        axes[0, 0].plot(xs, [100 * r["improvement"] for r in rr], marker="o", markersize=4,
                        color=c, label=rf"$\rho_B={rho_B:g}$")
        axes[0, 0].plot(xs, [100 * r["qs_improvement"] for r in rr], marker="s", markersize=3,
                        linestyle="--", color=c)
        axes[0, 1].plot(xs, [r["base_beta"] for r in rr], marker="o", markersize=4, color=c)
        axes[1, 0].plot(xs, [100 * r["dEW"] for r in rr], marker="o", markersize=4, color=c)
        axes[1, 0].plot(xs, [100 * r["dCost"] for r in rr], marker="s", markersize=3,
                        linestyle="--", color=c)
        axes[1, 1].plot(xs, [r["pred_p1_launch_per_cycle"] for r in rr], marker="o",
                        markersize=4, color=c)
    labels = ["ERP improvement [%] (solid: actual, dashed: quasi-static)",
              r"Base $\beta^*$ (dotted: quiet-phase optimum, quasi-static)",
              r"Pred/Base $-1$ [%] (solid: $E[W]$, dashed: Cost)",
              "Pred: P1 launches per cycle"]
    for ax, lab in zip(axes.flat, labels):
        ax.set_xscale("log")
        ax.set_xlabel(r"$r_1=\alpha/\sigma$", fontsize=14)
        ax.set_ylabel(lab, fontsize=11)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=10)
    axes[0, 1].set_yscale("log")
    # beta_qs は rho_B によらず同じ格子点になる (閑散期の到着率の違いが格子の間隔より小さい)
    for v in sorted({r["beta_qs"] for r in rows}):
        axes[0, 1].axhline(v, color="0.4", linestyle=":", linewidth=1)
    for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
        ax.axhline(0, color="0.5", linewidth=0.8)
    handles, labs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    paths = save_figure(fig, fig_dir, FIG_STEM)
    plt.close(fig)
    return paths


def report(rows: List[dict], figs: List[str]) -> str:
    lines = [
        "# 比較の地図の改善の仕組み", "",
        "results/compare_map/ の CSV と summary.csv から scripts/compare_map_mechanism.py で生成. "
        "定義は scripts/compare_map_mechanism.py の docstring.", "",
        "- 改善率: 地図の改善率. 準静的な改善率: Base の最良点の β* と Predictive の最良点の形の準静的な ERP の差.",
        "- β_qs: 閑散期の到着率のポアソン到着で ERP を最小にする β. 「小さい β の Base」: 実際の系で Base の β を β_qs にした点.",
        "- ΔE[W], ΔCost: Predictive の最良点 / Base の最良点 − 1.",
        "- 1 周期あたり: 単位時間あたりの回数を 0→1 の切り替わりの率 σ/2 で割ったもの. 繁忙期 (準静的) は Base の β* での繁忙期の起動回数.", "",
        "| ρ_B | r1 | 改善率 | 準静的な改善率 | Base β* | β_qs | Base E[W] / Cost | 小さい β の Base E[W] / Cost | Pred E[W] / Cost | (n_target, γ) | 起動/周期 Base | うち繁忙期 (準静的) | 起動/周期 Pred | P1 起動/周期 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["rho_B"], r["r1"])):
        lines.append(
            f"| {r['rho_B']:g} | {r['r1']:.3g} | {100 * r['improvement']:.2f}% | "
            f"{100 * r['qs_improvement']:.2f}% | {r['base_beta']:.3g} | {r['beta_qs']:.3g} | "
            f"{r['base_E_W']:.4f} / {r['base_Cost']:.4f} | {r['small_E_W']:.4f} / {r['small_Cost']:.4f} | "
            f"{r['pred_E_W']:.4f} / {r['pred_Cost']:.4f} | ({r['pred_n_target']}, {r['pred_gamma']:g}) | "
            f"{r['base_setups_per_cycle']:.2f} | {r['base_qs_burst_setups_per_cycle']:.2f} | "
            f"{r['pred_setups_per_cycle']:.2f} | {r['pred_p1_launch_per_cycle']:.2f} |")
    lines += ["", "## 図", ""] + [f"- {f}" for f in figs] + [""]
    text = "\n".join(lines)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main() -> None:
    rows = analyze()
    d.write_rows(OUT_CSV, rows, FIELDS)
    figs = plot(rows)
    print(report(rows, figs))


if __name__ == "__main__":
    main()
