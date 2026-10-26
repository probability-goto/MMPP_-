#!/usr/bin/env python
"""比較の地図の改善の仕組み: Base と Predictive の最良点を, フェーズの切り替わりの観点で比べる.

results/compare_map/ の CSV と summary.csv を読み, マスごとに次を計算する.
    1. 最良点どうしの E[W], Cost, P_block の変化 (Predictive / Base).
    2. 1 周期 (閑散期と繁忙期の 1 組. 単位時間あたり sigma/2 回) あたりの回数.
       Base: セットアップの完了 alpha E[S]. Predictive: セットアップの完了, 事前セットアップで起動した台数,
       セットアップの取り消し, Delayoff (いずれも CSV の診断列).
    3. Base の最良点の準静的な近似: 同じ (alpha, beta*) で, 閑散期の到着率 lambda_0 と繁忙期の到着率 lambda_1 を
       それぞれポアソン到着として解き, 時間平均を合わせる (各フェーズの時間の割合は 1/2).
       E[W]_qs = (E[N]_0 + E[N]_1) / (lambda_eff,0 + lambda_eff,1), Cost_qs = (Cost_0 + Cost_1) / 2,
       P_qs は到着の割合で重み付け. Base との差をフェーズの切り替わりで生じる分とみなす
       (フェーズが十分長いときの近似なので, r1 が小さいマスでは意味を持たない).
新しく解くのは 3. のポアソン到着の点 (マスあたり 2 点) だけ.

出力: results/compare_map/mechanism.csv, results/compare_map/mechanism.md, figures/compare_map_mechanism.{pdf,png}

使用例:
    python scripts/compare_map_mechanism.py
"""
import csv
import os
from typing import Dict, List

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
    "base_setups_per_cycle", "pred_setups_per_cycle", "pred_p1_launch_per_cycle",
    "pred_cancel_per_cycle", "pred_delayoff_per_cycle",
    "qs_E_W", "qs_Cost", "qs_P", "qs_ERP",
    "trans_share_P", "trans_share_EW", "trans_share_ERP", "improvement_over_trans",
]


def _f(r, k) -> float:
    return float(r[k])


def _find(rows: List[dict], series: str, beta: float, nt: int, g: float) -> dict:
    hit = [r for r in rows if r["series"] == series and _f(r, "beta") == beta
           and int(r["n_target"]) == nt and _f(r, "gamma") == g]
    if not hit:
        raise KeyError(f"{series} beta={beta} n_target={nt} gamma={g} の行がない")
    return hit[0]


def quasi_static(rho_B: float, beta: float) -> Dict[str, float]:
    """閑散期と繁忙期をそれぞれポアソン到着として解き, 時間平均を合わせる."""
    rho = d.rho_of(rho_B, cm.DELTA)
    ph = [d.solve_row("qs", rho * fac, 0.0, 1.0, alpha=cm.ALPHA, beta=beta, r1="")
          for fac in (1 - cm.DELTA, 1 + cm.DELTA)]
    en = sum(_f(r, "E_N") for r in ph)
    lam_eff = sum(_f(r, "lambda_eff") for r in ph)
    w_arr = [(1 - cm.DELTA) / 2, (1 + cm.DELTA) / 2]
    out = dict(qs_E_W=en / lam_eff, qs_Cost=sum(_f(r, "Cost") for r in ph) / 2,
               qs_P=sum(w * _f(r, "P_block_arrival_stable") for w, r in zip(w_arr, ph)))
    out["qs_ERP"] = out["qs_E_W"] * out["qs_Cost"]
    return out


def analyze() -> List[dict]:
    summary = list(csv.DictReader(open(os.path.join(cm.OUT_DIR, "summary.csv"), encoding="utf-8")))
    cell_rows = {rb: read_rows(cm.csv_path(rb)) for rb in cm.RHO_B_LEVELS}
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
        diag = (lambda k: _f(p, k) / cyc if p[k] not in ("", None) else 0.0)
        row = dict(
            rho_B=rho_B, r1=r1, sigma=sigma, improvement=_f(s, "improvement"),
            base_beta=_f(b, "beta"), base_E_W=_f(b, "E_W"), base_Cost=_f(b, "Cost"),
            base_P=_f(b, "P_block_arrival_stable"), base_ERP=_f(b, "ERP"),
            pred_beta=_f(p, "beta"), pred_n_target=int(p["n_target"]), pred_gamma=_f(p, "gamma"),
            pred_E_W=_f(p, "E_W"), pred_Cost=_f(p, "Cost"), pred_P=_f(p, "P_block_arrival_stable"),
            pred_ERP=_f(p, "ERP"),
            base_setups_per_cycle=cm.ALPHA * _f(b, "E_S") / cyc,
            pred_setups_per_cycle=(diag("setup_completion_rate") if p["series"] != "Base"
                                   else cm.ALPHA * _f(p, "E_S") / cyc),
            pred_p1_launch_per_cycle=diag("p1_launch_rate"),
            pred_cancel_per_cycle=diag("setup_cancel_rate"),
            pred_delayoff_per_cycle=diag("delayoff_rate"),
        )
        row["dEW"] = row["pred_E_W"] / row["base_E_W"] - 1
        row["dCost"] = row["pred_Cost"] / row["base_Cost"] - 1
        row["P_ratio"] = row["pred_P"] / row["base_P"]
        row.update(quasi_static(rho_B, row["base_beta"]))
        row["trans_share_P"] = 1 - row["qs_P"] / row["base_P"]
        row["trans_share_EW"] = 1 - row["qs_E_W"] / row["base_E_W"]
        row["trans_share_ERP"] = 1 - row["qs_ERP"] / row["base_ERP"]
        row["improvement_over_trans"] = (row["improvement"] / row["trans_share_ERP"]
                                         if row["trans_share_ERP"] > 0 else float("nan"))
        out.append(row)
        print(f"rho_B={rho_B:g}, r1={r1:.3g} 完了")
    return out


def plot(rows: List[dict], fig_dir: str = cm.FIG_DIR) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = {0.6: "tab:green", 0.8: "tab:blue", 1.12: "tab:red"}
    panels = [
        ("improvement", "ERP improvement [%]", 100),
        ("trans_share_ERP", "Base: share of ERP from phase switches [%]", 100),
        ("dEW", r"Pred/Base $-1$ [%] (solid: $E[W]$, dashed: Cost)", 100),
        ("pred_p1_launch_per_cycle", "Pred: P1 launches per cycle", 1),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (key, ylabel, scale) in zip(axes.flat, panels):
        for rho_B in cm.RHO_B_LEVELS:
            rr = sorted([r for r in rows if r["rho_B"] == rho_B], key=lambda r: r["r1"])
            xs = [r["r1"] for r in rr]
            ax.plot(xs, [scale * r[key] for r in rr], marker="o", markersize=4,
                    color=color[rho_B], label=rf"$\rho_B={rho_B:g}$")
            if key == "dEW":
                ax.plot(xs, [scale * r["dCost"] for r in rr], marker="s", markersize=3,
                        linestyle="--", color=color[rho_B])
        ax.set_xscale("log")
        ax.axhline(0, color="0.5", linewidth=0.8)
        ax.set_xlabel(r"$r_1=\alpha/\sigma$", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    paths = save_figure(fig, fig_dir, FIG_STEM)
    plt.close(fig)
    return paths


def report(rows: List[dict], figs: List[str]) -> str:
    lines = [
        "# 比較の地図の改善の仕組み", "",
        "results/compare_map/ の CSV と summary.csv から scripts/compare_map_mechanism.py で生成. "
        "定義は scripts/compare_map_mechanism.py の docstring.", "",
        "- 改善率: 地図の改善率 (Base の ERP 最小値に対する, P ≤ P_ref の Predictive の ERP 最小値の減少率).",
        "- ΔE[W], ΔCost: Predictive の最良点 / Base の最良点 − 1. P 比: Predictive / Base.",
        "- 1 周期あたり: 単位時間あたりの回数を 0→1 の切り替わりの率 σ/2 で割ったもの.",
        "- 切り替わりの分: 1 − (準静的な近似の値) / (Base の値). r1 が小さいマスでは近似が成り立たないので参考.", "",
        "| ρ_B | r1 | 改善率 | ΔE[W] | ΔCost | P 比 | (n_target, γ) | 起動/周期 Base | 起動/周期 Pred | P1 起動/周期 | 取り消し/周期 | Delayoff/周期 | 切り替わりの分 P | E[W] | ERP | 改善率 / ERP の切り替わりの分 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["rho_B"], r["r1"])):
        nt = f"({r['pred_n_target']}, {r['pred_gamma']:g})"
        lines.append(
            f"| {r['rho_B']:g} | {r['r1']:.3g} | {100 * r['improvement']:.2f}% | "
            f"{100 * r['dEW']:+.2f}% | {100 * r['dCost']:+.2f}% | {r['P_ratio']:.3f} | {nt} | "
            f"{r['base_setups_per_cycle']:.2f} | {r['pred_setups_per_cycle']:.2f} | "
            f"{r['pred_p1_launch_per_cycle']:.2f} | {r['pred_cancel_per_cycle']:.2f} | "
            f"{r['pred_delayoff_per_cycle']:.2f} | {100 * r['trans_share_P']:.1f}% | "
            f"{100 * r['trans_share_EW']:.1f}% | {100 * r['trans_share_ERP']:.1f}% | "
            f"{r['improvement_over_trans']:.2f} |")
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
