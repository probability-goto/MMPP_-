#!/usr/bin/env python
"""実験 1 (ベース) と実験 1-P (Predictive) の直接比較図を生成.

両方の CSV (scripts/experiment_1_traffic.py, scripts/experiment_1P_traffic.py
の出力) を読み込み, 同じ 4 指標をベース (実線) vs Predictive (破線) で重ね
描きする. 3 バースト水準 x 4 指標の 2x2 サブプロット図を作成し, 代表的な
rho 点でベースに対する Predictive の改善率を標準出力にサマライズする.

使用例:
    python scripts/compare_experiment_1_vs_1P.py \\
        --base-csv results/experiment_1.csv \\
        --pred-csv results/experiment_1P_nt10_g5.0.csv \\
        --meta "n_target=10, gamma=5.0"
"""
import argparse
import csv
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(filepath: str) -> List[Dict]:
    """CSV を辞書リストとして読み込む (数値列は float に変換する)."""
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            for k in row:
                if k != "burst_name":
                    try:
                        row[k] = float(row[k])
                    except ValueError:
                        pass
            rows.append(row)
    return rows


def plot_comparison(
    base_rows: List[Dict], pred_rows: List[Dict], out_path: str, meta: str = ""
) -> None:
    """ベース vs Predictive の 4 指標比較図を生成する."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Experiment 1 vs 1-P: Base vs Predictive\n({meta})", fontsize=13)

    colors = {"weak": "tab:blue", "medium": "tab:orange", "strong": "tab:red"}
    labels = {
        "weak": r"weak ($\delta$=0.3, $\sigma$=1.0)",
        "medium": r"medium ($\delta$=0.6, $\sigma$=0.1)",
        "strong": r"strong ($\delta$=0.9, $\sigma$=0.01)",
    }
    metrics_config = [
        (axes[0, 0], "P_block_arrival", r"$P^\mathrm{arrival}_\mathrm{block}$", True),
        (axes[0, 1], "E_W", r"$E[W]$", False),
        (axes[1, 0], "Cost", "Cost", False),
        (axes[1, 1], "ERP", "ERP", False),
    ]

    for burst_name in ["weak", "medium", "strong"]:
        base_filtered = [r for r in base_rows if r["burst_name"] == burst_name]
        base_filtered.sort(key=lambda r: r["rho"])
        base_rhos = [r["rho"] for r in base_filtered]

        pred_filtered = [r for r in pred_rows if r["burst_name"] == burst_name]
        pred_filtered.sort(key=lambda r: r["rho"])
        pred_rhos = [r["rho"] for r in pred_filtered]

        for ax, metric_name, ylabel, log_y in metrics_config:
            base_values = [r[metric_name] for r in base_filtered]
            pred_values = [r[metric_name] for r in pred_filtered]

            ax.plot(
                base_rhos, base_values,
                color=colors[burst_name], linestyle="-",
                marker="o", markersize=4,
                label=f"{labels[burst_name]} (Base)",
            )
            ax.plot(
                pred_rhos, pred_values,
                color=colors[burst_name], linestyle="--",
                marker="s", markersize=4, alpha=0.7,
                label=f"{labels[burst_name]} (Predictive)",
            )
            ax.set_xlabel(r"$\rho$")
            ax.set_ylabel(ylabel)
            if log_y:
                ax.set_yscale("log")
            ax.grid(True, alpha=0.3)

    axes[0, 0].legend(loc="best", fontsize=7, ncol=1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    plt.savefig(pdf_path, dpi=120, bbox_inches="tight")
    print(f"比較図保存: {out_path}, {pdf_path}")
    plt.close(fig)


def print_improvement_summary(base_rows: List[Dict], pred_rows: List[Dict]) -> None:
    """代表的な rho 点でベースに対する Predictive の改善率をサマライズして出力する.

    改善率がマイナス (Predictive が悪化) の場合もあり得る. 特に Cost は
    事前セットアップの分だけ Predictive で増加する可能性が高く, これは
    想定内である (ERP でトレードオフを評価する).
    """
    print("\n=== ベース vs Predictive の改善率サマリ ===")
    print(
        f"{'burst':>8s} {'rho':>6s} {'metric':>18s} "
        f"{'base':>12s} {'pred':>12s} {'改善率':>10s}"
    )
    for burst_name in ["weak", "medium", "strong"]:
        for rho_target in [0.3, 0.5, 0.7, 0.9]:
            base_pt = next(
                (
                    r for r in base_rows
                    if r["burst_name"] == burst_name
                    and abs(r["rho"] - rho_target) < 0.05
                ),
                None,
            )
            pred_pt = next(
                (
                    r for r in pred_rows
                    if r["burst_name"] == burst_name
                    and abs(r["rho"] - rho_target) < 0.05
                ),
                None,
            )
            if base_pt is None or pred_pt is None:
                continue
            for metric in ["P_block_arrival", "E_W", "Cost", "ERP"]:
                base_val = base_pt[metric]
                pred_val = pred_pt[metric]
                if base_val > 0:
                    improvement = (base_val - pred_val) / base_val * 100
                    print(
                        f"{burst_name:>8s} {rho_target:>6.2f} {metric:>18s} "
                        f"{base_val:>12.4g} {pred_val:>12.4g} {improvement:>9.1f}%"
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", required=True, help="ベース実験 1 の CSV パス")
    parser.add_argument("--pred-csv", required=True, help="Predictive 実験 1-P の CSV パス")
    parser.add_argument(
        "--out", default="figures/compare_experiment_1_vs_1P.png", help="出力パス"
    )
    parser.add_argument("--meta", default="", help="図タイトルの補足情報")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_rows = load_csv(args.base_csv)
    pred_rows = load_csv(args.pred_csv)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_comparison(base_rows, pred_rows, args.out, args.meta)
    print_improvement_summary(base_rows, pred_rows)


if __name__ == "__main__":
    main()
