#!/usr/bin/env python
"""実験 4 (ベース) と実験 4-P (Predictive) の直接比較図を生成.

両方の CSV (scripts/experiment_4_K_sensitivity.py,
scripts/experiment_4P_K_sensitivity.py の出力) を読み込み, 同じ 4 指標を
ベース (実線) vs Predictive (破線) で重ね描きする. K 4 水準 × 4 指標の
比較図を作成し, 代表的な rho 点でベースに対する Predictive の改善率を
標準出力にサマライズする.

比較図は指標 (4 枚) × K (4 水準) の grid で構成する.

使用例:
    python scripts/compare_experiment_4_vs_4P.py \\
        --burst medium \\
        --base-csv results/experiment_4_medium.csv \\
        --pred-csv results/experiment_4P_medium_nt10_g5.0.csv \\
        --meta "medium burst, n_target=10, gamma=5.0"
"""
import argparse
import csv
import os
from typing import Dict, List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


K_LEVELS = [100, 200, 500, 1000]

K_STYLE = {
    100: dict(color="tab:blue"),
    200: dict(color="tab:orange"),
    500: dict(color="tab:green"),
    1000: dict(color="tab:red"),
}


def load_csv(filepath: str) -> List[Dict]:
    """CSV を辞書リストとして読み込む (数値列は float に変換する)."""
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            for k in row:
                if k not in ("burst_name",):
                    try:
                        row[k] = float(row[k])
                    except ValueError:
                        pass
            rows.append(row)
    return rows


def plot_comparison(
    base_rows: List[Dict], pred_rows: List[Dict],
    out_path: str, burst_name: str, meta: str = "",
) -> None:
    """ベース vs Predictive の 4 指標比較図を生成する.

    2x2 のサブプロットで, 各サブプロットに 4 K 水準を重ね描きする.
    ベースは実線, Predictive は破線.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Experiment 4 vs 4-P ({burst_name} burst): Base vs Predictive\n({meta})",
        fontsize=13,
    )

    metrics_config = [
        (axes[0, 0], "P_block_arrival", r"$P^\mathrm{arrival}_\mathrm{block}$", True),
        (axes[0, 1], "E_W", r"$E[W]$", False),
        (axes[1, 0], "Cost", "Cost", False),
        (axes[1, 1], "ERP", "ERP", False),
    ]

    for K in K_LEVELS:
        color = K_STYLE[K]["color"]

        base_filtered = [r for r in base_rows if int(r["K"]) == K]
        base_filtered.sort(key=lambda r: r["rho"])
        base_rhos = [r["rho"] for r in base_filtered]

        pred_filtered = [r for r in pred_rows if int(r["K"]) == K]
        pred_filtered.sort(key=lambda r: r["rho"])
        pred_rhos = [r["rho"] for r in pred_filtered]

        for ax, metric_name, ylabel, log_y in metrics_config:
            base_values = [r[metric_name] for r in base_filtered]
            pred_values = [r[metric_name] for r in pred_filtered]

            ax.plot(
                base_rhos, base_values,
                color=color, linestyle="-",
                marker="o", markersize=4,
                label=f"K={K} (Base)",
            )
            ax.plot(
                pred_rhos, pred_values,
                color=color, linestyle="--",
                marker="s", markersize=4, alpha=0.7,
                label=f"K={K} (Predictive)",
            )
            ax.set_xlabel(r"$\rho$")
            ax.set_ylabel(ylabel)
            if log_y:
                ax.set_yscale("log")
            ax.grid(True, alpha=0.3)

    axes[0, 0].legend(loc="best", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    plt.savefig(pdf_path, dpi=120, bbox_inches="tight")
    print(f"比較図保存: {out_path}, {pdf_path}")
    plt.close(fig)


def print_improvement_summary(
    base_rows: List[Dict], pred_rows: List[Dict], burst_name: str
) -> None:
    """代表的な (K, rho) 点でベースに対する Predictive の改善率をサマライズ."""
    print(f"\n=== ベース vs Predictive の改善率サマリ ({burst_name} burst) ===")
    print(
        f"{'K':>6s} {'rho':>8s} {'metric':>18s} "
        f"{'base':>12s} {'pred':>12s} {'改善率':>10s}"
    )

    # 代表点: 各 K で rho = {0.3, 0.5, 0.7, 0.9} の 4 点
    rho_targets = [0.3, 0.5, 0.7, 0.9]

    for K in K_LEVELS:
        base_candidates = [r for r in base_rows if int(r["K"]) == K]
        pred_candidates = [r for r in pred_rows if int(r["K"]) == K]
        if not base_candidates or not pred_candidates:
            continue
        for rho_target in rho_targets:
            base_pt = min(base_candidates, key=lambda r: abs(r["rho"] - rho_target))
            pred_pt = min(pred_candidates, key=lambda r: abs(r["rho"] - rho_target))
            for metric in ["P_block_arrival", "E_W", "Cost", "ERP"]:
                base_val = base_pt[metric]
                pred_val = pred_pt[metric]
                if base_val > 0:
                    improvement = (base_val - pred_val) / base_val * 100
                    print(
                        f"{K:>6d} {rho_target:>8.2f} {metric:>18s} "
                        f"{base_val:>12.4g} {pred_val:>12.4g} {improvement:>9.1f}%"
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--burst", choices=["weak", "medium", "strong"], required=True,
        help="バースト水準 (図タイトル用)",
    )
    parser.add_argument("--base-csv", required=True, help="ベース実験 4 の CSV パス")
    parser.add_argument("--pred-csv", required=True, help="Predictive 実験 4-P の CSV パス")
    parser.add_argument(
        "--out", default=None,
        help="出力パス (デフォルト: figures/compare_experiment_4_vs_4P_{burst}.png)",
    )
    parser.add_argument("--meta", default="", help="図タイトルの補足情報")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_rows = load_csv(args.base_csv)
    pred_rows = load_csv(args.pred_csv)

    out_path = args.out or f"figures/compare_experiment_4_vs_4P_{args.burst}.png"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plot_comparison(base_rows, pred_rows, out_path, args.burst, args.meta)
    print_improvement_summary(base_rows, pred_rows, args.burst)


if __name__ == "__main__":
    main()
