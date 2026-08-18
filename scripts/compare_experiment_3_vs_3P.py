#!/usr/bin/env python
"""実験 3 (ベース) と実験 3-P (Predictive) の直接比較図を生成.

両方の CSV (scripts/experiment_3_burstiness.py, scripts/experiment_3P_burstiness.py
の出力) を読み込み, 同じ 4 指標をベース (実線) vs Predictive (破線) で
重ね描きする. 3 alpha 水準 x 4 指標の 2x2 サブプロット図を作成し, 代表的な
delta または sigma 点でベースに対する Predictive の改善率を標準出力に
サマライズする.

使用例:
    python scripts/compare_experiment_3_vs_3P.py \\
        --sweep delta \\
        --base-csv results/experiment_3_delta.csv \\
        --pred-csv results/experiment_3P_delta_nt10_g5.0.csv \\
        --meta "delta sweep, n_target=10, gamma=5.0"

    python scripts/compare_experiment_3_vs_3P.py \\
        --sweep sigma \\
        --base-csv results/experiment_3_sigma.csv \\
        --pred-csv results/experiment_3P_sigma_nt10_g5.0.csv \\
        --meta "sigma sweep, n_target=10, gamma=5.0"
"""
import argparse
import csv
import os
from typing import Dict, List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ALPHA_LEVELS = [0.1, 1.0, 10.0]


def load_csv(filepath: str) -> List[Dict]:
    """CSV を辞書リストとして読み込む (数値列は float に変換する)."""
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            for k in row:
                if k not in ("sweep_var",):
                    try:
                        row[k] = float(row[k])
                    except ValueError:
                        pass
            rows.append(row)
    return rows


def plot_comparison(
    base_rows: List[Dict], pred_rows: List[Dict],
    out_path: str, sweep_var: str, meta: str = "",
) -> None:
    """ベース vs Predictive の 4 指標比較図を生成する.

    sweep_var は "delta" または "sigma".
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sub_label = "A" if sweep_var == "delta" else "B"
    fig.suptitle(
        f"Experiment 3-{sub_label} vs 3-P-{sub_label} ({sweep_var} sweep): "
        f"Base vs Predictive\n({meta})",
        fontsize=13,
    )

    alpha_colors = {0.1: "tab:blue", 1.0: "tab:orange", 10.0: "tab:red"}
    alpha_labels = {
        0.1: r"$\alpha$=0.1",
        1.0: r"$\alpha$=1.0",
        10.0: r"$\alpha$=10",
    }

    metrics_config = [
        (axes[0, 0], "P_block_arrival", r"$P^\mathrm{arrival}_\mathrm{block}$", True),
        (axes[0, 1], "E_W", r"$E[W]$", False),
        (axes[1, 0], "Cost", "Cost", False),
        (axes[1, 1], "ERP", "ERP", False),
    ]

    x_label = r"$\delta$" if sweep_var == "delta" else r"$\sigma$"
    x_log = sweep_var == "sigma"

    for alpha in ALPHA_LEVELS:
        base_filtered = [r for r in base_rows if abs(r["alpha"] - alpha) < 1e-6]
        base_filtered.sort(key=lambda r: r[sweep_var])
        base_xs = [r[sweep_var] for r in base_filtered]

        pred_filtered = [r for r in pred_rows if abs(r["alpha"] - alpha) < 1e-6]
        pred_filtered.sort(key=lambda r: r[sweep_var])
        pred_xs = [r[sweep_var] for r in pred_filtered]

        for ax, metric_name, ylabel, log_y in metrics_config:
            base_values = [r[metric_name] for r in base_filtered]
            pred_values = [r[metric_name] for r in pred_filtered]

            ax.plot(
                base_xs, base_values,
                color=alpha_colors[alpha], linestyle="-",
                marker="o", markersize=4,
                label=f"{alpha_labels[alpha]} (Base)",
            )
            ax.plot(
                pred_xs, pred_values,
                color=alpha_colors[alpha], linestyle="--",
                marker="s", markersize=4, alpha=0.7,
                label=f"{alpha_labels[alpha]} (Predictive)",
            )
            ax.set_xlabel(x_label)
            if x_log:
                ax.set_xscale("log")
            ax.set_ylabel(ylabel)
            if log_y:
                ax.set_yscale("log")
            ax.grid(True, alpha=0.3)

    axes[0, 0].legend(loc="best", fontsize=7, ncol=1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"比較図保存: {out_path}")
    plt.close(fig)


def print_improvement_summary(
    base_rows: List[Dict], pred_rows: List[Dict], sweep_var: str
) -> None:
    """代表的な走査点でベースに対する Predictive の改善率をサマライズ."""
    print(f"\n=== ベース vs Predictive の改善率サマリ ({sweep_var} 走査) ===")
    print(
        f"{'alpha':>6s} {sweep_var:>10s} {'metric':>18s} "
        f"{'base':>12s} {'pred':>12s} {'改善率':>10s}"
    )

    # 代表点 (delta: 線形 5 点、sigma: 対数 5 点)
    if sweep_var == "delta":
        targets = [0.0, 0.3, 0.5, 0.7, 0.9]

        def distance(actual, target):
            return abs(actual - target)
    else:  # sigma
        targets = [1e-3, 1e-2, 1e-1, 1e0, 1e1]

        def distance(actual, target):
            return abs(np.log10(actual) - np.log10(target))

    for alpha in ALPHA_LEVELS:
        base_candidates = [r for r in base_rows if abs(r["alpha"] - alpha) < 1e-6]
        pred_candidates = [r for r in pred_rows if abs(r["alpha"] - alpha) < 1e-6]
        if not base_candidates or not pred_candidates:
            continue
        for target in targets:
            base_pt = min(
                base_candidates,
                key=lambda r: distance(r[sweep_var], target),
            )
            pred_pt = min(
                pred_candidates,
                key=lambda r: distance(r[sweep_var], target),
            )
            for metric in ["P_block_arrival", "E_W", "Cost", "ERP"]:
                base_val = base_pt[metric]
                pred_val = pred_pt[metric]
                if base_val > 0:
                    improvement = (base_val - pred_val) / base_val * 100
                    print(
                        f"{alpha:>6.1f} {target:>10.4g} {metric:>18s} "
                        f"{base_val:>12.4g} {pred_val:>12.4g} {improvement:>9.1f}%"
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep", choices=["delta", "sigma"], required=True,
        help="走査変数を指定",
    )
    parser.add_argument("--base-csv", required=True, help="ベース実験 3 の CSV パス")
    parser.add_argument("--pred-csv", required=True, help="Predictive 実験 3-P の CSV パス")
    parser.add_argument(
        "--out", default=None,
        help="出力パス (デフォルト: figures/compare_experiment_3_vs_3P_{sweep}.png)",
    )
    parser.add_argument("--meta", default="", help="図タイトルの補足情報")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_rows = load_csv(args.base_csv)
    pred_rows = load_csv(args.pred_csv)

    out_path = args.out or f"figures/compare_experiment_3_vs_3P_{args.sweep}.png"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plot_comparison(base_rows, pred_rows, out_path, args.sweep, args.meta)
    print_improvement_summary(base_rows, pred_rows, args.sweep)


if __name__ == "__main__":
    main()
