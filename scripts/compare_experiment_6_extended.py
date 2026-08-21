#!/usr/bin/env python
"""実験 6 の拡張走査結果を初期走査と統合してプロットする.

初期走査 (γ ∈ {1, 2, 5, 10, 20}) と拡張走査 (γ ∈ {20, 50, 100, 500, 1000})
の CSV を統合し, γ の広範囲 (1〜1000) にわたる ERP・4 指標の依存性を
可視化する. これにより真の最適 γ の位置および単調改善が続く領域を特定する.

使用例:
    python scripts/compare_experiment_6_extended.py --burst medium
    python scripts/compare_experiment_6_extended.py --burst all
    python scripts/compare_experiment_6_extended.py --burst medium \\
        --initial-csv results/experiment_6_medium_nt10.csv \\
        --extended-csv results/experiment_6_extended/experiment_6_medium_nt10.csv
"""
import argparse
import csv
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ALPHA_LEVELS = [0.1, 1.0, 10.0]

BURST_LEVELS_ORDER = ["weak", "medium", "strong"]


def load_csv(filepath: str) -> List[Dict]:
    """CSV を辞書リストとして読み込む (数値列は float に変換)."""
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


def merge_and_dedupe(
    initial_rows: List[Dict], extended_rows: List[Dict],
) -> List[Dict]:
    """初期走査と拡張走査の結果を統合し, γ 値で重複除去する.

    γ=20 が両方に存在する場合、拡張走査 (extended_rows) の値を優先する
    (再現性チェックのため両者の値を確認するのは呼び出し側の責務).
    """
    combined: Dict[tuple, Dict] = {}
    for row in initial_rows:
        key = (round(row["alpha"], 6), round(row["gamma"], 6))
        combined[key] = row
    for row in extended_rows:
        key = (round(row["alpha"], 6), round(row["gamma"], 6))
        combined[key] = row
    return sorted(combined.values(), key=lambda r: (r["alpha"], r["gamma"]))


def plot_extended(
    rows: List[Dict], out_path: str, burst_name: str, n_target: int,
) -> None:
    """γ 拡張走査の 4 指標プロット (横軸: γ 対数, 3 α 曲線)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Experiment 6 (Extended): Sensitivity to gamma ({burst_name} burst, "
        rf"$n_\mathrm{{target}}$={n_target})"
        f"\nγ ∈ [1, 1000], initial + extended sweeps merged",
        fontsize=13,
    )

    alpha_colors = {0.1: "tab:blue", 1.0: "tab:orange", 10.0: "tab:red"}
    alpha_labels = {
        0.1: r"$\alpha$=0.1 (setup slow)",
        1.0: r"$\alpha$=1.0 (setup medium)",
        10.0: r"$\alpha$=10 (setup fast)",
    }

    metrics_config = [
        (axes[0, 0], "P_block_arrival", r"$P^\mathrm{arrival}_\mathrm{block}$", True),
        (axes[0, 1], "E_W", r"$E[W]$", False),
        (axes[1, 0], "Cost", "Cost", False),
        (axes[1, 1], "ERP", "ERP", False),
    ]

    for alpha in ALPHA_LEVELS:
        alpha_rows = [r for r in rows if abs(r["alpha"] - alpha) < 1e-6]
        alpha_rows.sort(key=lambda r: r["gamma"])
        gammas = [r["gamma"] for r in alpha_rows]

        for ax, metric_name, ylabel, log_y in metrics_config:
            values = [r[metric_name] for r in alpha_rows]
            ax.plot(
                gammas, values,
                color=alpha_colors[alpha], marker="o", markersize=5,
                label=alpha_labels[alpha],
            )
            ax.set_xlabel(r"$\gamma$")
            ax.set_xscale("log")
            ax.set_ylabel(ylabel)
            if log_y:
                ax.set_yscale("log")
            ax.grid(True, alpha=0.3, which="both")

    axes[0, 0].legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    plt.savefig(pdf_path, dpi=120, bbox_inches="tight")
    print(f"図保存: {out_path}, {pdf_path}")
    plt.close(fig)


def find_optimal_gamma(rows: List[Dict], alpha: float) -> Dict:
    """指定 α で ERP を最小化する γ とその値を返す."""
    alpha_rows = [r for r in rows if abs(r["alpha"] - alpha) < 1e-6]
    if not alpha_rows:
        return {}
    return min(alpha_rows, key=lambda r: r["ERP"])


def print_optimal_summary(rows: List[Dict], burst_name: str) -> None:
    """各 α で ERP を最小化する γ と、代表値 γ=5.0 との差をサマライズ."""
    print(f"\n=== [{burst_name}] 拡張走査後の最適 γ ===")
    print(f"{'alpha':>8s} {'optimal γ':>12s} "
          f"{'ERP_opt':>10s} {'ERP_γ=5':>10s} "
          f"{'ERP_γ=1':>10s} {'ERP_γ=20':>10s} "
          f"{'γ=5 悪化':>10s}")

    for alpha in ALPHA_LEVELS:
        opt = find_optimal_gamma(rows, alpha)
        alpha_rows = [r for r in rows if abs(r["alpha"] - alpha) < 1e-6]
        g5 = next((r for r in alpha_rows if abs(r["gamma"] - 5.0) < 1e-6), None)
        g1 = next((r for r in alpha_rows if abs(r["gamma"] - 1.0) < 1e-6), None)
        g20 = next((r for r in alpha_rows if abs(r["gamma"] - 20.0) < 1e-6), None)

        if opt and g5:
            degradation = (g5["ERP"] - opt["ERP"]) / opt["ERP"] * 100
        else:
            degradation = float("nan")

        print(
            f"{alpha:>8.1f} {opt.get('gamma', 0):>12.1f} "
            f"{opt.get('ERP', 0):>10.4g} "
            f"{g5['ERP'] if g5 else 0:>10.4g} "
            f"{g1['ERP'] if g1 else 0:>10.4g} "
            f"{g20['ERP'] if g20 else 0:>10.4g} "
            f"{degradation:>9.1f}%"
        )


def check_gamma_20_consistency(
    initial_rows: List[Dict], extended_rows: List[Dict],
) -> None:
    """γ=20 が両走査に含まれる場合、再現性を確認する.

    同じ γ=20 で両走査の値が機械精度で一致すること (両方とも同一の LU
    分解ソルバーで計算されるため) を確認する.
    """
    print("\n=== γ=20 の再現性チェック (初期走査 vs 拡張走査) ===")
    for alpha in ALPHA_LEVELS:
        init_pt = next(
            (r for r in initial_rows
             if abs(r["alpha"] - alpha) < 1e-6 and abs(r["gamma"] - 20.0) < 1e-6),
            None,
        )
        ext_pt = next(
            (r for r in extended_rows
             if abs(r["alpha"] - alpha) < 1e-6 and abs(r["gamma"] - 20.0) < 1e-6),
            None,
        )
        if init_pt is None or ext_pt is None:
            print(f"  α={alpha}: γ=20 の点が片方に存在しないためスキップ")
            continue
        max_rel_diff = 0.0
        for metric in ["P_block_arrival", "E_W", "Cost", "ERP"]:
            init_val = init_pt[metric]
            ext_val = ext_pt[metric]
            if abs(init_val) > 1e-12:
                rel_diff = abs(init_val - ext_val) / abs(init_val)
                max_rel_diff = max(max_rel_diff, rel_diff)
        print(f"  α={alpha}: 4 指標の最大相対差 = {max_rel_diff:.3e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--burst", choices=["weak", "medium", "strong", "all"], default="all",
        help="バースト水準",
    )
    parser.add_argument(
        "--n-target", type=int, default=10,
        help="実験 6 で使用した n_target (ファイル名の解決に使用)",
    )
    parser.add_argument(
        "--initial-csv-dir", default="results",
        help="初期走査 CSV のディレクトリ (デフォルト: results)",
    )
    parser.add_argument(
        "--extended-csv-dir", default="results/experiment_6_extended",
        help="拡張走査 CSV のディレクトリ (デフォルト: results/experiment_6_extended)",
    )
    parser.add_argument(
        "--out-dir", default="figures/experiment_6_extended",
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--initial-csv", default=None,
        help="初期走査 CSV パスを直接指定 (--burst と併用不可)",
    )
    parser.add_argument(
        "--extended-csv", default=None,
        help="拡張走査 CSV パスを直接指定 (--burst と併用不可)",
    )
    return parser.parse_args()


def burst_levels_to_run(burst_arg: str) -> List[str]:
    if burst_arg == "all":
        return BURST_LEVELS_ORDER
    return [burst_arg]


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for burst_name in burst_levels_to_run(args.burst):
        if args.initial_csv and args.extended_csv:
            initial_path = args.initial_csv
            extended_path = args.extended_csv
        else:
            file_stem = f"experiment_6_{burst_name}_nt{args.n_target}"
            initial_path = os.path.join(args.initial_csv_dir, f"{file_stem}.csv")
            extended_path = os.path.join(args.extended_csv_dir, f"{file_stem}.csv")

        if not os.path.exists(initial_path):
            print(f"警告: 初期走査 CSV が見つかりません: {initial_path}")
            continue
        if not os.path.exists(extended_path):
            print(f"警告: 拡張走査 CSV が見つかりません: {extended_path}")
            continue

        print(f"\n########## burst level = {burst_name} ##########")
        print(f"初期走査 CSV: {initial_path}")
        print(f"拡張走査 CSV: {extended_path}")

        initial_rows = load_csv(initial_path)
        extended_rows = load_csv(extended_path)

        check_gamma_20_consistency(initial_rows, extended_rows)

        merged = merge_and_dedupe(initial_rows, extended_rows)
        out_path = os.path.join(
            args.out_dir, f"experiment_6_extended_{burst_name}.png"
        )
        plot_extended(merged, out_path, burst_name, args.n_target)

        print_optimal_summary(merged, burst_name)


if __name__ == "__main__":
    main()
