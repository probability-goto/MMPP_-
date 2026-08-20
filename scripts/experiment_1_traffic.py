"""実験 1 (トラフィック強度 rho に対する応答, 3 バースト水準) の実装.

実験計画 5.4 節に沿い, 先行研究 (Le-Anh & Phung-Duc 2025) Fig.10 のパラメータ規模
(c=20, K=200, b=5, 1/mu=1, 1/alpha=10s, 1/beta=200s) の下, 3 つのバースト水準
(弱・中・強) について rho = lambda_bar / (c*b*mu) を [0.1, 0.95] でスイープし,
4 指標 (P_block^arrival, E[W], Cost, ERP) を理論解析 (mmpp) のみで計算する.

実験 0 で理論とシミュレーションの一致は確認済みのため, 実験 1 以降は理論解析のみで
図を作成する方針 (スイープ点数が多くシミュレーションを含めると計算時間が大きくなること,
論文図として理論曲線の滑らかさを優先することが理由).

使用例:
    python scripts/experiment_1_traffic.py --n-points 15
    python scripts/experiment_1_traffic.py --quick  # 開発中の軽量実行 (n-points=3)
"""
import argparse
import csv
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    print("注意: tqdm が見つかりません (pip install tqdm を推奨). 素朴な進捗表示で継続します.")

    def tqdm(iterable, desc=None):
        total = len(iterable)
        for idx, item in enumerate(iterable, start=1):
            print(f"  [{desc}] {idx}/{total}")
            yield item

from mmpp import ModelParameters, build_generator, solve_stationary, Metrics
try:
    from _mmpp_burst import build_mmpp, compute_interarrival_cv
except ImportError:
    from scripts._mmpp_burst import build_mmpp, compute_interarrival_cv


# 実験計画のベースラインパラメータ (先行研究 Fig.10 と同じ規模)
BASELINE = dict(
    c=20,       # サーバー数 (先行研究 Fig.10 と同じ)
    K=200,      # バッファ容量 (cb*4 = 400 だが計算コスト考慮で 200)
    b=5,        # バッチサイズ (先行研究 6.3.1 節と同じ)
    mu=1.0,     # サービス率
    alpha=0.1,  # セットアップ率 (1/alpha = 10 s)
    beta=0.005,  # Delayoff 率 (1/beta = 200 s)
)

# バースト水準の 3 段階定義: (名前, delta, sigma)
BURST_LEVELS: List[Tuple[str, float, float]] = [
    ("weak", 0.3, 1.0),
    ("medium", 0.6, 0.1),
    ("strong", 0.9, 0.01),
]

# 水準ごとの表示色
LEVEL_STYLE: Dict[str, Dict[str, str]] = {
    "weak": dict(color="tab:blue"),
    "medium": dict(color="tab:orange"),
    "strong": dict(color="tab:red"),
}

# rho スイープ範囲の既定値
RHO_RANGE: Tuple[float, float] = (0.1, 0.95)

# (指標キー, Metrics 共通メソッド名, 表示ラベル)
METRIC_SPECS: List[Tuple[str, str, str]] = [
    ("P_block_arrival", "arrival_blocking_probability", r"$P_{\mathrm{block}}^{\mathrm{arrival}}$"),
    ("E[W]", "mean_waiting_time", r"$E[W]$ (mean response time)"),
    ("Cost", "energy_cost_paper", "Cost"),
    ("ERP", "erp_paper", "ERP"),
]


def build_params(rho: float, delta: float, sigma: float) -> ModelParameters:
    """rho, delta, sigma に対応する ModelParameters を構築する (BASELINE 固定)."""
    C0, C1 = build_mmpp(rho, delta, sigma, BASELINE["c"], BASELINE["b"], BASELINE["mu"])
    return ModelParameters(C0=C0, C1=C1, **BASELINE)


def run_theory(params: ModelParameters) -> Metrics:
    """理論解析: Q 構築 -> 定常分布 -> Metrics."""
    Q = build_generator(params)
    pi = solve_stationary(Q)
    return Metrics(params, pi)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-points", type=int, default=15, help="rho スイープ点数")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--csv-dir", default="results", help="CSV 出力ディレクトリ")
    parser.add_argument(
        "--quick", action="store_true",
        help="開発中の軽量実行用 (n-points=3)",
    )
    args = parser.parse_args()
    if args.quick:
        args.n_points = 3
    return args


def _conservation_and_flow_balance(
    c: int,
    b: int,
    mu: float,
    mean_busy: float,
    mean_idle: float,
    mean_setup: float,
    mean_off: float,
    lambda_eff: float,
) -> Tuple[float, float]:
    """保存則 E[B]+E[I]+E[S]+E[Off]=c と 流量バランス lambda_eff=b*mu*E[B] の誤差を返す.

    Returns:
        (conservation_error, flow_balance_error)
    """
    conservation_error = abs(mean_busy + mean_idle + mean_setup + mean_off - c)
    flow_balance_error = abs(lambda_eff - b * mu * mean_busy)
    return conservation_error, flow_balance_error


def main() -> None:
    args = parse_args()
    lo, hi = RHO_RANGE
    rho_values = np.linspace(lo, hi, args.n_points)

    # 結果格納: results[level_name][metric_key] = [値, ...] (rho 順)
    theory_vals: Dict[str, Dict[str, List[float]]] = {
        name: {k: [] for k, _, _ in METRIC_SPECS} for name, _, _ in BURST_LEVELS
    }

    max_conservation_error = 0.0
    max_flow_balance_error = 0.0
    csv_rows: List[Dict[str, object]] = []

    for level_name, delta, sigma in BURST_LEVELS:
        cv_mid = compute_interarrival_cv(
            *build_mmpp(0.5, delta, sigma, BASELINE["c"], BASELINE["b"], BASELINE["mu"])
        )
        print(
            f"\n########## burst level = {level_name} "
            f"(delta={delta}, sigma={sigma}, CV@rho=0.5={cv_mid:.4f}) ##########"
        )

        for rho in tqdm(rho_values, desc=f"burst={level_name}"):
            rho = float(rho)
            params = build_params(rho, delta, sigma)
            theory = run_theory(params)

            print(f"\n=== [{level_name}] rho = {rho:.4g} (lambda_bar = {params.lambda_bar:.4g}) ===")
            print(f"  {'metric':<18}{'theory':>14}")
            for key, method_name, _ in METRIC_SPECS:
                theory_val = getattr(theory, method_name)()
                theory_vals[level_name][key].append(theory_val)
                print(f"  {key:<18}{theory_val:>14.5g}")

            cons_err, flow_err = _conservation_and_flow_balance(
                params.c, params.b, params.mu,
                theory.mean_busy(), theory.mean_idle(),
                theory.mean_setup(), theory.mean_off(),
                theory.effective_arrival_rate(),
            )
            print(
                f"  [保存則・流量バランス] "
                f"E[B]+E[I]+E[S]+E[Off]-c = {cons_err:.3e}, "
                f"lambda_eff - b*mu*E[B] = {flow_err:.3e}"
            )
            max_conservation_error = max(max_conservation_error, cons_err)
            max_flow_balance_error = max(max_flow_balance_error, flow_err)

            csv_rows.append({
                "burst_name": level_name,
                "delta": delta,
                "sigma": sigma,
                "rho": rho,
                "P_block_arrival": theory_vals[level_name]["P_block_arrival"][-1],
                "E_W": theory_vals[level_name]["E[W]"][-1],
                "Cost": theory_vals[level_name]["Cost"][-1],
                "ERP": theory_vals[level_name]["ERP"][-1],
                "E_N": theory.mean_queue_length(),
                "lambda_eff": theory.effective_arrival_rate(),
                "E_B": theory.mean_busy(),
                "E_S": theory.mean_setup(),
                "E_I": theory.mean_idle(),
                "E_off": theory.mean_off(),
                "N_states": params.N,
            })

    print("\n=== サマリ ===")
    print(f"保存則の最大誤差: {max_conservation_error:.3e}")
    print(f"流量バランスの最大誤差: {max_flow_balance_error:.3e}")

    _print_caption_summary(rho_values, theory_vals)
    _save_csv(args, csv_rows)
    _plot(args, rho_values, theory_vals)


def _save_csv(args: argparse.Namespace, rows: List[Dict[str, object]]) -> None:
    """rho スイープ結果を CSV に保存する (実験 1-P との比較用に列を統一)."""
    os.makedirs(args.csv_dir, exist_ok=True)
    csv_path = os.path.join(args.csv_dir, "experiment_1.csv")
    fieldnames = [
        "burst_name", "delta", "sigma", "rho",
        "P_block_arrival", "E_W", "Cost", "ERP",
        "E_N", "lambda_eff", "E_B", "E_S", "E_I", "E_off",
        "N_states",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV 保存: {csv_path}")


def _print_caption_summary(
    rho_values: np.ndarray,
    theory_vals: Dict[str, Dict[str, List[float]]],
) -> None:
    """図のキャプション用サマリ (CV, 最軽/最重 rho での指標表) をコンソールに出力する."""
    print("\n=== キャプション用サマリ ===")
    print("[各バースト水準の CV (rho=0.5 での代表値)]")
    for level_name, delta, sigma in BURST_LEVELS:
        cv = compute_interarrival_cv(*build_mmpp(0.5, delta, sigma, BASELINE["c"], BASELINE["b"], BASELINE["mu"]))
        print(f"  {level_name:<8} delta={delta}, sigma={sigma}, CV={cv:.4f}")

    lo_idx, hi_idx = 0, len(rho_values) - 1
    print(f"\n[最軽 rho={rho_values[lo_idx]:.4g} と最重 rho={rho_values[hi_idx]:.4g} での指標値]")
    header = f"  {'level':<8}{'rho':>8}" + "".join(f"{k:>18}" for k, _, _ in METRIC_SPECS)
    print(header)
    for level_name, _, _ in BURST_LEVELS:
        for idx, tag in [(lo_idx, "theory(lo)"), (hi_idx, "theory(hi)")]:
            row = f"  {level_name:<8}{rho_values[idx]:>8.3g}"
            for key, _, _ in METRIC_SPECS:
                row += f"{theory_vals[level_name][key][idx]:>18.5g}"
            print(row + f"  [{tag}]")


def _plot(
    args: argparse.Namespace,
    rho_values: np.ndarray,
    theory_vals: Dict[str, Dict[str, List[float]]],
) -> None:
    """4 指標を 2x2 のサブプロットにまとめ, PDF と PNG に保存する."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    for ax, (key, _, ylabel) in zip(axes.flat, METRIC_SPECS):
        for level_name, delta, sigma in BURST_LEVELS:
            style = LEVEL_STYLE[level_name]
            cv = compute_interarrival_cv(*build_mmpp(0.5, delta, sigma, BASELINE["c"], BASELINE["b"], BASELINE["mu"]))
            label = f"{level_name} (delta={delta}, sigma={sigma}, CV={cv:.2f})"

            ax.plot(
                rho_values, theory_vals[level_name][key], "-",
                color=style["color"], marker="o", markersize=4,
                label=label if ax is axes.flat[0] else None,
            )

        if key == "P_block_arrival":
            ax.set_yscale("log")

        ax.set_xlabel(r"$\rho$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.3)

    axes.flat[0].legend(loc="best", fontsize=8)

    fig.suptitle(
        "Experiment 1: Response to Traffic Intensity "
        r"$\rho$ across Burstiness Levels"
        "\n"
        r"($c=20$, $K=200$, $b=5$, $1/\mu=1$, $1/\alpha=10$s, $1/\beta=200$s)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    os.makedirs(args.out_dir, exist_ok=True)
    pdf_path = os.path.join(args.out_dir, "experiment_1_traffic.pdf")
    png_path = os.path.join(args.out_dir, "experiment_1_traffic.png")
    fig.savefig(pdf_path, dpi=100)
    fig.savefig(png_path, dpi=150)
    print(f"\n図を保存しました: {pdf_path}, {png_path}")


if __name__ == "__main__":
    main()
