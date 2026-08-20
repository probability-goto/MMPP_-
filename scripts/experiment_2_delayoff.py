"""実験 2 (delayoff 率 beta に対する応答, 複数 alpha 水準) の実装.

実験計画 5.5 節に沿い, 先行研究 (Le-Anh & Phung-Duc 2025) Fig.8 のパラメータ規模
(c=20, K=200, b=5, rho=0.7, 1/mu=1) の下, 3 つの alpha 水準 (0.1, 1, 10) について
beta = delayoff 率を [1e-2, 1e2] で対数スイープし, 4 指標
(P_block^arrival, E[W], Cost, ERP) を理論解析 (mmpp) のみで計算する.

バースト性の影響は補助的に見る: メイン実行は中バースト (delta=0.6, sigma=0.1) を
固定し, --strong-burst フラグで強バースト (delta=0.9, sigma=0.01) の別図を生成する.

使用例:
    python scripts/experiment_2_delayoff.py --n-points 20
    python scripts/experiment_2_delayoff.py --n-points 20 --strong-burst
    python scripts/experiment_2_delayoff.py --quick  # 開発中の軽量実行 (n-points=5)
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


# 実験計画のベースラインパラメータ (先行研究 Fig.8 と同じ規模)
BASELINE = dict(
    c=20,       # サーバー数 (実験 1 と同じ)
    K=200,      # バッファ容量 (実験 1 と同じ)
    b=5,        # バッチサイズ (先行研究 Fig.8 と同じ)
    mu=1.0,     # サービス率
    # alpha は水準として走査するのでここには含めない
    # beta は走査変数なのでここには含めない
)

RHO_FIXED = 0.7  # 固定トラフィック強度 (先行研究 Fig.8 と同じ)

# alpha 水準の 3 段階定義 (先行研究 Fig.8 と同じ)
ALPHA_LEVELS: List[Tuple[str, float]] = [
    ("alpha=0.1", 0.1),
    ("alpha=1", 1.0),
    ("alpha=10", 10.0),
]

# 水準ごとの表示色
LEVEL_STYLE: Dict[str, Dict[str, str]] = {
    "alpha=0.1": dict(color="tab:blue", linestyle="-"),
    "alpha=1": dict(color="tab:orange", linestyle="-"),
    "alpha=10": dict(color="tab:red", linestyle="-"),
}

# バースト水準: メインは中バースト固定, 補助は強バースト固定
BURST_MAIN = ("medium", 0.6, 0.1)
BURST_AUX = ("strong", 0.9, 0.01)

# beta スイープ範囲の既定値 (対数スケール)
BETA_RANGE: Tuple[float, float] = (1e-2, 1e2)

# (指標キー, Metrics 共通メソッド名, 表示ラベル)
METRIC_SPECS: List[Tuple[str, str, str]] = [
    ("P_block_arrival", "arrival_blocking_probability", r"$P_{\mathrm{block}}^{\mathrm{arrival}}$"),
    ("E[W]", "mean_waiting_time", r"$E[W]$ (mean response time)"),
    ("Cost", "energy_cost_paper", "Cost"),
    ("ERP", "erp_paper", "ERP"),
]


def build_params(beta: float, alpha: float, delta: float, sigma: float) -> ModelParameters:
    """beta, alpha, delta, sigma に対応する ModelParameters を構築する (BASELINE + RHO_FIXED)."""
    C0, C1 = build_mmpp(RHO_FIXED, delta, sigma, BASELINE["c"], BASELINE["b"], BASELINE["mu"])
    return ModelParameters(C0=C0, C1=C1, alpha=alpha, beta=beta, **BASELINE)


def run_theory(params: ModelParameters) -> Metrics:
    """理論解析: Q 構築 -> 定常分布 -> Metrics."""
    Q = build_generator(params)
    pi = solve_stationary(Q)
    return Metrics(params, pi)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-points", type=int, default=15, help="beta スイープ点数")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--csv-dir", default="results", help="CSV 出力ディレクトリ")
    parser.add_argument(
        "--quick", action="store_true",
        help="開発中の軽量実行用 (n-points=5)",
    )
    parser.add_argument(
        "--strong-burst", action="store_true",
        help="強バースト水準 (delta=0.9, sigma=0.01) で実行し, ファイル名末尾に _strong を付ける",
    )
    args = parser.parse_args()
    if args.quick:
        args.n_points = 5
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


def _find_optimal_beta(
    beta_values: np.ndarray, values_list: List[float]
) -> Tuple[float, float]:
    """values_list (beta ごとの指標値) から, 指標が最小となる beta を返す."""
    idx = int(np.argmin(values_list))
    return float(beta_values[idx]), float(values_list[idx])


def main() -> None:
    args = parse_args()
    lo, hi = BETA_RANGE
    beta_values = np.logspace(np.log10(lo), np.log10(hi), args.n_points)

    burst_name, delta, sigma = BURST_AUX if args.strong_burst else BURST_MAIN

    # 結果格納: theory_vals[level_name][metric_key] = [値, ...] (beta 順)
    theory_vals: Dict[str, Dict[str, List[float]]] = {
        name: {k: [] for k, _, _ in METRIC_SPECS} for name, _ in ALPHA_LEVELS
    }

    max_conservation_error = 0.0
    max_flow_balance_error = 0.0
    csv_rows: List[Dict[str, object]] = []

    for level_name, alpha in ALPHA_LEVELS:
        print(
            f"\n########## alpha level = {level_name} "
            f"(burst={burst_name}, delta={delta}, sigma={sigma}) ##########"
        )

        for beta in tqdm(beta_values, desc=f"{level_name}"):
            beta = float(beta)
            params = build_params(beta, alpha, delta, sigma)
            theory = run_theory(params)

            print(f"\n=== [{level_name}] beta = {beta:.4g} ===")
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
                "burst_name": burst_name,
                "delta": delta,
                "sigma": sigma,
                "alpha": alpha,
                "beta": beta,
                "rho": RHO_FIXED,
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

    _print_caption_summary(beta_values, theory_vals, burst_name, delta, sigma)
    _save_csv(args, csv_rows, burst_name)
    _plot(args, beta_values, theory_vals, burst_name, delta, sigma)


def _save_csv(args: argparse.Namespace, rows: List[Dict[str, object]], burst_name: str) -> None:
    """alpha x beta スイープ結果を CSV に保存する (実験 2-P との比較用に列を統一)."""
    os.makedirs(args.csv_dir, exist_ok=True)
    csv_path = os.path.join(args.csv_dir, f"experiment_2_{burst_name}.csv")
    fieldnames = [
        "burst_name", "delta", "sigma", "alpha", "beta", "rho",
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
    beta_values: np.ndarray,
    theory_vals: Dict[str, Dict[str, List[float]]],
    burst_name: str,
    delta: float,
    sigma: float,
) -> None:
    """図のキャプション用サマリ (最小/最大 beta での指標表, 最適 beta) をコンソールに出力する."""
    print("\n=== キャプション用サマリ ===")
    print(f"[バースト水準] {burst_name} (delta={delta}, sigma={sigma})")

    lo_idx, hi_idx = 0, len(beta_values) - 1
    print(f"\n[最小 beta={beta_values[lo_idx]:.4g} と最大 beta={beta_values[hi_idx]:.4g} での指標値]")
    header = f"  {'level':<10}{'beta':>10}" + "".join(f"{k:>18}" for k, _, _ in METRIC_SPECS)
    print(header)
    for level_name, _ in ALPHA_LEVELS:
        for idx, tag in [(lo_idx, "theory(lo)"), (hi_idx, "theory(hi)")]:
            row = f"  {level_name:<10}{beta_values[idx]:>10.3g}"
            for key, _, _ in METRIC_SPECS:
                row += f"{theory_vals[level_name][key][idx]:>18.5g}"
            print(row + f"  [{tag}]")

    print("\n[各 alpha 水準での最適 beta (指標最小)]")
    for level_name, _ in ALPHA_LEVELS:
        for key, _, _ in METRIC_SPECS:
            if key == "P_block_arrival":
                continue  # 最小化対象外 (単調)
            beta_opt, val_opt = _find_optimal_beta(beta_values, theory_vals[level_name][key])
            print(f"  {level_name}: min({key}) = {val_opt:.4g} at beta = {beta_opt:.4g}")


def _plot(
    args: argparse.Namespace,
    beta_values: np.ndarray,
    theory_vals: Dict[str, Dict[str, List[float]]],
    burst_name: str,
    delta: float,
    sigma: float,
) -> None:
    """4 指標を 2x2 のサブプロットにまとめ, PDF と PNG に保存する."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    for ax, (key, _, ylabel) in zip(axes.flat, METRIC_SPECS):
        for level_name, alpha in ALPHA_LEVELS:
            style = LEVEL_STYLE[level_name]
            ax.plot(
                beta_values, theory_vals[level_name][key],
                color=style["color"], linestyle=style["linestyle"],
                label=level_name if ax is axes.flat[0] else None,
            )

        ax.set_xscale("log")
        if key == "P_block_arrival":
            ax.set_yscale("log")

        ax.set_xlabel(r"$\beta$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.3)

    axes.flat[0].legend(loc="best", fontsize=8)

    suffix = "_strong" if args.strong_burst else ""
    burst_label = "strong burst" if args.strong_burst else "medium burst"
    fig.suptitle(
        f"Experiment 2: Response to Delayoff Rate "
        r"$\beta$"
        f" ({burst_label})\n"
        rf"($c=20$, $K=200$, $b=5$, $\rho=0.7$, delta={delta}, sigma={sigma})"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    os.makedirs(args.out_dir, exist_ok=True)
    pdf_path = os.path.join(args.out_dir, f"experiment_2_delayoff{suffix}.pdf")
    png_path = os.path.join(args.out_dir, f"experiment_2_delayoff{suffix}.png")
    fig.savefig(pdf_path, dpi=100)
    fig.savefig(png_path, dpi=150)
    print(f"\n図を保存しました: {pdf_path}, {png_path}")


if __name__ == "__main__":
    main()
