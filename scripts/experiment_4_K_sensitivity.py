"""実験 4 (バッファ容量 K に対する感度分析) の実装.

有限バッファ K=200 という設計選択の妥当性, および先行研究の無限バッファ近似が
MMPP バースト到着下でどこまで妥当かを検証するため, K を {100, 200, 500, 1000}
で走査する. 実験 1 と同じ 3 つのバースト水準 (弱・中・強) それぞれについて,
rho = lambda_bar / (c*b*mu) を [0.1, 0.95] でスイープしながら 4 指標
(P_block^arrival, E[W], Cost, ERP) を理論解析 (mmpp) のみで計算し, K が
大きいほど指標がどこに収束していくかを観察する.

c, b, mu, alpha, beta は実験 1 と同じ値 (先行研究 Fig.10 の規模) に固定する.

使用例:
    python scripts/experiment_4_K_sensitivity.py --burst medium
    python scripts/experiment_4_K_sensitivity.py --burst all
    python scripts/experiment_4_K_sensitivity.py --burst medium --quick  # 開発中の軽量実行 (n-points=5)
"""
import argparse
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


# 実験計画のベースラインパラメータ (実験 1 と同じ規模. K は走査変数なので含めない)
BASELINE = dict(
    c=20,       # サーバー数 (実験 1 と同じ)
    b=5,        # バッチサイズ (実験 1 と同じ)
    mu=1.0,     # サービス率
    alpha=0.1,  # セットアップ率 (1/alpha = 10 s)
    beta=0.005,  # Delayoff 率 (1/beta = 200 s)
)

# 走査対象の K 水準 (有限バッファ設計の妥当性検証)
K_LEVELS: List[int] = [100, 200, 500, 1000]

# 水準ごとの表示色
K_STYLE: Dict[int, Dict[str, str]] = {
    100: dict(color="tab:blue"),
    200: dict(color="tab:orange"),
    500: dict(color="tab:green"),
    1000: dict(color="tab:red"),
}

# バースト水準の 3 段階定義 (実験 1 と同じ): (名前, delta, sigma)
BURST_LEVELS: List[Tuple[str, float, float]] = [
    ("weak", 0.3, 1.0),
    ("medium", 0.6, 0.1),
    ("strong", 0.9, 0.01),
]

# rho スイープ範囲・点数の既定値 (実験 1 と同じ)
RHO_RANGE: Tuple[float, float] = (0.1, 0.95)
RHO_N_POINTS = 15

# (指標キー, Metrics 共通メソッド名, 表示ラベル)
METRIC_SPECS: List[Tuple[str, str, str]] = [
    ("P_block_arrival", "arrival_blocking_probability", r"$P_{\mathrm{block}}^{\mathrm{arrival}}$"),
    ("E[W]", "mean_waiting_time", r"$E[W]$ (mean response time)"),
    ("Cost", "energy_cost_paper", "Cost"),
    ("ERP", "erp_paper", "ERP"),
]


def build_params(K: int, rho: float, delta: float, sigma: float) -> ModelParameters:
    """K, rho, delta, sigma に対応する ModelParameters を構築する (BASELINE 固定)."""
    C0, C1 = build_mmpp(rho, delta, sigma, BASELINE["c"], BASELINE["b"], BASELINE["mu"])
    return ModelParameters(C0=C0, C1=C1, K=K, **BASELINE)


def run_theory(params: ModelParameters) -> Metrics:
    """理論解析: Q 構築 -> 定常分布 -> Metrics."""
    Q = build_generator(params)
    pi = solve_stationary(Q)
    return Metrics(params, pi)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--burst", choices=["weak", "medium", "strong", "all"], default="all",
        help="バースト水準を指定 (既定: all で 3 水準すべてを実行)",
    )
    parser.add_argument("--n-points", type=int, default=RHO_N_POINTS, help="rho スイープ点数")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--csv-dir", default="results", help="CSV 出力ディレクトリ")
    parser.add_argument(
        "--quick", action="store_true",
        help="開発中の軽量実行用 (n-points=5)",
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


def _burst_levels_to_run(burst_arg: str) -> List[Tuple[str, float, float]]:
    """--burst 引数から実行対象のバースト水準リストを返す."""
    if burst_arg == "all":
        return BURST_LEVELS
    return [lvl for lvl in BURST_LEVELS if lvl[0] == burst_arg]


def main() -> None:
    args = parse_args()
    lo, hi = RHO_RANGE
    rho_values = np.linspace(lo, hi, args.n_points)

    max_conservation_error = 0.0
    max_flow_balance_error = 0.0

    for level_name, delta, sigma in _burst_levels_to_run(args.burst):
        # 結果格納: theory_vals[K][metric_key] = [値, ...] (rho 順)
        theory_vals: Dict[int, Dict[str, List[float]]] = {
            K: {k: [] for k, _, _ in METRIC_SPECS} for K in K_LEVELS
        }

        cv_mid = compute_interarrival_cv(
            *build_mmpp(0.5, delta, sigma, BASELINE["c"], BASELINE["b"], BASELINE["mu"])
        )
        print(
            f"\n########## burst level = {level_name} "
            f"(delta={delta}, sigma={sigma}, CV@rho=0.5={cv_mid:.4f}) ##########"
        )

        for K in K_LEVELS:
            for rho in tqdm(rho_values, desc=f"burst={level_name}, K={K}"):
                rho = float(rho)
                params = build_params(K, rho, delta, sigma)
                theory = run_theory(params)

                print(
                    f"\n=== [{level_name}, K={K}] rho = {rho:.4g} "
                    f"(lambda_bar = {params.lambda_bar:.4g}) ==="
                )
                print(f"  {'metric':<18}{'theory':>14}")
                for key, method_name, _ in METRIC_SPECS:
                    theory_val = getattr(theory, method_name)()
                    theory_vals[K][key].append(theory_val)
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

        _print_caption_summary(level_name, rho_values, theory_vals)
        _plot(args, level_name, delta, sigma, rho_values, theory_vals)
        _save_csv(args, level_name, delta, sigma, rho_values, theory_vals)

    print("\n=== サマリ ===")
    print(f"保存則の最大誤差: {max_conservation_error:.3e}")
    print(f"流量バランスの最大誤差: {max_flow_balance_error:.3e}")


def _print_caption_summary(
    level_name: str,
    rho_values: np.ndarray,
    theory_vals: Dict[int, Dict[str, List[float]]],
) -> None:
    """図のキャプション用サマリ (K 別の最軽/最重 rho での指標表, 収束の目安) をコンソールに出力する."""
    print(f"\n=== [{level_name}] キャプション用サマリ ===")
    lo_idx, hi_idx = 0, len(rho_values) - 1
    print(f"[最軽 rho={rho_values[lo_idx]:.4g} と最重 rho={rho_values[hi_idx]:.4g} での指標値 (K 別)]")
    header = f"  {'K':<8}{'rho':>8}" + "".join(f"{k:>18}" for k, _, _ in METRIC_SPECS)
    print(header)
    for K in K_LEVELS:
        for idx, tag in [(lo_idx, "theory(lo)"), (hi_idx, "theory(hi)")]:
            row = f"  {K:<8}{rho_values[idx]:>8.3g}"
            for key, _, _ in METRIC_SPECS:
                row += f"{theory_vals[K][key][idx]:>18.5g}"
            print(row + f"  [{tag}]")

    # 最大 2 水準の K 間の差 (無限バッファ近似への収束具合の目安, 最重 rho で評価)
    K_max, K_prev = K_LEVELS[-1], K_LEVELS[-2]
    print(f"\n[K={K_prev} -> K={K_max} での指標変化 (収束の目安, rho={rho_values[hi_idx]:.4g})]")
    for key, _, _ in METRIC_SPECS:
        v_prev = theory_vals[K_prev][key][hi_idx]
        v_max = theory_vals[K_max][key][hi_idx]
        diff = abs(v_max - v_prev)
        rel = diff / max(abs(v_max), 1e-12)
        print(
            f"  {key:<18} K={K_prev}: {v_prev:.5g} -> K={K_max}: {v_max:.5g} "
            f"(|diff|={diff:.3g}, rel={rel:.3e})"
        )


def _plot(
    args: argparse.Namespace,
    level_name: str,
    delta: float,
    sigma: float,
    rho_values: np.ndarray,
    theory_vals: Dict[int, Dict[str, List[float]]],
) -> None:
    """4 指標を 2x2 のサブプロットにまとめ, K 別に重ね描きして PNG に保存する."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    for ax, (key, _, ylabel) in zip(axes.flat, METRIC_SPECS):
        for K in K_LEVELS:
            style = K_STYLE[K]
            ax.plot(
                rho_values, theory_vals[K][key], "-",
                color=style["color"], marker="o", markersize=4,
                label=f"K={K}" if ax is axes.flat[0] else None,
            )

        if key == "P_block_arrival":
            ax.set_yscale("log")

        ax.set_xlabel(r"$\rho$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.3)

    axes.flat[0].legend(loc="best", fontsize=8)

    fig.suptitle(
        f"Experiment 4: Sensitivity to Buffer Capacity $K$ ({level_name} burst)\n"
        rf"($c=20$, $b=5$, $1/\mu=1$, $1/\alpha=10$s, $1/\beta=200$s, "
        rf"delta={delta}, sigma={sigma})"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    os.makedirs(args.out_dir, exist_ok=True)
    png_path = os.path.join(args.out_dir, f"experiment_4_K_sensitivity_{level_name}.png")
    fig.savefig(png_path, dpi=150)
    print(f"\n図を保存しました: {png_path}")


def _save_csv(
    args: argparse.Namespace,
    level_name: str,
    delta: float,
    sigma: float,
    rho_values: np.ndarray,
    theory_vals: Dict[int, Dict[str, List[float]]],
) -> None:
    """4 指標の K 別・rho 別データを CSV に書き出す (比較スクリプト用)."""
    import csv
    os.makedirs(args.csv_dir, exist_ok=True)
    csv_path = os.path.join(args.csv_dir, f"experiment_4_{level_name}.csv")

    fieldnames = [
        "burst_name", "delta", "sigma", "K", "rho",
        "P_block_arrival", "E_W", "Cost", "ERP",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for K in K_LEVELS:
            for i, rho in enumerate(rho_values):
                writer.writerow({
                    "burst_name": level_name,
                    "delta": delta,
                    "sigma": sigma,
                    "K": K,
                    "rho": float(rho),
                    "P_block_arrival": theory_vals[K]["P_block_arrival"][i],
                    "E_W": theory_vals[K]["E[W]"][i],
                    "Cost": theory_vals[K]["Cost"][i],
                    "ERP": theory_vals[K]["ERP"][i],
                })
    print(f"CSV 保存: {csv_path}")


if __name__ == "__main__":
    main()
