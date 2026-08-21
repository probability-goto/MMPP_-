#!/usr/bin/env python
"""実験 6: gamma (Delayoff 加速係数) 感度分析.

Predictive 拡張モデルの gamma パラメータに対する 4 指標の感度を分析する.
gamma は通常位相 (F=0) における Delayoff レートの加速係数で,
gamma=1 で加速無効 (Delayoff レートは beta のまま), gamma が大きいほど
通常時のサーバー停止が加速される.

固定条件: rho=0.7, n_target=10 (実験 1-P〜4-P と同じ代表値)
走査変数: gamma ∈ {1.0, 2.0, 5.0, 10.0, 20.0}
サブ実験: 3 バースト水準 × 3 α 水準

gamma=1 は Delayoff 加速機構を無効化した状態
(n_target=10 は有効なので, ベースと厳密一致するのは n_target=0 かつ
gamma=1.0 のときのみ. これは test_predictive_baseline_consistency.py で検証済み).

使用例:
    python scripts/experiment_6_gamma.py --burst medium --quick
    python scripts/experiment_6_gamma.py --burst all
    python scripts/experiment_6_gamma.py --n-target 5  # n_target を上書き
"""
import argparse
import csv
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    print("注意: tqdm が見つかりません. 素朴な進捗表示で継続します.")

    def tqdm(iterable, desc=None, total=None):
        total = total or len(iterable)
        for idx, item in enumerate(iterable, start=1):
            print(f"  [{desc}] {idx}/{total}")
            yield item

from mmpp_predictive import (
    PredictiveModelParameters, build_generator, solve_stationary, Metrics
)

try:
    from _mmpp_burst import build_mmpp
except ImportError:
    from scripts._mmpp_burst import build_mmpp


# ============================================================
# パラメータ
# ============================================================

BASELINE = dict(
    c=20,
    K=200,
    b=5,
    mu=1.0,
    alpha=0.1,   # alpha 水準で上書き
    beta=0.005,
)

RHO_FIXED = 0.7
ALPHA_LEVELS = [0.1, 1.0, 10.0]

BURST_LEVELS: List[Tuple[str, float, float]] = [
    ("weak", 0.3, 1.0),
    ("medium", 0.6, 0.1),
    ("strong", 0.9, 0.01),
]

# 走査対象の gamma 水準 (1 <= gamma <= 20 を対数スケールで 15 点走査.
# 境界値 gamma=1 (加速無効) と gamma=20 (拡張走査との接続点), および
# プロジェクト全体の代表値 gamma=5.0 を含む)
def _build_gamma_levels() -> List[float]:
    levels = list(np.logspace(0, np.log10(20.0), 15))
    nearest_idx = min(
        range(len(levels)), key=lambda i: abs(np.log10(levels[i]) - np.log10(5.0))
    )
    levels[nearest_idx] = 5.0
    return sorted(float(g) for g in levels)


GAMMA_LEVELS = _build_gamma_levels()

# n_target は代表値で固定
N_TARGET_FIXED = 10


@dataclass
class ExperimentResult:
    """1 点の実験結果."""

    burst_name: str
    delta: float
    sigma: float
    alpha: float
    beta: float
    rho: float
    n_target: int
    gamma: float
    P_block_arrival: float
    E_W: float
    Cost: float
    ERP: float
    E_N: float
    lambda_eff: float
    E_B: float
    E_S: float
    E_I: float
    E_off: float
    N_states: int


def run_single_point(
    rho: float, delta: float, sigma: float, alpha: float,
    n_target: int, gamma: float,
    verbose: bool = False,
) -> ExperimentResult:
    """指定パラメータで理論値を計算し ExperimentResult を返す."""
    c = BASELINE["c"]
    b = BASELINE["b"]
    mu = BASELINE["mu"]
    beta = BASELINE["beta"]

    C0, C1 = build_mmpp(rho, delta, sigma, c, b, mu)

    params_dict = {k: v for k, v in BASELINE.items() if k != "alpha"}
    params = PredictiveModelParameters(
        C0=C0, C1=C1, alpha=alpha,
        n_target=n_target, gamma=gamma,
        **params_dict,
    )

    if verbose:
        print(f"  Solving: {params.summary()}")

    Q = build_generator(params)
    pi = solve_stationary(Q)
    m = Metrics(params, pi)

    return ExperimentResult(
        burst_name="",
        delta=delta, sigma=sigma, alpha=alpha, beta=beta, rho=rho,
        n_target=n_target, gamma=gamma,
        P_block_arrival=m.arrival_blocking_probability(),
        E_W=m.mean_waiting_time(),
        Cost=m.energy_cost_paper(),
        ERP=m.erp_paper(),
        E_N=m.mean_queue_length(),
        lambda_eff=m.effective_arrival_rate(),
        E_B=m.mean_busy(),
        E_S=m.mean_setup(),
        E_I=m.mean_idle(),
        E_off=m.mean_off(),
        N_states=params.N,
    )


def run_all(
    burst_name: str, delta: float, sigma: float,
    n_target: int, gamma_levels: List[float],
    verbose: bool = False,
) -> List[ExperimentResult]:
    """全 alpha 水準 x 全 gamma 水準で計算."""
    results: List[ExperimentResult] = []

    total = len(ALPHA_LEVELS) * len(gamma_levels)
    with tqdm(total=total, desc=f"実験 6 (gamma 感度, {burst_name})") as pbar:
        for alpha in ALPHA_LEVELS:
            for gamma in gamma_levels:
                result = run_single_point(
                    RHO_FIXED, delta, sigma, alpha,
                    n_target, gamma, verbose,
                )
                result.burst_name = burst_name
                results.append(result)
                pbar.update(1)

    return results


def _burst_levels_to_run(burst_arg: str) -> List[Tuple[str, float, float]]:
    if burst_arg == "all":
        return BURST_LEVELS
    return [lvl for lvl in BURST_LEVELS if lvl[0] == burst_arg]


def save_csv(results: List[ExperimentResult], filepath: str) -> None:
    fieldnames = [
        "burst_name", "delta", "sigma",
        "alpha", "beta", "rho",
        "n_target", "gamma",
        "P_block_arrival", "E_W", "Cost", "ERP",
        "E_N", "lambda_eff", "E_B", "E_S", "E_I", "E_off",
        "N_states",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({fn: getattr(r, fn) for fn in fieldnames})
    print(f"CSV 保存: {filepath}")


def plot_results(
    results: List[ExperimentResult], filepath: str,
    burst_name: str, n_target: int, title_suffix: str = "",
) -> None:
    """4 指標の 2x2 サブプロット (横軸 gamma (対数), 3 alpha 曲線)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Experiment 6: Sensitivity to gamma ({burst_name} burst, "
        rf"$n_\mathrm{{target}}$={n_target})"
        f"\n({title_suffix})",
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
        alpha_results = [r for r in results if r.alpha == alpha]
        alpha_results.sort(key=lambda r: r.gamma)
        gammas = [r.gamma for r in alpha_results]

        for ax, metric_name, ylabel, log_y in metrics_config:
            values = [getattr(r, metric_name) for r in alpha_results]
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
            ax.grid(True, alpha=0.3)

    axes[0, 0].legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches="tight")
    pdf_filepath = os.path.splitext(filepath)[0] + ".pdf"
    plt.savefig(pdf_filepath, dpi=120, bbox_inches="tight")
    print(f"図保存: {filepath}, {pdf_filepath}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--burst", choices=["weak", "medium", "strong", "all"], default="all",
        help="バースト水準",
    )
    parser.add_argument(
        "--n-target", type=int, default=N_TARGET_FIXED,
        help=f"目標稼働サーバー数 (固定, デフォルト: {N_TARGET_FIXED})",
    )
    parser.add_argument(
        "--gamma-levels", type=float, nargs="+", default=GAMMA_LEVELS,
        help=f"gamma 走査水準 (デフォルト: {GAMMA_LEVELS})",
    )
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--csv-dir", default="results")
    parser.add_argument(
        "--quick", action="store_true",
        help="動作確認用 (gamma 3 水準に削減)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.gamma_levels = [1.0, 5.0, 20.0]
    return args


def main() -> None:
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.csv_dir, exist_ok=True)

    print("=== 実験 6: gamma 感度分析 ===")
    print(f"rho      = {RHO_FIXED} (固定)")
    print(f"n_target = {args.n_target} (固定)")
    print(f"gamma 水準 = {args.gamma_levels}")
    print(f"alpha 水準 = {ALPHA_LEVELS}")

    max_conservation_error = 0.0

    for burst_name, delta, sigma in _burst_levels_to_run(args.burst):
        print(f"\n########## burst level = {burst_name} "
              f"(delta={delta}, sigma={sigma}) ##########")

        results = run_all(
            burst_name, delta, sigma,
            args.n_target, args.gamma_levels, args.verbose,
        )

        # 保存則チェック
        for r in results:
            cons_err = abs(r.E_B + r.E_S + r.E_I + r.E_off - BASELINE["c"])
            max_conservation_error = max(max_conservation_error, cons_err)

        # CSV & 図の保存
        file_stem = f"experiment_6_{burst_name}_nt{args.n_target}"
        csv_path = os.path.join(args.csv_dir, f"{file_stem}.csv")
        save_csv(results, csv_path)

        fig_path = os.path.join(args.out_dir, f"{file_stem}.png")
        title_suffix = (
            f"c={BASELINE['c']}, K={BASELINE['K']}, b={BASELINE['b']}, "
            f"rho={RHO_FIXED}"
        )
        plot_results(results, fig_path, burst_name, args.n_target, title_suffix)

    print(f"\n=== サマリ ===")
    print(f"保存則の最大誤差: {max_conservation_error:.3e}")


if __name__ == "__main__":
    main()
