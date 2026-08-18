#!/usr/bin/env python
"""実験 3-P: Predictive 拡張版バースト性の直接影響.

実験 3 (ベース版, scripts/experiment_3_burstiness.py) と同じ設定
(c=20, K=200, b=5, rho=0.7, alpha in {0.1, 1, 10}) で Predictive 拡張
モデルを解析し, バースト性の 2 独立軸 (振幅 delta と時定数 sigma) を
連続的に走査して 4 指標を計算する.

実験 3-P-A (delta 走査): delta in [0.0, 0.95] 線形 25 点, sigma=0.1 固定
実験 3-P-B (sigma 走査): sigma in [1e-3, 1e1] 対数 25 点, delta=0.6 固定

使用例:
    python scripts/experiment_3P_burstiness.py --sweep delta --quick
    python scripts/experiment_3P_burstiness.py --sweep sigma --quick
    python scripts/experiment_3P_burstiness.py --sweep delta        # 3-P-A 本番
    python scripts/experiment_3P_burstiness.py --sweep sigma        # 3-P-B 本番
    python scripts/experiment_3P_burstiness.py --sweep delta \
        --n-target 16 --gamma 3.0
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
    print("注意: tqdm が見つかりません (pip install tqdm を推奨). 素朴な進捗表示で継続します.")

    def tqdm(iterable, desc=None, total=None):
        total = total or len(iterable)
        for idx, item in enumerate(iterable, start=1):
            print(f"  [{desc}] {idx}/{total}")
            yield item

from mmpp_predictive import PredictiveModelParameters, build_generator, solve_stationary, Metrics

try:
    from _mmpp_burst import build_mmpp
except ImportError:
    from scripts._mmpp_burst import build_mmpp


# ============================================================
# パラメータ (実験 3 ベース版と完全一致)
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

# 実験 3-A: delta 走査 (振幅走査)
DELTA_RANGE: Tuple[float, float] = (0.0, 0.95)
DELTA_N_POINTS = 25
DELTA_SIGMA_FIXED = 0.1

# 実験 3-B: sigma 走査 (時定数走査)
SIGMA_RANGE: Tuple[float, float] = (1e-3, 1e1)
SIGMA_N_POINTS = 25
SIGMA_DELTA_FIXED = 0.6

# Predictive 拡張パラメータの代表値 (実験 1-P, 2-P と統一)
DEFAULT_N_TARGET = 10
DEFAULT_GAMMA = 5.0


@dataclass
class ExperimentResult:
    """1 点の実験結果."""

    sweep_var: str        # "delta" または "sigma"
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


# ============================================================
# メイン計算
# ============================================================

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

    # BASELINE から alpha を除いた辞書を作成し, alpha 水準で上書きする
    params_dict = {k: v for k, v in BASELINE.items() if k != "alpha"}
    params = PredictiveModelParameters(
        C0=C0, C1=C1,
        alpha=alpha,
        n_target=n_target, gamma=gamma,
        **params_dict,
    )

    if verbose:
        print(f"  Solving: {params.summary()}")

    Q = build_generator(params)
    pi = solve_stationary(Q)
    m = Metrics(params, pi)

    return ExperimentResult(
        sweep_var="",  # 呼び出し側で埋める
        delta=delta,
        sigma=sigma,
        alpha=alpha,
        beta=beta,
        rho=rho,
        n_target=n_target,
        gamma=gamma,
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


def run_delta_sweep(
    n_target: int, gamma: float, n_points: int, verbose: bool = False
) -> List[ExperimentResult]:
    """実験 3-P-A: delta 走査 (sigma=0.1 固定)."""
    delta_values = np.linspace(DELTA_RANGE[0], DELTA_RANGE[1], n_points)
    sigma = DELTA_SIGMA_FIXED
    results: List[ExperimentResult] = []

    total = len(ALPHA_LEVELS) * len(delta_values)
    with tqdm(total=total, desc="実験 3-P-A (delta 走査)") as pbar:
        for alpha in ALPHA_LEVELS:
            for delta in delta_values:
                result = run_single_point(
                    RHO_FIXED, float(delta), sigma, alpha,
                    n_target, gamma, verbose,
                )
                result.sweep_var = "delta"
                results.append(result)
                pbar.update(1)

    return results


def run_sigma_sweep(
    n_target: int, gamma: float, n_points: int, verbose: bool = False
) -> List[ExperimentResult]:
    """実験 3-P-B: sigma 走査 (delta=0.6 固定, 対数スケール)."""
    sigma_values = np.logspace(
        np.log10(SIGMA_RANGE[0]), np.log10(SIGMA_RANGE[1]), n_points
    )
    delta = SIGMA_DELTA_FIXED
    results: List[ExperimentResult] = []

    total = len(ALPHA_LEVELS) * len(sigma_values)
    with tqdm(total=total, desc="実験 3-P-B (sigma 走査)") as pbar:
        for alpha in ALPHA_LEVELS:
            for sigma in sigma_values:
                result = run_single_point(
                    RHO_FIXED, delta, float(sigma), alpha,
                    n_target, gamma, verbose,
                )
                result.sweep_var = "sigma"
                results.append(result)
                pbar.update(1)

    return results


# ============================================================
# 出力: CSV
# ============================================================

def save_csv(results: List[ExperimentResult], filepath: str) -> None:
    """結果を CSV に書き出す (実験 3 ベース版と同じ形式 + Predictive パラメータ列)."""
    fieldnames = [
        "sweep_var",
        "delta", "sigma",
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


# ============================================================
# 出力: 図
# ============================================================

def plot_delta_sweep(
    results: List[ExperimentResult], filepath: str, title_suffix: str = ""
) -> None:
    """実験 3-P-A: delta 走査の 4 指標プロット (実験 3 ベース版と同じフォーマット).

    横軸: delta (線形), 3 alpha 水準を色分けした 3 曲線.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Experiment 3-P-A: Predictive Response to Burst Amplitude "
        r"$\delta$"
        f" ($\\sigma$={DELTA_SIGMA_FIXED} fixed)"
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
        (axes[0, 1], "E_W", r"$E[W]$ (mean response time)", False),
        (axes[1, 0], "Cost", "Cost", False),
        (axes[1, 1], "ERP", "ERP", False),
    ]

    for alpha in ALPHA_LEVELS:
        alpha_results = [r for r in results if r.alpha == alpha]
        alpha_results.sort(key=lambda r: r.delta)
        deltas = [r.delta for r in alpha_results]

        for ax, metric_name, ylabel, log_y in metrics_config:
            values = [getattr(r, metric_name) for r in alpha_results]
            ax.plot(
                deltas, values,
                color=alpha_colors[alpha], marker="o", markersize=4,
                label=alpha_labels[alpha],
            )
            ax.set_xlabel(r"$\delta$")
            ax.set_ylabel(ylabel)
            if log_y:
                ax.set_yscale("log")
            ax.grid(True, alpha=0.3)

    axes[0, 0].legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches="tight")
    print(f"図保存: {filepath}")
    plt.close(fig)


def plot_sigma_sweep(
    results: List[ExperimentResult], filepath: str, title_suffix: str = ""
) -> None:
    """実験 3-P-B: sigma 走査の 4 指標プロット (実験 3 ベース版と同じフォーマット).

    横軸: sigma (対数), 3 alpha 水準を色分けした 3 曲線.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Experiment 3-P-B: Predictive Response to Burst Time Constant "
        r"$\sigma$"
        f" ($\\delta$={SIGMA_DELTA_FIXED} fixed)"
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
        (axes[0, 1], "E_W", r"$E[W]$ (mean response time)", False),
        (axes[1, 0], "Cost", "Cost", False),
        (axes[1, 1], "ERP", "ERP", False),
    ]

    for alpha in ALPHA_LEVELS:
        alpha_results = [r for r in results if r.alpha == alpha]
        alpha_results.sort(key=lambda r: r.sigma)
        sigmas = [r.sigma for r in alpha_results]

        for ax, metric_name, ylabel, log_y in metrics_config:
            values = [getattr(r, metric_name) for r in alpha_results]
            ax.plot(
                sigmas, values,
                color=alpha_colors[alpha], marker="o", markersize=4,
                label=alpha_labels[alpha],
            )
            ax.set_xlabel(r"$\sigma$")
            ax.set_xscale("log")
            ax.set_ylabel(ylabel)
            if log_y:
                ax.set_yscale("log")
            ax.grid(True, alpha=0.3)

    axes[0, 0].legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches="tight")
    print(f"図保存: {filepath}")
    plt.close(fig)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep", choices=["delta", "sigma"], required=True,
        help="走査変数を指定 (delta: 実験 3-P-A, sigma: 実験 3-P-B)",
    )
    parser.add_argument(
        "--n-target", type=int, default=DEFAULT_N_TARGET,
        help=f"目標稼働サーバー数 (デフォルト: {DEFAULT_N_TARGET})",
    )
    parser.add_argument(
        "--gamma", type=float, default=DEFAULT_GAMMA,
        help=f"Delayoff 加速係数 (デフォルト: {DEFAULT_GAMMA})",
    )
    parser.add_argument(
        "--n-points", type=int, default=None,
        help="走査点数 (デフォルト: delta なら 25, sigma なら 25)",
    )
    parser.add_argument("--out-dir", default="figures", help="図出力ディレクトリ")
    parser.add_argument("--csv-dir", default="results", help="CSV 出力ディレクトリ")
    parser.add_argument(
        "--quick", action="store_true",
        help="動作確認用の高速モード (n-points=5)",
    )
    parser.add_argument("--verbose", action="store_true", help="各点の詳細を表示")
    args = parser.parse_args()

    if args.n_points is None:
        args.n_points = DELTA_N_POINTS if args.sweep == "delta" else SIGMA_N_POINTS
    if args.quick:
        args.n_points = 5

    return args


def main() -> None:
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.csv_dir, exist_ok=True)

    sub_label = "A" if args.sweep == "delta" else "B"
    fixed_label = (
        f"sigma={DELTA_SIGMA_FIXED} 固定" if args.sweep == "delta"
        else f"delta={SIGMA_DELTA_FIXED} 固定"
    )

    print(f"=== 実験 3-P-{sub_label}: Predictive 版 {args.sweep} 走査 ===")
    print(f"固定条件: {fixed_label}, rho={RHO_FIXED}")
    print(f"n_target = {args.n_target}")
    print(f"gamma    = {args.gamma}")
    print(f"n_points = {args.n_points}")
    print(f"alpha 水準: {ALPHA_LEVELS}")

    if args.sweep == "delta":
        results = run_delta_sweep(
            args.n_target, args.gamma, args.n_points, args.verbose
        )
        plot_fn = plot_delta_sweep
    else:
        results = run_sigma_sweep(
            args.n_target, args.gamma, args.n_points, args.verbose
        )
        plot_fn = plot_sigma_sweep

    file_stem = f"experiment_3P_{args.sweep}_nt{args.n_target}_g{args.gamma}"
    csv_path = os.path.join(args.csv_dir, f"{file_stem}.csv")
    save_csv(results, csv_path)

    fig_path = os.path.join(args.out_dir, f"{file_stem}.png")
    title_suffix = (
        f"c={BASELINE['c']}, K={BASELINE['K']}, b={BASELINE['b']}, "
        f"rho={RHO_FIXED}, n_target={args.n_target}, gamma={args.gamma}"
    )
    plot_fn(results, fig_path, title_suffix)

    # Poisson 極限テスト (delta 走査のみ, delta=0 で Poisson と一致することを検証)
    if args.sweep == "delta":
        print("\n=== Poisson 極限テスト (delta=0 の点) ===")
        for alpha in ALPHA_LEVELS:
            zero_delta_result = next(
                (r for r in results if r.alpha == alpha and r.delta == 0.0),
                None,
            )
            if zero_delta_result is not None:
                print(
                    f"  alpha={alpha}, delta=0: "
                    f"P_block^arrival={zero_delta_result.P_block_arrival:.6g}, "
                    f"E[W]={zero_delta_result.E_W:.6g}"
                )

    print("\n=== 保存則チェック ===")
    for r in [results[0], results[-1]]:
        conservation = r.E_B + r.E_S + r.E_I + r.E_off
        primary_val = r.delta if args.sweep == "delta" else r.sigma
        print(
            f"alpha={r.alpha}, {args.sweep}={primary_val:.4g}: "
            f"E[B]+E[S]+E[I]+E[Off] = {conservation:.4f} (c={BASELINE['c']})"
        )

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
