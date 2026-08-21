#!/usr/bin/env python
"""実験 1-P: Predictive 拡張版トラフィック強度に対する応答.

実験 1 (ベース版, scripts/experiment_1_traffic.py) と同じ設定
(c=20, K=200, b=5, 1/mu=1, 1/alpha=10s, 1/beta=200s, 3 バースト水準,
rho in [0.1, 0.95]) で Predictive 拡張モデル (mmpp_predictive) を解析し,
4 指標 (P_block^arrival, E[W], Cost, ERP) を計算する.

ベース版の出力と直接比較できるよう, CSV/図の形式を統一する
(比較図は scripts/compare_experiment_1_vs_1P.py で生成する).

使用例:
    python scripts/experiment_1P_traffic.py --quick  # 動作確認用
    python scripts/experiment_1P_traffic.py          # 本番実行
    python scripts/experiment_1P_traffic.py --n-target 16 --gamma 3.0
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
# パラメータ (実験 1 ベース版と完全一致)
# ============================================================

BASELINE = dict(
    c=20,
    K=200,
    b=5,
    mu=1.0,
    alpha=0.1,
    beta=0.005,
)

# バースト水準 3 段階: (名前, delta, sigma) (実験 1 と同じ)
BURST_LEVELS: List[Tuple[str, float, float]] = [
    ("weak", 0.3, 1.0),
    ("medium", 0.6, 0.1),
    ("strong", 0.9, 0.01),
]

RHO_RANGE: Tuple[float, float] = (0.1, 0.95)
RHO_N_POINTS = 15

# Predictive 拡張パラメータの代表値.
# n_target=10: c=20 の半分程度を目標稼働サーバー数とする中間的な設定
# (感度分析は実験 5/6 で別途行う).
# gamma=5.0: Delayoff レートを 5 倍に加速する代表値.
DEFAULT_N_TARGET = 10
DEFAULT_GAMMA = 5.0


@dataclass
class ExperimentResult:
    """1 点の実験結果."""

    rho: float
    burst_name: str
    delta: float
    sigma: float
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
    rho: float, delta: float, sigma: float,
    n_target: int, gamma: float,
    verbose: bool = False,
) -> ExperimentResult:
    """指定パラメータで理論値を計算し ExperimentResult を返す."""
    c = BASELINE["c"]
    b = BASELINE["b"]
    mu = BASELINE["mu"]

    C0, C1 = build_mmpp(rho, delta, sigma, c, b, mu)

    params = PredictiveModelParameters(
        C0=C0, C1=C1, n_target=n_target, gamma=gamma, **BASELINE
    )

    if verbose:
        print(f"  Solving: {params.summary()}")

    Q = build_generator(params)
    pi = solve_stationary(Q)
    m = Metrics(params, pi)

    return ExperimentResult(
        rho=rho,
        burst_name="",
        delta=delta,
        sigma=sigma,
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


def run_all(
    n_target: int, gamma: float, n_points: int, verbose: bool = False
) -> List[ExperimentResult]:
    """全バースト水準 x 全 rho 点で計算を実行する."""
    rho_values = np.linspace(RHO_RANGE[0], RHO_RANGE[1], n_points)
    results: List[ExperimentResult] = []

    total = len(BURST_LEVELS) * len(rho_values)
    with tqdm(total=total, desc="実験 1-P 計算中") as pbar:
        for burst_name, delta, sigma in BURST_LEVELS:
            for rho in rho_values:
                result = run_single_point(
                    float(rho), delta, sigma, n_target, gamma, verbose
                )
                result.burst_name = burst_name
                results.append(result)
                pbar.update(1)

    return results


# ============================================================
# 出力: CSV
# ============================================================

def save_csv(results: List[ExperimentResult], filepath: str) -> None:
    """結果を CSV に書き出す (ベース版と同じ形式 + Predictive パラメータ列)."""
    fieldnames = [
        "burst_name", "delta", "sigma", "rho",
        "n_target", "gamma",
        "P_block_arrival", "E_W", "Cost", "ERP",
        "E_N", "lambda_eff", "E_B", "E_S", "E_I", "E_off",
        "N_states",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "burst_name": r.burst_name,
                "delta": r.delta,
                "sigma": r.sigma,
                "rho": r.rho,
                "n_target": r.n_target,
                "gamma": r.gamma,
                "P_block_arrival": r.P_block_arrival,
                "E_W": r.E_W,
                "Cost": r.Cost,
                "ERP": r.ERP,
                "E_N": r.E_N,
                "lambda_eff": r.lambda_eff,
                "E_B": r.E_B,
                "E_S": r.E_S,
                "E_I": r.E_I,
                "E_off": r.E_off,
                "N_states": r.N_states,
            })
    print(f"CSV 保存: {filepath}")


# ============================================================
# 出力: 図
# ============================================================

def plot_results(
    results: List[ExperimentResult], filepath: str, title_suffix: str = ""
) -> None:
    """4 指標の 2x2 サブプロットを作成する (ベース版と同じ図フォーマット)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Experiment 1-P: Predictive Response to Traffic Intensity "
        r"$\rho$"
        f"\n({title_suffix})",
        fontsize=13,
    )

    colors = {"weak": "tab:blue", "medium": "tab:orange", "strong": "tab:red"}
    labels = {
        "weak": r"weak ($\delta$=0.3, $\sigma$=1.0)",
        "medium": r"medium ($\delta$=0.6, $\sigma$=0.1)",
        "strong": r"strong ($\delta$=0.9, $\sigma$=0.01)",
    }

    metrics_config = [
        (axes[0, 0], "P_block_arrival", r"$P^\mathrm{arrival}_\mathrm{block}$", True),
        (axes[0, 1], "E_W", r"$E[W]$ (mean response time)", False),
        (axes[1, 0], "Cost", "Cost", False),
        (axes[1, 1], "ERP", "ERP", False),
    ]

    for burst_name in ["weak", "medium", "strong"]:
        burst_results = [r for r in results if r.burst_name == burst_name]
        burst_results.sort(key=lambda r: r.rho)
        rhos = [r.rho for r in burst_results]

        for ax, metric_name, ylabel, log_y in metrics_config:
            values = [getattr(r, metric_name) for r in burst_results]
            ax.plot(
                rhos, values,
                color=colors[burst_name], marker="o", markersize=4,
                label=labels[burst_name],
            )
            ax.set_xlabel(r"$\rho$")
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


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-target", type=int, default=DEFAULT_N_TARGET,
        help=f"目標稼働サーバー数 (デフォルト: {DEFAULT_N_TARGET})",
    )
    parser.add_argument(
        "--gamma", type=float, default=DEFAULT_GAMMA,
        help=f"Delayoff 加速係数 (デフォルト: {DEFAULT_GAMMA})",
    )
    parser.add_argument(
        "--n-points", type=int, default=RHO_N_POINTS,
        help=f"rho 走査点数 (デフォルト: {RHO_N_POINTS})",
    )
    parser.add_argument("--out-dir", default="figures", help="図出力ディレクトリ")
    parser.add_argument("--csv-dir", default="results", help="CSV 出力ディレクトリ")
    parser.add_argument(
        "--quick", action="store_true", help="動作確認用の高速モード (n-points=5)"
    )
    parser.add_argument("--verbose", action="store_true", help="各点の詳細を表示")
    args = parser.parse_args()
    if args.quick:
        args.n_points = 5
    return args


def main() -> None:
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.csv_dir, exist_ok=True)

    print("=== 実験 1-P: Predictive 版トラフィック強度 ===")
    print(f"n_target = {args.n_target}")
    print(f"gamma    = {args.gamma}")
    print(f"n_points = {args.n_points}")
    print(f"バースト水準: {[name for name, _, _ in BURST_LEVELS]}")

    results = run_all(args.n_target, args.gamma, args.n_points, args.verbose)

    csv_path = os.path.join(
        args.csv_dir, f"experiment_1P_nt{args.n_target}_g{args.gamma}.csv"
    )
    save_csv(results, csv_path)

    fig_path = os.path.join(
        args.out_dir, f"experiment_1P_nt{args.n_target}_g{args.gamma}.png"
    )
    title_suffix = (
        f"c={BASELINE['c']}, K={BASELINE['K']}, b={BASELINE['b']}, "
        f"1/alpha={1 / BASELINE['alpha']:.0f}s, 1/beta={1 / BASELINE['beta']:.0f}s, "
        f"n_target={args.n_target}, gamma={args.gamma}"
    )
    plot_results(results, fig_path, title_suffix)

    print("\n=== 保存則チェック ===")
    for r in [results[0], results[-1]]:
        conservation = r.E_B + r.E_S + r.E_I + r.E_off
        print(
            f"rho={r.rho:.2f}, burst={r.burst_name}: "
            f"E[B]+E[S]+E[I]+E[Off] = {conservation:.4f} (c={BASELINE['c']})"
        )

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
