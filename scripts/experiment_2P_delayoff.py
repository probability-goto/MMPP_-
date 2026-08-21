#!/usr/bin/env python
"""実験 2-P: Predictive 拡張版 Delayoff 率に対する応答.

実験 2 (ベース版, scripts/experiment_2_delayoff.py) と同じ設定
(c=20, K=200, b=5, rho=0.7, alpha in {0.1, 1, 10}, beta 対数走査 40 点) で
Predictive 拡張モデル (mmpp_predictive) を解析し, 4 指標
(P_block^arrival, E[W], Cost, ERP) を計算する.

メイン実験: 中バースト水準 (delta=0.6, sigma=0.1)
補助実験: 強バースト水準 (delta=0.9, sigma=0.01), --strong-burst で切替

使用例:
    python scripts/experiment_2P_delayoff.py --quick            # 動作確認
    python scripts/experiment_2P_delayoff.py                    # メイン実験
    python scripts/experiment_2P_delayoff.py --strong-burst     # 補助実験
    python scripts/experiment_2P_delayoff.py --n-target 16 --gamma 3.0
"""
import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
# パラメータ (実験 2 ベース版と完全一致)
# ============================================================

BASELINE = dict(
    c=20,
    K=200,
    b=5,
    mu=1.0,
    alpha=0.1,   # alpha 水準で上書き
    beta=0.005,  # beta スイープで上書き
)

RHO_FIXED = 0.7
ALPHA_LEVELS = [0.1, 1.0, 10.0]

BETA_RANGE: Tuple[float, float] = (1e-2, 1e2)
BETA_N_POINTS = 15

BURST_LEVELS = {
    "medium": (0.6, 0.1),
    "strong": (0.9, 0.01),
}

# Predictive 拡張パラメータの代表値 (実験 1-P と統一).
DEFAULT_N_TARGET = 10
DEFAULT_GAMMA = 5.0


def parse_alpha_gamma_map(map_str: str) -> Dict[float, float]:
    """'0.1:1.0,1.0:100.0,10.0:1000.0' 形式を辞書に変換する."""
    result = {}
    for pair in map_str.split(","):
        alpha_str, gamma_str = pair.split(":")
        result[float(alpha_str.strip())] = float(gamma_str.strip())
    return result


def resolve_gamma(
    alpha: float, gamma: float, alpha_gamma_dict: Optional[Dict[float, float]],
) -> float:
    """alpha に対応する gamma 値を返す (alpha_gamma_dict が None なら gamma をそのまま返す)."""
    if alpha_gamma_dict is None:
        return gamma
    if alpha in alpha_gamma_dict:
        return alpha_gamma_dict[alpha]
    nearest_alpha = min(
        alpha_gamma_dict.keys(),
        key=lambda a: abs(math.log10(a) - math.log10(alpha)),
    )
    return alpha_gamma_dict[nearest_alpha]


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


# ============================================================
# メイン計算
# ============================================================

def run_single_point(
    rho: float, delta: float, sigma: float,
    alpha: float, beta: float,
    n_target: int, gamma: float,
    verbose: bool = False,
) -> ExperimentResult:
    """指定パラメータで理論値を計算し ExperimentResult を返す."""
    c = BASELINE["c"]
    b = BASELINE["b"]
    mu = BASELINE["mu"]

    C0, C1 = build_mmpp(rho, delta, sigma, c, b, mu)

    # BASELINE から alpha/beta を除いた辞書を作成し, スイープ値で上書きする
    params_dict = {k: v for k, v in BASELINE.items() if k not in ("alpha", "beta")}
    params = PredictiveModelParameters(
        C0=C0, C1=C1,
        alpha=alpha, beta=beta,
        n_target=n_target, gamma=gamma,
        **params_dict,
    )

    if verbose:
        print(f"  Solving: {params.summary()}")

    Q = build_generator(params)
    pi = solve_stationary(Q)
    m = Metrics(params, pi)

    return ExperimentResult(
        burst_name="",  # 呼び出し側で埋める
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


def run_all(
    burst_name: str, delta: float, sigma: float,
    n_target: int, gamma: float, n_points: int,
    alpha_gamma_dict: Optional[Dict[float, float]] = None,
    verbose: bool = False,
) -> List[ExperimentResult]:
    """全 alpha 水準 x 全 beta 点で計算を実行する.

    alpha_gamma_dict が指定された場合, alpha ごとに resolve_gamma で
    gamma を切り替える (指定なしなら gamma を共通使用).
    """
    beta_values = np.logspace(
        np.log10(BETA_RANGE[0]), np.log10(BETA_RANGE[1]), n_points
    )
    results: List[ExperimentResult] = []

    total = len(ALPHA_LEVELS) * len(beta_values)
    with tqdm(total=total, desc=f"実験 2-P 計算中 ({burst_name})") as pbar:
        for alpha in ALPHA_LEVELS:
            gamma_for_alpha = resolve_gamma(alpha, gamma, alpha_gamma_dict)
            for beta in beta_values:
                result = run_single_point(
                    RHO_FIXED, delta, sigma, alpha, float(beta),
                    n_target, gamma_for_alpha, verbose,
                )
                result.burst_name = burst_name
                results.append(result)
                pbar.update(1)

    return results


# ============================================================
# 出力: CSV
# ============================================================

def save_csv(results: List[ExperimentResult], filepath: str) -> None:
    """結果を CSV に書き出す (実験 2 ベース版と同じ形式 + Predictive パラメータ列)."""
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


# ============================================================
# 出力: 図
# ============================================================

def plot_results(
    results: List[ExperimentResult], filepath: str,
    burst_name: str, title_suffix: str = "",
) -> None:
    """4 指標の 2x2 サブプロットを作成する (実験 2 ベース版と同じ図フォーマット).

    横軸: beta (対数), 3 alpha 水準を色分けした 3 曲線.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Experiment 2-P: Predictive Response to Delayoff Rate "
        r"$\beta$"
        f" ({burst_name} burst)"
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
        alpha_results.sort(key=lambda r: r.beta)
        betas = [r.beta for r in alpha_results]

        for ax, metric_name, ylabel, log_y in metrics_config:
            values = [getattr(r, metric_name) for r in alpha_results]
            ax.plot(
                betas, values,
                color=alpha_colors[alpha], marker="o", markersize=4,
                label=alpha_labels[alpha],
            )
            ax.set_xlabel(r"$\beta$")
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


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strong-burst", action="store_true",
        help="強バースト水準で実施 (補助実験). デフォルトは中バースト.",
    )
    parser.add_argument(
        "--n-target", type=int, default=DEFAULT_N_TARGET,
        help=f"目標稼働サーバー数 (デフォルト: {DEFAULT_N_TARGET})",
    )
    parser.add_argument(
        "--gamma", type=float, default=DEFAULT_GAMMA,
        help=f"Delayoff 加速係数 (デフォルト: {DEFAULT_GAMMA}). "
             "--alpha-gamma-map 指定時は無視される.",
    )
    parser.add_argument(
        "--alpha-gamma-map", type=str, default=None,
        help="alpha 別 gamma の指定 (例: '0.1:1.0,1.0:100.0,10.0:1000.0'). "
             "指定時は --gamma より優先される.",
    )
    parser.add_argument(
        "--n-points", type=int, default=BETA_N_POINTS,
        help=f"beta 走査点数 (デフォルト: {BETA_N_POINTS})",
    )
    parser.add_argument("--out-dir", default="figures", help="図出力ディレクトリ")
    parser.add_argument("--csv-dir", default="results", help="CSV 出力ディレクトリ")
    parser.add_argument(
        "--quick", action="store_true",
        help="動作確認用の高速モード (n-points=6)",
    )
    parser.add_argument("--verbose", action="store_true", help="各点の詳細を表示")
    args = parser.parse_args()
    if args.quick:
        args.n_points = 6
    if args.alpha_gamma_map:
        args.alpha_gamma_dict = parse_alpha_gamma_map(args.alpha_gamma_map)
    else:
        args.alpha_gamma_dict = None
    return args


def main() -> None:
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.csv_dir, exist_ok=True)

    burst_name = "strong" if args.strong_burst else "medium"
    delta, sigma = BURST_LEVELS[burst_name]

    print("=== 実験 2-P: Predictive 版 Delayoff 率走査 ===")
    print(f"バースト水準: {burst_name} (delta={delta}, sigma={sigma})")
    print(f"n_target = {args.n_target}")
    if args.alpha_gamma_dict:
        print(f"gamma    = alpha 別 ({args.alpha_gamma_dict})")
    else:
        print(f"gamma    = {args.gamma}")
    print(f"n_points = {args.n_points}")
    print(f"alpha 水準: {ALPHA_LEVELS}")

    results = run_all(
        burst_name, delta, sigma,
        args.n_target, args.gamma, args.n_points,
        args.alpha_gamma_dict, args.verbose,
    )

    gamma_tag = "gmap" if args.alpha_gamma_dict else f"g{args.gamma}"
    file_stem = f"experiment_2P_{burst_name}_nt{args.n_target}_{gamma_tag}"
    csv_path = os.path.join(args.csv_dir, f"{file_stem}.csv")
    save_csv(results, csv_path)

    fig_path = os.path.join(args.out_dir, f"{file_stem}.png")
    gamma_desc = "alpha別" if args.alpha_gamma_dict else str(args.gamma)
    title_suffix = (
        f"c={BASELINE['c']}, K={BASELINE['K']}, b={BASELINE['b']}, "
        f"rho={RHO_FIXED}, n_target={args.n_target}, gamma={gamma_desc}"
    )
    plot_results(results, fig_path, burst_name, title_suffix)

    if args.alpha_gamma_dict:
        print("\n=== 使用した alpha 別 gamma マッピング ===")
        for alpha, gamma in sorted(args.alpha_gamma_dict.items()):
            print(f"  alpha={alpha}: gamma={gamma}")

    print("\n=== 保存則チェック ===")
    for r in [results[0], results[-1]]:
        conservation = r.E_B + r.E_S + r.E_I + r.E_off
        print(
            f"alpha={r.alpha}, beta={r.beta:.4g}: "
            f"E[B]+E[S]+E[I]+E[Off] = {conservation:.4f} (c={BASELINE['c']})"
        )

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
