"""理論解析 (mmpp) とシミュレーション (mmpp_sim) の性能指標比較プロット.

1 つのパラメータ (lambda, alpha, beta, mu のいずれか) をスイープし,
各点で理論解析 (Q 構築 -> 定常分布 -> Metrics) と
シミュレーション (Simulator.run -> SimMetrics) の両方から性能指標を計算する.
6 指標 (P_block, E[j], E[W], lambda_eff, rho, energy_cost) をサブプロットし,
理論を実線, シミュレーションを 95% CI 付きの点として重ね書きする.

使用例:
    python scripts/compare_theory_sim.py --sweep lambda --n-points 10
"""
import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mmpp import ModelParameters, build_generator, solve_stationary, Metrics
from mmpp_sim import SimMetrics, run_replications


# 中規模パラメータ (c=10, K=50, b=2, D_M=2) をベースラインとする
BASELINE = dict(c=10, K=50, b=2, mu=1.0, alpha=1.0, beta=0.3)
# C0 の非対角 (位相遷移率, スイープ対象以外は固定)
C0_OFFDIAG = np.array([[0.0, 0.3], [0.4, 0.0]])
# C1 の「形状」(位相別到着率の相対比. lambda スイープ時にこれをスケールする)
C1_UNIT = np.array([[0.4, 0.0], [0.0, 0.6]])

# スイープパラメータごとの既定範囲 (linspace の下限・上限)
SWEEP_RANGES: Dict[str, Tuple[float, float]] = {
    "lambda": (5.0, 35.0),
    "alpha": (0.1, 5.0),
    "beta": (0.05, 2.0),
    "mu": (0.3, 3.0),
}

# (指標キー, Metrics/SimMetrics 共通メソッド名, 表示ラベル)
#
# 注: 実行環境に日本語フォントが無いケースを考慮し, PNG 図中の文字は
#     文字化け (tofu box) を避けるため英語表記にする (コード自体のコメントは日本語のまま).
METRIC_SPECS: List[Tuple[str, str, str]] = [
    ("P_block", "blocking_probability", r"$P_{\mathrm{block}}$"),
    ("E[L]", "mean_queue_length", r"$E[L]$"),
    ("E[W]", "mean_waiting_time", r"$E[W]$"),
    ("lambda_eff", "effective_arrival_rate", r"$\lambda_{\mathrm{eff}}$"),
    ("rho", "utilization", r"$\rho$ (utilization)"),
    ("energy_cost", "energy_cost", "energy cost"),
]


def build_params(sweep_name: str, value: float) -> ModelParameters:
    """sweep_name の値を反映した ModelParameters を構築する.

    lambda: C1 全体を value 倍にスケール (到着強度そのものを変える).
    alpha/beta/mu: 該当スカラーパラメータを value に置き換える.
    いずれの場合も C0 の対角成分は (C0+C1)e=0 を満たすよう自動導出する.
    """
    kwargs = dict(BASELINE)
    C1 = C1_UNIT.copy()
    C0_off = C0_OFFDIAG.copy()

    if sweep_name == "lambda":
        C1 = C1 * value
    elif sweep_name == "alpha":
        kwargs["alpha"] = value
    elif sweep_name == "beta":
        kwargs["beta"] = value
    elif sweep_name == "mu":
        kwargs["mu"] = value
    else:
        raise ValueError(f"未知の sweep パラメータ: {sweep_name}")

    row_sum = C0_off.sum(axis=1) + C1.sum(axis=1)
    C0 = C0_off.copy()
    np.fill_diagonal(C0, -row_sum)

    return ModelParameters(C0=C0, C1=C1, **kwargs)


def run_theory(params: ModelParameters) -> Metrics:
    """理論解析: Q 構築 -> 定常分布 -> Metrics."""
    Q = build_generator(params)
    pi = solve_stationary(Q)
    return Metrics(params, pi)


def run_simulation(
    params: ModelParameters,
    seed: int,
    warmup_events: int,
    measurement_events: int,
    n_reps: int,
) -> SimMetrics:
    """シミュレーション: 独立レプリケーション法 (run_replications) -> SimMetrics."""
    replications = run_replications(
        params,
        n_reps=n_reps,
        seed0=seed,
        warmup_events=warmup_events,
        measurement_events=measurement_events,
    )
    return SimMetrics(params, replications)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep", choices=sorted(SWEEP_RANGES), default="lambda",
        help="スイープするパラメータ名",
    )
    parser.add_argument("--n-points", type=int, default=10, help="スイープ点数")
    parser.add_argument("--warmup-events", type=int, default=100_000)
    parser.add_argument("--measurement-events", type=int, default=1_000_000)
    parser.add_argument("--n-reps", type=int, default=20, help="独立レプリケーション数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lo, hi = SWEEP_RANGES[args.sweep]
    values = np.linspace(lo, hi, args.n_points)

    theory_vals: Dict[str, List[float]] = {k: [] for k, _, _ in METRIC_SPECS}
    sim_pts: Dict[str, List[float]] = {k: [] for k, _, _ in METRIC_SPECS}
    sim_err_lo: Dict[str, List[float]] = {k: [] for k, _, _ in METRIC_SPECS}
    sim_err_hi: Dict[str, List[float]] = {k: [] for k, _, _ in METRIC_SPECS}

    for value in values:
        params = build_params(args.sweep, value)
        theory = run_theory(params)
        sim_metrics = run_simulation(
            params, args.seed, args.warmup_events,
            args.measurement_events, args.n_reps,
        )

        for key, method_name, _ in METRIC_SPECS:
            theory_vals[key].append(getattr(theory, method_name)())
            pt, ci_lo, ci_hi = getattr(sim_metrics, f"{method_name}_ci")()
            sim_pts[key].append(pt)
            sim_err_lo[key].append(max(pt - ci_lo, 0.0))
            sim_err_hi[key].append(max(ci_hi - pt, 0.0))

        print(f"[{args.sweep}={value:.4g}] 完了")

    _plot(args, values, theory_vals, sim_pts, sim_err_lo, sim_err_hi)


def _plot(
    args: argparse.Namespace,
    values: np.ndarray,
    theory_vals: Dict[str, List[float]],
    sim_pts: Dict[str, List[float]],
    sim_err_lo: Dict[str, List[float]],
    sim_err_hi: Dict[str, List[float]],
) -> None:
    """6 指標を 2x3 のサブプロットにまとめて PNG に保存する."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (key, _, ylabel) in zip(axes.flat, METRIC_SPECS):
        ax.plot(values, theory_vals[key], "-", color="tab:blue", label="Theory")
        ax.errorbar(
            values, sim_pts[key],
            yerr=[sim_err_lo[key], sim_err_hi[key]],
            fmt="o", color="tab:orange", capsize=3, markersize=4,
            label="Simulation (95% CI)",
        )
        ax.set_xlabel(args.sweep)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    axes.flat[0].legend(loc="best", fontsize=8)
    fig.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"theory_vs_sim_{args.sweep}.png")
    fig.savefig(out_path, dpi=150)
    print(f"図を保存しました: {out_path}")


if __name__ == "__main__":
    main()
