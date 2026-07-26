"""実験 0 (理論とシミュレーションの整合性検証) の再実装.

実験計画のパラメータ (c=20, K=200) と対称 2 位相 MMPP (delta, sigma) を用い,
中バースト水準を固定して rho = lambda_bar / (c*b*mu) を [0.3, 0.95] でスイープする.
各点で理論解析 (mmpp) とシミュレーション (mmpp_sim, 独立レプリケーション法) の
6 指標 (P_block, E[N], E[W], lambda_eff, rho_server, Cost) を比較し,
理論値がシミュレーションの 95% CI に入っているか, 保存則 E[B]+E[I]+E[S]+E[Off]=c,
流量バランス lambda_eff = b*mu*E[B] が成り立つかをコンソールに出力する.

使用例:
    python scripts/experiment_0_validation.py --n-points 10 --n-reps 20 \
        --warmup-events 100000 --measurement-events 1000000
    python scripts/experiment_0_validation.py --quick  # 開発中の軽量実行
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
try:
    from _mmpp_burst import (
        build_mmpp as _build_mmpp_generic,
        compute_interarrival_cv,
        check_warmup_duration,
    )
except ImportError:
    from scripts._mmpp_burst import (
        build_mmpp as _build_mmpp_generic,
        compute_interarrival_cv,
        check_warmup_duration,
    )


# 実験計画のベースラインパラメータ (先行研究 6.3 節と同じ規模)
BASELINE = dict(
    c=20,       # サーバー数
    K=200,      # バッファ容量 (cb*5)
    b=5,        # バッチサイズ
    mu=1.0,     # サービス率
    alpha=0.1,  # セットアップ率
    beta=0.005,   # Delayoff 率
)

# 中バースト水準 (対称 2 位相 MMPP): delta = 到着率の相対振れ幅, sigma = 位相遷移率
DELTA = 0.3
SIGMA = 1.0

# rho スイープ範囲・点数の既定値
RHO_RANGE: Tuple[float, float] = (0.3, 0.95)

# (指標キー, Metrics/SimMetrics 共通メソッド名, 表示ラベル)
METRIC_SPECS: List[Tuple[str, str, str]] = [
    ("P_block", "blocking_probability", r"$P_{\mathrm{block}}$"),
    ("E[N]", "mean_queue_length", r"$E[N]$"),
    ("E[W]", "mean_waiting_time", r"$E[W]$"),
    ("lambda_eff", "effective_arrival_rate", r"$\lambda_{\mathrm{eff}}$"),
    ("rho_server", "utilization", r"$\rho_{\mathrm{server}}=E[B]/c$"),
    ("Cost", "energy_cost_paper", "Cost"),
]


def build_mmpp(
    rho: float,
    delta: float = DELTA,
    sigma: float = SIGMA,
    c: int = BASELINE["c"],
    b: int = BASELINE["b"],
    mu: float = BASELINE["mu"],
) -> Tuple[np.ndarray, np.ndarray]:
    """rho を与えて対称 2 位相 MMPP の C0, C1 を構築する (_mmpp_burst.build_mmpp の薄いラッパー)."""
    return _build_mmpp_generic(rho, delta, sigma, c, b, mu)


def build_params(rho: float) -> ModelParameters:
    """rho に対応する ModelParameters を構築する (中バースト水準, BASELINE 固定)."""
    C0, C1 = build_mmpp(rho)
    return ModelParameters(C0=C0, C1=C1, **BASELINE)


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
) -> Tuple[List, SimMetrics]:
    """シミュレーション: 独立レプリケーション法 (run_replications) -> SimMetrics."""
    replications = run_replications(
        params,
        n_reps=n_reps,
        seed0=seed,
        warmup_events=warmup_events,
        measurement_events=measurement_events,
    )
    return replications, SimMetrics(params, replications)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-points", type=int, default=10, help="rho スイープ点数")
    parser.add_argument("--n-reps", type=int, default=20, help="独立レプリケーション数")
    parser.add_argument("--warmup-events", type=int, default=100_000)
    parser.add_argument("--measurement-events", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument(
        "--quick", action="store_true",
        help="開発中の軽量実行用 (n-points=3, n-reps=3, warmup=5000, measurement=20000)",
    )
    args = parser.parse_args()

    if args.quick:
        args.n_points = 3
        args.n_reps = 3
        args.warmup_events = 5_000
        args.measurement_events = 20_000

    return args


def _conservation_and_flow_balance(
    label: str,
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
    print(
        f"    [{label}] E[B]+E[I]+E[S]+E[Off]-c = {conservation_error:.3e}, "
        f"lambda_eff - b*mu*E[B] = {flow_balance_error:.3e}"
    )
    return conservation_error, flow_balance_error


def main() -> None:
    args = parse_args()
    lo, hi = RHO_RANGE
    rho_values = np.linspace(lo, hi, args.n_points)

    theory_vals: Dict[str, List[float]] = {k: [] for k, _, _ in METRIC_SPECS}
    sim_pts: Dict[str, List[float]] = {k: [] for k, _, _ in METRIC_SPECS}
    sim_err_lo: Dict[str, List[float]] = {k: [] for k, _, _ in METRIC_SPECS}
    sim_err_hi: Dict[str, List[float]] = {k: [] for k, _, _ in METRIC_SPECS}

    n_in_ci = 0
    n_checked = 0
    max_conservation_error = 0.0
    max_flow_balance_error = 0.0

    for rho in rho_values:
        params = build_params(float(rho))
        theory = run_theory(params)
        replications, sim_metrics = run_simulation(
            params, args.seed, args.warmup_events,
            args.measurement_events, args.n_reps,
        )
        check_warmup_duration(params, replications)

        print(f"\n=== rho = {rho:.4g} (lambda_bar = {params.lambda_bar:.4g}) ===")
        print(f"  {'metric':<12}{'theory':>12}{'sim':>12}{'CI_lo':>12}{'CI_hi':>12}{'in_CI':>8}")

        point_in_ci = 0
        for key, method_name, _ in METRIC_SPECS:
            theory_val = getattr(theory, method_name)()
            pt, ci_lo, ci_hi = getattr(sim_metrics, f"{method_name}_ci")()

            theory_vals[key].append(theory_val)
            sim_pts[key].append(pt)
            sim_err_lo[key].append(max(pt - ci_lo, 0.0))
            sim_err_hi[key].append(max(ci_hi - pt, 0.0))

            in_ci = ci_lo <= theory_val <= ci_hi
            point_in_ci += int(in_ci)
            n_checked += 1
            n_in_ci += int(in_ci)
            print(
                f"  {key:<12}{theory_val:>12.5g}{pt:>12.5g}"
                f"{ci_lo:>12.5g}{ci_hi:>12.5g}{'OK' if in_ci else 'NG':>8}"
            )

        print(f"  -> {point_in_ci}/{len(METRIC_SPECS)} 指標で理論値が 95% CI 内")

        # 保存則・流量バランスチェック (理論: 定常分布から, シミュレーション: 合算統計の点推定から)
        print("  [保存則・流量バランス]")
        cons_err_t, flow_err_t = _conservation_and_flow_balance(
            "theory", params.c, params.b, params.mu,
            theory.mean_busy(), theory.mean_idle(), theory.mean_setup(), theory.mean_off(),
            theory.effective_arrival_rate(),
        )
        cons_err_s, flow_err_s = _conservation_and_flow_balance(
            "sim", params.c, params.b, params.mu,
            sim_metrics.mean_busy(), sim_metrics.mean_idle(),
            sim_metrics.mean_setup(), sim_metrics.mean_off(),
            sim_metrics.effective_arrival_rate(),
        )
        max_conservation_error = max(max_conservation_error, cons_err_t, cons_err_s)
        max_flow_balance_error = max(max_flow_balance_error, flow_err_t, flow_err_s)

    print(f"\n=== 合格基準サマリ ===")
    print(f"理論値が sim 95% CI 内: {n_in_ci}/{n_checked} (rho 点ごとに 6 指標中 5 つ以上が目安)")
    print(f"保存則の最大誤差: {max_conservation_error:.3e}")
    print(f"流量バランスの最大誤差: {max_flow_balance_error:.3e}")

    _plot(args, rho_values, theory_vals, sim_pts, sim_err_lo, sim_err_hi)


def _plot(
    args: argparse.Namespace,
    rho_values: np.ndarray,
    theory_vals: Dict[str, List[float]],
    sim_pts: Dict[str, List[float]],
    sim_err_lo: Dict[str, List[float]],
    sim_err_hi: Dict[str, List[float]],
) -> None:
    """6 指標を 2x3 のサブプロットにまとめて PNG に保存する."""
    # キャプション用 CV: rho スイープ全域での変動幅を報告する
    # (delta, sigma は rho に依らず固定だが, sigma/lambda_bar 比が変わるため
    #  CV はわずかに rho 依存する. 中バースト水準では変動幅は小さい).
    cvs = [compute_interarrival_cv(*build_mmpp(float(rho))) for rho in rho_values]
    cv_lo, cv_hi = min(cvs), max(cvs)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, _, ylabel) in zip(axes.flat, METRIC_SPECS):
        ax.plot(rho_values, theory_vals[key], "-", color="tab:blue", label="Theory")
        ax.errorbar(
            rho_values, sim_pts[key],
            yerr=[sim_err_lo[key], sim_err_hi[key]],
            fmt="o", color="tab:orange", capsize=3, markersize=4,
            label="Simulation (95% CI)",
        )
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    axes.flat[0].legend(loc="best", fontsize=8)

    if abs(cv_hi - cv_lo) < 1e-3:
        cv_label = f"CV of interarrival time = {cv_lo:.3f}"
    else:
        cv_label = f"CV of interarrival time = {cv_lo:.3f}-{cv_hi:.3f}"
    fig.suptitle(
        f"Experiment 0: Theory vs Simulation (medium burst, "
        f"delta={DELTA}, sigma={SIGMA}, {cv_label})"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "experiment_0_validation.png")
    fig.savefig(out_path, dpi=150)
    print(f"\n図を保存しました: {out_path}")


if __name__ == "__main__":
    main()
