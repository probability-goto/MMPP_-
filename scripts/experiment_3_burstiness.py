"""実験 3 (バースト性の直接的影響, delta 走査と sigma 走査) の実装.

実験計画 5.6 節に沿い, 実験 1・2 で固定していたバースト性パラメータ
(delta: 振幅, sigma: 時定数) 自体を走査軸とし, バースト性が性能に与える
直接的影響を定量化する.

2 サブ実験を --sweep で切り替える:
    - 実験 3-A (--sweep delta): 振幅 delta を走査 (sigma 固定)
    - 実験 3-B (--sweep sigma): 時定数 sigma を走査 (delta 固定)

3 つの alpha 水準 (0.1, 1, 10) について, rho=0.7 固定の下, 4 指標
(P_block^arrival, E[W], Cost, ERP) を理論解析 (mmpp) のみで計算する.

使用例:
    python scripts/experiment_3_burstiness.py --sweep delta
    python scripts/experiment_3_burstiness.py --sweep sigma
    python scripts/experiment_3_burstiness.py --sweep delta --quick  # 開発中の軽量実行 (n-points=5)
"""
import argparse
import csv
import os
import warnings
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


# 実験計画のベースラインパラメータ (実験 1・2 と同じ規模)
BASELINE = dict(
    c=20,       # サーバー数 (実験 1・2 と同じ)
    K=200,      # バッファ容量
    b=5,        # バッチサイズ (先行研究 Fig.8 と同じ)
    mu=1.0,     # サービス率
    beta=0.005, # Delayoff 率 (1/beta = 200 s, 先行研究 Fig.10 と同じ)
    # alpha は水準として走査するのでここには含めない
)

RHO_FIXED = 0.7  # 固定トラフィック強度 (実験 2 と同じ, 先行研究 Fig.8 と同じ)

# alpha 水準の 3 段階定義 (実験 2 と同じ)
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

# 実験 3-A (delta 走査): sigma を実験 1 中バースト水準に固定
SIGMA_FIXED_FOR_DELTA_SWEEP = 0.1
# delta=0 は Poisson, delta->1 は数値特異点回避のため 0.95 まで
DELTA_RANGE: Tuple[float, float] = (0.0, 0.95)

# 実験 3-B (sigma 走査): delta を実験 1 中バースト水準に固定
DELTA_FIXED_FOR_SIGMA_SWEEP = 0.6
# 対数 4 桁
SIGMA_RANGE: Tuple[float, float] = (1e-3, 1e1)

# (指標キー, Metrics 共通メソッド名, 表示ラベル)
METRIC_SPECS: List[Tuple[str, str, str]] = [
    ("P_block_arrival", "arrival_blocking_probability", r"$P_{\mathrm{block}}^{\mathrm{arrival}}$"),
    ("E[W]", "mean_waiting_time", r"$E[W]$ (mean response time)"),
    ("Cost", "energy_cost_paper", "Cost"),
    ("ERP", "erp_paper", "ERP"),
]


def build_params(sweep_var: str, sweep_value: float, alpha: float) -> ModelParameters:
    """sweep_var ('delta' or 'sigma') の値 sweep_value と alpha に対応する ModelParameters を構築する."""
    if sweep_var == "delta":
        delta = sweep_value
        sigma = SIGMA_FIXED_FOR_DELTA_SWEEP
    elif sweep_var == "sigma":
        delta = DELTA_FIXED_FOR_SIGMA_SWEEP
        sigma = sweep_value
    else:
        raise ValueError(f"Unknown sweep_var: {sweep_var}")

    C0, C1 = build_mmpp(
        RHO_FIXED, delta, sigma,
        BASELINE["c"], BASELINE["b"], BASELINE["mu"],
    )
    return ModelParameters(C0=C0, C1=C1, alpha=alpha, **BASELINE)


def run_theory(params: ModelParameters) -> Metrics:
    """理論解析: Q 構築 -> 定常分布 -> Metrics."""
    Q = build_generator(params)
    pi = solve_stationary(Q)
    return Metrics(params, pi)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep", choices=["delta", "sigma"], required=True,
        help="走査変数: delta (振幅) または sigma (時定数)",
    )
    parser.add_argument("--n-points", type=int, default=25, help="走査点数")
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


def _cv_at_sweep_value(sweep_var: str, sweep_value: float) -> float:
    """走査点における到着間隔 CV を返す."""
    if sweep_var == "delta":
        C0, C1 = build_mmpp(
            RHO_FIXED, sweep_value, SIGMA_FIXED_FOR_DELTA_SWEEP,
            BASELINE["c"], BASELINE["b"], BASELINE["mu"],
        )
    else:  # sigma
        C0, C1 = build_mmpp(
            RHO_FIXED, DELTA_FIXED_FOR_SIGMA_SWEEP, sweep_value,
            BASELINE["c"], BASELINE["b"], BASELINE["mu"],
        )
    # delta=0 では CV=1 (Poisson), compute_interarrival_cv は微小数値誤差を出す可能性がある
    if sweep_var == "delta" and sweep_value == 0.0:
        return 1.0
    return compute_interarrival_cv(C0, C1)


def _check_poisson_limit_if_applicable(
    sweep_var: str,
    sweep_value: float,
    theory: Metrics,
    params: ModelParameters,
) -> None:
    """delta=0 の場合, Poisson 到着 (D_M=1 相当) の解析値と一致することを確認する.

    実装上は 2 位相 MMPP で delta=0 なので, 位相分離が消えて実質 Poisson になる.
    比較用の Poisson パラメータを別途構築して指標を比較する.
    """
    if sweep_var != "delta" or sweep_value != 0.0:
        return

    # 同じ lambda_bar を持つ Poisson (D_M=1) パラメータを構築
    lambda_bar_val = RHO_FIXED * BASELINE["c"] * BASELINE["b"] * BASELINE["mu"]
    C0_poisson = np.array([[-lambda_bar_val]])
    C1_poisson = np.array([[lambda_bar_val]])
    params_poisson = ModelParameters(
        C0=C0_poisson, C1=C1_poisson, alpha=params.alpha, **BASELINE,
    )
    Q_poisson = build_generator(params_poisson)
    pi_poisson = solve_stationary(Q_poisson)
    metrics_poisson = Metrics(params_poisson, pi_poisson)

    # 主要指標の一致確認
    tol = 1e-6
    for key, method_name, _ in METRIC_SPECS:
        val_mmpp = getattr(theory, method_name)()
        val_poisson = getattr(metrics_poisson, method_name)()
        rel_err = abs(val_mmpp - val_poisson) / max(abs(val_poisson), 1e-12)
        print(
            f"    [Poisson limit check] {key}: MMPP(delta=0)={val_mmpp:.6g}, "
            f"Poisson={val_poisson:.6g}, rel_err={rel_err:.2e}"
        )
        if rel_err > tol:
            warnings.warn(
                f"delta=0 での Poisson 一致確認に失敗: {key} の相対誤差 {rel_err:.2e} > {tol}",
                RuntimeWarning,
            )


def main() -> None:
    args = parse_args()

    if args.sweep == "delta":
        lo, hi = DELTA_RANGE
        sweep_values = np.linspace(lo, hi, args.n_points)
    else:  # sigma
        lo, hi = SIGMA_RANGE
        sweep_values = np.logspace(np.log10(lo), np.log10(hi), args.n_points)

    theory_vals: Dict[str, Dict[str, List[float]]] = {
        name: {k: [] for k, _, _ in METRIC_SPECS} for name, _ in ALPHA_LEVELS
    }

    max_conservation_error = 0.0
    max_flow_balance_error = 0.0
    csv_rows: List[Dict[str, object]] = []

    for level_name, alpha in ALPHA_LEVELS:
        fixed_label = (
            f"sigma={SIGMA_FIXED_FOR_DELTA_SWEEP}"
            if args.sweep == "delta"
            else f"delta={DELTA_FIXED_FOR_SIGMA_SWEEP}"
        )
        print(
            f"\n########## alpha level = {level_name} "
            f"({fixed_label}, rho={RHO_FIXED}) ##########"
        )

        for sweep_value in tqdm(sweep_values, desc=f"{level_name}"):
            sweep_value = float(sweep_value)
            params = build_params(args.sweep, sweep_value, alpha)
            theory = run_theory(params)
            cv = _cv_at_sweep_value(args.sweep, sweep_value)

            print(f"\n=== [{level_name}] {args.sweep} = {sweep_value:.4g} (CV = {cv:.4f}) ===")
            print(f"  {'metric':<18}{'theory':>14}")
            for key, method_name, _ in METRIC_SPECS:
                theory_val = getattr(theory, method_name)()
                theory_vals[level_name][key].append(theory_val)
                print(f"  {key:<18}{theory_val:>14.5g}")

            _check_poisson_limit_if_applicable(args.sweep, sweep_value, theory, params)

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

            if args.sweep == "delta":
                delta_val, sigma_val = sweep_value, SIGMA_FIXED_FOR_DELTA_SWEEP
            else:
                delta_val, sigma_val = DELTA_FIXED_FOR_SIGMA_SWEEP, sweep_value

            csv_rows.append({
                "sweep_var": args.sweep,
                "delta": delta_val,
                "sigma": sigma_val,
                "alpha": alpha,
                "beta": BASELINE["beta"],
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

    _print_caption_summary(args.sweep, sweep_values, theory_vals)
    _save_csv(args, csv_rows)
    _plot(args, sweep_values, theory_vals)


def _save_csv(args: argparse.Namespace, rows: List[Dict[str, object]]) -> None:
    """delta/sigma スイープ結果を CSV に保存する (実験 3-P との比較用に列を統一)."""
    os.makedirs(args.csv_dir, exist_ok=True)
    csv_path = os.path.join(args.csv_dir, f"experiment_3_{args.sweep}.csv")
    fieldnames = [
        "sweep_var", "delta", "sigma", "alpha", "beta", "rho",
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
    sweep_var: str,
    sweep_values: np.ndarray,
    theory_vals: Dict[str, Dict[str, List[float]]],
) -> None:
    """図のキャプション用サマリ (走査両端での指標表) をコンソールに出力する."""
    print("\n=== キャプション用サマリ ===")

    lo_idx, hi_idx = 0, len(sweep_values) - 1
    cv_lo = _cv_at_sweep_value(sweep_var, float(sweep_values[lo_idx]))
    cv_hi = _cv_at_sweep_value(sweep_var, float(sweep_values[hi_idx]))
    print(
        f"[走査範囲: {sweep_var} = {sweep_values[lo_idx]:.4g} "
        f"(CV = {cv_lo:.3f}) から {sweep_values[hi_idx]:.4g} (CV = {cv_hi:.3f})]"
    )

    print(f"\n[両端の {sweep_var} 値での指標]")
    header = f"  {'level':<10}{sweep_var:>10}{'CV':>8}" + "".join(
        f"{k:>18}" for k, _, _ in METRIC_SPECS
    )
    print(header)
    for level_name, _ in ALPHA_LEVELS:
        for idx, tag in [(lo_idx, "theory(lo)"), (hi_idx, "theory(hi)")]:
            cv_at_idx = _cv_at_sweep_value(sweep_var, float(sweep_values[idx]))
            row = f"  {level_name:<10}{sweep_values[idx]:>10.3g}{cv_at_idx:>8.3f}"
            for key, _, _ in METRIC_SPECS:
                row += f"{theory_vals[level_name][key][idx]:>18.5g}"
            print(row + f"  [{tag}]")


def _plot(
    args: argparse.Namespace,
    sweep_values: np.ndarray,
    theory_vals: Dict[str, Dict[str, List[float]]],
) -> None:
    """4 指標を 2x2 のサブプロットにまとめ, PDF と PNG に保存する."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    for ax, (key, _, ylabel) in zip(axes.flat, METRIC_SPECS):
        for level_name, alpha in ALPHA_LEVELS:
            style = LEVEL_STYLE[level_name]
            ax.plot(
                sweep_values, theory_vals[level_name][key],
                color=style["color"], linestyle=style["linestyle"],
                label=level_name if ax is axes.flat[0] else None,
            )

        # 走査変数の軸スケール: sigma のみ対数
        if args.sweep == "sigma":
            ax.set_xscale("log")
        # P_block^arrival のみ縦軸対数
        if key == "P_block_arrival":
            ax.set_yscale("log")

        xlabel_map = {"delta": r"$\delta$ (burst amplitude)", "sigma": r"$\sigma$ (phase transition rate)"}
        ax.set_xlabel(xlabel_map[args.sweep], fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.3)

    axes.flat[0].legend(loc="best", fontsize=8)

    if args.sweep == "delta":
        title = (
            r"Experiment 3-A: Response to Burst Amplitude $\delta$"
            "\n"
            rf"($c=20$, $K=200$, $b=5$, $\rho={RHO_FIXED}$, "
            rf"$\sigma={SIGMA_FIXED_FOR_DELTA_SWEEP}$, $1/\beta=200$s)"
        )
    else:
        title = (
            r"Experiment 3-B: Response to Phase Transition Rate $\sigma$"
            "\n"
            rf"($c=20$, $K=200$, $b=5$, $\rho={RHO_FIXED}$, "
            rf"$\delta={DELTA_FIXED_FOR_SIGMA_SWEEP}$, $1/\beta=200$s)"
        )
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    os.makedirs(args.out_dir, exist_ok=True)
    pdf_path = os.path.join(args.out_dir, f"experiment_3_burstiness_{args.sweep}.pdf")
    png_path = os.path.join(args.out_dir, f"experiment_3_burstiness_{args.sweep}.png")
    fig.savefig(pdf_path, dpi=100)
    fig.savefig(png_path, dpi=150)
    print(f"\n図を保存しました: {pdf_path}, {png_path}")


if __name__ == "__main__":
    main()
