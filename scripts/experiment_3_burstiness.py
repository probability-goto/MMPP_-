"""実験 3: バースト性の 2 つの軸 (振幅 delta, フェーズの長さ r1) の影響 (ベースモデル).

--sweep で切り替える.
    3-A (--sweep delta): delta を [0, 0.95] の 20 点で走査する. 条件は r1 in {0.1, 17.8} x rho in {0.5, 0.7}.
        rho = 0.5 では delta <= 0.95 で rho_B = rho (1 + delta) < 1 のまま, rho = 0.7 では delta = 3/7 で 1 を越える.
    3-B (--sweep r1): r1 = alpha / sigma を [0.01, 1000] の対数 21 点で走査する. 条件は rho_B in {0.8, 1.12}, delta = 0.6.
alpha = 0.005, beta は損益分岐の値 0.6 alpha = 0.003 (mmpp_experiments/design.py, paper/review_spec.md).

使用例:
    python scripts/experiment_3_burstiness.py --sweep delta
    python scripts/experiment_3_burstiness.py --sweep r1
    python scripts/experiment_3_burstiness.py --sweep delta --quick   # 3 点だけ (動作確認用)
"""
import argparse
import os
from typing import List, Sequence

import numpy as np

from mmpp_experiments import design as d
from mmpp_experiments.plotting import plot_four_metrics

DELTA_GRID = [float(x) for x in np.round(np.linspace(0.0, 0.95, 20), 4)]
R1_GRID = [float(x) for x in np.logspace(-2, 3, 21)]
RHO_LEVELS_A = [0.5, 0.7]
R1_LEVELS_A = [d.R1_SHORT, d.R1_LONG]

RHO_COLOR = {0.5: "tab:blue", 0.7: "tab:red"}
RHO_B_COLOR = {0.8: "tab:blue", 1.12: "tab:red"}


def compute_delta(deltas: Sequence[float]) -> List[dict]:
    rows = []
    for rho in RHO_LEVELS_A:
        for r1 in R1_LEVELS_A:
            for delta in deltas:
                rows.append(d.solve_row(f"rho{rho:g}_r1{r1:.3g}", rho, delta, d.sigma_of(r1)))
            print(f"3-A: rho={rho:g}, r1={r1:.3g} 完了")
    return rows


def compute_r1(r1s: Sequence[float]) -> List[dict]:
    rows = []
    for rho_B in d.RHO_B_LEVELS:
        for r1 in r1s:
            rows.append(d.solve_row(f"rhoB{rho_B:g}", d.rho_of(rho_B), d.DELTA, d.sigma_of(r1)))
        print(f"3-B: rho_B={rho_B:g} 完了")
    return rows


def plot_delta(rows: Sequence[dict], out_dir: str) -> List[str]:
    series = []
    for rho in RHO_LEVELS_A:
        for r1, ls in zip(R1_LEVELS_A, ("--", "-")):
            name = f"rho{rho:g}_r1{r1:.3g}"
            series.append((rf"$\rho={rho:g}$, $r_1={r1:.3g}$",
                           [r for r in rows if r["condition"] == name],
                           dict(color=RHO_COLOR[rho], linestyle=ls)))
    return plot_four_metrics(series, "delta", r"$\delta$", out_dir,
                             "experiment_3_burstiness_delta")


def plot_r1(rows: Sequence[dict], out_dir: str) -> List[str]:
    series = [(rf"$\rho_B={rho_B:g}$", [r for r in rows if r["condition"] == f"rhoB{rho_B:g}"],
               dict(color=RHO_B_COLOR[rho_B])) for rho_B in d.RHO_B_LEVELS]
    return plot_four_metrics(series, "r1", r"$r_1=\alpha/\sigma$", out_dir,
                             "experiment_3_burstiness_r1", xlog=True, vline=1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", choices=["delta", "r1"], required=True)
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--csv-dir", default="results")
    parser.add_argument("--quick", action="store_true", help="3 点だけ計算する")
    args = parser.parse_args()
    if args.sweep == "delta":
        rows = compute_delta([0.0, 0.5, 0.95] if args.quick else DELTA_GRID)
        d.report_checks(rows)
        d.write_rows(os.path.join(args.csv_dir, "experiment_3_delta.csv"), rows)
        plot_delta(rows, args.out_dir)
    else:
        rows = compute_r1([0.01, 1.0, 1000.0] if args.quick else R1_GRID)
        d.report_checks(rows)
        d.write_rows(os.path.join(args.csv_dir, "experiment_3_r1.csv"), rows)
        plot_r1(rows, args.out_dir)
    print("キャプション用情報: c=20, K=200, b=5, 1/mu=1, alpha=0.005, beta=0.003 (損益分岐)")


if __name__ == "__main__":
    main()
