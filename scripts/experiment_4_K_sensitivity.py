"""実験 4: バッファ容量 K に対する感度 (ベースモデル).

K in {100, 200, 500, 1000} と rho (実験 1 と同じ 35 点) を走査し, 4 指標を GTH で計算する.
フェーズの長さは r1 = 0.1 (短) と 17.8 (長) の 2 つで, それぞれ別の CSV と図にする.
delta = 0.6, alpha = 0.005, beta は損益分岐の値 0.6 alpha = 0.003
(mmpp_experiments/design.py, paper/review_spec.md).

使用例:
    python scripts/experiment_4_K_sensitivity.py              # 短と長の両方
    python scripts/experiment_4_K_sensitivity.py --phase long
    python scripts/experiment_4_K_sensitivity.py --quick      # rho を 3 点だけ (動作確認用)
"""
import argparse
import os
from typing import Dict, List, Sequence

from mmpp_experiments import design as d
from mmpp_experiments.plotting import plot_four_metrics

K_LEVELS = [100, 200, 500, 1000]
PHASES: Dict[str, float] = {"short": d.R1_SHORT, "long": d.R1_LONG}
K_COLOR = {100: "tab:red", 200: "tab:orange", 500: "tab:green", 1000: "tab:blue"}


def compute(phase: str, rhos: Sequence[float]) -> List[dict]:
    r1 = PHASES[phase]
    rows = []
    for K in K_LEVELS:
        for rho in rhos:
            rows.append(d.solve_row(f"{phase}_K{K}", rho, d.DELTA, d.sigma_of(r1), K=K))
        print(f"{phase}: K={K} 完了")
    return rows


def plot(phase: str, rows: Sequence[dict], out_dir: str) -> List[str]:
    series = [(f"K={K}", [r for r in rows if int(r["K"]) == K], dict(color=K_COLOR[K]))
              for K in K_LEVELS]
    return plot_four_metrics(series, "rho", r"$\rho$", out_dir,
                             f"experiment_4_K_sensitivity_{phase}", vline=1 / (1 + d.DELTA),
                             legend_ncol=4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=["short", "long", "all"], default="all")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--csv-dir", default="results")
    parser.add_argument("--quick", action="store_true", help="rho を 3 点だけ計算する")
    args = parser.parse_args()
    rhos = [0.1, 0.5, 0.95] if args.quick else d.RHO_GRID
    for phase in (PHASES if args.phase == "all" else [args.phase]):
        rows = compute(phase, rhos)
        d.report_checks(rows)
        d.write_rows(os.path.join(args.csv_dir, f"experiment_4_{phase}.csv"), rows)
        plot(phase, rows, args.out_dir)
    print("キャプション用情報: c=20, b=5, 1/mu=1, delta=0.6, alpha=0.005, beta=0.003 (損益分岐), "
          "r1=0.1 (short), 17.8 (long). 点線は rho_B=1")


if __name__ == "__main__":
    main()
