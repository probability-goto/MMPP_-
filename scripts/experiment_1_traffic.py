"""実験 1: トラフィック強度 rho に対する応答 (ベースモデル).

rho を [0.1, 0.95] で 0.025 刻みに走査し, 4 指標 (P_block, E[W], Cost, ERP) を GTH で計算する.
条件はポアソン到着 (delta = 0) と, delta = 0.6 でフェーズの長さ r1 = alpha / sigma が
0.1, 1, 17.8 の 4 つ. alpha = 0.005, beta は損益分岐の値 0.6 alpha = 0.003
(mmpp_experiments/design.py, paper/review_spec.md).

使用例:
    python scripts/experiment_1_traffic.py
    python scripts/experiment_1_traffic.py --quick   # rho を 3 点だけ (動作確認用)
"""
import argparse
import os
from typing import Dict, List, Sequence, Tuple

from mmpp_experiments import design as d
from mmpp_experiments.plotting import plot_four_metrics

# (条件名, delta, r1). ポアソンでは sigma は結果に効かないので r1 = 1 の sigma を使い, r1 列は空にする
CONDITIONS: List[Tuple[str, float, float]] = [
    ("Poisson", 0.0, d.R1_MID),
    ("r1=0.1", d.DELTA, d.R1_SHORT),
    ("r1=1", d.DELTA, d.R1_MID),
    ("r1=17.8", d.DELTA, d.R1_LONG),
]

STYLE: Dict[str, Tuple[str, dict]] = {
    "Poisson": ("Poisson", dict(color="0.4", linestyle="--")),
    "r1=0.1": (r"$r_1=0.1$ (short phase)", dict(color="tab:red")),
    "r1=1": (r"$r_1=1$", dict(color="tab:orange")),
    "r1=17.8": (r"$r_1=17.8$ (long phase)", dict(color="tab:blue")),
}

CSV_NAME = "experiment_1.csv"
FIG_STEM = "experiment_1_traffic"


def compute(rhos: Sequence[float]) -> List[dict]:
    rows = []
    for name, delta, r1 in CONDITIONS:
        for rho in rhos:
            rows.append(d.solve_row(name, rho, delta, d.sigma_of(r1), r1="" if delta == 0 else r1))
        print(f"条件 {name} 完了")
    return rows


def plot(rows: Sequence[dict], out_dir: str) -> List[str]:
    series = [(STYLE[name][0], [r for r in rows if r["condition"] == name], STYLE[name][1])
              for name, _, _ in CONDITIONS]
    return plot_four_metrics(series, "rho", r"$\rho$", out_dir, FIG_STEM,
                             vline=1 / (1 + d.DELTA))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--csv-dir", default="results")
    parser.add_argument("--quick", action="store_true", help="rho を 3 点だけ計算する")
    args = parser.parse_args()
    rhos = [0.1, 0.5, 0.95] if args.quick else d.RHO_GRID
    rows = compute(rhos)
    d.report_checks(rows)
    d.write_rows(os.path.join(args.csv_dir, CSV_NAME), rows)
    plot(rows, args.out_dir)
    print("キャプション用情報: c=20, K=200, b=5, 1/mu=1, alpha=0.005, beta=0.003 (損益分岐), "
          "delta=0.6 (Poisson は delta=0), 点線は rho_B=1 (rho=0.625)")


if __name__ == "__main__":
    main()
