"""実験 4: バッファ容量 K に対する感度 (ベースモデル).

K in {100, 200, 500, 1000} と rho (実験 1 と同じ 35 点) を走査し, 4 指標を GTH で計算する.
フェーズの長さは r1 = 0.1 (短) と 17.8 (長) の 2 つで, それぞれ別の CSV と図にする.
長フェーズでは K = 1000 の P_block の原因別の内訳も experiment_4_split.csv に書く.
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

# 長フェーズの P_block の原因別の内訳 (論文の表 tab:exp4-split)
SPLIT_K = 1000
SPLIT_RHOS = [0.6, 0.625, 0.65, 0.7, 0.95]
ALWAYS_ON_BETA = 1e-8  # サーバーを切らない場合の beta
SPLIT_FIELDS = ["rho", "rho_B", "P_block_arrival_stable", "excess", "startup",
                "P_block_always_on", "always_on_minus_excess", "check"]


def compute(phase: str, rhos: Sequence[float]) -> List[dict]:
    r1 = PHASES[phase]
    rows = []
    for K in K_LEVELS:
        for rho in rhos:
            rows.append(d.solve_row(f"{phase}_K{K}", rho, d.DELTA, d.sigma_of(r1), K=K))
        print(f"{phase}: K={K} 完了")
    return rows


def split(rows: Sequence[dict]) -> List[dict]:
    """長フェーズ (K = 1000) の P_block を超過による棄却と立ち上がりによる棄却に分ける.

    超過による棄却は (1 + delta)/2 * (rho_B - 1)/rho_B (rho_B <= 1 では 0), 立ち上がりによる棄却は
    P_block との差. サーバーを切らない場合 (beta = 1e-8) の P_block も解き, 超過による棄却との差を記録する
    (この差が小さければ, 立ち上がりによる棄却はサーバーを切ることで生じている).
    """
    out = []
    for rho in SPLIT_RHOS:
        row = next(r for r in rows if int(r["K"]) == SPLIT_K and abs(float(r["rho"]) - rho) < 1e-9)
        rho_g = float(row["rho"])
        rho_B = rho_g * (1 + d.DELTA)
        excess = (1 + d.DELTA) / 2 * (rho_B - 1) / rho_B if rho_B > 1 + 1e-12 else 0.0
        p_block = float(row["P_block_arrival_stable"])
        on = d.solve_row("long_K1000_always_on", rho_g, d.DELTA, d.sigma_of(d.R1_LONG),
                         beta=ALWAYS_ON_BETA, K=SPLIT_K)
        out.append(dict(rho=rho_g, rho_B=rho_B, P_block_arrival_stable=p_block, excess=excess,
                        startup=p_block - excess, P_block_always_on=on["P_block_arrival_stable"],
                        always_on_minus_excess=on["P_block_arrival_stable"] - excess,
                        check=on["check"]))
    return out


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
        if phase == "long" and not args.quick:
            d.write_rows(os.path.join(args.csv_dir, "experiment_4_split.csv"), split(rows),
                         SPLIT_FIELDS)
    print("キャプション用情報: c=20, b=5, 1/mu=1, delta=0.6, alpha=0.005, beta=0.003 (損益分岐), "
          "r1=0.1 (short), 17.8 (long). 点線は rho_B=1")


if __name__ == "__main__":
    main()
