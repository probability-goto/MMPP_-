"""実験 2: Delayoff 率 beta に対する応答 (ベースモデル).

パート A: beta を [1e-8, 1e2] で対数 41 点 (1 桁あたり 4 点) に走査し, 4 指標を GTH で計算する.
    条件は rho_B in {0.8, 1.12} x sigma in {0.05, 2.81e-4} (alpha = 0.005 で r1 = 0.1, 17.8)
    x alpha in {0.005, 0.1, 1}. sigma を作業負荷の性質として固定するので, alpha を変えると r1 も変わる.
    各 alpha の損益分岐の値 beta_BE = 0.6 alpha も計算に加える.
パート B (フェーズ別): 閑散期の到着率 lambda_0 = rho (1 - delta) c b mu と繁忙期の到着率
    lambda_1 = rho (1 + delta) c b mu をそれぞれポアソン到着とし, 同じ beta の格子を走査する.
    フェーズが十分長いときに, 各フェーズで ERP を最小にする beta を近似的に求める.
    閑散期の系で ERP を最小にする beta (beta_0) は, この格子 (1 桁あたり 4 点) の上の値とする.
パート A の beta* (実際の系で ERP を最小にする beta): 格子 (stage = grid) の最小が走査範囲の内部にある条件では,
    実験 5 (scripts/compare_map.py) の Base と同じ規則でもう一度探す (search_beta_star. stage = coarse,
    fine, extend, fine2). sigma / 1000〜100 sigma の対数 9 点の最小点の両隣の間を対数 21 点で細かくし,
    最小が 9 点の端なら 1 桁外まで 3 点広げて, 広げた点が最小ならその両隣の間を 21 点で細かくする.
    beta* は格子とこの探索の点を合わせた中の ERP 最小点. 同じ系 (rho_B, alpha = 0.005, 長フェーズ) では
    探索の点が最小になり, 実験 5 の Base の beta* と一致する. 格子の最小が走査範囲の端 (1e-8 など) の条件は探索せず, beta* はその端 (論文では「下限」).

使用例:
    python scripts/experiment_2_delayoff.py
    python scripts/experiment_2_delayoff.py --quick   # beta を 5 点だけ, beta* の探索なし (動作確認用)
    python scripts/experiment_2_delayoff.py --plot-only   # 計算せず, CSV から図だけ描き直す
    python scripts/experiment_2_delayoff.py --search-only # 格子は CSV のまま, beta* の探索だけして CSV に加える
"""
import argparse
import os
from typing import List, Sequence, Tuple

import numpy as np

from mmpp_experiments import design as d
from mmpp_experiments.csvio import read_rows
from mmpp_experiments.plotting import save_figure, set_log_xaxis, ylim_from_zero

try:
    import compare_map as cm
except ImportError:
    import scripts.compare_map as cm

ALPHA_LEVELS = [0.005, 0.1, 1.0]
BETA_GRID = [float(x) for x in np.logspace(-8, 2, 41)]
# (フェーズの長さの名前, sigma). alpha = 0.005 での r1 で名前を付ける
PHASES: List[Tuple[str, float]] = [("short", d.sigma_of(d.R1_SHORT)),
                                   ("long", d.sigma_of(d.R1_LONG))]
# パート B のフェーズ: (名前, 到着率の倍率)
PHASEWISE: List[Tuple[str, float]] = [("quiet", 1 - d.DELTA), ("burst", 1 + d.DELTA)]

ALPHA_STYLE = {0.005: dict(color="tab:green"), 0.1: dict(color="tab:blue"),
               1.0: dict(color="tab:orange")}

CSV_A = "experiment_2.csv"
# P_block の図の縦軸の範囲 (線形目盛). 4 つのパネルで共通にする. 最大は 0.542.
# 0 付近の線 (alpha = 1 など) が軸の枠に重ならないよう, 下限を 0 より少し下げる
PBLOCK_YLIM = ylim_from_zero(0.6)
CSV_B = "experiment_2_phasewise.csv"
FIELDS_A = ["phase", "stage"] + d.ROW_FIELDS
FIELDS_B = ["phase", "rho_B_mmpp"] + d.ROW_FIELDS


def betas_for(alpha: float, grid: Sequence[float]) -> List[float]:
    return sorted(set(grid) | {d.break_even_beta(alpha)})


def compute_A(grid: Sequence[float]) -> List[dict]:
    rows = []
    for rho_B in d.RHO_B_LEVELS:
        for phase, sigma in PHASES:
            for alpha in ALPHA_LEVELS:
                for beta in betas_for(alpha, grid):
                    row = d.solve_row(f"rhoB{rho_B:g}_{phase}_a{alpha:g}", d.rho_of(rho_B),
                                      d.DELTA, sigma, alpha=alpha, beta=beta)
                    rows.append(dict(row, phase=phase, stage="grid"))
            print(f"パート A: rho_B={rho_B:g}, {phase} 完了")
    return rows


def search_beta_star(solve, sigma: float) -> List[dict]:
    """実験 5 (scripts/compare_map.py の _refine) の Base と同じ規則で beta を探し, 計算した点を返す.

    solve(beta, stage) は 1 点の行を返す関数.
    """
    betas = cm.coarse_betas(sigma)
    rows = [solve(b, "coarse") for b in betas]

    def best(rr):
        return min(rr, key=lambda r: float(r["ERP"]))

    def add(pts, stage):
        have = {float(r["beta"]) for r in rows}
        rows.extend(solve(b, stage) for b in pts if b not in have)

    add(cm.e8._fine_between(betas, float(best(rows)["beta"]), cm.N_FINE_BASE), "fine")
    bb = float(best(rows)["beta"])
    lo, hi = betas[0], betas[-1]
    if bb <= lo * (1 + 1e-12):
        ext = [lo * 10 ** (-j / cm.N_EXTEND) for j in range(1, cm.N_EXTEND + 1)]
    elif bb >= hi * (1 - 1e-12):
        ext = [hi * 10 ** (j / cm.N_EXTEND) for j in range(1, cm.N_EXTEND + 1)]
    else:
        return rows
    add(ext, "extend")
    on_grid = [r for r in rows if r["stage"] in ("coarse", "extend")]
    gbest = best(on_grid)
    if gbest["stage"] == "extend":
        grid = sorted(float(r["beta"]) for r in on_grid)
        add(cm.e8._fine_between(grid, float(gbest["beta"]), cm.N_FINE_BASE), "fine2")
    return rows


def _at_scan_edge(rows: Sequence[dict]) -> bool:
    betas = [float(r["beta"]) for r in rows]
    b = float(min(rows, key=lambda r: float(r["ERP"]))["beta"])
    return b <= min(betas) * (1 + 1e-12) or b >= max(betas) * (1 - 1e-12)


def compute_search(grid_rows: Sequence[dict]) -> List[dict]:
    """格子の最小が走査範囲の内部にある条件で, beta* を探索する (search_beta_star)."""
    rows = []
    for rho_B in d.RHO_B_LEVELS:
        for phase, sigma in PHASES:
            for alpha in ALPHA_LEVELS:
                g = _select(grid_rows, rho_B=float(d.rho_of(rho_B) * (1 + d.DELTA)), phase=phase,
                            alpha=alpha, stage="grid")
                if _at_scan_edge(g):
                    continue

                def solve(beta, stage):
                    row = d.solve_row(f"rhoB{rho_B:g}_{phase}_a{alpha:g}", d.rho_of(rho_B),
                                      d.DELTA, sigma, alpha=alpha, beta=beta)
                    return dict(row, phase=phase, stage=stage)

                found = search_beta_star(solve, sigma)
                rows += found
                b = min(found, key=lambda r: float(r["ERP"]))
                print(f"beta* の探索: rho_B={rho_B:g}, {phase}, alpha={alpha:g}: "
                      f"beta*={float(b['beta']):.3g} ({len(found)} 点)")
    return rows


def beta_star_row(rows: Sequence[dict], **cond) -> dict:
    """条件の beta* の行: 格子と探索の点を合わせた中の ERP 最小点."""
    return min(_select(rows, **cond), key=lambda r: float(r["ERP"]))


def compute_B(grid: Sequence[float]) -> List[dict]:
    rows = []
    for rho_B in d.RHO_B_LEVELS:
        for phase, factor in PHASEWISE:
            rho = d.rho_of(rho_B) * factor
            for alpha in ALPHA_LEVELS:
                for beta in betas_for(alpha, grid):
                    row = d.solve_row(f"rhoB{rho_B:g}_{phase}_a{alpha:g}", rho, 0.0, 1.0,
                                      alpha=alpha, beta=beta, r1="")
                    rows.append(dict(row, phase=phase, rho_B_mmpp=rho_B))
        print(f"パート B: rho_B={rho_B:g} 完了")
    return rows


def _select(rows, **cond) -> List[dict]:
    out = [r for r in rows if all(abs(float(r[k]) - v) < 1e-12 * max(1, abs(v))
                                  if isinstance(v, float) else r[k] == v
                                  for k, v in cond.items())]
    return sorted(out, key=lambda r: float(r["beta"]))


def plot_A(rows: Sequence[dict], out_dir: str, metric: str, ylabel: str,
           stem: str, ylim=None) -> List[str]:
    """2x2 (行 rho_B, 列 フェーズの長さ) に, alpha ごとの線と損益分岐の点, ERP を最小にする点を描く.

    横軸 beta は対数目盛 (ラベルは 1 桁ごと), 縦軸は線形目盛. ylim を与えると 4 つのパネルの縦軸を
    その範囲で共通にする.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=ylim is not None)
    for i, rho_B in enumerate(d.RHO_B_LEVELS):
        for j, (phase, sigma) in enumerate(PHASES):
            ax = axes[i, j]
            for alpha in ALPHA_STYLE:
                cond = dict(rho_B=float(d.rho_of(rho_B) * (1 + d.DELTA)), phase=phase, alpha=alpha)
                rr = _select(rows, stage="grid", **cond)
                xs, ys = [float(r["beta"]) for r in rr], [float(r[metric]) for r in rr]
                ax.plot(xs, ys, marker="o", markersize=2.5,
                        label=rf"$\alpha={alpha:g}$" if (i, j) == (0, 0) else None,
                        **ALPHA_STYLE[alpha])
                be = [r for r in rr if abs(float(r["beta"]) - d.break_even_beta(alpha)) < 1e-15]
                ax.plot([float(be[0]["beta"])], [float(be[0][metric])], marker="*",
                        markersize=13, markeredgecolor="black", linestyle="none",
                        **ALPHA_STYLE[alpha])
                best = beta_star_row(rows, **cond)
                ax.plot([float(best["beta"])], [float(best[metric])], marker="o",
                        markersize=13, markerfacecolor="none", markeredgecolor="black",
                        markeredgewidth=1.5, linestyle="none", zorder=4)
            set_log_xaxis(ax)
            ax.tick_params(axis="x", labelsize=9)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.set_title(rf"$\rho_B={rho_B:g}$, $\sigma={sigma:.3g}$ "
                         rf"($r_1={d.ALPHA / sigma:.3g}$ at $\alpha=0.005$)", fontsize=12)
            ax.set_xlabel(r"$\beta$", fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.grid(alpha=0.3)
    from matplotlib.lines import Line2D
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([], [], marker="*", markersize=13, color="0.6",
                          markeredgecolor="black", linestyle="none"))
    labels.append(r"$\beta_{\mathrm{BE}}=0.6\alpha$")
    handles.append(Line2D([], [], marker="o", markersize=13, markerfacecolor="none",
                          markeredgecolor="black", markeredgewidth=1.5, linestyle="none"))
    labels.append(r"$\beta^*$ (minimum ERP)")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5,
               fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    paths = save_figure(fig, out_dir, stem)
    plt.close(fig)
    return paths


def plot(rows_a: Sequence[dict], out_dir: str) -> List[str]:
    """パート A の図 (ERP と P_block). パート B は論文で表にするので図は描かない."""
    return (plot_A(rows_a, out_dir, "ERP", "ERP", "experiment_2_delayoff_erp")
            + plot_A(rows_a, out_dir, "P_block_arrival_stable", r"$P_{\mathrm{block}}$",
                     "experiment_2_delayoff_pblock", ylim=PBLOCK_YLIM))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--csv-dir", default="results")
    parser.add_argument("--quick", action="store_true", help="beta を 5 点だけ計算する")
    parser.add_argument("--plot-only", action="store_true", help="計算せず, CSV から図だけ描く")
    parser.add_argument("--search-only", action="store_true",
                        help="格子は CSV のまま, beta* の探索だけして CSV に加え, 図を描く")
    args = parser.parse_args()
    path_a = os.path.join(args.csv_dir, CSV_A)
    if args.plot_only:
        plot(read_rows(path_a), args.out_dir)
        return
    if args.search_only:
        grid_rows = [dict(r, stage=r.get("stage") or "grid") for r in read_rows(path_a)
                     if (r.get("stage") or "grid") == "grid"]
        rows_a = grid_rows + compute_search(grid_rows)
        d.report_checks(rows_a)
        d.write_rows(path_a, rows_a, FIELDS_A)
        plot(rows_a, args.out_dir)
        return
    grid = [float(x) for x in np.logspace(-8, 2, 5)] if args.quick else BETA_GRID
    rows_a, rows_b = compute_A(grid), compute_B(grid)
    if not args.quick:
        rows_a += compute_search(rows_a)
    d.report_checks(rows_a + rows_b)
    d.write_rows(os.path.join(args.csv_dir, CSV_A), rows_a, FIELDS_A)
    d.write_rows(os.path.join(args.csv_dir, CSV_B), rows_b, FIELDS_B)
    plot(rows_a, args.out_dir)
    print("キャプション用情報: c=20, K=200, b=5, 1/mu=1, delta=0.6. 星は損益分岐の beta_BE = 0.6 alpha, 黒の円は ERP を最小にする beta.")


if __name__ == "__main__":
    main()
