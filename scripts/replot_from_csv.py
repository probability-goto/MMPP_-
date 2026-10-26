"""既存の結果 CSV から実験図だけを描き直す (定常分布の再計算はしない).

各実験スクリプト (experiment_*.py) の描画関数をそのまま呼び出し,
CSV の P_block_arrival_stable 列など図に使う列を読み込んで渡す。
理論計算をやり直さないため, 図の見た目だけを変更したときや,
プロットする列を切り替えたときの再描画に使う。

対象:
    実験 1, 2 (パート A と B), 3 (delta/r1), 4 (short/long),
    実験 1-P, 2-P, 3-P, 4-P, 5, 6 (results/ にある CSV すべて)

ベース vs Predictive の比較図 (compare_experiment_*.py) はもともと CSV を
読むスクリプトなので, archive/old_comparison_beta_fixed_no_protect/README.md の
手順でそのまま再生成できる。

使用例:
    python scripts/replot_from_csv.py
    python scripts/replot_from_csv.py --csv-dir results --out-dir figures
"""
import argparse
import csv
import dataclasses
import glob
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiment_1_traffic as e1  # noqa: E402
import experiment_2_delayoff as e2  # noqa: E402
import experiment_3_burstiness as e3  # noqa: E402
import experiment_4_K_sensitivity as e4  # noqa: E402
import experiment_4P_K_sensitivity as e4p  # noqa: E402
import experiment_1P_traffic as e1p  # noqa: E402
import experiment_2P_delayoff as e2p  # noqa: E402
import experiment_3P_burstiness as e3p  # noqa: E402
import experiment_5_n_target as e5  # noqa: E402
import experiment_6_gamma as e6  # noqa: E402

# METRIC_SPECS のキー -> CSV の列名
METRIC_KEY_TO_COL = {
    "P_block_arrival_stable": "P_block_arrival_stable",
    "E[W]": "E_W",
    "Cost": "Cost",
    "ERP": "ERP",
}


def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _sorted_unique(values) -> np.ndarray:
    return np.array(sorted({float(v) for v in values}))


def _theory_vals(rows, group_of, x_col, metric_specs, groups):
    """rows を group ごとに x_col 昇順で並べ, METRIC_SPECS のキーで値リストを作る."""
    by_group = defaultdict(list)
    for r in rows:
        by_group[group_of(r)].append(r)
    vals = {}
    for g in groups:
        sub = sorted(by_group[g], key=lambda r: float(r[x_col]))
        vals[g] = {
            key: [float(r[METRIC_KEY_TO_COL[key]]) for r in sub]
            for key, _, _ in metric_specs
        }
    return vals


def _load_results(cls, path: str) -> list:
    """CSV 行を各実験の ExperimentResult (dataclass) に変換する.

    dataclass にない列 (旧 CSV に残る P_block_arrival など) は無視する。
    """
    casts = {"int": lambda v: int(float(v)), "float": float, "str": str}
    fields = [(f.name, casts[getattr(f.type, "__name__", f.type)])
              for f in dataclasses.fields(cls)]
    return [cls(**{name: cast(r[name]) for name, cast in fields})
            for r in load_rows(path)]


# ---------------------------------------------------------------------------
# ベースモデル (METRIC_SPECS 型)
# ---------------------------------------------------------------------------

def replot_exp1(csv_dir, out_dir):
    e1.plot(load_rows(os.path.join(csv_dir, e1.CSV_NAME)), out_dir)


def replot_exp2(csv_dir, out_dir):
    e2.plot(load_rows(os.path.join(csv_dir, e2.CSV_A)), out_dir)


def replot_exp3(csv_dir, out_dir):
    e3.plot_delta(load_rows(os.path.join(csv_dir, "experiment_3_delta.csv")), out_dir)
    e3.plot_r1(load_rows(os.path.join(csv_dir, "experiment_3_r1.csv")), out_dir)


def _replot_k_sensitivity(module, csv_path, args):
    rows = load_rows(csv_path)
    level_name = rows[0]["burst_name"]
    delta, sigma = float(rows[0]["delta"]), float(rows[0]["sigma"])
    vals = _theory_vals(rows, lambda r: int(float(r["K"])), "rho",
                        module.METRIC_SPECS, module.K_LEVELS)
    rho_values = _sorted_unique(r["rho"] for r in rows)
    module._plot(args, level_name, delta, sigma, rho_values, vals)


def replot_exp4(csv_dir, out_dir):
    for phase in e4.PHASES:
        e4.plot(phase, load_rows(os.path.join(csv_dir, f"experiment_4_{phase}.csv")), out_dir)


def replot_exp4p(csv_dir, out_dir):
    pattern = re.compile(r"experiment_4P_(\w+)_nt(\d+)_g([\d.]+)\.csv$")
    for path in sorted(glob.glob(os.path.join(csv_dir, "experiment_4P_*.csv"))):
        m = pattern.search(os.path.basename(path))
        if not m:
            continue
        args = argparse.Namespace(
            out_dir=out_dir, n_target=int(m.group(2)), gamma=float(m.group(3)),
        )
        _replot_k_sensitivity(e4p, path, args)


# ---------------------------------------------------------------------------
# Predictive 系 (ExperimentResult 型)
# ---------------------------------------------------------------------------

def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def replot_exp1p(csv_dir, out_dir):
    for path in sorted(glob.glob(os.path.join(csv_dir, "experiment_1P_*.csv"))):
        results = _load_results(e1p.ExperimentResult, path)
        e1p.plot_results(results, os.path.join(out_dir, f"{_stem(path)}.png"),
                         f"replot from {os.path.basename(path)}")


def replot_exp2p(csv_dir, out_dir):
    for path in sorted(glob.glob(os.path.join(csv_dir, "experiment_2P_*.csv"))):
        results = _load_results(e2p.ExperimentResult, path)
        e2p.plot_results(results, os.path.join(out_dir, f"{_stem(path)}.png"),
                         results[0].burst_name, f"replot from {os.path.basename(path)}")


def replot_exp3p(csv_dir, out_dir):
    for path in sorted(glob.glob(os.path.join(csv_dir, "experiment_3P_*.csv"))):
        results = _load_results(e3p.ExperimentResult, path)
        plot_fn = (e3p.plot_delta_sweep if "_delta_" in os.path.basename(path)
                   else e3p.plot_sigma_sweep)
        plot_fn(results, os.path.join(out_dir, f"{_stem(path)}.png"),
                f"replot from {os.path.basename(path)}")


def replot_exp5(csv_dir, out_dir):
    for path in sorted(glob.glob(os.path.join(csv_dir, "experiment_5_*.csv"))):
        results = _load_results(e5.ExperimentResult, path)
        e5.plot_results(results, os.path.join(out_dir, f"{_stem(path)}.png"),
                        results[0].burst_name, results[0].gamma,
                        f"replot from {os.path.basename(path)}")


def replot_exp6(csv_dir, out_dir):
    for path in sorted(glob.glob(os.path.join(csv_dir, "experiment_6_*.csv"))):
        results = _load_results(e6.ExperimentResult, path)
        e6.plot_results(results, os.path.join(out_dir, f"{_stem(path)}.png"),
                        results[0].burst_name, results[0].n_target,
                        f"replot from {os.path.basename(path)}")


REPLOTTERS = [
    replot_exp1, replot_exp2, replot_exp3, replot_exp4, replot_exp4p,
    replot_exp1p, replot_exp2p, replot_exp3p, replot_exp5, replot_exp6,
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", default="results", help="読み込む CSV のディレクトリ")
    parser.add_argument("--out-dir", default="figures", help="図の出力ディレクトリ")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for fn in REPLOTTERS:
        print(f"\n##### {fn.__name__} #####")
        fn(args.csv_dir, args.out_dir)


if __name__ == "__main__":
    main()
