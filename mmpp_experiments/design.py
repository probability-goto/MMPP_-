"""第4章の実験 1〜4 で共通の設計 (paper/review_spec.md).

軸は比較の地図 (scripts/compare_map.py) と同じで, バースト期の負荷 rho_B = rho (1 + delta) と,
フェーズの平均の長さとセットアップの平均時間の比 r1 = alpha / sigma で条件を決める.
Delayoff 率は, 指定がなければ損益分岐の固定値 beta_BE = (C_i / C_s) alpha とする.
"""
import csv
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from mmpp import ModelParameters
from mmpp_experiments.checks import point_checks
from mmpp_experiments.evaluate import METRIC_FIELDS, base_metrics, solve_base
from mmpp_experiments.params import SYSTEM, build_mmpp

ALPHA = 0.005
DELTA = 0.6

# mmpp.metrics の energy_cost_paper の C_i / C_s (C_i = 0.6 C_b, C_s = C_b)
IDLE_TO_SETUP_POWER = 0.6

# フェーズの長さの水準 (短と長は比較の地図の格子 r1 = 0.1, 10^1.25 にそろえる)
R1_SHORT = 0.1
R1_MID = 1.0
R1_LONG = float(10 ** 1.25)

# バースト期の負荷の代表値 (delta = 0.6 で rho = 0.5, 0.7)
RHO_B_LEVELS = [0.8, 1.12]

# rho の格子 (0.025 刻み. rho_B = 1 となる rho = 0.625 を含む)
RHO_GRID = [float(x) for x in np.round(np.linspace(0.1, 0.95, 35), 4)]


def break_even_beta(alpha: float) -> float:
    """損益分岐の Delayoff 率: Idle で待つ電力がセットアップのエネルギーと等しくなる平均待ち時間の逆数."""
    return IDLE_TO_SETUP_POWER * alpha


def sigma_of(r1: float, alpha: float = ALPHA) -> float:
    return alpha / r1


def rho_of(rho_B: float, delta: float = DELTA) -> float:
    return rho_B / (1 + delta)


def base_params(rho: float, delta: float, sigma: float, alpha: float = ALPHA,
                beta: Optional[float] = None, K: int = SYSTEM["K"]) -> ModelParameters:
    """対称 2 位相 MMPP 到着のベースモデルのパラメータ (beta を省くと損益分岐の値)."""
    C0, C1 = build_mmpp(rho, delta, sigma, SYSTEM["c"], SYSTEM["b"], SYSTEM["mu"])
    beta = break_even_beta(alpha) if beta is None else beta
    return ModelParameters(C0=C0, C1=C1, alpha=alpha, beta=beta, **{**SYSTEM, "K": K})


LABEL_FIELDS = ["condition", "rho", "rho_B", "delta", "sigma", "r1", "alpha", "beta", "K"]
ROW_FIELDS = LABEL_FIELDS + METRIC_FIELDS + ["N_states", "min_pi", "cons_err", "flow_rel_err",
                                             "check"]


def solve_row(condition: str, rho: float, delta: float, sigma: float, alpha: float = ALPHA,
              beta: Optional[float] = None, K: int = SYSTEM["K"],
              r1: Optional[float] = None) -> Dict[str, object]:
    """1 点を GTH で解き, 条件, 指標, 検査の列を持つ行を返す.

    r1 を省くと alpha / sigma. ポアソン (delta = 0) のように r1 が意味を持たない行では r1 = "" を渡す.
    """
    params = base_params(rho, delta, sigma, alpha, beta, K)
    theory = solve_base(params)
    row: Dict[str, object] = dict(
        condition=condition, rho=rho, rho_B=rho * (1 + delta), delta=delta, sigma=sigma,
        r1=alpha / sigma if r1 is None else r1, alpha=alpha, beta=params.beta, K=K,
    )
    row.update(base_metrics(theory))
    row["N_states"] = params.N
    row.update(point_checks(params, theory))
    return row


def write_rows(path: str, rows: Sequence[dict], fieldnames: Sequence[str] = ROW_FIELDS) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"CSV 保存: {path}")


def report_checks(rows: Sequence[dict]) -> int:
    """検査違反の行を表示し, その数を返す."""
    bad = [r for r in rows if r["check"] != "ok"]
    for r in bad:
        print(f"  [検査違反] {r['condition']} rho={r['rho']} beta={r['beta']}: {r['check']}")
    print(f"検査違反の点数: {len(bad)} / {len(rows)}")
    return len(bad)


def as_float(rows: Sequence[dict], key: str) -> List[float]:
    return [float(r[key]) for r in rows]
