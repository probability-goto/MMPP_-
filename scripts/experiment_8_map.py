#!/usr/bin/env python
"""実験 8: 位相の情報に価値がある領域を, 公平な比較で 2 枚の地図にする.

各マスで, すべての方策が β を最適化してから比べる (実験 7 と同じ公平な比較).
仮説: 反応的な起動はキューに b 件たまるとすぐ始まるので, P1 が稼げる時間は
サービス時間の尺度に限られ, セットアップが長いほど相対的な効果は小さくなる.

固定: c=20, K=200, b=5, mu=1, delta=0.6. 到着は build_mmpp. rho = rho_B / (1 + delta).
地図 A: alpha=0.1 固定, r1 = alpha/sigma (対数 5 点 [0.1, 100]) x rho_B in {0.6, 0.8, 1.12}
地図 B: rho_B=0.8 固定, r1 x alpha in {0.005, 0.02, 0.1, 1, 10}
        (alpha=0.1 の行は地図 A の rho_B=0.8 の行と同じなので計算を共有する)
sigma = alpha / r1.

系列 (SERIES): Base (ベースモデル), P2 only (n_target=0, gamma in {3,10,30,100,1000}),
P1 only (protect) (n_target in {5,10,15,20}, gamma=1), P1+P2 (protect)
(n_target in {5,10,20} x gamma in {3,10,30,100}). (protect) は
(protect_presetup, protect_delayoff)=(True, True).

β の探し方:
    1. 粗い探索 (stage=coarse): β を [sigma/1000, sigma*100] の対数 9 点. 全系列・全組み合わせ.
    2. 細かい探索 (stage=fine): 各系列で粗い探索の ERP 最小の (n_target, gamma) を固定し,
       その β の両隣の格子点の間を対数 7 点 (両端と中央は計算済み). Base は 21 点.
    3. 範囲の拡張 (stage=extend, fine2): 最良点が β の範囲の端なら, その (n_target, gamma)
       で範囲を 1 桁広げて 3 点を追加し, 最良点が広げた側に移ればその周りを 2. と同じく
       細かくする. n_target と gamma の端はレポートに注記する.
整合性の検査 (stage=check): 各マスで 1 点 (粗い探索の中央の β), Predictive (n_target=0,
gamma=1, 保護なし) がベースモデルと全指標で相対誤差 1e-10 以下で一致することを確かめる.

計算結果は 1 点ごとに results/experiment_8/experiment_8_<行>.csv に追記し, 再実行時は
計算済みの点を飛ばして再開する (次に計算する点は CSV の内容から決まる).
進捗は results/experiment_8/progress.log に追記する.

使用例:
    python scripts/experiment_8_map.py              # 全マスを計算
    python scripts/experiment_8_map.py --row A_rhoB0.6
    python scripts/experiment_8_map.py --report     # 図とレポート
"""
import argparse
import csv
import os
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from mmpp.model import ModelParameters
from mmpp.generator import build_generator as build_base_generator
from mmpp.solver import solve_stationary as solve_base
from mmpp.metrics import Metrics as BaseMetrics

from mmpp_predictive import (
    PredictiveModelParameters, build_generator, solve_stationary, Metrics,
)

try:
    from _mmpp_burst import build_mmpp
    from experiment_7_frontier import pareto_front, interp_cost, dominance, COMMON_KEYS
except ImportError:
    from scripts._mmpp_burst import build_mmpp
    from scripts.experiment_7_frontier import (
        pareto_front, interp_cost, dominance, COMMON_KEYS,
    )


# ============================================================
# パラメータ
# ============================================================

BASELINE = dict(c=20, K=200, b=5, mu=1.0)
DELTA = 0.6
R1_LEVELS = [float(x) for x in np.logspace(-1, 2, 5)]
N_BETA = 9
N_FINE = 7
N_FINE_BASE = 21
N_EXTEND = 3
CONSISTENCY_TOL = 1e-10

# 地図の行: 行の名前 -> (rho_B, alpha)
ROWS: Dict[str, Tuple[float, float]] = {
    "A_rhoB0.6": (0.6, 0.1),
    "A_rhoB0.8": (0.8, 0.1),
    "A_rhoB1.12": (1.12, 0.1),
    "B_alpha0.005": (0.8, 0.005),
    "B_alpha0.02": (0.8, 0.02),
    "B_alpha1": (0.8, 1.0),
    "B_alpha10": (0.8, 10.0),
}
MAP_A = ["A_rhoB0.6", "A_rhoB0.8", "A_rhoB1.12"]          # 行は rho_B
MAP_B = ["B_alpha0.005", "B_alpha0.02", "A_rhoB0.8", "B_alpha1", "B_alpha10"]  # 行は alpha

# 系列名 -> ((n_target, gamma) の一覧, 保護)
SERIES: Dict[str, Tuple[List[Tuple[int, float]], bool]] = {
    "P2 only": ([(0, g) for g in (3.0, 10.0, 30.0, 100.0, 1000.0)], False),
    "P1 only (protect)": ([(nt, 1.0) for nt in (5, 10, 15, 20)], True),
    "P1+P2 (protect)": ([(nt, g) for nt in (5, 10, 20) for g in (3.0, 10.0, 30.0, 100.0)],
                        True),
}
SERIES_ORDER = ["Base"] + list(SERIES)

OUT_DIR = os.path.join("results", "experiment_8")
FIG_DIR = "figures"
LOG_NAME = "progress.log"

warnings.filterwarnings("ignore", message="Predictive モデルの縮約後の帯幅が大きい")


def cell_params(row: str, r1: float) -> dict:
    rho_B, alpha = ROWS[row]
    sigma = alpha / r1
    return dict(rho_B=rho_B, rho=rho_B / (1 + DELTA), alpha=alpha, r1=r1, sigma=sigma)


def coarse_betas(sigma: float) -> List[float]:
    return [float(x) for x in np.logspace(np.log10(sigma / 1000), np.log10(sigma * 100),
                                          N_BETA)]


# ============================================================
# CSV
# ============================================================

METRIC_FIELDS = ["P_block", "P_block_arrival_stable", "E_N", "E_B", "E_I", "E_S",
                 "E_off", "lambda_eff", "E_W", "rho_server", "Cost", "ERP"]
DIAG_FIELDS = ["p1_fire_rate", "p1_launch_rate", "setup_completion_rate",
               "setup_cancel_rate", "delayoff_rate"]
FIELDNAMES = (["row", "rho_B", "rho", "alpha", "r1", "sigma", "delta", "c", "K", "b", "mu",
               "model", "series", "stage", "beta", "n_target", "gamma",
               "protect_presetup", "protect_delayoff"] + METRIC_FIELDS + DIAG_FIELDS +
              ["consistency_max_relerr", "min_pi", "elapsed_s", "check"])


def csv_path(row: str, out_dir: str = OUT_DIR) -> str:
    return os.path.join(out_dir, f"experiment_8_{row}.csv")


def read_rows(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_row(path: str, row: dict) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if new:
            w.writeheader()
        w.writerow(row)


def pkey(r1, series, beta, nt, g) -> Tuple:
    return (float(r1), series, float(beta), int(nt), float(g))


def rkey(r) -> Tuple:
    return pkey(r["r1"], r["series"], r["beta"], r["n_target"], r["gamma"])


# ============================================================
# 1 点の計算
# ============================================================

def _checks(E_B, E_I, E_S, E_off, lambda_eff, P_arr, min_pi) -> str:
    c, b, mu = BASELINE["c"], BASELINE["b"], BASELINE["mu"]
    out = []
    if abs(E_B + E_I + E_S + E_off - c) > 1e-9:
        out.append(f"conservation:{E_B + E_I + E_S + E_off - c:.3e}")
    if abs(lambda_eff - b * mu * E_B) > 1e-9 * abs(lambda_eff):
        out.append(f"flow:{(lambda_eff - b * mu * E_B) / lambda_eff:.3e}")
    if min_pi < 0:
        out.append(f"min_pi:{min_pi:.3e}")
    if not P_arr > 0:
        out.append("P_block_nonpositive")
    return ";".join(out) if out else "ok"


def _mmpp(cp):
    return build_mmpp(cp["rho"], DELTA, cp["sigma"], BASELINE["c"], BASELINE["b"],
                      BASELINE["mu"])


def _row_head(row, cp, model, series, stage, beta, nt, g, protect):
    return {"row": row, "rho_B": cp["rho_B"], "rho": cp["rho"], "alpha": cp["alpha"],
            "r1": cp["r1"], "sigma": cp["sigma"], "delta": DELTA, **BASELINE,
            "model": model, "series": series, "stage": stage, "beta": beta,
            "n_target": nt, "gamma": g, "protect_presetup": protect,
            "protect_delayoff": protect}


def compute_predictive(row, cp, series, stage, beta, nt, g, protect):
    C0, C1 = _mmpp(cp)
    params = PredictiveModelParameters(
        C0=C0, C1=C1, alpha=cp["alpha"], beta=beta, n_target=nt, gamma=g,
        protect_presetup=protect, protect_delayoff=protect, **BASELINE)
    t = time.time()
    pi = solve_stationary(build_generator(params), solver="gth")
    m = Metrics(params, pi)
    out = _row_head(row, cp, "predictive", series, stage, beta, nt, g, protect)
    out.update({
        "P_block": m.blocking_probability(),
        "P_block_arrival_stable": m.arrival_blocking_probability_stable(),
        "E_N": m.E_N, "E_B": m.E_B, "E_I": m.E_I, "E_S": m.E_S, "E_off": m.E_off,
        "lambda_eff": m.lambda_eff, "E_W": m.mean_waiting_time(),
        "rho_server": m.utilization(), "Cost": m.energy_cost_paper(), "ERP": m.erp_paper(),
        "p1_fire_rate": m.p1_fire_rate, "p1_launch_rate": m.p1_launch_rate,
        "setup_completion_rate": m.setup_completion_rate,
        "setup_cancel_rate": m.setup_cancel_rate, "delayoff_rate": m.delayoff_rate,
        "consistency_max_relerr": "", "min_pi": float(pi.min()),
        "elapsed_s": time.time() - t,
    })
    out["check"] = _checks(m.E_B, m.E_I, m.E_S, m.E_off, m.lambda_eff,
                           m.P_block_arrival_stable, float(pi.min()))
    return out, m


def compute_base(row, cp, stage, beta):
    C0, C1 = _mmpp(cp)
    params = ModelParameters(C0=C0, C1=C1, alpha=cp["alpha"], beta=beta, **BASELINE)
    t = time.time()
    pi = solve_base(build_base_generator(params), solver="gth")
    m = BaseMetrics(params, pi)
    d = m.all_metrics()
    out = _row_head(row, cp, "base", "Base", stage, beta, 0, 1.0, False)
    out.update({
        "P_block": d["P_block"], "P_block_arrival_stable": d["P_block_arrival_stable"],
        "E_N": d["E[j]"], "E_B": d["E[B]"], "E_I": d["E[I]"], "E_S": d["E[S]"],
        "E_off": d["E[Off]"], "lambda_eff": d["lambda_eff"], "E_W": d["E[W]"],
        "rho_server": d["rho"], "Cost": d["cost_paper"], "ERP": d["ERP_paper"],
        # 診断指標はベースモデルの Metrics にはない (P1 は発動しないので 0)
        "p1_fire_rate": 0.0, "p1_launch_rate": 0.0, "setup_completion_rate": "",
        "setup_cancel_rate": "", "delayoff_rate": "",
        "consistency_max_relerr": "", "min_pi": float(pi.min()),
        "elapsed_s": time.time() - t,
    })
    out["check"] = _checks(d["E[B]"], d["E[I]"], d["E[S]"], d["E[Off]"], d["lambda_eff"],
                           d["P_block_arrival_stable"], float(pi.min()))
    return out, m


class ConsistencyError(RuntimeError):
    pass


# ============================================================
# 次に計算する点 (CSV の内容から決める)
# ============================================================

def _f(r, k):
    return float(r[k])


def _series_rows(rows, series, nt=None, g=None):
    out = [r for r in rows if r["series"] == series]
    if nt is not None:
        out = [r for r in out if int(r["n_target"]) == nt and float(r["gamma"]) == g]
    return out


def _fine_between(grid: List[float], beta_best: float, n: int) -> List[float]:
    """格子 grid (昇順) 上の beta_best の両隣の間を対数 n 点 (両端を含む)."""
    i = int(np.argmin([abs(np.log(beta_best / x)) for x in grid]))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    return [float(x) for x in np.logspace(np.log10(lo), np.log10(hi), n)]


def _combos(series):
    if series == "Base":
        return [(0, 1.0)]
    return SERIES[series][0]


def next_points(row: str, r1: float, rows: List[dict]) -> List[Tuple]:
    """マス (row, r1) で次に計算する点の一覧 (series, stage, beta, n_target, gamma).

    空なら, そのマスは完了.
    """
    cp = cell_params(row, r1)
    betas = coarse_betas(cp["sigma"])
    done = {rkey(r) for r in rows}

    def missing(pts):
        return [p for p in pts if pkey(r1, p[0], p[2], p[3], p[4]) not in done]

    # 1. 粗い探索と整合性の検査
    pts = [("Base", "coarse", b, 0, 1.0) for b in betas]
    pts += [("Check", "check", betas[N_BETA // 2], 0, 1.0)]
    for s in SERIES:
        pts += [(s, "coarse", b, nt, g) for nt, g in _combos(s) for b in betas]
    m = missing(pts)
    if m:
        return m

    out = []
    for s in SERIES_ORDER:
        srows = _series_rows(rows, s)
        coarse = [r for r in srows if r["stage"] == "coarse"]
        cb = min(coarse, key=lambda r: _f(r, "ERP"))
        nt, g = int(cb["n_target"]), float(cb["gamma"])
        n = N_FINE_BASE if s == "Base" else N_FINE
        # 2. 細かい探索
        fine = [(s, "fine", b, nt, g) for b in _fine_between(betas, _f(cb, "beta"), n)]
        mf = missing(fine)
        if mf:
            out += mf
            continue
        # 3. 範囲の拡張 (最良点が粗い格子の端にあるとき)
        fixed = _series_rows(rows, s, nt, g)
        best = min(fixed, key=lambda r: _f(r, "ERP"))
        bb = _f(best, "beta")
        lo_edge, hi_edge = betas[0], betas[-1]
        ext = []
        if bb <= lo_edge * (1 + 1e-12):
            ext = [lo_edge * 10 ** (-j / N_EXTEND) for j in range(1, N_EXTEND + 1)]
        elif bb >= hi_edge * (1 - 1e-12):
            ext = [hi_edge * 10 ** (j / N_EXTEND) for j in range(1, N_EXTEND + 1)]
        ext_done = [r for r in fixed if r["stage"] in ("extend", "fine2")]
        if ext or ext_done:
            if not ext:  # 拡張済みで最良点が端から離れた場合
                ext_betas = sorted({_f(r, "beta") for r in fixed if r["stage"] == "extend"})
            else:
                ext_betas = ext
            me = missing([(s, "extend", b, nt, g) for b in ext_betas])
            if me:
                out += me
                continue
            # 格子の点 (粗い探索と拡張) の中の最良点が広げた側に移ったら,
            # その周りを細かくする (細かい探索の点は基準にしないので繰り返さない)
            fixed = _series_rows(rows, s, nt, g)
            on_grid = [r for r in fixed if r["stage"] in ("coarse", "extend")]
            gbest = min(on_grid, key=lambda r: _f(r, "ERP"))
            if gbest["stage"] == "extend":
                grid = sorted(_f(r, "beta") for r in on_grid)
                f2 = [(s, "fine2", b, nt, g)
                      for b in _fine_between(grid, _f(gbest, "beta"), n)]
                out += missing(f2)
    return out


# ============================================================
# 実行
# ============================================================

def estimated_points_per_cell() -> int:
    coarse = N_BETA * (1 + sum(len(v[0]) for v in SERIES.values())) + 1
    fine = (N_FINE_BASE - 3) + len(SERIES) * (N_FINE - 3)
    return coarse + fine


def log_progress(out_dir, done, total, t0, avg, last):
    line = (f"{datetime.now().isoformat(timespec='seconds')} "
            f"done={done}/{total} elapsed={time.time() - t0:.1f}s "
            f"eta={max(total - done, 0) * avg:.1f}s last={last}")
    with open(os.path.join(out_dir, LOG_NAME), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def all_cells(rows_order: List[str]) -> List[Tuple[str, float]]:
    return [(row, r1) for row in rows_order for r1 in R1_LEVELS]


def run(rows_order: List[str], out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    unique_rows = list(dict.fromkeys(MAP_A + MAP_B))
    cells_all = all_cells(unique_rows)
    total = len(cells_all) * estimated_points_per_cell()
    done = sum(len(read_rows(csv_path(r, out_dir))) for r in unique_rows)
    print(f"総点数 (見込み) {total} (計算済み {done}), マス {len(cells_all)}", flush=True)
    t0 = time.time()
    n_new = 0
    last_log = t0
    for row in rows_order:
        path = csv_path(row, out_dir)
        for r1 in R1_LEVELS:
            cp = cell_params(row, r1)
            while True:
                rows = [r for r in read_rows(path) if float(r["r1"]) == r1]
                pts = next_points(row, r1, rows)
                if not pts:
                    break
                for series, stage, beta, nt, g in pts:
                    last = (f"{row},r1={r1:.4g},{series},{stage},beta={beta:.4g},"
                            f"n_target={nt},gamma={g:g}")
                    if series == "Base":
                        out, _ = compute_base(row, cp, stage, beta)
                    elif series == "Check":
                        out, m_p = compute_predictive(row, cp, "Check", stage, beta, 0, 1.0,
                                                      False)
                        _, m_b = compute_base(row, cp, "check", beta)
                        dp, db = m_p.all_metrics(), m_b.all_metrics()
                        rel = max(0.0 if max(abs(db[k]), abs(dp[k])) == 0 else
                                  abs(db[k] - dp[k]) / max(abs(db[k]), abs(dp[k]))
                                  for k in COMMON_KEYS)
                        out["consistency_max_relerr"] = rel
                        if rel > CONSISTENCY_TOL:
                            out["check"] = (out["check"] + ";" if out["check"] != "ok"
                                            else "") + f"consistency:{rel:.2e}"
                            append_row(path, out)
                            log_progress(out_dir, done + 1, total, t0, 0.0,
                                         last + " CONSISTENCY_FAIL")
                            raise ConsistencyError(
                                f"{row}, r1={r1:.4g}, beta={beta:.4g}: ベースモデルと "
                                f"Predictive (n_target=0, gamma=1) が一致しない (rel={rel:.2e})")
                    else:
                        protect = SERIES[series][1]
                        out, _ = compute_predictive(row, cp, series, stage, beta, nt, g,
                                                    protect)
                    append_row(path, out)
                    done += 1
                    n_new += 1
                    if out["check"] != "ok":
                        print(f"  検査違反: {last}: {out['check']}", flush=True)
                    if done % 10 == 0 or time.time() - last_log >= 60:
                        log_progress(out_dir, done, total, t0, (time.time() - t0) / n_new,
                                     last)
                        last_log = time.time()
            if n_new:
                log_progress(out_dir, done, total, t0, (time.time() - t0) / n_new,
                             f"{row},r1={r1:.4g} マス完了")
                last_log = time.time()
        print(f"行 {row} 完了", flush=True)


# ============================================================
# 集計
# ============================================================

def _convert(rows: List[dict]) -> List[dict]:
    for r in rows:
        for k in METRIC_FIELDS + DIAG_FIELDS + ["beta", "gamma", "r1", "sigma", "alpha",
                                                 "rho_B", "rho"]:
            r[k] = float(r[k]) if str(r[k]).strip() != "" else float("nan")
        r["n_target"] = int(r["n_target"])
    return rows


def load_row(row: str, out_dir: str = OUT_DIR) -> List[dict]:
    return _convert(read_rows(csv_path(row, out_dir)))


def cell_summary(rows: List[dict]) -> dict:
    """1 マスの各系列の量 (ERP の相対差, Cost の削減率の最大値, 支配, 最良点)."""
    base = [r for r in rows if r["series"] == "Base"]
    base_best = min(base, key=lambda r: r["ERP"])
    fb = pareto_front([(r["E_W"], r["Cost"], i) for i, r in enumerate(base)])
    xmin, xmax = fb[0][0], fb[-1][0]
    levels = [xmin + f * (xmax - xmin) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
    check = [r for r in rows if r["series"] == "Check"]
    out = {"base_best": base_best, "n_base_front": len(fb),
           "consistency": max(float(r["consistency_max_relerr"]) for r in check)
           if check else float("nan"),
           "n_check_bad": sum(1 for r in rows if r["check"] != "ok"),
           "series": {}}
    for s in SERIES:
        sr = [r for r in rows if r["series"] == s]
        best = min(sr, key=lambda r: r["ERP"])
        fs = pareto_front([(r["E_W"], r["Cost"], i) for i, r in enumerate(sr)])
        reds = []
        for x in levels:
            cs = interp_cost(fs, x)
            if cs is not None:
                reds.append(1 - cs / interp_cost(fb, x))
        d = dominance(fs, fb)
        coarse = coarse_betas(best["sigma"])
        notes = []
        allb = sorted({r["beta"] for r in sr if r["n_target"] == best["n_target"]
                       and r["gamma"] == best["gamma"]})
        if best["beta"] == allb[0] or best["beta"] == allb[-1]:
            ext = any(r["stage"] == "extend" for r in sr)
            notes.append("β が範囲の端" + (" (広げた後も)" if ext else ""))
        nts = sorted({nt for nt, _ in SERIES[s][0]})
        gs = sorted({g for _, g in SERIES[s][0]})
        if len(nts) > 1 and best["n_target"] in (nts[0], nts[-1]):
            notes.append("n_target が" + ("下端" if best["n_target"] == nts[0] else
                                           "上端 (=c)" if nts[-1] == BASELINE["c"] else "上端"))
        if len(gs) > 1 and best["gamma"] in (gs[0], gs[-1]):
            notes.append("γ が" + ("下端" if best["gamma"] == gs[0] else "上端"))
        out["series"][s] = {
            "best": best, "erp_rel": best["ERP"] / base_best["ERP"] - 1,
            "cost_red_max": max(reds) if reds else float("nan"),
            "dom": d, "beta_star": best["beta"], "r2_star": best["sigma"] / best["beta"],
            "notes": notes, "coarse_edge": (coarse[0], coarse[-1]),
        }
    return out


def summaries(out_dir: str = OUT_DIR) -> Dict[Tuple[str, float], dict]:
    res = {}
    for row in dict.fromkeys(MAP_A + MAP_B):
        rows = load_row(row, out_dir)
        for r1 in R1_LEVELS:
            cr = [r for r in rows if r["r1"] == r1]
            if cr and any(r["series"] == "Base" for r in cr) and \
                    all(any(r["series"] == s for r in cr) for s in SERIES):
                res[(row, r1)] = cell_summary(cr)
    return res


def _row_label(row: str, which: str) -> str:
    rho_B, alpha = ROWS[row]
    return f"ρ_B={rho_B:g}" if which == "A" else f"α={alpha:g}"


def _matrix(sm, rows_order, value_fn) -> np.ndarray:
    M = np.full((len(rows_order), len(R1_LEVELS)), np.nan)
    for i, row in enumerate(rows_order):
        for j, r1 in enumerate(R1_LEVELS):
            if (row, r1) in sm:
                M[i, j] = value_fn(sm[(row, r1)])
    return M


def _md_matrix(title, sm, rows_order, which, value_fn, fmt) -> List[str]:
    lines = [title, "", "| | " + " | ".join(f"r1={r1:.3g}" for r1 in R1_LEVELS) + " |",
             "|---|" + "---|" * len(R1_LEVELS)]
    for row in rows_order:
        cells = []
        for r1 in R1_LEVELS:
            cells.append(fmt(value_fn(sm[(row, r1)])) if (row, r1) in sm else "—")
        lines.append(f"| {_row_label(row, which)} | " + " | ".join(cells) + " |")
    return lines + [""]


def plot_heatmaps(sm, rows_order, which, fig_dir=FIG_DIR) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    x = np.log10(R1_LEVELS)
    ylabels = [_row_label(r, which).replace("ρ_B", r"$\rho_B$").replace("α", r"$\alpha$")
               for r in rows_order]
    for stem, fn, cbar in [
        ("erp", lambda c, s: 100 * c["series"][s]["erp_rel"], r"$\Delta$ERP vs Base [%]"),
        ("cost", lambda c, s: 100 * c["series"][s]["cost_red_max"],
         "max Cost reduction at equal $E[W]$ [%]"),
    ]:
        mats = {s: _matrix(sm, rows_order, lambda c, s=s: fn(c, s)) for s in SERIES}
        vmax = max(np.nanmax(np.abs(M)) for M in mats.values())
        vmax = vmax if vmax > 0 else 1.0
        fig, axes = plt.subplots(1, len(SERIES), figsize=(5.2 * len(SERIES), 0.9 *
                                 len(rows_order) + 2.2), sharey=True)
        for ax, s in zip(axes, SERIES):
            M = mats[s]
            # 改善 (ERP の相対差が負, Cost の削減率が正) を青にそろえる
            cmap = "RdBu_r" if stem == "erp" else "RdBu"
            im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto",
                           origin="lower")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if np.isfinite(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                                fontsize=10,
                                color="white" if abs(M[i, j]) > 0.6 * vmax else "black")
            ax.set_xticks(range(len(x)))
            ax.set_xticklabels([f"{v:.2f}" for v in x])
            ax.set_xlabel(r"$\log_{10} r_1$", fontsize=14)
            ax.set_title(s, fontsize=13)
        axes[0].set_yticks(range(len(rows_order)))
        axes[0].set_yticklabels(ylabels, fontsize=12)
        fig.tight_layout(rect=(0, 0, 0.92, 1))
        cax = fig.add_axes([0.93, 0.2, 0.012, 0.6])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(cbar, fontsize=11)
        png = os.path.join(fig_dir, f"experiment_8_{which}_{stem}.png")
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(png, dpi=120, bbox_inches="tight")
        fig.savefig(os.path.splitext(png)[0] + ".pdf", dpi=120, bbox_inches="tight")
        plt.close(fig)
        files += [png, os.path.splitext(png)[0] + ".pdf"]
    return files


def plot_alpha_lines(sm, fig_dir=FIG_DIR) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    alphas = [ROWS[r][1] for r in MAP_B]
    for r1 in R1_LEVELS:
        ys = [100 * sm[(row, r1)]["series"]["P1 only (protect)"]["erp_rel"]
              if (row, r1) in sm else np.nan for row in MAP_B]
        ax.plot(alphas, ys, marker="o", markersize=5, label=f"$r_1$={r1:.3g}")
    ax.axhline(0, color="black", linestyle=":", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$", fontsize=16)
    ax.set_ylabel(r"$\Delta$ERP vs Base [%]", fontsize=16)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5,
               fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    png = os.path.join(fig_dir, "experiment_8_B_p1_alpha.png")
    fig.savefig(png, dpi=120, bbox_inches="tight")
    fig.savefig(os.path.splitext(png)[0] + ".pdf", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [png, os.path.splitext(png)[0] + ".pdf"]


def fire_points(sm) -> List[dict]:
    """P1 only (protect) の ERP 最小点での p1_fire_rate (重複のないマス)."""
    out = []
    for (row, r1), c in sm.items():
        b = c["series"]["P1 only (protect)"]["best"]
        sigma = b["sigma"]
        out.append({"row": row, "r1": r1, "p1_fire_rate": b["p1_fire_rate"],
                    # 位相 0->1 の遷移 1 回あたりの発動確率 (sigma * varpi_0 で割る;
                    # 対称 2 位相なので varpi_0 = 1/2)
                    "fire_per_onset": b["p1_fire_rate"] / (sigma * 0.5),
                    "p_stage2": 1 / (1 + 1 / r1), "n_target": b["n_target"]})
    return out


def plot_fire(sm, fig_dir=FIG_DIR) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = fire_points(sm)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    rows_order = list(dict.fromkeys(MAP_A + MAP_B))
    cmap = plt.get_cmap("tab10")
    for k, row in enumerate(rows_order):
        p = sorted([q for q in pts if q["row"] == row], key=lambda q: q["r1"])
        if not p:
            continue
        rho_B, alpha = ROWS[row]
        lab = rf"$\rho_B$={rho_B:g}, $\alpha$={alpha:g}"
        xs = [np.log10(q["r1"]) for q in p]
        axes[0].plot(xs, [q["p1_fire_rate"] for q in p], marker="o", color=cmap(k),
                     label=lab)
        axes[1].plot(xs, [q["fire_per_onset"] for q in p], marker="o", color=cmap(k))
    xx = np.linspace(-1, 2, 200)
    axes[1].plot(xx, 1 / (1 + 10 ** (-xx)), color="black", linestyle="--", linewidth=2,
                 label=r"$1/(1+1/r_1)$")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("p1_fire_rate", fontsize=16)
    axes[1].set_ylabel(r"p1_fire_rate / $(\sigma \varpi_0)$", fontsize=16)
    for ax in axes:
        ax.set_xlabel(r"$\log_{10} r_1$", fontsize=16)
        ax.grid(True, alpha=0.3)
    axes[1].legend(fontsize=12, loc="lower right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
               fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    png = os.path.join(fig_dir, "experiment_8_p1_fire.png")
    fig.savefig(png, dpi=120, bbox_inches="tight")
    fig.savefig(os.path.splitext(png)[0] + ".pdf", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [png, os.path.splitext(png)[0] + ".pdf"]


def _pct(v):
    return "—" if not np.isfinite(v) else f"{100 * v:+.2f}%"


def report(out_dir: str = OUT_DIR, fig_dir: str = FIG_DIR) -> str:
    sm = summaries(out_dir)
    print("キャプション用情報: c=20, K=200, b=5, mu=1, delta=0.6, rho=rho_B/1.6, "
          "sigma=alpha/r1, r1 は [0.1, 100] の対数 5 点. 地図 A は alpha=0.1, "
          "地図 B は rho_B=0.8. 値は各系列の ERP 最小値の, β を細かくした Base の "
          "ERP 最小値に対する相対差, または同じ E[W] での Cost の削減率の最大値.")
    lines = ["# 実験 8: 位相の情報に価値がある領域の地図", "",
             "results/experiment_8/experiment_8_<行>.csv から生成. 固定: c=20, K=200, b=5, "
             "μ=1, δ=0.6, ρ=ρ_B/1.6, σ=α/r1, r1 ∈ {" +
             ", ".join(f"{r:.3g}" for r in R1_LEVELS) + "}. 地図 A: α=0.1, ρ_B ∈ {0.6, 0.8, 1.12}. "
             "地図 B: ρ_B=0.8, α ∈ {0.005, 0.02, 0.1, 1, 10} (α=0.1 の行は地図 A の ρ_B=0.8 と共有).",
             "",
             f"β の探し方: 粗い探索 (各系列・各組み合わせで [σ/1000, σ×100] の対数 {N_BETA} 点), "
             f"細かい探索 (粗い探索の最良の (n_target, γ) を固定し, 最良の β の両隣の間を対数 "
             f"{N_FINE} 点; Base は {N_FINE_BASE} 点), 最良点が β の範囲の端なら 1 桁広げて "
             f"{N_EXTEND} 点を追加. 相対差と削減率は, 細かくした Base の ERP 最小値と "
             "フロンティアに対するもの. 改善 = −(ERP の相対差).", ""]

    # 整合性の検査
    cons = [(k, v["consistency"]) for k, v in sm.items()]
    bad = [k for k, v in cons if not v <= CONSISTENCY_TOL]
    nbad = sum(v["n_check_bad"] for v in sm.values())
    lines += ["## 整合性の検査と検査違反", "",
              f"各マスの 1 点 (粗い探索の中央の β) で Predictive (n_target=0, γ=1, 保護なし) と "
              f"ベースモデルの全指標を比べた. {len(cons)} マスの最大の相対誤差は "
              f"{max(v for _, v in cons):.2e} で, 1e-10 を超えたマスは {len(bad)} 個. "
              f"check 列の違反 (保存則, 流量バランス, min π ≥ 0, P_block > 0) は {nbad} 件.", ""]

    for which, rows_order in [("A", MAP_A), ("B", MAP_B)]:
        title = "地図 A (α=0.1, 行は ρ_B)" if which == "A" else "地図 B (ρ_B=0.8, 行は α)"
        lines += [f"## {title}", ""]
        for s in SERIES:
            lines += _md_matrix(f"### {s}: ERP の相対差 (対 Base)", sm, rows_order, which,
                                lambda c, s=s: c["series"][s]["erp_rel"], _pct)
            lines += _md_matrix(f"### {s}: 同じ E[W] での Cost の削減率の最大値", sm,
                                rows_order, which,
                                lambda c, s=s: c["series"][s]["cost_red_max"], _pct)
            lines += _md_matrix(f"### {s}: Base のフロンティアに対する支配 (支配される点 / 全点)",
                                sm, rows_order, which,
                                lambda c, s=s: c["series"][s]["dom"],
                                lambda d: f"{d[1]}/{d[2]}")
        lines += [f"### ERP 最小点 (β*, r2*=σ/β*, n_target, γ, p1_fire_rate, 注記)", "",
                  "| 行 | r1 | 系列 | β* | r2* | n_target | γ | p1_fire_rate | 注記 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for row in rows_order:
            for r1 in R1_LEVELS:
                if (row, r1) not in sm:
                    continue
                c = sm[(row, r1)]
                bb = c["base_best"]
                lines.append(f"| {_row_label(row, which)} | {r1:.3g} | Base | {bb['beta']:.3g} | "
                             f"{bb['sigma'] / bb['beta']:.3g} | — | — | — | |")
                for s in SERIES:
                    v = c["series"][s]
                    b = v["best"]
                    lines.append(f"| | | {s} | {b['beta']:.3g} | {v['r2_star']:.3g} | "
                                 f"{b['n_target']} | {b['gamma']:g} | {b['p1_fire_rate']:.3e} | "
                                 f"{', '.join(v['notes'])} |")
        lines.append("")

    # 改善が 0.5% を超えるマス
    lines += ["## 改善が 0.5% を超えるマス", ""]
    for s in SERIES:
        cells = [(k, -v["series"][s]["erp_rel"]) for k, v in sm.items()
                 if -v["series"][s]["erp_rel"] > 0.005]
        if cells:
            kbest = max(cells, key=lambda t: t[1])
            desc = ", ".join(f"{row} r1={r1:.3g} ({100 * imp:.2f}%)"
                             for (row, r1), imp in sorted(cells))
            lines.append(f"- {s}: {len(cells)} マス. 最大 {100 * kbest[1]:.2f}% "
                         f"({kbest[0][0]}, r1={kbest[0][1]:.3g}). 該当: {desc}")
        else:
            lines.append(f"- {s}: 該当なし")
    lines.append("")

    # r1=0.1 の列
    r1min = R1_LEVELS[0]
    col = [(row, s, -sm[(row, r1min)]["series"][s]["erp_rel"])
           for row in dict.fromkeys(MAP_A + MAP_B) if (row, r1min) in sm for s in SERIES]
    if col:
        mx = max(col, key=lambda t: t[2])
        lines += ["## r1=0.1 の列", "",
                  f"r1=0.1 の {len(col) // len(SERIES)} マス × {len(SERIES)} 系列で, 改善 "
                  f"(−ERP の相対差) の最大は {100 * mx[2]:+.3f}% ({mx[0]}, {mx[1]}). "
                  f"改善が 0.1% を超える組は {sum(1 for t in col if t[2] > 0.001)} 個, "
                  f"0.5% を超える組は {sum(1 for t in col if t[2] > 0.005)} 個.", ""]

    # 地図 B の α の傾向
    lines += ["## 地図 B: α と P1 only (protect) の改善", "",
              "改善 = −(ERP の相対差). 各 r1 で α を小さい順に並べたとき, 改善が α とともに"
              "単調に増えるか (α が小さいほど改善が小さいか) と, log10 α と改善の順位相関 "
              "(Spearman).", "",
              "| r1 | " + " | ".join(f"α={ROWS[r][1]:g}" for r in MAP_B) +
              " | 単調 | Spearman |", "|---|" + "---|" * (len(MAP_B) + 2)]
    from scipy.stats import spearmanr, pearsonr
    for r1 in R1_LEVELS:
        imps = [-sm[(row, r1)]["series"]["P1 only (protect)"]["erp_rel"]
                if (row, r1) in sm else np.nan for row in MAP_B]
        if any(not np.isfinite(v) for v in imps):
            continue
        mono = all(imps[i] <= imps[i + 1] + 1e-12 for i in range(len(imps) - 1))
        rho_s = spearmanr(np.log10([ROWS[r][1] for r in MAP_B]), imps).correlation
        lines.append(f"| {r1:.3g} | " + " | ".join(f"{100 * v:+.3f}%" for v in imps) +
                     f" | {'はい' if mono else 'いいえ'} | {rho_s:+.2f} |")
    lines.append("")

    # p1_fire_rate と 1/(1+1/r1)
    fp = fire_points(sm)
    if len(fp) >= 3:
        ps = [q["p_stage2"] for q in fp]
        r_raw = pearsonr(ps, [q["p1_fire_rate"] for q in fp])[0]
        r_norm = pearsonr(ps, [q["fire_per_onset"] for q in fp])[0]
        lines += ["## p1_fire_rate と 1/(1+1/r1) の相関", "",
                  f"P1 only (protect) の ERP 最小点での p1_fire_rate と 1/(1+1/r1) の Pearson の"
                  f"相関係数 ({len(fp)} マス, 重複なし): {r_raw:+.3f}. "
                  f"p1_fire_rate は単位時間あたりの率で σ に比例する成分を含むので, 位相 0→1 の"
                  f"遷移 1 回あたりの発動確率 p1_fire_rate/(σ ϖ₀) (ϖ₀=1/2) との相関も示す: "
                  f"{r_norm:+.3f}.", ""]

    figs = []
    for which, rows_order in [("A", MAP_A), ("B", MAP_B)]:
        figs += plot_heatmaps(sm, rows_order, which, fig_dir)
    figs += plot_alpha_lines(sm, fig_dir)
    figs += plot_fire(sm, fig_dir)
    concl = os.path.join(out_dir, "conclusion.md")
    if os.path.exists(concl):
        with open(concl, encoding="utf-8") as f:
            lines += [f.read().rstrip(), ""]
    lines += ["## 図", ""] + [f"- {f}" for f in figs] + [""]
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--row", choices=list(ROWS) + ["all"], default="all")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        print(report())
        return
    rows_order = list(dict.fromkeys(MAP_A + MAP_B)) if args.row == "all" else [args.row]
    run(rows_order)


if __name__ == "__main__":
    main()
