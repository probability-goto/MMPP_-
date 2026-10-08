#!/usr/bin/env python
"""実験 7: ベースモデルと Predictive を, それぞれの最良の設定どうしで比較する.

各方策が自分のパラメータ (beta, n_target, gamma, 保護) を最適に選べるようにして,
(E[W], Cost) 平面のパレートフロンティアで比較する. 判定したいのは次の 2 点.
    (1) 位相の情報に, beta の調整では得られない価値があるか
    (2) その価値は, 位相を使わずに取り消しを止めるだけ (NoCancel) でも得られるか

固定: c=20, K=200, b=5, mu=1. 到着は build_mmpp (対称 2 位相).
条件: C1〜C5 (CONDITIONS). beta は条件ごとに [sigma/1000, sigma*100] の対数等間隔 13 点.
系列 (SERIES_SPECS):
    Base                ベースモデル (mmpp)
    NoCancel            n_target=0, gamma=1, never_cancel_setup=True
    P2 only             n_target=0, gamma in GAMMAS
    P1 only             n_target in N_TARGETS, gamma=1, 保護なし
    P1 only (protect)   同上, (protect_presetup, protect_delayoff)=(True, True)
    P1+P2               n_target in N_TARGETS, gamma in GAMMAS, 保護なし
    P1+P2 (protect)     同上, 保護あり
整合性の検査: 各 beta で Predictive (n_target=0, gamma=1, 保護なし) を計算し
(系列名 Check), ベースモデルと共通の全指標が相対誤差 1e-10 以下で一致することを
確かめる. 一致しなければ計算を止める.

Base 行の診断指標 (setup_completion_rate 等) は, 同じ beta の Check 行の値を写す
(Check はベースモデルと全指標が一致することを確かめたモデル. diag_source 列で明示).

計算結果は 1 点ごとに results/experiment_7/experiment_7_<条件名>.csv に追記し,
再実行時は計算済みの点 (パラメータの組がキー) を飛ばして再開する.
進捗は results/experiment_7/progress.log に追記する.

補足 (--refine): Base の β 13 点は Predictive の系列 (γ, n_target の組み合わせで
点が多い) より粗いので, フロンティアの差が格子の粗さによるものでないかを確かめる.
Base を同じ範囲の β 121 点 (10 倍細かい) で, NoCancel を 13 点での ERP 最小点の
両隣の β の間を 21 点で計算し, results/experiment_7/experiment_7_refine_<条件名>.csv
に保存する (本計算の CSV は変えない).

補足 2 (--extend): 最良点が走査範囲の端にあった系列について範囲を広げ, 本計算の CSV
に追記する (EXTENSIONS). C4 では β の下限を広げるので, 比較の基準である β を細かくした
Base も同じ下限まで広げ, 補足の CSV に追記する.

使用例:
    python scripts/experiment_7_frontier.py                 # 全条件を計算
    python scripts/experiment_7_frontier.py --condition C1  # 1 条件だけ計算
    python scripts/experiment_7_frontier.py --refine        # 補足: β を細かくした Base / NoCancel
    python scripts/experiment_7_frontier.py --extend        # 補足 2: 端にあった最良点の範囲を広げる
    python scripts/experiment_7_frontier.py --report        # 図とレポートを作る
    python scripts/experiment_7_frontier.py --count         # 点数だけ表示
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
except ImportError:
    from scripts._mmpp_burst import build_mmpp


# ============================================================
# パラメータ
# ============================================================

BASELINE = dict(c=20, K=200, b=5, mu=1.0)

# 名前 -> (rho, delta, sigma, alpha, 位置づけ)
CONDITIONS: Dict[str, Tuple[float, float, float, float, str]] = {
    "C1": (0.5, 0.6, 0.02, 0.1, "有効候補"),
    "C2": (0.7, 0.6, 0.02, 0.1, "容量不足"),
    "C3": (0.5, 0.6, 1.0, 0.1, "位相が短い"),
    "C4": (0.5, 0.6, 0.001, 0.005, "動機の尺度"),
    "C5": (0.5, 0.6, 2.0, 10.0, "高速起動"),
}

N_BETA = 13
GAMMAS = [3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
N_TARGETS = [5, 10, 15, 20]
CONSISTENCY_TOL = 1e-10

# ベースモデルと Predictive の all_metrics に共通のキー (整合性の検査に使う)
COMMON_KEYS = ["P_block", "P_block_arrival_stable", "E[j]", "E[B]", "E[I]", "E[S]",
               "E[Off]", "lambda_eff", "E[W]", "rho", "cost_paper", "ERP_paper"]

SERIES_ORDER = ["Base", "NoCancel", "P2 only", "P1 only", "P1 only (protect)",
                "P1+P2", "P1+P2 (protect)"]

OUT_DIR = os.path.join("results", "experiment_7")
LOG_NAME = "progress.log"
FIG_DIR = "figures"

warnings.filterwarnings("ignore", message="Predictive モデルの縮約後の帯幅が大きい")


def beta_levels(sigma: float) -> List[float]:
    """[sigma/1000, sigma*100] の対数等間隔 N_BETA 点."""
    return [float(x) for x in np.logspace(np.log10(sigma / 1000),
                                          np.log10(sigma * 100), N_BETA)]


def series_specs() -> List[Tuple[str, int, float, bool, bool, bool]]:
    """Predictive の系列の (系列名, n_target, gamma, pp, pd, nc) の一覧 (Base, Check を除く)."""
    specs = [("NoCancel", 0, 1.0, False, False, True)]
    specs += [("P2 only", 0, g, False, False, False) for g in GAMMAS]
    for protect, suffix in [(False, ""), (True, " (protect)")]:
        specs += [("P1 only" + suffix, nt, 1.0, protect, protect, False)
                  for nt in N_TARGETS]
    for protect, suffix in [(False, ""), (True, " (protect)")]:
        specs += [("P1+P2" + suffix, nt, g, protect, protect, False)
                  for nt in N_TARGETS for g in GAMMAS]
    return specs


def condition_points(cond: str) -> List[Tuple]:
    """条件の全点. 各要素は (系列名, beta, n_target, gamma, pp, pd, nc).

    先頭に Base と Check (整合性の検査) を beta ごとに並べ, その後に他の系列を置く.
    """
    _, _, sigma, _, _ = CONDITIONS[cond]
    betas = beta_levels(sigma)
    pts = []
    for beta in betas:
        pts.append(("Check", beta, 0, 1.0, False, False, False))
        pts.append(("Base", beta, 0, 1.0, False, False, False))
    for beta in betas:
        for name, nt, g, pp, pd, nc in series_specs():
            pts.append((name, beta, nt, g, pp, pd, nc))
    return pts


# ============================================================
# CSV
# ============================================================

KEY_FIELDS = ["series", "beta", "n_target", "gamma", "protect_presetup",
              "protect_delayoff", "never_cancel_setup"]
METRIC_FIELDS = ["P_block", "P_block_arrival_stable", "E_N", "E_B", "E_I", "E_S",
                 "E_off", "lambda_eff", "E_W", "rho_server", "Cost", "ERP"]
DIAG_FIELDS = ["p1_fire_rate", "p1_launch_rate", "setup_completion_rate",
               "setup_cancel_rate", "delayoff_rate", "rho_B"]
FIELDNAMES = (["condition", "rho", "delta", "sigma", "alpha", "c", "K", "b", "mu",
               "model"] + KEY_FIELDS + METRIC_FIELDS + DIAG_FIELDS +
              ["diag_source", "consistency_max_relerr", "min_pi", "elapsed_s", "check"])


def csv_path(cond: str, out_dir: str = OUT_DIR) -> str:
    return os.path.join(out_dir, f"experiment_7_{cond}.csv")


def point_key(pt) -> Tuple:
    name, beta, nt, g, pp, pd, nc = pt
    return (name, float(beta), int(nt), float(g), bool(pp), bool(pd), bool(nc))


def row_key(r) -> Tuple:
    return (r["series"], float(r["beta"]), int(r["n_target"]), float(r["gamma"]),
            r["protect_presetup"] == "True", r["protect_delayoff"] == "True",
            r["never_cancel_setup"] == "True")


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


# ============================================================
# 1 点の計算
# ============================================================

def _checks(E_B, E_I, E_S, E_off, lambda_eff, P_arr, min_pi) -> str:
    c, b, mu = BASELINE["c"], BASELINE["b"], BASELINE["mu"]
    out = []
    total = E_B + E_I + E_S + E_off
    if abs(total - c) > 1e-9:
        out.append(f"conservation:{total - c:.3e}")
    flow = b * mu * E_B
    if abs(lambda_eff - flow) > 1e-9 * abs(lambda_eff):
        out.append(f"flow:{(lambda_eff - flow) / lambda_eff:.3e}")
    if min_pi < 0:
        out.append(f"min_pi:{min_pi:.3e}")
    if not P_arr > 0:
        out.append("P_block_nonpositive")
    return ";".join(out) if out else "ok"


def _base_row(cond: str) -> dict:
    rho, delta, sigma, alpha, _ = CONDITIONS[cond]
    return {"condition": cond, "rho": rho, "delta": delta, "sigma": sigma,
            "alpha": alpha, **BASELINE}


def solve_predictive(cond, beta, nt, g, pp, pd, nc):
    rho, delta, sigma, alpha, _ = CONDITIONS[cond]
    C0, C1 = build_mmpp(rho, delta, sigma, BASELINE["c"], BASELINE["b"], BASELINE["mu"])
    params = PredictiveModelParameters(
        C0=C0, C1=C1, alpha=alpha, beta=beta, n_target=nt, gamma=g,
        protect_presetup=pp, protect_delayoff=pd, never_cancel_setup=nc,
        **BASELINE,
    )
    pi = solve_stationary(build_generator(params), solver="gth")
    return params, pi, Metrics(params, pi)


def solve_base_model(cond, beta):
    rho, delta, sigma, alpha, _ = CONDITIONS[cond]
    C0, C1 = build_mmpp(rho, delta, sigma, BASELINE["c"], BASELINE["b"], BASELINE["mu"])
    params = ModelParameters(C0=C0, C1=C1, alpha=alpha, beta=beta, **BASELINE)
    pi = solve_base(build_base_generator(params), solver="gth")
    return params, pi, BaseMetrics(params, pi)


def predictive_row(cond, pt, elapsed, pi, m) -> dict:
    name, beta, nt, g, pp, pd, nc = pt
    row = _base_row(cond)
    row.update({
        "model": "predictive", "series": name, "beta": beta, "n_target": nt,
        "gamma": g, "protect_presetup": pp, "protect_delayoff": pd,
        "never_cancel_setup": nc,
        "P_block": m.blocking_probability(),
        "P_block_arrival_stable": m.arrival_blocking_probability_stable(),
        "E_N": m.E_N, "E_B": m.E_B, "E_I": m.E_I, "E_S": m.E_S, "E_off": m.E_off,
        "lambda_eff": m.lambda_eff, "E_W": m.mean_waiting_time(),
        "rho_server": m.utilization(), "Cost": m.energy_cost_paper(),
        "ERP": m.erp_paper(),
        "p1_fire_rate": m.p1_fire_rate, "p1_launch_rate": m.p1_launch_rate,
        "setup_completion_rate": m.setup_completion_rate,
        "setup_cancel_rate": m.setup_cancel_rate,
        "delayoff_rate": m.delayoff_rate, "rho_B": m.rho_B,
        "diag_source": "self", "consistency_max_relerr": "",
        "min_pi": float(pi.min()), "elapsed_s": elapsed,
    })
    row["check"] = _checks(m.E_B, m.E_I, m.E_S, m.E_off, m.lambda_eff,
                           m.P_block_arrival_stable, float(pi.min()))
    rho, delta = row["rho"], row["delta"]
    if abs(m.rho_B - rho * (1 + delta)) > 1e-12:
        row["check"] = (row["check"] + ";" if row["check"] != "ok" else "") + "rho_B"
    return row


def base_row(cond, beta, elapsed, pi, mb, check_row: Optional[dict],
             series: str = "Base") -> dict:
    row = _base_row(cond)
    d = mb.all_metrics()
    row.update({
        "model": "base", "series": series, "beta": beta, "n_target": 0,
        "gamma": 1.0, "protect_presetup": False, "protect_delayoff": False,
        "never_cancel_setup": False,
        "P_block": d["P_block"], "P_block_arrival_stable": d["P_block_arrival_stable"],
        "E_N": d["E[j]"], "E_B": d["E[B]"], "E_I": d["E[I]"], "E_S": d["E[S]"],
        "E_off": d["E[Off]"], "lambda_eff": d["lambda_eff"], "E_W": d["E[W]"],
        "rho_server": d["rho"], "Cost": d["cost_paper"], "ERP": d["ERP_paper"],
        "diag_source": "Check" if check_row is not None else "none",
        "consistency_max_relerr": (check_row["consistency_max_relerr"]
                                   if check_row is not None else ""),
        "min_pi": float(pi.min()), "elapsed_s": elapsed,
    })
    for f in DIAG_FIELDS:
        # 診断指標はベースモデルの Metrics にはないので Check 行から写す (なければ空)
        row[f] = check_row[f] if check_row is not None else ""
    if check_row is None:
        rho, delta = row["rho"], row["delta"]
        row["rho_B"] = rho * (1 + delta)
        row["p1_fire_rate"] = 0.0
        row["p1_launch_rate"] = 0.0
    row["check"] = _checks(d["E[B]"], d["E[I]"], d["E[S]"], d["E[Off]"],
                           d["lambda_eff"], d["P_block_arrival_stable"], float(pi.min()))
    return row


class ConsistencyError(RuntimeError):
    pass


def run_base_and_check(cond, beta) -> Tuple[dict, dict]:
    """Check (Predictive n_target=0, gamma=1) と Base を計算し, 全指標の一致を確かめる."""
    pt = ("Check", beta, 0, 1.0, False, False, False)
    t = time.time()
    _, pi_p, m_p = solve_predictive(cond, *pt[1:])
    check = predictive_row(cond, pt, time.time() - t, pi_p, m_p)

    t = time.time()
    _, pi_b, m_b = solve_base_model(cond, beta)
    elapsed_b = time.time() - t

    dp, db = m_p.all_metrics(), m_b.all_metrics()
    relerr = {}
    for k in COMMON_KEYS:
        a, b = db[k], dp[k]
        denom = max(abs(a), abs(b))
        relerr[k] = 0.0 if denom == 0 else abs(a - b) / denom
    worst = max(relerr.values())
    check["consistency_max_relerr"] = worst
    if worst > CONSISTENCY_TOL:
        bad = {k: v for k, v in relerr.items() if v > CONSISTENCY_TOL}
        check["check"] = (check["check"] + ";" if check["check"] != "ok" else "") + \
            "consistency:" + "/".join(f"{k}={v:.2e}" for k, v in bad.items())
    base = base_row(cond, beta, elapsed_b, pi_b, m_b, check)
    return check, base


# ============================================================
# 実行
# ============================================================

def total_points(conds: List[str]) -> int:
    return sum(len(condition_points(c)) for c in conds)


def log_progress(out_dir, done, total, t0, avg, last):
    elapsed = time.time() - t0
    line = (f"{datetime.now().isoformat(timespec='seconds')} "
            f"done={done}/{total} elapsed={elapsed:.1f}s "
            f"eta={(total - done) * avg:.1f}s last={last}")
    with open(os.path.join(out_dir, LOG_NAME), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run(conds: List[str], out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    all_conds = list(CONDITIONS)
    total = total_points(all_conds)
    done_keys = {c: {row_key(r) for r in read_rows(csv_path(c, out_dir))}
                 for c in all_conds}
    done = sum(len([p for p in condition_points(c) if point_key(p) in done_keys[c]])
               for c in all_conds)
    todo_n = sum(len([p for p in condition_points(c) if point_key(p) not in done_keys[c]])
                 for c in conds)
    print(f"総点数 {total} (計算済み {done}, 今回計算 {todo_n})", flush=True)

    t0 = time.time()
    n_new = 0
    last_log = t0
    for cond in conds:
        path = csv_path(cond, out_dir)
        pts = [p for p in condition_points(cond) if point_key(p) not in done_keys[cond]]
        i = 0
        while i < len(pts):
            pt = pts[i]
            last = (f"{cond},{pt[0]},beta={pt[1]:.4g},n_target={pt[2]},"
                    f"gamma={pt[3]:g},protect={pt[4]},never_cancel={pt[6]}")
            if pt[0] in ("Check", "Base"):
                # Check と Base は対で計算する (Base の診断指標は Check から写す)
                check, base = run_base_and_check(cond, pt[1])
                for row in (check, base):
                    if point_key((row["series"], row["beta"], row["n_target"],
                                  row["gamma"], row["protect_presetup"],
                                  row["protect_delayoff"],
                                  row["never_cancel_setup"])) not in done_keys[cond]:
                        append_row(path, row)
                        done += 1
                        n_new += 1
                # 対のもう片方が pts に残っていれば飛ばす
                pair = {point_key(("Check",) + pt[1:]), point_key(("Base",) + pt[1:])}
                while i < len(pts) and point_key(pts[i]) in pair:
                    i += 1
                if check["consistency_max_relerr"] > CONSISTENCY_TOL:
                    log_progress(out_dir, done, total, t0, 0.0, last + " CONSISTENCY_FAIL")
                    raise ConsistencyError(
                        f"{cond}, beta={pt[1]:.6g}: ベースモデルと Predictive "
                        f"(n_target=0, gamma=1) が一致しない ({check['check']})")
                row = base
            else:
                t = time.time()
                _, pi, m = solve_predictive(cond, *pt[1:])
                row = predictive_row(cond, pt, time.time() - t, pi, m)
                append_row(path, row)
                done += 1
                n_new += 1
                i += 1
            if row["check"] != "ok":
                print(f"  検査違反: {last}: {row['check']}", flush=True)
            cond_end = i >= len(pts)
            if done % 10 == 0 or time.time() - last_log >= 60 or cond_end:
                log_progress(out_dir, done, total, t0, (time.time() - t0) / n_new, last)
                last_log = time.time()
        print(f"条件 {cond} 完了", flush=True)


# ============================================================
# 補足: β の格子の粗さの影響
# ============================================================

N_BETA_DENSE = 121
N_REFINE = 21


def refine_csv_path(cond: str, out_dir: str = OUT_DIR) -> str:
    return os.path.join(out_dir, f"experiment_7_refine_{cond}.csv")


def dense_beta_levels(sigma: float) -> List[float]:
    """beta_levels と同じ範囲の対数等間隔 N_BETA_DENSE 点 (13 点を含む)."""
    return [float(x) for x in np.logspace(np.log10(sigma / 1000),
                                          np.log10(sigma * 100), N_BETA_DENSE)]


def refine_points(cond: str, out_dir: str = OUT_DIR) -> List[Tuple]:
    """補足計算の点 (系列名, beta, n_target, gamma, pp, pd, nc).

    NoCancel (refine) は本計算の 13 点での ERP 最小点の両隣の β の間を細かくとるので,
    本計算の NoCancel 13 点が揃っていることを前提とする.
    """
    _, _, sigma, _, _ = CONDITIONS[cond]
    pts = [("Base (dense)", b, 0, 1.0, False, False, False)
           for b in dense_beta_levels(sigma)]
    nc = [r for r in read_rows(csv_path(cond, out_dir)) if r["series"] == "NoCancel"]
    betas = beta_levels(sigma)
    if len(nc) == len(betas):
        best = min(nc, key=lambda r: float(r["ERP"]))
        k = int(np.argmin([abs(np.log(float(best["beta"]) / b)) for b in betas]))
        lo, hi = betas[max(k - 1, 0)], betas[min(k + 1, len(betas) - 1)]
        pts += [("NoCancel (refine)", float(b), 0, 1.0, False, False, True)
                for b in np.logspace(np.log10(lo), np.log10(hi), N_REFINE)]
    return pts


def run_refine(conds: List[str], out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    plan = {c: refine_points(c, out_dir) for c in conds}
    done_keys = {c: {row_key(r) for r in read_rows(refine_csv_path(c, out_dir))}
                 for c in conds}
    total = sum(len(v) for v in plan.values())
    done = sum(len([p for p in plan[c] if point_key(p) in done_keys[c]]) for c in conds)
    print(f"[refine] 総点数 {total} (計算済み {done}, 今回計算 {total - done})", flush=True)
    t0 = time.time()
    last_log = t0
    n_new = 0
    for cond in conds:
        todo = [p for p in plan[cond] if point_key(p) not in done_keys[cond]]
        for k, pt in enumerate(todo):
            t = time.time()
            if pt[0] == "Base (dense)":
                _, pi, mb = solve_base_model(cond, pt[1])
                row = base_row(cond, pt[1], time.time() - t, pi, mb, None,
                               series="Base (dense)")
            else:
                _, pi, m = solve_predictive(cond, *pt[1:])
                row = predictive_row(cond, pt, time.time() - t, pi, m)
            append_row(refine_csv_path(cond, out_dir), row)
            done += 1
            n_new += 1
            last = f"[refine] {cond},{pt[0]},beta={pt[1]:.4g}"
            if row["check"] != "ok":
                print(f"  検査違反: {last}: {row['check']}", flush=True)
            cond_end = k + 1 == len(todo)
            if done % 10 == 0 or time.time() - last_log >= 60 or cond_end:
                log_progress(out_dir, done, total, t0, (time.time() - t0) / n_new, last)
                last_log = time.time()
        print(f"[refine] 条件 {cond} 完了", flush=True)


# ============================================================
# 補足 2: 最良点が走査範囲の端にあった系列の範囲を広げる
# ============================================================

# 条件 -> (広げる量, 系列の一覧, 追加する値)
#   beta: 追加する β (n_target と γ は本計算の ERP 最小点の値とその前後の格子点)
#   gamma: 追加する γ (β は ERP 最小点を中心とする 5 点, n_target は最小点の値と前後)
#   n_target: 追加する n_target (β と γ は ERP 最小点の値とその前後の格子点)
EXTENSIONS: Dict[str, Tuple[str, List[str], List[float]]] = {
    "C4": ("beta", ["P2 only", "P1+P2", "P1+P2 (protect)"],
           [float(x) for x in np.logspace(-8, -6, 7)[:-1]]),
    "C2": ("gamma", ["P2 only", "P1+P2", "P1+P2 (protect)"], [3000.0, 10000.0, 30000.0]),
    "C5": ("n_target", ["P1 only", "P1 only (protect)", "P1+P2", "P1+P2 (protect)"],
           [1, 2, 3, 4]),
}
# β を下に広げる条件では, β を細かくした Base も同じ間隔で同じ下限まで広げる
DENSE_EXTENSIONS: Dict[str, List[float]] = {
    "C4": [float(x) for x in np.logspace(-8, -6, 49)[:-1]],
}


def _index_of(levels: List[float], v: float) -> int:
    i = int(np.argmin([abs(np.log(v / x)) if x > 0 and v > 0 else abs(v - x)
                       for x in levels]))
    return i


def _neighbors(levels: List[float], v: float) -> List[float]:
    """格子 levels 上の v とその両隣 (端ならある側だけ)."""
    i = _index_of(levels, v)
    return list(levels[max(i - 1, 0): i + 2])


def _window5(levels: List[float], v: float) -> List[float]:
    """v を中心とする 5 点 (端にかかるときは 5 点を保つよう窓をずらす)."""
    i = _index_of(levels, v)
    lo = max(0, min(i - 2, len(levels) - 5))
    return list(levels[lo: lo + 5])


def grid_best(cond: str, series: str, out_dir: str = OUT_DIR) -> dict:
    """本計算の格子 (condition_points) の点だけで見た系列の ERP 最小点."""
    keys = {point_key(p) for p in condition_points(cond)}
    rows = [r for r in read_rows(csv_path(cond, out_dir))
            if r["series"] == series and row_key(r) in keys]
    return min(rows, key=lambda r: float(r["ERP"]))


def _series_flags(name: str) -> Tuple[bool, bool]:
    protect = name.endswith("(protect)")
    return protect, protect


def extension_points(cond: str, out_dir: str = OUT_DIR) -> List[Tuple]:
    """補足 2 の点 (系列名, beta, n_target, gamma, pp, pd, nc).

    基準にする ERP 最小点は本計算の格子の点だけから選ぶので, 追記後に再実行しても
    同じ点の一覧になる.
    """
    if cond not in EXTENSIONS:
        return []
    kind, series, values = EXTENSIONS[cond]
    betas = beta_levels(CONDITIONS[cond][2])
    pts = []
    for name in series:
        b = grid_best(cond, name, out_dir)
        beta0, nt0, g0 = float(b["beta"]), int(b["n_target"]), float(b["gamma"])
        pp, pd = _series_flags(name)
        has_p1 = name.startswith("P1")
        has_p2 = name.startswith("P2") or name.startswith("P1+P2")
        nts = _neighbors(N_TARGETS, nt0) if has_p1 else [0]
        gs = _neighbors(GAMMAS, g0) if has_p2 else [1.0]
        if kind == "beta":
            grid = [(beta, nt, g) for beta in values for nt in nts for g in gs]
        elif kind == "gamma":
            grid = [(beta, nt, g) for beta in _window5(betas, beta0) for nt in nts
                    for g in values]
        else:  # n_target
            grid = [(beta, nt, g) for beta in _neighbors(betas, beta0) for g in gs
                    for nt in values]
        pts += [(name, beta, int(nt), float(g), pp, pd, False) for beta, nt, g in grid]
    return pts


def dense_extension_points(cond: str) -> List[Tuple]:
    return [("Base (dense)", b, 0, 1.0, False, False, False)
            for b in DENSE_EXTENSIONS.get(cond, [])]


def run_extension(conds: List[str], out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    conds = [c for c in conds if c in EXTENSIONS or c in DENSE_EXTENSIONS]
    plan = {c: [(p, "main") for p in extension_points(c, out_dir)] +
            [(p, "refine") for p in dense_extension_points(c)] for c in conds}
    done_keys = {c: {row_key(r) for r in read_rows(csv_path(c, out_dir))} |
                 {row_key(r) for r in read_rows(refine_csv_path(c, out_dir))}
                 for c in conds}
    total = sum(len(v) for v in plan.values())
    done = sum(len([p for p, _ in plan[c] if point_key(p) in done_keys[c]]) for c in conds)
    print(f"[extend] 総点数 {total} (計算済み {done}, 今回計算 {total - done})", flush=True)
    t0 = time.time()
    last_log = t0
    n_new = 0
    for cond in conds:
        todo = [(p, dest) for p, dest in plan[cond] if point_key(p) not in done_keys[cond]]
        for k, (pt, dest) in enumerate(todo):
            t = time.time()
            if pt[0] == "Base (dense)":
                _, pi, mb = solve_base_model(cond, pt[1])
                row = base_row(cond, pt[1], time.time() - t, pi, mb, None,
                               series="Base (dense)")
            else:
                _, pi, m = solve_predictive(cond, *pt[1:])
                row = predictive_row(cond, pt, time.time() - t, pi, m)
            path = csv_path(cond, out_dir) if dest == "main" else refine_csv_path(cond, out_dir)
            append_row(path, row)
            done += 1
            n_new += 1
            last = (f"[extend] {cond},{pt[0]},beta={pt[1]:.4g},n_target={pt[2]},"
                    f"gamma={pt[3]:g}")
            if row["check"] != "ok":
                print(f"  検査違反: {last}: {row['check']}", flush=True)
            cond_end = k + 1 == len(todo)
            if done % 10 == 0 or time.time() - last_log >= 60 or cond_end:
                log_progress(out_dir, done, total, t0, (time.time() - t0) / n_new, last)
                last_log = time.time()
        print(f"[extend] 条件 {cond} 完了", flush=True)


# ============================================================
# フロンティアと集計
# ============================================================

def pareto_front(points: List[Tuple[float, float, int]]) -> List[Tuple[float, float, int]]:
    """(x, y, id) の点集合のうち, x と y をともに最小化する意味で支配されない点.

    x の昇順に並べて返す (y は降順になる). 同じ (x, y) の点は 1 つだけ残す.
    """
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    front = []
    best_y = float("inf")
    for x, y, i in pts:
        if y < best_y:
            front.append((x, y, i))
            best_y = y
    return front


def interp_cost(front: List[Tuple[float, float, int]], x: float) -> Optional[float]:
    """フロンティアを x について線形補間した y.

    x がフロンティアの x の最小値より小さければ到達できないので None.
    最大値より大きければ, 最大の x の点の y (それより小さい x で達成できる) を返す.
    """
    xs = [p[0] for p in front]
    ys = [p[1] for p in front]
    if x < xs[0]:
        return None
    if x >= xs[-1]:
        return ys[-1]
    return float(np.interp(x, xs, ys))


def _dominates(a: Tuple[float, float], b: Tuple[float, float], rtol=1e-9) -> bool:
    """a が b を (x, y の最小化の意味で) 支配するか. 差が rtol 以下は同じとみなす."""
    le_x = a[0] <= b[0] * (1 + rtol)
    le_y = a[1] <= b[1] * (1 + rtol)
    lt = a[0] < b[0] * (1 - rtol) or a[1] < b[1] * (1 - rtol)
    return le_x and le_y and lt


def dominance(front_s, front_ref) -> Tuple[str, int, int]:
    """系列のフロンティアが参照フロンティアを支配するか.

    参照フロンティアの各点について, 系列フロンティアのいずれかの (実在の) 点が
    それを支配するかを調べ, 支配される点の数で 全域 / 一部 / しない を返す.
    """
    n = len(front_ref)
    k = sum(1 for r in front_ref if any(_dominates(s[:2], r[:2]) for s in front_s))
    label = "全域" if k == n else ("一部" if k > 0 else "しない")
    return label, k, n


# ============================================================
# 図とレポート
# ============================================================

SERIES_STYLE = {
    "Base": dict(color="black", marker="o", linestyle="-"),
    "NoCancel": dict(color="tab:gray", marker="x", linestyle="--"),
    "P2 only": dict(color="tab:blue", marker="s", linestyle="-"),
    "P1 only": dict(color="tab:orange", marker="^", linestyle="-"),
    "P1 only (protect)": dict(color="tab:orange", marker="v", linestyle="--"),
    "P1+P2": dict(color="tab:red", marker="D", linestyle="-"),
    "P1+P2 (protect)": dict(color="tab:purple", marker="P", linestyle="--"),
}


def _to_float(v) -> float:
    return float(v) if str(v).strip() != "" else float("nan")


def _convert(rows: List[dict]) -> List[dict]:
    for r in rows:
        for k in METRIC_FIELDS + DIAG_FIELDS + ["beta", "gamma"]:
            r[k] = _to_float(r[k])
        r["n_target"] = int(r["n_target"])
    return rows


def load_condition(cond: str, out_dir: str = OUT_DIR) -> Tuple[List[dict], List[dict]]:
    """本計算の行 (Check を除く) と補足計算の行を返す."""
    rows = _convert(read_rows(csv_path(cond, out_dir)))
    refine = _convert(read_rows(refine_csv_path(cond, out_dir)))
    return [r for r in rows if r["series"] != "Check"], refine


def fronts_for(rows, xkey, ykey, series=None) -> Dict[str, List[Tuple[float, float, int]]]:
    out = {}
    for s in (series or SERIES_ORDER):
        pts = [(r[xkey], r[ykey], i) for i, r in enumerate(rows) if r["series"] == s]
        if pts:
            out[s] = pareto_front(pts)
    return out


def plot_condition(cond, rows, refine=(), fig_dir=FIG_DIR) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rho, delta, sigma, alpha, desc = CONDITIONS[cond]
    print(f"キャプション用情報 ({cond}): {desc}, c={BASELINE['c']}, K={BASELINE['K']}, "
          f"b={BASELINE['b']}, mu={BASELINE['mu']}, rho={rho}, delta={delta}, "
          f"sigma={sigma}, alpha={alpha}, rho_B={rho * (1 + delta):.2f}, "
          f"beta in [{sigma / 1000:g}, {sigma * 100:g}] (対数 {N_BETA} 点). "
          f"点線は Base を β {N_BETA_DENSE} 点で計算したフロンティア. "
          "E[W]-Cost の図の下段は, 各系列のフロンティアの点の Cost の, 点線 (β を細かくした "
          "Base のフロンティア) を線形補間した Cost に対する差 (%)")
    dense = [r for r in refine if r["series"] == "Base (dense)"]
    files = []
    for xkey, ykey, xlabel, ylabel, logy, stem in [
        ("E_W", "Cost", r"$E[W]$", "Cost", False, "EW_cost"),
        ("Cost", "P_block_arrival_stable", "Cost", r"$P_{\mathrm{block}}$", True,
         "cost_pblock"),
    ]:
        rel_panel = stem == "EW_cost"
        if rel_panel:
            fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 9.5), sharex=True,
                                          gridspec_kw={"height_ratios": [2, 1]})
        else:
            fig, ax = plt.subplots(figsize=(9, 6.5))
        fronts = fronts_for(rows, xkey, ykey)
        ref = pareto_front([(r[xkey], r[ykey], i) for i, r in enumerate(dense)]) \
            if dense else fronts["Base"]
        ys_all = []
        for s in SERIES_ORDER:
            if s not in fronts:
                continue
            st = SERIES_STYLE[s]
            xs = [r[xkey] for r in rows if r["series"] == s]
            ys = [r[ykey] for r in rows if r["series"] == s]
            ys_all += ys
            ax.scatter(xs, ys, s=8, color=st["color"], marker=st["marker"], alpha=0.25,
                       linewidths=0.6)
            fx = [p[0] for p in fronts[s]]
            fy = [p[1] for p in fronts[s]]
            ax.plot(fx, fy, color=st["color"], linestyle=st["linestyle"],
                    marker=st["marker"], markersize=5, linewidth=1.8, label=s)
            if rel_panel:
                pairs = [(x, 100 * (y / interp_cost(ref, x) - 1)) for x, y in zip(fx, fy)
                         if interp_cost(ref, x) is not None and x <= ref[-1][0]]
                if pairs:
                    ax2.plot([p[0] for p in pairs], [p[1] for p in pairs],
                             color=st["color"], linestyle=st["linestyle"],
                             marker=st["marker"], markersize=4, linewidth=1.4)
        if dense:
            ax.plot([p[0] for p in ref], [p[1] for p in ref], color="black",
                    linestyle=":", linewidth=1.5, label=f"Base ($\\beta$ {N_BETA_DENSE} points)")
        ax.set_ylabel(ylabel, fontsize=16)
        if logy:
            ax.set_yscale("log")
            # ブロッキング確率なので上端は 1 を超えないが, データより大きく空けない
            ax.set_ylim(top=min(1.0, 3 * max(ys_all)))
        ax.grid(True, alpha=0.3)
        if rel_panel:
            ax2.axhline(0, color="black", linestyle=":", linewidth=1.2)
            ax2.set_ylabel(r"$\Delta$Cost [%]", fontsize=16)
            ax2.set_xlabel(xlabel, fontsize=16)
            ax2.grid(True, alpha=0.3)
            # 横軸は参照フロンティアの範囲に合わせる (散布の右側の遠い点は上段で見る)
            ax2.set_xlim(ref[0][0] - 0.02 * (ref[-1][0] - ref[0][0]),
                         ref[-1][0] + 0.02 * (ref[-1][0] - ref[0][0]))
        else:
            ax.set_xlabel(xlabel, fontsize=16)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                   ncol=4, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.9 if rel_panel else 0.88))
        os.makedirs(fig_dir, exist_ok=True)
        png = os.path.join(fig_dir, f"experiment_7_{cond}_{stem}.png")
        pdf = os.path.splitext(png)[0] + ".pdf"
        fig.savefig(png, dpi=120, bbox_inches="tight")
        fig.savefig(pdf, dpi=120, bbox_inches="tight")
        plt.close(fig)
        files += [png, pdf]
    return files


def _fmt_params(r) -> str:
    if r["series"].startswith("Base"):
        return f"β={r['beta']:.4g}"
    return f"β={r['beta']:.4g}, n_target={r['n_target']}, γ={r['gamma']:g}"


def _cost_reduction_table(fronts, ref, names) -> Tuple[List[str], List[float], dict]:
    xmin, xmax = ref[0][0], ref[-1][0]
    levels = [xmin + f * (xmax - xmin) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
    lines = ["| 系列 | " + " | ".join(f"E[W]={x:.4f}" for x in levels) + " |",
             "|---|" + "---|" * len(levels)]
    red = {}
    for s in names:
        if s not in fronts:
            continue
        cells, red[s] = [], []
        for x in levels:
            cb, cs = interp_cost(ref, x), interp_cost(fronts[s], x)
            if cs is None:
                cells.append("—")
                red[s].append(None)
            else:
                red[s].append(1 - cs / cb)
                cells.append(f"{100 * (1 - cs / cb):+.2f}%")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    return lines, levels, red


def edge_note(r: dict, series_rows: List[dict]) -> str:
    """最良点のパラメータが, その系列で計算した範囲の端にあるかを書く."""
    notes = []
    for key, label in [("beta", "β"), ("gamma", "γ"), ("n_target", "n_target")]:
        vals = sorted({x[key] for x in series_rows})
        if key == "n_target":
            vals = [v for v in vals if v > 0]
        if len(vals) < 2 or r[key] not in vals:
            continue
        if r[key] == vals[0]:
            if key == "gamma" and r[key] == GAMMAS[0]:
                notes.append(f"{label} が下端 (より小さい γ=1 は P1 only の系列)")
            else:
                notes.append(f"{label} が下端")
        elif r[key] == vals[-1]:
            if key == "n_target" and r[key] == BASELINE["c"]:
                notes.append(f"{label} が上端 (=c, 取りうる最大)")
            else:
                notes.append(f"{label} が上端")
    return ", ".join(notes)


def report_condition(cond, rows, refine=()) -> Tuple[List[str], dict]:
    """1 条件の (a)〜(e) の表と補足の表 (Markdown の行) と, 要約用の数値を返す."""
    rho, delta, sigma, alpha, desc = CONDITIONS[cond]
    lines = [f"## {cond} {desc}", "",
             f"ρ={rho}, δ={delta}, σ={sigma}, α={alpha}, ρ_B={rho * (1 + delta):.2f}, "
             f"α/σ={alpha / sigma:g}, β ∈ [{sigma / 1000:g}, {sigma * 100:g}] "
             f"(対数 {N_BETA} 点). 点数 {len(rows)} (Check 行を除く), "
             f"check 違反 {sum(1 for r in rows if r['check'] != 'ok')} 件.", ""]
    betas = beta_levels(sigma)
    by = {s: [r for r in rows if r["series"] == s] for s in SERIES_ORDER}
    best = {s: min(v, key=lambda r: r["ERP"]) for s, v in by.items() if v}
    summary = {"best": best}

    # (a)
    lines += ["### (a) 各系列の ERP 最小点", "",
              "| 系列 | パラメータ | E[W] | Cost | P_block_arrival_stable | ERP | 備考 |",
              "|---|---|---|---|---|---|---|"]
    for s, r in best.items():
        edge = edge_note(r, by[s])
        lines.append(f"| {s} | {_fmt_params(r)} | {r['E_W']:.4f} | {r['Cost']:.4f} | "
                     f"{r['P_block_arrival_stable']:.4e} | {r['ERP']:.4f} | {edge} |")

    # (b)
    erp_base = best["Base"]["ERP"]
    lines += ["", "### (b) Base の ERP 最小値に対する各系列の ERP 最小値の相対差", "",
              "| 系列 | ERP 最小値 | 相対差 |", "|---|---|---|"]
    rel = {}
    for s, r in best.items():
        rel[s] = r["ERP"] / erp_base - 1
        lines.append(f"| {s} | {r['ERP']:.4f} | {100 * rel[s]:+.2f}% |")
    summary["erp_rel"] = rel

    # (c)
    fronts = fronts_for(rows, "E_W", "Cost")
    fb = fronts["Base"]
    tbl, _, red = _cost_reduction_table(fronts, fb, [s for s in SERIES_ORDER if s != "Base"])
    lines += ["", "### (c) 同じ E[W] での Cost の削減率 (Base のフロンティアに対して)", "",
              f"Base のフロンティアの E[W] の範囲 [{fb[0][0]:.4f}, {fb[-1][0]:.4f}] の "
              "10, 30, 50, 70, 90% の位置で, 各系列のフロンティアを線形補間した Cost を比べる "
              "(正の値が削減. — はその E[W] に系列のフロンティアが届かないこと. "
              "系列のフロンティアの最大 E[W] より右では, その点の Cost を用いる).", ""] + tbl
    summary["cost_red"] = red

    # (d)
    lines += ["", "### (d) フロンティアの支配関係", "",
              "参照フロンティアの各点を, 系列のフロンティアの実在の点が支配するか "
              "(E[W] と Cost がともに以下で少なくとも一方が小さい; 相対差 1e-9 以下は同じとみなす). "
              "k/n は支配される参照点の数.", "",
              "| 系列 | 対 Base | 対 NoCancel |", "|---|---|---|"]
    dom = {}
    fmt = lambda d: d[0] if d[0] == "—" else f"{d[0]} ({d[1]}/{d[2]})"
    for s in SERIES_ORDER:
        if s not in fronts:
            continue
        d_base = dominance(fronts[s], fb) if s != "Base" else ("—", 0, 0)
        d_nc = dominance(fronts[s], fronts["NoCancel"]) if s != "NoCancel" else ("—", 0, 0)
        dom[s] = (d_base, d_nc)
        lines.append(f"| {s} | {fmt(d_base)} | {fmt(d_nc)} |")
    lines += ["", "NoCancel と Base の比較: NoCancel のフロンティアが Base のフロンティアを支配するか "
              f"= {fmt(dom['NoCancel'][0])}, Base のフロンティアが NoCancel のフロンティアを支配するか "
              f"= {fmt(dom['Base'][1])}."]
    summary["dom"] = dom

    # (e)
    lines += ["", "### (e) フロンティア上の点での p1_fire_rate と setup_cancel_rate の範囲", "",
              "| 系列 | フロンティアの点数 | p1_fire_rate | setup_cancel_rate |",
              "|---|---|---|---|"]
    for s in SERIES_ORDER:
        if s not in fronts:
            continue
        rs = [rows[p[2]] for p in fronts[s]]
        f = [r["p1_fire_rate"] for r in rs]
        cr = [r["setup_cancel_rate"] for r in rs]
        lines.append(f"| {s} | {len(rs)} | [{min(f):.3e}, {max(f):.3e}] | "
                     f"[{min(cr):.3e}, {max(cr):.3e}] |")

    # 補足: β を細かくした Base と NoCancel
    dense = [r for r in refine if r["series"] == "Base (dense)"]
    nc_ref = [r for r in refine if r["series"] == "NoCancel (refine)"]
    if dense:
        # NoCancel は本計算の 13 点と ERP 最小点の近くの 21 点を合わせる
        fd = pareto_front([(r["E_W"], r["Cost"], i) for i, r in enumerate(dense)])
        f_all = dict(fronts)
        nc_rows = [r for r in rows if r["series"] == "NoCancel"] + nc_ref
        f_all["NoCancel"] = pareto_front([(r["E_W"], r["Cost"], i)
                                          for i, r in enumerate(nc_rows)])
        best_d = min(dense, key=lambda r: r["ERP"])
        best_nc = min(nc_rows, key=lambda r: r["ERP"])
        erp_d = best_d["ERP"]
        lines += ["", f"### 補足: β を細かくした Base ({N_BETA_DENSE} 点) と NoCancel "
                  f"(ERP 最小点の両隣の間を {N_REFINE} 点)", "",
                  "Base の β 13 点は, γ や n_target の組み合わせで点の多い Predictive の系列より "
                  "粗い. 差が格子の粗さによるものでないかを確かめるため, Base を同じ範囲の "
                  f"β {N_BETA_DENSE} 点で, NoCancel を 13 点での ERP 最小点の両隣の β の間を "
                  f"{N_REFINE} 点で計算した (experiment_7_refine_{cond}.csv). "
                  "Predictive の系列は 13 点のままなので, この比較は Predictive に不利な側に寄る.", "",
                  "| 系列 | ERP 最小値 | パラメータ | 対 Base (13 点) | 対 Base (β を細かく) |",
                  "|---|---|---|---|---|",
                  f"| Base (β を細かく) | {erp_d:.4f} | β={best_d['beta']:.4g} | "
                  f"{100 * (erp_d / erp_base - 1):+.2f}% | +0.00% |",
                  f"| NoCancel (β を細かく) | {best_nc['ERP']:.4f} | β={best_nc['beta']:.4g} | "
                  f"{100 * (best_nc['ERP'] / erp_base - 1):+.2f}% | "
                  f"{100 * (best_nc['ERP'] / erp_d - 1):+.2f}% |"]
        rel_d = {"NoCancel": best_nc["ERP"] / erp_d - 1}
        for s, r in best.items():
            if s in ("Base", "NoCancel"):
                continue
            rel_d[s] = r["ERP"] / erp_d - 1
            lines.append(f"| {s} | {r['ERP']:.4f} | {_fmt_params(r)} | "
                         f"{100 * rel[s]:+.2f}% | {100 * rel_d[s]:+.2f}% |")
        tbl, _, red_d = _cost_reduction_table(
            f_all, fd, [s for s in SERIES_ORDER if s != "Base"])
        lines += ["", f"同じ E[W] での Cost の削減率 (β を細かくした Base のフロンティア "
                  f"[{fd[0][0]:.4f}, {fd[-1][0]:.4f}] に対して, (c) と同じ方法):", ""] + tbl
        lines += ["", "β を細かくした Base のフロンティアに対する支配関係 ((d) と同じ判定):", "",
                  "| 系列 | 対 Base (β を細かく) |", "|---|---|"]
        dom_d = {}
        for s in SERIES_ORDER:
            if s == "Base" or s not in f_all:
                continue
            dom_d[s] = dominance(f_all[s], fd)
            lines.append(f"| {s} | {fmt(dom_d[s])} |")
        d_rev = dominance(fd, f_all["NoCancel"])
        lines += ["", f"β を細かくした Base のフロンティアが NoCancel のフロンティアを支配するか: "
                  f"{fmt(d_rev)}."]
        summary.update({"erp_rel_dense": rel_d, "cost_red_dense": red_d,
                        "dom_dense": dom_d, "dense_over_nc": d_rev,
                        "best_dense": best_d, "best_nc_all": best_nc})
    lines.append("")
    return lines, summary


SUMMARY_SERIES = ["NoCancel", "P2 only", "P1 only", "P1 only (protect)", "P1+P2",
                  "P1+P2 (protect)"]


def cross_condition_summary(summaries: dict) -> List[str]:
    """5 条件を通した要約の表."""
    fmt = lambda d: f"{d[0]} ({d[1]}/{d[2]})"
    lines = ["## 5 条件を通した要約", "",
             "### ERP 最小値の Base に対する相対差 (上: Base 13 点, 下: β を細かくした Base)", "",
             "| 条件 | " + " | ".join(SUMMARY_SERIES) + " |",
             "|---|" + "---|" * len(SUMMARY_SERIES)]
    for c, sm in summaries.items():
        lines.append(f"| {c} (13 点) | " + " | ".join(
            f"{100 * sm['erp_rel'][s]:+.2f}%" for s in SUMMARY_SERIES) + " |")
        if "erp_rel_dense" in sm:
            lines.append(f"| {c} (細かく) | " + " | ".join(
                f"{100 * sm['erp_rel_dense'][s]:+.2f}%" for s in SUMMARY_SERIES) + " |")
    lines += ["", "### Base のフロンティアに対する支配関係 (上: Base 13 点, 下: β を細かくした Base)", "",
              "| 条件 | " + " | ".join(SUMMARY_SERIES) + " |",
              "|---|" + "---|" * len(SUMMARY_SERIES)]
    for c, sm in summaries.items():
        lines.append(f"| {c} (13 点) | " + " | ".join(
            fmt(sm["dom"][s][0]) for s in SUMMARY_SERIES) + " |")
        if "dom_dense" in sm:
            lines.append(f"| {c} (細かく) | " + " | ".join(
                fmt(sm["dom_dense"][s]) for s in SUMMARY_SERIES) + " |")
    lines += ["", "### 同じ E[W] での Cost の削減率の最大値 (5 水準のうち; β を細かくした Base に対して)", "",
              "| 条件 | " + " | ".join(SUMMARY_SERIES) + " |",
              "|---|" + "---|" * len(SUMMARY_SERIES)]
    for c, sm in summaries.items():
        if "cost_red_dense" not in sm:
            continue
        cells = []
        for s in SUMMARY_SERIES:
            v = [x for x in sm["cost_red_dense"].get(s, []) if x is not None]
            cells.append(f"{100 * max(v):+.2f}%" if v else "—")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def report(conds: List[str], out_dir: str = OUT_DIR, fig_dir: str = FIG_DIR):
    lines = ["# 実験 7: ベースモデルと Predictive の最良の設定どうしの比較", "",
             f"results/experiment_7/experiment_7_<条件>.csv (と補足の "
             f"experiment_7_refine_<条件>.csv) から生成. "
             f"固定: c={BASELINE['c']}, K={BASELINE['K']}, b={BASELINE['b']}, "
             f"mu={BASELINE['mu']}. 系列: Base (ベースモデル), NoCancel "
             "(n_target=0, γ=1, never_cancel_setup=True), P2 only (n_target=0, "
             f"γ∈{{{', '.join(f'{g:g}' for g in GAMMAS)}}}), P1 only (n_target∈"
             f"{{{', '.join(map(str, N_TARGETS))}}}, γ=1), P1+P2 (n_target × γ). "
             "(protect) は (protect_presetup, protect_delayoff)=(True, True). "
             "整合性の検査 (各 β で Predictive (n_target=0, γ=1, 保護なし) とベースモデルの "
             "全指標の一致) の結果は各 CSV の Check 行の consistency_max_relerr 列.", ""]
    figs = []
    summaries = {}
    checks = []
    for cond in conds:
        rows, refine = load_condition(cond, out_dir)
        if not rows:
            continue
        ch = [r for r in read_rows(csv_path(cond, out_dir)) if r["series"] == "Check"]
        checks.append((cond, len(ch), max(float(r["consistency_max_relerr"]) for r in ch)))
        ls, sm = report_condition(cond, rows, refine)
        lines += ls
        summaries[cond] = sm
        figs += plot_condition(cond, rows, refine, fig_dir)
    lines += ["## 整合性の検査", "",
              "| 条件 | 点数 | 最大の相対誤差 |", "|---|---|---|"] + \
        [f"| {c} | {n} | {m:.2e} |" for c, n, m in checks] + [""]
    lines += cross_condition_summary(summaries)
    concl = os.path.join(out_dir, "conclusion.md")
    if os.path.exists(concl):
        with open(concl, encoding="utf-8") as f:
            lines += [f.read().rstrip(), ""]
    lines += ["## 図", ""] + [f"- {f}" for f in figs] + [""]
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return text, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--condition", choices=list(CONDITIONS) + ["all"], default="all")
    parser.add_argument("--report", action="store_true", help="図とレポートを作る")
    parser.add_argument("--count", action="store_true", help="点数だけ表示する")
    parser.add_argument("--extend", action="store_true",
                        help="補足 2: 最良点が走査範囲の端にあった系列の範囲を広げる")
    parser.add_argument("--refine", action="store_true",
                        help="補足: β を細かくした Base と NoCancel を計算する")
    args = parser.parse_args()
    conds = list(CONDITIONS) if args.condition == "all" else [args.condition]
    if args.count:
        for c in conds:
            print(c, len(condition_points(c)))
        print("合計", total_points(conds))
        return
    if args.extend:
        run_extension(conds)
        return
    if args.refine:
        run_refine(conds)
        return
    if args.report:
        text, _ = report(conds)
        print(text)
        return
    run(conds)


if __name__ == "__main__":
    main()
