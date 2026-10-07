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

使用例:
    python scripts/experiment_7_frontier.py                 # 全条件を計算
    python scripts/experiment_7_frontier.py --condition C1  # 1 条件だけ計算
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


def base_row(cond, beta, elapsed, pi, mb, check_row: dict) -> dict:
    row = _base_row(cond)
    d = mb.all_metrics()
    row.update({
        "model": "base", "series": "Base", "beta": beta, "n_target": 0,
        "gamma": 1.0, "protect_presetup": False, "protect_delayoff": False,
        "never_cancel_setup": False,
        "P_block": d["P_block"], "P_block_arrival_stable": d["P_block_arrival_stable"],
        "E_N": d["E[j]"], "E_B": d["E[B]"], "E_I": d["E[I]"], "E_S": d["E[S]"],
        "E_off": d["E[Off]"], "lambda_eff": d["lambda_eff"], "E_W": d["E[W]"],
        "rho_server": d["rho"], "Cost": d["cost_paper"], "ERP": d["ERP_paper"],
        "diag_source": "Check",
        "consistency_max_relerr": check_row["consistency_max_relerr"],
        "min_pi": float(pi.min()), "elapsed_s": elapsed,
    })
    for f in DIAG_FIELDS:
        row[f] = check_row[f]
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


def load_condition(cond: str, out_dir: str = OUT_DIR) -> List[dict]:
    rows = read_rows(csv_path(cond, out_dir))
    for r in rows:
        for k in METRIC_FIELDS + DIAG_FIELDS + ["beta", "gamma"]:
            r[k] = float(r[k])
        r["n_target"] = int(r["n_target"])
    return [r for r in rows if r["series"] != "Check"]


def fronts_for(rows, xkey, ykey) -> Dict[str, List[Tuple[float, float, int]]]:
    out = {}
    for s in SERIES_ORDER:
        pts = [(r[xkey], r[ykey], i) for i, r in enumerate(rows) if r["series"] == s]
        if pts:
            out[s] = pareto_front(pts)
    return out


def plot_condition(cond, rows, fig_dir=FIG_DIR) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rho, delta, sigma, alpha, desc = CONDITIONS[cond]
    print(f"キャプション用情報 ({cond}): {desc}, c={BASELINE['c']}, K={BASELINE['K']}, "
          f"b={BASELINE['b']}, mu={BASELINE['mu']}, rho={rho}, delta={delta}, "
          f"sigma={sigma}, alpha={alpha}, rho_B={rho * (1 + delta):.2f}, "
          f"beta in [{sigma / 1000:g}, {sigma * 100:g}] (対数 {N_BETA} 点)")
    files = []
    for xkey, ykey, xlabel, ylabel, logy, stem in [
        ("E_W", "Cost", r"$E[W]$", "Cost", False, "EW_cost"),
        ("Cost", "P_block_arrival_stable", "Cost", r"$P_{\mathrm{block}}$", True,
         "cost_pblock"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6.5))
        fronts = fronts_for(rows, xkey, ykey)
        for s in SERIES_ORDER:
            if s not in fronts:
                continue
            st = SERIES_STYLE[s]
            xs = [r[xkey] for r in rows if r["series"] == s]
            ys = [r[ykey] for r in rows if r["series"] == s]
            ax.scatter(xs, ys, s=8, color=st["color"], marker=st["marker"], alpha=0.25,
                       linewidths=0.6)
            fx = [p[0] for p in fronts[s]]
            fy = [p[1] for p in fronts[s]]
            ax.plot(fx, fy, color=st["color"], linestyle=st["linestyle"],
                    marker=st["marker"], markersize=5, linewidth=1.8, label=s)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(top=1)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                   ncol=4, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        os.makedirs(fig_dir, exist_ok=True)
        png = os.path.join(fig_dir, f"experiment_7_{cond}_{stem}.png")
        pdf = os.path.splitext(png)[0] + ".pdf"
        fig.savefig(png, dpi=120, bbox_inches="tight")
        fig.savefig(pdf, dpi=120, bbox_inches="tight")
        plt.close(fig)
        files += [png, pdf]
    return files


def _fmt_params(r) -> str:
    if r["series"] == "Base":
        return f"β={r['beta']:.4g}"
    return f"β={r['beta']:.4g}, n_target={r['n_target']}, γ={r['gamma']:g}"


def report_condition(cond, rows) -> Tuple[List[str], dict]:
    """1 条件の (a)〜(e) の表 (Markdown の行) と, 要約用の数値を返す."""
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
        edge = ""
        if r["beta"] in (betas[0], betas[-1]):
            edge = "β が走査範囲の端"
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
    xmin, xmax = fb[0][0], fb[-1][0]
    levels = [xmin + f * (xmax - xmin) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
    lines += ["", "### (c) 同じ E[W] での Cost の削減率 (Base のフロンティアに対して)", "",
              "Base のフロンティアの E[W] の範囲 "
              f"[{xmin:.4f}, {xmax:.4f}] の 10, 30, 50, 70, 90% の位置で, "
              "各系列のフロンティアを線形補間した Cost を比べる "
              "(正の値が削減. — はその E[W] に系列のフロンティアが届かないこと. "
              "系列のフロンティアの最大 E[W] より右では, その点の Cost を用いる).", "",
              "| 系列 | " + " | ".join(f"E[W]={x:.4f}" for x in levels) + " |",
              "|---|" + "---|" * len(levels)]
    red = {}
    for s in SERIES_ORDER:
        if s not in fronts or s == "Base":
            continue
        cells = []
        red[s] = []
        for x in levels:
            cb = interp_cost(fb, x)
            cs = interp_cost(fronts[s], x)
            if cs is None:
                cells.append("—")
                red[s].append(None)
            else:
                v = 1 - cs / cb
                red[s].append(v)
                cells.append(f"{100 * v:+.2f}%")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    summary["cost_red"] = red

    # (d)
    lines += ["", "### (d) フロンティアの支配関係", "",
              "参照フロンティアの各点を, 系列のフロンティアの実在の点が支配するか "
              "(E[W] と Cost がともに以下で少なくとも一方が小さい; 相対差 1e-9 以下は同じとみなす). "
              "k/n は支配される参照点の数.", "",
              "| 系列 | 対 Base | 対 NoCancel |", "|---|---|---|"]
    dom = {}
    for s in SERIES_ORDER:
        if s not in fronts:
            continue
        d_base = dominance(fronts[s], fb) if s != "Base" else ("—", 0, 0)
        d_nc = dominance(fronts[s], fronts["NoCancel"]) if s != "NoCancel" else ("—", 0, 0)
        dom[s] = (d_base, d_nc)
        fmt = lambda d: d[0] if d[0] == "—" else f"{d[0]} ({d[1]}/{d[2]})"
        lines.append(f"| {s} | {fmt(d_base)} | {fmt(d_nc)} |")
    # NoCancel を Base と比べる逆向き (Base が NoCancel を支配するか) も示す
    d_rev = dominance(fb, fronts["NoCancel"])
    lines += ["", f"参考: Base のフロンティアが NoCancel のフロンティアを支配するか: "
              f"{d_rev[0]} ({d_rev[1]}/{d_rev[2]})."]
    summary["dom"] = dom
    summary["dom_base_over_nc"] = d_rev

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
    lines.append("")
    return lines, summary


def report(conds: List[str], out_dir: str = OUT_DIR, fig_dir: str = FIG_DIR) -> str:
    lines = ["# 実験 7: ベースモデルと Predictive の最良の設定どうしの比較", "",
             f"results/experiment_7/experiment_7_<条件>.csv から生成. "
             f"固定: c={BASELINE['c']}, K={BASELINE['K']}, b={BASELINE['b']}, "
             f"mu={BASELINE['mu']}. 系列: Base (ベースモデル), NoCancel "
             "(n_target=0, γ=1, never_cancel_setup=True), P2 only (n_target=0, "
             f"γ∈{{{', '.join(f'{g:g}' for g in GAMMAS)}}}), P1 only (n_target∈"
             f"{{{', '.join(map(str, N_TARGETS))}}}, γ=1), P1+P2 (n_target × γ). "
             "(protect) は (protect_presetup, protect_delayoff)=(True, True).", ""]
    figs = []
    summaries = {}
    for cond in conds:
        rows = load_condition(cond, out_dir)
        if not rows:
            continue
        ls, sm = report_condition(cond, rows)
        lines += ls
        summaries[cond] = sm
        figs += plot_condition(cond, rows, fig_dir)
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
    args = parser.parse_args()
    conds = list(CONDITIONS) if args.condition == "all" else [args.condition]
    if args.count:
        for c in conds:
            print(c, len(condition_points(c)))
        print("合計", total_points(conds))
        return
    if args.report:
        text, _ = report(conds)
        print(text)
        return
    run(conds)


if __name__ == "__main__":
    main()
