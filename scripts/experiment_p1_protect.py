#!/usr/bin/env python
"""事前セットアップの保護 (protect_presetup) の有無による P1 診断指標の比較.

固定条件: c=20, K=200, b=5, mu=1, beta=0.005, n_target=10, gamma=5
条件 (ブロック): (medium, rho=0.7), (strong, rho=0.7), (medium, rho=0.3)
走査: alpha in {0.1, 1, 10} x protect_presetup in {False, True}

計算結果は 1 点ごとに results/p1_protect/p1_protect.csv に追記し, 再実行時は
計算済みの点 (パラメータの組がキー) を飛ばして再開する. 進捗は
results/p1_protect/progress.log に追記する.

使用例:
    python scripts/experiment_p1_protect.py                  # 全ブロック
    python scripts/experiment_p1_protect.py --block medium_rho0.7
    python scripts/experiment_p1_protect.py --summary        # CSV から比較表を出力
"""
import argparse
import csv
import os
import time
import warnings
from datetime import datetime

import numpy as np

from mmpp_predictive import (
    PredictiveModelParameters, build_generator, solve_stationary, Metrics,
)

try:
    from _mmpp_burst import build_mmpp
except ImportError:
    from scripts._mmpp_burst import build_mmpp


BASELINE = dict(c=20, K=200, b=5, mu=1.0, beta=0.005)
N_TARGET = 10
GAMMA = 5.0
ALPHA_LEVELS = [0.1, 1.0, 10.0]
PROTECT_LEVELS = [False, True]

# (ブロック名, バースト名, delta, sigma, rho)
BLOCKS = [
    ("medium_rho0.7", "medium", 0.6, 0.1, 0.7),
    ("strong_rho0.7", "strong", 0.9, 0.01, 0.7),
    ("medium_rho0.3", "medium", 0.6, 0.1, 0.3),
]

# 標準設定では縮約後の N*p*q が閾値を超えて毎点警告が出る (既知, 計算は正常)
warnings.filterwarnings("ignore", message="Predictive モデルの縮約後の帯幅が大きい")

OUT_DIR = os.path.join("results", "p1_protect")
CSV_PATH = os.path.join(OUT_DIR, "p1_protect.csv")
LOG_PATH = os.path.join(OUT_DIR, "progress.log")

KEY_FIELDS = ["burst_name", "rho", "alpha", "n_target", "gamma", "protect_presetup"]
FIELDNAMES = KEY_FIELDS + [
    "delta", "sigma", "beta", "c", "K", "b",
    "P_block_arrival_stable", "E_W", "Cost", "ERP",
    "E_N", "lambda_eff", "E_B", "E_S", "E_I", "E_off",
    "p1_fire_rate", "p1_launch_rate", "setup_completion_rate",
    "setup_cancel_rate", "rho_B",
    "min_pi", "elapsed_s", "check",
]


def point_key(burst_name, rho, alpha, protect):
    return (burst_name, float(rho), float(alpha), int(N_TARGET), float(GAMMA),
            bool(protect))


def row_key(row):
    return (row["burst_name"], float(row["rho"]), float(row["alpha"]),
            int(row["n_target"]), float(row["gamma"]),
            row["protect_presetup"] == "True")


def load_done():
    if not os.path.exists(CSV_PATH):
        return set()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return {row_key(r) for r in csv.DictReader(f)}


def append_row(row):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if new:
            w.writeheader()
        w.writerow(row)


def log_progress(done, total, t0, t_point_avg, last):
    elapsed = time.time() - t0
    remaining = (total - done) * t_point_avg
    line = (f"{datetime.now().isoformat(timespec='seconds')} "
            f"done={done}/{total} elapsed={elapsed:.1f}s "
            f"eta={remaining:.1f}s last={last}")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run_point(burst_name, delta, sigma, rho, alpha, protect):
    c, b, mu = BASELINE["c"], BASELINE["b"], BASELINE["mu"]
    C0, C1 = build_mmpp(rho, delta, sigma, c, b, mu)
    params = PredictiveModelParameters(
        C0=C0, C1=C1, alpha=alpha, n_target=N_TARGET, gamma=GAMMA,
        protect_presetup=protect, **BASELINE,
    )
    t = time.time()
    pi = solve_stationary(build_generator(params), solver="gth")
    m = Metrics(params, pi)
    elapsed = time.time() - t

    checks = []
    total = m.E_B + m.E_I + m.E_S + m.E_off
    if abs(total - c) > 1e-9:
        checks.append(f"conservation:{total - c:.3e}")
    flow = b * mu * m.E_B
    if abs(m.lambda_eff - flow) > 1e-9 * abs(m.lambda_eff):
        checks.append(f"flow:{(m.lambda_eff - flow) / m.lambda_eff:.3e}")
    if pi.min() < 0:
        checks.append(f"min_pi:{pi.min():.3e}")
    if not m.P_block_arrival_stable > 0:
        checks.append("P_block_nonpositive")
    if abs(m.rho_B - rho * (1 + delta)) > 1e-12:
        checks.append("rho_B")

    return {
        "burst_name": burst_name, "rho": rho, "alpha": alpha,
        "n_target": N_TARGET, "gamma": GAMMA, "protect_presetup": protect,
        "delta": delta, "sigma": sigma, "beta": BASELINE["beta"],
        "c": c, "K": BASELINE["K"], "b": b,
        "P_block_arrival_stable": m.arrival_blocking_probability_stable(),
        "E_W": m.mean_waiting_time(),
        "Cost": m.energy_cost_paper(),
        "ERP": m.erp_paper(),
        "E_N": m.mean_queue_length(),
        "lambda_eff": m.effective_arrival_rate(),
        "E_B": m.E_B, "E_S": m.E_S, "E_I": m.E_I, "E_off": m.E_off,
        "p1_fire_rate": m.p1_fire_rate,
        "p1_launch_rate": m.p1_launch_rate,
        "setup_completion_rate": m.setup_completion_rate,
        "setup_cancel_rate": m.setup_cancel_rate,
        "rho_B": m.rho_B,
        "min_pi": float(pi.min()),
        "elapsed_s": elapsed,
        "check": ";".join(checks) if checks else "ok",
    }


def run(blocks):
    os.makedirs(OUT_DIR, exist_ok=True)
    # 総点数・完了点数は全ブロックで数え, 計算は指定ブロックのみ行う
    points = [(blk, alpha, protect)
              for blk in BLOCKS for alpha in ALPHA_LEVELS for protect in PROTECT_LEVELS]
    done_keys = load_done()
    remaining = [pt for pt in points
                 if point_key(pt[0][1], pt[0][4], pt[1], pt[2]) not in done_keys]
    todo = [pt for pt in remaining if pt[0] in blocks]
    total = len(points)
    done = total - len(remaining)
    print(f"総点数 {total} (計算済み {done}, 今回計算 {len(todo)})", flush=True)

    t0 = time.time()
    last_log = t0
    for k, ((blk_name, burst_name, delta, sigma, rho), alpha, protect) in enumerate(todo):
        row = run_point(burst_name, delta, sigma, rho, alpha, protect)
        append_row(row)
        done += 1
        if row["check"] != "ok":
            print(f"  検査違反: {row['check']}", flush=True)
        block_end = k + 1 == len(todo) or todo[k + 1][0][0] != blk_name
        if done % 10 == 0 or time.time() - last_log >= 60 or block_end:
            avg = (time.time() - t0) / (k + 1)
            last = (f"{burst_name},rho={rho},alpha={alpha},"
                    f"protect={protect}")
            log_progress(done, total, t0, avg, last)
            last_log = time.time()
        if block_end:
            print(f"ブロック {blk_name} 完了", flush=True)


def summary():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"固定条件: c={BASELINE['c']}, K={BASELINE['K']}, b={BASELINE['b']}, "
          f"mu={BASELINE['mu']}, beta={BASELINE['beta']}, n_target={N_TARGET}, "
          f"gamma={GAMMA}")
    print("| 条件 | alpha | rho_B | p1_fire (F) | p1_fire (T) | cancel (F) | cancel (T) | "
          "cancel 比 T/F | P_block (F) | P_block (T) | ERP (F) | ERP (T) | check |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, burst_name, _, _, rho in BLOCKS:
        for alpha in ALPHA_LEVELS:
            sel = {r["protect_presetup"]: r for r in rows
                   if r["burst_name"] == burst_name and float(r["rho"]) == rho
                   and float(r["alpha"]) == alpha}
            if set(sel) != {"False", "True"}:
                continue
            f_, t_ = sel["False"], sel["True"]
            g = lambda r, k: float(r[k])
            ratio = g(t_, "setup_cancel_rate") / g(f_, "setup_cancel_rate")
            print(f"| {burst_name}, ρ={rho} | {alpha} | {g(f_, 'rho_B'):.2f} | "
                  f"{g(f_, 'p1_fire_rate'):.4e} | {g(t_, 'p1_fire_rate'):.4e} | "
                  f"{g(f_, 'setup_cancel_rate'):.4e} | {g(t_, 'setup_cancel_rate'):.4e} | "
                  f"{ratio:.3f} | "
                  f"{g(f_, 'P_block_arrival_stable'):.4e} | {g(t_, 'P_block_arrival_stable'):.4e} | "
                  f"{g(f_, 'ERP'):.4f} | {g(t_, 'ERP'):.4f} | "
                  f"{f_['check']}/{t_['check']} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--block", choices=[b[0] for b in BLOCKS] + ["all"],
                        default="all")
    parser.add_argument("--summary", action="store_true",
                        help="計算はせず CSV から比較表を出力する")
    args = parser.parse_args()
    if args.summary:
        summary()
        return
    blocks = BLOCKS if args.block == "all" else [b for b in BLOCKS if b[0] == args.block]
    run(blocks)


if __name__ == "__main__":
    main()
