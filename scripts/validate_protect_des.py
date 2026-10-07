#!/usr/bin/env python
"""保護ありの方策 (both) と位相を使わない対照 (NoCancel) の理論と DES の照合.

実験 0-P と同じ設定 (c=20, K=200, b=5, mu=1, alpha=0.1, beta=0.005,
中バースト delta=0.6, sigma=0.1, 20 レプリケーション, ウォームアップ 1e5,
計測 1e6 事象) で, rho を 5 点とり, 次の 2 方策について 6 指標
(P_block, E[N], E[W], lambda_eff, rho_server, Cost) の理論値が DES の
95% 信頼区間に入るかを調べる. 補助として P_block^arrival
(理論は P_block_arrival_stable) と setup_cancel_rate も照合する.

    both     : n_target=10, gamma=5, protect_presetup=protect_delayoff=True
    nocancel : n_target=0,  gamma=1, never_cancel_setup=True

計算結果は 1 点 ((方策, rho) の組) ごとに results/protect_des/protect_des.csv に
追記し, 再実行時は計算済みの点を飛ばして再開する. 進捗は
results/protect_des/progress.log に追記する.

使用例:
    python scripts/validate_protect_des.py
    python scripts/validate_protect_des.py --summary
"""
import argparse
import csv
import os
import time
import warnings
from datetime import datetime

import numpy as np
from scipy import stats as scipy_stats

from mmpp_predictive import (
    PredictiveModelParameters, build_generator, solve_stationary, Metrics,
)
from mmpp_predictive_sim import PredictiveSimulator, SimMetrics

try:
    from _mmpp_burst import build_mmpp, check_warmup_duration
except ImportError:
    from scripts._mmpp_burst import build_mmpp, check_warmup_duration


BASELINE = dict(c=20, K=200, b=5, mu=1.0, alpha=0.1, beta=0.005)
DELTA, SIGMA = 0.6, 0.1
RHO_LEVELS = [float(r) for r in np.linspace(0.3, 0.95, 5)]
N_REPS = 20
WARMUP_EVENTS = 100_000
MEASURE_EVENTS = 1_000_000
SEED0 = 42

# (名前, n_target, gamma, protect_presetup, protect_delayoff, never_cancel_setup)
POLICIES = [
    ("both", 10, 5.0, True, True, False),
    ("nocancel", 0, 1.0, False, False, True),
]

# (列名, Metrics のメソッド, SimMetrics の CI メソッド)
METRICS = [
    ("P_block", "blocking_probability", "blocking_probability_ci"),
    ("E_N", "mean_queue_length", "mean_queue_length_ci"),
    ("E_W", "mean_waiting_time", "mean_waiting_time_ci"),
    ("lambda_eff", "effective_arrival_rate", "effective_arrival_rate_ci"),
    ("rho_server", "utilization", "utilization_ci"),
    ("Cost", "energy_cost_paper", "energy_cost_paper_ci"),
]
# 補助の照合 (6 指標の判定には含めない)
EXTRA = [
    ("P_block_arrival", "arrival_blocking_probability_stable",
     "arrival_blocking_probability_ci"),
]

warnings.filterwarnings("ignore", message="Predictive モデルの縮約後の帯幅が大きい")

OUT_DIR = os.path.join("results", "protect_des")
CSV_PATH = os.path.join(OUT_DIR, "protect_des.csv")
LOG_PATH = os.path.join(OUT_DIR, "progress.log")
SUMMARY_PATH = os.path.join(OUT_DIR, "summary.md")

KEY_FIELDS = ["policy", "rho", "n_target", "gamma", "protect_presetup",
              "protect_delayoff", "never_cancel_setup", "n_reps",
              "warmup_events", "measure_events", "seed0"]
METRIC_COLUMNS = []
for name, _, _ in METRICS + EXTRA + [("setup_cancel_rate", None, None)]:
    METRIC_COLUMNS += [f"{name}_theory", f"{name}_sim", f"{name}_lo",
                       f"{name}_hi", f"{name}_in_ci"]
# P_block_arrival_stable は理論値 (P_block_arrival_theory と同じ値). results/ の
# CSV に共通の列名 (tests/test_stable_blocking.py が正値性を検査する).
FIELDNAMES = KEY_FIELDS + ["delta", "sigma", "c", "K", "b", "alpha", "beta",
                           "P_block_arrival_stable"] + \
    METRIC_COLUMNS + ["n_in_ci_6", "min_pi", "elapsed_s", "check"]


def point_key(policy, rho):
    return (policy, round(float(rho), 10), N_REPS, WARMUP_EVENTS, MEASURE_EVENTS, SEED0)


def row_key(r):
    return (r["policy"], round(float(r["rho"]), 10), int(r["n_reps"]),
            int(r["warmup_events"]), int(r["measure_events"]), int(r["seed0"]))


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


def t_ci(values):
    v = np.asarray(values, dtype=float)
    mean = float(v.mean())
    half = float(scipy_stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return mean, mean - half, mean + half


def run_point(policy, rho):
    name, n_target, gamma, pp, pd, nc = policy
    c, b, mu = BASELINE["c"], BASELINE["b"], BASELINE["mu"]
    C0, C1 = build_mmpp(rho, DELTA, SIGMA, c, b, mu)
    params = PredictiveModelParameters(
        C0=C0, C1=C1, n_target=n_target, gamma=gamma,
        protect_presetup=pp, protect_delayoff=pd, never_cancel_setup=nc,
        **BASELINE,
    )
    t = time.time()
    pi = solve_stationary(build_generator(params), solver="gth")
    m = Metrics(params, pi)

    reps, cancel_rates = [], []
    for k in range(N_REPS):
        sim = PredictiveSimulator(params, seed=SEED0 + k)
        st = sim.run(warmup_events=WARMUP_EVENTS, measurement_events=MEASURE_EVENTS)
        reps.append(st)
        cancel_rates.append(sim.setup_cancel_count / st.total_duration)
    check_warmup_duration(params, reps)
    sm = SimMetrics(params, reps)
    elapsed = time.time() - t

    row = {
        "policy": name, "rho": rho, "n_target": n_target, "gamma": gamma,
        "protect_presetup": pp, "protect_delayoff": pd, "never_cancel_setup": nc,
        "n_reps": N_REPS, "warmup_events": WARMUP_EVENTS,
        "measure_events": MEASURE_EVENTS, "seed0": SEED0,
        "delta": DELTA, "sigma": SIGMA, "c": c, "K": BASELINE["K"], "b": b,
        "alpha": BASELINE["alpha"], "beta": BASELINE["beta"],
    }
    n_in = 0
    for col, th_method, ci_method in METRICS + EXTRA:
        theory = getattr(m, th_method)()
        mean, lo, hi = getattr(sm, ci_method)()
        inside = bool(lo <= theory <= hi)
        row.update({f"{col}_theory": theory, f"{col}_sim": mean, f"{col}_lo": lo,
                    f"{col}_hi": hi, f"{col}_in_ci": inside})
        if (col, th_method, ci_method) in METRICS:
            n_in += inside
    mean, lo, hi = t_ci(cancel_rates)
    if nc:
        inside = m.setup_cancel_rate == 0.0 and max(cancel_rates) == 0.0
    else:
        inside = bool(lo <= m.setup_cancel_rate <= hi)
    row.update({"setup_cancel_rate_theory": m.setup_cancel_rate,
                "setup_cancel_rate_sim": mean, "setup_cancel_rate_lo": lo,
                "setup_cancel_rate_hi": hi, "setup_cancel_rate_in_ci": inside})
    row["n_in_ci_6"] = n_in
    row["P_block_arrival_stable"] = m.P_block_arrival_stable

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
    row["min_pi"] = float(pi.min())
    row["elapsed_s"] = elapsed
    row["check"] = ";".join(checks) if checks else "ok"
    return row


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    points = [(pol, rho) for pol in POLICIES for rho in RHO_LEVELS]
    done_keys = load_done()
    todo = [pt for pt in points if point_key(pt[0][0], pt[1]) not in done_keys]
    total = len(points)
    done = total - len(todo)
    print(f"総点数 {total} (計算済み {done}, 今回計算 {len(todo)})", flush=True)

    t0 = time.time()
    for k, (pol, rho) in enumerate(todo):
        row = run_point(pol, rho)
        append_row(row)
        done += 1
        if row["check"] != "ok":
            print(f"  検査違反: {row['check']}", flush=True)
        # 1 点が 1 分を超えるので毎点記録する
        avg = (time.time() - t0) / (k + 1)
        log_progress(done, total, t0, avg,
                     f"policy={pol[0]},rho={rho:.4f},in_ci={row['n_in_ci_6']}/6")
        if k + 1 == len(todo) or todo[k + 1][0][0] != pol[0]:
            print(f"ブロック {pol[0]} 完了", flush=True)


def summary():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lines = [
        "# 保護ありの方策と対照の DES 照合 (results/protect_des/protect_des.csv から生成)",
        "",
        f"固定条件: c={BASELINE['c']}, K={BASELINE['K']}, b={BASELINE['b']}, "
        f"mu={BASELINE['mu']}, alpha={BASELINE['alpha']}, beta={BASELINE['beta']}, "
        f"delta={DELTA}, sigma={SIGMA}, {N_REPS} レプリケーション, "
        f"ウォームアップ {WARMUP_EVENTS} 事象, 計測 {MEASURE_EVENTS} 事象. "
        "both: n_target=10, gamma=5, protect_presetup=protect_delayoff=True. "
        "nocancel: n_target=0, gamma=1, never_cancel_setup=True.",
        "",
        "| 方策 | ρ | 6 指標の CI 内 | CI 外の指標 | P_block^arr | cancel | check |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        out = [col for col, _, _ in METRICS if r[f"{col}_in_ci"] != "True"]
        lines.append(
            f"| {r['policy']} | {float(r['rho']):.4f} | {r['n_in_ci_6']}/6 | "
            f"{', '.join(out) if out else '-'} | "
            f"{'内' if r['P_block_arrival_in_ci'] == 'True' else '外'} | "
            f"{'内' if r['setup_cancel_rate_in_ci'] == 'True' else '外'} | {r['check']} |")
    lines += ["", "## CI 外の指標の詳細", "",
              "| 方策 | ρ | 指標 | 理論 | DES 平均 | CI 下限 | CI 上限 | 相対差 |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for col, _, _ in METRICS + EXTRA + [("setup_cancel_rate", None, None)]:
            if r[f"{col}_in_ci"] == "True":
                continue
            th, sm = float(r[f"{col}_theory"]), float(r[f"{col}_sim"])
            rel = (sm - th) / th if th != 0 else float("nan")
            lines.append(
                f"| {r['policy']} | {float(r['rho']):.4f} | {col} | {th:.6g} | {sm:.6g} | "
                f"{float(r[f'{col}_lo']):.6g} | {float(r[f'{col}_hi']):.6g} | {rel:+.2e} |")
    text = "\n".join(lines)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"保存: {SUMMARY_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", action="store_true",
                        help="計算はせず CSV から照合結果の表を出力する")
    args = parser.parse_args()
    if args.summary:
        summary()
    else:
        run()


if __name__ == "__main__":
    main()
