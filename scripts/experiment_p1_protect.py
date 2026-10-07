#!/usr/bin/env python
"""事前セットアップ・Delayoff の保護の有無による P1 診断指標と 4 指標の比較.

固定条件: c=20, K=200, b=5, mu=1, n_target=10
条件 (ブロック): (medium, rho=0.7), (strong, rho=0.7), (medium, rho=0.3)
走査: alpha in {0.1, 1, 10} x beta in {0.005, 0.5} x gamma in {1, 5}
      x 保護 3 通り (protect_presetup, protect_delayoff):
        none     = (False, False)  保護なし
        presetup = (True,  False)  取り消しのみ保護
        both     = (True,  True)   取り消しと Delayoff の両方を保護

計算結果は 1 点ごとに results/p1_protect/p1_protect.csv に追記し, 再実行時は
計算済みの点 (パラメータの組がキー) を飛ばして再開する. 進捗は
results/p1_protect/progress.log に追記する.

protect_delayoff 導入前の結果 (beta=0.005, gamma=5, none/presetup) は
results/p1_protect/p1_protect_v1.csv に残してあり, 同じパラメータの点は
計算時に照合して, 一致しなければ check 列に v1_mismatch を記録する.

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

from mmpp_predictive import (
    PredictiveModelParameters, build_generator, solve_stationary, Metrics,
)

try:
    from _mmpp_burst import build_mmpp
except ImportError:
    from scripts._mmpp_burst import build_mmpp


BASELINE = dict(c=20, K=200, b=5, mu=1.0)
N_TARGET = 10
ALPHA_LEVELS = [0.1, 1.0, 10.0]
BETA_LEVELS = [0.005, 0.5]
GAMMA_LEVELS = [1.0, 5.0]
# (名前, protect_presetup, protect_delayoff)
MODES = [
    ("none", False, False),
    ("presetup", True, False),
    ("both", True, True),
]

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
V1_CSV_PATH = os.path.join(OUT_DIR, "p1_protect_v1.csv")
LOG_PATH = os.path.join(OUT_DIR, "progress.log")
SUMMARY_PATH = os.path.join(OUT_DIR, "summary.md")

KEY_FIELDS = ["burst_name", "rho", "alpha", "beta", "gamma", "n_target",
              "protect_presetup", "protect_delayoff"]
METRIC_FIELDS = [
    "P_block_arrival_stable", "E_W", "Cost", "ERP",
    "E_N", "lambda_eff", "E_B", "E_S", "E_I", "E_off",
    "p1_fire_rate", "p1_launch_rate", "setup_completion_rate",
    "setup_cancel_rate", "delayoff_rate", "rho_B",
]
FIELDNAMES = KEY_FIELDS + ["mode", "delta", "sigma", "c", "K", "b"] + METRIC_FIELDS + [
    "min_pi", "elapsed_s", "check",
]
# v1 の CSV と照合する列 (v1 に delayoff_rate はない)
V1_COMPARE_FIELDS = [f for f in METRIC_FIELDS if f != "delayoff_rate"]


def point_key(burst_name, rho, alpha, beta, gamma, protect_presetup, protect_delayoff):
    return (burst_name, float(rho), float(alpha), float(beta), float(gamma),
            int(N_TARGET), bool(protect_presetup), bool(protect_delayoff))


def row_key(row):
    return (row["burst_name"], float(row["rho"]), float(row["alpha"]),
            float(row["beta"]), float(row["gamma"]), int(row["n_target"]),
            row["protect_presetup"] == "True", row["protect_delayoff"] == "True")


def load_done():
    if not os.path.exists(CSV_PATH):
        return set()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return {row_key(r) for r in csv.DictReader(f)}


def load_v1():
    """v1 の結果 (protect_delayoff 導入前) をキー -> 行 の辞書で返す."""
    if not os.path.exists(V1_CSV_PATH):
        return {}
    with open(V1_CSV_PATH, newline="", encoding="utf-8") as f:
        return {
            point_key(r["burst_name"], r["rho"], r["alpha"], r["beta"], r["gamma"],
                      r["protect_presetup"] == "True", False): r
            for r in csv.DictReader(f)
        }


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


def run_point(burst_name, delta, sigma, rho, alpha, beta, gamma, mode, v1_row=None):
    mode_name, protect_presetup, protect_delayoff = mode
    c, b, mu = BASELINE["c"], BASELINE["b"], BASELINE["mu"]
    C0, C1 = build_mmpp(rho, delta, sigma, c, b, mu)
    params = PredictiveModelParameters(
        C0=C0, C1=C1, alpha=alpha, beta=beta, n_target=N_TARGET, gamma=gamma,
        protect_presetup=protect_presetup, protect_delayoff=protect_delayoff,
        **BASELINE,
    )
    t = time.time()
    pi = solve_stationary(build_generator(params), solver="gth")
    m = Metrics(params, pi)
    elapsed = time.time() - t

    row = {
        "burst_name": burst_name, "rho": rho, "alpha": alpha, "beta": beta,
        "gamma": gamma, "n_target": N_TARGET,
        "protect_presetup": protect_presetup, "protect_delayoff": protect_delayoff,
        "mode": mode_name, "delta": delta, "sigma": sigma,
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
        "delayoff_rate": m.delayoff_rate,
        "rho_B": m.rho_B,
        "min_pi": float(pi.min()),
        "elapsed_s": elapsed,
    }

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
    if v1_row is not None:
        bad = [f for f in V1_COMPARE_FIELDS
               if abs(row[f] - float(v1_row[f])) > 1e-12 * abs(float(v1_row[f]))]
        if bad:
            checks.append("v1_mismatch:" + "/".join(bad))
    row["check"] = ";".join(checks) if checks else "ok"
    return row


def all_points():
    return [(blk, alpha, beta, gamma, mode)
            for blk in BLOCKS for alpha in ALPHA_LEVELS for beta in BETA_LEVELS
            for gamma in GAMMA_LEVELS for mode in MODES]


def key_of(pt):
    (_, burst_name, _, _, rho), alpha, beta, gamma, mode = pt
    return point_key(burst_name, rho, alpha, beta, gamma, mode[1], mode[2])


def run(blocks):
    os.makedirs(OUT_DIR, exist_ok=True)
    # 総点数・完了点数は全ブロックで数え, 計算は指定ブロックのみ行う
    points = all_points()
    done_keys = load_done()
    v1 = load_v1()
    remaining = [pt for pt in points if key_of(pt) not in done_keys]
    todo = [pt for pt in remaining if pt[0] in blocks]
    total = len(points)
    done = total - len(remaining)
    print(f"総点数 {total} (計算済み {done}, 今回計算 {len(todo)})", flush=True)

    t0 = time.time()
    last_log = t0
    for k, pt in enumerate(todo):
        (blk_name, burst_name, delta, sigma, rho), alpha, beta, gamma, mode = pt
        row = run_point(burst_name, delta, sigma, rho, alpha, beta, gamma, mode,
                        v1_row=v1.get(key_of(pt)))
        append_row(row)
        done += 1
        if row["check"] != "ok":
            print(f"  検査違反: {row['check']}", flush=True)
        block_end = k + 1 == len(todo) or todo[k + 1][0][0] != blk_name
        if done % 10 == 0 or time.time() - last_log >= 60 or block_end:
            avg = (time.time() - t0) / (k + 1)
            last = (f"{burst_name},rho={rho},alpha={alpha},beta={beta},"
                    f"gamma={gamma},mode={mode[0]}")
            log_progress(done, total, t0, avg, last)
            last_log = time.time()
        if block_end:
            print(f"ブロック {blk_name} 完了", flush=True)


def summary_table(rows, alpha):
    """alpha を固定した比較表 (Markdown) の行リストを返す."""
    lines = [
        f"### alpha = {alpha}",
        "",
        "| 条件 | β | γ | 保護 | p1_fire_rate | setup_cancel_rate | "
        "P_block_arrival_stable | E[W] | Cost | ERP | check |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, burst_name, _, _, rho in BLOCKS:
        for beta in BETA_LEVELS:
            for gamma in GAMMA_LEVELS:
                for mode_name, pp, pd in MODES:
                    key = point_key(burst_name, rho, alpha, beta, gamma, pp, pd)
                    r = rows.get(key)
                    if r is None:
                        continue
                    g = lambda k: float(r[k])
                    lines.append(
                        f"| {burst_name}, ρ={rho} | {beta} | {gamma:g} | {mode_name} | "
                        f"{g('p1_fire_rate'):.4e} | {g('setup_cancel_rate'):.4e} | "
                        f"{g('P_block_arrival_stable'):.4e} | {g('E_W'):.4f} | "
                        f"{g('Cost'):.4f} | {g('ERP'):.4f} | {r['check']} |")
    return lines


def summary():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = {row_key(r): r for r in csv.DictReader(f)}
    caption = (f"固定条件: c={BASELINE['c']}, K={BASELINE['K']}, b={BASELINE['b']}, "
               f"mu={BASELINE['mu']}, n_target={N_TARGET}. "
               "保護: none=(protect_presetup, protect_delayoff)=(False, False), "
               "presetup=(True, False), both=(True, True)")
    lines = ["# 保護の有無による比較 (results/p1_protect/p1_protect.csv から生成)",
             "", caption, ""]
    for alpha in ALPHA_LEVELS:
        lines += summary_table(rows, alpha) + [""]
    text = "\n".join(lines)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"保存: {SUMMARY_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--block", choices=[b[0] for b in BLOCKS] + ["all"],
                        default="all")
    parser.add_argument("--summary", action="store_true",
                        help="計算はせず CSV から比較表を出力する (summary.md にも保存)")
    args = parser.parse_args()
    if args.summary:
        summary()
        return
    blocks = BLOCKS if args.block == "all" else [b for b in BLOCKS if b[0] == args.block]
    run(blocks)


if __name__ == "__main__":
    main()
