"""V6: 既存の全実験 (ベースモデル, 実験 0〜4) の全パラメータ点で
GTH と splu を比較する.

各スクリプトが定義する BASELINE / スイープ範囲 / build_params を直接
再利用し (CSV・図の I/O はスキップ), 全格子点で両ソルバーを実行して
Metrics.all_metrics() の相対差を集計する。

合格基準: P_block >= 1e-12 の点で相対差 > 1e-4 のものがあれば FAIL。
その他指標は最大相対差を表にするのみ (閾値なし)。

使用例:
    python scripts/validate_gth_v6_base.py
"""
import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from mmpp import build_generator, Metrics
from mmpp.solver import solve_stationary


def rel_diff(a, b):
    denom = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / denom


def compare_point(params, tag, rows, timing):
    Q = build_generator(params)

    t0 = time.perf_counter()
    pi_gth = solve_stationary(Q, solver="gth")
    t_gth = time.perf_counter() - t0

    t0 = time.perf_counter()
    pi_splu = solve_stationary(Q, solver="splu")
    t_splu = time.perf_counter() - t0

    m_gth = Metrics(params, pi_gth).all_metrics()
    m_splu = Metrics(params, pi_splu).all_metrics()

    row = {"tag": tag}
    for k in m_gth:
        row[f"diff__{k}"] = rel_diff(m_gth[k], m_splu[k])
        row[f"val__{k}"] = m_gth[k]
    rows.append(row)
    timing.append((tag, params.N, t_gth, t_splu))


def main():
    rows = []
    timing = []

    # ---------------- 実験 0 ----------------
    import experiment_0_validation as e0
    rho_values = np.linspace(*e0.RHO_RANGE, 10)
    for rho in rho_values:
        params = e0.build_params(float(rho))
        compare_point(params, f"exp0/rho={rho:.3f}", rows, timing)

    # ---------------- 実験 1 ----------------
    import experiment_1_traffic as e1
    rho_values = np.linspace(*e1.RHO_RANGE, 15)
    for name, delta, sigma in e1.BURST_LEVELS:
        for rho in rho_values:
            params = e1.build_params(float(rho), delta, sigma)
            compare_point(params, f"exp1/{name}/rho={rho:.3f}", rows, timing)

    # ---------------- 実験 2 ----------------
    import experiment_2_delayoff as e2
    beta_values = np.logspace(np.log10(e2.BETA_RANGE[0]), np.log10(e2.BETA_RANGE[1]), 15)
    for burst_name, delta, sigma in (e2.BURST_MAIN, e2.BURST_AUX):
        for level_name, alpha in e2.ALPHA_LEVELS:
            for beta in beta_values:
                params = e2.build_params(float(beta), alpha, delta, sigma)
                compare_point(params, f"exp2/{burst_name}/{level_name}/beta={beta:.3g}", rows, timing)

    # ---------------- 実験 3 ----------------
    import experiment_3_burstiness as e3
    delta_values = np.linspace(*e3.DELTA_RANGE, 15)
    sigma_values = np.logspace(np.log10(e3.SIGMA_RANGE[0]), np.log10(e3.SIGMA_RANGE[1]), 15)
    for level_name, alpha in e3.ALPHA_LEVELS:
        for delta in delta_values:
            params = e3.build_params("delta", float(delta), alpha)
            compare_point(params, f"exp3/delta/{level_name}/delta={delta:.3f}", rows, timing)
        for sigma in sigma_values:
            params = e3.build_params("sigma", float(sigma), alpha)
            compare_point(params, f"exp3/sigma/{level_name}/sigma={sigma:.3g}", rows, timing)

    # ---------------- 実験 4 ----------------
    import experiment_4_K_sensitivity as e4
    rho_values = np.linspace(*e4.RHO_RANGE, e4.RHO_N_POINTS)
    for burst_name, delta, sigma in e4.BURST_LEVELS:
        for K in e4.K_LEVELS:
            for rho in rho_values:
                params = e4.build_params(K, float(rho), delta, sigma)
                compare_point(params, f"exp4/{burst_name}/K={K}/rho={rho:.3f}", rows, timing)

    # ---------------- 集計 ----------------
    print(f"\n総比較点数: {len(rows)}")

    diff_keys = sorted(k for k in rows[0] if k.startswith("diff__"))
    max_diff = {k: 0.0 for k in diff_keys}
    for r in rows:
        for k in diff_keys:
            max_diff[k] = max(max_diff[k], r[k])

    print("\n=== 指標ごとの最大相対差 (GTH vs splu, 全点) ===")
    for k in diff_keys:
        name = k[len('diff__'):]
        val_key = f"val__{name}"
        # P_block 系は val がほぼ 0 (トラフィックが軽い点) だと 1-lambda_eff/lambda_bar
        # のような引き算で相対差が桁落ちにより見かけ上大きくなるため, val>=1e-12 の
        # 点に限定した最大値も併記する (V6 の合格基準と同じ足切り).
        if val_key in rows[0]:
            gated = [r[k] for r in rows if r[val_key] >= 1e-12]
            gated_max = max(gated) if gated else float("nan")
            print(f"  {name:<20} max_rel_diff(全点) = {max_diff[k]:.3e}   max_rel_diff(val>=1e-12) = {gated_max:.3e}")
        else:
            print(f"  {name:<20} max_rel_diff = {max_diff[k]:.3e}")

    # 参考: 指標側の桁落ちを除いた「純粋なソルバー差」
    # P_block_arrival_stable は引き算を含まないため, ここでの相対差は
    # π そのものの精度差 (= GTH と splu の差) だけを反映する.
    stable_key = "diff__P_block_arrival_stable"
    if stable_key in rows[0]:
        gated = [
            (r["tag"], r[stable_key], r["val__P_block_arrival_stable"])
            for r in rows if r["val__P_block_arrival_stable"] >= 1e-12
        ]
        if gated:
            worst = max(gated, key=lambda t: t[1])
            over = [g for g in gated if g[1] > 1e-4]
            print(f"\n=== 参考: 安定版指標での純粋なソルバー差 (val>=1e-12 の {len(gated)} 点) ===")
            print(f"  最大相対差: {worst[1]:.3e} @ {worst[0]} (値={worst[2]:.3e})")
            print(f"  1e-4 超過点数: {len(over)}")

    # 合格基準: P_block (or P_block_arrival) >= 1e-12 の点で,
    # その指標自体の相対差 (diff__<同じキー>) > 1e-4 なら違反.
    fail_rows = []
    for r in rows:
        for key in ("P_block", "P_block_arrival"):
            val_key, diff_key = f"val__{key}", f"diff__{key}"
            if val_key not in r:
                continue
            pb = r[val_key]
            if pb >= 1e-12 and r[diff_key] > 1e-4:
                fail_rows.append((r["tag"], key, pb, r[diff_key]))

    print(f"\n合格基準違反件数 (P_block>=1e-12 かつ相対差>1e-4): {len(fail_rows)}")
    for tag, key, pb, diff in fail_rows[:20]:
        print(f"  FAIL {tag}: {key}={pb:.3e} rel_diff={diff:.3e}")

    print("\n=== 計算時間 (GTH vs splu, 代表点) ===")
    total_gth = sum(t[2] for t in timing)
    total_splu = sum(t[3] for t in timing)
    print(f"  合計 GTH 時間:  {total_gth:.2f} s ({len(timing)} 点, 平均 {total_gth/len(timing)*1000:.2f} ms/点)")
    print(f"  合計 splu 時間: {total_splu:.2f} s ({len(timing)} 点, 平均 {total_splu/len(timing)*1000:.2f} ms/点)")

    ok = len(fail_rows) == 0
    print("\nV6 (base models) result:", "PASS" if ok else "FAIL")
    return rows, timing, ok


if __name__ == "__main__":
    main()
