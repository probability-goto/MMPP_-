"""GTH と splu の計算時間・帯格納メモリを設定ごとに計測する.

出力: results_gth/benchmark.md (Markdown 表)

使用例:
    python scripts/benchmark_gth_vs_splu.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _mmpp_burst import build_mmpp
from mmpp import ModelParameters, build_generator
from mmpp.solver import solve_stationary
from mmpp.gth_solver import bandwidths

# (ラベル, c, K, b, delta, sigma, rho)
CONFIGS = [
    ("小規模 (c=5,K=60,b=2)", 5, 60, 2, 0.6, 0.1, 0.7),
    ("標準 (c=20,K=200,b=5)", 20, 200, 5, 0.6, 0.1, 0.7),
    ("K=500 (c=20,b=5)", 20, 500, 5, 0.6, 0.1, 0.7),
    ("K=1000 (c=20,b=5)", 20, 1000, 5, 0.6, 0.1, 0.7),
]

N_REPEAT = 3


def bench_base():
    rows = []
    for label, c, K, b, delta, sigma, rho in CONFIGS:
        C0, C1 = build_mmpp(rho, delta, sigma, c, b, 1.0)
        params = ModelParameters(
            c=c, K=K, b=b, mu=1.0, alpha=0.1, beta=0.005, C0=C0, C1=C1
        )
        Q = build_generator(params)
        N = params.N
        p, q = bandwidths(Q)
        W = p + q + 1
        mem_mb = N * W * 8 / 1024 ** 2

        solve_stationary(Q, solver="gth")  # JIT ウォームアップ

        def timeit(solver):
            best = float("inf")
            for _ in range(N_REPEAT):
                t0 = time.perf_counter()
                solve_stationary(Q, solver=solver)
                best = min(best, time.perf_counter() - t0)
            return best

        t_gth = timeit("gth")
        t_splu = timeit("splu")

        rows.append(dict(
            label=label, N=N, p=p, q=q, W=W, mem_mb=mem_mb,
            t_gth=t_gth, t_splu=t_splu,
            cost=N * p * q,
        ))
        print(f"{label}: N={N} p={p} q={q} GTH={t_gth*1000:.1f}ms splu={t_splu*1000:.1f}ms mem={mem_mb:.1f}MB")
    return rows


def bench_predictive():
    """Predictive モデル (縮約後) の帯幅と計算時間."""
    try:
        from mmpp_predictive import PredictiveModelParameters
        from mmpp_predictive import build_generator as build_gen_p
        from mmpp_predictive.solver import solve_stationary as solve_p
        from scipy.sparse.csgraph import breadth_first_order
    except ImportError as e:
        print(f"Predictive モジュール読み込み失敗: {e}")
        return []

    rows = []
    for label, c, K, b, delta, sigma, rho in CONFIGS[:2]:
        C0, C1 = build_mmpp(rho, delta, sigma, c, b, 1.0)
        params = PredictiveModelParameters(
            c=c, K=K, b=b, mu=1.0, alpha=0.1, beta=0.005,
            C0=C0, C1=C1, n_target=min(10, c), gamma=5.0,
        )
        Q = build_gen_p(params)
        N_nominal = Q.shape[0]

        Qc = Q.tocsr()
        adj = Qc.copy()
        adj.setdiag(0)
        adj.eliminate_zeros()
        reach = np.sort(breadth_first_order(
            adj, i_start=0, directed=True, return_predecessors=False))
        Qr = Q[reach][:, reach].tocsr()
        N_red = Qr.shape[0]
        p, q = bandwidths(Qr)
        W = p + q + 1
        mem_mb = N_red * W * 8 / 1024 ** 2

        solve_p(Q, solver="gth")  # ウォームアップ

        t0 = time.perf_counter()
        solve_p(Q, solver="gth")
        t_gth = time.perf_counter() - t0

        t0 = time.perf_counter()
        solve_p(Q, solver="splu")
        t_splu = time.perf_counter() - t0

        rows.append(dict(
            label=f"Predictive {label}", N_nominal=N_nominal, N=N_red,
            p=p, q=q, W=W, mem_mb=mem_mb, t_gth=t_gth, t_splu=t_splu,
            cost=N_red * p * q,
        ))
        print(f"Predictive {label}: 名目N={N_nominal} 縮約N={N_red} p={p} q={q} "
              f"GTH={t_gth:.2f}s splu={t_splu:.2f}s mem={mem_mb:.1f}MB")
    return rows


def main():
    print("=== ベースモデル ===")
    base_rows = bench_base()
    print("\n=== Predictive モデル ===")
    pred_rows = bench_predictive()

    os.makedirs("results_gth", exist_ok=True)
    out = os.path.join("results_gth", "benchmark.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# GTH vs splu 計算時間・メモリ計測\n\n")
        f.write(f"計測環境: {os.cpu_count()} コア, numba JIT ウォームアップ後の最良値 "
                f"({N_REPEAT} 回中)\n\n")

        f.write("## ベースモデル (mmpp)\n\n")
        f.write("| 設定 | N | 下帯幅 p | 上帯幅 q | 帯幅 W | 帯格納メモリ | GTH | splu | GTH/splu |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in base_rows:
            f.write(f"| {r['label']} | {r['N']} | {r['p']} | {r['q']} | {r['W']} | "
                    f"{r['mem_mb']:.1f} MB | {r['t_gth']*1000:.1f} ms | "
                    f"{r['t_splu']*1000:.1f} ms | {r['t_gth']/r['t_splu']:.1f}x |\n")

        if pred_rows:
            f.write("\n## Predictive モデル (mmpp_predictive, 到達可能状態に縮約後)\n\n")
            f.write("| 設定 | 名目 N | 縮約 N | 下帯幅 p | 上帯幅 q | 帯格納メモリ | GTH | splu | GTH/splu |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for r in pred_rows:
                f.write(f"| {r['label']} | {r['N_nominal']} | {r['N']} | {r['p']} | {r['q']} | "
                        f"{r['mem_mb']:.1f} MB | {r['t_gth']:.2f} s | "
                        f"{r['t_splu']:.2f} s | {r['t_gth']/r['t_splu']:.0f}x |\n")
            f.write("\nPredictive モデルは状態 (i, s, j, F) の縮約状態空間がベースモデルの "
                    "j-major 順序と噛み合わず, 帯幅 p, q がベースモデルより 1 桁大きくなる。"
                    "GTH の計算量は O(N p q) のため計算時間もその分増大する "
                    "(mmpp_predictive.solver が N*p*q > 5e8 で警告を出す)。\n")

    print(f"\n{out} に書き出しました。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
