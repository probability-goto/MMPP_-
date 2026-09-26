"""V2: mpmath (50 桁) による GTH の高精度検証.

タスク仕様 V2:
    c=5, K=60, b=2, alpha=0.5, beta=0.2, sigma0=sigma1=0.1,
    (lambda0, lambda1) = s * (0.6, 4.0), s in {0.05, 0.1, 0.2, 0.4, 1, 2}.
    mpmath (50 桁) で同じ GTH アルゴリズムを実行した真値に対し,
    P_block の相対誤差が GTH (float64) は 1e-13 以下であることを確認する.
    splu の誤差も並べて表にする.

使用例:
    python scripts/validate_gth_mpmath.py
"""
import sys
import os

import numpy as np
import mpmath as mp
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mmpp import ModelParameters, build_generator, Metrics
from mmpp.solver import solve_stationary
from mmpp.gth_solver import bandwidths, to_banded, check_reachability


S_VALUES = [0.05, 0.1, 0.2, 0.4, 1, 2]
BASELINE = dict(c=5, K=60, b=2, mu=1.0, alpha=0.5, beta=0.2)
SIGMA = 0.1


def build_params(s: float) -> ModelParameters:
    lambda0 = s * 0.6
    lambda1 = s * 4.0
    C1 = np.array([[lambda0, 0.0], [0.0, lambda1]])
    C0_off = np.array([[0.0, SIGMA], [SIGMA, 0.0]])
    row_sum = C0_off.sum(axis=1) + C1.sum(axis=1)
    C0 = C0_off.copy()
    np.fill_diagonal(C0, -row_sum)
    return ModelParameters(C0=C0, C1=C1, **BASELINE)


def gth_mpmath(Q: csr_matrix, dps: int = 50):
    """gth.py (参照実装) と同一のアルゴリズムを mpmath 任意精度で実行する."""
    mp.mp.dps = dps
    Qc = Q.tocoo()
    N = Q.shape[0]
    p, q = bandwidths(Q)
    W = p + q + 1

    check_reachability(Q)

    Bnd = [[mp.mpf(0) for _ in range(W)] for _ in range(N)]
    off = Qc.row != Qc.col
    for r, c, v in zip(Qc.row[off], Qc.col[off], Qc.data[off]):
        if v != 0.0:
            Bnd[r][c - r + p] = mp.mpf(float(v))

    alpha = [mp.mpf(0) for _ in range(N)]
    for k in range(N - 1):
        hi_c = min(k + q, N - 1)
        hi_r = min(k + p, N - 1)
        a = mp.mpf(0)
        for c in range(k + 1, hi_c + 1):
            a += Bnd[k][c - k + p]
        alpha[k] = a
        if a == 0:
            continue
        for r in range(k + 1, hi_r + 1):
            grk = Bnd[r][k - r + p]
            if grk == 0:
                continue
            factor = grk / a
            for c in range(k + 1, hi_c + 1):
                if c == r:
                    continue
                gkc = Bnd[k][c - k + p]
                if gkc == 0:
                    continue
                Bnd[r][c - r + p] += factor * gkc

    x = [mp.mpf(0) for _ in range(N)]
    x[N - 1] = mp.mpf(1)
    for k in range(N - 2, -1, -1):
        hi_r = min(k + p, N - 1)
        s = mp.mpf(0)
        for r in range(k + 1, hi_r + 1):
            s += x[r] * Bnd[r][k - r + p]
        x[k] = s / alpha[k]

    total = mp.fsum(x)
    pi = [xi / total for xi in x]
    return pi


def main():
    rows = []
    for s in S_VALUES:
        params = build_params(s)
        Q = build_generator(params)

        pi_gth = solve_stationary(Q, solver="gth")
        pi_splu = solve_stationary(Q, solver="splu")
        pi_true_mp = gth_mpmath(Q, dps=50)

        ss_block = params.D * params.K  # start index of level K block
        block_true = mp.fsum(pi_true_mp[ss_block: ss_block + params.D])
        block_gth = float(np.sum(pi_gth[ss_block: ss_block + params.D]))
        block_splu = float(np.sum(pi_splu[ss_block: ss_block + params.D]))

        block_true_f = float(block_true)
        rel_err_gth = abs(block_gth - block_true_f) / block_true_f if block_true_f != 0 else float("nan")
        rel_err_splu = abs(block_splu - block_true_f) / block_true_f if block_true_f != 0 else float("nan")

        rows.append(dict(
            s=s, N=params.N,
            P_block_true=block_true_f,
            P_block_gth=block_gth,
            P_block_splu=block_splu,
            rel_err_gth=rel_err_gth,
            rel_err_splu=rel_err_splu,
        ))

        print(
            f"s={s:5.2f}  N={params.N:5d}  "
            f"P_block(mpmath)={block_true_f:.6e}  "
            f"rel_err(GTH)={rel_err_gth:.3e}  "
            f"rel_err(splu)={rel_err_splu:.3e}"
        )

    print()
    max_gth = max(r["rel_err_gth"] for r in rows)
    max_splu = max(r["rel_err_splu"] for r in rows)
    print(f"max rel_err GTH  = {max_gth:.3e}  (threshold 1e-13)")
    print(f"max rel_err splu = {max_splu:.3e}")

    ok = max_gth <= 1e-13
    print("V2 result:", "PASS" if ok else "FAIL")
    return rows, ok


if __name__ == "__main__":
    main()
