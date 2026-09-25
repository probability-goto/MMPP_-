"""GTH 法 (O'Cinneide 1996, GTH1-3) による定常分布ソルバー.

splu (scipy.sparse.linalg.spsolve) は引き算を伴う通常の LU 分解であり,
絶対誤差が浮動小数点の丸め (~1e-17) で頭打ちになる. π の成分が
1e-15 を下回る領域では相対誤差が破綻し, 負の確率さえ出うる.

GTH (Grassmann-Taksar-Heyman) 法は生成行列 G の非対角成分 (非負) のみを
使い, 引き算を一切行わない消去法 (状態削減法) である. 各成分が相対精度で
求まるため, P_block のように著しく小さくなりうる量の計算に適する.

参照: O'Cinneide, C. A. (1993). Entrywise perturbation theory and error
analysis for Markov chains. (1996). Relative-error bounds for the LU
decomposition via the GTH algorithm.

このモジュールは検証用の素朴な帯格納実装 (帯全体を numpy 配列に展開し,
GTH1/GTH2 の消去掃引と GTH3 の後退代入を numba @njit で高速化する) を
提供する. 対角要素は一切参照しない (仕様どおり).

帯格納:
    Bnd[r, c - r + p] = g_rc   (非対角のみ. 対角 r==c は格納しない)
    p: 下帯幅 (r > c となる最大 r - c), q: 上帯幅 (c > r となる最大 c - r)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import breadth_first_order

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mmpp.gth_solver には numba が必要です. `pip install numba` を実行してください."
    ) from exc


__all__ = [
    "bandwidths",
    "check_reachability",
    "to_banded",
    "gth_sweep",
    "solve_stationary_gth",
    "build_LU",
]


# ============================================================
# 帯幅の測定
# ============================================================

def bandwidths(Q) -> Tuple[int, int]:
    """生成行列 Q の非対角成分から下帯幅 p, 上帯幅 q を測定する.

    p = max_{g_rc != 0, r>c} (r - c),  q = max_{g_rc != 0, c>r} (c - r).
    非対角成分が全く無い (N=1 等) 場合は (0, 0) を返す.
    """
    Qc = Q.tocoo()
    off = Qc.row != Qc.col
    row = Qc.row[off]
    col = Qc.col[off]
    data = Qc.data[off]
    nz = data != 0
    row = row[nz]
    col = col[nz]
    if row.size == 0:
        return 0, 0
    diff = row - col
    p = int(max(diff.max(), 0))
    q = int(max((-diff).max(), 0))
    return p, q


# ============================================================
# 到達可能性チェック (GTH の前提条件)
# ============================================================

def check_reachability(Q, target: int = None) -> None:
    """全状態が状態 target (既定: 最終状態 N-1) に到達可能かを検証する.

    GTH 法は「全状態が最終状態から到達可能」(既約なら成立) であることを
    前提とする. 満たさない場合は ValueError を送出する.
    """
    N = Q.shape[0]
    if target is None:
        target = N - 1
    Qc = Q.tocsr().copy()
    Qc.setdiag(0)
    Qc.eliminate_zeros()
    # k -> target への到達可能性は, 逆向きグラフ (= Q^T) 上で target から
    # BFS した到達集合に k が含まれるかと同値.
    rev = Qc.transpose().tocsr()
    reached = breadth_first_order(
        rev, i_start=target, directed=True, return_predecessors=False
    )
    if len(reached) != N:
        reached_set = set(reached.tolist())
        missing = [k for k in range(N) if k not in reached_set]
        n_show = missing[:10]
        raise ValueError(
            f"GTH の前提条件違反: {len(missing)} 個の状態が最終状態 "
            f"(idx={target}) に到達不能です (例: {n_show}). "
            "Q が既約でない可能性があります。"
        )


# ============================================================
# 帯格納への変換
# ============================================================

def to_banded(Q, p: int, q: int) -> np.ndarray:
    """疎行列 Q を帯格納 Bnd[r, c-r+p] = g_rc (非対角のみ) に変換する."""
    Qc = Q.tocoo()
    N = Q.shape[0]
    W = p + q + 1
    Bnd = np.zeros((N, W), dtype=np.float64)
    off = Qc.row != Qc.col
    row = Qc.row[off]
    col = Qc.col[off]
    data = Qc.data[off]
    Bnd[row, col - row + p] = data
    return Bnd


# ============================================================
# numba による GTH1+GTH2 (前進消去) / GTH3 (後退代入)
# ============================================================

@njit(cache=True)
def _gth_forward(Bnd: np.ndarray, N: int, p: int, q: int) -> np.ndarray:
    W = Bnd.shape[1]
    alpha = np.zeros(N, dtype=np.float64)
    for k in range(N - 1):
        hi_c = k + q
        if hi_c > N - 1:
            hi_c = N - 1
        hi_r = k + p
        if hi_r > N - 1:
            hi_r = N - 1

        a = 0.0
        for c in range(k + 1, hi_c + 1):
            a += Bnd[k, c - k + p]
        alpha[k] = a
        if a == 0.0:
            continue

        for r in range(k + 1, hi_r + 1):
            grk = Bnd[r, k - r + p]
            if grk == 0.0:
                continue
            factor = grk / a
            for c in range(k + 1, hi_c + 1):
                if c == r:
                    continue
                gkc = Bnd[k, c - k + p]
                if gkc == 0.0:
                    continue
                col = c - r + p
                if col < 0 or col >= W:
                    continue
                Bnd[r, col] += factor * gkc
    return alpha


@njit(cache=True)
def _gth_backward(Bnd: np.ndarray, alpha: np.ndarray, N: int, p: int) -> np.ndarray:
    x = np.zeros(N, dtype=np.float64)
    x[N - 1] = 1.0
    for k in range(N - 2, -1, -1):
        hi_r = k + p
        if hi_r > N - 1:
            hi_r = N - 1
        s = 0.0
        for r in range(k + 1, hi_r + 1):
            s += x[r] * Bnd[r, k - r + p]
        x[k] = s / alpha[k]
    return x


def gth_sweep(Q, p: int = None, q: int = None) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """GTH1+GTH2 の前進消去を実行し, スイープ後の帯格納行列 S と alpha を返す.

    Returns:
        S: 帯格納 (N, p+q+1) 配列. 消去掃引後の非対角成分.
        alpha: 各 k の alpha_k (k=0..N-2; alpha[N-1] は未使用で 0).
        p, q: 実際に使用した下帯幅 / 上帯幅.
    """
    N = Q.shape[0]
    if p is None or q is None:
        p, q = bandwidths(Q)
    Bnd = to_banded(Q, p, q)
    alpha = _gth_forward(Bnd, N, p, q)
    return Bnd, alpha, p, q


def solve_stationary_gth(
    Q,
    return_fill: bool = False,
    return_factors: bool = False,
    check_irreducible: bool = True,
) -> np.ndarray:
    """GTH 法で定常分布 pi (pi Q = 0, sum pi = 1) を計算する.

    Args:
        Q: (N, N) 生成行列 (scipy.sparse). 対角要素は使用しない.
        return_fill: True の場合 (pi, nnz_after_sweep) を返す.
        return_factors: True の場合 (pi, S, alpha, p, q) を返す
            (S, alpha は build_LU に渡して L, U を構築できる).
        check_irreducible: True の場合, 消去前に全状態が最終状態へ
            到達可能かを検証する (GTH の前提条件).

    Returns:
        pi: 定常分布 (1D numpy 配列, サイズ N). return_fill/return_factors
            が True の場合はタプルで追加情報も返す.
    """
    N = Q.shape[0]
    if check_irreducible:
        check_reachability(Q)

    p, q = bandwidths(Q)
    Bnd = to_banded(Q, p, q)
    alpha = _gth_forward(Bnd, N, p, q)
    x = _gth_backward(Bnd, alpha, N, p)
    pi = x / x.sum()

    if return_factors:
        return pi, Bnd, alpha, p, q
    if return_fill:
        return pi, int((Bnd != 0).sum())
    return pi


# ============================================================
# LU 分解の構築 (return_factors=True の後段, 検証用 V4)
# ============================================================

def build_LU(S: np.ndarray, alpha: np.ndarray, p: int, q: int) -> Tuple[csr_matrix, csr_matrix]:
    """前進消去後のスイープ行列 S (帯格納) と alpha から L, U を組み立てる.

    GTH の前進消去 (GTH1+GTH2, 全て非負の加算/乗算のみ) は, 行列 A = -G
    (対角 alpha_k > 0, 非対角 -G_ij <= 0) に対する標準ガウス消去 (ピボット
    交換なし) と等価である. 消去段階 k の直前において

        U の k 行目 (対角右側) = -S の k 行目 (c>k)  -- k 行目は段 k 以降
                                                         二度と更新されないので
                                                         S の最終値と一致する
        L の k 列目 (対角下側) = -S の k 列目 (r>k) / alpha_k -- 同様に列 k は
                                                         段 k 以降更新されない

    ため, 前進消去を最後まで行った後の S から直接 L, U を復元できる
    (S 自体は G の非対角成分の絶対値をそのまま掃引したもので符号は正).
    U_kk = alpha_k (k < N-1), U_{N-1,N-1} = 0 (G は特異, rank N-1).
    L は単位下三角 (対角 1).

    結果として -G ≈ L @ U (最大成分の絶対誤差 ~1e-13 程度) となる
    (V4 で ||(-G) - L@U||_max / ||G||_max <= 1e-13 を検証する).
    """
    N = S.shape[0]

    rows_L, cols_L, data_L = [], [], []
    rows_U, cols_U, data_U = [], [], []

    for k in range(N):
        rows_L.append(k)
        cols_L.append(k)
        data_L.append(1.0)
        u_kk = alpha[k] if k < N - 1 else 0.0
        rows_U.append(k)
        cols_U.append(k)
        data_U.append(u_kk)

        hi_c = min(k + q, N - 1)
        for c in range(k + 1, hi_c + 1):
            val = S[k, c - k + p]
            if val != 0.0:
                rows_U.append(k)
                cols_U.append(c)
                data_U.append(-val)

        if k < N - 1 and alpha[k] != 0.0:
            hi_r = min(k + p, N - 1)
            for r in range(k + 1, hi_r + 1):
                val = S[r, k - r + p]
                if val != 0.0:
                    rows_L.append(r)
                    cols_L.append(k)
                    data_L.append(-val / alpha[k])

    L = coo_matrix((data_L, (rows_L, cols_L)), shape=(N, N)).tocsr()
    U = coo_matrix((data_U, (rows_U, cols_U)), shape=(N, N)).tocsr()
    return L, U
