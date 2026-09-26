"""定常分布 π Q = 0 のソルバー.

πQ = 0 かつ sum(π) = 1 を満たす確率ベクトル π を求める.

主要手法:
    - 疎行列 LU 分解 (推奨): scipy.sparse.linalg.spsolve
    - 密行列 LU 分解 (検証用): numpy.linalg.solve

理論的背景 (splu, solver="splu"):
    - Q は生成行列 (Q e = 0) なので階数落ち (rank(Q) = N-1)
    - x_N = 1 固定により (N-1) x (N-1) 正方系に帰着
    - Q^T の左上 (N-1) x (N-1) 部分行列を A とすると -A は非特異 M 行列
    - よって A y = -u は非負一意解を持つ (M 行列理論の帰結)
    - 欠点: 引き算を伴う通常の LU 分解であり, 絶対誤差が浮動小数点の丸め
      (~1e-17) で頭打ちになる. π の成分が 1e-15 を下回ると相対誤差が破綻し
      うる (負の確率も出うる).

GTH 法 (solver="gth", 既定):
    - O'Cinneide (1996) の GTH1-3 を帯格納で実装 (mmpp.gth_solver).
    - 非対角成分 (非負) のみを使い, 引き算を一切行わない消去法.
    - 各成分が相対精度で求まるため, splu が破綻する極小成分でも安定.
    - 前提: 全状態が最終状態から到達可能 (既約なら成立). 満たさない場合は
      ValueError を送出する.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from mmpp.gth_solver import solve_stationary_gth


def solve_stationary(
    Q, method: str = "sparse", solver: str = "gth", tol: float = 1e-8
) -> np.ndarray:
    """定常分布 π を計算する.

    Args:
        Q: (N, N) の生成行列 (scipy.sparse または numpy array).
        method: solver="splu" のときのみ有効. 'sparse' (疎行列 LU 分解) or
                'dense' (密行列 LU 分解). 'sparse' が主用途. 'dense' は
                小規模検証用.
        solver: 'gth' (既定, O'Cinneide の GTH 法, 相対精度) or
                'splu' (旧来の疎行列 LU 分解, method で sparse/dense を選択).
        tol: 数値検証時のトレランス.

    Returns:
        pi: 定常分布 (1D numpy 配列, サイズ N).

    前提:
        生成行列 Q は既約 (irreducible) であること. 具体的には
        MMPP 位相過程 (C0 + C1) が既約であることが十分条件で,
        これは ModelParameters の検証時にチェックされる.
        非既約な Q に対しては (solver="splu" の場合) 数値誤差の範囲で
        「もっともらしい間違った解」を返す可能性があるため, 呼び出し側で
        既約性を保証すること (solver="gth" は到達可能性チェックで検出する).

    Raises:
        RuntimeError: 数値解が検証条件を満たさない場合.
        ValueError: 未知の method/solver 指定, または (solver="gth" で)
            GTH の前提条件 (全状態が最終状態へ到達可能) を満たさない場合.
    """
    N = Q.shape[0]

    if method not in ("sparse", "dense"):
        raise ValueError(f"Unknown method '{method}', use 'sparse' or 'dense'")

    if solver == "gth":
        pi = solve_stationary_gth(Q)
    elif solver == "splu":
        pi = _solve_sparse(Q) if method == "sparse" else _solve_dense(Q)
    else:
        raise ValueError(f"Unknown solver '{solver}', use 'gth' or 'splu'")

    # 検証
    residual = float(np.abs(pi @ Q).max())
    if residual > tol:
        raise RuntimeError(
            f"数値解の残差過大: ||pi Q||_inf = {residual:.2e} > tol={tol:.2e}"
        )
    total = float(pi.sum())
    if abs(total - 1.0) > tol:
        raise RuntimeError(f"正規化違反: sum(pi) = {total}")
    if pi.min() < -tol:
        raise RuntimeError(f"非負性違反: min(pi) = {pi.min():.2e}")

    # 数値誤差による僅かな負値をクリップ、再正規化
    pi = np.maximum(pi, 0.0)
    pi = pi / pi.sum()

    return pi


def _solve_sparse(Q) -> np.ndarray:
    """疎行列 LU 分解による直接法.

    手順:
        1. Q^T の左上 (N-1) x (N-1) 部分 A と最右列先頭 N-1 成分 u を抽出
        2. spsolve で A y = -u を解く (LU 分解 + 前進代入 + 後退代入)
        3. y に x_N = 1 を復元し、正規化
    """
    # csr でも csc でもよいが csc は列スライスに有利
    # scipy.sparse.csr_matrix の T は csc になる
    QT = Q.T.tocsr()
    A = QT[:-1, :-1].tocsc()  # (N-1) x (N-1) sub-matrix, CSC 形式 (spsolve が期待する形)
    u = np.asarray(QT[:-1, -1].todense()).flatten()  # 最右列の先頭 N-1 成分

    # 疎行列 LU 分解 + 前進代入 + 後退代入
    y = spsolve(A, -u)

    # 復元と正規化
    x = np.append(y, 1.0)
    pi = x / x.sum()
    return pi


def _solve_dense(Q) -> np.ndarray:
    """密行列 LU 分解 (検証用, 小規模のみ).

    scipy.sparse でも numpy array でも入力可.
    """
    Q_dense = Q.toarray() if hasattr(Q, "toarray") else np.asarray(Q)
    QT = Q_dense.T
    A = QT[:-1, :-1]
    u = QT[:-1, -1]
    y = np.linalg.solve(A, -u)
    x = np.append(y, 1.0)
    pi = x / x.sum()
    return pi
