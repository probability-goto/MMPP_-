"""定常分布の計算.

Predictive モデルの名目状態空間 (i, s, j, F) は, i + s <= c を満たす
全ての組み合わせを許容するように構築されている (mmpp_predictive.generator).
しかし実際に到達可能な状態はその真部分集合でしかない. 特に n_target=0
(ベースモデル相当) の場合, s は (i, j) から一意に定まる
(compute_required_s(i, j, b, c)) ため, ほとんどの (i, s) の組は到達不能に
なる. n_target > 0 の場合でも, 事前セットアップで一度に増やせる量には
上限があるため, 名目状態空間の全域が到達可能とは限らない.

到達不能な状態を残したまま生成行列 Q を解こうとすると, Q が可約になり
(複数の連結成分に分解し), ベースの LU 分解ソルバーが特異行列エラーで
失敗する (GTH 法についても同様: 全状態が最終状態から到達可能という前提
(mmpp.gth_solver.check_reachability) が成り立たなくなる). そこで, 定常
分布を計算する前に「全サーバー停止・空システム・通常位相」状態
(i=0, s=0, j=0, F=0; これは常に到達可能かつ, 系が安定であれば再帰的に
訪れる状態) から到達可能な部分状態空間のみを抽出し, その既約な部分生成
行列上でベースの solve_stationary (mmpp.solver.solve_stationary) を再利用
して解く. 得られた解は名目状態空間全体のサイズのベクトルに復元し
(到達不能な状態には確率 0 を割り当てる), mmpp_predictive.metrics.Metrics
がそのまま扱えるようにする.

縮約後の部分空間は BFS 到達集合を元のインデックス順にソートしたもの
(np.sort(reachable)) で間引くため, 帯幅は元の名目状態空間の帯幅以下に
留まる. ただし縮約の結果, 元の (i, j-major) 順序が想定する帯構造から
外れて帯幅が大きくなる場合があるため, solver="gth" 利用時は実際に測定
した帯幅 (p, q) を報告する.
"""
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

from mmpp.solver import solve_stationary as _solve_stationary_base
from mmpp.gth_solver import bandwidths as _bandwidths

__all__ = ["solve_stationary"]

# 縮約後の計算量 N*p*q がこの値を超えたら警告する (GTH の計算量は
# O(N*p*q) であり, predictive モデルは (i,s,j,F) の縮約状態空間が
# ベースモデルの j-major 順序と噛み合わず, 個々の p, q が大きくなくても
# 積が非常に大きくなりうるため, 積そのものを監視する).
_COST_WARN_THRESHOLD = 5e8


def solve_stationary(
    Q, start_index: int = 0, solver: str = "gth", method: str = "sparse"
) -> np.ndarray:
    """定常分布 pi を計算する (到達不能な状態を除いてから解く).

    Args:
        Q: (N, N) の生成行列 (scipy.sparse.csr_matrix 等).
        start_index: 到達可能性 BFS の起点となる状態インデックス.
            既定値 0 は (i=0, s=0, j=0, F=0) に対応する
            (mmpp_predictive.state_space のインデックス化規約による).
        solver: 'gth' (既定) or 'splu'. mmpp.solver.solve_stationary に
            そのまま渡す.
        method: solver='splu' のときのみ有効 ('sparse'/'dense').

    Returns:
        pi: 定常分布 (1D numpy 配列, サイズ N). 到達不能な状態の
            確率は 0.
    """
    N = Q.shape[0]
    Q = Q.tocsr()

    # 対角成分 (負) を除いた隣接構造で到達可能性を判定する
    adjacency = Q.copy()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    reachable = breadth_first_order(
        adjacency, i_start=start_index, directed=True, return_predecessors=False
    )

    if len(reachable) == N:
        Q_reduced = Q
        reachable = None
    else:
        reachable = np.sort(reachable)
        Q_reduced = Q[reachable][:, reachable].tocsr()

    if solver == "gth":
        p, q = _bandwidths(Q_reduced)
        cost = Q_reduced.shape[0] * p * q
        if cost > _COST_WARN_THRESHOLD:
            warnings.warn(
                f"Predictive モデルの縮約後の帯幅が大きい "
                f"(N={Q_reduced.shape[0]}, p={p}, q={q}, N*p*q={cost:.3g}). "
                "GTH の計算量は O(N*p*q) のため, 計算時間・メモリが想定より "
                "増大する可能性があります。",
                RuntimeWarning,
                stacklevel=2,
            )

    pi_reduced = _solve_stationary_base(Q_reduced, solver=solver, method=method)

    if reachable is None:
        return pi_reduced

    pi = np.zeros(N)
    pi[reachable] = pi_reduced
    return pi
