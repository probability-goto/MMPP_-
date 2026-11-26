"""生成行列の既約性の確認 (前向きと逆向きの幅優先探索).

mmpp_predictive.solver.solve_stationary は, 起点 x0 (既定は idx=0) から前向きの幅優先探索で
到達できる集合 R を求め, 生成行列を R の行と列に縮約して解く. このとき R の各状態から x0 へ
戻れなければ, 縮約した行列は既約にならない. このモジュールはそれを確かめる.

隣接の定義は solver と同じ: 対角を 0 にし, 0 の要素を取り除いた行列の非零要素.
    前向き: x の隣接は Q[x, y] != 0 (y != x) となる y (Q の行 x の列番号)
    逆向き: x の隣接は Q[y, x] != 0 (y != x) となる y (Q^T の行 x の列番号)
生成行列の非対角要素は非負なので, 非零は正と同じ (負の非対角要素の数は check_irreducible が数える).

R は前向きの遷移について閉じている (R の状態からの遷移先は R に入る) ので, R の状態から x0 への
道は R の中に収まる. そのため, 逆向きの探索で x0 から見つかる集合を B とすると,
R ⊆ B と, 縮約した行列が既約であることは同値である.

求解の処理 (GTH, 縮約) は変えない. 確認のためだけに使う.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

__all__ = [
    "offdiag_adjacency",
    "forward_reachable",
    "backward_reachable",
    "IrreducibilityResult",
    "check_irreducible",
    "offdiag_rates_below",
]


def offdiag_adjacency(Q) -> csr_matrix:
    """隣接行列: Q の対角を 0 にして 0 の要素を除いたもの (mmpp_predictive.solver と同じ作り方)."""
    adjacency = csr_matrix(Q, copy=True)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency


def _bfs(adjacency: csr_matrix, start: int) -> np.ndarray:
    """adjacency の行 x の非零の列を x の隣接として, start から幅優先探索した集合 (昇順)."""
    found = breadth_first_order(adjacency, i_start=start, directed=True,
                                return_predecessors=False)
    return np.sort(found)


def forward_reachable(Q, start: int = 0) -> np.ndarray:
    """start から前向きにたどって到達できる状態の集合 R (昇順の番号)."""
    return _bfs(offdiag_adjacency(Q), start)


def backward_reachable(Q, start: int = 0) -> np.ndarray:
    """start へ到達できる状態の集合 B (昇順の番号). 隣接は Q[y, x] != 0 (y != x) となる y."""
    return _bfs(offdiag_adjacency(Q).transpose().tocsr(), start)


@dataclass
class IrreducibilityResult:
    """check_irreducible の結果.

    Attributes:
        n_states: Q の状態数 (|S| または |S^P|).
        reachable: 前向きの到達集合 R (昇順).
        backward: 逆向きの到達集合 B (昇順).
        not_back: R \\ B (R の中で start へ戻れない状態, 昇順).
        n_negative_offdiag: 負の非対角要素の数 (生成行列なら 0).
        n_scc_reduced: R に縮約した行列の強連結成分の数 (別の方法による照合. 既約なら 1).
    """
    n_states: int
    reachable: np.ndarray
    backward: np.ndarray
    not_back: np.ndarray
    n_negative_offdiag: int
    n_scc_reduced: int

    @property
    def n_reachable(self) -> int:
        return len(self.reachable)

    @property
    def n_back_in_R(self) -> int:
        return self.n_reachable - len(self.not_back)

    @property
    def irreducible(self) -> bool:
        """R ⊆ B (縮約した行列が既約)."""
        return len(self.not_back) == 0

    @property
    def full(self) -> bool:
        """R = S (縮約しなくても全状態に到達できる)."""
        return self.n_reachable == self.n_states


def check_irreducible(Q, start: int = 0) -> IrreducibilityResult:
    """R (start から前向き), B (start へ逆向き) を求め, R ⊆ B かを判定する."""
    Q = csr_matrix(Q)
    adjacency = offdiag_adjacency(Q)
    n_negative = int((adjacency.data < 0).sum())
    reachable = _bfs(adjacency, start)
    backward = _bfs(adjacency.transpose().tocsr(), start)
    not_back = np.setdiff1d(reachable, backward, assume_unique=True)
    reduced = adjacency[reachable][:, reachable]
    n_scc, _ = connected_components(reduced, directed=True, connection="strong")
    return IrreducibilityResult(
        n_states=Q.shape[0], reachable=reachable, backward=backward, not_back=not_back,
        n_negative_offdiag=n_negative, n_scc_reduced=int(n_scc),
    )


def offdiag_rates_below(Q, threshold: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """0 < Q[x, y] < threshold となる非対角要素 (x != y) の (行, 列, 値)."""
    coo = csr_matrix(Q).tocoo()
    mask = (coo.row != coo.col) & (coo.data > 0) & (coo.data < threshold)
    return coo.row[mask], coo.col[mask], coo.data[mask]
