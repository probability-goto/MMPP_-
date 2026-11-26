"""mmpp_experiments.irreducibility (縮約した生成行列の既約性の確認) のテスト."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mmpp import ModelParameters
from mmpp import build_generator as build_base_generator
from mmpp_experiments.irreducibility import (
    backward_reachable,
    check_irreducible,
    forward_reachable,
    offdiag_rates_below,
)
from mmpp_predictive import PredictiveModelParameters, build_generator, solve_stationary

A, B, C, D, E = range(5)


def _generator(n, edges):
    """率 1 の遷移 (src, dst) を持つ n 状態の生成行列 (対角は行和が 0 になる値)."""
    Q = np.zeros((n, n))
    for src, dst in edges:
        Q[src, dst] = 1.0
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return csr_matrix(Q)


def test_irreducible_example():
    """A<->B<->C と D->A, E->A. x0 = A から R = {A,B,C}, B は全状態, 既約."""
    Q = _generator(5, [(A, B), (B, A), (B, C), (C, B), (D, A), (E, A)])
    assert forward_reachable(Q, A).tolist() == [A, B, C]
    assert backward_reachable(Q, A).tolist() == [A, B, C, D, E]
    res = check_irreducible(Q, A)
    assert res.irreducible
    assert res.n_reachable == 3 and res.n_back_in_R == 3
    assert res.not_back.tolist() == []
    assert res.n_scc_reduced == 1
    assert not res.full


def test_not_irreducible_example():
    """A->B, B<->C. x0 = A から R = {A,B,C}, B = {A}, 既約でない (R \\ B = {B, C})."""
    Q = _generator(3, [(A, B), (B, C), (C, B)])
    assert forward_reachable(Q, A).tolist() == [A, B, C]
    assert backward_reachable(Q, A).tolist() == [A]
    res = check_irreducible(Q, A)
    assert not res.irreducible
    assert res.n_reachable == 3 and res.n_back_in_R == 1
    assert res.not_back.tolist() == [B, C]
    assert res.n_scc_reduced == 2


def test_diagonal_is_not_adjacency():
    """対角要素は隣接に含めない (対角だけの状態からは, どこへも行けない)."""
    Q = csr_matrix(np.array([[-1.0, 0.0], [1.0, -1.0]]))
    assert forward_reachable(Q, 0).tolist() == [0]
    assert backward_reachable(Q, 0).tolist() == [0, 1]


def test_offdiag_rates_below():
    Q = _generator(3, [(A, B), (B, A), (B, C), (C, B)]).tolil()
    Q[A, C] = 1e-13
    Q[A, A] -= 1e-13
    rows, cols, vals = offdiag_rates_below(Q.tocsr(), 1e-12)
    assert rows.tolist() == [A] and cols.tolist() == [C]
    assert vals.tolist() == [1e-13]


def _small_mmpp():
    C1 = np.array([[0.8, 0.0], [0.0, 3.2]])
    C0 = np.array([[-1.1, 0.3], [0.3, -3.5]])  # (C0 + C1) e = 0
    return C0, C1


@pytest.mark.parametrize("n_target,protect", [(0, False), (2, False), (2, True), (4, True)])
def test_forward_set_matches_solver_support(n_target, protect):
    """前向きの集合 R が, solver が縮約に使う集合 (定常分布が正の状態) と一致し, 既約である."""
    C0, C1 = _small_mmpp()
    params = PredictiveModelParameters(
        c=4, K=12, b=2, mu=1.0, alpha=0.2, beta=0.1, C0=C0, C1=C1,
        n_target=n_target, gamma=3.0, protect_presetup=protect, protect_delayoff=protect)
    Q = build_generator(params)
    pi = solve_stationary(Q)
    res = check_irreducible(Q, 0)
    assert np.flatnonzero(pi > 0).tolist() == res.reachable.tolist()
    assert res.irreducible and res.n_scc_reduced == 1
    assert res.n_negative_offdiag == 0


def test_base_model_full_and_irreducible():
    C0, C1 = _small_mmpp()
    params = ModelParameters(c=4, K=12, b=2, mu=1.0, alpha=0.2, beta=0.1, C0=C0, C1=C1)
    res = check_irreducible(build_base_generator(params), 0)
    assert res.full and res.irreducible
