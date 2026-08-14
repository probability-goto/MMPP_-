"""mmpp_predictive.generator のテスト."""
import pytest
import numpy as np

from mmpp_predictive.model import PredictiveModelParameters
from mmpp_predictive.generator import build_generator
from mmpp_predictive.state_space import num_states

try:
    from scripts._mmpp_burst import build_mmpp
except ImportError:
    from _mmpp_burst import build_mmpp


def make_test_params(n_target=0, gamma=1.0):
    """小規模テスト用のパラメータ (c=3, K=5, D_M=2)."""
    c, K, b, mu = 3, 5, 1, 1.0
    C0, C1 = build_mmpp(rho=0.5, delta=0.6, sigma=0.1, c=c, b=b, mu=mu)
    return PredictiveModelParameters(
        c=c, K=K, b=b, mu=mu, alpha=0.5, beta=0.1,
        C0=C0, C1=C1,
        n_target=n_target, gamma=gamma,
    )


def test_generator_shape():
    p = make_test_params()
    Q = build_generator(p)
    N = num_states(p.c, p.K, p.D_M)
    assert Q.shape == (N, N)


def test_generator_row_sums_zero():
    """各行の和が 0 であること (生成行列の必要条件)."""
    p = make_test_params()
    Q = build_generator(p)
    row_sums = np.array(Q.sum(axis=1)).flatten()
    np.testing.assert_allclose(row_sums, 0, atol=1e-12)


def test_generator_off_diagonals_nonneg():
    """非対角成分が非負であること."""
    p = make_test_params()
    Q = build_generator(p).toarray()
    N = Q.shape[0]
    for i in range(N):
        for j in range(N):
            if i != j:
                assert Q[i, j] >= 0, f"Q[{i},{j}]={Q[i,j]} is negative"


def test_generator_diagonals_nonpos():
    """対角成分が非正であること."""
    p = make_test_params()
    Q = build_generator(p).toarray()
    diag = np.diag(Q)
    assert np.all(diag <= 0)


def test_generator_row_sums_zero_with_predictive_kick():
    """事前セットアップが有効な場合も生成行列の性質が保たれること."""
    p = make_test_params(n_target=2, gamma=3.0)
    Q = build_generator(p)
    row_sums = np.array(Q.sum(axis=1)).flatten()
    np.testing.assert_allclose(row_sums, 0, atol=1e-12)
    diag = np.array(Q.diagonal())
    assert np.all(diag <= 0)
