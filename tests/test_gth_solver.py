"""GTH ソルバー (mmpp.gth_solver) の検証テスト.

タスク仕様の検証項目に対応:
    V1(a): M/M/1/K の解析解 (pi_n ∝ rho^n) との一致
    V1(b): O'Cinneide (1996) sec.4 の 20 状態三重対角例, 通常 Gauss 消去との対比
    V4:    分解の検証 (||G - LU||_max / ||G||_max <= 1e-13, 符号規約は
           build_LU の docstring どおり -G ≈ L @ U)
    V5:    性質の確認 (pi >= 0, sum pi = 1, 残差 ||pi Q||_inf)
"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mmpp import ModelParameters, build_generator
from mmpp.gth_solver import (
    bandwidths,
    check_reachability,
    solve_stationary_gth,
    build_LU,
    to_banded,
)


# ============================================================
# V1(a): M/M/1/K 解析解
# ============================================================

def _mm1k_generator(K: int, lam: float, mu: float) -> csr_matrix:
    """M/M/1/K の生成行列を直接構築する (状態 0..K, i.i.d. Q_ij)."""
    N = K + 1
    rows, cols, vals = [], [], []
    for n in range(N):
        if n < K:
            rows += [n, n]; cols += [n + 1, n]; vals += [lam, -lam]
        if n > 0:
            rows += [n, n]; cols += [n - 1, n]; vals += [mu, -mu]
    Q = csr_matrix((vals, (rows, cols)), shape=(N, N))
    Q.sum_duplicates()
    return Q


class TestMM1K:
    @pytest.mark.parametrize("lam,mu,K", [(0.3, 1.0, 10), (2.0, 1.0, 20), (1.0, 1.0, 15)])
    def test_analytical_match(self, lam, mu, K):
        Q = _mm1k_generator(K, lam, mu)
        pi = solve_stationary_gth(Q)

        rho = lam / mu
        n = np.arange(K + 1)
        expected = rho ** n
        expected = expected / expected.sum()

        rel_err = np.abs(pi - expected) / expected
        assert rel_err.max() < 1e-13


# ============================================================
# V1(b): O'Cinneide (1996) sec.4 の 20 状態例
# ============================================================

class TestOCinneideExample:
    def _build(self, n=20, p_up=0.1, p_down=0.8):
        P = np.zeros((n, n))
        for i in range(n):
            if i + 1 < n:
                P[i, i + 1] = p_up
            if i - 1 >= 0:
                P[i, i - 1] = p_down
            P[i, i] = 1.0 - P[i].sum()
        G = P - np.eye(n)
        return csr_matrix(G), n

    def test_gth_relative_error(self):
        Q, n = self._build()
        pi = solve_stationary_gth(Q)
        expected = np.array([8.0 ** (-i) for i in range(n)])
        expected = expected / expected.sum()
        rel_err = np.abs(pi - expected) / expected
        assert rel_err.max() <= 1e-13

    def test_naive_gauss_elimination_fails(self):
        """通常の Gauss 消去 (splu 相当) は極小成分で破綻しうることを示す."""
        Q, n = self._build()
        G = Q.toarray()
        QT = G.T.copy()
        A = QT[:-1, :-1]
        u = QT[:-1, -1]
        y = np.linalg.solve(A, -u)
        x = np.append(y, 1.0)
        pi_gauss = x / x.sum()

        expected = np.array([8.0 ** (-i) for i in range(n)])
        expected = expected / expected.sum()
        rel_err_gauss = np.abs(pi_gauss - expected) / expected

        # 通常の Gauss 消去は GTH よりはるかに悪い相対誤差 (負値も出うる)
        assert rel_err_gauss.max() > 1e-6


# ============================================================
# V4: LU 分解の検証
# ============================================================

class TestLUFactorization:
    def test_small_random_generator(self):
        rng = np.random.default_rng(0)
        n = 12
        A = rng.random((n, n))
        np.fill_diagonal(A, 0.0)
        A[A < 0.5] = 0.0  # まばらにする
        G = A.copy()
        np.fill_diagonal(G, -A.sum(axis=1))
        Q = csr_matrix(G)
        check_reachability(Q)  # 前提を満たすことを確認 (満たさなければ skip 相当)

        pi, S, alpha, p, q = solve_stationary_gth(Q, return_factors=True)
        L, U = build_LU(S, alpha, p, q)
        LU = (L @ U).toarray()
        err = np.abs((-G) - LU).max() / np.abs(G).max()
        assert err <= 1e-13

    def test_model_generator(self, batch_params):
        Q = build_generator(batch_params)
        G = Q.toarray()
        pi, S, alpha, p, q = solve_stationary_gth(Q, return_factors=True)
        L, U = build_LU(S, alpha, p, q)
        LU = (L @ U).toarray()
        err = np.abs((-G) - LU).max() / np.abs(G).max()
        assert err <= 1e-13


# ============================================================
# V5: 基本性質
# ============================================================

class TestProperties:
    def test_nonneg_sum_residual(self, small_params, medium_params, batch_params, mm1k_like_params):
        for params in (small_params, medium_params, batch_params, mm1k_like_params):
            Q = build_generator(params)
            pi = solve_stationary_gth(Q)
            assert (pi >= 0).all()
            assert abs(pi.sum() - 1.0) < 1e-10
            residual = np.abs(pi @ Q).max()
            assert residual < 1e-8

    def test_agrees_with_splu(self, small_params, medium_params, batch_params):
        from mmpp.solver import solve_stationary
        for params in (small_params, medium_params, batch_params):
            Q = build_generator(params)
            pi_gth = solve_stationary(Q, solver="gth")
            pi_splu = solve_stationary(Q, solver="splu")
            np.testing.assert_allclose(pi_gth, pi_splu, atol=1e-9)


# ============================================================
# 到達可能性の前提条件チェック
# ============================================================

class TestReachabilityGuard:
    def test_reducible_raises(self):
        # 2 つの非連結な 2 状態 CTMC を対角ブロックで並べる (既約でない)
        G = np.array([
            [-1.0, 1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
            [0.0, 0.0, 1.0, -1.0],
        ])
        Q = csr_matrix(G)
        with pytest.raises(ValueError, match="到達不能"):
            solve_stationary_gth(Q)

    def test_bandwidths_reasonable(self, batch_params):
        Q = build_generator(batch_params)
        p, q = bandwidths(Q)
        D = batch_params.D
        assert p == batch_params.b * D
        assert q == D


# ============================================================
# V3: モデルの同一性 (build_generator が返す Q がそのまま GTH に渡ること)
# ============================================================

class TestModelIdentity:
    """本リポジトリには build_Q_direct / build_Q_kron のような複数実装は
    存在せず, Q の構築経路は build_generator の 1 つのみである. そのため
    V3 の本質は「ソルバーに渡る過程で Q の値が一切変質しないこと」の確認
    に帰着する. to_banded による帯格納への変換が, build_generator が出力
    した非対角成分を厳密に (丸め誤差なく) 保持することを確認する.
    """

    def test_banded_roundtrip_exact(self, small_params, medium_params, batch_params):
        for params in (small_params, medium_params, batch_params):
            Q = build_generator(params)
            p, q = bandwidths(Q)
            Bnd = to_banded(Q, p, q)

            Qc = Q.tocoo()
            off = Qc.row != Qc.col
            for r, c, v in zip(Qc.row[off], Qc.col[off], Qc.data[off]):
                assert Bnd[r, c - r + p] == v
