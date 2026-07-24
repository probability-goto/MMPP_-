"""定常分布ソルバーのテスト.

検証項目:
    - pi >= 0, sum pi = 1
    - pi Q = 0 (残差ゼロ)
    - sparse と dense で同じ解
    - 手計算可能な小さいケースとの一致
"""
import numpy as np
import pytest

from mmpp import (
    ModelParameters,
    build_generator,
    solve_stationary,
)


class TestBasicProperties:
    """pi の基本性質."""

    def test_non_negative(self, small_params):
        """pi >= 0."""
        Q = build_generator(small_params)
        pi = solve_stationary(Q)
        assert (pi >= 0).all()

    def test_sums_to_one(self, small_params):
        """sum pi = 1."""
        Q = build_generator(small_params)
        pi = solve_stationary(Q)
        assert abs(pi.sum() - 1.0) < 1e-10

    def test_residual_small(self, small_params):
        """||pi Q||_inf ≈ 0."""
        Q = build_generator(small_params)
        pi = solve_stationary(Q)
        residual = np.abs(pi @ Q).max()
        assert residual < 1e-10

    def test_correct_shape(self, small_params):
        """pi のサイズ = N."""
        Q = build_generator(small_params)
        pi = solve_stationary(Q)
        assert pi.shape == (small_params.N,)


class TestSparseVsDense:
    """sparse と dense のソルバー一致."""

    def test_agreement_small(self, small_params):
        Q = build_generator(small_params)
        pi_sparse = solve_stationary(Q, method="sparse")
        pi_dense = solve_stationary(Q, method="dense")
        np.testing.assert_allclose(pi_sparse, pi_dense, atol=1e-10)

    def test_agreement_batch(self, batch_params):
        Q = build_generator(batch_params)
        pi_sparse = solve_stationary(Q, method="sparse")
        pi_dense = solve_stationary(Q, method="dense")
        np.testing.assert_allclose(pi_sparse, pi_dense, atol=1e-10)

    def test_agreement_medium(self, medium_params):
        """中規模でも sparse == dense (メモリ許す限り)."""
        Q = build_generator(medium_params)
        pi_sparse = solve_stationary(Q, method="sparse")
        pi_dense = solve_stationary(Q, method="dense")
        np.testing.assert_allclose(pi_sparse, pi_dense, atol=1e-10)


class TestKnownSolutions:
    """既知の解との比較."""

    def test_two_state_analytical(self):
        """4 状態の CTMC の解析解との一致.

        c=1, K=1, b=1, D_M=1. N=4.

        インデックス (j-major, idx = j*D + i*D_M + F, D=2, D_M=1):
            (i=0, j=0, F=0) -> idx = 0
            (i=1, j=0, F=0) -> idx = 1
            (i=0, j=1, F=0) -> idx = 2
            (i=1, j=1, F=0) -> idx = 3

        遷移 (λ=μ=1, α=100, β=0.01):
            (0, 0): λ=1 で (0, 1)
            (0, 1): α*S=100*1=100 で (1, 1)
            (1, 0): β*I=0.01*1=0.01 で (0, 0), λ=1 で (1, 1)
            (1, 1): μ*B=1 で (1, 0)

        バランス方程式:
            π(0,0)  出: 1; 入: 0.01 * π(1,0)
                     => π(0,0) = 0.01 * π(1,0)
            π(0,1)  出: 100; 入: 1 * π(0,0)
                     => 100 * π(0,1) = π(0,0), つまり π(0,1) = π(0,0)/100
            π(1,0)  出: 0.01 + 1 = 1.01; 入: 1 * π(1,1)
                     => 1.01 * π(1,0) = π(1,1)
            π(1,1)  出: 1; 入: 100 * π(0,1) + 1 * π(1,0)
                     => π(1,1) = π(0,0) + π(1,0)

        c = π(1,0) とおくと:
            π(0,0) = 0.01 c
            π(0,1) = 10^-4 c
            π(1,0) = c
            π(1,1) = 1.01 c
        正規化: c * (0.01 + 10^-4 + 1 + 1.01) = 1
                c ≈ 1 / 2.0201 ≈ 0.49502
        """
        C0 = np.array([[-1.0]])
        C1 = np.array([[1.0]])
        params = ModelParameters(
            c=1, K=1, b=1, mu=1.0, alpha=100.0, beta=0.01,
            C0=C0, C1=C1,
        )
        Q = build_generator(params)
        pi = solve_stationary(Q)

        # 解析解
        c_val = 1.0 / 2.0201
        expected = np.array([
            0.01 * c_val,       # (0, 0)
            c_val,              # (1, 0)
            1e-4 * c_val,       # (0, 1)
            1.01 * c_val,       # (1, 1)
        ])
        np.testing.assert_allclose(pi, expected, atol=1e-8)

        # i=1 状態の総質量 (Delayoff 稀のためほとんど i=1)
        pi_i1 = pi[1] + pi[3]  # (1, 0) と (1, 1)
        assert pi_i1 > 0.99

    def test_stationary_MMPP_marginal(self, small_params):
        """位相ごとのマージナル分布が MMPP 定常分布と (近似的に) 一致."""
        Q = build_generator(small_params)
        pi = solve_stationary(Q)
        # F 別のマージナル: pi_F = sum over (i, j) of pi(i, j, F)
        pi_F = np.zeros(small_params.D_M)
        for idx in range(small_params.N):
            _, _, F = ((idx % small_params.D) // small_params.D_M,
                       idx // small_params.D,
                       idx % small_params.D_M)
            # 上記は state_space.decode と同じロジック
            j, rem = divmod(idx, small_params.D)
            i, F = divmod(rem, small_params.D_M)
            pi_F[F] += pi[idx]
        # 位相定常分布と一致するはず
        pi_mmpp = small_params.phase_stationary
        np.testing.assert_allclose(pi_F, pi_mmpp, atol=1e-8)


class TestEdgeCases:
    """エッジケース."""

    def test_single_phase(self, mm1k_like_params):
        """D_M=1 のとき (Poisson 到着) でも動作."""
        Q = build_generator(mm1k_like_params)
        pi = solve_stationary(Q)
        assert abs(pi.sum() - 1.0) < 1e-10
        assert (pi >= -1e-12).all()

    def test_high_stiffness_still_solvable(self):
        """剛性が高くても sparse ソルバーが動く.

        μ=1, β=1e-6 で剛性比 10^6.
        """
        C0 = np.array([[-0.5]])
        C1 = np.array([[0.5]])
        params = ModelParameters(
            c=2, K=10, b=1, mu=1.0, alpha=100.0, beta=1e-6,
            C0=C0, C1=C1,
        )
        Q = build_generator(params)
        pi = solve_stationary(Q)
        assert abs(pi.sum() - 1.0) < 1e-10

    def test_unknown_method_raises(self, small_params):
        """未知の method 指定でエラー."""
        Q = build_generator(small_params)
        with pytest.raises(ValueError, match="method"):
            solve_stationary(Q, method="foo")


class TestFixationTechnique:
    """x_N = 1 固定法の妥当性."""

    def test_fixation_gives_positive_probability(self, small_params):
        """固定変数 (最後の状態) の正規化後の確率も正しい."""
        Q = build_generator(small_params)
        pi = solve_stationary(Q)
        # 最終状態 (i=c, j=K, F=D_M-1) の確率
        pi_last = pi[-1]
        # 有限な非負値であるべき
        assert 0 <= pi_last <= 1
        # 実際の値は小さいはず (満杯かつフェーズ D_M-1 は稀)
        # ただし特定の値を仮定はしない
