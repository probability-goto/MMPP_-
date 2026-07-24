"""ModelParameters のテスト.

検証項目:
    - 有効なパラメータでの作成
    - パラメータ妥当性検証 (負値, 型不整合, MMPP 整合性等)
    - 派生量 (D_S, D_M, D, N, lambdas, phase_stationary, lambda_bar)
"""
import numpy as np
import pytest

from mmpp import ModelParameters


class TestModelParametersValid:
    """有効なパラメータでの作成テスト."""

    def test_create_basic(self):
        """基本的な作成が成功."""
        C0 = np.array([[-1.0, 0.0], [0.0, -1.0]])
        C1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        p = ModelParameters(
            c=2, K=5, b=1, mu=1.0, alpha=0.5, beta=0.1,
            C0=C0, C1=C1,
        )
        assert p.c == 2
        assert p.K == 5
        assert p.b == 1

    def test_derived_D_S(self, small_params):
        """D_S = c + 1."""
        assert small_params.D_S == small_params.c + 1

    def test_derived_D_M(self, small_params):
        """D_M = C0 の次元."""
        assert small_params.D_M == 2

    def test_derived_D(self, small_params):
        """D = D_S * D_M."""
        assert small_params.D == small_params.D_S * small_params.D_M

    def test_derived_N(self, small_params):
        """N = D * (K+1)."""
        assert small_params.N == small_params.D * (small_params.K + 1)

    def test_lambdas(self, small_params):
        """lambdas = C1 e (行和)."""
        expected = small_params.C1 @ np.ones(small_params.D_M)
        np.testing.assert_allclose(small_params.lambdas, expected)

    def test_phase_stationary_sums_to_one(self, small_params):
        """位相定常分布の和は 1."""
        pi_mmpp = small_params.phase_stationary
        assert abs(pi_mmpp.sum() - 1.0) < 1e-12

    def test_phase_stationary_is_stationary(self, small_params):
        """pi_mmpp @ (C0 + C1) = 0."""
        Q_mmpp = small_params.C0 + small_params.C1
        pi_mmpp = small_params.phase_stationary
        residual = pi_mmpp @ Q_mmpp
        np.testing.assert_allclose(residual, 0, atol=1e-10)

    def test_lambda_bar_positive(self, small_params):
        """平均到着率は正."""
        assert small_params.lambda_bar > 0


class TestModelParametersValidation:
    """パラメータ妥当性検証テスト."""

    def _base_C(self, D_M=2):
        C0 = -np.eye(D_M)
        C1 = np.eye(D_M)
        return C0, C1

    def test_negative_c_rejected(self):
        C0, C1 = self._base_C()
        with pytest.raises(ValueError, match="c="):
            ModelParameters(c=-1, K=5, b=1, mu=1.0, alpha=0.5, beta=0.1,
                            C0=C0, C1=C1)

    def test_zero_b_rejected(self):
        C0, C1 = self._base_C()
        with pytest.raises(ValueError, match="b="):
            ModelParameters(c=2, K=5, b=0, mu=1.0, alpha=0.5, beta=0.1,
                            C0=C0, C1=C1)

    def test_K_less_than_b_rejected(self):
        """K < b は無効."""
        C0, C1 = self._base_C()
        with pytest.raises(ValueError, match="K="):
            ModelParameters(c=2, K=1, b=2, mu=1.0, alpha=0.5, beta=0.1,
                            C0=C0, C1=C1)

    def test_zero_mu_rejected(self):
        C0, C1 = self._base_C()
        with pytest.raises(ValueError, match="mu="):
            ModelParameters(c=2, K=5, b=1, mu=0.0, alpha=0.5, beta=0.1,
                            C0=C0, C1=C1)

    def test_negative_beta_rejected(self):
        C0, C1 = self._base_C()
        with pytest.raises(ValueError, match="beta="):
            ModelParameters(c=2, K=5, b=1, mu=1.0, alpha=0.5, beta=-0.1,
                            C0=C0, C1=C1)

    def test_non_square_C0_rejected(self):
        C0 = np.zeros((2, 3))
        C1 = np.zeros((2, 3))
        with pytest.raises(ValueError, match="square"):
            ModelParameters(c=2, K=5, b=1, mu=1.0, alpha=0.5, beta=0.1,
                            C0=C0, C1=C1)

    def test_C0_C1_shape_mismatch_rejected(self):
        C0 = np.array([[-1.0, 0.0], [0.0, -1.0]])
        C1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with pytest.raises(ValueError, match="shape"):
            ModelParameters(c=2, K=5, b=1, mu=1.0, alpha=0.5, beta=0.1,
                            C0=C0, C1=C1)

    def test_C0_off_diagonal_negative_rejected(self):
        """C0 の非対角成分が負ならエラー."""
        C0 = np.array([[-1.0, -0.1], [0.1, -1.0]])
        C1 = np.array([[1.1, 0.0], [0.0, 0.9]])
        with pytest.raises(ValueError, match="C0"):
            ModelParameters(c=2, K=5, b=1, mu=1.0, alpha=0.5, beta=0.1,
                            C0=C0, C1=C1)

    def test_C1_negative_rejected(self):
        """C1 に負の成分があるとエラー."""
        C0 = np.array([[-0.5, 0.0], [0.0, -0.5]])
        C1 = np.array([[0.6, -0.1], [0.0, 0.5]])
        with pytest.raises(ValueError, match="C1"):
            ModelParameters(c=2, K=5, b=1, mu=1.0, alpha=0.5, beta=0.1,
                            C0=C0, C1=C1)

    def test_MMPP_inconsistency_rejected(self):
        """(C0 + C1) e != 0 ならエラー."""
        C0 = np.array([[-2.0, 0.5], [0.5, -2.0]])
        C1 = np.array([[1.0, 0.0], [0.0, 1.0]])  # 行和 -0.5, -0.5
        with pytest.raises(ValueError, match="MMPP"):
            ModelParameters(c=2, K=5, b=1, mu=1.0, alpha=0.5, beta=0.1,
                            C0=C0, C1=C1)


class TestDifferentPhases:
    """異なる D_M でのテスト."""

    def test_single_phase(self):
        """D_M = 1 (Poisson 到着)."""
        C0 = np.array([[-0.5]])
        C1 = np.array([[0.5]])
        p = ModelParameters(c=1, K=3, b=1, mu=1.0, alpha=0.5, beta=0.1,
                            C0=C0, C1=C1)
        assert p.D_M == 1
        assert p.lambda_bar == 0.5

    def test_three_phases(self):
        """D_M = 3."""
        C0 = np.array([
            [-1.0, 0.1, 0.1],
            [0.1, -1.0, 0.1],
            [0.1, 0.1, -1.0],
        ])
        C1 = np.array([
            [0.8, 0.0, 0.0],
            [0.0, 0.8, 0.0],
            [0.0, 0.0, 0.8],
        ])
        p = ModelParameters(c=1, K=3, b=1, mu=1.0, alpha=0.5, beta=0.1,
                            C0=C0, C1=C1)
        assert p.D_M == 3
        # 対称なので pi_mmpp = (1/3, 1/3, 1/3)
        np.testing.assert_allclose(p.phase_stationary, [1/3, 1/3, 1/3], atol=1e-10)
        assert abs(p.lambda_bar - 0.8) < 1e-10
