"""統合テスト.

end-to-end のワークフロー確認と、既知の解析解との比較.
"""
import numpy as np
import pytest

from mmpp import (
    ModelParameters,
    build_generator,
    solve_stationary,
    Metrics,
)


class TestEndToEnd:
    """パラメータ設定 → Q 構築 → 定常分布計算 → 指標計算 のフロー."""

    def test_basic_workflow(self):
        """基本的なワークフローが動作."""
        C0 = np.array([[-1.0, 0.05], [0.05, -1.0]])
        C1 = np.array([[0.95, 0.0], [0.0, 0.95]])
        params = ModelParameters(
            c=5, K=20, b=2, mu=1.0, alpha=0.5, beta=0.1,
            C0=C0, C1=C1,
        )
        Q = build_generator(params)
        pi = solve_stationary(Q)
        metrics = Metrics(params, pi)

        d = metrics.all_metrics()
        assert 0 <= d["P_block"] <= 1
        assert 0 <= d["E[j]"] <= params.K
        assert 0 <= d["rho"] <= 1
        # サーバー数保存
        total = d["E[B]"] + d["E[I]"] + d["E[S]"] + d["E[Off]"]
        assert abs(total - params.c) < 1e-10
        # 流量バランス
        assert abs(
            d["lambda_eff"] - params.b * params.mu * d["E[B]"]
        ) < 1e-8


class TestMM1KLimit:
    """M/M/1/K 極限との比較.

    D_M=1, c=1, b=1, alpha → 大, beta → 小 で M/M/1/K に一致.
    ただし本モデルには (0, j) 系の transient 状態があるので
    完全一致ではなく漸近一致.
    """

    def _mm1k_stationary(self, lam, mu, K):
        """M/M/1/K の解析解.

        pi_n = (1 - rho) * rho^n / (1 - rho^(K+1)) if rho != 1
        pi_n = 1 / (K + 1)                        if rho == 1
        """
        rho = lam / mu
        if abs(rho - 1) < 1e-10:
            return np.ones(K + 1) / (K + 1)
        return (1 - rho) * rho ** np.arange(K + 1) / (1 - rho ** (K + 1))

    def test_limit_matches_MM1K(self):
        """alpha 大, beta 小の極限で M/M/1/K に近づく."""
        lam = 0.6
        mu = 1.0
        K = 8
        # alpha 十分大 & beta 十分小 & c=1
        C0 = np.array([[-lam]])
        C1 = np.array([[lam]])
        params = ModelParameters(
            c=1, K=K, b=1, mu=mu,
            alpha=1e5,  # setup 実質瞬時
            beta=1e-8,  # Delayoff 実質発生せず
            C0=C0, C1=C1,
        )
        Q = build_generator(params)
        pi = solve_stationary(Q)

        # レベル別マージナル
        pi_j = np.array([
            pi[j * params.D:(j + 1) * params.D].sum()
            for j in range(K + 1)
        ])
        # M/M/1/K 解析解
        pi_j_analytical = self._mm1k_stationary(lam, mu, K)

        # 近似的な一致 (transient 状態のため完全一致はしない)
        np.testing.assert_allclose(pi_j, pi_j_analytical, atol=1e-4)


class TestSymmetryProperties:
    """対称なパラメータで対称な解."""

    def test_symmetric_phases(self):
        """C0, C1 が位相対称なら定常分布も位相対称."""
        # C0 と C1 が位相 (0, 1) について対称
        C0 = np.array([[-1.0, 0.1], [0.1, -1.0]])
        C1 = np.array([[0.9, 0.0], [0.0, 0.9]])
        params = ModelParameters(
            c=3, K=10, b=1, mu=1.0, alpha=0.5, beta=0.1,
            C0=C0, C1=C1,
        )
        Q = build_generator(params)
        pi = solve_stationary(Q)

        # 位相マージナル
        pi_F = np.zeros(2)
        for idx in range(params.N):
            j, rem = divmod(idx, params.D)
            i, F = divmod(rem, params.D_M)
            pi_F[F] += pi[idx]

        # 対称性より 0.5, 0.5
        np.testing.assert_allclose(pi_F, [0.5, 0.5], atol=1e-10)


class TestMonotoneParametric:
    """パラメータ変化に対する単調性 (定性的テスト)."""

    def _build_and_solve(self, c, K, b, mu, alpha, beta, lam):
        C0 = np.array([[-lam]])
        C1 = np.array([[lam]])
        params = ModelParameters(
            c=c, K=K, b=b, mu=mu, alpha=alpha, beta=beta,
            C0=C0, C1=C1,
        )
        Q = build_generator(params)
        pi = solve_stationary(Q)
        return Metrics(params, pi), params

    def test_higher_lambda_higher_blocking(self):
        """到着率が上がると P_block が (弱) 単調非減少."""
        lambdas = [0.2, 0.5, 0.8]
        P_blocks = []
        for lam in lambdas:
            m, _ = self._build_and_solve(
                c=2, K=8, b=1, mu=1.0, alpha=1.0, beta=0.1, lam=lam,
            )
            P_blocks.append(m.blocking_probability())
        # 単調非減少
        for i in range(len(P_blocks) - 1):
            assert P_blocks[i] <= P_blocks[i + 1] + 1e-10

    def test_higher_lambda_higher_utilization(self):
        """到着率が上がると ρ が (弱) 単調非減少."""
        lambdas = [0.2, 0.5, 0.8]
        rhos = []
        for lam in lambdas:
            m, _ = self._build_and_solve(
                c=2, K=8, b=1, mu=1.0, alpha=1.0, beta=0.1, lam=lam,
            )
            rhos.append(m.utilization())
        for i in range(len(rhos) - 1):
            assert rhos[i] <= rhos[i + 1] + 1e-10

    def test_higher_alpha_lower_blocking(self):
        """setup 完了率が上がると (バッファに滞留する時間が減り) P_block が下がる.

        直感: alpha 大 → setup 早く終わる → サーバー稼働率上がる → 消化早い
             → 満杯確率下がる.
        """
        alphas = [0.1, 1.0, 10.0]
        P_blocks = []
        for alpha in alphas:
            m, _ = self._build_and_solve(
                c=2, K=8, b=1, mu=1.0, alpha=alpha, beta=0.1, lam=0.7,
            )
            P_blocks.append(m.blocking_probability())
        # 単調非増加
        for i in range(len(P_blocks) - 1):
            assert P_blocks[i] >= P_blocks[i + 1] - 1e-8


class TestNumericalStability:
    """数値安定性のテスト."""

    def test_high_dynamic_range(self):
        """動的レンジが大きくても sparse solver が動く.

        大きな K で pi_K が極端に小さくなる状況.
        """
        lam = 0.3
        mu = 1.0
        C0 = np.array([[-lam]])
        C1 = np.array([[lam]])
        params = ModelParameters(
            c=1, K=30, b=1, mu=mu,
            alpha=1000.0, beta=1e-6,
            C0=C0, C1=C1,
        )
        Q = build_generator(params)
        pi = solve_stationary(Q, tol=1e-6)
        # 検証は解けたことのみ
        assert abs(pi.sum() - 1.0) < 1e-8
        assert (pi >= -1e-8).all()

    def test_multiple_calls_reproducible(self, small_params):
        """複数回呼び出しても同じ結果."""
        Q = build_generator(small_params)
        pi1 = solve_stationary(Q)
        pi2 = solve_stationary(Q)
        np.testing.assert_array_equal(pi1, pi2)
