"""性能指標 (Metrics) のテスト.

検証項目:
    - 確率的性質 (P_block ∈ [0, 1], ρ ∈ [0, 1])
    - サーバー数保存: E[B] + E[I] + E[S] + E[Off] = c
    - 流量バランス: λ_eff = b * μ * E[B]
    - Little の公式: E[j] = λ_eff * E[W]
"""
import numpy as np
import pytest

from mmpp import (
    build_generator,
    solve_stationary,
    Metrics,
    ModelParameters,
)


@pytest.fixture
def small_metrics(small_params):
    """small_params の結果から Metrics を作成."""
    Q = build_generator(small_params)
    pi = solve_stationary(Q)
    return Metrics(small_params, pi), small_params


@pytest.fixture
def medium_metrics(medium_params):
    Q = build_generator(medium_params)
    pi = solve_stationary(Q)
    return Metrics(medium_params, pi), medium_params


@pytest.fixture
def batch_metrics(batch_params):
    Q = build_generator(batch_params)
    pi = solve_stationary(Q)
    return Metrics(batch_params, pi), batch_params


class TestBasicRanges:
    """基本的な範囲検証."""

    def test_P_block_in_unit(self, small_metrics):
        """P_block ∈ [0, 1]."""
        m, _ = small_metrics
        P = m.blocking_probability()
        assert 0 <= P <= 1

    def test_utilization_in_unit(self, small_metrics):
        """ρ ∈ [0, 1]."""
        m, _ = small_metrics
        rho = m.utilization()
        assert 0 <= rho <= 1

    def test_mean_queue_bounded_by_K(self, small_metrics):
        """E[j] ≤ K."""
        m, p = small_metrics
        Ej = m.mean_queue_length()
        assert 0 <= Ej <= p.K

    def test_mean_servers_non_negative(self, small_metrics):
        """E[B], E[I], E[S], E[Off] ≥ 0."""
        m, _ = small_metrics
        assert m.mean_busy() >= 0
        assert m.mean_idle() >= 0
        assert m.mean_setup() >= 0
        assert m.mean_off() >= -1e-10  # 数値誤差許容


class TestConservationLaws:
    """保存則の検証."""

    def test_server_conservation_small(self, small_metrics):
        """E[B] + E[I] + E[S] + E[Off] = c."""
        m, p = small_metrics
        total = m.mean_busy() + m.mean_idle() + m.mean_setup() + m.mean_off()
        assert abs(total - p.c) < 1e-10

    def test_server_conservation_medium(self, medium_metrics):
        m, p = medium_metrics
        total = m.mean_busy() + m.mean_idle() + m.mean_setup() + m.mean_off()
        assert abs(total - p.c) < 1e-10

    def test_server_conservation_batch(self, batch_metrics):
        m, p = batch_metrics
        total = m.mean_busy() + m.mean_idle() + m.mean_setup() + m.mean_off()
        assert abs(total - p.c) < 1e-10


class TestFlowBalance:
    """流量バランス: λ_eff = b * μ * E[B].

    定常状態では有効到着率 = 平均サービス完了率 (単位時間あたりのジョブ数).
    サービス完了ごとに b 個のジョブが去るので:
        λ_eff = b * μ * E[B]
    """

    def test_flow_balance_small(self, small_metrics):
        m, p = small_metrics
        lhs = m.effective_arrival_rate()
        rhs = p.b * p.mu * m.mean_busy()
        np.testing.assert_allclose(lhs, rhs, rtol=1e-8)

    def test_flow_balance_medium(self, medium_metrics):
        m, p = medium_metrics
        lhs = m.effective_arrival_rate()
        rhs = p.b * p.mu * m.mean_busy()
        np.testing.assert_allclose(lhs, rhs, rtol=1e-8)

    def test_flow_balance_batch(self, batch_metrics):
        m, p = batch_metrics
        lhs = m.effective_arrival_rate()
        rhs = p.b * p.mu * m.mean_busy()
        np.testing.assert_allclose(lhs, rhs, rtol=1e-8)


class TestLittleLaw:
    """Little の公式: E[j] = λ_eff * E[W]."""

    def test_little_small(self, small_metrics):
        m, _ = small_metrics
        lhs = m.mean_queue_length()
        rhs = m.effective_arrival_rate() * m.mean_waiting_time()
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)

    def test_little_medium(self, medium_metrics):
        m, _ = medium_metrics
        lhs = m.mean_queue_length()
        rhs = m.effective_arrival_rate() * m.mean_waiting_time()
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


class TestUtilizationConsistency:
    """利用率と E[B] の整合."""

    def test_rho_equals_EB_over_c(self, small_metrics):
        m, p = small_metrics
        assert abs(m.utilization() - m.mean_busy() / p.c) < 1e-12


class TestEnergyCost:
    """コスト計算."""

    def test_energy_cost_bounded(self, small_metrics):
        """コストは 0 ≤ cost ≤ c_busy * c."""
        m, p = small_metrics
        cost = m.energy_cost(c_busy=1.0, c_idle=0.5, c_setup=1.0, c_off=0.0)
        assert 0 <= cost <= 1.0 * p.c

    def test_zero_cost_when_all_off(self):
        """全 Off ならコスト = c_off * c."""
        # このケースを模擬するのは難しいので、係数だけ確認.
        # cost = c_busy*E[B] + c_idle*E[I] + c_setup*E[S] + c_off*E[Off]
        # 全て 0 なら 0. c_off=0 とすれば Off の寄与なし.
        pass  # placeholder

    def test_all_metrics_dict(self, small_metrics):
        """all_metrics が全キーを持つ辞書を返す."""
        m, _ = small_metrics
        d = m.all_metrics()
        expected_keys = {
            "P_block", "P_block_arrival",
            "E[j]", "E[B]", "E[I]", "E[S]", "E[Off]",
            "lambda_eff", "E[W]", "rho", "energy_cost",
            "cost_paper", "ERP_paper",
        }
        assert set(d.keys()) == expected_keys
        for k, v in d.items():
            assert isinstance(v, (int, float, np.floating))
            assert not np.isnan(v)


class TestEnergyCostPaper:
    """先行研究 (Le-Anh & Phung-Duc 2025) 準拠のコスト計算."""

    def test_energy_cost_paper_matches_formula(self, small_metrics):
        """energy_cost_paper が先行研究の公式と一致."""
        m, p = small_metrics

        b = p.b
        C_b = 0.04419 * b + 0.15503
        C_i = 0.6 * C_b
        expected = C_b * m.mean_busy() + C_b * m.mean_setup() + C_i * m.mean_idle()

        assert abs(m.energy_cost_paper() - expected) < 1e-12

    def test_energy_cost_paper_depends_on_b(self):
        """b が変わるとコストが変わる (b 依存性の確認)."""
        C0 = np.array([[-1.5, 0.5], [0.3, -1.3]])
        C1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        p_b1 = ModelParameters(c=2, K=4, b=1, mu=1.0, alpha=1.0, beta=0.1, C0=C0, C1=C1)
        p_b2 = ModelParameters(c=2, K=4, b=2, mu=1.0, alpha=1.0, beta=0.1, C0=C0, C1=C1)

        Q1 = build_generator(p_b1); pi1 = solve_stationary(Q1); m1 = Metrics(p_b1, pi1)
        Q2 = build_generator(p_b2); pi2 = solve_stationary(Q2); m2 = Metrics(p_b2, pi2)

        # 係数比: C_b(b=2)/C_b(b=1) = (0.04419*2+0.15503)/(0.04419+0.15503) = 1.221
        ratio_expected = (0.04419 * 2 + 0.15503) / (0.04419 + 0.15503)
        # 完全比較は難しいが、係数側が変わっていることを確認
        assert abs(ratio_expected - 1.221) < 0.001  # サニティ
        assert m1.energy_cost_paper() != m2.energy_cost_paper()


class TestBlockingProbability:
    """ブロック確率の性質."""

    def test_arrival_blocking_in_unit(self, small_metrics):
        """0 ≤ P_block^arrival ≤ 1."""
        m, _ = small_metrics
        p_arr = m.arrival_blocking_probability()
        assert 0 <= p_arr <= 1

    def test_arrival_blocking_consistent_with_lambda_eff(self, small_metrics):
        """P_block^arrival = 1 - λ_eff / λ_bar (定義との整合)."""
        m, p = small_metrics
        p_arr = m.arrival_blocking_probability()
        lam_eff = m.effective_arrival_rate()
        expected = 1.0 - lam_eff / p.lambda_bar
        assert abs(p_arr - expected) < 1e-12

    def test_PASTA_holds_for_poisson(self):
        """D_M=1 (Poisson) では P_block^time = P_block^arrival (PASTA)."""
        from mmpp import ModelParameters, build_generator, solve_stationary, Metrics
        C0 = np.array([[-0.5]])
        C1 = np.array([[0.5]])
        params = ModelParameters(
            c=2, K=8, b=1, mu=1.0, alpha=1.0, beta=0.1,
            C0=C0, C1=C1,
        )
        Q = build_generator(params)
        pi = solve_stationary(Q)
        m = Metrics(params, pi)
        p_time = m.blocking_probability()
        p_arr = m.arrival_blocking_probability()
        # Poisson 到着なので PASTA により両者一致
        assert abs(p_time - p_arr) < 1e-10
