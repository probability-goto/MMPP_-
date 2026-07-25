"""mmpp_sim (離散事象シミュレーション) のテスト.

検証項目:
    - 単一状態でのイベントレートの正しさ (シード固定で期待値と統計的に一致)
    - 状態遷移の正しさ (バッチサービス完了, セットアップ, Delayoff 等)
    - 再現性 (同一シードで同一結果)
    - 独立レプリケーション (異なるシードで異なる結果, run_replications の基本動作)
    - 保存則: E[B] + E[I] + E[S] + E[Off] = c
    - 流量バランス: lambda_eff = b * mu * E[B]
    - 単一 SimStats では CI 計算がエラーになること (SimMetrics の型分岐)
    - レプリケーションから信頼区間 (CI) が計算できること
    - 小規模ケースでの理論解析との CI 内一致
"""
from collections import defaultdict

import numpy as np
import pytest

from mmpp import ModelParameters, build_generator, solve_stationary, Metrics
from mmpp_sim import EventType, Simulator, SimMetrics, run_replications


@pytest.fixture
def small_params():
    """独立レプリケーション/CI 系テスト向けの小規模パラメータ (D_M=2)."""
    C0 = np.array([[-1.5, 0.1], [0.2, -2.0]])
    C1 = np.array([[1.4, 0.0], [0.0, 1.8]])
    return ModelParameters(
        c=2, K=5, b=1, mu=1.0, alpha=1.0, beta=0.3,
        C0=C0, C1=C1,
    )


@pytest.fixture
def tiny_params():
    """理論解析と比較する極小ケース (c=1, K=3, D_M=1)."""
    C0 = np.array([[-0.7]])
    C1 = np.array([[0.7]])
    return ModelParameters(
        c=1, K=3, b=1, mu=1.0, alpha=1.0, beta=0.3,
        C0=C0, C1=C1,
    )


@pytest.fixture
def rate_params():
    """イベントレート/状態遷移検証用パラメータ (D_M=1, c=5, K=10)."""
    C0 = np.array([[-0.6]])
    C1 = np.array([[0.6]])
    return ModelParameters(
        c=5, K=10, b=1, mu=1.0, alpha=1.0, beta=1.0,
        C0=C0, C1=C1,
    )


class TestEventRates:
    """単一状態から発生するイベントのレートが正しいことの統計的検証."""

    def _sample_from_state(self, params, state, n_trials, seed=123):
        """state に強制リセットしながら n_trials 回 step() し, 結果を集計する."""
        sim = Simulator(params, seed=seed)
        dts = []
        event_counts = defaultdict(int)
        _, j, _ = state
        for _ in range(n_trials):
            sim.state = state
            sim.time = 0.0
            sim.fifo.clear()
            sim.fifo.extend([0.0] * j)
            result = sim.step()
            dts.append(result.dt)
            event_counts[result.event_type] += 1
        return np.array(dts), event_counts

    def test_rates_with_idle_no_setup(self, rate_params):
        """i=3, j=2: B=2, I=1, S=0 の状態でのレート検証 (ARRIVAL/SERVICE/DELAYOFF)."""
        p = rate_params
        state = (3, 2, 0)
        n = 20_000
        dts, counts = self._sample_from_state(p, state, n)

        lam = p.lambdas[0]
        B, I = 2, 1
        rate_service = B * p.mu
        rate_delayoff = I * p.beta
        total_rate = lam + rate_service + rate_delayoff

        # 平均待ち時間の検証 (4 sigma 以内)
        mean_dt = dts.mean()
        se_dt = (1.0 / total_rate) / np.sqrt(n)
        assert abs(mean_dt - 1.0 / total_rate) < 4 * se_dt

        # イベント種別の発生比率の検証
        expected_probs = {
            EventType.ARRIVAL: lam / total_rate,
            EventType.SERVICE_COMPLETION: rate_service / total_rate,
            EventType.DELAYOFF_TIMEOUT: rate_delayoff / total_rate,
        }
        assert EventType.SETUP_COMPLETION not in counts
        for ev, p_expected in expected_probs.items():
            p_hat = counts[ev] / n
            se = np.sqrt(p_expected * (1 - p_expected) / n)
            assert abs(p_hat - p_expected) < 4 * se

    def test_rates_with_setup_no_idle(self, rate_params):
        """i=1, j=5: B=1, I=0, S=4 の状態でのレート検証 (ARRIVAL/SERVICE/SETUP)."""
        p = rate_params
        state = (1, 5, 0)
        n = 20_000
        dts, counts = self._sample_from_state(p, state, n)

        lam = p.lambdas[0]
        B, S = 1, 4
        rate_service = B * p.mu
        rate_setup = S * p.alpha
        total_rate = lam + rate_service + rate_setup

        mean_dt = dts.mean()
        se_dt = (1.0 / total_rate) / np.sqrt(n)
        assert abs(mean_dt - 1.0 / total_rate) < 4 * se_dt

        expected_probs = {
            EventType.ARRIVAL: lam / total_rate,
            EventType.SERVICE_COMPLETION: rate_service / total_rate,
            EventType.SETUP_COMPLETION: rate_setup / total_rate,
        }
        assert EventType.DELAYOFF_TIMEOUT not in counts
        for ev, p_expected in expected_probs.items():
            p_hat = counts[ev] / n
            se = np.sqrt(p_expected * (1 - p_expected) / n)
            assert abs(p_hat - p_expected) < 4 * se


class TestStateTransitions:
    """状態遷移の正しさ (各イベント種別ごとの状態変化)."""

    def _run_many(self, params, state, n_trials, seed=7):
        sim = Simulator(params, seed=seed)
        _, j, _ = state
        results = []
        for _ in range(n_trials):
            sim.state = state
            sim.time = 0.0
            sim.fifo.clear()
            sim.fifo.extend([0.0] * j)
            results.append(sim.step())
        return results

    def test_service_completion_decrements_j_by_b(self):
        """バッチサービス完了で j が b 減り, i と F は不変."""
        C0 = np.array([[-0.5, 0.2], [0.2, -0.5]])
        C1 = np.array([[0.3, 0.0], [0.0, 0.3]])
        params = ModelParameters(
            c=4, K=12, b=3, mu=1.0, alpha=0.5, beta=0.5, C0=C0, C1=C1,
        )
        state = (3, 6, 1)  # B = min(3, 6//3) = 2 > 0
        results = self._run_many(params, state, 3_000)
        found = False
        for r in results:
            if r.event_type == EventType.SERVICE_COMPLETION:
                found = True
                i0, j0, F0 = r.old_state
                i1, j1, F1 = r.new_state
                assert j1 == j0 - params.b
                assert i1 == i0
                assert F1 == F0
                assert r.departure_count == params.b
        assert found

    def test_setup_completion_increments_i(self):
        """セットアップ完了で i が 1 増え, j と F は不変."""
        C0 = np.array([[-0.5]])
        C1 = np.array([[0.5]])
        params = ModelParameters(
            c=5, K=10, b=1, mu=1.0, alpha=1.0, beta=0.5, C0=C0, C1=C1,
        )
        state = (1, 5, 0)  # S = min(5,5) - 1 = 4 > 0
        results = self._run_many(params, state, 3_000)
        found = False
        for r in results:
            if r.event_type == EventType.SETUP_COMPLETION:
                found = True
                i0, j0, F0 = r.old_state
                i1, j1, F1 = r.new_state
                assert i1 == i0 + 1
                assert j1 == j0
                assert F1 == F0
        assert found

    def test_delayoff_decrements_i(self):
        """Delayoff タイムアウトで i が 1 減り, j と F は不変."""
        C0 = np.array([[-0.5]])
        C1 = np.array([[0.5]])
        params = ModelParameters(
            c=5, K=10, b=1, mu=1.0, alpha=1.0, beta=0.5, C0=C0, C1=C1,
        )
        state = (3, 2, 0)  # I = 3 - min(3,2) = 1 > 0
        results = self._run_many(params, state, 3_000)
        found = False
        for r in results:
            if r.event_type == EventType.DELAYOFF_TIMEOUT:
                found = True
                i0, j0, F0 = r.old_state
                i1, j1, F1 = r.new_state
                assert i1 == i0 - 1
                assert j1 == j0
                assert F1 == F0
        assert found

    def test_arrival_admitted_increments_j(self):
        """j < K での到着は j を 1 増やし, ブロックされない."""
        C0 = np.array([[-0.5]])
        C1 = np.array([[0.5]])
        params = ModelParameters(
            c=3, K=5, b=1, mu=1.0, alpha=1.0, beta=0.5, C0=C0, C1=C1,
        )
        state = (1, 2, 0)
        results = self._run_many(params, state, 3_000)
        found = False
        for r in results:
            if r.event_type == EventType.ARRIVAL:
                found = True
                assert r.blocked is False
                i0, j0, F0 = r.old_state
                i1, j1, F1 = r.new_state
                assert j1 == j0 + 1
                assert i1 == i0
        assert found

    def test_arrival_blocked_at_K(self):
        """j = K での到着はブロックされ, j は増えない."""
        C0 = np.array([[-0.5]])
        C1 = np.array([[0.5]])
        params = ModelParameters(
            c=3, K=5, b=1, mu=1.0, alpha=1.0, beta=0.5, C0=C0, C1=C1,
        )
        state = (2, 5, 0)  # j = K
        results = self._run_many(params, state, 3_000)
        found = False
        for r in results:
            if r.event_type == EventType.ARRIVAL:
                found = True
                assert r.blocked is True
                i0, j0, F0 = r.old_state
                i1, j1, F1 = r.new_state
                assert j1 == j0 == params.K
                assert i1 == i0
        assert found


def test_simulator_reproducibility(small_params):
    """同一シードで 2 回実行して同じ結果になる."""
    sim1 = Simulator(small_params, seed=42)
    stats1 = sim1.run(warmup_events=1000, measurement_events=5000)
    sim2 = Simulator(small_params, seed=42)
    stats2 = sim2.run(warmup_events=1000, measurement_events=5000)
    assert stats1.total_duration == stats2.total_duration
    assert stats1.arrival_attempts == stats2.arrival_attempts

    m1 = SimMetrics(small_params, stats1).all_metrics()
    m2 = SimMetrics(small_params, stats2).all_metrics()
    for key in m1:
        assert m1[key] == m2[key]


def test_replications_different_seeds(small_params):
    """異なるシードで異なる結果になる."""
    reps = run_replications(
        small_params, n_reps=3, seed0=42,
        warmup_events=500, measurement_events=2000,
    )
    durations = [r.total_duration for r in reps]
    assert len(set(durations)) == 3  # 全部異なる


def test_server_conservation(small_params):
    """E[B] + E[I] + E[S] + E[Off] = c (シミュレーション側でも)."""
    reps = run_replications(
        small_params, n_reps=5, seed0=42,
        warmup_events=1000, measurement_events=10000,
    )
    m = SimMetrics(small_params, reps)
    total = m.mean_busy() + m.mean_idle() + m.mean_setup() + m.mean_off()
    assert abs(total - small_params.c) < 1e-10


def test_flow_balance(small_params):
    """λ_eff ≈ b * μ * E[B] (定常時)."""
    reps = run_replications(
        small_params, n_reps=10, seed0=42,
        warmup_events=5000, measurement_events=50000,
    )
    m = SimMetrics(small_params, reps)
    lhs = m.effective_arrival_rate()
    rhs = small_params.b * small_params.mu * m.mean_busy()
    # 有限サンプルなので緩めのトレランス
    assert abs(lhs - rhs) / max(abs(lhs), 1e-10) < 0.05


def test_ci_requires_replications(small_params):
    """単一 SimStats では CI 計算がエラー."""
    sim = Simulator(small_params, seed=42)
    stats = sim.run(warmup_events=500, measurement_events=2000)
    m = SimMetrics(small_params, stats)
    with pytest.raises(RuntimeError, match="レプリケーション"):
        m.blocking_probability_ci()


def test_ci_from_replications(small_params):
    """レプリケーションから CI が計算できる."""
    reps = run_replications(
        small_params, n_reps=10, seed0=42,
        warmup_events=1000, measurement_events=10000,
    )
    m = SimMetrics(small_params, reps)
    pt, lo, hi = m.blocking_probability_ci()
    assert lo <= pt <= hi
    assert 0 <= pt <= 1


def test_theory_within_ci_low_load(small_params):
    """理論値がシミュレーションの CI 内に (おおむね) 入る.

    95% CI なので指標ごとに約 5% の確率で理論値を外れうる. 単一指標の
    際どい外れでテストが不安定化しないよう, 4 指標中 3 指標以上が
    CI 内に収まることを要求する (旧 batch means 版のテストと同じ方針).
    """
    Q = build_generator(small_params)
    pi = solve_stationary(Q)
    theory = Metrics(small_params, pi)

    reps = run_replications(
        small_params, n_reps=20, seed0=42,
        warmup_events=10000, measurement_events=100000,
    )
    sim = SimMetrics(small_params, reps)

    methods = ["blocking_probability", "mean_queue_length", "utilization", "mean_waiting_time"]
    n_ok = 0
    for method in methods:
        theory_val = getattr(theory, method)()
        _, lo, hi = getattr(sim, f"{method}_ci")()
        if lo <= theory_val <= hi:
            n_ok += 1
    assert n_ok >= 3, f"少なくとも {len(methods)} 指標中 3 指標が理論値を CI 内に含むべき"
