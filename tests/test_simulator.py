"""mmpp_sim (離散事象シミュレーション) のテスト.

検証項目:
    - 単一状態でのイベントレートの正しさ (シード固定で期待値と統計的に一致)
    - 状態遷移の正しさ (バッチサービス完了, セットアップ, Delayoff 等)
    - ウォームアップ期間が正しく除外されること
    - 保存則: E[B] + E[I] + E[S] + E[Off] = c
    - 流量バランス: lambda_eff = b * mu * E[B]
    - 小規模ケースでの理論解析との CI 内一致
    - 再現性 (同一シードで同一結果)
"""
from collections import defaultdict

import numpy as np
import pytest

from mmpp import ModelParameters, build_generator, solve_stationary, Metrics
from mmpp_sim import EventType, Simulator, SimMetrics


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


class TestWarmupExclusion:
    """ウォームアップ期間の統計が結果に含まれないことの検証."""

    @pytest.mark.parametrize("warmup_events", [0, 1_000, 10_000, 50_000])
    def test_recorded_count_matches_measurement_events(
        self, rate_params, warmup_events
    ):
        """記録されるイベント数は warmup_events に依らず常に measurement_events."""
        measurement_events = 5_000
        sim = Simulator(rate_params, seed=1)
        stats = sim.run(
            warmup_events=warmup_events,
            measurement_events=measurement_events,
            n_batches=5,
        )
        assert stats.n_records == measurement_events


class TestConservationLaw:
    """保存則: E[B] + E[I] + E[S] + E[Off] が c と (ほぼ) 一致すること."""

    def test_server_conservation(self, rate_params):
        sim = Simulator(rate_params, seed=11)
        stats = sim.run(warmup_events=5_000, measurement_events=50_000, n_batches=20)
        m = SimMetrics(rate_params, stats)

        # バッチごとの合計値のばらつきから sigma を推定
        batch_totals = []
        for b in stats.batches:
            bm = SimMetrics(rate_params, b)
            batch_totals.append(
                bm.mean_busy() + bm.mean_idle() + bm.mean_setup() + bm.mean_off()
            )
        batch_totals = np.array(batch_totals)
        se = batch_totals.std(ddof=1) / np.sqrt(len(batch_totals))
        sigma = max(se, 1e-12)

        total = m.mean_busy() + m.mean_idle() + m.mean_setup() + m.mean_off()
        assert abs(total - rate_params.c) < 3 * sigma + 1e-9


class TestFlowBalance:
    """流量バランス: lambda_eff と b*mu*E[B] が (統計的に) 一致すること."""

    def test_flow_balance(self, rate_params):
        p = rate_params
        sim = Simulator(p, seed=22)
        stats = sim.run(warmup_events=5_000, measurement_events=80_000, n_batches=20)

        diffs = []
        for b in stats.batches:
            bm = SimMetrics(p, b)
            diffs.append(bm.effective_arrival_rate() - p.b * p.mu * bm.mean_busy())
        diffs = np.array(diffs)
        mean_diff = diffs.mean()
        se = diffs.std(ddof=1) / np.sqrt(len(diffs))
        assert abs(mean_diff) < 3 * se + 1e-9


class TestTheoryComparison:
    """小規模ケース (c=1, K=3, D_M=1) での理論解析との CI 内一致."""

    def test_matches_theory_within_ci(self, tiny_params):
        p = tiny_params
        Q = build_generator(p)
        pi = solve_stationary(Q)
        theory = Metrics(p, pi)

        sim = Simulator(p, seed=99)
        stats = sim.run(warmup_events=10_000, measurement_events=300_000, n_batches=30)
        sim_metrics = SimMetrics(p, stats)

        checks = [
            ("P_block", theory.blocking_probability(), sim_metrics.blocking_probability_ci),
            ("E[j]", theory.mean_queue_length(), sim_metrics.mean_queue_length_ci),
            ("rho", theory.utilization(), sim_metrics.utilization_ci),
            ("E[W]", theory.mean_waiting_time(), sim_metrics.mean_waiting_time_ci),
        ]
        n_ok = 0
        for name, theory_val, ci_fn in checks:
            _, lo, hi = ci_fn()
            if lo <= theory_val <= hi:
                n_ok += 1
        assert n_ok >= 3, "少なくとも 4 指標中 3 指標が理論値を CI 内に含むべき"


class TestReproducibility:
    """同一シードでの再現性."""

    def test_same_seed_gives_same_result(self, rate_params):
        sim1 = Simulator(rate_params, seed=42)
        stats1 = sim1.run(warmup_events=1_000, measurement_events=5_000, n_batches=5)

        sim2 = Simulator(rate_params, seed=42)
        stats2 = sim2.run(warmup_events=1_000, measurement_events=5_000, n_batches=5)

        m1 = SimMetrics(rate_params, stats1).all_metrics()
        m2 = SimMetrics(rate_params, stats2).all_metrics()
        for key in m1:
            assert m1[key] == m2[key]
