"""事前セットアップの保護 (protect_presetup) と P1 の診断指標のテスト.

- protect_presetup=False (既定) で既存の Predictive の結果が変わらないこと
- protect_presetup=True でも n_target=0, gamma=1 でベースモデルに一致すること
- protect_presetup=True でバースト中に i+s < n_target となる確率が下がること
- 診断指標 (p1_fire_rate, setup_cancel_rate 等) の恒等式と DES との一致
"""
import csv
import os

import numpy as np
import pytest
from scipy import stats as scipy_stats

from mmpp.generator import setup_servers as compute_required_s
from mmpp.model import ModelParameters
from mmpp.generator import build_generator as build_base_generator
from mmpp.solver import solve_stationary as solve_base
from mmpp.metrics import Metrics as BaseMetrics

from mmpp_predictive import (
    PredictiveModelParameters, build_generator, solve_stationary, Metrics,
)
from mmpp_predictive.state_space import (
    build_is_index, pi_by_level, state_to_idx,
)
from mmpp_predictive_sim import PredictiveSimulator
from mmpp_predictive_sim.event import EventType

try:
    from scripts._mmpp_burst import build_mmpp
except ImportError:
    from _mmpp_burst import build_mmpp


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_params(c=6, K=30, b=2, rho=0.7, delta=0.6, sigma=0.1,
                alpha=0.1, beta=0.005, n_target=4, gamma=5.0,
                protect_presetup=False):
    C0, C1 = build_mmpp(rho, delta, sigma, c, b, 1.0)
    return PredictiveModelParameters(
        c=c, K=K, b=b, mu=1.0, alpha=alpha, beta=beta, C0=C0, C1=C1,
        n_target=n_target, gamma=gamma, protect_presetup=protect_presetup,
    )


def solve(params):
    pi = solve_stationary(build_generator(params), solver="gth")
    return pi, Metrics(params, pi)


# ============================================================
# 1. protect_presetup=False で既存の結果が変わらない
# ============================================================

# 変更前のコード (protect_presetup 導入前, コミット a4075fa) で計算した値.
REFERENCE_POINTS = [
    (
        dict(c=5, K=20, b=2, rho=0.7, delta=0.6, sigma=0.1, alpha=0.1,
             beta=0.005, n_target=3, gamma=5.0),
        {"P_block_arrival_stable": 0.1439269767892233,
         "E[W]": 1.5856082487546423, "cost_paper": 0.9788363636850794,
         "ERP_paper": 1.552051012440061, "E[S]": 0.2792714229485006,
         "E[j]": 9.501775129975327, "E[I]": 1.2430353148741604},
    ),
    (
        dict(c=6, K=30, b=2, rho=0.5, delta=0.9, sigma=0.01, alpha=1.0,
             beta=0.05, n_target=6, gamma=1.0),
        {"P_block_arrival_stable": 0.038634797953834704,
         "E[W]": 1.5855295949067572, "cost_paper": 0.8553102380994296,
         "ERP_paper": 1.3561196953333907, "E[S]": 0.04844391280595212,
         "E[j]": 9.145637876146251, "E[I]": 0.968878256119042},
    ),
    (
        dict(c=4, K=10, b=1, rho=0.3, delta=0.3, sigma=1.0, alpha=10.0,
             beta=0.2, n_target=2, gamma=3.0),
        {"P_block_arrival_stable": 4.244581761131609e-05,
         "E[W]": 1.0404457278686652, "cost_paper": 0.37290319511542347,
         "ERP_paper": 0.38798553626641763, "E[S]": 0.040887118930610644,
         "E[j]": 1.2484818783588787, "E[I]": 1.051633123757727},
    ),
]


class TestDefaultUnchanged:

    def test_default_is_false(self):
        params = make_params()
        assert params.protect_presetup is False

    @pytest.mark.parametrize("kw,expected", REFERENCE_POINTS)
    def test_reference_points(self, kw, expected):
        _, m = solve(make_params(protect_presetup=False, **kw))
        d = m.all_metrics()
        for key, ref in expected.items():
            assert d[key] == pytest.approx(ref, rel=1e-12), key

    def test_standard_point_matches_experiment_5_csv(self):
        """標準設定 (medium, rho=0.7, alpha=0.1, n_target=10, gamma=5) が既存 CSV と一致."""
        path = os.path.join(REPO_ROOT, "results", "experiment_5_medium_g5.0.csv")
        if not os.path.exists(path):
            pytest.skip("results/experiment_5_medium_g5.0.csv がない")
        with open(path, newline="", encoding="utf-8") as f:
            row = next(r for r in csv.DictReader(f)
                       if float(r["alpha"]) == 0.1 and int(r["n_target"]) == 10)
        params = make_params(c=20, K=200, b=5, rho=0.7, delta=0.6, sigma=0.1,
                             alpha=0.1, beta=0.005, n_target=10, gamma=5.0)
        _, m = solve(params)
        assert m.P_block_arrival_stable == pytest.approx(
            float(row["P_block_arrival_stable"]), rel=1e-10)
        assert m.mean_waiting_time() == pytest.approx(float(row["E_W"]), rel=1e-10)
        assert m.energy_cost_paper() == pytest.approx(float(row["Cost"]), rel=1e-10)
        assert m.erp_paper() == pytest.approx(float(row["ERP"]), rel=1e-10)


# ============================================================
# 2. protect_presetup=True でも n_target=0, gamma=1 でベースに一致
# ============================================================

@pytest.mark.parametrize("rho", [0.3, 0.5, 0.7, 0.9])
def test_protect_reduces_to_base(rho):
    c, K, b, delta, sigma = 5, 20, 2, 0.6, 0.1
    C0, C1 = build_mmpp(rho, delta, sigma, c, b, 1.0)
    base_p = ModelParameters(c=c, K=K, b=b, mu=1.0, alpha=0.1, beta=0.005,
                             C0=C0, C1=C1)
    m_base = BaseMetrics(base_p, solve_base(build_base_generator(base_p), solver="gth"))
    pred_p = make_params(c=c, K=K, b=b, rho=rho, delta=delta, sigma=sigma,
                         n_target=0, gamma=1.0, protect_presetup=True)
    _, m_pred = solve(pred_p)

    for method in ["arrival_blocking_probability_stable", "mean_waiting_time",
                   "energy_cost_paper", "erp_paper", "mean_setup",
                   "mean_queue_length", "mean_idle"]:
        a = getattr(m_base, method)()
        p = getattr(m_pred, method)()
        assert abs(a - p) / abs(a) < 1e-10, f"{method}: base={a} pred={p}"


def test_protect_generator_identical_when_n_target_zero():
    """n_target=0 では protect_presetup の有無で Q が完全に同じ."""
    Q_off = build_generator(make_params(n_target=0, protect_presetup=False))
    Q_on = build_generator(make_params(n_target=0, protect_presetup=True))
    assert (Q_off != Q_on).nnz == 0


# ============================================================
# 3. 保護の効果: 遷移規則と F=1 で i+s < n_target の確率
# ============================================================

class TestProtectRule:
    """c=6, b=2, n_target=4. 状態 (i,s,j)=(1,3,2) からのバッチ完了 (j -> 0)."""

    def _service_dst(self, params, F):
        Q = build_generator(params).tocsr()
        _, pairs = build_is_index(params.c)
        is_index = {pair: k for k, pair in enumerate(pairs)}

        def idx(i, s, j, F_):
            return state_to_idx(i, s, j, F_, params.c, params.K, params.D_M, is_index)

        src = idx(1, 3, 2, F)
        keep = Q[src, idx(1, 3, 0, F)]
        cancel = Q[src, idx(1, 2, 0, F)]
        return keep, cancel

    def test_default_cancels_in_burst(self):
        keep, cancel = self._service_dst(make_params(protect_presetup=False), F=1)
        assert keep == 0 and cancel == pytest.approx(1.0)

    def test_protect_keeps_in_burst(self):
        keep, cancel = self._service_dst(make_params(protect_presetup=True), F=1)
        assert keep == pytest.approx(1.0) and cancel == 0

    def test_protect_cancels_in_normal_phase(self):
        keep, cancel = self._service_dst(make_params(protect_presetup=True), F=0)
        assert keep == 0 and cancel == pytest.approx(1.0)

    def test_protect_cancels_above_n_target(self):
        """i+s > n_target (n_target=3) では保護しない."""
        params = make_params(n_target=3, protect_presetup=True)
        keep, cancel = self._service_dst(params, F=1)
        assert keep == 0 and cancel == pytest.approx(1.0)

    @pytest.mark.parametrize("protect,F,expected_s", [
        (False, 1, 2), (True, 1, 3), (True, 0, 2),
    ])
    def test_des_same_rule(self, protect, F, expected_s):
        """DES のバッチ完了でも同じ規則になる."""
        params = make_params(protect_presetup=protect)
        sim = PredictiveSimulator(params, seed=3)
        n_checked = 0
        for _ in range(2000):
            sim.state = (1, 3, 2, F)
            sim.fifo.clear()
            sim.fifo.extend([0.0, 0.0])
            r = sim.step()
            if r.event_type == EventType.SERVICE_COMPLETION:
                assert r.new_state == (1, expected_s, 0, F)
                n_checked += 1
        assert n_checked > 0


def _burst_under_target_mass(params, pi):
    _, pairs = build_is_index(params.c)
    P = pi_by_level(pi, params.c, params.K, params.D_M)
    mask = np.array([i + s < params.n_target for i, s in pairs])
    return float(P[:, mask, 1].sum())


@pytest.mark.parametrize("kw", [
    dict(), dict(alpha=1.0), dict(alpha=10.0), dict(rho=0.3),
    dict(delta=0.9, sigma=0.01),
])
def test_protect_reduces_burst_under_target_probability(kw):
    pi_off, _ = solve(make_params(protect_presetup=False, **kw))
    pi_on, _ = solve(make_params(protect_presetup=True, **kw))
    p_off = _burst_under_target_mass(make_params(**kw), pi_off)
    p_on = _burst_under_target_mass(make_params(**kw), pi_on)
    assert p_off > 0
    assert p_on < p_off


# ============================================================
# 4. 診断指標
# ============================================================

def test_no_p1_when_n_target_zero():
    _, m = solve(make_params(n_target=0, protect_presetup=True))
    assert m.p1_fire_rate == 0.0
    assert m.p1_launch_rate == 0.0


def test_rho_B():
    _, m = solve(make_params(rho=0.7, delta=0.6))
    assert m.rho_B == pytest.approx(0.7 * 1.6, rel=1e-14)


@pytest.mark.parametrize("protect", [False, True])
@pytest.mark.parametrize("kw", [dict(), dict(rho=0.3, alpha=1.0)])
def test_setup_flow_balance(protect, kw):
    """セットアップの開始率 (反応的 + P1) = 完了率 + 取り消し率."""
    params = make_params(protect_presetup=protect, **kw)
    pi, m = solve(params)
    _, pairs = build_is_index(params.c)
    P = pi_by_level(pi, params.c, params.K, params.D_M)
    reactive = 0.0
    for k, (i, s) in enumerate(pairs):
        for j in range(params.K):
            if compute_required_s(i, j + 1, params.b, params.c) > s and s < params.c - i:
                reactive += float(P[j, k] @ params.lambdas)
    start = reactive + m.p1_launch_rate
    end = m.setup_completion_rate + m.setup_cancel_rate
    assert start == pytest.approx(end, rel=1e-10)
    assert m.p1_fire_rate <= m.p1_launch_rate
    assert 0 < m.p1_fire_rate <= params.C0[0, 1] * params.phase_stationary[0]


@pytest.mark.parametrize("protect", [False, True])
def test_p1_rates_match_des(protect):
    """DES の P1 発動回数・取り消し回数の率が理論値と 95% CI の範囲で一致."""
    params = make_params(c=4, K=10, b=2, rho=0.6, delta=0.6, sigma=0.3,
                         alpha=0.5, beta=0.2, n_target=3, gamma=2.0,
                         protect_presetup=protect)
    _, m = solve(params)

    fire, cancel = [], []
    for k in range(10):
        sim = PredictiveSimulator(params, seed=100 + k)
        st = sim.run(warmup_events=5000, measurement_events=50000)
        fire.append(sim.proactive_setup_count / st.total_duration)
        cancel.append(sim.setup_cancel_count / st.total_duration)

    for name, values, theory in [("p1_fire_rate", fire, m.p1_fire_rate),
                                 ("setup_cancel_rate", cancel, m.setup_cancel_rate)]:
        v = np.asarray(values)
        half = scipy_stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        assert abs(v.mean() - theory) <= half, (
            f"{name}: DES={v.mean():.5g}±{half:.3g}, theory={theory:.5g}")
