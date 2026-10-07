"""バースト中の Delayoff の保護 (protect_delayoff) のテスト.

- protect_delayoff=False で既存の結果 (results/p1_protect/p1_protect_v1.csv) が変わらないこと
- protect_presetup=protect_delayoff=True でも n_target=0, gamma=1 でベースモデルに一致すること
- 両方 True のとき, P1 発動直後の状態からバースト位相に留まる経路で
  i+s < n_target の状態に入らないこと
- DES と理論の p1_fire_rate, setup_cancel_rate の一致
"""
import csv
import os
from collections import deque

import numpy as np
import pytest
from scipy import stats as scipy_stats
from scipy.sparse.csgraph import breadth_first_order

from mmpp.model import ModelParameters
from mmpp.generator import build_generator as build_base_generator
from mmpp.solver import solve_stationary as solve_base
from mmpp.metrics import Metrics as BaseMetrics

from mmpp_predictive import (
    PredictiveModelParameters, build_generator, solve_stationary, Metrics,
)
from mmpp_predictive.state_space import (
    build_is_index, idx_to_state, state_to_idx, setup_target_delta,
)
from mmpp_predictive_sim import PredictiveSimulator
from mmpp_predictive_sim.event import EventType

try:
    from scripts._mmpp_burst import build_mmpp
except ImportError:
    from _mmpp_burst import build_mmpp


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODES = [(False, False), (True, False), (True, True)]


def make_params(c=6, K=30, b=2, rho=0.7, delta=0.6, sigma=0.1,
                alpha=0.1, beta=0.005, n_target=4, gamma=5.0,
                protect_presetup=False, protect_delayoff=False):
    C0, C1 = build_mmpp(rho, delta, sigma, c, b, 1.0)
    return PredictiveModelParameters(
        c=c, K=K, b=b, mu=1.0, alpha=alpha, beta=beta, C0=C0, C1=C1,
        n_target=n_target, gamma=gamma,
        protect_presetup=protect_presetup, protect_delayoff=protect_delayoff,
    )


def solve(params):
    pi = solve_stationary(build_generator(params), solver="gth")
    return pi, Metrics(params, pi)


# ============================================================
# 1. protect_delayoff=False で既存の結果が変わらない
# ============================================================

def test_default_is_false():
    assert make_params().protect_delayoff is False


V1_CSV = os.path.join(REPO_ROOT, "results", "p1_protect", "p1_protect_v1.csv")
V1_POINTS = [
    ("medium", 0.7, 1.0, "False"),
    ("strong", 0.7, 10.0, "True"),
    ("medium", 0.3, 0.1, "True"),
]
V1_COLUMNS = [
    "P_block_arrival_stable", "E_W", "Cost", "ERP", "E_N", "lambda_eff",
    "E_B", "E_S", "E_I", "E_off", "p1_fire_rate", "p1_launch_rate",
    "setup_completion_rate", "setup_cancel_rate", "rho_B",
]


@pytest.mark.parametrize("burst,rho,alpha,protect", V1_POINTS)
def test_matches_p1_protect_v1(burst, rho, alpha, protect):
    """protect_delayoff=False で results/p1_protect/p1_protect_v1.csv の値を再現する."""
    if not os.path.exists(V1_CSV):
        pytest.skip("results/p1_protect/p1_protect_v1.csv がない")
    with open(V1_CSV, newline="", encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f)
                   if r["burst_name"] == burst and float(r["rho"]) == rho
                   and float(r["alpha"]) == alpha and r["protect_presetup"] == protect)
    params = make_params(
        c=20, K=200, b=5, rho=rho, delta=float(row["delta"]),
        sigma=float(row["sigma"]), alpha=alpha, beta=float(row["beta"]),
        n_target=int(row["n_target"]), gamma=float(row["gamma"]),
        protect_presetup=(protect == "True"), protect_delayoff=False,
    )
    _, m = solve(params)
    got = {
        "P_block_arrival_stable": m.P_block_arrival_stable,
        "E_W": m.mean_waiting_time(), "Cost": m.energy_cost_paper(),
        "ERP": m.erp_paper(), "E_N": m.E_N, "lambda_eff": m.lambda_eff,
        "E_B": m.E_B, "E_S": m.E_S, "E_I": m.E_I, "E_off": m.E_off,
        "p1_fire_rate": m.p1_fire_rate, "p1_launch_rate": m.p1_launch_rate,
        "setup_completion_rate": m.setup_completion_rate,
        "setup_cancel_rate": m.setup_cancel_rate, "rho_B": m.rho_B,
    }
    for col in V1_COLUMNS:
        assert got[col] == pytest.approx(float(row[col]), rel=1e-12), col


# ============================================================
# 2. 両方 True でも n_target=0, gamma=1 でベースに一致
# ============================================================

@pytest.mark.parametrize("rho", [0.3, 0.5, 0.7, 0.9])
def test_both_protect_reduce_to_base(rho):
    c, K, b, delta, sigma = 5, 20, 2, 0.6, 0.1
    C0, C1 = build_mmpp(rho, delta, sigma, c, b, 1.0)
    base_p = ModelParameters(c=c, K=K, b=b, mu=1.0, alpha=0.1, beta=0.005,
                             C0=C0, C1=C1)
    m_base = BaseMetrics(base_p, solve_base(build_base_generator(base_p), solver="gth"))
    pred_p = make_params(c=c, K=K, b=b, rho=rho, delta=delta, sigma=sigma,
                         n_target=0, gamma=1.0,
                         protect_presetup=True, protect_delayoff=True)
    _, m_pred = solve(pred_p)

    for method in ["arrival_blocking_probability_stable", "mean_waiting_time",
                   "energy_cost_paper", "erp_paper", "mean_setup",
                   "mean_queue_length", "mean_idle"]:
        a = getattr(m_base, method)()
        p = getattr(m_pred, method)()
        assert abs(a - p) / abs(a) < 1e-10, f"{method}: base={a} pred={p}"


def test_generator_identical_when_n_target_zero():
    Q_off = build_generator(make_params(n_target=0))
    Q_on = build_generator(make_params(n_target=0, protect_presetup=True,
                                       protect_delayoff=True))
    assert (Q_off != Q_on).nnz == 0


# ============================================================
# 3. 遷移規則と, バースト中に i+s >= n_target が保たれること
# ============================================================

class TestDelayoffRule:
    """c=6, b=2, n_target=4. 状態 (i,s,j)=(2,2,0) (2 台ともアイドル) からの Delayoff."""

    def _delayoff_rate(self, params, F):
        Q = build_generator(params).tocsr()
        _, pairs = build_is_index(params.c)
        is_index = {pair: k for k, pair in enumerate(pairs)}
        src = state_to_idx(2, 2, 0, F, params.c, params.K, params.D_M, is_index)
        dst = state_to_idx(1, 2, 0, F, params.c, params.K, params.D_M, is_index)
        return Q[src, dst]

    def test_default_allows_in_burst(self):
        params = make_params(beta=0.5)
        assert self._delayoff_rate(params, F=1) == pytest.approx(2 * 0.5)

    def test_protect_blocks_in_burst(self):
        params = make_params(beta=0.5, protect_delayoff=True)
        assert self._delayoff_rate(params, F=1) == 0

    def test_protect_allows_in_normal_phase(self):
        params = make_params(beta=0.5, gamma=5.0, protect_delayoff=True)
        assert self._delayoff_rate(params, F=0) == pytest.approx(2 * 0.5 * 5.0)

    def test_protect_allows_above_n_target(self):
        params = make_params(beta=0.5, n_target=3, protect_delayoff=True)
        assert self._delayoff_rate(params, F=1) == pytest.approx(2 * 0.5)

    @pytest.mark.parametrize("protect,F,expect_delayoff", [
        (False, 1, True), (True, 1, False), (True, 0, True),
    ])
    def test_des_same_rule(self, protect, F, expect_delayoff):
        params = make_params(beta=0.5, protect_delayoff=protect)
        sim = PredictiveSimulator(params, seed=5)
        seen = False
        for _ in range(3000):
            sim.state = (2, 2, 0, F)
            sim.fifo.clear()
            r = sim.step()
            if r.event_type == EventType.DELAYOFF_TIMEOUT:
                seen = True
                assert r.new_state == (1, 2, 0, F)
        assert seen == expect_delayoff


def _burst_paths_from_p1(params):
    """P1 発動直後の状態から, バースト位相 F=1 に留まる経路で訪れる状態の集合."""
    Q = build_generator(params).tocsr()
    c, K, D_M = params.c, params.K, params.D_M
    is_index, pairs = build_is_index(c)

    adjacency = Q.copy()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    reachable = breadth_first_order(adjacency, i_start=0, directed=True,
                                    return_predecessors=False)

    starts = set()
    for idx in reachable:
        i, s, j, F = idx_to_state(int(idx), c, K, D_M, pairs)
        if F != 0:
            continue
        delta_eff = setup_target_delta(i, s, params.n_target, c)
        if delta_eff > 0:
            starts.add(state_to_idx(i, s + delta_eff, j, 1, c, K, D_M, is_index))
    assert starts, "P1 が発動する状態が到達可能であること"

    visited = set(starts)
    queue = deque(starts)
    while queue:
        src = queue.popleft()
        for dst in adjacency.indices[adjacency.indptr[src]:adjacency.indptr[src + 1]]:
            dst = int(dst)
            if dst in visited or idx_to_state(dst, c, K, D_M, pairs)[3] != 1:
                continue
            visited.add(dst)
            queue.append(dst)
    states = [idx_to_state(idx, c, K, D_M, pairs) for idx in visited]
    return states, reachable


@pytest.mark.parametrize("kw", [dict(), dict(rho=0.3, beta=0.5),
                                dict(delta=0.9, sigma=0.01, beta=0.5)])
def test_both_protect_keep_n_target_in_burst(kw):
    params = make_params(protect_presetup=True, protect_delayoff=True, **kw)
    states, reachable = _burst_paths_from_p1(params)
    under = [(i, s, j) for i, s, j, F in states if i + s < params.n_target]
    assert under == []

    # 到達可能な F=1 の状態全体でも i+s < n_target は現れない
    # (build_mmpp の C1 は対角なので F=1 へは位相遷移 (P1) でしか入らない)
    _, pairs = build_is_index(params.c)
    for idx in reachable:
        i, s, j, F = idx_to_state(int(idx), params.c, params.K, params.D_M, pairs)
        assert not (F == 1 and i + s < params.n_target)


@pytest.mark.parametrize("protect_presetup,protect_delayoff",
                         [(True, False), (False, True)])
def test_single_protection_leaves_hole(protect_presetup, protect_delayoff):
    """片方の保護だけでは, 同じ経路で i+s < n_target に入る (上のテストが無意味でない)."""
    params = make_params(beta=0.5, protect_presetup=protect_presetup,
                         protect_delayoff=protect_delayoff)
    states, _ = _burst_paths_from_p1(params)
    assert any(i + s < params.n_target for i, s, j, F in states)


# ============================================================
# 4. 診断指標
# ============================================================

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kw", [dict(), dict(rho=0.3, beta=0.5)])
def test_server_flow_balance(mode, kw):
    """i の増加率 (セットアップ完了) = i の減少率 (Delayoff)."""
    params = make_params(protect_presetup=mode[0], protect_delayoff=mode[1], **kw)
    _, m = solve(params)
    assert m.delayoff_rate == pytest.approx(m.setup_completion_rate, rel=1e-10)


@pytest.mark.parametrize("mode", MODES)
def test_p1_rates_match_des(mode):
    """DES の P1 発動回数・取り消し回数の率が理論値と 95% CI の範囲で一致."""
    params = make_params(c=4, K=10, b=2, rho=0.6, delta=0.6, sigma=0.3,
                         alpha=0.5, beta=0.2, n_target=3, gamma=2.0,
                         protect_presetup=mode[0], protect_delayoff=mode[1])
    _, m = solve(params)

    fire, cancel = [], []
    for k in range(10):
        sim = PredictiveSimulator(params, seed=200 + k)
        st = sim.run(warmup_events=5000, measurement_events=50000)
        fire.append(sim.proactive_setup_count / st.total_duration)
        cancel.append(sim.setup_cancel_count / st.total_duration)

    for name, values, theory in [("p1_fire_rate", fire, m.p1_fire_rate),
                                 ("setup_cancel_rate", cancel, m.setup_cancel_rate)]:
        v = np.asarray(values)
        half = scipy_stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        assert abs(v.mean() - theory) <= half, (
            f"{name}: DES={v.mean():.5g}±{half:.3g}, theory={theory:.5g}")
