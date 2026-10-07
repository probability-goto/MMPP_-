"""位相の情報を使わない対照 (never_cancel_setup) のテスト.

- never_cancel_setup=False で既存の結果 (results/p1_protect/p1_protect.csv) が変わらないこと
- セットアップの流量の保存と, Delayoff 率 = セットアップ完了率
- DES と理論の一致 (P_block_arrival_stable, E[W], Cost, setup_cancel_rate=0)
"""
import csv
import os

import pytest

from mmpp.generator import setup_servers as compute_required_s

from mmpp_predictive import (
    PredictiveModelParameters, build_generator, solve_stationary, Metrics,
)
from mmpp_predictive.state_space import (
    build_is_index, pi_by_level, setup_cancelled,
)
from mmpp_predictive_sim import PredictiveSimulator, SimMetrics

try:
    from scripts._mmpp_burst import build_mmpp
except ImportError:
    from _mmpp_burst import build_mmpp


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_params(c=6, K=30, b=2, rho=0.7, delta=0.6, sigma=0.1,
                alpha=0.1, beta=0.005, n_target=0, gamma=1.0,
                protect_presetup=False, protect_delayoff=False,
                never_cancel_setup=False):
    C0, C1 = build_mmpp(rho, delta, sigma, c, b, 1.0)
    return PredictiveModelParameters(
        c=c, K=K, b=b, mu=1.0, alpha=alpha, beta=beta, C0=C0, C1=C1,
        n_target=n_target, gamma=gamma,
        protect_presetup=protect_presetup, protect_delayoff=protect_delayoff,
        never_cancel_setup=never_cancel_setup,
    )


def solve(params):
    pi = solve_stationary(build_generator(params), solver="gth")
    return pi, Metrics(params, pi)


# ============================================================
# 判定関数と既定値
# ============================================================

def test_default_is_false():
    assert make_params().never_cancel_setup is False


def test_setup_cancelled_never():
    # 需要 0 で s=3: 通常は取り消すが, never_cancel_setup では取り消さない
    assert setup_cancelled(1, 3, 0, 0, 2, 6, 4, False) is True
    assert setup_cancelled(1, 3, 0, 0, 2, 6, 4, False, True) is False
    assert setup_cancelled(1, 3, 0, 1, 2, 6, 0, True, True) is False


# ============================================================
# never_cancel_setup=False で既存の結果が変わらない
# ============================================================

P1_CSV = os.path.join(REPO_ROOT, "results", "p1_protect", "p1_protect.csv")
P1_POINTS = [
    # (burst, rho, alpha, beta, gamma, protect_presetup, protect_delayoff)
    ("medium", 0.3, 0.1, 0.5, 1.0, True, True),
    ("strong", 0.7, 1.0, 0.005, 5.0, True, False),
    ("medium", 0.7, 10.0, 0.5, 5.0, False, False),
]
P1_COLUMNS = [
    "P_block_arrival_stable", "E_W", "Cost", "ERP", "E_N", "lambda_eff",
    "E_B", "E_S", "E_I", "E_off", "p1_fire_rate", "p1_launch_rate",
    "setup_completion_rate", "setup_cancel_rate", "delayoff_rate", "rho_B",
]


@pytest.mark.parametrize("burst,rho,alpha,beta,gamma,pp,pd", P1_POINTS)
def test_matches_p1_protect(burst, rho, alpha, beta, gamma, pp, pd):
    if not os.path.exists(P1_CSV):
        pytest.skip("results/p1_protect/p1_protect.csv がない")
    with open(P1_CSV, newline="", encoding="utf-8") as f:
        row = next(
            r for r in csv.DictReader(f)
            if r["burst_name"] == burst and float(r["rho"]) == rho
            and float(r["alpha"]) == alpha and float(r["beta"]) == beta
            and float(r["gamma"]) == gamma
            and r["protect_presetup"] == str(pp) and r["protect_delayoff"] == str(pd)
            and r.get("never_cancel_setup", "False") == "False"
        )
    params = make_params(
        c=20, K=200, b=5, rho=rho, delta=float(row["delta"]),
        sigma=float(row["sigma"]), alpha=alpha, beta=beta,
        n_target=int(row["n_target"]), gamma=gamma,
        protect_presetup=pp, protect_delayoff=pd, never_cancel_setup=False,
    )
    _, m = solve(params)
    got = {
        "P_block_arrival_stable": m.P_block_arrival_stable,
        "E_W": m.mean_waiting_time(), "Cost": m.energy_cost_paper(),
        "ERP": m.erp_paper(), "E_N": m.E_N, "lambda_eff": m.lambda_eff,
        "E_B": m.E_B, "E_S": m.E_S, "E_I": m.E_I, "E_off": m.E_off,
        "p1_fire_rate": m.p1_fire_rate, "p1_launch_rate": m.p1_launch_rate,
        "setup_completion_rate": m.setup_completion_rate,
        "setup_cancel_rate": m.setup_cancel_rate,
        "delayoff_rate": m.delayoff_rate, "rho_B": m.rho_B,
    }
    for col in P1_COLUMNS:
        assert got[col] == pytest.approx(float(row[col]), rel=1e-12), col


# ============================================================
# 流量の保存
# ============================================================

@pytest.mark.parametrize("kw", [
    dict(), dict(rho=0.3, beta=0.5), dict(n_target=4, gamma=5.0, beta=0.5),
])
def test_flow_balances(kw):
    """開始率 = 完了率 + 取り消し率 (取り消し率は 0), Delayoff 率 = 完了率."""
    params = make_params(never_cancel_setup=True, **kw)
    pi, m = solve(params)
    assert m.setup_cancel_rate == 0.0

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
    assert m.delayoff_rate == pytest.approx(m.setup_completion_rate, rel=1e-10)


def test_never_cancel_raises_setup_count():
    """取り消さないので, 平均セットアップ台数は保護なしより大きい."""
    _, m_off = solve(make_params(rho=0.3, beta=0.5))
    _, m_on = solve(make_params(rho=0.3, beta=0.5, never_cancel_setup=True))
    assert m_off.setup_cancel_rate > 0
    assert m_on.E_S > m_off.E_S


# ============================================================
# DES と理論の一致
# ============================================================

def test_des_matches_theory():
    """P_block_arrival_stable, E[W], Cost が 95% CI 内, 取り消し回数は 0."""
    params = make_params(c=4, K=10, b=2, rho=0.6, delta=0.6, sigma=0.3,
                         alpha=0.5, beta=0.2, never_cancel_setup=True)
    _, m = solve(params)
    assert m.setup_cancel_rate == 0.0

    reps = []
    for k in range(20):
        sim = PredictiveSimulator(params, seed=300 + k)
        reps.append(sim.run(warmup_events=5000, measurement_events=50000))
        assert sim.setup_cancel_count == 0
    sm = SimMetrics(params, reps)

    for name, theory, ci in [
        ("P_block_arrival", m.P_block_arrival_stable, sm.arrival_blocking_probability_ci()),
        ("E[W]", m.mean_waiting_time(), sm.mean_waiting_time_ci()),
        ("Cost", m.energy_cost_paper(), sm.energy_cost_paper_ci()),
    ]:
        _, lo, hi = ci
        assert lo <= theory <= hi, f"{name}: theory={theory:.6g}, CI=[{lo:.6g}, {hi:.6g}]"
