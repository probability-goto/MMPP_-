"""到着平均ブロッキング確率 P_block^arrival (arrival_blocking_probability_stable) の検証.

P_block^arrival は引き算を含まない式
    P_block^arrival = sum_{i,F} pi(i,K,F) lambda_F / lambda_bar
で計算する (定義式 1 - lambda_eff/lambda_bar は桁落ちするためライブラリから削除済み)。

検証内容:
    (a) 全実験格子で値が厳密に正 (かつ 1 以下) であること
        - results/ の全 CSV の P_block_arrival_stable 列 (実際の実験格子)
        - 実験格子と同じ rho / バースト水準 / K 水準を縮小モデルで走査し,
          ライブラリ関数自体が正値を返すこと
    (b) 高精度領域 (P_block^arrival が十分大きく, 引き算でも桁落ちしない領域) で,
        lambda_bar - lambda_eff を状態空間上の直接和で求めた参照値と一致すること
"""
import csv
import glob
import os

import numpy as np
import pytest

from mmpp import ModelParameters, build_generator, Metrics
from mmpp.solver import solve_stationary
from mmpp.state_space import StateSpace

try:
    from scripts._mmpp_burst import build_mmpp
except ImportError:
    from _mmpp_burst import build_mmpp


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 実験 1〜6 と同じ走査水準 (rho は実験 1/4 の 15 点格子)
RHO_GRID = list(np.linspace(0.1, 0.95, 15))
BURST_LEVELS = [("weak", 0.3, 1.0), ("medium", 0.6, 0.1), ("strong", 0.9, 0.01)]

# (b) で「高精度領域」とみなす下限. この値以上なら 1 - x の引き算で失う桁は
# 高々 6 桁程度なので, 参照値として 1e-9 の相対一致を要求できる.
HIGH_PRECISION_FLOOR = 1e-6


def make_params(rho, c=5, K=60, b=2, delta=0.6, sigma=0.1):
    C0, C1 = build_mmpp(rho=rho, delta=delta, sigma=sigma, c=c, b=b, mu=1.0)
    return ModelParameters(
        c=c, K=K, b=b, mu=1.0, alpha=0.1, beta=0.005, C0=C0, C1=C1
    )


def make_pred_params(rho, c=5, K=30, b=2, delta=0.6, sigma=0.1,
                     n_target=3, gamma=5.0):
    from mmpp_predictive.model import PredictiveModelParameters

    C0, C1 = build_mmpp(rho=rho, delta=delta, sigma=sigma, c=c, b=b, mu=1.0)
    return PredictiveModelParameters(
        c=c, K=K, b=b, mu=1.0, alpha=0.1, beta=0.005,
        C0=C0, C1=C1, n_target=n_target, gamma=gamma,
    )


def solve_base(params):
    return Metrics(params, solve_stationary(build_generator(params)))


def solve_pred(params):
    from mmpp_predictive.generator import build_generator as build_pred
    from mmpp_predictive.solver import solve_stationary as solve_pred_pi
    from mmpp_predictive.metrics import Metrics as PredMetrics

    return PredMetrics(params, solve_pred_pi(build_pred(params)))


def direct_reference_base(params, pi):
    """(lambda_bar - lambda_eff) / lambda_bar を状態空間上の直接和で求める.

    lambda_bar = sum_{全状態} pi(i,j,F) lambda_F,
    lambda_eff = sum_{j<K} pi(i,j,F) lambda_F
    をライブラリの指標メソッドを介さずに計算する (参照値専用).
    """
    ss = StateSpace(params)
    lam_bar = 0.0
    lam_eff = 0.0
    for (i, j, F) in ss.iter_states():
        w = pi[ss.index(i, j, F)] * params.lambdas[F]
        lam_bar += w
        if j < params.K:
            lam_eff += w
    return (lam_bar - lam_eff) / lam_bar


def direct_reference_pred(params, pi):
    """Predictive モデル版の直接和参照値 (状態順は j -> (i,s) -> F)."""
    from mmpp_predictive.state_space import pi_by_level

    K = params.K
    pi_3d = pi_by_level(pi, params.c, K, params.D_M)
    lam_by_j = pi_3d.sum(axis=1) @ params.lambdas  # shape (K+1,)
    lam_bar = float(lam_by_j.sum())
    lam_eff = float(lam_by_j[:K].sum())
    return (lam_bar - lam_eff) / lam_bar


# ---------------------------------------------------------------------------
# (a) 正値性
# ---------------------------------------------------------------------------

def _result_csvs():
    return sorted(glob.glob(os.path.join(REPO_ROOT, "results", "**", "*.csv"),
                            recursive=True))


class TestPositivityOnExperimentGrids:
    """全実験格子で P_block^arrival が厳密に正 (0 < P <= 1)."""

    @pytest.mark.skipif(not _result_csvs(), reason="results/ に CSV がない")
    @pytest.mark.parametrize(
        "path", _result_csvs(), ids=lambda p: os.path.relpath(p, REPO_ROOT)
    )
    def test_result_csv_column_positive(self, path):
        """実際の実験格子 (results/ の全 CSV) で安定版の値が正."""
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows, f"空の CSV: {path}"
        # DES 照合の CSV は理論値を theory_P_block_arrival_stable 列に持つ
        col = ("P_block_arrival_stable" if "P_block_arrival_stable" in rows[0]
               else "theory_P_block_arrival_stable")
        assert col in rows[0], f"列がない: {path}"
        vals = [float(r[col]) for r in rows]
        bad = [v for v in vals if not (0.0 < v <= 1.0)]
        assert not bad, f"{path}: 非正または 1 超の値 {bad[:5]}"

    @pytest.mark.parametrize("burst", BURST_LEVELS, ids=lambda b: b[0])
    @pytest.mark.parametrize("K", [20, 60])
    def test_base_library_positive_on_rho_grid(self, burst, K):
        """ベースモデル: 実験と同じ rho 格子・バースト水準で正値かつ rho に単調増加."""
        _, delta, sigma = burst
        vals = []
        for rho in RHO_GRID:
            m = solve_base(make_params(rho, K=K, delta=delta, sigma=sigma))
            v = m.arrival_blocking_probability_stable()
            assert 0.0 < v <= 1.0, f"rho={rho}: {v}"
            vals.append(v)
        # 引き算版で桁落ちしていた低 rho 側でも物理的な単調性が保たれる
        for prev, cur in zip(vals, vals[1:]):
            assert cur > prev, f"単調性違反 (K={K}, {burst[0]}): {vals}"

    @pytest.mark.parametrize("burst", BURST_LEVELS, ids=lambda b: b[0])
    def test_predictive_library_positive_on_rho_grid(self, burst):
        """Predictive モデル: 同じ rho 格子・バースト水準で正値."""
        _, delta, sigma = burst
        for rho in RHO_GRID:
            m = solve_pred(make_pred_params(rho, delta=delta, sigma=sigma))
            v = m.arrival_blocking_probability_stable()
            assert 0.0 < v <= 1.0, f"rho={rho}: {v}"

    def test_reaches_tiny_values_without_breakdown(self):
        """引き算版が破綻する 1e-13 未満の領域まで正値で到達できること."""
        # この点の真値は ~1e-28. 引き算版は丸め誤差 (~1e-16) しか返せない
        m = solve_base(make_params(0.1, K=150, delta=0.3, sigma=1.0))
        v = m.arrival_blocking_probability_stable()
        assert 0.0 < v < 1e-20, f"想定した極小領域に入っていない: {v}"


# ---------------------------------------------------------------------------
# (b) 高精度領域での直接和との一致
# ---------------------------------------------------------------------------

class TestAgreementWithDirectSum:
    """高精度領域で (lambda_bar - lambda_eff) / lambda_bar の直接和と一致."""

    @pytest.mark.parametrize("burst", BURST_LEVELS, ids=lambda b: b[0])
    @pytest.mark.parametrize("rho", [0.5, 0.7, 0.9, 0.95])
    def test_base(self, rho, burst):
        _, delta, sigma = burst
        params = make_params(rho, K=20, delta=delta, sigma=sigma)
        pi = solve_stationary(build_generator(params))
        stable = Metrics(params, pi).arrival_blocking_probability_stable()
        ref = direct_reference_base(params, pi)

        assert ref > HIGH_PRECISION_FLOOR, f"高精度領域でない: ref={ref}"
        assert abs(stable - ref) / ref < 1e-9, f"stable={stable} ref={ref}"

    @pytest.mark.parametrize("rho", [0.5, 0.7, 0.9])
    def test_predictive(self, rho):
        params = make_pred_params(rho, K=15)
        m = solve_pred(params)
        stable = m.arrival_blocking_probability_stable()
        ref = direct_reference_pred(params, m.pi)

        assert ref > HIGH_PRECISION_FLOOR, f"高精度領域でない: ref={ref}"
        assert abs(stable - ref) / ref < 1e-9, f"stable={stable} ref={ref}"


class TestAllMetricsColumns:
    """all_metrics の列構成: 安定版のみを出力し, 引き算版の列は持たない."""

    def test_base(self):
        d = solve_base(make_params(0.7)).all_metrics()
        assert "P_block_arrival_stable" in d
        assert "P_block_arrival" not in d

    def test_predictive(self):
        d = solve_pred(make_pred_params(0.7)).all_metrics()
        assert "P_block_arrival_stable" in d
        assert "P_block_arrival" not in d
