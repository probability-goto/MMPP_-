"""P_block^arrival の安定版計算 (arrival_blocking_probability_stable) の検証.

現行の 1 - lambda_eff/lambda_bar は引き算を含むため, P_block^arrival が
極小になる領域で桁落ちする (負の値も出る)。安定版は
    P_block^arrival = sum_{i,F} pi(i,K,F) lambda_F / lambda_bar
であり数学的に厳密に等価だが引き算を含まない。

検証内容:
    - 値が十分大きい領域では両者が一致すること (等価性)
    - 極小領域では安定版が正値を保ち, 現行版が破綻すること
"""
import numpy as np
import pytest

from mmpp import ModelParameters, build_generator, Metrics
from mmpp.solver import solve_stationary

try:
    from scripts._mmpp_burst import build_mmpp
except ImportError:
    from _mmpp_burst import build_mmpp


def make_params(rho, c=5, K=60, b=2, delta=0.6, sigma=0.1):
    C0, C1 = build_mmpp(rho=rho, delta=delta, sigma=sigma, c=c, b=b, mu=1.0)
    return ModelParameters(
        c=c, K=K, b=b, mu=1.0, alpha=0.1, beta=0.005, C0=C0, C1=C1
    )


class TestEquivalence:
    """値が十分大きい領域での等価性."""

    @pytest.mark.parametrize("rho", [0.5, 0.7, 0.9])
    def test_agrees_when_not_tiny(self, rho):
        params = make_params(rho)
        pi = solve_stationary(params_Q := build_generator(params))
        m = Metrics(params, pi)

        a = m.arrival_blocking_probability()
        b = m.arrival_blocking_probability_stable()

        # 値が 1e-10 以上なら両者は機械精度レベルで一致するはず
        assert a > 1e-10, "この rho では P_block^arrival が十分大きいこと"
        assert abs(a - b) / max(abs(a), abs(b)) < 1e-9, f"a={a} b={b}"

    def test_in_all_metrics(self):
        params = make_params(0.7)
        m = Metrics(params, solve_stationary(build_generator(params)))
        d = m.all_metrics()
        assert "P_block_arrival" in d
        assert "P_block_arrival_stable" in d


class TestStabilityAtTinyValues:
    """極小領域での安定性."""

    @pytest.mark.parametrize("rho", [0.05, 0.1, 0.15])
    def test_stable_version_stays_positive(self, rho):
        """安定版は極小領域でも厳密に正の値を返す."""
        params = make_params(rho)
        pi = solve_stationary(build_generator(params))
        m = Metrics(params, pi)

        stable = m.arrival_blocking_probability_stable()
        assert stable > 0.0, f"安定版が非正: {stable}"
        # ブロック確率なので 1 以下
        assert stable <= 1.0

    def test_stable_is_monotone_in_rho(self):
        """安定版は rho に対して単調増加 (物理的に自然な性質).

        現行版は桁落ちのためこの単調性が低 rho で崩れる。
        """
        rhos = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
        stable_vals = []
        for rho in rhos:
            params = make_params(rho)
            pi = solve_stationary(build_generator(params))
            stable_vals.append(
                Metrics(params, pi).arrival_blocking_probability_stable()
            )

        for prev, cur in zip(stable_vals, stable_vals[1:]):
            assert cur > prev, f"単調性違反: {stable_vals}"


class TestPredictiveStable:
    """Predictive モデルでも同じ性質が成り立つこと."""

    def test_equivalence_and_positivity(self):
        from mmpp_predictive.model import PredictiveModelParameters
        from mmpp_predictive.generator import build_generator as build_pred
        from mmpp_predictive.solver import solve_stationary as solve_pred
        from mmpp_predictive.metrics import Metrics as PredMetrics

        C0, C1 = build_mmpp(rho=0.7, delta=0.6, sigma=0.1, c=5, b=2, mu=1.0)
        params = PredictiveModelParameters(
            c=5, K=30, b=2, mu=1.0, alpha=0.1, beta=0.005,
            C0=C0, C1=C1, n_target=3, gamma=5.0,
        )
        pi = solve_pred(build_pred(params))
        m = PredMetrics(params, pi)

        a = m.arrival_blocking_probability()
        b = m.arrival_blocking_probability_stable()
        assert a > 1e-10
        assert abs(a - b) / max(abs(a), abs(b)) < 1e-9, f"a={a} b={b}"
        assert "P_block_arrival_stable" in m.all_metrics()
