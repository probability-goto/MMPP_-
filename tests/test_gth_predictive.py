"""Predictive モデルでの GTH ソルバーの検証.

Predictive モデルは名目状態空間に到達不能状態を含むため, 到達可能部分
空間に縮約してから GTH を適用する. その経路が正しく動くこと, および
solver="gth" / "splu" が一致することを確認する.
"""
import numpy as np
import pytest

from mmpp_predictive.model import PredictiveModelParameters
from mmpp_predictive.generator import build_generator as build_pred_generator
from mmpp_predictive.solver import solve_stationary as solve_pred
from mmpp_predictive.metrics import Metrics as PredMetrics

from mmpp.gth_solver import bandwidths, check_reachability

try:
    from scripts._mmpp_burst import build_mmpp
except ImportError:
    from _mmpp_burst import build_mmpp


def make_pred_params(c=5, K=20, b=2, n_target=3, gamma=5.0, rho=0.5):
    C0, C1 = build_mmpp(rho=rho, delta=0.6, sigma=0.1, c=c, b=b, mu=1.0)
    return PredictiveModelParameters(
        c=c, K=K, b=b, mu=1.0, alpha=0.1, beta=0.005,
        C0=C0, C1=C1, n_target=n_target, gamma=gamma,
    )


class TestPredictiveGTH:
    """Predictive モデルでの GTH の基本性質."""

    def test_properties(self):
        params = make_pred_params()
        Q = build_pred_generator(params)
        pi = solve_pred(Q, solver="gth")

        assert (pi >= 0).all(), "全成分が非負であること"
        assert abs(pi.sum() - 1.0) < 1e-10, "正規化されていること"
        residual = np.abs(pi @ Q).max()
        assert residual < 1e-8, f"残差過大: {residual:.3e}"

    def test_gth_matches_splu(self):
        """GTH と splu が指標レベルで一致する."""
        params = make_pred_params()
        Q = build_pred_generator(params)

        pi_gth = solve_pred(Q, solver="gth")
        pi_splu = solve_pred(Q, solver="splu")

        m_gth = PredMetrics(params, pi_gth).all_metrics()
        m_splu = PredMetrics(params, pi_splu).all_metrics()

        for key in m_gth:
            a, b = m_gth[key], m_splu[key]
            denom = max(abs(a), abs(b), 1e-300)
            # P_block 系は極小値になりうるので, 値が十分大きい場合のみ厳密に比較
            if denom < 1e-12:
                continue
            assert abs(a - b) / denom < 1e-6, f"{key}: gth={a} splu={b}"

    @pytest.mark.parametrize("n_target,gamma", [(0, 1.0), (3, 5.0), (5, 2.0)])
    def test_various_settings(self, n_target, gamma):
        """n_target / gamma を変えても GTH が解けること."""
        params = make_pred_params(n_target=n_target, gamma=gamma)
        Q = build_pred_generator(params)
        pi = solve_pred(Q, solver="gth")
        assert (pi >= 0).all()
        assert abs(pi.sum() - 1.0) < 1e-10

    def test_unreachable_states_get_zero(self):
        """到達不能な状態の確率は 0 になる."""
        params = make_pred_params(n_target=0, gamma=1.0)
        Q = build_pred_generator(params)
        pi = solve_pred(Q, solver="gth")
        # 名目状態空間のサイズを保ったまま返ること
        assert pi.shape == (Q.shape[0],)
        # n_target=0 では到達不能状態が多数あり, それらは厳密に 0
        assert (pi == 0.0).any()


class TestReducedSpacePrecondition:
    """縮約後の部分空間が GTH の前提条件を満たすこと."""

    def test_reduced_space_is_irreducible(self):
        from scipy.sparse.csgraph import breadth_first_order

        params = make_pred_params()
        Q = build_pred_generator(params).tocsr()

        adj = Q.copy()
        adj.setdiag(0)
        adj.eliminate_zeros()
        reach = np.sort(breadth_first_order(
            adj, i_start=0, directed=True, return_predecessors=False))
        Q_reduced = Q[reach][:, reach].tocsr()

        # 縮約後は GTH の前提条件 (全状態が最終状態へ到達可能) を満たす
        check_reachability(Q_reduced)  # 例外が出なければ合格

    def test_reduced_bandwidth_is_measured(self):
        """縮約後の帯幅が実測され, 妥当な範囲にあること."""
        from scipy.sparse.csgraph import breadth_first_order

        params = make_pred_params()
        Q = build_pred_generator(params).tocsr()
        adj = Q.copy()
        adj.setdiag(0)
        adj.eliminate_zeros()
        reach = np.sort(breadth_first_order(
            adj, i_start=0, directed=True, return_predecessors=False))
        Q_reduced = Q[reach][:, reach].tocsr()

        p, q = bandwidths(Q_reduced)
        N = Q_reduced.shape[0]
        assert 0 < p < N
        assert 0 < q < N
