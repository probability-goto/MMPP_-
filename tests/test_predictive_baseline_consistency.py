"""Predictive モデルがベースモデルの拡張として整合することを検証.

理論的には n_target=0, gamma=1 で Predictive と Base は同じ挙動になる.
この一致を性能指標レベルで機械精度で検証する
(Predictive モデル実装の正しさの証となる最重要テスト).
"""
import pytest
import numpy as np

from mmpp.model import ModelParameters
from mmpp.generator import build_generator as build_base_generator
from mmpp.solver import solve_stationary as solve_base
from mmpp.metrics import Metrics as BaseMetrics

from mmpp_predictive.model import PredictiveModelParameters
from mmpp_predictive.generator import build_generator as build_pred_generator
from mmpp_predictive.solver import solve_stationary as solve_pred
from mmpp_predictive.metrics import Metrics as PredMetrics

try:
    from scripts._mmpp_burst import build_mmpp
except ImportError:
    from _mmpp_burst import build_mmpp


def make_matched_params(c=5, K=20, b=2, delta=0.6, sigma=0.1, rho=0.5):
    """ベースと Predictive で同じパラメータを構築する."""
    mu = 1.0
    C0, C1 = build_mmpp(rho=rho, delta=delta, sigma=sigma, c=c, b=b, mu=mu)

    base_params = ModelParameters(
        c=c, K=K, b=b, mu=mu, alpha=0.1, beta=0.005,
        C0=C0, C1=C1,
    )
    pred_params = PredictiveModelParameters(
        c=c, K=K, b=b, mu=mu, alpha=0.1, beta=0.005,
        C0=C0, C1=C1,
        n_target=0, gamma=1.0,  # ベースと一致するパラメータ
    )
    return base_params, pred_params


@pytest.mark.parametrize("rho", [0.3, 0.5, 0.7, 0.9])
def test_predictive_reduces_to_base_for_metrics(rho):
    """4 指標がベースと機械精度で一致することを確認."""
    base_p, pred_p = make_matched_params(rho=rho)

    # ベースモデルで計算
    Q_base = build_base_generator(base_p)
    pi_base = solve_base(Q_base)
    m_base = BaseMetrics(base_p, pi_base)

    # Predictive モデル (n_target=0, gamma=1) で計算
    Q_pred = build_pred_generator(pred_p)
    pi_pred = solve_pred(Q_pred)
    m_pred = PredMetrics(pred_p, pi_pred)

    # 4 指標を比較
    tol = 1e-8
    for metric_name in ["arrival_blocking_probability", "mean_waiting_time",
                         "energy_cost_paper", "erp_paper"]:
        val_base = getattr(m_base, metric_name)()
        val_pred = getattr(m_pred, metric_name)()
        rel_err = abs(val_base - val_pred) / max(abs(val_base), 1e-12)
        assert rel_err < tol, (
            f"{metric_name} at rho={rho}: base={val_base:.6g}, "
            f"pred={val_pred:.6g}, rel_err={rel_err:.2e}"
        )


@pytest.mark.parametrize("rho", [0.5, 0.9])
def test_predictive_reduces_to_base_for_server_counts(rho):
    """E[B], E[I], E[S], E[Off] もベースと機械精度で一致することを確認."""
    base_p, pred_p = make_matched_params(rho=rho)

    Q_base = build_base_generator(base_p)
    pi_base = solve_base(Q_base)
    m_base = BaseMetrics(base_p, pi_base)

    Q_pred = build_pred_generator(pred_p)
    pi_pred = solve_pred(Q_pred)
    m_pred = PredMetrics(pred_p, pi_pred)

    tol = 1e-8
    for metric_name in ["mean_busy", "mean_idle", "mean_setup", "mean_off"]:
        val_base = getattr(m_base, metric_name)()
        val_pred = getattr(m_pred, metric_name)()
        rel_err = abs(val_base - val_pred) / max(abs(val_base), 1e-12)
        assert rel_err < tol, (
            f"{metric_name} at rho={rho}: base={val_base:.6g}, "
            f"pred={val_pred:.6g}, rel_err={rel_err:.2e}"
        )
