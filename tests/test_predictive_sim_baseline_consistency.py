"""Predictive シミュレータが n_target=0, gamma=1 でベースシミュレータと
一致することを検証する.

n_target=0 のとき, s は常に反応的セットアップの目標値
required_setup_count(i,j,b,c) (= ベースモデルの setup_servers(i,j,b,c)) に
帰納的に一致し続け (mmpp_predictive.generator の docstring と同じ議論),
gamma=1 のとき Delayoff レートは常に beta (加速なし) になる. したがって
イベントメニューの候補・順序・レートはベースシミュレータと状態空間の
次元 (s の有無) を除いて完全に一致し, 同一シードなら乱数呼び出し列も
完全に一致するため, 統計量は機械精度で一致するはずである
(理論モデルレベルでの一致は tests/test_predictive_baseline_consistency.py
で既に検証済み; 本テストは DES 実装レベルでの一致を検証する).
"""
import numpy as np
import pytest
from scipy.stats import ttest_ind

from mmpp.model import ModelParameters
from mmpp_sim.simulator import Simulator as BaseSimulator
from mmpp_sim.metrics import SimMetrics as BaseSimMetrics
from mmpp_sim.simulator import run_replications as run_base_replications

from mmpp_predictive.model import PredictiveModelParameters
from mmpp_predictive_sim.simulator import PredictiveSimulator
from mmpp_predictive_sim.metrics import SimMetrics as PredSimMetrics
from mmpp_predictive_sim.replication import run_replications as run_pred_replications

try:
    from scripts._mmpp_burst import build_mmpp
except ImportError:
    from _mmpp_burst import build_mmpp


def _matched_params(c=5, K=20, b=2, delta=0.6, sigma=0.1, rho=0.5):
    mu, alpha, beta = 1.0, 0.5, 0.1
    C0, C1 = build_mmpp(rho=rho, delta=delta, sigma=sigma, c=c, b=b, mu=mu)

    base_params = ModelParameters(
        c=c, K=K, b=b, mu=mu, alpha=alpha, beta=beta, C0=C0, C1=C1,
    )
    pred_params = PredictiveModelParameters(
        c=c, K=K, b=b, mu=mu, alpha=alpha, beta=beta, C0=C0, C1=C1,
        n_target=0, gamma=1.0,
    )
    return base_params, pred_params


def test_predictive_sim_matches_base_exactly_single_run():
    """n_target=0, gamma=1, 同一シードで, 全指標が機械精度で一致する.

    (i,s,j,F) の s は常に (i,j) から一意に定まるため, 乱数呼び出し列が
    ベースシミュレータと完全に一致し, 統計量も厳密一致するはず.
    """
    base_params, pred_params = _matched_params()
    seed = 2024

    base_sim = BaseSimulator(base_params, seed=seed)
    base_stats = base_sim.run(warmup_events=5_000, measurement_events=20_000)
    base_metrics = BaseSimMetrics(base_params, base_stats).all_metrics()

    pred_sim = PredictiveSimulator(pred_params, seed=seed)
    pred_stats = pred_sim.run(warmup_events=5_000, measurement_events=20_000)
    pred_metrics = PredSimMetrics(pred_params, pred_stats).all_metrics()

    assert base_stats.total_duration == pytest.approx(pred_stats.total_duration, rel=1e-12)
    assert base_stats.arrival_attempts == pred_stats.arrival_attempts
    assert base_stats.arrival_blocked == pred_stats.arrival_blocked
    assert base_stats.departure_count == pred_stats.departure_count

    for key in ["P_block", "P_block_arrival", "E[j]", "E[B]", "E[I]", "E[S]",
                "E[Off]", "lambda_eff", "E[W]", "rho"]:
        assert base_metrics[key] == pytest.approx(pred_metrics[key], rel=1e-9, abs=1e-12), (
            f"{key}: base={base_metrics[key]:.10g}, pred={pred_metrics[key]:.10g}"
        )


def test_predictive_sim_matches_base_when_disabled():
    """n_target=0, gamma=1 で Predictive シミュレータがベースと統計的に一致.

    独立レプリケーション法で P_block の平均を比較し, Welch の t 検定で
    有意差がないことを確認する (実装バグがあれば高確率で棄却されるはずの
    緩やかなチェック; 上の exact-match テストがより強い検証).
    """
    base_params, pred_params = _matched_params()

    n_reps = 10
    base_p_block = []
    pred_p_block = []
    for rep in range(n_reps):
        seed = 42 + rep

        base_sim = BaseSimulator(base_params, seed=seed)
        base_result = base_sim.run(warmup_events=10_000, measurement_events=100_000)
        base_p_block.append(BaseSimMetrics(base_params, base_result).blocking_probability())

        pred_sim = PredictiveSimulator(pred_params, seed=seed)
        pred_result = pred_sim.run(warmup_events=10_000, measurement_events=100_000)
        pred_p_block.append(PredSimMetrics(pred_params, pred_result).blocking_probability())

    t_stat, p_value = ttest_ind(base_p_block, pred_p_block, equal_var=False)
    assert p_value > 0.05, (
        f"P_block が有意に異なる: base={np.mean(base_p_block):.4f}, "
        f"pred={np.mean(pred_p_block):.4f}, p={p_value:.4f}"
    )


def test_predictive_sim_matches_base_conservation_and_flow():
    """独立レプリケーション法での SimMetrics 経由でも保存則・流量バランスが両者一致."""
    base_params, pred_params = _matched_params(rho=0.7)

    base_reps = run_base_replications(
        base_params, n_reps=5, seed0=100, warmup_events=5_000, measurement_events=20_000,
    )
    pred_reps = run_pred_replications(
        pred_params, n_reps=5, seed0=100, warmup_events=5_000, measurement_events=20_000,
    )

    base_m = BaseSimMetrics(base_params, base_reps)
    pred_m = PredSimMetrics(pred_params, pred_reps)

    assert base_m.mean_busy() == pytest.approx(pred_m.mean_busy(), rel=1e-9)
    assert base_m.mean_idle() == pytest.approx(pred_m.mean_idle(), rel=1e-9)
    assert base_m.mean_setup() == pytest.approx(pred_m.mean_setup(), rel=1e-9)
    assert base_m.mean_off() == pytest.approx(pred_m.mean_off(), rel=1e-9)
