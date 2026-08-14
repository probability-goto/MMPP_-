"""独立レプリケーション法.

mmpp_sim では run_replications() を simulator.py 内に置いているが,
本パッケージではファイル構成の見通しのため独立ファイルに分離する
(ロジックは mmpp_sim.simulator.run_replications と同一方針: n_reps 個の
独立な PredictiveSimulator インスタンスを異なるシードで実行する).
"""
from typing import List

from mmpp_sim.stats import SimStats

from mmpp_predictive_sim.simulator import PredictiveSimulator


def run_replications(
    params,
    n_reps: int = 20,
    seed0: int = 42,
    warmup_events: int = 100_000,
    measurement_events: int = 1_000_000,
) -> List[SimStats]:
    """独立レプリケーション法によるシミュレーション実行.

    n_reps 個の独立な PredictiveSimulator インスタンスを生成し, それぞれ
    異なるシードで warmup_events + measurement_events を実行する. 得られた
    SimStats のリストを返す. 信頼区間はこのリストを使って
    mmpp_predictive_sim.metrics.SimMetrics 側で計算する.

    Args:
        params: PredictiveModelParameters.
        n_reps: レプリケーション数 (推奨 20 以上).
        seed0: 開始シード. k 番目のレプリケーションは seed=seed0+k を使う.
        warmup_events: 各レプリケーションのウォームアップイベント数.
        measurement_events: 各レプリケーションの本計測イベント数.

    Returns:
        List[SimStats]: 各レプリケーションの統計. 長さ n_reps.
    """
    replications: List[SimStats] = []
    for k in range(n_reps):
        sim = PredictiveSimulator(params, seed=seed0 + k)
        stats = sim.run(warmup_events=warmup_events, measurement_events=measurement_events)
        replications.append(stats)
    return replications
