"""独立レプリケーション法.

ベースの DES と同じ手順 (mmpp_sim.simulator.run_replications) を, PredictiveSimulator で実行する.
"""
from typing import List

from mmpp_sim.simulator import run_replications as _run_replications
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
    異なるシード (k 番目は seed0+k) で warmup_events + measurement_events を実行する.
    得られた SimStats のリストを返す. 信頼区間はこのリストを使って
    mmpp_predictive_sim.metrics.SimMetrics 側で計算する.

    Args:
        params: PredictiveModelParameters.
        n_reps, seed0, warmup_events, measurement_events: mmpp_sim.simulator.run_replications と同じ.
    """
    return _run_replications(params, n_reps, seed0, warmup_events, measurement_events,
                             simulator_cls=PredictiveSimulator)
