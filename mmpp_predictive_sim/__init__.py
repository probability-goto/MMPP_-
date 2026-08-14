"""MMPP/M/c/SET-BATCH/Delayoff/Predictive の離散事象シミュレーション (DES).

mmpp_predictive パッケージ (理論解析実装) の正当性検証用の独立実装.
PredictiveModelParameters のみ mmpp_predictive と共有し, 生成行列やソルバー
には依存しない.

主要 API:
    PredictiveSimulator: DES 本体
    run_replications: 独立レプリケーション法によるシミュレーション実行
    SimStats: 時間重み付き統計収集 (mmpp_sim.stats を再利用)
    SimMetrics: シミュレーション統計からの性能指標計算 (mmpp_predictive.Metrics と同一 API)
    EventType: イベント種別
"""
from mmpp_sim.stats import SimStats

from mmpp_predictive_sim.event import EventType, StepResult
from mmpp_predictive_sim.simulator import (
    PredictiveSimulator,
    required_setup_count,
    setup_target_delta,
)
from mmpp_predictive_sim.replication import run_replications
from mmpp_predictive_sim.metrics import SimMetrics

__all__ = [
    "PredictiveSimulator",
    "run_replications",
    "SimStats",
    "SimMetrics",
    "EventType",
    "StepResult",
    "required_setup_count",
    "setup_target_delta",
]

__version__ = "0.1.0"
