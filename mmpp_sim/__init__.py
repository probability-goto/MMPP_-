"""MMPP/M/c/SET-BATCH/delayoff の離散事象シミュレーション (DES).

mmpp パッケージ (理論解析実装) の正当性検証用の独立実装.
ModelParameters のみ mmpp と共有し, 生成行列やソルバーには依存しない.

主要 API:
    Simulator: DES 本体
    SimStats: 時間重み付き統計収集
    SimMetrics: シミュレーション統計からの性能指標計算 (mmpp.Metrics と同一 API)
    EventType: イベント種別
"""
from mmpp_sim.event import EventType, EventQueue, StepResult
from mmpp_sim.simulator import (
    Simulator,
    busy_servers,
    idle_servers,
    setup_servers,
    off_servers,
)
from mmpp_sim.stats import SimStats
from mmpp_sim.metrics import SimMetrics

__all__ = [
    "Simulator",
    "SimStats",
    "SimMetrics",
    "EventType",
    "EventQueue",
    "StepResult",
    "busy_servers",
    "idle_servers",
    "setup_servers",
    "off_servers",
]

__version__ = "0.1.0"
