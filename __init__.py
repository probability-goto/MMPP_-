"""MMPP/M/c/SET-BATCH/delayoff queueing model.

主要 API:
    ModelParameters: モデルパラメータのコンテナ
    StateSpace: 状態空間のインデックス変換
    build_generator: 生成行列 Q の構築
    solve_stationary: 定常分布ソルバー
    Metrics: 性能指標
"""
from mmpp.model import ModelParameters
from mmpp.state_space import StateSpace
from mmpp.generator import (
    build_generator,
    busy_servers,
    idle_servers,
    setup_servers,
    off_servers,
)
from mmpp.solver import solve_stationary
from mmpp.metrics import Metrics

__all__ = [
    "ModelParameters",
    "StateSpace",
    "build_generator",
    "busy_servers",
    "idle_servers",
    "setup_servers",
    "off_servers",
    "solve_stationary",
    "Metrics",
]

__version__ = "0.1.0"
