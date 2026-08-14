"""イベント種別と 1 イベント実行結果の型定義.

DES 本体 (simulator.py) が扱うイベントの種別と, `PredictiveSimulator.step()` の
戻り値の型を定義する軽量なモジュール. mmpp_predictive パッケージには依存しない
(mmpp_sim.event と同じ 5 種のイベント; 状態タプルが (i, j, F) から
(i, s, j, F) に拡張されている点のみが違い).
"""
from enum import Enum, auto
from typing import NamedTuple, Optional, Tuple


class EventType(Enum):
    """DES で発生しうるイベントの種別."""

    ARRIVAL = auto()
    """到着 (現位相の lambda_F で発生)."""

    SERVICE_COMPLETION = auto()
    """バッチサービス完了 (率 B(i,j) * mu). 1 イベントで j が b 減る."""

    SETUP_COMPLETION = auto()
    """セットアップ完了 (率 s * alpha). i が 1 増え, s が 1 減る."""

    DELAYOFF_TIMEOUT = auto()
    """Delayoff タイムアウト (率 I(i,j) * beta_F). i が 1 減る."""

    PHASE_TRANSITION = auto()
    """MMPP 位相遷移 (到着を伴わない C0 由来の遷移).

    F: 0 -> 1 の場合のみ, 事前セットアップ (Predictive 拡張) を伴い
    s が delta_eff だけ瞬時に増加する (同一イベント内で処理される).
    """


class StepResult(NamedTuple):
    """PredictiveSimulator.step() の 1 イベント分の実行結果.

    Attributes:
        dt: 直前の状態での滞在時間.
        event_type: 発生したイベント種別.
        old_state: 遷移前の状態 (i, s, j, F).
        new_state: 遷移後の状態 (i, s, j, F).
        blocked: ARRIVAL の場合のみ有効 (ブロックされたか否か). それ以外は None.
        departure_count: SERVICE_COMPLETION の場合の departure ジョブ数 (それ以外は 0).
        avg_sojourn: SERVICE_COMPLETION の場合の平均 sojourn time (それ以外は None).
        proactive_setup_delta: PHASE_TRANSITION (F: 0->1) で発動した
            事前セットアップの台数 (delta_eff). それ以外は 0.
    """

    dt: float
    event_type: EventType
    old_state: Tuple[int, int, int, int]
    new_state: Tuple[int, int, int, int]
    blocked: Optional[bool]
    departure_count: int
    avg_sojourn: Optional[float]
    proactive_setup_delta: int
