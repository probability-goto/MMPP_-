"""イベント種別と優先度付きキューの定義.

DES 本体 (simulator.py) で使う「次の 1 イベント」を表現するための
軽量なデータ構造を提供する. イベントは (time, event_type, metadata) の
tuple として heapq ベースの優先度キューで管理する.

このモジュールは mmpp パッケージに一切依存しない (独立検証のため).
"""
import heapq
import itertools
from enum import Enum, auto
from typing import Any, Dict, NamedTuple, Optional, Tuple


class EventType(Enum):
    """DES で発生しうるイベントの種別."""

    ARRIVAL = auto()
    """到着 (現位相の lambda_F で発生)."""

    SERVICE_COMPLETION = auto()
    """バッチサービス完了 (率 B(i,j) * mu). 1 イベントで j が b 減る."""

    SETUP_COMPLETION = auto()
    """セットアップ完了 (率 S(i,j) * alpha). i が 1 増える."""

    DELAYOFF_TIMEOUT = auto()
    """Delayoff タイムアウト (率 I(i,j) * beta). i が 1 減る."""

    PHASE_TRANSITION = auto()
    """MMPP 位相遷移 (到着を伴わない C0 由来の遷移)."""


class EventQueue:
    """heapq ベースの優先度付きイベントキュー (時刻昇順).

    同時刻イベントの比較時に EventType/metadata の比較エラーが起きないよう,
    単調増加カウンタをタイブレークキーとして併用する.
    """

    def __init__(self) -> None:
        self._heap: list = []
        self._counter = itertools.count()

    def push(
        self,
        time: float,
        event_type: EventType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """イベントをキューに追加する."""
        entry = (time, next(self._counter), event_type, metadata or {})
        heapq.heappush(self._heap, entry)

    def pop(self) -> Tuple[float, EventType, Dict[str, Any]]:
        """最も時刻の早いイベントを取り出す."""
        time, _, event_type, metadata = heapq.heappop(self._heap)
        return time, event_type, metadata

    def is_empty(self) -> bool:
        """キューが空かどうか."""
        return len(self._heap) == 0

    def __len__(self) -> int:
        return len(self._heap)


class StepResult(NamedTuple):
    """Simulator.step() の 1 イベント分の実行結果.

    Attributes:
        dt: 直前の状態での滞在時間.
        event_type: 発生したイベント種別.
        old_state: 遷移前の状態 (i, j, F).
        new_state: 遷移後の状態 (i, j, F).
        blocked: ARRIVAL の場合のみ有効 (ブロックされたか否か). それ以外は None.
        departure_count: SERVICE_COMPLETION の場合の departure ジョブ数 (それ以外は 0).
        avg_sojourn: SERVICE_COMPLETION の場合の平均 sojourn time (それ以外は None).
    """

    dt: float
    event_type: EventType
    old_state: Tuple[int, int, int]
    new_state: Tuple[int, int, int]
    blocked: Optional[bool]
    departure_count: int
    avg_sojourn: Optional[float]
