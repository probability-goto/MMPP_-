"""時間重み付き統計収集.

DES 実行中に以下を蓄積する:
    - 状態 (i, j, F) ごとの累積滞在時間 (時間重み付き平均の元データ)
    - 到着 attempt 総数とブロック数
    - departure したジョブの sojourn time 総和と総数

batch means 法による信頼区間計算のため, 記録期間を等分割した
複数の SimStats を並行して蓄積できるよう `batches` 属性を持つ
(Simulator.run() が生成し, 集計後にセットする).
"""
from collections import defaultdict
from typing import Dict, List, Tuple


class SimStats:
    """時間重み付き統計を蓄積するクラス."""

    def __init__(self) -> None:
        # 辞書。キーは状態 (i, j, F) 、値はその状態に滞在した累積時間。\{(0,0,0): 12.3, (1,0,0): 8.5, \ldots\} のような形。
        self.state_duration: Dict[Tuple[int, int, int], float] = defaultdict(float)
        # 記録期間の総時間 T
        self.total_duration: float = 0.0
        # 到着 attempt の総回数(ブロックされたものも含む)。
        self.arrival_attempts: int = 0
        # そのうちブロックされた回数。
        self.arrival_blocked: int = 0
        # departure したジョブの滞在時間の総和。
        self.departure_sojourn_sum: float = 0.0
        # departure したジョブの総数。
        self.departure_count: int = 0
        # 信頼区間計算用に、記録期間を分割した子 SimStats のリスト。
        self.batches: List["SimStats"] = []
        # record_state 呼び出し回数 (= 記録されたイベント数. ウォームアップ除外の検証用)
        self.n_records: int = 0

    def record_state(self, state: Tuple[int, int, int], duration: float) -> None:
        """状態 (i, j, F) に duration だけ滞在したことを記録."""
        self.state_duration[state] += duration
        self.total_duration += duration
        self.n_records += 1

    def record_arrival_attempt(self, blocked: bool) -> None:
        """到着 attempt を記録 (ブロックされたか否か)."""
        self.arrival_attempts += 1
        if blocked:
            self.arrival_blocked += 1

    def record_departure(self, sojourn_time: float, count: int) -> None:
        """ジョブが departure したことを記録.

        Args:
            sojourn_time: 今回 departure した count 個のジョブの平均 sojourn time.
            count: departure したジョブ数 (バッチなら count = b).
        """
        self.departure_sojourn_sum += sojourn_time * count
        self.departure_count += count
