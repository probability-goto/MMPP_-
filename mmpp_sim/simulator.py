"""MMPP/M/c/SET-BATCH/delayoff の離散事象シミュレーション (DES) 本体.

mmpp パッケージ (理論解析実装) の正当性検証のため, 独立に実装する.
そのため busy_servers 等の補助関数もここで独立に再実装し,
mmpp.generator には依存しない. ModelParameters のみ共通で使う.

シミュレーション方式:
    現状態 (i, j, F) から発生しうる全遷移を「競合指数分布」として扱う
    (Gillespie 法/競合レート法). 各時刻で
        1. 全遷移の合計率 total_rate で待ち時間 dt ~ Exp(total_rate) を生成
        2. 発生するイベント種別を各遷移率に比例した確率で選択
    という 2 段階で「次の 1 イベント」だけを生成する. 全事象が指数分布
    (無記憶性) なので, これは真の CTMC 軌道と統計的に同一の分布を持つ.
"""
from collections import deque
from typing import Deque, List, Tuple

import numpy as np

from mmpp_sim.event import EventType, StepResult
from mmpp_sim.stats import SimStats


# ============================================================
# 補助関数 (Busy/Idle/Setup/Off サーバー数)
#
# mmpp.generator の同名関数と独立に再実装したもの (検証のため import しない).
# ============================================================

def busy_servers(i: int, j: int, b: int) -> int:
    """処理中サーバー数 B(i,j) = min(i, floor(j/b))."""
    return min(i, j // b)


def idle_servers(i: int, j: int, b: int) -> int:
    """アイドル (Delayoff 中) サーバー数 I(i,j) = i - B(i,j)."""
    return i - busy_servers(i, j, b)


def setup_servers(i: int, j: int, b: int, c: int) -> int:
    """セットアップ中サーバー数 S(i,j) = max(0, min(c, floor(j/b)) - i)."""
    target = min(c, j // b)
    return max(0, target - i)


def off_servers(i: int, j: int, b: int, c: int) -> int:
    """オフサーバー数 = c - i - S(i,j)."""
    return c - i - setup_servers(i, j, b, c)


class Simulator:
    """MMPP/M/c/SET-BATCH/delayoff モデルの DES シミュレータ.

    使用例:
        sim = Simulator(params, seed=42)
        stats = sim.run(warmup_events=100_000, measurement_events=1_000_000)
        metrics = SimMetrics(params, stats)

    Notes:
        sojourn time 追跡について:
        SERVICE_COMPLETION 時に FIFO キューから最古の b 個の到着時刻を pop して
        sojourn time を計算しているが, これは物理モデルとしては厳密ではない.
        指数分布の無記憶性により, B(i,j) > 1 の状態では実際に完了するのは
        B 個のバッチから一様ランダムに選ばれる 1 つであり, 必ずしも最古とは限らない.

        しかし集計 E[W] は影響を受けない. 任意の到着列 {a_k} と departure 列 {d_k}
        の間のどの 1 対 1 対応 σ についても
            Σ (d_k - a_σ(k)) = Σ d_k - Σ a_k (bijection 不変性)
        が成立するため, 個別ジョブへの帰属を FIFO で決めても実物理で決めても
        総和は同一で, したがって全体平均も同一である. これにより Little の公式
        との整合性も保たれる.

        FIFO を採用している理由は実装の単純さのみで, 統計的な意味付けはない.
    """

    def __init__(self, params, seed: int = 42) -> None:
        """
        Args:
            params: ModelParameters インスタンス.
            seed: 乱数生成器のシード (再現性確保のため全乱数使用箇所で共有する).
        """
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.state: Tuple[int, int, int] = (0, 0, 0)
        self.time: float = 0.0
        # 系内ジョブの到着時刻を FIFO で保持 (departure 時に sojourn time を計算するため)
        self.fifo: Deque[float] = deque()

    def reset(self) -> None:
        """状態を初期状態 (i=0, j=0, F=0), 時刻 0 にリセットする (乱数系列は継続)."""
        self.state = (0, 0, 0)
        self.time = 0.0
        self.fifo.clear()

    # ------------------------------------------------------------------
    # イベントメニュー構築
    # ------------------------------------------------------------------

    def _event_menu(
        self, state: Tuple[int, int, int]
    ) -> List[Tuple[float, EventType, dict]]:
        """状態 state から発生しうる全イベントとその率のリストを返す.

        各要素は (rate, event_type, metadata) の tuple.
        """
        i, j, F = state
        p = self.params
        B = busy_servers(i, j, p.b)
        I = idle_servers(i, j, p.b)
        S = setup_servers(i, j, p.b, p.c)

        candidates: List[Tuple[float, EventType, dict]] = []

        # 到着 (位相遷移を伴う場合を含む). j=K でもブロックされた attempt として発生させる
        # (P_block^arrival の統計を正しく取るため, 自己ループ (Fp=F かつ j=K) も候補に含める).
        for Fp in range(p.D_M):
            rate = p.C1[F, Fp]
            if rate > 0:
                candidates.append((rate, EventType.ARRIVAL, {"Fp": Fp}))

        # MMPP 内部遷移 (到着を伴わない位相遷移)
        for Fp in range(p.D_M):
            if Fp != F:
                rate = p.C0[F, Fp]
                if rate > 0:
                    candidates.append((rate, EventType.PHASE_TRANSITION, {"Fp": Fp}))

        # バッチサービス完了
        if B > 0:
            candidates.append((B * p.mu, EventType.SERVICE_COMPLETION, {}))

        # セットアップ完了
        if S > 0 and i < p.c:
            candidates.append((S * p.alpha, EventType.SETUP_COMPLETION, {}))

        # Delayoff タイムアウト
        if I > 0 and i > 0:
            candidates.append((I * p.beta, EventType.DELAYOFF_TIMEOUT, {}))

        return candidates

    # ------------------------------------------------------------------
    # 1 イベント実行
    # ------------------------------------------------------------------

    def step(self) -> StepResult:
        """次の 1 イベントを生成・適用し, その結果を返す."""
        old_state = self.state
        i, j, F = old_state
        p = self.params

        candidates = self._event_menu(old_state)
        total_rate = sum(rate for rate, _, _ in candidates)
        if total_rate <= 0:
            raise RuntimeError(f"状態 {old_state} から遷移可能なイベントがありません")

        dt = self.rng.exponential(1.0 / total_rate)

        # 競合レート法: 率に比例した確率で発生イベントを選択
        u = self.rng.random() * total_rate
        cum = 0.0
        rate, event_type, metadata = candidates[-1]
        for rate, event_type, metadata in candidates:
            cum += rate
            if u <= cum:
                break

        self.time += dt

        blocked = None
        departure_count = 0
        avg_sojourn = None

        if event_type == EventType.ARRIVAL:
            Fp = metadata["Fp"]
            if j < p.K:
                self.fifo.append(self.time)
                new_state = (i, j + 1, Fp)
                blocked = False
            else:
                new_state = (i, j, Fp)
                blocked = True
        elif event_type == EventType.SERVICE_COMPLETION:
            b = p.b
            sojourns = [self.time - self.fifo.popleft() for _ in range(b)]
            avg_sojourn = sum(sojourns) / b
            departure_count = b
            new_state = (i, j - b, F)
        elif event_type == EventType.SETUP_COMPLETION:
            new_state = (i + 1, j, F)
        elif event_type == EventType.DELAYOFF_TIMEOUT:
            new_state = (i - 1, j, F)
        elif event_type == EventType.PHASE_TRANSITION:
            Fp = metadata["Fp"]
            new_state = (i, j, Fp)
        else:
            raise ValueError(f"未知のイベント種別: {event_type}")

        self.state = new_state
        return StepResult(
            dt=dt,
            event_type=event_type,
            old_state=old_state,
            new_state=new_state,
            blocked=blocked,
            departure_count=departure_count,
            avg_sojourn=avg_sojourn,
        )

    # ------------------------------------------------------------------
    # シミュレーション実行
    # ------------------------------------------------------------------

    def run(
        self,
        warmup_events: int = 100_000,
        measurement_events: int = 1_000_000,
    ) -> SimStats:
        """シミュレーションを実行し, ウォームアップ後の統計を返す.

        Args:
            warmup_events: 統計を記録しないウォームアップイベント数.
            measurement_events: 統計を記録するイベント数.

        Returns:
            SimStats: 記録期間の統計. `warmup_duration` にウォームアップ期間の
                実時間が記録される (遅い時定数 1/beta 等に対する妥当性確認用).
                信頼区間計算には run_replications() を使うこと.
        """
        stats = SimStats()

        # ウォームアップ (統計は記録しない)
        for _ in range(warmup_events):
            self.step()
        stats.warmup_duration = self.time

        # 本計測
        for _ in range(measurement_events):
            result = self.step()

            stats.record_state(result.old_state, result.dt)

            if result.event_type == EventType.ARRIVAL:
                stats.record_arrival_attempt(result.blocked)
            elif result.event_type == EventType.SERVICE_COMPLETION:
                stats.record_departure(result.avg_sojourn, result.departure_count)

        return stats


# ============================================================
# 独立レプリケーション法
# ============================================================

def run_replications(
    params,
    n_reps: int = 20,
    seed0: int = 42,
    warmup_events: int = 100_000,
    measurement_events: int = 1_000_000,
) -> List[SimStats]:
    """独立レプリケーション法によるシミュレーション実行.

    n_reps 個の独立な Simulator インスタンスを生成し, それぞれ異なるシードで
    warmup_events + measurement_events を実行する. 得られた SimStats のリストを返す.
    信頼区間はこのリストを使って SimMetrics 側で計算する.

    Args:
        params: ModelParameters.
        n_reps: レプリケーション数 (推奨 20 以上).
        seed0: 開始シード. i 番目のレプリケーションは seed=seed0+i を使う.
        warmup_events: 各レプリケーションのウォームアップイベント数.
        measurement_events: 各レプリケーションの本計測イベント数.

    Returns:
        List[SimStats]: 各レプリケーションの統計. 長さ n_reps.
    """
    replications: List[SimStats] = []
    for k in range(n_reps):
        sim = Simulator(params, seed=seed0 + k)
        stats = sim.run(warmup_events=warmup_events, measurement_events=measurement_events)
        replications.append(stats)
    return replications
