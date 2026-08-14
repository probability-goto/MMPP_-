"""MMPP/M/c/SET-BATCH/Delayoff/Predictive の離散事象シミュレーション (DES) 本体.

mmpp_predictive パッケージ (理論解析実装) の正当性検証のため, 独立に実装する.
そのため s に関する補助関数 (required_setup_count, setup_target_delta) も
ここで独立に再実装し, mmpp_predictive.generator/state_space には依存しない.
PredictiveModelParameters のみ共通で使う.

busy_servers/idle_servers (i, j のみに依存し Predictive 拡張と無関係な
共通ロジック) は mmpp_sim から再利用する (ベースシミュレータとの構造統一).

シミュレーション方式 (mmpp_sim.simulator と同じ Gillespie 法):
    現状態 (i, s, j, F) から発生しうる全遷移を「競合指数分布」として扱う.
    各時刻で
        1. 全遷移の合計率 total_rate で待ち時間 dt ~ Exp(total_rate) を生成
        2. 発生するイベント種別を各遷移率に比例した確率で選択
    という 2 段階で「次の 1 イベント」だけを生成する.

状態表現について:
    理論モデル (mmpp_predictive.state_space) の状態空間は (i, s, j, F) の
    4 タプルであり, s (セットアップ中サーバー数) はベースモデルと異なり
    (i, j) のみからは定まらない明示的な状態変数である (事前セットアップに
    より, キューの実需要を超えてセットアップが先行しうるため). このシミュ
    レータも同じ状態表現を採用する. サーバーは統計的に同一なので個々の
    サーバー識別子を追跡する必要はなく (Predictive 拡張の 2 機構はいずれも
    「何台が SETUP/IDLE 状態か」という集約カウントのみに依存する), 集約
    状態の競合指数分布シミュレーションで理論モデルと厳密に対応する.
"""
from collections import deque
from typing import Deque, List, Tuple

import numpy as np

from mmpp_sim.simulator import busy_servers, idle_servers

from mmpp_predictive_sim.event import EventType, StepResult
from mmpp_sim.stats import SimStats


# ============================================================
# Predictive 拡張の補助関数 (s に関する反応的/事前セットアップ判定)
#
# mmpp_predictive.state_space の同名関数と独立に再実装したもの
# (検証のため import しない).
# ============================================================

def required_setup_count(i: int, j: int, b: int, c: int) -> int:
    """反応的セットアップの目標値: 現在の (i,j) を処理するのに必要な s.

    ベースモデルの setup_servers(i,j,b,c) と同じ式
    (target = min(c, floor(j/b)); 目標 = max(0, target - i)).
    """
    target = min(c, j // b)
    return max(0, target - i)


def setup_target_delta(i: int, s: int, n_target: int, c: int) -> int:
    """事前セットアップ (規則 P1, F: 0->1 発生時) の発動台数 Delta_eff.

    Delta = max(n_target - i - s, 0)
    Delta_eff = min(Delta, c - i - s)  (オフサーバー不足によるクリップ)
    """
    delta = max(n_target - i - s, 0)
    return min(delta, c - i - s)


class PredictiveSimulator:
    """MMPP/M/c/SET-BATCH/Delayoff/Predictive モデルの DES シミュレータ.

    使用例:
        sim = PredictiveSimulator(params, seed=42)
        stats = sim.run(warmup_events=100_000, measurement_events=1_000_000)
        metrics = SimMetrics(params, stats)

    Notes:
        sojourn time の扱いは mmpp_sim.Simulator と同じ FIFO 近似を用いる
        (総和 = departure 時刻の和 - 到着時刻の和 は帰属方法によらず不変な
        ため, 平均 E[W] には影響しない. 詳細は mmpp_sim.simulator の
        docstring 参照).
    """

    def __init__(self, params, seed: int = 42) -> None:
        """
        Args:
            params: PredictiveModelParameters インスタンス.
            seed: 乱数生成器のシード (再現性確保のため全乱数使用箇所で共有する).
        """
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.state: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.time: float = 0.0
        self.fifo: Deque[float] = deque()

    def reset(self) -> None:
        """状態を初期状態 (i=0, s=0, j=0, F=0), 時刻 0 にリセットする (乱数系列は継続)."""
        self.state = (0, 0, 0, 0)
        self.time = 0.0
        self.fifo.clear()

    # ------------------------------------------------------------------
    # イベントメニュー構築
    # ------------------------------------------------------------------

    def _event_menu(
        self, state: Tuple[int, int, int, int]
    ) -> List[Tuple[float, EventType, dict]]:
        """状態 state から発生しうる全イベントとその率のリストを返す.

        各要素は (rate, event_type, metadata) の tuple.
        """
        i, s, j, F = state
        p = self.params
        B = busy_servers(i, j, p.b)
        I = idle_servers(i, j, p.b)

        candidates: List[Tuple[float, EventType, dict]] = []

        # 到着 (位相遷移を伴う場合を含む). j=K でもブロックされた attempt として発生させる
        # (P_block^arrival の統計を正しく取るため, 自己ループ (Fp=F かつ j=K) も候補に含める).
        for Fp in range(p.D_M):
            rate = p.C1[F, Fp]
            if rate > 0:
                candidates.append((rate, EventType.ARRIVAL, {"Fp": Fp}))

        # MMPP 内部遷移 (到着を伴わない位相遷移). F=0->1 は事前セットアップを伴う.
        for Fp in range(p.D_M):
            if Fp != F:
                rate = p.C0[F, Fp]
                if rate > 0:
                    candidates.append((rate, EventType.PHASE_TRANSITION, {"Fp": Fp}))

        # バッチサービス完了
        if B > 0:
            candidates.append((B * p.mu, EventType.SERVICE_COMPLETION, {}))

        # セットアップ完了 (s が明示的な状態変数なので i<c の付帯条件は不要:
        # i + s <= c かつ s > 0 は自動的に i < c を含意する)
        if s > 0:
            candidates.append((s * p.alpha, EventType.SETUP_COMPLETION, {}))

        # Delayoff タイムアウト (F=0 では gamma 倍に加速)
        if I > 0:
            beta_F = p.gamma * p.beta if F == 0 else p.beta
            candidates.append((I * beta_F, EventType.DELAYOFF_TIMEOUT, {}))

        return candidates

    # ------------------------------------------------------------------
    # 1 イベント実行
    # ------------------------------------------------------------------

    def step(self) -> StepResult:
        """次の 1 イベントを生成・適用し, その結果を返す."""
        old_state = self.state
        i, s, j, F = old_state
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
        proactive_setup_delta = 0

        if event_type == EventType.ARRIVAL:
            Fp = metadata["Fp"]
            if j < p.K:
                j_new = j + 1
                s_needed = required_setup_count(i, j_new, p.b, p.c)
                s_new = s + 1 if (s_needed > s and s < p.c - i) else s
                self.fifo.append(self.time)
                new_state = (i, s_new, j_new, Fp)
                blocked = False
            else:
                new_state = (i, s, j, Fp)
                blocked = True
        elif event_type == EventType.SERVICE_COMPLETION:
            b = p.b
            sojourns = [self.time - self.fifo.popleft() for _ in range(b)]
            avg_sojourn = sum(sojourns) / b
            departure_count = b
            j_new = j - b
            s_needed = required_setup_count(i, j_new, p.b, p.c)
            s_new = s - 1 if s_needed < s else s
            new_state = (i, s_new, j_new, F)
        elif event_type == EventType.SETUP_COMPLETION:
            new_state = (i + 1, s - 1, j, F)
        elif event_type == EventType.DELAYOFF_TIMEOUT:
            new_state = (i - 1, s, j, F)
        elif event_type == EventType.PHASE_TRANSITION:
            Fp = metadata["Fp"]
            if F == 0 and Fp == 1:
                delta_eff = setup_target_delta(i, s, p.n_target, p.c)
                new_state = (i, s + delta_eff, j, Fp)
                proactive_setup_delta = delta_eff
            else:
                new_state = (i, s, j, Fp)
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
            proactive_setup_delta=proactive_setup_delta,
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
            SimStats: 記録期間の統計 (mmpp_sim.stats.SimStats をそのまま再利用.
                状態タプルの長さに依存しない実装のため互換). 事前セットアップ
                の発動回数・平均台数は self.proactive_setup_count /
                self.proactive_setup_delta_sum に別途集計する.
        """
        stats = SimStats()
        self.proactive_setup_count = 0
        self.proactive_setup_delta_sum = 0

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
            elif result.event_type == EventType.PHASE_TRANSITION and result.proactive_setup_delta > 0:
                self.proactive_setup_count += 1
                self.proactive_setup_delta_sum += result.proactive_setup_delta

        return stats
