"""シミュレーション統計からの性能指標計算.

mmpp.Metrics と同一の API (メソッド名) を提供する. 定常分布 pi の代わりに
SimStats (時間重み付き統計) から同じ指標を計算する.

加えて batch means 法による 95% 信頼区間を返す `*_ci` 系メソッドを持つ.
"""
from typing import Dict, Tuple

import numpy as np
from scipy import stats as scipy_stats

from mmpp_sim.simulator import busy_servers, idle_servers, setup_servers
from mmpp_sim.stats import SimStats


class SimMetrics:
    """シミュレーション統計から性能指標を計算するクラス.

    使用例:
        metrics = SimMetrics(params, stats)
        print(metrics.blocking_probability())
        print(metrics.blocking_probability_ci())
        print(metrics.all_metrics())
    """

    def __init__(self, params, stats: SimStats):
        """
        Args:
            params: ModelParameters インスタンス.
            stats: Simulator.run() が返す SimStats インスタンス.
        """
        self.params = params
        self.stats = stats

    # ---------- 点推定 (内部ヘルパー: 任意の SimStats に対して計算) ----------

    def _blocking_probability(self, stats: SimStats) -> float:
        p = self.params
        total = sum(
            dur for (i, j, F), dur in stats.state_duration.items() if j == p.K
        )
        return total / stats.total_duration

    def _mean_queue_length(self, stats: SimStats) -> float:
        total = sum(j * dur for (i, j, F), dur in stats.state_duration.items())
        return total / stats.total_duration

    def _server_type_mean(self, stats: SimStats, count_fn) -> float:
        total = sum(
            count_fn(i, j) * dur for (i, j, F), dur in stats.state_duration.items()
        )
        return total / stats.total_duration

    def _mean_busy(self, stats: SimStats) -> float:
        p = self.params
        return self._server_type_mean(stats, lambda i, j: busy_servers(i, j, p.b))

    def _mean_idle(self, stats: SimStats) -> float:
        p = self.params
        return self._server_type_mean(stats, lambda i, j: idle_servers(i, j, p.b))

    def _mean_setup(self, stats: SimStats) -> float:
        p = self.params
        return self._server_type_mean(
            stats, lambda i, j: setup_servers(i, j, p.b, p.c)
        )

    def _mean_off(self, stats: SimStats) -> float:
        return (
            self.params.c
            - self._mean_busy(stats)
            - self._mean_idle(stats)
            - self._mean_setup(stats)
        )

    def _effective_arrival_rate(self, stats: SimStats) -> float:
        admitted = stats.arrival_attempts - stats.arrival_blocked
        return admitted / stats.total_duration

    def _arrival_blocking_probability(self, stats: SimStats) -> float:
        if stats.arrival_attempts == 0:
            return float("nan")
        return stats.arrival_blocked / stats.arrival_attempts

    def _mean_waiting_time(self, stats: SimStats) -> float:
        if stats.departure_count == 0:
            return float("inf")
        return stats.departure_sojourn_sum / stats.departure_count

    def _utilization(self, stats: SimStats) -> float:
        return self._mean_busy(stats) / self.params.c

    def _energy_cost(
        self,
        stats: SimStats,
        c_busy: float = 1.0,
        c_idle: float = 0.6,
        c_setup: float = 1.0,
        c_off: float = 0.0,
    ) -> float:
        return (
            c_busy * self._mean_busy(stats)
            + c_idle * self._mean_idle(stats)
            + c_setup * self._mean_setup(stats)
            + c_off * self._mean_off(stats)
        )

    # ---------- 点推定 (公開 API, mmpp.Metrics と同一名) ----------

    def blocking_probability(self) -> float:
        """P_block = 状態 j=K の総滞在時間 / 総記録時間."""
        return self._blocking_probability(self.stats)

    def mean_queue_length(self) -> float:
        """E[j] = sum_states j * tau_state / T."""
        return self._mean_queue_length(self.stats)

    def mean_busy(self) -> float:
        """E[B]: 平均処理中サーバー数."""
        return self._mean_busy(self.stats)

    def mean_idle(self) -> float:
        """E[I]: 平均アイドル (Delayoff 中) サーバー数."""
        return self._mean_idle(self.stats)

    def mean_setup(self) -> float:
        """E[S]: 平均セットアップ中サーバー数."""
        return self._mean_setup(self.stats)

    def mean_off(self) -> float:
        """E[Off]: 平均オフサーバー数 (= c - E[B] - E[I] - E[S])."""
        return self._mean_off(self.stats)

    def effective_arrival_rate(self) -> float:
        """lambda_eff = 非ブロック到着数 / 総記録時間."""
        return self._effective_arrival_rate(self.stats)

    def arrival_blocking_probability(self) -> float:
        """P_block^arrival = ブロック数 / 到着 attempt 総数."""
        return self._arrival_blocking_probability(self.stats)

    def mean_waiting_time(self) -> float:
        """E[W] = departure 済みジョブの sojourn time の平均."""
        return self._mean_waiting_time(self.stats)

    def utilization(self) -> float:
        """rho = E[B] / c."""
        return self._utilization(self.stats)

    def energy_cost(
        self,
        c_busy: float = 1.0,
        c_idle: float = 0.6,
        c_setup: float = 1.0,
        c_off: float = 0.0,
    ) -> float:
        """電力コスト = c_busy*E[B] + c_idle*E[I] + c_setup*E[S] + c_off*E[Off]."""
        return self._energy_cost(self.stats, c_busy, c_idle, c_setup, c_off)

    def all_metrics(self) -> Dict[str, float]:
        """全指標を辞書として返す (mmpp.Metrics.all_metrics と同一キー)."""
        return {
            "P_block": self.blocking_probability(),
            "P_block_arrival": self.arrival_blocking_probability(),
            "E[j]": self.mean_queue_length(),
            "E[B]": self.mean_busy(),
            "E[I]": self.mean_idle(),
            "E[S]": self.mean_setup(),
            "E[Off]": self.mean_off(),
            "lambda_eff": self.effective_arrival_rate(),
            "E[W]": self.mean_waiting_time(),
            "rho": self.utilization(),
            "energy_cost": self.energy_cost(),
        }

    # ---------- batch means 法による信頼区間 ----------

    def _get_batches(self, n_batches: int) -> list:
        """記録済みバッチを取得する (Simulator.run の n_batches と一致必須)."""
        batches = self.stats.batches
        if not batches:
            raise ValueError(
                "SimStats に batches が記録されていません. "
                "Simulator.run() の戻り値を使ってください."
            )
        if len(batches) != n_batches:
            raise ValueError(
                f"n_batches={n_batches} が Simulator.run() 実行時の "
                f"バッチ数 {len(batches)} と一致しません."
            )
        return batches

    def _ci_from_values(self, values: list) -> Tuple[float, float, float]:
        """バッチごとの指標値のリストから (点推定, CI下限, CI上限) を計算する.

        batch means 法: 各バッチの指標値を独立な標本とみなし,
        標本平均の標準誤差 (SE) を用いて t 分布近似で 95% CI を求める.
        """
        arr = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
        n = arr.size
        mean = float(arr.mean())
        if n < 2:
            return mean, mean, mean
        se = float(arr.std(ddof=1) / np.sqrt(n))
        if se == 0.0:
            return mean, mean, mean
        t_val = float(scipy_stats.t.ppf(0.975, df=n - 1))
        half_width = t_val * se
        return mean, mean - half_width, mean + half_width

    def blocking_probability_ci(
        self, n_batches: int = 30
    ) -> Tuple[float, float, float]:
        """P_block の (点推定, CI下限, CI上限)."""
        batches = self._get_batches(n_batches)
        values = [self._blocking_probability(b) for b in batches]
        return self._ci_from_values(values)

    def mean_queue_length_ci(self, n_batches: int = 30) -> Tuple[float, float, float]:
        """E[j] の (点推定, CI下限, CI上限)."""
        batches = self._get_batches(n_batches)
        values = [self._mean_queue_length(b) for b in batches]
        return self._ci_from_values(values)

    def mean_busy_ci(self, n_batches: int = 30) -> Tuple[float, float, float]:
        """E[B] の (点推定, CI下限, CI上限)."""
        batches = self._get_batches(n_batches)
        values = [self._mean_busy(b) for b in batches]
        return self._ci_from_values(values)

    def mean_idle_ci(self, n_batches: int = 30) -> Tuple[float, float, float]:
        """E[I] の (点推定, CI下限, CI上限)."""
        batches = self._get_batches(n_batches)
        values = [self._mean_idle(b) for b in batches]
        return self._ci_from_values(values)

    def mean_setup_ci(self, n_batches: int = 30) -> Tuple[float, float, float]:
        """E[S] の (点推定, CI下限, CI上限)."""
        batches = self._get_batches(n_batches)
        values = [self._mean_setup(b) for b in batches]
        return self._ci_from_values(values)

    def mean_off_ci(self, n_batches: int = 30) -> Tuple[float, float, float]:
        """E[Off] の (点推定, CI下限, CI上限)."""
        batches = self._get_batches(n_batches)
        values = [self._mean_off(b) for b in batches]
        return self._ci_from_values(values)

    def effective_arrival_rate_ci(
        self, n_batches: int = 30
    ) -> Tuple[float, float, float]:
        """lambda_eff の (点推定, CI下限, CI上限)."""
        batches = self._get_batches(n_batches)
        values = [self._effective_arrival_rate(b) for b in batches]
        return self._ci_from_values(values)

    def arrival_blocking_probability_ci(
        self, n_batches: int = 30
    ) -> Tuple[float, float, float]:
        """P_block^arrival の (点推定, CI下限, CI上限)."""
        batches = self._get_batches(n_batches)
        values = [self._arrival_blocking_probability(b) for b in batches]
        return self._ci_from_values(values)

    def mean_waiting_time_ci(self, n_batches: int = 30) -> Tuple[float, float, float]:
        """E[W] の (点推定, CI下限, CI上限)."""
        batches = self._get_batches(n_batches)
        values = [self._mean_waiting_time(b) for b in batches]
        return self._ci_from_values(values)

    def utilization_ci(self, n_batches: int = 30) -> Tuple[float, float, float]:
        """rho の (点推定, CI下限, CI上限)."""
        batches = self._get_batches(n_batches)
        values = [self._utilization(b) for b in batches]
        return self._ci_from_values(values)

    def energy_cost_ci(
        self,
        n_batches: int = 30,
        c_busy: float = 1.0,
        c_idle: float = 0.6,
        c_setup: float = 1.0,
        c_off: float = 0.0,
    ) -> Tuple[float, float, float]:
        """energy_cost の (点推定, CI下限, CI上限)."""
        batches = self._get_batches(n_batches)
        values = [
            self._energy_cost(b, c_busy, c_idle, c_setup, c_off) for b in batches
        ]
        return self._ci_from_values(values)

    def all_metrics_ci(self, n_batches: int = 30) -> Dict[str, Tuple[float, float, float]]:
        """全指標の (点推定, CI下限, CI上限) を辞書として返す."""
        return {
            "P_block": self.blocking_probability_ci(n_batches),
            "P_block_arrival": self.arrival_blocking_probability_ci(n_batches),
            "E[j]": self.mean_queue_length_ci(n_batches),
            "E[B]": self.mean_busy_ci(n_batches),
            "E[I]": self.mean_idle_ci(n_batches),
            "E[S]": self.mean_setup_ci(n_batches),
            "E[Off]": self.mean_off_ci(n_batches),
            "lambda_eff": self.effective_arrival_rate_ci(n_batches),
            "E[W]": self.mean_waiting_time_ci(n_batches),
            "rho": self.utilization_ci(n_batches),
            "energy_cost": self.energy_cost_ci(n_batches),
        }
