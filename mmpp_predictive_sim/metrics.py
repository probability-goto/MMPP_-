"""シミュレーション統計からの性能指標計算 (Predictive 拡張版).

mmpp_predictive.Metrics と同一の API (メソッド名) を提供する. 定常分布 pi
の代わりに SimStats (時間重み付き統計) から同じ指標を計算する.

ベース版 (mmpp_sim.metrics.SimMetrics) との違いは, 状態タプルが
(i, s, j, F) の 4 要素であること (E[S] は setup_servers(i,j) 関数からの
導出ではなく, 明示的な状態変数 s の期待値として直接計算する) と,
energy_cost_paper が params.C_b/C_s/C_i (PredictiveModelParameters が
保持するコスト係数) を使うことの 2 点のみ.

信頼区間は独立レプリケーション法で計算する: 複数の独立な SimStats
(run_replications() の戻り値) それぞれについて指標を計算し, その値の
リストから t 分布近似で 95% CI を求める. `*_ci` 系メソッドはレプリケーション
(SimStats のリスト) が与えられている場合のみ使用できる.
"""
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy import stats as scipy_stats

from mmpp_sim.simulator import busy_servers, idle_servers
from mmpp_sim.stats import SimStats


class SimMetrics:
    """シミュレーション統計から性能指標を計算するクラス (Predictive 拡張版).

    使用例:
        # 点推定のみ (単一 SimStats, 開発中の高速確認向け)
        stats = PredictiveSimulator(params, seed=42).run(...)
        metrics = SimMetrics(params, stats)
        print(metrics.blocking_probability())

        # 点推定 + 信頼区間 (独立レプリケーション, 本番向け)
        replications = run_replications(params, n_reps=20)
        metrics = SimMetrics(params, replications)
        print(metrics.blocking_probability_ci())
    """

    def __init__(self, params, stats: Union[SimStats, List[SimStats]]):
        """
        Args:
            params: PredictiveModelParameters インスタンス.
            stats: 単一 SimStats または SimStats のリスト.
                リストの場合は独立レプリケーションと解釈し, CI 計算が可能になる.
        """
        self.params = params
        if isinstance(stats, SimStats):
            self.replications = [stats]
            self._is_replicated = False
        elif isinstance(stats, list) and all(isinstance(s, SimStats) for s in stats):
            self.replications = stats
            self._is_replicated = True
        else:
            raise TypeError(
                "stats は SimStats または SimStats のリストでなければならない"
            )

    def _aggregate_stats(self) -> SimStats:
        """全レプリケーションを合算した SimStats を返す (点推定用)."""
        agg = SimStats()
        for rep in self.replications:
            for state, dur in rep.state_duration.items():
                agg.state_duration[state] += dur
            agg.total_duration += rep.total_duration
            agg.arrival_attempts += rep.arrival_attempts
            agg.arrival_blocked += rep.arrival_blocked
            agg.departure_sojourn_sum += rep.departure_sojourn_sum
            agg.departure_count += rep.departure_count
        return agg

    def _require_replicated(self, method_name: str) -> None:
        if not self._is_replicated:
            raise RuntimeError(
                f"{method_name} には SimStats のリスト (レプリケーション) が必要. "
                "run_replications() の戻り値を SimMetrics に渡してください."
            )

    # ---------- 点推定 (内部ヘルパー: 任意の SimStats に対して計算) ----------

    def _blocking_probability(self, stats: SimStats) -> float:
        p = self.params
        total = sum(
            dur for (i, s, j, F), dur in stats.state_duration.items() if j == p.K
        )
        return total / stats.total_duration

    def _mean_queue_length(self, stats: SimStats) -> float:
        total = sum(j * dur for (i, s, j, F), dur in stats.state_duration.items())
        return total / stats.total_duration

    def _server_type_mean(self, stats: SimStats, count_fn) -> float:
        total = sum(
            count_fn(i, s, j) * dur for (i, s, j, F), dur in stats.state_duration.items()
        )
        return total / stats.total_duration

    def _mean_busy(self, stats: SimStats) -> float:
        p = self.params
        return self._server_type_mean(stats, lambda i, s, j: busy_servers(i, j, p.b))

    def _mean_idle(self, stats: SimStats) -> float:
        p = self.params
        return self._server_type_mean(stats, lambda i, s, j: idle_servers(i, j, p.b))

    def _mean_setup(self, stats: SimStats) -> float:
        return self._server_type_mean(stats, lambda i, s, j: s)

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

    def _energy_cost_paper(self, stats: SimStats) -> float:
        p = self.params
        return (
            p.C_b * self._mean_busy(stats)
            + p.C_s * self._mean_setup(stats)
            + p.C_i * self._mean_idle(stats)
        )

    def _erp_paper(self, stats: SimStats) -> float:
        return self._mean_waiting_time(stats) * self._energy_cost_paper(stats)

    # ---------- 点推定 (公開 API, mmpp_predictive.Metrics と同一名) ----------
    # 全レプリケーションを合算した SimStats に対して計算する.

    def blocking_probability(self) -> float:
        """P_block = 状態 j=K の総滞在時間 / 総記録時間."""
        return self._blocking_probability(self._aggregate_stats())

    def mean_queue_length(self) -> float:
        """E[j] = sum_states j * tau_state / T."""
        return self._mean_queue_length(self._aggregate_stats())

    def mean_busy(self) -> float:
        """E[B]: 平均処理中サーバー数."""
        return self._mean_busy(self._aggregate_stats())

    def mean_idle(self) -> float:
        """E[I]: 平均アイドル (Delayoff 中) サーバー数."""
        return self._mean_idle(self._aggregate_stats())

    def mean_setup(self) -> float:
        """E[S]: 平均セットアップ中サーバー数 (明示的な状態変数 s の期待値)."""
        return self._mean_setup(self._aggregate_stats())

    def mean_off(self) -> float:
        """E[Off]: 平均オフサーバー数 (= c - E[B] - E[I] - E[S])."""
        return self._mean_off(self._aggregate_stats())

    def effective_arrival_rate(self) -> float:
        """lambda_eff = 非ブロック到着数 / 総記録時間."""
        return self._effective_arrival_rate(self._aggregate_stats())

    def arrival_blocking_probability(self) -> float:
        """P_block^arrival = ブロック数 / 到着 attempt 総数."""
        return self._arrival_blocking_probability(self._aggregate_stats())

    def mean_waiting_time(self) -> float:
        """E[W] = departure 済みジョブの sojourn time の平均."""
        return self._mean_waiting_time(self._aggregate_stats())

    def utilization(self) -> float:
        """rho = E[B] / c."""
        return self._utilization(self._aggregate_stats())

    def energy_cost(
        self,
        c_busy: float = 1.0,
        c_idle: float = 0.6,
        c_setup: float = 1.0,
        c_off: float = 0.0,
    ) -> float:
        """電力コスト = c_busy*E[B] + c_idle*E[I] + c_setup*E[S] + c_off*E[Off]."""
        return self._energy_cost(self._aggregate_stats(), c_busy, c_idle, c_setup, c_off)

    def energy_cost_paper(self) -> float:
        """先行研究 (Le-Anh & Phung-Duc 2025) 5.2 節の公式に基づく電力コスト
        (params.C_b/C_s/C_i を使用)."""
        return self._energy_cost_paper(self._aggregate_stats())

    def erp_paper(self) -> float:
        """ERP = E[W] * Cost (先行研究定義, paper cost formula 使用)."""
        return self._erp_paper(self._aggregate_stats())

    def all_metrics(self) -> Dict[str, float]:
        """全指標を辞書として返す (mmpp_predictive.Metrics.all_metrics と同一キー)."""
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
            "cost_paper": self.energy_cost_paper(),
            "ERP_paper": self.erp_paper(),
        }

    # ---------- 独立レプリケーション法による信頼区間 ----------

    def _ci_from_values(self, values: list) -> Tuple[float, float, float]:
        """レプリケーションごとの指標値のリストから (点推定, CI下限, CI上限) を計算する.

        inf/nan を含むレプリケーションは除外する. 除外が起きた場合は警告する
        (通常はレプリケーション長 (measurement_events) 不足のシグナル).
        """
        arr_raw = np.asarray(values, dtype=float)
        finite_mask = np.isfinite(arr_raw)
        n_excluded = int((~finite_mask).sum())
        if n_excluded > 0:
            warnings.warn(
                f"{n_excluded}/{arr_raw.size} レプリケーションが inf/nan のため除外されました. "
                "measurement_events を増やすか, パラメータ設定を確認してください.",
                RuntimeWarning,
                stacklevel=2,
            )

        arr = arr_raw[finite_mask]
        n = arr.size
        if n == 0:
            return float("nan"), float("nan"), float("nan")
        mean = float(arr.mean())
        if n < 2:
            return mean, mean, mean

        se = float(arr.std(ddof=1) / np.sqrt(n))
        if se == 0.0:
            warnings.warn(
                f"標本標準偏差が 0. 全レプリケーションで値が完全一致しています "
                f"(mean={mean:.6g}). CI 幅が 0 になるため注意 (低負荷など, 指標が "
                f"該当測定時間内にほぼ発生しないパラメータ設定でも起こりうる. "
                f"意図しない場合はパラメータ設定やシミュレーション実装を確認してください).",
                RuntimeWarning,
                stacklevel=2,
            )
            return mean, mean, mean

        t_val = float(scipy_stats.t.ppf(0.975, df=n - 1))
        half_width = t_val * se
        return mean, mean - half_width, mean + half_width

    def blocking_probability_ci(self) -> Tuple[float, float, float]:
        """P_block の (点推定, CI下限, CI上限)."""
        self._require_replicated("blocking_probability_ci")
        values = [self._blocking_probability(rep) for rep in self.replications]
        return self._ci_from_values(values)

    def mean_queue_length_ci(self) -> Tuple[float, float, float]:
        """E[j] の (点推定, CI下限, CI上限)."""
        self._require_replicated("mean_queue_length_ci")
        values = [self._mean_queue_length(rep) for rep in self.replications]
        return self._ci_from_values(values)

    def mean_busy_ci(self) -> Tuple[float, float, float]:
        """E[B] の (点推定, CI下限, CI上限)."""
        self._require_replicated("mean_busy_ci")
        values = [self._mean_busy(rep) for rep in self.replications]
        return self._ci_from_values(values)

    def mean_idle_ci(self) -> Tuple[float, float, float]:
        """E[I] の (点推定, CI下限, CI上限)."""
        self._require_replicated("mean_idle_ci")
        values = [self._mean_idle(rep) for rep in self.replications]
        return self._ci_from_values(values)

    def mean_setup_ci(self) -> Tuple[float, float, float]:
        """E[S] の (点推定, CI下限, CI上限)."""
        self._require_replicated("mean_setup_ci")
        values = [self._mean_setup(rep) for rep in self.replications]
        return self._ci_from_values(values)

    def mean_off_ci(self) -> Tuple[float, float, float]:
        """E[Off] の (点推定, CI下限, CI上限)."""
        self._require_replicated("mean_off_ci")
        values = [self._mean_off(rep) for rep in self.replications]
        return self._ci_from_values(values)

    def effective_arrival_rate_ci(self) -> Tuple[float, float, float]:
        """lambda_eff の (点推定, CI下限, CI上限)."""
        self._require_replicated("effective_arrival_rate_ci")
        values = [self._effective_arrival_rate(rep) for rep in self.replications]
        return self._ci_from_values(values)

    def arrival_blocking_probability_ci(self) -> Tuple[float, float, float]:
        """P_block^arrival の (点推定, CI下限, CI上限)."""
        self._require_replicated("arrival_blocking_probability_ci")
        values = [self._arrival_blocking_probability(rep) for rep in self.replications]
        return self._ci_from_values(values)

    def mean_waiting_time_ci(self) -> Tuple[float, float, float]:
        """E[W] の (点推定, CI下限, CI上限)."""
        self._require_replicated("mean_waiting_time_ci")
        values = [self._mean_waiting_time(rep) for rep in self.replications]
        return self._ci_from_values(values)

    def utilization_ci(self) -> Tuple[float, float, float]:
        """rho の (点推定, CI下限, CI上限)."""
        self._require_replicated("utilization_ci")
        values = [self._utilization(rep) for rep in self.replications]
        return self._ci_from_values(values)

    def energy_cost_ci(
        self,
        c_busy: float = 1.0,
        c_idle: float = 0.6,
        c_setup: float = 1.0,
        c_off: float = 0.0,
    ) -> Tuple[float, float, float]:
        """energy_cost の (点推定, CI下限, CI上限)."""
        self._require_replicated("energy_cost_ci")
        values = [
            self._energy_cost(rep, c_busy, c_idle, c_setup, c_off)
            for rep in self.replications
        ]
        return self._ci_from_values(values)

    def energy_cost_paper_ci(self) -> Tuple[float, float, float]:
        """energy_cost_paper の (点推定, CI下限, CI上限)."""
        self._require_replicated("energy_cost_paper_ci")
        values = [self._energy_cost_paper(rep) for rep in self.replications]
        return self._ci_from_values(values)

    def erp_paper_ci(self) -> Tuple[float, float, float]:
        """erp_paper の (点推定, CI下限, CI上限)."""
        self._require_replicated("erp_paper_ci")
        values = [self._erp_paper(rep) for rep in self.replications]
        return self._ci_from_values(values)

    def all_metrics_ci(self) -> Dict[str, Tuple[float, float, float]]:
        """全指標の (点推定, CI下限, CI上限) を辞書として返す."""
        return {
            "P_block": self.blocking_probability_ci(),
            "P_block_arrival": self.arrival_blocking_probability_ci(),
            "E[j]": self.mean_queue_length_ci(),
            "E[B]": self.mean_busy_ci(),
            "E[I]": self.mean_idle_ci(),
            "E[S]": self.mean_setup_ci(),
            "E[Off]": self.mean_off_ci(),
            "lambda_eff": self.effective_arrival_rate_ci(),
            "E[W]": self.mean_waiting_time_ci(),
            "rho": self.utilization_ci(),
            "energy_cost": self.energy_cost_ci(),
            "cost_paper": self.energy_cost_paper_ci(),
            "ERP_paper": self.erp_paper_ci(),
        }
