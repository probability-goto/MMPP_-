"""シミュレーション統計からの性能指標計算 (Predictive 拡張版).

mmpp_predictive.Metrics と同一の API (メソッド名) を提供する. 定常分布 pi
の代わりに SimStats (時間重み付き統計) から同じ指標を計算する.

計算はベース版 (mmpp_sim.metrics.SimMetrics) を継承し, 違う 2 点だけを上書きする:
状態タプルが (i, s, j, F) の 4 要素で, E[S] を setup_servers(i,j) からの導出では
なく明示的な状態変数 s の期待値として直接計算することと, energy_cost_paper が
params.C_b/C_s/C_i (PredictiveModelParameters が保持するコスト係数) を使うこと.
信頼区間 (`*_ci` 系) もベース版と同じく独立レプリケーション法で計算する.
"""
from typing import Tuple

from mmpp_sim.metrics import SimMetrics as BaseSimMetrics
from mmpp_sim.stats import SimStats


class SimMetrics(BaseSimMetrics):
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

    params は PredictiveModelParameters.
    """

    def _mean_setup(self, stats: SimStats) -> float:
        """E[S]: 明示的な状態変数 s の期待値."""
        return self._server_type_mean(stats, lambda st: st[1])

    def _cost_coefficients(self) -> Tuple[float, float, float]:
        p = self.params
        return p.C_b, p.C_s, p.C_i
