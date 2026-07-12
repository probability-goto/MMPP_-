"""性能指標の計算.

定常分布 π から以下の指標を導出:
    - ブロック確率 P_block
    - 平均系内ジョブ数 E[j]
    - サーバー種別ごとの平均数 (E[B], E[I], E[S], E[Off])
    - 有効到着率 λ_eff
    - 平均滞在時間 E[W] (Little の公式)
    - サーバー利用率 ρ
    - エネルギーコスト
"""
from typing import Dict
import numpy as np

from mmpp.state_space import StateSpace
from mmpp.generator import (
    busy_servers,
    idle_servers,
    setup_servers,
)


class Metrics:
    """定常分布から性能指標を計算するクラス.

    使用例:
        metrics = Metrics(params, pi)
        print(metrics.blocking_probability())
        print(metrics.all_metrics())
    """

    def __init__(self, params, pi: np.ndarray):
        """
        Args:
            params: ModelParameters インスタンス.
            pi: 定常分布 (1D 配列, サイズ N).
        """
        self.params = params
        self.pi = np.asarray(pi)
        self.ss = StateSpace(params)
        if self.pi.shape != (params.N,):
            raise ValueError(f"pi shape {self.pi.shape} != ({params.N},)")

    # ---------- ブロック確率 ----------

    def blocking_probability(self) -> float:
        """P_block = P(j = K) = sum_{i, F} π(i, K, F).

        バッファ満杯時に到着したジョブが失われる確率.
        """
        p = self.params
        block = self.ss.block_slice(p.K)
        return float(self.pi[block].sum())

    # ---------- 平均量 ----------

    def mean_queue_length(self) -> float:
        """E[j] = sum_{i,j,F} j * π(i,j,F).

        平均系内ジョブ数 (待機中 + 処理中を含む).
        """
        p = self.params
        # レベル j のマージナル確率
        pi_j = np.array(
            [self.pi[self.ss.block_slice(j)].sum() for j in range(p.K + 1)]
        )
        j_values = np.arange(p.K + 1)
        return float((j_values * pi_j).sum())

    def _server_type_mean(self, count_fn) -> float:
        """サーバー数系指標の共通計算.

        count_fn(i, j) がスカラーを返す関数.
        """
        p = self.params
        total = 0.0
        for (i, j, F) in self.ss.iter_states():
            total += count_fn(i, j) * self.pi[self.ss.index(i, j, F)]
        return float(total)

    def mean_busy(self) -> float:
        """E[B]: 平均処理中サーバー数."""
        p = self.params
        return self._server_type_mean(lambda i, j: busy_servers(i, j, p.b))

    def mean_idle(self) -> float:
        """E[I]: 平均アイドル (Delayoff 中) サーバー数."""
        p = self.params
        return self._server_type_mean(lambda i, j: idle_servers(i, j, p.b))

    def mean_setup(self) -> float:
        """E[S]: 平均セットアップ中サーバー数."""
        p = self.params
        return self._server_type_mean(
            lambda i, j: setup_servers(i, j, p.b, p.c)
        )

    def mean_off(self) -> float:
        """E[Off]: 平均オフサーバー数.

        E[Off] = c - E[B] - E[I] - E[S] の関係を利用.
        """
        return (
            self.params.c
            - self.mean_busy()
            - self.mean_idle()
            - self.mean_setup()
        )

    # ---------- 到着率と待ち時間 ----------

    def effective_arrival_rate(self) -> float:
        """λ_eff = 実際に系に入るジョブの単位時間あたりの数.

        直接計算: λ_eff = Σ_{j<K, i, F} π(i, j, F) * λ_F.

        注: MMPP では位相 F の分布が j に依存するため、
             単純に λ_bar * (1 - P_block) では計算できない
             (Poisson 到着では PASTA により一致するが MMPP では非一致).
        """
        p = self.params
        lambdas = p.lambdas
        total = 0.0
        # j < K のみ集計 (j = K は到着ブロック)
        for j in range(p.K):
            block = self.pi[self.ss.block_slice(j)]  # shape (D,)
            # block を (D_S, D_M) にリシェイプ. i-major, F-minor.
            block_2d = block.reshape(p.D_S, p.D_M)
            # F 別に i を集約
            block_F = block_2d.sum(axis=0)  # shape (D_M,)
            total += float(block_F @ lambdas)
        return total

    def arrival_blocking_probability(self) -> float:
        """到着平均ブロック確率 P_block^arrival = 1 - λ_eff / λ_bar.

        時間平均 P_block (= P(j=K)) とは異なる (MMPP 到着の場合).
        Poisson 到着 (D_M=1) では PASTA により両者一致.
        """
        return 1.0 - self.effective_arrival_rate() / self.params.lambda_bar

    def mean_waiting_time(self) -> float:
        """E[W] = E[j] / λ_eff (Little の公式).

        Little の公式: L = λ W より、平均滞在時間.
        """
        lam_eff = self.effective_arrival_rate()
        if lam_eff <= 0:
            return float("inf")
        return self.mean_queue_length() / lam_eff

    # ---------- 利用率 ----------

    def utilization(self) -> float:
        """ρ = E[B] / c: 平均サーバー利用率."""
        return self.mean_busy() / self.params.c

    # ---------- コスト ----------

    def energy_cost(
        self,
        c_busy: float = 1.0,
        c_idle: float = 0.6,
        c_setup: float = 1.0,
        c_off: float = 0.0,
    ) -> float:
        """電力コスト = c_busy*E[B] + c_idle*E[I] + c_setup*E[S] + c_off*E[Off].

        Args:
            c_busy: Busy サーバー1台あたりの電力係数.
            c_idle: Idle サーバー1台あたりの電力係数.
            c_setup: Setup サーバー1台あたりの電力係数.
            c_off: Off サーバー1台あたりの電力係数.
        """
        return (
            c_busy * self.mean_busy()
            + c_idle * self.mean_idle()
            + c_setup * self.mean_setup()
            + c_off * self.mean_off()
        )

    # ---------- 一括取得 ----------

    def all_metrics(self) -> Dict[str, float]:
        """全指標を辞書として返す."""
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
