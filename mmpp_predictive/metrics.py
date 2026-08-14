"""性能指標の計算.

ベース (mmpp/metrics.py) と同じ指標を計算する. E[S] (平均セットアップ中
サーバー数) はベースでは S(i,j) 関数から導出するが, Predictive モデルで
は s が明示的な状態変数であるため, その期待値として直接計算する.
"""
from typing import Dict

import numpy as np

from mmpp_predictive.model import PredictiveModelParameters
from mmpp_predictive.state_space import build_is_index, busy_count, idle_count


class Metrics:
    """定常分布から各性能指標を計算するクラス."""

    def __init__(self, params: PredictiveModelParameters, pi: np.ndarray):
        self.params = params
        self.pi = np.asarray(pi)
        if self.pi.shape != (params.N,):
            raise ValueError(f"pi shape {self.pi.shape} != ({params.N},)")
        self._compute_moments()

    def _compute_moments(self):
        p = self.params
        c, K, b = p.c, p.K, p.b
        D_M = p.D_M
        is_index, is_pairs = build_is_index(c)
        lambdas = p.lambdas

        E_B = 0.0
        E_S = 0.0
        E_I = 0.0
        E_N = 0.0
        P_block_time = 0.0
        lambda_eff = 0.0

        for (i, s) in is_pairs:
            for j in range(K + 1):
                base = (is_index[(i, s)] * (K + 1) + j) * D_M
                pi_slice = self.pi[base:base + D_M]
                pi_sum = float(pi_slice.sum())
                if pi_sum == 0.0:
                    continue

                B = busy_count(i, j, b)
                I = idle_count(i, j, b)

                E_B += B * pi_sum
                E_S += s * pi_sum
                E_I += I * pi_sum
                E_N += j * pi_sum

                if j == K:
                    P_block_time += pi_sum
                else:
                    lambda_eff += float(pi_slice @ lambdas)

        self.E_B = E_B
        self.E_S = E_S
        self.E_I = E_I
        self.E_off = c - E_B - E_S - E_I
        self.E_N = E_N
        self.P_block_time = P_block_time
        self.lambda_eff = lambda_eff

        lambda_bar = p.lambda_bar
        self.P_block_arrival = (
            1.0 - self.lambda_eff / lambda_bar if lambda_bar > 0 else 0.0
        )

    # ---------- ブロック確率 ----------

    def blocking_probability(self) -> float:
        """P_block = P(j = K) (時間平均)."""
        return self.P_block_time

    def arrival_blocking_probability(self) -> float:
        """到着平均ブロック確率 P_block^arrival = 1 - lambda_eff / lambda_bar."""
        return self.P_block_arrival

    # ---------- 平均量 ----------

    def mean_queue_length(self) -> float:
        """E[j]: 平均系内ジョブ数."""
        return self.E_N

    def mean_busy(self) -> float:
        """E[B]: 平均処理中サーバー数."""
        return self.E_B

    def mean_setup(self) -> float:
        """E[S]: 平均セットアップ中サーバー数."""
        return self.E_S

    def mean_idle(self) -> float:
        """E[I]: 平均アイドル (Delayoff 中) サーバー数."""
        return self.E_I

    def mean_off(self) -> float:
        """E[Off]: 平均オフサーバー数 (= c - E[B] - E[S] - E[I])."""
        return self.E_off

    # ---------- 到着率と待ち時間 ----------

    def effective_arrival_rate(self) -> float:
        """lambda_eff: 実効到着率 (j < K の状態でのみ到着が成立)."""
        return self.lambda_eff

    def mean_waiting_time(self) -> float:
        """Little の公式より E[W] = E[j] / lambda_eff."""
        if self.lambda_eff <= 0:
            return float("inf")
        return self.E_N / self.lambda_eff

    # ---------- 利用率 ----------

    def utilization(self) -> float:
        """rho = E[B] / c."""
        return self.E_B / self.params.c

    # ---------- コスト ----------

    def energy_cost_paper(self) -> float:
        """先行研究と同じ Cost 関数. Cost = C_b*E[B] + C_s*E[S] + C_i*E[I]."""
        p = self.params
        return p.C_b * self.E_B + p.C_s * self.E_S + p.C_i * self.E_I

    def erp_paper(self) -> float:
        """ERP = E[W] * Cost."""
        return self.mean_waiting_time() * self.energy_cost_paper()

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
            "cost_paper": self.energy_cost_paper(),
            "ERP_paper": self.erp_paper(),
        }
