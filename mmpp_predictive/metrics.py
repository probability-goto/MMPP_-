"""性能指標の計算.

ベース (mmpp/metrics.py) と同じ指標を計算する. E[S] (平均セットアップ中
サーバー数) はベースでは S(i,j) 関数から導出するが, Predictive モデルで
は s が明示的な状態変数であるため, その期待値として直接計算する.

あわせて事前セットアップ (P1) の診断指標 (発動率, 起動台数の率,
セットアップ完了率, セットアップ取り消し率, Delayoff 率, バースト期の負荷)
を計算する.
"""
from typing import Dict

import numpy as np

from mmpp_predictive.model import PredictiveModelParameters
from mmpp_predictive.state_space import (
    build_is_index,
    busy_count,
    idle_count,
    pi_by_level,
    setup_target_delta,
    setup_cancelled,
    delayoff_blocked,
)


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
        _, is_pairs = build_is_index(c)
        lambdas = p.lambdas
        pi3 = pi_by_level(self.pi, c, K, D_M)  # 添字 [j, iota(i,s), F]

        E_B = 0.0
        E_S = 0.0
        E_I = 0.0
        E_N = 0.0
        P_block_time = 0.0
        lambda_eff = 0.0
        # j=K のブロック状態における到着率の重み付き質量。
        # P_block^arrival を引き算なしで求めるために使う (下記参照)。
        block_arrival_mass = 0.0
        # P1 の診断用: F=0 の質量のうち Delta_eff > 0 の部分と Delta_eff の重み付き和,
        # およびバッチ完了でセットアップが取り消される率 (mu 倍する前).
        fire_mass = 0.0
        launch_mass = 0.0
        cancel_mass = 0.0
        # Delayoff 率 (保護で止められる状態を除く)
        delayoff_rate = 0.0

        for is_idx, (i, s) in enumerate(is_pairs):
            for j in range(K + 1):
                pi_slice = pi3[j, is_idx]
                pi_sum = float(pi_slice.sum())
                if pi_sum == 0.0:
                    continue

                B = busy_count(i, j, b)
                I = idle_count(i, j, b)

                E_B += B * pi_sum
                E_S += s * pi_sum
                E_I += I * pi_sum
                E_N += j * pi_sum

                delta_eff = setup_target_delta(i, s, p.n_target, c)
                if delta_eff > 0:
                    fire_mass += float(pi_slice[0])
                    launch_mass += delta_eff * float(pi_slice[0])

                if I > 0 and i >= 1:
                    for F in range(D_M):
                        if not delayoff_blocked(i, s, F, p.n_target,
                                                p.protect_delayoff):
                            beta_F = p.gamma * p.beta if F == 0 else p.beta
                            delayoff_rate += I * beta_F * float(pi_slice[F])

                if B > 0 and j >= b:
                    for F in range(D_M):
                        if setup_cancelled(i, s, j - b, F, b, c,
                                           p.n_target, p.protect_presetup):
                            cancel_mass += B * float(pi_slice[F])

                if j == K:
                    P_block_time += pi_sum
                    block_arrival_mass += float(pi_slice @ lambdas)
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
        # P_block^arrival は定義式 1 - lambda_eff / lambda_bar ではなく,
        # 引き算を含まない等価式で求める (ベースモデルの
        # mmpp.metrics.Metrics.arrival_blocking_probability_stable と同じ導出):
        #   pi の位相マージナルは MMPP 位相定常分布に一致するので
        #   lambda_bar - lambda_eff = sum_{i,s,F} pi(i,s,K,F) lambda_F
        # これにより P_block^arrival が極小の領域でも桁落ちしない。
        self.P_block_arrival_stable = (
            block_arrival_mass / lambda_bar if lambda_bar > 0 else 0.0
        )

        # P1 の診断指標. 位相 0 -> 1 の率は C0[0, 1] から取る.
        sigma_01 = float(p.C0[0, 1]) if D_M >= 2 else 0.0
        self.p1_fire_rate = sigma_01 * fire_mass
        self.p1_launch_rate = sigma_01 * launch_mass
        self.setup_completion_rate = p.alpha * E_S
        self.setup_cancel_rate = p.mu * cancel_mass
        self.delayoff_rate = delayoff_rate
        # バースト位相 (F=1) の負荷 lambda_1 / (c b mu).
        # build_mmpp で作った MMPP では rho * (1 + delta) に等しい.
        self.rho_B = (
            float(lambdas[1]) / (c * b * p.mu) if D_M >= 2 else float("nan")
        )

    # ---------- ブロック確率 ----------

    def blocking_probability(self) -> float:
        """P_block = P(j = K) (時間平均)."""
        return self.P_block_time

    def arrival_blocking_probability_stable(self) -> float:
        """到着平均ブロック確率 P_block^arrival (引き算を含まない式).

        P_block^arrival = sum_{i,s,F} pi(i,s,K,F) lambda_F / lambda_bar
        (= 1 - lambda_eff / lambda_bar と厳密に等価だが, 桁落ちしない)
        """
        return self.P_block_arrival_stable

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

    # ---------- 事前セットアップ (P1) の診断 ----------

    def p1_fire_count_rate(self) -> float:
        """P1 の発動率 sigma_01 * sum_{i,s,j} pi(i,s,j,0) 1[Delta_eff(i,s) > 0].

        単位時間あたりに F: 0 -> 1 で事前セットアップが 1 台以上起動される回数.
        """
        return self.p1_fire_rate

    def p1_launch_count_rate(self) -> float:
        """P1 で起動を始める台数の率 sigma_01 * sum pi(i,s,j,0) Delta_eff(i,s)."""
        return self.p1_launch_rate

    def setup_completion_count_rate(self) -> float:
        """セットアップ完了率 sum pi(i,s,j,F) s alpha (= alpha E[S])."""
        return self.setup_completion_rate

    def setup_cancel_count_rate(self) -> float:
        """セットアップ取り消し率 sum pi(i,s,j,F) B(i,j) mu 1[取り消しが起こる]."""
        return self.setup_cancel_rate

    def delayoff_count_rate(self) -> float:
        """Delayoff 率 sum pi(i,s,j,F) I(i,j) beta_F 1[Delayoff が止められていない].

        定常状態ではセットアップ完了率 (i の増加率) と等しい.
        """
        return self.delayoff_rate

    def burst_load(self) -> float:
        """バースト位相の負荷 rho_B = lambda_1 / (c b mu) (D_M < 2 なら nan)."""
        return self.rho_B

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
            "P_block_arrival_stable": self.arrival_blocking_probability_stable(),
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
            "p1_fire_rate": self.p1_fire_count_rate(),
            "p1_launch_rate": self.p1_launch_count_rate(),
            "setup_completion_rate": self.setup_completion_count_rate(),
            "setup_cancel_rate": self.setup_cancel_count_rate(),
            "delayoff_rate": self.delayoff_count_rate(),
            "rho_B": self.burst_load(),
        }
