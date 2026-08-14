"""Predictive モデルのパラメータ定義.

MMPP/M/c/SET-BATCH/Delayoff/Predictive モデルのパラメータを保持する.
ベースモデル共通部分 (c, K, b, mu, alpha, beta, C0, C1) の妥当性検証と
派生量計算は mmpp.model.ModelParameters に委譲し (再利用), Predictive
拡張独自のパラメータ (n_target, gamma) のみをここで追加検証する.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from mmpp.model import ModelParameters

from mmpp_predictive.state_space import num_states


@dataclass
class PredictiveModelParameters:
    """MMPP/M/c/SET-BATCH/Delayoff/Predictive モデルのパラメータ.

    Attributes:
        c, K, b, mu, alpha, beta, C0, C1: ベースモデルと共通 (意味は
            mmpp.model.ModelParameters と同じ).
        n_target: バースト予測時 (MMPP 位相 F=0 -> 1) の目標稼働
            (アクティブ+セットアップ) サーバー数. 0 <= n_target <= c.
        gamma: Delayoff 加速係数. F=0 (通常位相) では Delayoff レートが
            beta -> gamma*beta に加速される. gamma >= 1.
        C_b, C_s, C_i: コスト関数の係数. None ならば先行研究
            (Le-Anh & Phung-Duc 2025) のデフォルト式を用いる.
    """
    c: int
    K: int
    b: int
    mu: float
    alpha: float
    beta: float
    C0: np.ndarray
    C1: np.ndarray

    n_target: int
    gamma: float

    C_b: Optional[float] = None
    C_s: Optional[float] = None
    C_i: Optional[float] = None

    def __post_init__(self):
        # ベースモデルの妥当性検証 (符号規約, MMPP 整合性, 既約性等) と
        # 派生量計算 (D_M, lambdas, phase_stationary, lambda_bar) を再利用.
        self._base = ModelParameters(
            c=self.c, K=self.K, b=self.b, mu=self.mu,
            alpha=self.alpha, beta=self.beta,
            C0=self.C0, C1=self.C1,
        )
        # float 化された C0, C1 で置き換える
        self.C0 = self._base.C0
        self.C1 = self._base.C1

        if not (0 <= self.n_target <= self.c):
            raise ValueError(
                f"n_target={self.n_target} must be in [0, c={self.c}]"
            )
        if self.gamma < 1:
            raise ValueError(f"gamma={self.gamma} must be >= 1")

        # デフォルトのコスト関数 (先行研究 Le-Anh & Phung-Duc 2025 と同じ)
        if self.C_b is None:
            self.C_b = 0.04419 * self.b + 0.15503
        if self.C_s is None:
            self.C_s = self.C_b
        if self.C_i is None:
            self.C_i = 0.6 * self.C_b

    # ---------- 派生量 (ベースモデルへの委譲) ----------

    @property
    def D_M(self) -> int:
        """MMPP フェーズ数."""
        return self._base.D_M

    @property
    def lambdas(self) -> np.ndarray:
        """フェーズごとの到着率 lambda_F = (C1 e)_F."""
        return self._base.lambdas

    @property
    def phase_stationary(self) -> np.ndarray:
        """MMPP 位相過程 (C0 + C1) の定常分布."""
        return self._base.phase_stationary

    @property
    def lambda_bar(self) -> float:
        """位相過程の定常分布下での平均到着率."""
        return self._base.lambda_bar

    # ---------- Predictive 拡張の派生量 ----------

    @property
    def N(self) -> int:
        """状態空間の総サイズ = ((c+1)(c+2)/2) * (K+1) * D_M."""
        return num_states(self.c, self.K, self.D_M)

    def summary(self) -> str:
        """パラメータの要約文字列."""
        return (
            f"PredictiveModelParameters(c={self.c}, K={self.K}, b={self.b}, "
            f"mu={self.mu}, alpha={self.alpha}, beta={self.beta}, "
            f"n_target={self.n_target}, gamma={self.gamma}, "
            f"D_M={self.D_M}, N={self.N}, lambda_bar={self.lambda_bar:.4g})"
        )
