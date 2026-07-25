"""モデルパラメータの定義.

MMPP/M/c/SET-BATCH/delayoff モデルの全パラメータを保持する
データクラス. 妥当性検証と派生量 (D_S, D_M, N, λ_bar 等) を提供.
"""
from dataclasses import dataclass
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class ModelParameters:
    """MMPP/M/c/SET-BATCH/delayoff モデルのパラメータ.

    Attributes:
        c: サーバー総数 (>0)
        K: バッファ容量 (>=b)
        b: バッチサイズ (>0)
        mu: バッチサービス完了率 (>0)
        alpha: セットアップ完了率 (>0)
        beta: Delayoff タイムアウト率 (>0)
        C0: MMPP 内部遷移行列 (D_M x D_M). 対角は負, 非対角は非負.
        C1: MMPP 到着遷移行列 (D_M x D_M). 全成分非負.

    整合性条件:
        (C0 + C1) e = 0 (行和ゼロ)

    Notes:
        - 全て指数分布 (連続時間 Markov 連鎖).
        - フルバッチ方式: b 個揃わないと処理開始しない.
    """
    c: int
    K: int
    b: int
    mu: float
    alpha: float
    beta: float
    C0: np.ndarray
    C1: np.ndarray

    def __post_init__(self):
        # numpy 配列化
        self.C0 = np.asarray(self.C0, dtype=float)
        self.C1 = np.asarray(self.C1, dtype=float)
        self._validate()

    def _validate(self):
        """パラメータの妥当性を検証"""
        # スカラーパラメータ
        if self.c <= 0:
            raise ValueError(f"c={self.c} must be > 0")
        if self.b <= 0:
            raise ValueError(f"b={self.b} must be > 0")
        if self.K < self.b:
            raise ValueError(f"K={self.K} must be >= b={self.b}")
        if self.mu <= 0:
            raise ValueError(f"mu={self.mu} must be > 0")
        if self.alpha <= 0:
            raise ValueError(f"alpha={self.alpha} must be > 0")
        if self.beta <= 0:
            raise ValueError(f"beta={self.beta} must be > 0")

        # 行列
        if self.C0.ndim != 2 or self.C0.shape[0] != self.C0.shape[1]:
            raise ValueError(f"C0 must be square 2D, got shape {self.C0.shape}")
        if self.C1.shape != self.C0.shape:
            raise ValueError(
                f"C1 shape {self.C1.shape} must equal C0 shape {self.C0.shape}"
            )

        # 符号
        D_M = self.C0.shape[0]
        C0_off = self.C0.copy()
        np.fill_diagonal(C0_off, 0)
        if (C0_off < -1e-12).any():
            raise ValueError("C0 の非対角成分は非負でなければならない")
        if (np.diag(self.C0) > 1e-12).any():
            raise ValueError("C0 の対角成分は非正でなければならない")
        if (self.C1 < -1e-12).any():
            raise ValueError("C1 の全成分は非負でなければならない")

        # (C0 + C1) e = 0 (MMPP 整合性)
        row_sums = (self.C0 + self.C1) @ np.ones(D_M)
        if not np.allclose(row_sums, 0, atol=1e-10):
            raise ValueError(
                f"MMPP 整合性違反: (C0+C1)e = {row_sums} (should be zero)"
            )

        # MMPP 位相過程の既約性検証
        # C0+C1 の非零成分をグラフとみなし, 強連結成分が 1 つであることを確認.
        # 複数の強連結成分があると Q 全体が reducible になり, 定常分布が非一意になる.
        C_sum = self.C0 + self.C1
        adjacency = (np.abs(C_sum) > 1e-12).astype(int)
        np.fill_diagonal(adjacency, 0)  # 自己ループは既約性判定に無関係
        n_components, _ = connected_components(
            csr_matrix(adjacency), directed=True, connection="strong"
        )
        if n_components > 1:
            raise ValueError(
                f"MMPP 位相過程が既約でない (強連結成分数 = {n_components}). "
                "C0 と C1 の設定を確認してください. "
                "全ての位相が互いに到達可能である必要があります."
            )

        # 到達不能状態の警告
        # floor(K/b) < c だと i > floor(K/b) の状態が setup 経路から到達不能になる.
        # 数学的には問題ないが, パラメータ設定ミスの可能性があるため警告.
        max_reachable_i = self.K // self.b
        if max_reachable_i < self.c:
            warnings.warn(
                f"floor(K/b) = {max_reachable_i} < c = {self.c} のため, "
                f"i > {max_reachable_i} の状態は setup 経路から到達不能になります "
                f"(K={self.K}, b={self.b}, c={self.c}). "
                "意図した設定であれば無視してください.",
                UserWarning,
                stacklevel=2,
            )

    # ---------- 派生量 (プロパティ) ----------

    @property
    def D_S(self) -> int:
        """アクティブサーバー数の場合の数 = c+1"""
        return self.c + 1

    @property
    def D_M(self) -> int:
        """MMPP フェーズ数"""
        return self.C0.shape[0]

    @property
    def D(self) -> int:
        """1 レベル (j 固定) あたりの状態数 = D_S * D_M"""
        return self.D_S * self.D_M

    @property
    def N(self) -> int:
        """状態空間の総サイズ = D * (K+1)"""
        return self.D * (self.K + 1)

    @property
    def lambdas(self) -> np.ndarray:
        """フェーズごとの到着率 λ_F = (C1 e)_F"""
        return self.C1 @ np.ones(self.D_M)

    @property
    def phase_stationary(self) -> np.ndarray:
        """MMPP 位相過程 (C0 + C1) の定常分布.

        pi_mmpp (C0 + C1) = 0, sum(pi_mmpp) = 1 を満たす.
        """
        Q = self.C0 + self.C1
        D_M = self.D_M
        # 正規化条件を用いて可解にする
        M = Q.T.copy()
        M[-1, :] = 1.0
        b = np.zeros(D_M)
        b[-1] = 1.0
        return np.linalg.solve(M, b)

    @property
    def lambda_bar(self) -> float:
        """位相過程の定常分布下での平均到着率 = pi_mmpp @ lambdas"""
        return float(self.phase_stationary @ self.lambdas)

    def summary(self) -> str:
        """パラメータの要約文字列"""
        return (
            f"ModelParameters(c={self.c}, K={self.K}, b={self.b}, "
            f"mu={self.mu}, alpha={self.alpha}, beta={self.beta}, "
            f"D_M={self.D_M}, N={self.N}, lambda_bar={self.lambda_bar:.4g})"
        )
