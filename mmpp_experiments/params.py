"""数値実験で共通のパラメータと, 対称 2 位相 MMPP 到着の構築."""
from typing import Dict, List, Tuple

import numpy as np

# 共通のシステム規模 (先行研究 Fig.10 と同じ規模. K は cb*4 = 400 だが計算コストを考えて 200)
SYSTEM = dict(c=20, K=200, b=5, mu=1.0)

# バースト水準の 3 段階: (名前, delta, sigma)
BURST_LEVELS: List[Tuple[str, float, float]] = [
    ("weak", 0.3, 1.0),
    ("medium", 0.6, 0.1),
    ("strong", 0.9, 0.01),
]
BURST_BY_NAME: Dict[str, Tuple[str, float, float]] = {lv[0]: lv for lv in BURST_LEVELS}


def build_mmpp(
    rho: float,
    delta: float,
    sigma: float,
    c: int,
    b: int,
    mu: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """rho を与えて対称 2 位相 MMPP の C0, C1 を構築する.

    平均到着率: lambda_bar = rho * c * b * mu
    位相別到着率: lambda_0 = lambda_bar * (1-delta), lambda_1 = lambda_bar * (1+delta)
    位相遷移: sigma_0 = sigma_1 = sigma (対称)
    """
    lambda_bar = rho * c * b * mu
    lambda_0 = lambda_bar * (1 - delta)
    lambda_1 = lambda_bar * (1 + delta)

    C1 = np.array([[lambda_0, 0.0], [0.0, lambda_1]])
    C0_off = np.array([[0.0, sigma], [sigma, 0.0]])
    row_sum = C0_off.sum(axis=1) + C1.sum(axis=1)
    C0 = C0_off.copy()
    np.fill_diagonal(C0, -row_sum)

    return C0, C1


def compute_interarrival_cv(C0: np.ndarray, C1: np.ndarray) -> float:
    """MMPP 到着間隔の CV (変動係数) を数値計算する.

    到着間隔分布は phase-type 分布であり, 平均は 1/lambda_bar,
    2 次モーメントは Fischer & Meier-Hellstern (1993) "The MMPP cookbook" の
    標準結果 E[X^2] = 2 * p_arrival @ (-C0)^{-2} @ e で計算する
    (p_arrival = pi_mmpp @ C1 / lambda_bar は到着直後の位相分布).
    """
    D_M = C0.shape[0]
    Q_mmpp = C0 + C1
    M = Q_mmpp.T.copy()
    M[-1, :] = 1.0
    rhs = np.zeros(D_M)
    rhs[-1] = 1.0
    varpi = np.linalg.solve(M, rhs)

    lambdas = C1 @ np.ones(D_M)
    lambda_bar = varpi @ lambdas
    mean_iat = 1.0 / lambda_bar

    p_arrival = varpi @ C1 / lambda_bar
    neg_C0_inv = np.linalg.inv(-C0)
    second_moment = 2.0 * p_arrival @ (neg_C0_inv @ neg_C0_inv) @ np.ones(D_M)

    variance = second_moment - mean_iat ** 2
    cv = np.sqrt(variance) / mean_iat
    return float(cv)
