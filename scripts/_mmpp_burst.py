"""実験 0 / 実験 1 共通のバースト MMPP 構築ユーティリティ.

対称 2 位相 MMPP (delta, sigma) の構築, 到着間隔 CV の計算,
ウォームアップ実時間の診断を提供する.
"""
import warnings
from typing import List, Tuple

import numpy as np


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


def check_warmup_duration(params, replications: List) -> None:
    """ウォームアップ実時間が遅い時定数 1/beta に対して十分かを診断する."""
    warmup_duration = replications[0].warmup_duration
    warmup_ratio = warmup_duration * params.beta
    if warmup_ratio < 5.0:
        warnings.warn(
            f"ウォームアップが遅い時定数 1/beta = {1 / params.beta:.2f} に対して "
            f"{warmup_ratio:.2f} 倍しかありません. warmup_events を増やすことを検討してください.",
            RuntimeWarning,
            stacklevel=2,
        )
