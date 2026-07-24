"""共有フィクスチャ.

各テストモジュールが参照する ModelParameters インスタンスをここで定義する.
"""
import numpy as np
import pytest

from mmpp import ModelParameters


@pytest.fixture
def small_params():
    """小規模パラメータ (D_M=2, b=1). 網羅的なループを伴うテスト向け."""
    C0 = np.array([[-1.0, 0.3], [0.4, -1.5]])
    C1 = np.array([[0.7, 0.0], [0.0, 1.1]])
    return ModelParameters(
        c=3, K=5, b=1, mu=1.0, alpha=0.5, beta=0.1,
        C0=C0, C1=C1,
    )


@pytest.fixture
def medium_params():
    """中規模パラメータ (D_M=2). c=8, K=30 で index(5, 25, 1) が有効な範囲."""
    C0 = np.array([[-1.0, 0.3], [0.4, -1.5]])
    C1 = np.array([[0.7, 0.0], [0.0, 1.1]])
    return ModelParameters(
        c=8, K=30, b=1, mu=1.0, alpha=1.0, beta=0.2,
        C0=C0, C1=C1,
    )


@pytest.fixture
def batch_params():
    """バッチサイズ b>1 のパラメータ (D_M=2)."""
    C0 = np.array([[-1.0, 0.3], [0.4, -1.5]])
    C1 = np.array([[0.7, 0.0], [0.0, 1.1]])
    return ModelParameters(
        c=4, K=10, b=2, mu=1.0, alpha=0.5, beta=0.1,
        C0=C0, C1=C1,
    )


@pytest.fixture
def mm1k_like_params():
    """D_M=1 (Poisson 到着) の M/M/1/K 相当パラメータ."""
    C0 = np.array([[-0.5]])
    C1 = np.array([[0.5]])
    return ModelParameters(
        c=2, K=6, b=1, mu=1.0, alpha=1.0, beta=0.1,
        C0=C0, C1=C1,
    )
