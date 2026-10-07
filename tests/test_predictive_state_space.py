"""mmpp_predictive.state_space のテスト."""
import numpy as np
import pytest

from mmpp_predictive.state_space import (
    enumerate_is_pairs, build_is_index, state_to_idx, idx_to_state,
    num_states, num_is_pairs, level_slice, pi_by_level,
    busy_count, idle_count, setup_target_delta
)


def test_enumerate_is_pairs_count():
    c = 20
    pairs = enumerate_is_pairs(c)
    assert len(pairs) == (c + 1) * (c + 2) // 2  # = 231 for c=20


def test_enumerate_is_pairs_constraint():
    c = 5
    pairs = enumerate_is_pairs(c)
    for i, s in pairs:
        assert 0 <= i <= c
        assert 0 <= s <= c - i


def test_idx_roundtrip():
    """全状態でインデックス化と逆変換が一致することを確認."""
    c, K, D_M = 3, 5, 2
    is_index, is_pairs = build_is_index(c)
    N = num_states(c, K, D_M)
    for idx in range(N):
        i, s, j, F = idx_to_state(idx, c, K, D_M, is_pairs)
        idx2 = state_to_idx(i, s, j, F, c, K, D_M, is_index)
        assert idx == idx2, f"Roundtrip failed at idx={idx}"


def test_idx_is_j_major():
    """idx = (j * N_IS + iota(i,s)) * D_M + F (j が最上位) であること."""
    c, K, D_M = 3, 5, 2
    is_index, is_pairs = build_is_index(c)
    n_is = num_is_pairs(c)
    assert n_is == len(is_pairs)
    expected = 0
    for j in range(K + 1):
        for (i, s) in is_pairs:
            for F in range(D_M):
                assert state_to_idx(i, s, j, F, c, K, D_M, is_index) == expected
                assert expected == (j * n_is + is_index[(i, s)]) * D_M + F
                expected += 1
    assert expected == num_states(c, K, D_M)


def test_origin_is_index_zero():
    """BFS の起点 idx=0 が (i, s, j, F) = (0, 0, 0, 0) に対応すること.

    mmpp_predictive.solver.solve_stationary の start_index=0 はこれを前提にする.
    """
    c, K, D_M = 20, 200, 2
    is_index, is_pairs = build_is_index(c)
    assert state_to_idx(0, 0, 0, 0, c, K, D_M, is_index) == 0
    assert idx_to_state(0, c, K, D_M, is_pairs) == (0, 0, 0, 0)


def test_level_slice_and_pi_by_level():
    """level_slice(j) と pi_by_level(pi)[j] が同じ状態集合を指すこと."""
    c, K, D_M = 3, 5, 2
    is_index, is_pairs = build_is_index(c)
    N = num_states(c, K, D_M)
    pi = np.arange(N, dtype=float)
    pi3 = pi_by_level(pi, c, K, D_M)
    assert pi3.shape == (K + 1, num_is_pairs(c), D_M)
    for j in range(K + 1):
        sl = level_slice(j, c, D_M)
        assert np.array_equal(pi[sl], pi3[j].ravel())
        for idx in range(N)[sl]:
            assert idx_to_state(idx, c, K, D_M, is_pairs)[2] == j
        for is_idx, (i, s) in enumerate(is_pairs):
            for F in range(D_M):
                assert pi3[j, is_idx, F] == state_to_idx(i, s, j, F, c, K, D_M, is_index)


def test_num_states():
    c, K, D_M = 20, 200, 2
    assert num_states(c, K, D_M) == 231 * 201 * 2  # = 92,862


def test_busy_idle_count():
    b = 5
    # j=0: 全アイドル
    assert busy_count(i=3, j=0, b=b) == 0
    assert idle_count(i=3, j=0, b=b) == 3
    # j=5 (1 バッチ分): 1 台 busy, 2 台アイドル
    assert busy_count(i=3, j=5, b=b) == 1
    assert idle_count(i=3, j=5, b=b) == 2
    # j=15 (3 バッチ分): 3 台 busy, 0 台アイドル
    assert busy_count(i=3, j=15, b=b) == 3
    assert idle_count(i=3, j=15, b=b) == 0
    # j=20 (4 バッチ分だが i=3): 3 台 busy (全て), 0 台アイドル
    assert busy_count(i=3, j=20, b=b) == 3
    assert idle_count(i=3, j=20, b=b) == 0


def test_setup_target_delta():
    c = 20
    # 既に目標達成: 追加起動なし
    assert setup_target_delta(i=15, s=0, n_target=10, c=c) == 0
    # アクティブ 5, 目標 10: 5 台追加
    assert setup_target_delta(i=5, s=0, n_target=10, c=c) == 5
    # アクティブ 5 + セットアップ中 3 = 8, 目標 10: 2 台追加
    assert setup_target_delta(i=5, s=3, n_target=10, c=c) == 2
    # オフサーバーが不足するケース: n_target=15, 既に i=10, s=0, オフは 10
    # -> Delta=5, Delta_eff=min(5, 10)=5
    assert setup_target_delta(i=10, s=0, n_target=15, c=c) == 5
    # オフサーバー枯渇ケース: n_target=15, i=8, s=10, オフは 2
    # -> Delta=max(15-8-10, 0)=0
    assert setup_target_delta(i=8, s=10, n_target=15, c=c) == 0
