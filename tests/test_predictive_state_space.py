"""mmpp_predictive.state_space のテスト."""
import pytest

from mmpp_predictive.state_space import (
    enumerate_is_pairs, build_is_index, state_to_idx, idx_to_state,
    num_states, busy_count, idle_count, setup_target_delta
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
