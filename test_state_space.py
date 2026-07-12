"""StateSpace のテスト.

検証項目:
    - index/decode の round-trip
    - iter_states が正確に N 個返す
    - block_slice のインデックス範囲
    - 範囲外の入力に対する例外
"""
import numpy as np
import pytest

from mmpp import StateSpace


class TestIndexDecode:
    """インデックス変換の round-trip テスト."""

    def test_roundtrip_small(self, small_params):
        """(i, j, F) -> idx -> (i, j, F) が可逆."""
        ss = StateSpace(small_params)
        for i in range(small_params.c + 1):
            for j in range(small_params.K + 1):
                for F in range(small_params.D_M):
                    idx = ss.index(i, j, F)
                    i2, j2, F2 = ss.decode(idx)
                    assert (i, j, F) == (i2, j2, F2)

    def test_all_indices_covered(self, small_params):
        """全 (i, j, F) が [0, N) を網羅する."""
        ss = StateSpace(small_params)
        indices = set()
        for i in range(small_params.c + 1):
            for j in range(small_params.K + 1):
                for F in range(small_params.D_M):
                    indices.add(ss.index(i, j, F))
        assert indices == set(range(small_params.N))

    def test_j_major_ordering(self, small_params):
        """j-major 順序: 同じ j 内は連続."""
        ss = StateSpace(small_params)
        for j in range(small_params.K + 1):
            indices_at_j = []
            for i in range(small_params.c + 1):
                for F in range(small_params.D_M):
                    indices_at_j.append(ss.index(i, j, F))
            indices_at_j = sorted(indices_at_j)
            # 差はすべて 1 (連続)
            diffs = np.diff(indices_at_j)
            assert (diffs == 1).all()

    def test_block_slice_size(self, small_params):
        """block_slice のサイズは D."""
        ss = StateSpace(small_params)
        for j in range(small_params.K + 1):
            sl = ss.block_slice(j)
            assert sl.stop - sl.start == small_params.D


class TestIterStates:
    """iter_states のテスト."""

    def test_count(self, small_params):
        """反復数 = N."""
        ss = StateSpace(small_params)
        count = sum(1 for _ in ss.iter_states())
        assert count == small_params.N

    def test_unique(self, small_params):
        """全状態がユニーク."""
        ss = StateSpace(small_params)
        states = list(ss.iter_states())
        assert len(set(states)) == len(states)

    def test_all_in_range(self, small_params):
        """各状態が正しい範囲."""
        ss = StateSpace(small_params)
        for (i, j, F) in ss.iter_states():
            assert 0 <= i <= small_params.c
            assert 0 <= j <= small_params.K
            assert 0 <= F < small_params.D_M


class TestOutOfRange:
    """範囲外の入力に対する例外テスト."""

    def test_index_i_out_of_range(self, small_params):
        ss = StateSpace(small_params)
        with pytest.raises(ValueError, match="i="):
            ss.index(small_params.c + 1, 0, 0)
        with pytest.raises(ValueError, match="i="):
            ss.index(-1, 0, 0)

    def test_index_j_out_of_range(self, small_params):
        ss = StateSpace(small_params)
        with pytest.raises(ValueError, match="j="):
            ss.index(0, small_params.K + 1, 0)

    def test_index_F_out_of_range(self, small_params):
        ss = StateSpace(small_params)
        with pytest.raises(ValueError, match="F="):
            ss.index(0, 0, small_params.D_M)

    def test_decode_out_of_range(self, small_params):
        ss = StateSpace(small_params)
        with pytest.raises(ValueError, match="idx="):
            ss.decode(small_params.N)
        with pytest.raises(ValueError, match="idx="):
            ss.decode(-1)


class TestDifferentSizes:
    """異なるパラメータサイズでの動作."""

    def test_medium(self, medium_params):
        ss = StateSpace(medium_params)
        assert len(ss) == medium_params.N
        # 中間の状態でも round-trip 成立
        idx = ss.index(5, 25, 1)
        assert ss.decode(idx) == (5, 25, 1)

    def test_batch(self, batch_params):
        ss = StateSpace(batch_params)
        assert len(ss) == batch_params.N
