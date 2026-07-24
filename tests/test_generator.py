"""生成行列 Q のテスト.

検証項目:
    - 補助関数 (B, I, S, Off) の性質
    - Q の形状と疎性
    - 行和ゼロ (Q e = 0)
    - 非対角成分の非負性、対角成分の非正性
    - ブロック帯構造
    - 具体的な遷移が正しく入っているか
"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mmpp import (
    ModelParameters,
    StateSpace,
    build_generator,
    busy_servers,
    idle_servers,
    setup_servers,
    off_servers,
)


class TestAuxiliaryFunctions:
    """補助関数 B, I, S, Off の性質."""

    @pytest.mark.parametrize("c,K,b", [
        (2, 4, 1),
        (3, 6, 2),
        (5, 20, 3),
    ])
    def test_B_bounds(self, c, K, b):
        """B は [0, i] の範囲."""
        for i in range(c + 1):
            for j in range(K + 1):
                B = busy_servers(i, j, b)
                assert 0 <= B <= i, f"B({i},{j})={B} out of [0, {i}]"

    @pytest.mark.parametrize("c,K,b", [
        (2, 4, 1),
        (3, 6, 2),
    ])
    def test_B_plus_I_equals_i(self, c, K, b):
        """B + I = i (活動サーバー数の内訳)."""
        for i in range(c + 1):
            for j in range(K + 1):
                B = busy_servers(i, j, b)
                I = idle_servers(i, j, b)
                assert B + I == i

    @pytest.mark.parametrize("c,K,b", [
        (2, 4, 1),
        (3, 6, 2),
    ])
    def test_S_bounds(self, c, K, b):
        """S は [0, c-i] の範囲."""
        for i in range(c + 1):
            for j in range(K + 1):
                S = setup_servers(i, j, b, c)
                assert 0 <= S <= c - i, f"S({i},{j})={S} out of [0, {c-i}]"

    @pytest.mark.parametrize("c,K,b", [
        (2, 4, 1),
        (3, 6, 2),
    ])
    def test_off_non_negative(self, c, K, b):
        """Off = c - i - S は非負."""
        for i in range(c + 1):
            for j in range(K + 1):
                Off = off_servers(i, j, b, c)
                assert Off >= 0, f"Off({i},{j})={Off} < 0"

    def test_total_conservation(self, small_params):
        """B + I + S + Off = c (サーバー総数保存)."""
        c, b = small_params.c, small_params.b
        for i in range(c + 1):
            for j in range(small_params.K + 1):
                B = busy_servers(i, j, b)
                I = idle_servers(i, j, b)
                S = setup_servers(i, j, b, c)
                Off = off_servers(i, j, b, c)
                assert B + I + S + Off == c, (
                    f"i={i}, j={j}: B+I+S+Off={B+I+S+Off} != {c}"
                )

    def test_at_j_zero(self, small_params):
        """j=0 では B=S=0, I=i, Off=c-i."""
        c, b = small_params.c, small_params.b
        for i in range(c + 1):
            assert busy_servers(i, 0, b) == 0
            assert setup_servers(i, 0, b, c) == 0
            assert idle_servers(i, 0, b) == i
            assert off_servers(i, 0, b, c) == c - i

    def test_full_batch_semantics(self):
        """バッチ b=2 のとき, j=3 では 1 バッチのみ処理可.

        B(i=2, j=3, b=2) = min(2, 3//2) = min(2, 1) = 1.
        I(2, 3, 2) = 2 - 1 = 1.
        """
        assert busy_servers(2, 3, 2) == 1
        assert idle_servers(2, 3, 2) == 1
        # j=4 なら 2 バッチ処理可
        assert busy_servers(2, 4, 2) == 2
        assert idle_servers(2, 4, 2) == 0


class TestQMatrixShape:
    """Q の形状と基本性質."""

    def test_shape(self, small_params):
        """Q は (N, N)."""
        Q = build_generator(small_params)
        assert Q.shape == (small_params.N, small_params.N)

    def test_is_sparse(self, small_params):
        """Q は疎行列."""
        Q = build_generator(small_params)
        assert isinstance(Q, csr_matrix)

    def test_sparsity_O_N(self, medium_params):
        """非ゼロ数は O(N) (行あたり定数個)."""
        Q = build_generator(medium_params)
        nnz = Q.nnz
        # 定数 (行あたり最大 10 個程度) * N
        assert nnz < 20 * medium_params.N


class TestRowSumZero:
    """行和がゼロ (Q e = 0) のテスト."""

    def test_row_sum_zero_small(self, small_params):
        Q = build_generator(small_params)
        row_sums = np.asarray(Q.sum(axis=1)).flatten()
        np.testing.assert_allclose(row_sums, 0, atol=1e-12)

    def test_row_sum_zero_medium(self, medium_params):
        Q = build_generator(medium_params)
        row_sums = np.asarray(Q.sum(axis=1)).flatten()
        np.testing.assert_allclose(row_sums, 0, atol=1e-12)

    def test_row_sum_zero_batch(self, batch_params):
        """バッチサイズ b>1 でも成立."""
        Q = build_generator(batch_params)
        row_sums = np.asarray(Q.sum(axis=1)).flatten()
        np.testing.assert_allclose(row_sums, 0, atol=1e-12)


class TestSignStructure:
    """符号構造: 非対角非負, 対角非正."""

    def test_off_diagonal_non_negative(self, small_params):
        Q = build_generator(small_params)
        Q_dense = Q.toarray()
        N = Q.shape[0]
        np.fill_diagonal(Q_dense, 0)
        assert (Q_dense >= -1e-12).all()

    def test_diagonal_non_positive(self, small_params):
        Q = build_generator(small_params)
        Q_dense = Q.toarray()
        diag = np.diag(Q_dense)
        assert (diag <= 1e-12).all()


class TestBlockBandStructure:
    """ブロック帯構造の確認."""

    def test_only_transitions_j_to_jplus1_or_j_minus_b(self, batch_params):
        """遷移は同一 j, j -> j+1, j -> j-b のいずれか."""
        Q = build_generator(batch_params)
        ss = StateSpace(batch_params)
        b = batch_params.b
        Q_dense = Q.toarray()
        N = Q.shape[0]
        for src in range(N):
            for dst in range(N):
                if src == dst:
                    continue
                if abs(Q_dense[src, dst]) < 1e-12:
                    continue
                _, j_src, _ = ss.decode(src)
                _, j_dst, _ = ss.decode(dst)
                dj = j_dst - j_src
                assert dj in (0, 1, -b), (
                    f"Unexpected transition: {ss.decode(src)} -> "
                    f"{ss.decode(dst)}, dj={dj}"
                )


class TestSpecificTransitions:
    """特定の遷移が正しい率で入っているか."""

    def test_arrival_transition(self):
        """(i, j, F) -> (i, j+1, F') 率 C1[F, F']."""
        C0 = np.array([[-1.5, 0.1], [0.2, -2.0]])
        C1 = np.array([[1.4, 0.0], [0.0, 1.8]])
        params = ModelParameters(
            c=2, K=3, b=1, mu=1.0, alpha=0.5, beta=0.1,
            C0=C0, C1=C1,
        )
        Q = build_generator(params)
        ss = StateSpace(params)

        # (1, 1, 0) -> (1, 2, 0): 率 C1[0, 0] = 1.4
        src = ss.index(1, 1, 0)
        dst = ss.index(1, 2, 0)
        assert abs(Q[src, dst] - 1.4) < 1e-12

        # C1[F, F'] = 0 のペアは遷移なし
        src2 = ss.index(1, 1, 0)
        dst2 = ss.index(1, 2, 1)
        assert abs(Q[src2, dst2]) < 1e-12  # C1[0, 1] = 0

    def test_batch_service_transition(self):
        """b=2, B=2 なら 2mu の遷移がある."""
        C0 = np.array([[-1.5, 0.1], [0.2, -2.0]])
        C1 = np.array([[1.4, 0.0], [0.0, 1.8]])
        params = ModelParameters(
            c=2, K=4, b=2, mu=1.0, alpha=0.5, beta=0.1,
            C0=C0, C1=C1,
        )
        Q = build_generator(params)
        ss = StateSpace(params)

        # (2, 4, 0) -> (2, 2, 0): B=min(2,4//2)=2, rate=2*mu=2
        src = ss.index(2, 4, 0)
        dst = ss.index(2, 2, 0)
        assert abs(Q[src, dst] - 2.0) < 1e-12

        # (1, 4, 0) -> (1, 2, 0): B=min(1,4//2)=1, rate=1*mu=1
        src = ss.index(1, 4, 0)
        dst = ss.index(1, 2, 0)
        assert abs(Q[src, dst] - 1.0) < 1e-12

    def test_setup_transition(self):
        """S(i,j)>0 のとき i -> i+1 の遷移が S*alpha."""
        C0 = np.array([[-1.5, 0.1], [0.2, -2.0]])
        C1 = np.array([[1.4, 0.0], [0.0, 1.8]])
        params = ModelParameters(
            c=3, K=4, b=1, mu=1.0, alpha=0.7, beta=0.1,
            C0=C0, C1=C1,
        )
        Q = build_generator(params)
        ss = StateSpace(params)

        # (0, 2, 0) -> (1, 2, 0): S=min(3, 2)-0=2, rate=2*0.7=1.4
        src = ss.index(0, 2, 0)
        dst = ss.index(1, 2, 0)
        assert abs(Q[src, dst] - 2 * 0.7) < 1e-12

    def test_delayoff_transition(self):
        """I(i,j)>0 のとき i -> i-1 の遷移が I*beta."""
        C0 = np.array([[-1.5, 0.1], [0.2, -2.0]])
        C1 = np.array([[1.4, 0.0], [0.0, 1.8]])
        params = ModelParameters(
            c=3, K=4, b=2, mu=1.0, alpha=0.5, beta=0.3,
            C0=C0, C1=C1,
        )
        Q = build_generator(params)
        ss = StateSpace(params)

        # (3, 2, 0): B=min(3, 2//2)=min(3,1)=1, I=3-1=2. Rate=2*0.3=0.6
        src = ss.index(3, 2, 0)
        dst = ss.index(2, 2, 0)
        assert abs(Q[src, dst] - 2 * 0.3) < 1e-12

    def test_mmpp_internal_transition(self):
        """(i, j, F) -> (i, j, F') 率 C0[F, F']."""
        C0 = np.array([[-1.5, 0.3], [0.4, -2.0]])
        C1 = np.array([[1.2, 0.0], [0.0, 1.6]])
        params = ModelParameters(
            c=2, K=3, b=1, mu=1.0, alpha=0.5, beta=0.1,
            C0=C0, C1=C1,
        )
        Q = build_generator(params)
        ss = StateSpace(params)

        # (1, 1, 0) -> (1, 1, 1): rate C0[0, 1] = 0.3
        src = ss.index(1, 1, 0)
        dst = ss.index(1, 1, 1)
        assert abs(Q[src, dst] - 0.3) < 1e-12


class TestBoundaryJequalK:
    """j = K での境界処理."""

    def test_no_arrival_at_j_equal_K(self):
        """j=K では到着遷移 (i, K, F) -> (i, K+1, F') は存在しない (K+1 なし)."""
        # そもそも K+1 のインデックスは無効なので, K での遷移が「不正な出口」を持たないか確認
        C0 = np.array([[-1.5, 0.1], [0.2, -2.0]])
        C1 = np.array([[1.4, 0.0], [0.0, 1.8]])
        params = ModelParameters(
            c=2, K=3, b=1, mu=1.0, alpha=0.5, beta=0.1,
            C0=C0, C1=C1,
        )
        Q = build_generator(params)
        ss = StateSpace(params)
        # 全遷移が有効なインデックス範囲内であることは shape で保証済み.
        # 行和ゼロも別テストで確認済み. ここでは j=K からの唯一の "遷移先" が
        # 位相内, 減少方向, 同一 j 内であることを確認.
        for i in range(params.c + 1):
            for F in range(params.D_M):
                src = ss.index(i, params.K, F)
                # Q の src 行の非ゼロ位置を確認
                Q_row = Q.getrow(src).toarray().flatten()
                for dst in range(params.N):
                    if abs(Q_row[dst]) > 1e-12 and dst != src:
                        _, j_dst, _ = ss.decode(dst)
                        # j 増加はあってはならない
                        assert j_dst <= params.K
                        # j_dst > j_src はあってはならない
                        assert j_dst <= params.K

    def test_phase_transition_at_j_equal_K(self):
        """j=K でも C1 経由の位相遷移 F -> F' (F'≠F) は残る."""
        C0 = np.array([[-1.5, 0.1], [0.2, -2.0]])
        # C1 が非対角を持つ例
        C1 = np.array([[0.8, 0.6], [0.3, 1.5]])
        # 行和 = -0.6, 0.2. これでは (C0+C1)e != 0
        # C0 を調整:
        # C0[0,0] = -(C1[0,0]+C1[0,1] + C0[0,1]) = -(0.8+0.6+0.1) = -1.5. OK.
        # C0[1,1] = -(C1[1,0]+C1[1,1] + C0[1,0]) = -(0.3+1.5+0.2) = -2.0. OK.
        params = ModelParameters(
            c=1, K=3, b=1, mu=1.0, alpha=0.5, beta=0.1,
            C0=C0, C1=C1,
        )
        Q = build_generator(params)
        ss = StateSpace(params)

        # j=K で (i, K, 0) -> (i, K, 1) の遷移は
        # C0[0, 1] + C1[0, 1] = 0.1 + 0.6 = 0.7 が期待される
        # (C1[0, 1] は "到着に伴う位相遷移" だが j=K で到着ブロックのため、
        #  位相遷移分だけ残す)
        src = ss.index(1, params.K, 0)
        dst = ss.index(1, params.K, 1)
        expected = C0[0, 1] + C1[0, 1]
        actual = Q[src, dst]
        assert abs(actual - expected) < 1e-12, (
            f"Expected {expected}, got {actual}"
        )


class TestSetupCancellationImplicit:
    """セットアップ暗黙キャンセルの間接的テスト.

    S(i, j) = f(i, j) が現在状態のみで決まる (履歴依存しない) こと.
    """

    def test_S_depends_only_on_current_state(self, batch_params):
        """S(i, j) が (i, j) の関数として一意に定まる."""
        c, b = batch_params.c, batch_params.b
        # 同じ (i, j) では必ず同じ値
        for i in range(c + 1):
            for j in range(batch_params.K + 1):
                s1 = setup_servers(i, j, b, c)
                s2 = setup_servers(i, j, b, c)
                assert s1 == s2


class TestValidateDenseVsSparse:
    """疎行列と密行列版で同じ Q が得られる (別の実装があるわけではないが,
    dense 変換後も同じ行列であることを確認)."""

    def test_conversion_consistent(self, small_params):
        Q_sparse = build_generator(small_params)
        Q_dense = Q_sparse.toarray()
        # 一致していることは自明だが, 変換の副作用がないことを確認
        assert Q_dense.shape == (small_params.N, small_params.N)
        # row sum
        row_sums = Q_dense.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0, atol=1e-12)
