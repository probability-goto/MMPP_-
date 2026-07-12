"""状態空間 (i, j, F) の列挙とインデックス変換.

状態空間: S = {(i, j, F) : 0 <= i <= c, 0 <= j <= K, 0 <= F < D_M}
サイズ: N = (c+1) * (K+1) * D_M = D * (K+1)

インデックス順序: j-major (レベル j が最上位)
    idx = j * D + i * D_M + F

この順序で並べると Q が明示的なブロック帯構造を持つ.
レベル j に対応するブロック = 状態インデックス [j*D, (j+1)*D)
"""
from typing import Iterator, Tuple


class StateSpace:
    """状態空間 (i, j, F) のインデックス変換.

    j-major 順序:
        - レベル j が最上位 (0, 1, ..., K)
        - 次に i (0, 1, ..., c)
        - 最後に F (0, 1, ..., D_M-1)

    数式:
        idx = j * D + i * D_M + F
        j   = idx // D
        i   = (idx % D) // D_M
        F   = idx % D_M
    """

    def __init__(self, params):
        self.params = params

    def index(self, i: int, j: int, F: int) -> int:
        """(i, j, F) -> インデックス.

        Raises:
            ValueError: 範囲外の値が渡された場合.
        """
        p = self.params
        if not (0 <= i <= p.c):
            raise ValueError(f"i={i} out of range [0, {p.c}]")
        if not (0 <= j <= p.K):
            raise ValueError(f"j={j} out of range [0, {p.K}]")
        if not (0 <= F < p.D_M):
            raise ValueError(f"F={F} out of range [0, {p.D_M})")
        return j * p.D + i * p.D_M + F

    def decode(self, idx: int) -> Tuple[int, int, int]:
        """インデックス -> (i, j, F).

        Raises:
            ValueError: idx が範囲外の場合.
        """
        p = self.params
        if not (0 <= idx < p.N):
            raise ValueError(f"idx={idx} out of range [0, {p.N})")
        j, rem = divmod(idx, p.D)
        i, F = divmod(rem, p.D_M)
        return i, j, F

    def iter_states(self) -> Iterator[Tuple[int, int, int]]:
        """全状態を j-major 順序で反復.

        Yields:
            (i, j, F) の 3-tuple.
        """
        p = self.params
        for j in range(p.K + 1):
            for i in range(p.c + 1):
                for F in range(p.D_M):
                    yield (i, j, F)

    def block_slice(self, j: int) -> slice:
        """レベル j の状態インデックスのスライス.

        使用例:
            block_j = pi[state_space.block_slice(j)]  # shape (D,)
        """
        p = self.params
        if not (0 <= j <= p.K):
            raise ValueError(f"j={j} out of range [0, {p.K}]")
        return slice(j * p.D, (j + 1) * p.D)

    def __len__(self) -> int:
        return self.params.N
