"""Predictive モデルの状態空間管理.

状態は 4 タプル (i, s, j, F) で表現される:
    - i: アクティブサーバー数 (0 <= i <= c)
    - s: セットアップ中サーバー数 (0 <= s <= c - i)
    - j: 系内ジョブ数 (0 <= j <= K)
    - F: MMPP 位相 (0 <= F < D_M)

ベースモデル (mmpp/) では s はセットアップ判定関数 S(i,j) から都度導出
されるだけで独立した状態次元を持たない. Predictive モデルでは事前
セットアップにより s が (i, j) だけから決まらなくなる (キューの実需要を
超えてセットアップが先行しうる) ため, s を明示的な状態次元として持つ.

拘束条件 i + s <= c を満たす状態のみを列挙する.
状態空間サイズ: N = ((c+1)(c+2)/2) * (K+1) * D_M
"""
from typing import List, Tuple


def enumerate_is_pairs(c: int) -> List[Tuple[int, int]]:
    """拘束条件 i + s <= c を満たす (i, s) の組を列挙する.

    順序: i の昇順 -> 同じ i 内で s の昇順.
    総数: (c+1)(c+2)/2
    """
    pairs = []
    for i in range(c + 1):
        for s in range(c - i + 1):
            pairs.append((i, s))
    return pairs


def build_is_index(c: int) -> Tuple[dict, List[Tuple[int, int]]]:
    """(i, s) -> 線形インデックス の辞書と, 逆引きリストを返す."""
    pairs = enumerate_is_pairs(c)
    is_index = {pair: idx for idx, pair in enumerate(pairs)}
    return is_index, pairs


def state_to_idx(i: int, s: int, j: int, F: int,
                  c: int, K: int, D_M: int,
                  is_index: dict) -> int:
    """状態 (i, s, j, F) を線形インデックスに変換.

    インデックス化: ((is_index[(i,s)]) * (K+1) + j) * D_M + F
    """
    return (is_index[(i, s)] * (K + 1) + j) * D_M + F


def idx_to_state(idx: int, c: int, K: int, D_M: int,
                  is_pairs: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """線形インデックスを状態 (i, s, j, F) に逆変換."""
    F = idx % D_M
    idx //= D_M
    j = idx % (K + 1)
    idx //= (K + 1)
    i, s = is_pairs[idx]
    return i, s, j, F


def num_states(c: int, K: int, D_M: int) -> int:
    """状態空間の総サイズを返す."""
    n_is = (c + 1) * (c + 2) // 2
    return n_is * (K + 1) * D_M


def busy_count(i: int, j: int, b: int) -> int:
    """ビジーサーバー数 B(i, j) = min(i, floor(j/b))."""
    return min(i, j // b)


def idle_count(i: int, j: int, b: int) -> int:
    """アイドル (Delayoff 中) サーバー数 I(i, j) = i - B(i, j).

    busy_count と同じ floor(j/b) 基準で計算する (ceil を使うと
    busy_count との整合性が崩れる).
    """
    return i - busy_count(i, j, b)


def setup_target_delta(i: int, s: int, n_target: int, c: int) -> int:
    """事前セットアップの発動台数 Delta_eff を計算する.

    Delta = max(n_target - i - s, 0)
    Delta_eff = min(Delta, c - i - s)  (オフサーバー不足によるクリップ)
    """
    delta = max(n_target - i - s, 0)
    delta_eff = min(delta, c - i - s)
    return delta_eff
