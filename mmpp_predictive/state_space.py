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
状態空間サイズ: N = N_IS * (K+1) * D_M,  N_IS = (c+1)(c+2)/2

インデックス順序: j-major (レベル j が最上位. ベースモデル mmpp.state_space と
同じ構造で, ベースの i の位置に (i, s) の線形番号 iota(i, s) が入る)
    idx = (j * N_IS + iota(i, s)) * D_M + F

この順序ではセットアップ完了 (i,s)->(i+1,s-1) と Delayoff (i,s)->(i-1,s) が
同じレベル j の中に収まり, レベルをまたぐ遷移は到着 (j+1) とバッチ完了 (j-b)
だけになる. そのため Q の帯幅は K に依存しない.
レベル j の状態 = インデックス [j*N_IS*D_M, (j+1)*N_IS*D_M) (level_slice)
"""
from typing import List, Tuple

import numpy as np

from mmpp.generator import setup_servers as compute_required_s


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


def num_is_pairs(c: int) -> int:
    """(i, s) の組の数 N_IS = (c+1)(c+2)/2."""
    return (c + 1) * (c + 2) // 2


def build_is_index(c: int) -> Tuple[dict, List[Tuple[int, int]]]:
    """(i, s) -> 線形インデックス の辞書と, 逆引きリストを返す."""
    pairs = enumerate_is_pairs(c)
    is_index = {pair: idx for idx, pair in enumerate(pairs)}
    return is_index, pairs


def state_to_idx(i: int, s: int, j: int, F: int,
                  c: int, K: int, D_M: int,
                  is_index: dict) -> int:
    """状態 (i, s, j, F) を線形インデックスに変換.

    インデックス化 (j-major): (j * N_IS + is_index[(i,s)]) * D_M + F
    """
    return (j * len(is_index) + is_index[(i, s)]) * D_M + F


def idx_to_state(idx: int, c: int, K: int, D_M: int,
                  is_pairs: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """線形インデックスを状態 (i, s, j, F) に逆変換."""
    idx, F = divmod(idx, D_M)
    j, is_idx = divmod(idx, len(is_pairs))
    i, s = is_pairs[is_idx]
    return i, s, j, F


def num_states(c: int, K: int, D_M: int) -> int:
    """状態空間の総サイズを返す."""
    return num_is_pairs(c) * (K + 1) * D_M


def level_slice(j: int, c: int, D_M: int) -> slice:
    """レベル j の状態インデックスのスライス (mmpp.state_space.block_slice 相当)."""
    width = num_is_pairs(c) * D_M
    return slice(j * width, (j + 1) * width)


def pi_by_level(pi, c: int, K: int, D_M: int) -> np.ndarray:
    """1 次元の pi を形 (K+1, N_IS, D_M) の配列 (添字 [j, iota(i,s), F]) として見る.

    インデックス順序 (j-major) を前提にした reshape はこの関数に集約する.
    """
    return np.asarray(pi).reshape(K + 1, num_is_pairs(c), D_M)


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


def setup_cancelled(i: int, s: int, j_new: int, F: int,
                    b: int, c: int, n_target: int,
                    protect_presetup: bool,
                    never_cancel_setup: bool = False) -> bool:
    """バッチ完了 (j -> j_new = j-b) でセットアップが 1 台取り消されるか.

    既定 (protect_presetup=False): 需要 S(i, j_new) が s を下回れば取り消す.
    protect_presetup=True: バースト位相 F=1 で i+s <= n_target の間は
    取り消さない (事前セットアップで揃えた台数を n_target 未満に落とさない).
    never_cancel_setup=True: 位相によらず一切取り消さない (位相を使わない対照).
    """
    if never_cancel_setup:
        return False
    cancel = compute_required_s(i, j_new, b, c) < s
    if protect_presetup and F == 1 and i + s <= n_target:
        cancel = False
    return cancel


def delayoff_blocked(i: int, s: int, F: int, n_target: int,
                     protect_delayoff: bool) -> bool:
    """Delayoff (i,s) -> (i-1,s) が保護により止められるか.

    protect_delayoff=True のとき, バースト位相 F=1 で i+s <= n_target の間は
    Delayoff を起こさない (i+s を n_target 未満に落とさない).
    """
    return protect_delayoff and F == 1 and i + s <= n_target
