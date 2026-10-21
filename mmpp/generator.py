"""生成行列 Q の構築.

各状態 (i, j, F) からの遷移を直接列挙して疎行列を組み立てる.
COO 形式で組んでから CSR に変換 (scipy.sparse の推奨方法).

遷移の分類:
    1. 到着 (j < K):        (i, j, F) -> (i, j+1, F')  率 C1[F, F']
                             (j = K では到着ブロック. C1 の F'≠F 位相遷移のみ保持)
    2. バッチサービス完了:   (i, j, F) -> (i, j-b, F)   率 B(i,j) * mu
    3. セットアップ完了:     (i, j, F) -> (i+1, j, F)   率 S(i,j) * alpha
    4. Delayoff タイムアウト: (i, j, F) -> (i-1, j, F)   率 I(i,j) * beta
    5. MMPP 内部遷移:        (i, j, F) -> (i, j, F')    率 C0[F, F'] (F' != F)
"""
from scipy.sparse import csr_matrix

from mmpp.state_space import StateSpace


# ============================================================
# 補助関数 (Busy/Idle/Setup/Off サーバー数)
# ============================================================

def busy_servers(i: int, j: int, b: int) -> int:
    """処理中サーバー数 B(i,j) = min(i, floor(j/b)).

    バッチを持っているサーバーの数. フルバッチ方式のため
    floor(j/b) 個の完全なバッチが待機中. それを処理できるサーバーは
    アクティブサーバー i 個までに制限される.
    """
    return min(i, j // b)


def idle_servers(i: int, j: int, b: int) -> int:
    """アイドル (Delayoff 中) サーバー数 I(i,j) = i - B(i,j).

    アクティブだがバッチを持っていないサーバー.
    Delayoff タイマーが動作中.
    """
    return i - busy_servers(i, j, b)


def setup_servers(i: int, j: int, b: int, c: int) -> int:
    """セットアップ中サーバー数 S(i,j) = max(0, min(c, floor(j/b)) - i).

    現在の待機バッチ数 floor(j/b) を処理するために必要なサーバー数と
    現在アクティブな i を比較. 不足分をセットアップ中とする.
    ただしサーバー総数 c を超えない.
    """
    target = min(c, j // b)
    return max(0, target - i)


def off_servers(i: int, j: int, b: int, c: int) -> int:
    """オフサーバー数 = c - i - S(i,j).

    完全にオフ (電源断) のサーバー数.
    """
    return c - i - setup_servers(i, j, b, c)


class TransitionBuilder:
    """生成行列の成分を COO 形式 (行, 列, 値のリスト) で積み, CSR にする.

    add(src, dst, rate) で遷移 src -> dst を追加すると, 対角 (src, src) に -rate も
    同時に積む. 同じ (行, 列) への重複は to_csr() で足し合わせる.
    ベースモデルと Predictive (mmpp_predictive.generator) の両方の生成行列で使う.
    """

    def __init__(self, N: int):
        self.N = N
        self.rows = []
        self.cols = []
        self.vals = []

    def add(self, src: int, dst: int, rate: float) -> None:
        """遷移 src -> dst (率 rate) を追加. 率が 0 以下か src == dst なら何もしない."""
        if rate <= 0 or src == dst:
            return
        # 非対角 (遷移先)
        self.rows.append(src)
        self.cols.append(dst)
        self.vals.append(rate)
        # 対角 (自身のインデックス, 負の寄与)
        self.rows.append(src)
        self.cols.append(src)
        self.vals.append(-rate)

    def to_csr(self) -> csr_matrix:
        """COO -> CSR (同一 (row, col) への重複エントリを集約)."""
        Q = csr_matrix((self.vals, (self.rows, self.cols)), shape=(self.N, self.N))
        Q.sum_duplicates()
        return Q


# ============================================================
# Q 行列構築 (メイン関数)
# ============================================================

def build_generator(params) -> csr_matrix:
    """MMPP/M/c/SET-BATCH/delayoff の生成行列 Q を疎行列として構築.

    Args:
        params: ModelParameters インスタンス.

    Returns:
        Q: (N, N) の scipy.sparse.csr_matrix 形式の生成行列.

    Notes:
        - 各行の和は 0 (生成行列の性質).
        - 非対角成分は非負, 対角成分は非正.
        - ブロック帯構造 (帯幅 (b+1) * D).
    """
    ss = StateSpace(params)
    N = params.N
    D_M = params.D_M
    c = params.c
    K = params.K
    b = params.b
    mu = params.mu
    alpha = params.alpha
    beta = params.beta
    C0 = params.C0
    C1 = params.C1

    # COO 形式 (Coordinate format（座標形式）)で構築 (行, 列, 値のリスト)
    builder = TransitionBuilder(N)
    add_transition = builder.add

    for (i, j, F) in ss.iter_states():
        """ 「遷移元 (source) の行番号」"""
        src = ss.index(i, j, F)

        # 補助量
        B = busy_servers(i, j, b)
        I = idle_servers(i, j, b)
        S = setup_servers(i, j, b, c)

        # --- 1. 到着 ---
        if j < K:
            # 通常の到着: (i, j, F) -> (i, j+1, F')
            for Fp in range(D_M):
                rate = C1[F, Fp]  # "FからFpへの遷移"
                if rate > 0:
                    dst = ss.index(i, j + 1, Fp)
                    add_transition(src, dst, rate)
        else:
            # j = K: 到着ブロック.
            # C1 の非対角成分 (F' != F) は "位相遷移を伴う到着 attempt" として
            # 位相のみ更新する遷移を残す. C1[F, F] (位相不変の到着) は自己ループとなり
            # add_transition の src==dst ガードで自動的に除外されるが, ここでは
            # モデル意味論の明示性のため Fp != F を条件として書いておく.
            for Fp in range(D_M):
                if Fp != F:
                    rate = C1[F, Fp]  # "FからFpへの遷移"
                    if rate > 0:
                        dst = ss.index(i, K, Fp)
                        add_transition(src, dst, rate)

        # --- 2. バッチサービス完了 ---
        if B > 0 and j >= b:
            rate = B * mu
            dst = ss.index(i, j - b, F)
            add_transition(src, dst, rate)

        # --- 3. セットアップ完了 ---
        if S > 0 and i < c:
            rate = S * alpha
            dst = ss.index(i + 1, j, F)
            add_transition(src, dst, rate)

        # --- 4. Delayoff タイムアウト ---
        if I > 0 and i > 0:
            rate = I * beta
            dst = ss.index(i - 1, j, F)
            add_transition(src, dst, rate)

        # --- 5. MMPP 内部遷移 (F' != F) ---
        for Fp in range(D_M):
            if Fp != F:
                rate = C0[F, Fp]
                if rate > 0:
                    dst = ss.index(i, j, Fp)
                    add_transition(src, dst, rate)

    return builder.to_csr()
