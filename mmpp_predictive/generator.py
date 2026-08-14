"""Predictive モデルの生成行列 Q の構築.

状態 (i, s, j, F) からの遷移を直接列挙して疎行列を組み立てる (ベースの
mmpp/generator.py と同じ COO -> CSR の組み立て方針).

遷移の分類:
    1. 到着 (j < K):         (i, s, j, F) -> (i, s', j+1, F')  率 C1[F, F']
                              j = K では到着ブロック (F' != F の位相遷移のみ残る).
                              s' はベースモデルと同じ反応的セットアップ判定
                              (compute_required_s) で j の増加に追従して
                              最大 1 だけ増える.
    2. セットアップ完了:      (i, s, j, F) -> (i+1, s-1, j, F)   率 s * alpha
    3. バッチサービス完了:    (i, s, j, F) -> (i, s'', j-b, F)   率 B(i,j) * mu
                              s'' は 1. と対称に, j の減少に追従して
                              最大 1 だけ減る (compute_required_s が
                              下がった分だけベースモデルと同じく「不要になった
                              反応的セットアップ」を 1 単位取り消す).
    4. Delayoff:              (i, s, j, F) -> (i-1, s, j, F)     率 I(i,j) * beta_F
                              beta_F は F=0 (通常位相) で gamma 倍に加速.
    5. MMPP 位相遷移 (F=0 以外の宛先, または F=0 -> F=1 以外):
                              (i, s, j, F) -> (i, s, j, F')      率 C0[F, F']
    6. MMPP 位相遷移 (F=0 -> F=1, 事前セットアップ):
                              (i, s, j, 0) -> (i, s+Delta_eff, j, 1)  率 C0[0, 1]

n_target=0 のとき Delta_eff は常に 0 になり, かつ 1. と 3. の s
反応的増減はベースモデルの S(i,j) = compute_required_s(i,j,b,c) に
帰納的に一致し続けるため (s は常に compute_required_s(i,j,b,c) と
一致する), Predictive モデルはベースモデルに厳密に一致する
(tests/test_predictive_baseline_consistency.py で検証).
"""
import numpy as np
from scipy.sparse import csr_matrix

from mmpp.generator import setup_servers as compute_required_s

from mmpp_predictive.model import PredictiveModelParameters
from mmpp_predictive.state_space import (
    build_is_index,
    num_states,
    busy_count,
    idle_count,
    setup_target_delta,
)


def build_generator(params: PredictiveModelParameters) -> csr_matrix:
    """Predictive モデルの生成行列 Q を構築する (疎行列).

    Returns:
        Q: N x N の scipy.sparse.csr_matrix 形式の生成行列.
            各行の和は 0. 非対角成分は非負, 対角成分は非正.
    """
    c, K, b = params.c, params.K, params.b
    mu, alpha, beta = params.mu, params.alpha, params.beta
    n_target, gamma = params.n_target, params.gamma
    C0, C1 = params.C0, params.C1
    D_M = params.D_M

    N = num_states(c, K, D_M)
    is_index, is_pairs = build_is_index(c)

    def idx(i, s, j, F):
        return (is_index[(i, s)] * (K + 1) + j) * D_M + F

    rows = []
    cols = []
    vals = []

    def add_transition(src: int, dst: int, rate: float):
        """遷移 src -> dst (率 rate) を追加. 対角成分も同時に更新."""
        if rate <= 0 or src == dst:
            return
        rows.append(src)
        cols.append(dst)
        vals.append(rate)
        rows.append(src)
        cols.append(src)
        vals.append(-rate)

    for (i, s) in is_pairs:
        for j in range(K + 1):
            for F in range(D_M):
                src = idx(i, s, j, F)

                # --- 1. 到着 ---
                if j < K:
                    j_new = j + 1
                    s_needed = compute_required_s(i, j_new, b, c) #今 j 個ジョブがあって i 台アクティブなとき、追加で何台セットアップしていないと足りないか」という理想値
                    s_new = s + 1 if (s_needed > s and s < c - i) else s
                    for Fp in range(D_M):
                        rate = C1[F, Fp]
                        if rate > 0:
                            dst = idx(i, s_new, j_new, Fp)
                            add_transition(src, dst, rate)
                else:
                    # j = K: 到着ブロック. 位相遷移を伴う到着のみ残す.
                    for Fp in range(D_M):
                        if Fp == F:
                            continue
                        rate = C1[F, Fp]
                        if rate > 0:
                            dst = idx(i, s, K, Fp)
                            add_transition(src, dst, rate)

                # --- 2. セットアップ完了 ---
                if s >= 1:
                    dst = idx(i + 1, s - 1, j, F)
                    add_transition(src, dst, s * alpha)

                # --- 3. バッチサービス完了 ---
                B = busy_count(i, j, b)
                if B > 0 and j >= b:
                    j_new = j - b
                    s_needed = compute_required_s(i, j_new, b, c)
                    s_new = s - 1 if s_needed < s else s
                    dst = idx(i, s_new, j_new, F)
                    add_transition(src, dst, B * mu)

                # --- 4. Delayoff ---
                I = idle_count(i, j, b)
                if I > 0 and i >= 1:
                    beta_F = gamma * beta if F == 0 else beta
                    dst = idx(i - 1, s, j, F)
                    add_transition(src, dst, I * beta_F)

                # --- 5, 6. MMPP 位相遷移 ---
                for Fp in range(D_M):
                    if Fp == F:
                        continue
                    rate = C0[F, Fp]
                    if rate <= 0:
                        continue
                    if F == 0 and Fp == 1:
                        delta_eff = setup_target_delta(i, s, n_target, c)
                        dst = idx(i, s + delta_eff, j, Fp)
                    else:
                        dst = idx(i, s, j, Fp)
                    add_transition(src, dst, rate)

    Q = csr_matrix((vals, (rows, cols)), shape=(N, N))
    Q.sum_duplicates()
    return Q
