"""V2 の Predictive 版: mpmath 50 桁による高精度検証.

Predictive モデル (mmpp_predictive) の到達可能状態に縮約した生成行列に対し,
float64 の GTH / splu と mpmath 50 桁の GTH (真値) を比較する。

標準設定 (c=20, K=200) は縮約後 N=12292, p=760, q=152 で
mpmath では計算量 1.4e9 となり現実的でないため, 小規模設定
(c=5, K=60, b=2; 縮約 N=788, p=48, q=24) を用いる。
rho を振って P_block が極小になる領域まで走査する。

使用例:
    python scripts/validate_gth_mpmath_predictive.py
"""
import os
import sys

import numpy as np
import mpmath as mp
from scipy.sparse.csgraph import breadth_first_order

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _mmpp_burst import build_mmpp
from mmpp_predictive.model import PredictiveModelParameters
from mmpp_predictive.generator import build_generator
from mmpp_predictive.solver import solve_stationary
from mmpp_predictive.metrics import Metrics
from mmpp_predictive.state_space import level_slice
from mmpp.gth_solver import bandwidths
from validate_gth_mpmath import gth_mpmath

RHO_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
C, K, B = 5, 60, 2
N_TARGET, GAMMA = 3, 5.0

# コマンドライン引数で rho 走査値を上書きできる:
#   python scripts/validate_gth_mpmath_predictive.py 0.05 0.08 0.1


def reduce_Q(Q):
    """到達可能部分空間に縮約した Q と, その元インデックスを返す."""
    Q = Q.tocsr()
    adj = Q.copy()
    adj.setdiag(0)
    adj.eliminate_zeros()
    reach = np.sort(breadth_first_order(
        adj, i_start=0, directed=True, return_predecessors=False))
    return Q[reach][:, reach].tocsr(), reach


def block_states(params):
    """j == K のブロック状態の名目インデックス一覧を返す."""
    return list(range(params.N)[level_slice(params.K, params.c, params.D_M)])


def main():
    rho_values = [float(a) for a in sys.argv[1:]] or RHO_VALUES
    print(f"設定: c={C}, K={K}, b={B}, n_target={N_TARGET}, gamma={GAMMA}")
    rows = []

    for rho in rho_values:
        C0, C1 = build_mmpp(rho=rho, delta=0.6, sigma=0.1, c=C, b=B, mu=1.0)
        params = PredictiveModelParameters(
            c=C, K=K, b=B, mu=1.0, alpha=0.1, beta=0.005,
            C0=C0, C1=C1, n_target=N_TARGET, gamma=GAMMA,
        )
        Q = build_generator(params)
        Q_red, reach = reduce_Q(Q)
        p, q = bandwidths(Q_red)
        N_red = Q_red.shape[0]

        pi_gth = solve_stationary(Q, solver="gth")
        pi_splu = solve_stationary(Q, solver="splu")

        # 真値: 縮約 Q に対して mpmath 50 桁の GTH
        pi_true_red = gth_mpmath(Q_red, dps=50)

        # ブロック状態 (j=K) の質量を集計する。真値は縮約空間の局所
        # インデックスに写してから集計する (到達不能状態の確率は 0)。
        orig_to_local = {int(o): l for l, o in enumerate(reach)}
        blocks = block_states(params)

        pb_true = float(mp.fsum([
            pi_true_red[orig_to_local[b]] for b in blocks if b in orig_to_local
        ]))
        pb_gth = float(sum(pi_gth[b] for b in blocks))
        pb_splu = float(sum(pi_splu[b] for b in blocks))

        # Metrics 経由の値と一致することを確認 (インデックス規約の自己検証)
        pb_metrics = Metrics(params, pi_gth).blocking_probability()
        assert abs(pb_gth - pb_metrics) <= 1e-12 * max(abs(pb_metrics), 1e-300), (
            f"ブロック状態の抽出が Metrics と不一致: {pb_gth} vs {pb_metrics}"
        )

        err_gth = abs(pb_gth - pb_true) / pb_true if pb_true else float("nan")
        err_splu = abs(pb_splu - pb_true) / pb_true if pb_true else float("nan")
        rows.append((rho, N_red, p, q, pb_true, err_gth, err_splu))

        print(f"rho={rho:.2f}  縮約N={N_red} p={p} q={q}  "
              f"P_block(真値)={pb_true:.4e}  "
              f"GTH誤差={err_gth:.3e}  splu誤差={err_splu:.3e}")

    print()
    max_gth = max(r[5] for r in rows)
    max_splu = max(r[6] for r in rows)
    print(f"max rel_err GTH  = {max_gth:.3e}  (閾値 1e-13)")
    print(f"max rel_err splu = {max_splu:.3e}")
    ok = max_gth <= 1e-13
    print("V2-Predictive result:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
