"""Base (ベースモデル) と比べた判定の区分と, 判定が真になるパラメータの範囲の書き方.

rel_erp は ERP の Base に対する相対差 (負なら改善), p_ratio は P_block^arrival の Base に対する比.
"""
from typing import List

# 相対差がこれ以下なら 0 とみなす (しきい値なしの判定)
RTOL = 1e-9

# しきい値付きの判定: |ERP の相対差| <= ERP_TIE と P の比が [P_LO, P_HI] は同等
ERP_TIE = 0.001
P_LO, P_HI = 0.95, 1.05
CATEGORIES5 = ["改善", "同等", "棄却のみ改善", "引き換え", "悪化"]


def category(rel_erp: float, rel_p: float) -> str:
    """しきい値なしの 3 区分 (改善 / 引き換え / 悪化). rel_p は P の相対差."""
    erp_better = rel_erp < 0
    p_ok = rel_p <= RTOL
    if erp_better and p_ok:
        return "改善"
    if not erp_better and not p_ok:
        return "悪化"
    return "引き換え"


def erp_state(rel_erp: float) -> str:
    if rel_erp < -ERP_TIE:
        return "better"
    if rel_erp > ERP_TIE:
        return "worse"
    return "equal"


def p_state(p_ratio: float) -> str:
    if p_ratio < P_LO:
        return "better"
    if p_ratio > P_HI:
        return "worse"
    return "equal"


def category5(rel_erp: float, p_ratio: float) -> str:
    """しきい値付きの 5 区分.

    改善: ERP が 0.1% 以上下がり, P が 5% 以上上がらない.
    同等: ERP も P も同等.
    棄却のみ改善: ERP は同等で, P が 5% 以上下がる.
    引き換え: 片方が改善し, もう片方が悪化する.
    悪化: どちらかが悪化し, どちらも改善しない.
    """
    e, p = erp_state(rel_erp), p_state(p_ratio)
    if e == "better" and p != "worse":
        return "改善"
    if e == "equal" and p == "equal":
        return "同等"
    if e == "equal" and p == "better":
        return "棄却のみ改善"
    if (e == "better" and p == "worse") or (e == "worse" and p == "better"):
        return "引き換え"
    return "悪化"


def intervals(vals: List[float], flags: List[bool], fmt) -> str:
    """格子 vals (昇順) 上で flags が真の値を連続区間にまとめる."""
    runs, cur = [], []
    for v, f in zip(vals, flags):
        if f:
            cur.append(v)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    if not runs:
        return "なし"
    if len(runs) == 1 and len(runs[0]) == len(vals):
        return "全域"
    return " ∪ ".join(fmt(r[0]) if len(r) == 1 else f"[{fmt(r[0])}, {fmt(r[-1])}]"
                      for r in runs)
