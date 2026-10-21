"""(E[W], Cost) などの 2 指標をともに最小化する意味でのフロンティア (パレート最適な点) と支配."""
from typing import List, Optional, Tuple

import numpy as np


def pareto_front(points: List[Tuple[float, float, int]]) -> List[Tuple[float, float, int]]:
    """(x, y, id) の点集合のうち, x と y をともに最小化する意味で支配されない点.

    x の昇順に並べて返す (y は降順になる). 同じ (x, y) の点は 1 つだけ残す.
    """
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    front = []
    best_y = float("inf")
    for x, y, i in pts:
        if y < best_y:
            front.append((x, y, i))
            best_y = y
    return front


def interp_cost(front: List[Tuple[float, float, int]], x: float) -> Optional[float]:
    """フロンティアを x について線形補間した y.

    x がフロンティアの x の最小値より小さければ到達できないので None.
    最大値より大きければ, 最大の x の点の y (それより小さい x で達成できる) を返す.
    """
    xs = [p[0] for p in front]
    ys = [p[1] for p in front]
    if x < xs[0]:
        return None
    if x >= xs[-1]:
        return ys[-1]
    return float(np.interp(x, xs, ys))


def _dominates(a: Tuple[float, float], b: Tuple[float, float], rtol=1e-9) -> bool:
    """a が b を (x, y の最小化の意味で) 支配するか. 差が rtol 以下は同じとみなす."""
    le_x = a[0] <= b[0] * (1 + rtol)
    le_y = a[1] <= b[1] * (1 + rtol)
    lt = a[0] < b[0] * (1 - rtol) or a[1] < b[1] * (1 - rtol)
    return le_x and le_y and lt


def dominance(front_s, front_ref) -> Tuple[str, int, int]:
    """系列のフロンティアが参照フロンティアを支配するか.

    参照フロンティアの各点について, 系列フロンティアのいずれかの (実在の) 点が
    それを支配するかを調べ, 支配される点の数で 全域 / 一部 / しない を返す.
    """
    n = len(front_ref)
    k = sum(1 for r in front_ref if any(_dominates(s[:2], r[:2]) for s in front_s))
    label = "全域" if k == n else ("一部" if k > 0 else "しない")
    return label, k, n
