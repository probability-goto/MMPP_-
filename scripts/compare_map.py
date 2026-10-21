#!/usr/bin/env python
"""比較の地図: alpha=0.005 で Base と Predictive (保護あり) の 2 者を比べる.

各マスで Base と Predictive がそれぞれパラメータを最適化してから比べる.
主な基準: P_ref = Base の ERP 最小点の P_block_arrival_stable とし,
P_block_arrival_stable <= P_ref を満たす Predictive の点の中の ERP 最小値を,
Base の ERP 最小値と比べる.

固定: c=20, K=200, b=5, mu=1, delta=0.6, alpha=0.005 (1/alpha=200).
到着は build_mmpp. rho = rho_B / (1 + delta), sigma = alpha / r1.
マス: rho_B in {0.6, 0.8, 1.12} x r1 in R1_LEVELS (experiment_8_map.py と同じ, 対数 5 点 [0.1, 100]).
定常分布は GTH.

Base の探索 (experiment_8_map.py と同じ):
    1. 粗い探索 (stage=coarse): beta を [sigma/1000, sigma*100] の対数 9 点.
    2. 細かい探索 (stage=fine): 粗い探索の最良の beta の両隣の間を対数 21 点.
    3. 範囲の拡張 (stage=extend, fine2): 最良点が beta の範囲の端なら 1 桁広げて 3 点を追加し,
       最良点が広げた側に移ればその周りを 2. と同じく細かくする.
Predictive の探索 (保護あり: protect_presetup=protect_delayoff=True):
    1. 粗い探索: n_target in {0,5,10,15,20} x gamma in {1,3,10,30,100,1000} の直積から
       (0, 1) (= Base) を除いた 29 組それぞれで, beta を Base と同じ 9 点.
    2. 細かい探索: Base の探索が終わってから P_ref を決め, 粗い探索の中で主な基準
       (P <= P_ref の中の ERP 最小) での最良の組を固定し, その beta の両隣の間を対数 7 点.
       P <= P_ref の点が粗い探索に 1 つもなければ, ERP 最小の組で同じことをする.
    3. beta の範囲の拡張: Base と同じ規則 (最良点は主な基準で選ぶ).
    4. gamma の範囲の拡張 (stage=gamma_ext): 最良点の gamma が 1000 なら, その n_target で,
       最良の beta とその両隣 (その組で計算済みの beta の中の隣) の 3 点で
       gamma in {3000, 10000} を追加する.
整合性の検査 (stage=check): 各マスで粗い格子の中央の beta で, Predictive (n_target=0, gamma=1)
とベースモデルの全指標が相対誤差 1e-10 以下で一致することを確かめる. 一致しなければ止まる.

既存の点の再利用: rho_B=0.8 の行では results/experiment_8/experiment_8_B_alpha0.005.csv の点のうち,
パラメータ (r1, beta, n_target, gamma, 保護の有無) が一致する点を再計算せずに使う.
保護なしの "P2 only" (n_target=0) の点も, n_target=0 では保護の有無で結果が変わらない
(保護の条件 i+s <= 0 は保護の効く遷移が起こらない状態でしか成り立たない. 3 点で全指標の
相対誤差 0 を確認済み) ので, 保護ありの点として使う. source 列に
"computed" / "reused:<元のファイル名>" を残す.

計算結果は 1 点ごとに results/compare_map/compare_map_rhoB<rho_B>.csv に追記し, 再実行時は
計算済みの点を飛ばして再開する. 進捗は results/compare_map/progress.log に追記する.

使用例:
    python scripts/compare_map.py                 # 全マスを計算
    python scripts/compare_map.py --rho-b 0.6     # 1 行だけ
    python scripts/compare_map.py --report        # 集計 (report.md, summary.csv, 図)
"""
import argparse
import csv
import itertools
import os
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mmpp_experiments.csvio import append_row, log_line, read_rows
from mmpp_experiments.evaluate import COMMON_KEYS

try:
    import experiment_8_map as e8
except ImportError:
    import scripts.experiment_8_map as e8


# ============================================================
# パラメータ
# ============================================================

BASELINE = e8.BASELINE
DELTA = e8.DELTA
ALPHA = 0.005
RHO_B_LEVELS = [0.6, 0.8, 1.12]
R1_LEVELS = e8.R1_LEVELS
N_BETA = e8.N_BETA
N_FINE_BASE = 21
N_FINE_PRED = 7
N_EXTEND = 3
CONSISTENCY_TOL = 1e-10

NT_LEVELS = [0, 5, 10, 15, 20]
GAMMA_LEVELS = [1.0, 3.0, 10.0, 30.0, 100.0, 1000.0]
GAMMA_EXT = [3000.0, 10000.0]
# (0, 1) は Base と同じなので除く (29 組)
COMBOS: List[Tuple[int, float]] = [(nt, g) for nt, g in itertools.product(NT_LEVELS, GAMMA_LEVELS)
                                   if (nt, g) != (0, 1.0)]

# 判定: 改善率がこれを超えれば「改善」, 以下なら「同等」
IMPROVE_THRESHOLD = 0.001

OUT_DIR = os.path.join("results", "compare_map")
FIG_DIR = "figures"
LOG_NAME = "progress.log"
REUSE_SOURCES = {0.8: os.path.join("results", "experiment_8", "experiment_8_B_alpha0.005.csv")}

warnings.filterwarnings("ignore", message="Predictive モデルの縮約後の帯幅が大きい")


def row_name(rho_B: float) -> str:
    return f"rhoB{rho_B:g}"


def cell_params(rho_B: float, r1: float) -> dict:
    return dict(rho_B=rho_B, rho=rho_B / (1 + DELTA), alpha=ALPHA, r1=r1, sigma=ALPHA / r1)


def coarse_betas(sigma: float) -> List[float]:
    return e8.coarse_betas(sigma)


# ============================================================
# CSV
# ============================================================

METRIC_FIELDS = e8.METRIC_FIELDS
FIELDNAMES = e8.FIELDNAMES + ["source"]


def csv_path(rho_B: float, out_dir: str = OUT_DIR) -> str:
    return os.path.join(out_dir, f"compare_map_{row_name(rho_B)}.csv")


def pkey(r1, series, beta, nt, g) -> Tuple:
    return (float(r1), series, float(beta), int(nt), float(g))


def rkey(r) -> Tuple:
    return pkey(r["r1"], r["series"], r["beta"], r["n_target"], r["gamma"])


# ============================================================
# 既存の点の再利用
# ============================================================

def load_reuse_index(rho_B: float) -> Dict[Tuple, dict]:
    """再利用できる既存の点: (r1, "Base"/"Predictive", beta, n_target, gamma) -> 行."""
    path = REUSE_SOURCES.get(rho_B)
    if path is None or not os.path.exists(path):
        return {}
    name = os.path.basename(path)
    out = {}
    for r in read_rows(path):
        if not (float(r["rho_B"]) == rho_B and float(r["alpha"]) == ALPHA
                and float(r["delta"]) == DELTA
                and all(float(r[k]) == float(v) for k, v in BASELINE.items())
                and r["check"] == "ok"):
            continue
        protect = r["protect_presetup"] == "True" and r["protect_delayoff"] == "True"
        nt = int(r["n_target"])
        if r["series"] == "Base":
            series = "Base"
        elif r["series"] == "Check":
            continue
        elif protect or nt == 0:
            # n_target=0 では保護の有無で結果が変わらない (docstring 参照)
            series = "Predictive"
        else:
            continue
        r = dict(r, _source=f"reused:{name}")
        out[pkey(r["r1"], series, r["beta"], nt, r["gamma"])] = r
    return out


def reused_row(src: dict, cp: dict, series: str, stage: str) -> dict:
    """既存の行を新しい CSV の行に直す (指標はそのまま, 名前と保護の列を付け直す)."""
    out = {k: src.get(k, "") for k in e8.FIELDNAMES}
    out.update({"row": row_name(cp["rho_B"]), "series": series, "stage": stage,
                "model": "base" if series == "Base" else "predictive",
                "protect_presetup": series != "Base", "protect_delayoff": series != "Base",
                "source": src["_source"]})
    return out


# ============================================================
# 1 点の計算と整合性の検査
# ============================================================

def consistency(cp: dict, beta: float) -> Tuple[dict, float]:
    """Predictive (n_target=0, gamma=1, 保護あり) とベースモデルの全指標の最大相対誤差."""
    out, m_p = e8.compute_predictive(row_name(cp["rho_B"]), cp, "Check", "check", beta, 0, 1.0,
                                     True)
    _, m_b = e8.compute_base(row_name(cp["rho_B"]), cp, "check", beta)
    dp, db = m_p.all_metrics(), m_b.all_metrics()
    rel = max(0.0 if max(abs(db[k]), abs(dp[k])) == 0 else
              abs(db[k] - dp[k]) / max(abs(db[k]), abs(dp[k])) for k in COMMON_KEYS)
    out["consistency_max_relerr"] = rel
    if rel > CONSISTENCY_TOL:
        out["check"] = (out["check"] + ";" if out["check"] != "ok" else "") + \
            f"consistency:{rel:.2e}"
    return out, rel


class ConsistencyError(RuntimeError):
    pass


# ============================================================
# 次に計算する点 (CSV の内容から決める)
# ============================================================

def _f(r, k):
    return float(r[k])


def _fixed(rows, series, nt, g):
    return [r for r in rows if r["series"] == series and int(r["n_target"]) == nt
            and float(r["gamma"]) == g]


def best_main(rows: Sequence[dict], p_ref: float) -> Optional[dict]:
    """主な基準: P_block_arrival_stable <= p_ref の点の中の ERP 最小点 (なければ None)."""
    feas = [r for r in rows if _f(r, "P_block_arrival_stable") <= p_ref]
    return min(feas, key=lambda r: _f(r, "ERP")) if feas else None


def best_for_search(rows: Sequence[dict], p_ref: Optional[float]) -> dict:
    """探索の向きを決める最良点: 主な基準で選び, 該当がなければ ERP 最小."""
    b = best_main(rows, p_ref) if p_ref is not None else None
    return b if b is not None else min(rows, key=lambda r: _f(r, "ERP"))


def base_p_ref(rows: Sequence[dict]) -> float:
    base = [r for r in rows if r["series"] == "Base"]
    return _f(min(base, key=lambda r: _f(r, "ERP")), "P_block_arrival_stable")


def _refine(rows, series, nt, g, betas, n, p_ref, missing) -> List[Tuple]:
    """1 つの組で, 細かい探索と beta の範囲の拡張の, 次に計算する点 (experiment_8_map と同じ規則)."""
    fixed = _fixed(rows, series, nt, g)
    coarse = [r for r in fixed if r["stage"] == "coarse"]
    cb = best_for_search(coarse, p_ref)
    fine = [(series, "fine", b, nt, g) for b in e8._fine_between(betas, _f(cb, "beta"), n)]
    mf = missing(fine)
    if mf:
        return mf
    best = best_for_search(fixed, p_ref)
    bb = _f(best, "beta")
    lo_edge, hi_edge = betas[0], betas[-1]
    ext = []
    if bb <= lo_edge * (1 + 1e-12):
        ext = [lo_edge * 10 ** (-j / N_EXTEND) for j in range(1, N_EXTEND + 1)]
    elif bb >= hi_edge * (1 - 1e-12):
        ext = [hi_edge * 10 ** (j / N_EXTEND) for j in range(1, N_EXTEND + 1)]
    ext_done = [r for r in fixed if r["stage"] in ("extend", "fine2")]
    if not (ext or ext_done):
        return []
    ext_betas = ext if ext else sorted({_f(r, "beta") for r in fixed if r["stage"] == "extend"})
    me = missing([(series, "extend", b, nt, g) for b in ext_betas])
    if me:
        return me
    fixed = _fixed(rows, series, nt, g)
    on_grid = [r for r in fixed if r["stage"] in ("coarse", "extend")]
    gbest = best_for_search(on_grid, p_ref)
    if gbest["stage"] == "extend":
        grid = sorted(_f(r, "beta") for r in on_grid)
        return missing([(series, "fine2", b, nt, g)
                        for b in e8._fine_between(grid, _f(gbest, "beta"), n)])
    return []


def next_points(rho_B: float, r1: float, rows: List[dict],
                combos: Sequence[Tuple[int, float]] = COMBOS) -> List[Tuple]:
    """マス (rho_B, r1) で次に計算する点の一覧 (series, stage, beta, n_target, gamma).

    空なら, そのマスは完了.
    """
    cp = cell_params(rho_B, r1)
    betas = coarse_betas(cp["sigma"])
    done = {rkey(r) for r in rows}

    def missing(pts):
        return [p for p in pts if pkey(r1, p[0], p[2], p[3], p[4]) not in done]

    # 1. 粗い探索と整合性の検査
    pts = [("Base", "coarse", b, 0, 1.0) for b in betas]
    pts += [("Check", "check", betas[N_BETA // 2], 0, 1.0)]
    pts += [("Predictive", "coarse", b, nt, g) for nt, g in combos for b in betas]
    m = missing(pts)
    if m:
        return m

    # 2. Base の細かい探索と拡張 (P_ref はこれが終わってから決める)
    m = _refine(rows, "Base", 0, 1.0, betas, N_FINE_BASE, None, missing)
    if m:
        return m
    p_ref = base_p_ref(rows)

    # 3. Predictive: 粗い探索の主な基準での最良の組を固定して細かくする
    pred = [r for r in rows if r["series"] == "Predictive"]
    cb = best_for_search([r for r in pred if r["stage"] == "coarse"], p_ref)
    nt, g = int(cb["n_target"]), float(cb["gamma"])
    m = _refine(rows, "Predictive", nt, g, betas, N_FINE_PRED, p_ref, missing)
    if m:
        return m

    # 4. gamma の範囲の拡張: 最良点 (その組の中) の gamma が上端なら
    best = best_for_search(_fixed(rows, "Predictive", nt, g), p_ref)
    if float(best["gamma"]) == GAMMA_LEVELS[-1] and g == GAMMA_LEVELS[-1]:
        bs = sorted({_f(r, "beta") for r in _fixed(rows, "Predictive", nt, g)})
        i = bs.index(_f(best, "beta"))
        around = bs[max(i - 1, 0): i + 2]
        return missing([("Predictive", "gamma_ext", b, nt, ge)
                        for ge in GAMMA_EXT for b in around])
    return []


# ============================================================
# 実行
# ============================================================

def log_progress(out_dir, msg):
    log_line(os.path.join(out_dir, LOG_NAME), msg)


def run_cell(rho_B: float, r1: float, out_dir: str = OUT_DIR,
             combos: Sequence[Tuple[int, float]] = COMBOS, reuse: bool = True) -> Dict[str, int]:
    """1 マスを計算する (計算済みの点は飛ばす). 計算した点と再利用した点の数を返す."""
    os.makedirs(out_dir, exist_ok=True)
    path = csv_path(rho_B, out_dir)
    cp = cell_params(rho_B, r1)
    index = load_reuse_index(rho_B) if reuse else {}
    stats = {"computed": 0, "reused": 0, "bad": 0}
    t0 = time.time()
    last_log = t0
    while True:
        rows = [r for r in read_rows(path) if float(r["r1"]) == r1]
        pts = next_points(rho_B, r1, rows, combos)
        if not pts:
            break
        for series, stage, beta, nt, g in pts:
            label = (f"{row_name(rho_B)},r1={r1:.4g},{series},{stage},beta={beta:.4g},"
                     f"n_target={nt},gamma={g:g}")
            src = index.get(pkey(r1, series, beta, nt, g))
            if src is not None:
                out = reused_row(src, cp, series, stage)
                stats["reused"] += 1
            elif series == "Check":
                out, rel = consistency(cp, beta)
                out["source"] = "computed"
                if rel > CONSISTENCY_TOL:
                    append_row(path, FIELDNAMES, out)
                    log_progress(out_dir, label + " CONSISTENCY_FAIL")
                    raise ConsistencyError(
                        f"{label}: ベースモデルと Predictive (n_target=0, gamma=1) が一致しない "
                        f"(rel={rel:.2e})")
                stats["computed"] += 1
            elif series == "Base":
                out, _ = e8.compute_base(row_name(rho_B), cp, stage, beta)
                out["source"] = "computed"
                stats["computed"] += 1
            else:
                out, _ = e8.compute_predictive(row_name(rho_B), cp, series, stage, beta, nt, g,
                                               True)
                out["source"] = "computed"
                stats["computed"] += 1
            append_row(path, FIELDNAMES, out)
            if out["check"] != "ok":
                stats["bad"] += 1
                print(f"  検査違反: {label}: {out['check']}", flush=True)
            if time.time() - last_log >= 60:
                log_progress(out_dir, f"computed={stats['computed']} reused={stats['reused']} "
                                      f"elapsed={time.time() - t0:.0f}s last={label}")
                last_log = time.time()
    log_progress(out_dir, f"{row_name(rho_B)},r1={r1:.4g} マス完了: computed={stats['computed']} "
                          f"reused={stats['reused']} bad={stats['bad']} "
                          f"elapsed={time.time() - t0:.0f}s")
    return stats


def run(rho_bs: Sequence[float], out_dir: str = OUT_DIR) -> None:
    for rho_B in rho_bs:
        for r1 in R1_LEVELS:
            run_cell(rho_B, r1, out_dir)
        print(f"行 {row_name(rho_B)} 完了", flush=True)


# ============================================================
# 集計
# ============================================================

def _convert(rows: List[dict]) -> List[dict]:
    for r in rows:
        for k in METRIC_FIELDS + ["beta", "gamma", "r1", "sigma", "alpha", "rho_B", "rho"]:
            r[k] = float(r[k]) if str(r[k]).strip() != "" else float("nan")
        r["n_target"] = int(r["n_target"])
    return rows


def cell_summary(rows: List[dict]) -> dict:
    """1 マスの集計 (数値はすべて CSV の行から)."""
    base = [r for r in rows if r["series"] == "Base"]
    pred = [r for r in rows if r["series"] == "Predictive"]
    check = [r for r in rows if r["series"] == "Check"]
    bb = min(base, key=lambda r: r["ERP"])
    p_ref = bb["P_block_arrival_stable"]
    # Base の点も Predictive の (n_target=0, gamma=1) の点として候補に含める
    cand = pred + base
    pb = best_main(cand, p_ref)
    ub = min(cand, key=lambda r: r["ERP"])
    imp = (bb["ERP"] - pb["ERP"]) / bb["ERP"]

    notes = []
    if pb["series"] == "Predictive":
        same = [r for r in pred if r["n_target"] == pb["n_target"] and r["gamma"] == pb["gamma"]]
        bs = sorted({r["beta"] for r in same})
        if pb["beta"] in (bs[0], bs[-1]):
            ext = any(r["stage"] == "extend" for r in same)
            notes.append("β が範囲の端" + (" (広げた後も)" if ext else ""))
        if pb["n_target"] == NT_LEVELS[-1]:
            notes.append(f"n_target が上端 ({NT_LEVELS[-1]}=c)")
        g_max = max(r["gamma"] for r in pred)
        if pb["gamma"] == g_max:
            notes.append(f"γ が上端 ({g_max:g})")
    else:
        bs = sorted({r["beta"] for r in base})
        if pb["beta"] in (bs[0], bs[-1]):
            notes.append("β が範囲の端 (Base の点)")
    return {
        "rho_B": bb["rho_B"], "r1": bb["r1"], "sigma": bb["sigma"],
        "base": bb, "p_ref": p_ref, "pred": pb, "pred_is_base": pb["series"] == "Base",
        "improvement": imp, "verdict": "改善" if imp > IMPROVE_THRESHOLD else "同等",
        "unc": ub, "unc_p_ratio": ub["P_block_arrival_stable"] / p_ref,
        "unc_improvement": (bb["ERP"] - ub["ERP"]) / bb["ERP"],
        "notes": notes,
        "consistency": max(float(r["consistency_max_relerr"]) for r in check)
        if check else float("nan"),
        "n_points": len(rows),
        "n_reused": sum(1 for r in rows if r["source"].startswith("reused:")),
        "n_check_bad": sum(1 for r in rows if r["check"] != "ok"),
    }


def summaries(out_dir: str = OUT_DIR) -> Dict[Tuple[float, float], dict]:
    res = {}
    for rho_B in RHO_B_LEVELS:
        rows = _convert(read_rows(csv_path(rho_B, out_dir)))
        for r1 in R1_LEVELS:
            cr = [r for r in rows if r["r1"] == r1]
            if cr and not next_points(rho_B, r1, [r for r in read_rows(csv_path(rho_B, out_dir))
                                                   if float(r["r1"]) == r1]):
                res[(rho_B, r1)] = cell_summary(cr)
    return res


SUMMARY_FIELDS = [
    "rho_B", "r1", "sigma",
    "base_beta", "base_E_W", "base_Cost", "P_ref", "base_ERP",
    "pred_beta", "pred_n_target", "pred_gamma", "pred_E_W", "pred_Cost", "pred_P",
    "pred_P_over_Pref", "pred_ERP", "pred_is_base", "improvement", "verdict",
    "unc_beta", "unc_n_target", "unc_gamma", "unc_ERP", "unc_P_over_Pref", "unc_improvement",
    "edge_notes", "n_points", "n_reused", "n_check_bad", "consistency_max_relerr",
]


def summary_row(c: dict) -> dict:
    b, p, u = c["base"], c["pred"], c["unc"]
    return {
        "rho_B": c["rho_B"], "r1": c["r1"], "sigma": c["sigma"],
        "base_beta": b["beta"], "base_E_W": b["E_W"], "base_Cost": b["Cost"],
        "P_ref": c["p_ref"], "base_ERP": b["ERP"],
        "pred_beta": p["beta"], "pred_n_target": p["n_target"], "pred_gamma": p["gamma"],
        "pred_E_W": p["E_W"], "pred_Cost": p["Cost"], "pred_P": p["P_block_arrival_stable"],
        "pred_P_over_Pref": p["P_block_arrival_stable"] / c["p_ref"], "pred_ERP": p["ERP"],
        "pred_is_base": c["pred_is_base"], "improvement": c["improvement"],
        "verdict": c["verdict"],
        "unc_beta": u["beta"], "unc_n_target": u["n_target"], "unc_gamma": u["gamma"],
        "unc_ERP": u["ERP"], "unc_P_over_Pref": c["unc_p_ratio"],
        "unc_improvement": c["unc_improvement"],
        "edge_notes": "; ".join(c["notes"]), "n_points": c["n_points"],
        "n_reused": c["n_reused"], "n_check_bad": c["n_check_bad"],
        "consistency_max_relerr": c["consistency"],
    }


def _md_matrix(title, sm, fn) -> List[str]:
    lines = [title, "", "| | " + " | ".join(f"r1={r1:.3g}" for r1 in R1_LEVELS) + " |",
             "|---|" + "---|" * len(R1_LEVELS)]
    for rho_B in RHO_B_LEVELS:
        cells = [fn(sm[(rho_B, r1)]) if (rho_B, r1) in sm else "—" for r1 in R1_LEVELS]
        lines.append(f"| ρ_B={rho_B:g} | " + " | ".join(cells) + " |")
    return lines + [""]


def _combo_label(c: dict) -> str:
    return "Base" if c["pred_is_base"] else f"({c['pred']['n_target']}, {c['pred']['gamma']:g})"


def plot_improvement(sm, fig_dir: str = FIG_DIR) -> List[str]:
    """改善率 [%] のヒートマップ (行 rho_B, 列 log10 r1). 同等のマスは灰色."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize

    M = np.full((len(RHO_B_LEVELS), len(R1_LEVELS)), np.nan)
    for i, rho_B in enumerate(RHO_B_LEVELS):
        for j, r1 in enumerate(R1_LEVELS):
            if (rho_B, r1) in sm:
                M[i, j] = 100 * sm[(rho_B, r1)]["improvement"]
    improved = np.zeros_like(M, dtype=bool)
    for (rho_B, r1), c in sm.items():
        improved[RHO_B_LEVELS.index(rho_B), R1_LEVELS.index(r1)] = c["verdict"] == "改善"
    vmax = np.nanmax(np.where(improved, M, np.nan)) if improved.any() else 1.0
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    # 薄すぎる色は灰色 (同等) や背景と区別しにくいので Blues の 0.15 から使う.
    # マスとカラーバーで同じ色の対応にする
    cmap = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.15, 1.0, 256)))
    norm = Normalize(vmin=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    rgba = np.ones(M.shape + (4,))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isfinite(M[i, j]):
                rgba[i, j] = (1, 1, 1, 1)
            elif improved[i, j]:
                rgba[i, j] = cmap(norm(M[i, j]))
            else:
                rgba[i, j] = (0.85, 0.85, 0.85, 1)
    ax.imshow(rgba, aspect="auto", origin="lower")
    for (rho_B, r1), c in sm.items():
        i, j = RHO_B_LEVELS.index(rho_B), R1_LEVELS.index(r1)
        dark = improved[i, j] and norm(M[i, j]) > 0.6
        color = "white" if dark else "black"
        ax.text(j, i + 0.12, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=13,
                color=color)
        ax.text(j, i - 0.2, _combo_label(c), ha="center", va="center", fontsize=10,
                color=color)
    # マスの境界 (2px 相当の白線)
    ax.set_xticks(np.arange(-0.5, len(R1_LEVELS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(RHO_B_LEVELS)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(len(R1_LEVELS)))
    ax.set_xticklabels([f"{np.log10(r):.2f}" for r in R1_LEVELS], fontsize=11)
    ax.set_yticks(range(len(RHO_B_LEVELS)))
    ax.set_yticklabels([rf"$\rho_B$={v:g}" for v in RHO_B_LEVELS], fontsize=12)
    ax.set_xlabel(r"$\log_{10} r_1$", fontsize=14)
    sm_ = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm_, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("ERP improvement vs Base [%]", fontsize=11)
    fig.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    png = os.path.join(fig_dir, "compare_map_improvement.png")
    pdf = os.path.splitext(png)[0] + ".pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def report(out_dir: str = OUT_DIR, fig_dir: str = FIG_DIR) -> str:
    sm = summaries(out_dir)
    caption = (
        "キャプション用情報: c=20, K=200, b=5, 1/mu=1, delta=0.6, alpha=0.005 (1/alpha=200), "
        "rho=rho_B/1.6, sigma=alpha/r1, r1 は [0.1, 100] の対数 5 点. "
        "値は ERP の改善率 (Base の ERP 最小値に対する, P_block <= P_ref を満たす Predictive の "
        "ERP 最小値の減少率) [%]. 下の括弧は最良点の (n_target, gamma). "
        f"灰色は改善率 {100 * IMPROVE_THRESHOLD:g}% 以下 (同等). "
        "Predictive は保護あり, beta, n_target in {0,5,10,15,20}, "
        "gamma in {1,3,10,30,100,1000} を最適化.")
    print(caption)

    rows_out = [summary_row(sm[k]) for k in sorted(sm)]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows_out)

    n_pts = sum(c["n_points"] for c in sm.values())
    n_reused = sum(c["n_reused"] for c in sm.values())
    n_bad = sum(c["n_check_bad"] for c in sm.values())
    cons = max((c["consistency"] for c in sm.values()), default=float("nan"))
    lines = [
        "# 比較の地図 (α=0.005, Base と Predictive)", "",
        "results/compare_map/compare_map_rhoB<ρ_B>.csv から scripts/compare_map.py --report で生成.", "",
        caption.replace("キャプション用情報: ", "固定と定義: "), "",
        "主な基準: P_ref = Base の ERP 最小点の P_block_arrival_stable. P_block_arrival_stable ≤ P_ref "
        "を満たす点 (Predictive の全点と, (n_target, γ)=(0, 1) とみなした Base の全点) の中の ERP 最小点を "
        f"Predictive の最良点とする. 改善率 = (Base の ERP − Predictive の ERP) / Base の ERP. "
        f"改善率 > {100 * IMPROVE_THRESHOLD:g}% を「改善」, それ以下を「同等」とする.", "",
        "## 点数と検査", "",
        f"- 完了したマス: {len(sm)} / {len(RHO_B_LEVELS) * len(R1_LEVELS)}",
        f"- 点数: {n_pts} (うち再利用 {n_reused})",
        f"- check 列の違反 (保存則, 流量バランス, min π ≥ 0, P_block > 0): {n_bad} 件",
        f"- 整合性の検査 (Predictive (0, 1) とベースモデル, 全指標) の最大相対誤差: {cons:.2e}", "",
        "## 地図", "",
    ]
    lines += _md_matrix("### ERP の改善率 [%] (判定)", sm,
                        lambda c: f"{100 * c['improvement']:.2f} ({c['verdict']})")
    lines += _md_matrix("### 最良点の n_target* (Base の点なら —)", sm,
                        lambda c: "—" if c["pred_is_base"] else str(c["pred"]["n_target"]))
    lines += _md_matrix("### 最良点の γ* (Base の点なら —)", sm,
                        lambda c: "—" if c["pred_is_base"] else f"{c['pred']['gamma']:g}")
    lines += _md_matrix("### 最良点の P / P_ref", sm,
                        lambda c: f"{c['pred']['P_block_arrival_stable'] / c['p_ref']:.3f}")
    lines += _md_matrix("### 補助: 制約なしの ERP 最小点の改善率 [%] と P / P_ref", sm,
                        lambda c: f"{100 * c['unc_improvement']:.2f} (P/P_ref={c['unc_p_ratio']:.3g})")

    lines += ["## 各マスの詳細", "",
              "| ρ_B | r1 | 点 | β* | n_target* | γ* | E[W] | Cost | P | P/P_ref | ERP | 注記 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (rho_B, r1) in sorted(sm):
        c = sm[(rho_B, r1)]
        for lab, r in [("Base (ERP 最小)", c["base"]), ("Predictive (主な基準)", c["pred"]),
                       ("Predictive (制約なし)", c["unc"])]:
            nt = "—" if r["series"] == "Base" else str(r["n_target"])
            g = "—" if r["series"] == "Base" else f"{r['gamma']:g}"
            note = "; ".join(c["notes"]) if lab.startswith("Predictive (主") else ""
            if r["series"] == "Base" and lab.startswith("Predictive"):
                note = ("Base の点; " + note).rstrip("; ")
            lines.append(
                f"| {rho_B:g} | {r1:.3g} | {lab} | {r['beta']:.4g} | {nt} | {g} | "
                f"{r['E_W']:.5g} | {r['Cost']:.5g} | {r['P_block_arrival_stable']:.4e} | "
                f"{r['P_block_arrival_stable'] / c['p_ref']:.3f} | {r['ERP']:.6g} | {note} |")
    lines.append("")

    edge = [(k, c["notes"]) for k, c in sorted(sm.items()) if c["notes"]]
    lines += ["## 探索範囲の端にある最良点", ""]
    lines += [f"- ρ_B={k[0]:g}, r1={k[1]:.3g}: {', '.join(n)}" for k, n in edge] or ["- なし"]
    lines.append("")

    figs = plot_improvement(sm, fig_dir) if sm else []
    lines += ["## 図", ""] + [f"- {f}" for f in figs] + [""]
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rho-b", type=float, choices=RHO_B_LEVELS, default=None,
                        help="1 行だけ計算する (既定: 全行)")
    parser.add_argument("--report", action="store_true", help="集計 (report.md, summary.csv, 図)")
    args = parser.parse_args()
    if args.report:
        print(report())
        return
    run(RHO_B_LEVELS if args.rho_b is None else [args.rho_b])


if __name__ == "__main__":
    main()
