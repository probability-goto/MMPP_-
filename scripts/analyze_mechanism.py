#!/usr/bin/env python
"""実験 7・8 の結果から, Predictive が優位になる仕組みを集計する.

1. 実験 8 の 35 マスで, R = β*(P1 only (protect)) / β*(Base) と P1 only (protect) の改善
   (−ERP の相対差) の相関 (Pearson, Spearman; 全マスと改善が 0.5% を超えるマス).
2. 電力の内訳: 実験 7 の 5 条件と実験 8 で P1 only (protect) の改善が最大の 3 マスで,
   Base と P1 only (protect) の ERP 最小点の C_b·E[B], C_s·E[S], C_i·E[I], Cost.
   (実験 7 の Base は β を細かくした Base (121 点) の ERP 最小点.)
3. 位相ごとの内訳: 同じ点を GTH で解き直し, 位相 F=0, 1 ごとの E[B], E[S], E[I] (寄与,
   位相で足すと全体), Cost, P_block^arrival への寄与を定常分布から計算する.
   Base の点はベースモデルと一致する Predictive (n_target=0, gamma=1, 保護なし) として解き,
   全体の値が既存 CSV と一致することを確かめる.

手順 1・2 は既存の CSV から, 手順 3 は解き直した点の CSV
(results/experiment_8/mechanism_points.csv) から計算し, results/experiment_8/mechanism.md に書く.

使用例:
    python scripts/analyze_mechanism.py
"""
import csv
import os
import warnings
from typing import Dict, List

import numpy as np
from scipy.stats import pearsonr, spearmanr

from mmpp_predictive import (
    PredictiveModelParameters, build_generator, solve_stationary, Metrics,
)
from mmpp_predictive.state_space import (
    build_is_index, busy_count, idle_count, pi_by_level,
)

try:
    import experiment_7_frontier as e7
    import experiment_8_map as e8
    from _mmpp_burst import build_mmpp
except ImportError:
    import scripts.experiment_7_frontier as e7
    import scripts.experiment_8_map as e8
    from scripts._mmpp_burst import build_mmpp

warnings.filterwarnings("ignore", message="Predictive モデルの縮約後の帯幅が大きい")

OUT_DIR = os.path.join("results", "experiment_8")
POINTS_CSV = os.path.join(OUT_DIR, "mechanism_points.csv")
REPORT = os.path.join(OUT_DIR, "mechanism.md")
P1 = "P1 only (protect)"
BASELINE = dict(c=20, K=200, b=5, mu=1.0)


# ============================================================
# 対象の点
# ============================================================

def _f(r, k):
    return float(r[k])


def exp7_points() -> List[dict]:
    """実験 7 の 5 条件の Base (β を細かくしたもの) と P1 only (protect) の ERP 最小点."""
    out = []
    for cond in e7.CONDITIONS:
        main = e7.read_rows(e7.csv_path(cond))
        dense = [r for r in e7.read_rows(e7.refine_csv_path(cond))
                 if r["series"] == "Base (dense)"]
        base = min(dense, key=lambda r: _f(r, "ERP"))
        p1 = min([r for r in main if r["series"] == P1], key=lambda r: _f(r, "ERP"))
        for label, r in (("Base", base), (P1, p1)):
            out.append({"case": f"実験7 {cond}", "series": label, "src": r})
    return out


def exp8_top_cells(n: int = 3):
    sm = e8.summaries()
    cells = sorted(sm, key=lambda k: sm[k]["series"][P1]["erp_rel"])[:n]
    return sm, cells


def exp8_points(sm, cells) -> List[dict]:
    out = []
    for row, r1 in cells:
        c = sm[(row, r1)]
        for label, r in (("Base", c["base_best"]), (P1, c["series"][P1]["best"])):
            out.append({"case": f"実験8 {row} r1={r1:.3g}", "series": label, "src": r})
    return out


# ============================================================
# 手順 3: 解き直して位相ごとに分ける
# ============================================================

PHASE_FIELDS = ["E_B", "E_S", "E_I", "Cost", "P_arr_contrib", "P_time", "mass"]
POINT_FIELDS = (["case", "series", "rho", "delta", "sigma", "alpha", "beta", "n_target",
                 "gamma", "protect", "E_B", "E_S", "E_I", "Cost", "ERP",
                 "P_block_arrival_stable"] +
                [f"{k}_F{F}" for F in (0, 1) for k in PHASE_FIELDS] +
                ["csv_max_relerr", "min_pi", "check"])


def solve_point(p: dict) -> dict:
    src = p["src"]
    is_base = p["series"] == "Base"
    rho, delta, sigma, alpha = (_f(src, k) for k in ("rho", "delta", "sigma", "alpha"))
    beta = _f(src, "beta")
    nt = 0 if is_base else int(src["n_target"])
    protect = not is_base
    C0, C1 = build_mmpp(rho, delta, sigma, BASELINE["c"], BASELINE["b"], BASELINE["mu"])
    params = PredictiveModelParameters(
        C0=C0, C1=C1, alpha=alpha, beta=beta, n_target=nt, gamma=1.0,
        protect_presetup=protect, protect_delayoff=protect, **BASELINE)
    pi = solve_stationary(build_generator(params), solver="gth")
    m = Metrics(params, pi)
    c, K, b = params.c, params.K, params.b
    _, pairs = build_is_index(c)
    P = pi_by_level(pi, c, K, params.D_M)  # [j, iota(i,s), F]
    Bm = np.array([[busy_count(i, j, b) for (i, s) in pairs] for j in range(K + 1)])
    Im = np.array([[idle_count(i, j, b) for (i, s) in pairs] for j in range(K + 1)])
    Sm = np.array([[s for (i, s) in pairs] for j in range(K + 1)], dtype=float)
    out = {"case": p["case"], "series": p["series"], "rho": rho, "delta": delta,
           "sigma": sigma, "alpha": alpha, "beta": beta, "n_target": nt, "gamma": 1.0,
           "protect": protect, "E_B": m.E_B, "E_S": m.E_S, "E_I": m.E_I,
           "Cost": m.energy_cost_paper(), "ERP": m.erp_paper(),
           "P_block_arrival_stable": m.P_block_arrival_stable}
    for F in (0, 1):
        PF = P[:, :, F]
        eb, es, ei = float((PF * Bm).sum()), float((PF * Sm).sum()), float((PF * Im).sum())
        out.update({
            f"E_B_F{F}": eb, f"E_S_F{F}": es, f"E_I_F{F}": ei,
            f"Cost_F{F}": params.C_b * eb + params.C_s * es + params.C_i * ei,
            f"P_arr_contrib_F{F}": float(PF[K].sum()) * params.lambdas[F] / params.lambda_bar,
            f"P_time_F{F}": float(PF[K].sum()),
            f"mass_F{F}": float(PF.sum()),
        })
    # 既存 CSV の値との一致
    rel = max(abs(out[k] / _f(src, k2) - 1) for k, k2 in
              [("E_B", "E_B"), ("E_S", "E_S"), ("E_I", "E_I"), ("Cost", "Cost"),
               ("ERP", "ERP"), ("P_block_arrival_stable", "P_block_arrival_stable")])
    out["csv_max_relerr"] = rel
    out["min_pi"] = float(pi.min())
    checks = []
    if abs(m.E_B + m.E_I + m.E_S + m.E_off - c) > 1e-9:
        checks.append("conservation")
    if abs(m.lambda_eff - b * params.mu * m.E_B) > 1e-9 * m.lambda_eff:
        checks.append("flow")
    if pi.min() < 0:
        checks.append("min_pi")
    if not m.P_block_arrival_stable > 0:
        checks.append("P_block_nonpositive")
    if rel > 1e-9:
        checks.append(f"csv_mismatch:{rel:.1e}")
    for k in ("E_B", "E_S", "E_I", "Cost", "P_block_arrival_stable"):
        part = out[f"{k}_F0" if k != "P_block_arrival_stable" else "P_arr_contrib_F0"] + \
            out[f"{k}_F1" if k != "P_block_arrival_stable" else "P_arr_contrib_F1"]
        if abs(part - out[k]) > 1e-9 * max(abs(out[k]), 1e-300):
            checks.append(f"phase_sum:{k}")
    out["check"] = ";".join(checks) if checks else "ok"
    return out


def solve_points(points: List[dict]) -> List[dict]:
    """解き直した点を CSV に 1 点ずつ追記する (計算済みの点は飛ばす)."""
    done = {}
    if os.path.exists(POINTS_CSV):
        with open(POINTS_CSV, newline="", encoding="utf-8") as f:
            done = {(r["case"], r["series"]): r for r in csv.DictReader(f)}
    for p in points:
        if (p["case"], p["series"]) in done:
            continue
        row = solve_point(p)
        new = not os.path.exists(POINTS_CSV)
        with open(POINTS_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=POINT_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"解き直し: {p['case']} {p['series']} check={row['check']}", flush=True)
    with open(POINTS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ============================================================
# レポート
# ============================================================

def coefs():
    b = BASELINE["b"]
    cb = 0.04419 * b + 0.15503
    return cb, cb, 0.6 * cb


def step1(sm) -> List[str]:
    rows = []
    for (row, r1), c in sm.items():
        bb = c["base_best"]["beta"]
        bp = c["series"][P1]["best"]["beta"]
        rows.append((row, r1, bp, bb, bp / bb, -c["series"][P1]["erp_rel"]))
    rows.sort(key=lambda t: -t[5])
    lines = ["## 1. β* の比 R と P1 only (protect) の改善", "",
             "R = β*(P1 only (protect)) / β*(Base). β* は各マスの ERP 最小点の β "
             "(Base は細かくした探索を含む). 改善 = −(ERP の相対差). 改善の大きい順.", "",
             "| 行 | r1 | β*(P1) | β*(Base) | R | log10 R | 改善 |", "|---|---|---|---|---|---|---|"]
    for row, r1, bp, bb, R, imp in rows:
        lines.append(f"| {row} | {r1:.3g} | {bp:.4g} | {bb:.4g} | {R:.3g} | "
                     f"{np.log10(R):+.2f} | {100 * imp:+.3f}% |")
    lines += ["", "| 対象 | マス数 | Pearson (R) | Spearman (R) | Pearson (log10 R) |",
              "|---|---|---|---|---|"]
    stats = {}
    for name, sel in [("全マス", rows), ("改善 > 0.5%", [t for t in rows if t[5] > 0.005])]:
        R = np.array([t[4] for t in sel])
        imp = np.array([t[5] for t in sel])
        pr = pearsonr(R, imp)[0]
        sr = spearmanr(R, imp).correlation
        pl = pearsonr(np.log10(R), imp)[0]
        stats[name] = (len(sel), pr, sr, pl)
        lines.append(f"| {name} | {len(sel)} | {pr:+.3f} | {sr:+.3f} | {pl:+.3f} |")
    n_gt1 = sum(1 for t in rows if t[4] > 1)
    n_gt1_imp = sum(1 for t in rows if t[4] > 1 and t[5] > 0.005)
    n_imp = sum(1 for t in rows if t[5] > 0.005)
    lines += ["", f"R > 1 (P1 only (protect) の方が β* が大きい) のマスは {len(rows)} マス中 {n_gt1} マス. "
              f"改善が 0.5% を超える {n_imp} マスのうち R > 1 は {n_gt1_imp} マス.", ""]
    return lines, stats


def step2(points: List[dict]) -> List[str]:
    cb, cs, ci = coefs()
    lines = ["## 2. 電力の内訳 (ERP 最小点)", "",
             f"C_b = C_s = {cb:.5f}, C_i = {ci:.6f} (b=5). 各値は既存 CSV の E[B], E[S], E[I] から計算. "
             "ΔCost = Cost(P1 only (protect)) − Cost(Base) と, その成分ごとの差.", "",
             "| 対象 | 系列 | β* | n_target | C_b·E[B] | C_s·E[S] | C_i·E[I] | Cost | E[W] | ERP |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    diffs = []
    cases = list(dict.fromkeys(p["case"] for p in points))
    for case in cases:
        pair = {p["series"]: p["src"] for p in points if p["case"] == case}
        vals = {}
        for s in ("Base", P1):
            r = pair[s]
            v = (cb * _f(r, "E_B"), cs * _f(r, "E_S"), ci * _f(r, "E_I"))
            vals[s] = v
            nt = "—" if s == "Base" else r["n_target"]
            lines.append(f"| {case} | {s} | {_f(r, 'beta'):.4g} | {nt} | {v[0]:.4f} | {v[1]:.4f} | "
                         f"{v[2]:.4f} | {sum(v):.4f} | {_f(r, 'E_W'):.4f} | {_f(r, 'ERP'):.4f} |")
        d = [vals[P1][k] - vals["Base"][k] for k in range(3)]
        diffs.append((case, d, sum(d),
                      _f(pair[P1], "E_W") / _f(pair["Base"], "E_W") - 1,
                      _f(pair[P1], "ERP") / _f(pair["Base"], "ERP") - 1))
    lines += ["", "| 対象 | Δ C_b·E[B] | Δ C_s·E[S] | Δ C_i·E[I] | ΔCost | ΔCost/Cost(Base) | "
              "ΔE[W]/E[W](Base) | ΔERP/ERP(Base) | ΔCost の主な成分 |",
              "|---|---|---|---|---|---|---|---|---|"]
    names = ["C_b·E[B]", "C_s·E[S]", "C_i·E[I]"]
    for case, d, dc, dw, de in diffs:
        base_cost = None
        for p in points:
            if p["case"] == case and p["series"] == "Base":
                base_cost = _f(p["src"], "Cost")
        main = names[int(np.argmax([abs(x) for x in d]))]
        lines.append(f"| {case} | {d[0]:+.4f} | {d[1]:+.4f} | {d[2]:+.4f} | {dc:+.4f} | "
                     f"{100 * dc / base_cost:+.2f}% | {100 * dw:+.2f}% | {100 * de:+.2f}% | {main} |")
    lines.append("")
    return lines, diffs


def step3(prow: List[dict]) -> List[str]:
    lines = ["## 3. 位相ごとの内訳 (同じ点を GTH で解き直したもの)", "",
             "mechanism_points.csv から. E[·]_F は位相 F の状態での寄与 (F=0 と F=1 を足すと全体). "
             "P_arr 寄与は Σ π(·,K,F) λ_F / λ̄ (足すと P_block_arrival_stable). "
             "Base はベースモデルと一致する Predictive (n_target=0, γ=1, 保護なし) として解き, "
             "全体の値と既存 CSV の値の最大の相対誤差を csv 列に示す.", "",
             "| 対象 | 系列 | F | E[B]_F | E[S]_F | E[I]_F | Cost_F | P_arr 寄与_F | check | csv |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in prow:
        for F in (0, 1):
            lines.append(f"| {r['case'] if F == 0 else ''} | {r['series'] if F == 0 else ''} | {F} | "
                         f"{_f(r, f'E_B_F{F}'):.4f} | {_f(r, f'E_S_F{F}'):.4f} | "
                         f"{_f(r, f'E_I_F{F}'):.4f} | {_f(r, f'Cost_F{F}'):.4f} | "
                         f"{_f(r, f'P_arr_contrib_F{F}'):.4e} | "
                         f"{r['check'] if F == 0 else ''} | "
                         f"{_f(r, 'csv_max_relerr'):.1e} |" if F == 0 else
                         f"| | | {F} | {_f(r, f'E_B_F{F}'):.4f} | {_f(r, f'E_S_F{F}'):.4f} | "
                         f"{_f(r, f'E_I_F{F}'):.4f} | {_f(r, f'Cost_F{F}'):.4f} | "
                         f"{_f(r, f'P_arr_contrib_F{F}'):.4e} | | |")
    lines += ["", "P1 only (protect) − Base の差 (位相ごと):", "",
              "| 対象 | ΔCost_F0 | ΔCost_F1 | ΔE[I]_F0 | ΔE[S]_F0 | ΔE[B]_F1 | ΔE[S]_F1 | "
              "ΔP_arr_F0 | ΔP_arr_F1 |", "|---|---|---|---|---|---|---|---|---|"]
    by = {}
    for r in prow:
        by.setdefault(r["case"], {})[r["series"]] = r
    out = []
    for case, d in by.items():
        b, p = d["Base"], d[P1]
        g = lambda k: _f(p, k) - _f(b, k)
        out.append((case, g("Cost_F0"), g("Cost_F1"), g("P_arr_contrib_F0"),
                    g("P_arr_contrib_F1")))
        lines.append(f"| {case} | {g('Cost_F0'):+.4f} | {g('Cost_F1'):+.4f} | "
                     f"{g('E_I_F0'):+.4f} | {g('E_S_F0'):+.4f} | {g('E_B_F1'):+.4f} | "
                     f"{g('E_S_F1'):+.4f} | {g('P_arr_contrib_F0'):+.3e} | "
                     f"{g('P_arr_contrib_F1'):+.3e} |")
    lines.append("")
    return lines, out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sm, cells = exp8_top_cells(3)
    points = exp7_points() + exp8_points(sm, cells)
    prow = solve_points(points)
    order = [(p["case"], p["series"]) for p in points]
    prow.sort(key=lambda r: order.index((r["case"], r["series"])))

    l1, stats = step1(sm)
    l2, diffs = step2(points)
    l3, phase = step3(prow)
    head = ["# Predictive が優位になる仕組み (実験 7・8 の集計)", "",
            "実験 8 の CSV (results/experiment_8/experiment_8_<行>.csv), 実験 7 の CSV "
            "(results/experiment_7/), 解き直した点の CSV (results/experiment_8/mechanism_points.csv) "
            "から scripts/analyze_mechanism.py で生成. 実験 8 で P1 only (protect) の改善が最大の 3 マスは "
            + ", ".join(f"{row} r1={r1:.3g}" for row, r1 in cells) + ".", ""]
    text = "\n".join(head + l1 + l2 + l3)
    concl = os.path.join(OUT_DIR, "mechanism_conclusion.md")
    if os.path.exists(concl):
        with open(concl, encoding="utf-8") as f:
            text += "\n" + f.read().rstrip() + "\n"
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
