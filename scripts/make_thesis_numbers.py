#!/usr/bin/env python
"""論文で引用する数値・表・図を, results/ の CSV とレポートの集計から生成する (新しい計算はしない).

出力:
    thesis_assets/numbers.tex   数値マクロ (\\newcommand). 各マクロの直前に出典と計算方法のコメント.
    thesis_assets/tables/*.tex  実験 7・8・9 の主要な表 (table 環境; 数値はマクロか CSV から).
    thesis_assets/figures/      論文で使う図 (figures/ からのコピー).
    thesis_assets/README.md     図・表と論文の節の対応表.

マクロ名は「Exp + 実験番号 (英単語) + 内容」の英字だけで付ける (LaTeX のマクロ名に数字は使えない).

使用例:
    python scripts/make_thesis_numbers.py
"""
import csv
import os
import re
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

try:
    import experiment_7_frontier as e7
    import experiment_8_map as e8
    import experiment_9_sensitivity as e9
    import blocking_reassessment as br
    import evaluate_predictive_comparison as ev
except ImportError:
    import scripts.experiment_7_frontier as e7
    import scripts.experiment_8_map as e8
    import scripts.experiment_9_sensitivity as e9
    import scripts.blocking_reassessment as br
    import scripts.evaluate_predictive_comparison as ev

OUT = "thesis_assets"
P = "P_block_arrival_stable"
WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]


# ============================================================
# マクロ
# ============================================================

class Macros:
    def __init__(self):
        self.items: List[Tuple[str, str, str]] = []  # (名前, 値, コメント)

    def add(self, name: str, value: str, comment: str) -> None:
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"マクロ名は英字だけ: {name}")
        if any(n == name for n, _, _ in self.items):
            raise ValueError(f"マクロ名の重複: {name}")
        self.items.append((name, value, comment))

    def write(self, path: str) -> None:
        lines = ["% 論文で引用する数値. scripts/make_thesis_numbers.py で生成 (手で編集しない).", ""]
        for name, value, comment in self.items:
            for c in comment.splitlines():
                lines.append(f"% {c}")
            lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
            lines.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def word_num(s: str) -> str:
    """文字列の中の数字を英単語に置き換える (例: C1 -> COne)."""
    return "".join(WORDS[int(ch)] if ch.isdigit() else ch for ch in s)


def pct(v: float, digits: int = 2) -> str:
    """符号なしの百分率 (例: 3.79\\%)."""
    return f"{100 * v:.{digits}f}\\%"


def spct(v: float, digits: int = 2) -> str:
    """符号付きの百分率 (例: $-1.19$\\%)."""
    return f"${100 * v:+.{digits}f}$\\%"


def num(v: float, fmt: str = ".4f") -> str:
    return format(v, fmt)


SERIES_NAME = {"Base": "Base", "NoCancel": "NoCancel", "P2 only": "PTwoOnly",
               "P1 only": "POneOnly", "P1 only (protect)": "POneOnlyProtect",
               "P1+P2": "POneTwo", "P1+P2 (protect)": "POneTwoProtect"}


def cell_tex(row: str, r1: float) -> str:
    rho_B, alpha = e8.ROWS[row]
    head = f"$\\rho_B={rho_B:g}$" if row.startswith("A_") else f"$\\alpha={alpha:g}$"
    return f"{head}, $r_1={r1:.3g}$"


# ============================================================
# 実験 0
# ============================================================

def exp0(m: Macros) -> None:
    rep = os.path.join("results_gth", "VALIDATION_REPORT.md")
    text = open(rep, encoding="utf-8").read()
    mt = re.search(r"\| 実験 0 \| ベース[^|]*\| 全 (\d+) 点・全 (\d+) 指標が CI 内", text)
    npts, nmet = int(mt.group(1)), int(mt.group(2))
    src = f"出典: {rep} の V-Sim の表 (実験 0 の CSV はなく, 結果はこの表にだけある)"
    m.add("ExpZeroBasePoints", str(npts), f"{src}\n実験 0 (ベースモデル) で DES と照合した ρ の点の数")
    m.add("ExpZeroMetrics", str(nmet), f"{src}\n照合した指標の数 (P_block, E[N], E[W], λ_eff, ρ_server, Cost)")
    m.add("ExpZeroBaseInCI", str(npts * nmet),
          f"{src}\n理論値が DES の 95% 信頼区間に入った (点, 指標) の組の数 (全点・全指標が CI 内)")

    path = os.path.join("results", "protect_des", "protect_des.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for pol, name in [("both", "Both"), ("nocancel", "NoCancel")]:
        rs = [r for r in rows if r["policy"] == pol]
        inci = sum(int(r["n_in_ci_6"]) for r in rs)
        m.add(f"ExpZeroProtect{name}InCI", str(inci),
              f"出典: {path} (policy={pol})\n6 指標の理論値が DES の 95% 信頼区間に入った組の数 (n_in_ci_6 の和)")
        m.add(f"ExpZeroProtect{name}Total", str(6 * len(rs)),
              f"出典: {path} (policy={pol})\n照合した組の総数 (ρ の点の数 × 6 指標)")
    unobs = sum(1 for r in rows if r["P_block_in_ci"] != "True"
                and float(r["P_block_lo"]) == 0.0 and float(r["P_block_hi"]) == 0.0)
    m.add("ExpZeroProtectUnobservable", str(unobs),
          f"出典: {path}\nDES でブロックが 1 回も観測されず (CI が [0, 0]) CI 外と判定された P_block の数")


# ============================================================
# 実験 7
# ============================================================

def exp7(m: Macros) -> dict:
    data = br.load_exp7()
    res = {}
    for cond, groups in data.items():
        a = br.assess(groups, br.E7_SERIES)
        res[cond] = a
        cn = word_num(cond)
        src = (f"出典: results/experiment_7/experiment_7_{cond}.csv と experiment_7_refine_{cond}.csv "
               "(scripts/blocking_reassessment.py の load_exp7 と assess)")
        for s in ["Base"] + br.E7_SERIES:
            v = a["series"][s]
            sn = SERIES_NAME[s]
            m.add(f"ExpSeven{cn}{sn}ERPMin", num(v["best"]["ERP"]),
                  f"{src}\n条件 {cond}, 系列 {s} の ERP 最小値 (Base は β を細かくした 121 点を含む)")
            if s == "Base":
                continue
            m.add(f"ExpSeven{cn}{sn}ERPRel", spct(v["erp_rel"]),
                  f"{src}\n条件 {cond}, 系列 {s} の ERP 最小値の, Base の ERP 最小値に対する相対差")
            m.add(f"ExpSeven{cn}{sn}PRatio", f"{v['p_ratio']:.2f}",
                  f"{src}\n条件 {cond}, 系列 {s} の ERP 最小点の P_block_arrival_stable の, Base の ERP 最小点の値に対する比")
            m.add(f"ExpSeven{cn}{sn}SameRelImprovement",
                  "該当なし" if v["b_rel"] is None else spct(-v["b_rel"]),
                  f"{src}\n条件 {cond}, 系列 {s} の同じ信頼性での改善: P ≤ P_ref (Base の ERP 最小点の P) を"
                  "\n満たす点の中の ERP 最小値の, Base の ERP 最小値に対する −(相対差)")
        m.add(f"ExpSeven{cn}BasePBlock", f"{a['p_ref']:.3g}",
              f"{src}\n条件 {cond}, Base の ERP 最小点の P_block_arrival_stable (= P_ref)")
        n_ok = sum(1 for s in ["Base"] + br.E7_SERIES if a["series"][s]["c"][1e-2]["pt"] is not None)
        m.add(f"ExpSeven{cn}EpsTwoFeasibleCount", str(n_ok),
              f"{src}\n条件 {cond}, P_block_arrival_stable ≤ 1e-2 を満たす点をもつ系列の数 (Base を含む 7 系列中)")
    src = ("出典: results/experiment_7/ の CSV (scripts/blocking_reassessment.py の assess)\n"
           "位相を使う 5 系列 (P2 only, P1 only, P1 only (protect), P1+P2, P1+P2 (protect)) と 5 条件で")
    phase = [s for s in br.E7_SERIES if s != "NoCancel"]
    m.add("ExpSevenMaxERPImprovement",
          pct(max(-res[c]["series"][s]["erp_rel"] for c in res for s in phase)),
          f"{src}\nERP 最小値の改善 (−相対差, 対 β を細かくした Base) の最大値")
    m.add("ExpSevenMaxSameRelImprovement",
          pct(max(-res[c]["series"][s]["b_rel"] for c in res for s in phase
                  if res[c]["series"][s]["b_rel"] is not None)),
          f"{src}\n同じ信頼性 (P ≤ P_ref) での改善の最大値")
    nc = [res[c]["series"]["NoCancel"]["erp_rel"] for c in res]
    m.add("ExpSevenNoCancelWorseMin", pct(min(nc)),
          f"{src.splitlines()[0]}\nNoCancel の ERP 最小値の, Base に対する相対差の最小値 (5 条件すべてで正)")
    m.add("ExpSevenNoCancelWorseMax", pct(max(nc)), f"{src.splitlines()[0]}\n同じく最大値")
    return res


# ============================================================
# 旧比較 (β 固定) と保護の比較
# ============================================================

def old_comparison(m: Macros) -> None:
    """実験 1-P〜4-P と 1〜4 の旧比較 (保護なしの旧モデル, Base の β を固定)."""
    for label, s14, s23, name in [("γ=5", "g5.0", "g5.0", "GammaFive"),
                                  ("γ を最適化", "g1.0", "gmap", "GammaOpt")]:
        pts = ev.load_experiment_1p("results", 10, s14)
        for b in ("medium", "strong"):
            pts += ev.load_experiment_2p("results", 10, s23, b)
        for sw in ("delta", "sigma"):
            pts += ev.load_experiment_3p("results", 10, s23, sw)
        for b in ("weak", "medium", "strong"):
            pts += ev.load_experiment_4p("results", 10, s14, b)
        erp = np.array([p.improvement_pct for p in pts if p.metric == "ERP"])
        src = (f"出典: results/experiment_{{1..4}}*.csv と experiment_{{1P..4P}}_*_{{{s14},{s23}}}.csv "
               "(scripts/evaluate_predictive_comparison.py の load_experiment_*)\n"
               f"旧比較 (保護なしの旧モデル, Base は β を走査の値に固定, Predictive は n_target=10, {label}) の ERP の改善率")
        m.add(f"ExpOldBetaFixed{name}MaxERPImprovement", f"{erp.max():.2f}\\%", f"{src}\n405 点中の最大値")
        m.add(f"ExpOldBetaFixed{name}CountOverFive", str(int((erp > 5).sum())),
              f"{src}\n改善率が 5% を超える点の数 (405 点中)")
        m.add(f"ExpOldBetaFixed{name}MeanERPImprovement", f"${erp.mean():+.2f}$\\%", f"{src}\n405 点の平均")


def protect_comparison(m: Macros) -> None:
    """保護の比較 (results/p1_protect/p1_protect.csv)."""
    path = os.path.join("results", "p1_protect", "p1_protect.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))

    def get(burst, rho, alpha, beta, gamma, mode):
        for r in rows:
            if (r["burst_name"] == burst and float(r["rho"]) == rho and float(r["alpha"]) == alpha
                    and float(r["beta"]) == beta and float(r["gamma"]) == gamma and r["mode"] == mode):
                return r
        raise KeyError((burst, rho, alpha, beta, gamma, mode))

    src = (f"出典: {path}\n条件 medium, ρ=0.3, α=0.1, β=0.5 で, 保護なし (none, n_target=10, γ=1) の ERP に対する相対差")
    none = float(get("medium", 0.3, 0.1, 0.5, 1.0, "none")["ERP"])
    for mode, name in [("presetup", "Presetup"), ("both", "Both")]:
        v = float(get("medium", 0.3, 0.1, 0.5, 1.0, mode)["ERP"]) / none - 1
        m.add(f"ProtectExample{name}ERPChange", spct(v), f"{src}\n保護の設定 {mode} (n_target=10, γ=1)")
    v = float(get("medium", 0.3, 0.1, 0.5, 1.0, "nocancel")["ERP"]) / none - 1
    m.add("ProtectExampleNoCancelERPChange", spct(v), f"{src}\nNoCancel (never_cancel_setup, n_target=0, γ=1)")
    # NoCancel が同じ γ=1 の none より悪い条件の数
    worse, total = 0, 0
    for r in rows:
        if r["mode"] != "nocancel":
            continue
        n = get(r["burst_name"], float(r["rho"]), float(r["alpha"]), float(r["beta"]), 1.0, "none")
        total += 1
        worse += float(r["ERP"]) > float(n["ERP"])
    m.add("ProtectNoCancelWorseCount", str(worse),
          f"出典: {path}\nNoCancel の ERP が同じ (条件, α, β) の none (γ=1) より大きい組の数")
    m.add("ProtectNoCancelTotal", str(total), f"出典: {path}\n上の組の総数")


# ============================================================
# 実験 8
# ============================================================

E8_SERIES = list(e8.SERIES)


def exp8(m: Macros) -> dict:
    sm = e8.summaries()
    src = "出典: results/experiment_8/experiment_8_<行>.csv (scripts/experiment_8_map.py の summaries)"
    imp = {k: {s: -c["series"][s]["erp_rel"] for s in E8_SERIES} for k, c in sm.items()}
    core = [k for k, v in imp.items() if max(v.values()) > 0.005]
    best_core = [max(imp[k].values()) for k in core]
    m.add("ExpEightCoreCellCount", str(len(core)),
          f"{src}\n3 系列のいずれかで改善 (−ERP の相対差, 対 β を細かくした Base) が 0.5% を超えるマスの数")
    m.add("ExpEightCellCount", str(len(sm)), f"{src}\n地図 A・B の重複のないマスの数")
    m.add("ExpEightCoreImprovementMin", pct(min(best_core)),
          f"{src}\n改善が 0.5% を超えるマスでの, 3 系列のうち最良の改善の最小値")
    m.add("ExpEightCoreImprovementMax", pct(max(best_core)),
          f"{src}\n改善が 0.5% を超えるマスでの, 3 系列のうち最良の改善の最大値")
    # 中心の領域: r1=17.8 の列のうち ρ_B ≤ 0.8 かつ α ≤ 1 のマスと, α=1, r1=3.16 のマス
    center = [k for k in sm if (abs(k[1] / e8.R1_LEVELS[3] - 1) < 1e-9 and e8.ROWS[k[0]][0] <= 0.8
                                and e8.ROWS[k[0]][1] <= 1.0)
              or (k[0] == "B_alpha1" and abs(k[1] / e8.R1_LEVELS[2] - 1) < 1e-9)]
    cbest = [max(imp[k].values()) for k in center]
    srcc = (f"{src}\n中心の領域 = r1=17.8 の列のうち ρ_B ≤ 0.8 かつ α ≤ 1 のマスと, α=1, r1=3.16 のマス "
            "(地図 A・B で重複のないもの)\n該当: " + ", ".join(f"{r} r1={x:.3g}" for r, x in sorted(center)))
    m.add("ExpEightCenterCellCount", str(len(center)), f"{srcc}\nマスの数")
    m.add("ExpEightCenterImprovementMin", pct(min(cbest)),
          f"{srcc}\n3 系列のうち最良の改善 (−ERP の相対差) の最小値")
    m.add("ExpEightCenterImprovementMax", pct(max(cbest)), f"{srcc}\n同じく最大値")
    kmax, smax = max(((k, s) for k in sm for s in E8_SERIES), key=lambda t: imp[t[0]][t[1]])
    m.add("ExpEightMaxImprovement", pct(imp[kmax][smax]),
          f"{src}\n全マス・全系列で改善 (−ERP の相対差) の最大値")
    m.add("ExpEightMaxImprovementSeries", smax, f"{src}\n改善が最大の系列")
    m.add("ExpEightMaxImprovementCell", cell_tex(*kmax), f"{src}\n改善が最大のマス")
    for s in E8_SERIES:
        sn = SERIES_NAME[s]
        k = max(sm, key=lambda k: imp[k][s])
        m.add(f"ExpEight{sn}MaxImprovement", pct(imp[k][s]), f"{src}\n系列 {s} の改善の最大値")
        m.add(f"ExpEight{sn}MaxImprovementCell", cell_tex(*k), f"{src}\n系列 {s} の改善が最大のマス")
        m.add(f"ExpEight{sn}CoreCellCount", str(sum(1 for v in imp.values() if v[s] > 0.005)),
              f"{src}\n系列 {s} の改善が 0.5% を超えるマスの数")
    # r1=0.1 の列
    col = [imp[(row, e8.R1_LEVELS[0])][s] for row in dict.fromkeys(e8.MAP_A + e8.MAP_B)
           for s in E8_SERIES]
    m.add("ExpEightROneMinMaxImprovement", pct(max(col)),
          f"{src}\nr1=0.1 の列の 7 マス × 3 系列での改善の最大値")
    # R と改善の相関
    R = np.array([c["series"]["P1 only (protect)"]["best"]["beta"] / c["base_best"]["beta"]
                  for c in sm.values()])
    I = np.array([imp[k]["P1 only (protect)"] for k in sm])
    sel = I > 0.005
    srcR = (f"{src}\nR = β*(P1 only (protect)) / β*(Base) と P1 only (protect) の改善の相関 "
            "(results/experiment_8/mechanism.md の手順 1 と同じ)")
    m.add("ExpEightRPearsonAll", f"{pearsonr(R, I)[0]:+.3f}", f"{srcR}\n全マス, Pearson")
    m.add("ExpEightRSpearmanAll", f"{spearmanr(R, I).correlation:+.3f}", f"{srcR}\n全マス, Spearman")
    m.add("ExpEightRPearsonCore", f"{pearsonr(R[sel], I[sel])[0]:+.3f}",
          f"{srcR}\n改善が 0.5% を超えるマス, Pearson")
    m.add("ExpEightRSpearmanCore", f"{spearmanr(R[sel], I[sel]).correlation:+.3f}",
          f"{srcR}\n改善が 0.5% を超えるマス, Spearman")
    m.add("ExpEightRCoreCount", str(int(sel.sum())), f"{srcR}\n改善が 0.5% を超えるマスの数 (P1 only (protect))")
    m.add("ExpEightRAboveOneCore", str(int(((R > 1) & sel).sum())),
          f"{srcR}\n改善が 0.5% を超えるマスのうち R > 1 のマスの数")
    # p1_fire_rate の相関
    fp = e8.fire_points(sm)
    ps = [q["p_stage2"] for q in fp]
    m.add("ExpEightFirePearsonRaw", f"{pearsonr(ps, [q['p1_fire_rate'] for q in fp])[0]:+.3f}",
          f"{src}\nP1 only (protect) の ERP 最小点の p1_fire_rate と 1/(1+1/r1) の Pearson の相関係数")
    m.add("ExpEightFirePearsonPerOnset",
          f"{pearsonr(ps, [q['fire_per_onset'] for q in fp])[0]:+.3f}",
          f"{src}\n位相 0→1 の遷移 1 回あたりの発動確率 p1_fire_rate/(σϖ₀) と 1/(1+1/r1) の相関")
    # α の傾向 (地図 B, P1 only (protect))
    for r1, nm in [(e8.R1_LEVELS[3], "ROneSeventeen"), (e8.R1_LEVELS[4], "ROneHundred")]:
        v = [imp[(row, r1)]["P1 only (protect)"] for row in e8.MAP_B]
        rho = spearmanr(np.log10([e8.ROWS[r][1] for r in e8.MAP_B]), v).correlation
        m.add(f"ExpEightAlphaSpearman{nm}", f"{rho:+.2f}",
              f"{src}\n地図 B, r1={r1:.3g} での log10 α と P1 only (protect) の改善の Spearman の順位相関")
    # 同じ信頼性での改善 (blocking reassessment)
    data = br.load_exp8()
    ares = {k: br.assess(g, br.E8_SERIES) for k, g in data.items()}
    srcb = ("出典: results/experiment_8/experiment_8_<行>.csv (scripts/blocking_reassessment.py の assess)\n"
            "同じ信頼性 (P ≤ P_ref) の中の ERP 最小値の, Base の ERP 最小値に対する −(相対差)")
    for s in E8_SERIES:
        vals = [(-c["series"][s]["b_rel"], k) for k, c in ares.items()
                if c["series"][s]["b_rel"] is not None]
        v, k = max(vals)
        sn = SERIES_NAME[s]
        m.add(f"ExpEight{sn}SameRelMaxImprovement", pct(v), f"{srcb}\n系列 {s} の最大値")
        m.add(f"ExpEight{sn}SameRelImprovedCells", str(sum(1 for x, _ in vals if x > 0.001)),
              f"{srcb}\n系列 {s} で 0.1% を超えて改善するマスの数")
        pr = [c["series"][s]["p_ratio"] for c in ares.values()]
        m.add(f"ExpEight{sn}PRatioMax", f"{max(pr):.2f}",
              f"出典: results/experiment_8/ の CSV (assess)\n系列 {s} の ERP 最小点の P の, Base の ERP 最小点の P に対する比の最大値")
    infeas = [k for k, c in ares.items() if c["base_eps"][1e-2] is None]
    m.add("ExpEightBaseInfeasibleEpsTwoCount", str(len(infeas)),
          "出典: results/experiment_8/ の CSV (assess)\nBase も 3 系列も P ≤ 1e-2 を満たせないマスの数\n"
          "該当: " + ", ".join(f"{r} r1={x:.3g}" for r, x in sorted(infeas)))
    # ERP だけの判定と同じ信頼性の判定で区分が変わった組 (blocking_reassessment と同じ定義)
    changes = sum(1 for c in ares.values() for s in E8_SERIES
                  if br._cat(c["series"][s]["erp_rel"]) != br._cat(c["series"][s]["b_rel"]))
    m.add("ExpEightCategoryChangedCount", str(changes),
          "出典: results/experiment_8/blocking_reassessment.md と同じ計算\n"
          "ERP だけの判定と同じ信頼性での判定で, 区分 (改善 > 0.1%, 同等, 悪化, 該当なし) が変わった (マス, 系列) の数")
    mechanism(m)
    return sm


def mechanism(m: Macros) -> None:
    path = os.path.join("results", "experiment_8", "mechanism_points.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    by = {}
    for r in rows:
        by.setdefault(r["case"], {})[r["series"]] = r
    sel = [c for c in by if c == "実験7 C1" or c.startswith("実験8")]
    d = lambda c, k: float(by[c]["P1 only (protect)"][k]) - float(by[c]["Base"][k])
    rel = lambda c, k: float(by[c]["P1 only (protect)"][k]) / float(by[c]["Base"][k]) - 1
    src = (f"出典: {path}\n実験 7 の C1 と実験 8 で改善が最大の 3 マスの, P1 only (protect) − Base (ERP 最小点どうし)")
    idle = [-d(c, "E_I_F0") for c in sel]
    m.add("MechIdleFZeroReductionMin", f"{min(idle):.2f}", f"{src}\n通常期 F=0 のアイドル台数 E[I]_F0 の減少量の最小値")
    m.add("MechIdleFZeroReductionMax", f"{max(idle):.2f}", f"{src}\n同じく最大値")
    c0 = [d(c, "Cost_F0") for c in sel]
    c1 = [d(c, "Cost_F1") for c in sel]
    m.add("MechCostFZeroChangeMin", f"${min(c0):+.2f}$", f"{src}\n通常期の Cost_F0 の差の最小値")
    m.add("MechCostFZeroChangeMax", f"${max(c0):+.2f}$", f"{src}\n同じく最大値")
    m.add("MechCostFOneChangeMin", f"${min(c1):+.2f}$", f"{src}\nバースト期の Cost_F1 の差の最小値")
    m.add("MechCostFOneChangeMax", f"${max(c1):+.2f}$", f"{src}\n同じく最大値")
    ct = [rel(c, "Cost") for c in sel]
    m.add("MechCostChangeMin", spct(min(ct)), f"{src}\nCost の相対差の最小値")
    m.add("MechCostChangeMax", spct(max(ct)), f"{src}\nCost の相対差の最大値")


# ============================================================
# 実験 9
# ============================================================

B_NAMES = {"P1 only (protect) n_target=10": "POneNTen", "P1 only (protect) n_target=20": "POneNTwenty",
           "P2 only γ=3": "PTwoGammaThree", "P2 only γ=10": "PTwoGammaTen"}
CAT_NAMES = {"改善": "Improved", "同等": "Equal", "棄却のみ改善": "BlockingOnly",
             "引き換え": "TradeOff", "悪化": "Worse"}


def exp9(m: Macros) -> dict:
    path = os.path.join("results", "experiment_9", "experiment_9_B.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    src = (f"出典: {path} (しきい値付きの 5 区分: scripts/experiment_9_sensitivity.py の category5;\n"
           "|ERP の差| ≤ 0.1%, P の比 0.95〜1.05 を同等. 比較の相手は同じ β* の Base)")
    out = {}
    for name, short in B_NAMES.items():
        cr = [r for r in rows if r["value"] == name]
        cats = {c: [r for r in cr if e9.category5(float(r["rel_ERP"]), 1 + float(r["rel_P"])) == c]
                for c in e9.CATEGORIES5}
        out[name] = cats
        for c, cn in CAT_NAMES.items():
            m.add(f"ExpNine{short}{cn}Count", str(len(cats[c])), f"{src}\n設定 {name} で区分「{c}」のマスの数")
        m.add(f"ExpNine{short}PUpCount", str(sum(1 for r in cr if 1 + float(r["rel_P"]) > e9.P_HI)),
              f"{src}\n設定 {name} で P の比 (対同じ β* の Base) が 1.05 を超えるマスの数 (35 マス中)")
        if cats["改善"]:
            b = min(cats["改善"], key=lambda r: float(r["rel_ERP"]))
            m.add(f"ExpNine{short}MaxImprovement", pct(-float(b["rel_ERP"])),
                  f"{src}\n設定 {name} の, 区分「改善」の中で ERP が最も下がるマスの −(ERP の相対差)")
            m.add(f"ExpNine{short}MaxImprovementCell", cell_tex(b["row"], float(b["r1"])),
                  f"{src}\n上のマス")
        bad = cats["悪化"] + cats["引き換え"]
        if bad:
            w = max(bad, key=lambda r: float(r["rel_ERP"]))
            m.add(f"ExpNine{short}MaxWorseERP", spct(float(w["rel_ERP"])),
                  f"{src}\n設定 {name} の, 区分「悪化」「引き換え」の中で ERP の相対差の最大値")
            m.add(f"ExpNine{short}MaxWorseERPCell", cell_tex(w["row"], float(w["r1"])), f"{src}\n上のマス")
            wp = max(bad, key=lambda r: float(r["rel_P"]))
            m.add(f"ExpNine{short}MaxPRatio", f"{1 + float(wp['rel_P']):.3f}",
                  f"{src}\n設定 {name} の, 区分「悪化」「引き換え」の中で P の比 (対同じ β* の Base) の最大値")
            m.add(f"ExpNine{short}MaxPRatioCell", cell_tex(wp["row"], float(wp["r1"])), f"{src}\n上のマス")
    # パート A: β×2 で P_ref を 5% より多く超える組の数
    pa = os.path.join("results", "experiment_9", "experiment_9_A.csv")
    ra = list(csv.DictReader(open(pa, encoding="utf-8")))
    b2 = [r for r in ra if r["sweep"] == "beta_mult" and float(r["value"]) == 2.0]
    over = [float(r["p_ratio"]) for r in b2 if float(r["p_ratio"]) > e9.P_HI]
    m.add("ExpNineBetaTwoOverCount", str(len(over)),
          f"出典: {pa}\n基準点の β を 2 倍にしたとき P の比 (対 P_ref) が 1.05 を超える組の数")
    m.add("ExpNineBetaTwoTotal", str(len(b2)), f"出典: {pa}\n基準点の組の数")
    m.add("ExpNineBetaTwoMaxPRatio", f"{max(over):.3f}", f"出典: {pa}\n上の組の P の比の最大値")
    return out


# ============================================================
# 計算時間と帯幅
# ============================================================

def bench(m: Macros) -> None:
    path = os.path.join("results_gth", "benchmark.md")
    text = open(path, encoding="utf-8").read()
    src = f"出典: {path} (2 コア, numba JIT のウォームアップ後の 3 回中の最良値)"
    b = re.search(r"\| 標準 \(c=20,K=200,b=5\) \| (\d+) \| (\d+) \| (\d+) \| \d+ \| [\d.]+ MB \| ([\d.]+) ms", text)
    m.add("BenchBaseN", f"{int(b.group(1)):,}".replace(",", "{,}"), f"{src}\nベースモデル標準設定の状態数 N")
    m.add("BenchBaseP", b.group(2), f"{src}\nベースモデル標準設定の下帯幅 p")
    m.add("BenchBaseQ", b.group(3), f"{src}\nベースモデル標準設定の上帯幅 q")
    m.add("BenchBaseGTHms", b.group(4), f"{src}\nベースモデル標準設定の GTH の計算時間 [ms]")
    p = re.search(r"\| Predictive 標準 \(c=20,K=200,b=5\) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| "
                  r"[\d.]+ MB \| ([\d.]+) s", text)
    m.add("BenchPredNominalN", f"{int(p.group(1)):,}".replace(",", "{,}"),
          f"{src}\nPredictive 標準設定の名目の状態数")
    m.add("BenchPredReducedN", f"{int(p.group(2)):,}".replace(",", "{,}"),
          f"{src}\nPredictive 標準設定の到達可能状態に縮約した後の状態数")
    m.add("BenchPredP", p.group(3), f"{src}\nPredictive 標準設定の縮約後の下帯幅 p")
    m.add("BenchPredQ", p.group(4), f"{src}\nPredictive 標準設定の縮約後の上帯幅 q")
    m.add("BenchPredGTHs", p.group(5), f"{src}\nPredictive 標準設定の GTH の計算時間 [s]")
    k = re.search(r"\| Predictive K=1000 \(c=20,b=5\) \| (\d+) \| (\d+) \| \d+ \| \d+ \| [\d.]+ MB \| ([\d.]+) s",
                  text)
    m.add("BenchPredKThousandReducedN", f"{int(k.group(2)):,}".replace(",", "{,}"),
          f"{src}\nPredictive K=1000 の縮約後の状態数")
    m.add("BenchPredKThousandGTHs", k.group(3), f"{src}\nPredictive K=1000 の GTH の計算時間 [s]")


# ============================================================
# 表
# ============================================================

def _table(caption: str, label: str, colspec: str, header: str, body: List[str],
           note: str = "", wide: bool = False) -> str:
    """table 環境の文字列. wide=True なら表を版面の幅に縮める (graphicx の \\resizebox)."""
    lines = ["% scripts/make_thesis_numbers.py で生成 (手で編集しない)",
             "\\begin{table}[tbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\small"]
    if wide:
        lines.append("\\resizebox{\\linewidth}{!}{%")
    lines += [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"]
    lines += [b + " \\\\" for b in body]
    lines += ["\\bottomrule", "\\end{tabular}" + ("}" if wide else "")]
    if note:
        lines += ["\\par\\smallskip", f"\\footnotesize {note}"]
    lines += ["\\end{table}", ""]
    return "\n".join(lines)


SER_TEX = {"NoCancel": "NoCancel", "P2 only": "P2 only", "P1 only": "P1 only",
           "P1 only (protect)": "P1 only (prot.)", "P1+P2": "P1+P2", "P1+P2 (protect)": "P1+P2 (prot.)"}


def tables(sm, out9) -> List[str]:
    d = os.path.join(OUT, "tables")
    os.makedirs(d, exist_ok=True)
    files = []

    # 実験 7: ERP 最小値
    conds = list(e7.CONDITIONS)
    body = []
    for s in ["Base"] + br.E7_SERIES:
        sn = SERIES_NAME[s]
        body.append(f"{SER_TEX.get(s, s)} & " + " & ".join(
            f"\\ExpSeven{word_num(c)}{sn}ERPMin" for c in conds))
    t = _table("実験 7: 各系列の ERP 最小値", "tab:exp7-erp", "l" + "r" * len(conds),
               "系列 & " + " & ".join(conds), body,
               "Base は $\\beta$ を細かくした 121 点を含む. 出典: results/experiment\\_7/.")
    files.append(_write(d, "exp7_erp.tex", t))

    # 実験 7: ERP だけの相対差と同じ信頼性での改善
    body = []
    for s in br.E7_SERIES:
        sn = SERIES_NAME[s]
        body.append(f"{SER_TEX[s]} & " + " & ".join(
            f"\\ExpSeven{word_num(c)}{sn}ERPRel & \\ExpSeven{word_num(c)}{sn}SameRelImprovement"
            for c in conds))
    t = _table("実験 7: ERP 最小値の相対差 (対 Base) と, 同じ信頼性 ($P \\le P_{\\mathrm{ref}}$) での改善",
               "tab:exp7-samerel", "l" + "rr" * len(conds),
               "系列 & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{c}}}" for c in conds) +
               " \\\\\n & " + " & ".join("ERP 差 & 改善" for _ in conds), body,
               "ERP 差 = ERP 最小値の相対差 (負が良い). 改善 = $P \\le P_{\\mathrm{ref}}$ の中の ERP 最小値の"
               "$-$(相対差) (正が良い). 出典: results/experiment\\_7/blocking\\_reassessment.md.",
               wide=True)
    files.append(_write(d, "exp7_samerel.tex", t))

    # 実験 8: 地図 A・B の ERP の相対差 (3 系列)
    for which, rows_order in [("A", e8.MAP_A), ("B", e8.MAP_B)]:
        body = []
        for s in E8_SERIES:
            body.append(f"\\multicolumn{{{1 + len(e8.R1_LEVELS)}}}{{l}}{{\\textit{{{s}}}}}")
            for row in rows_order:
                rho_B, alpha = e8.ROWS[row]
                lab = f"$\\rho_B={rho_B:g}$" if which == "A" else f"$\\alpha={alpha:g}$"
                body.append(lab + " & " + " & ".join(
                    f"${100 * sm[(row, r1)]['series'][s]['erp_rel']:+.2f}$" for r1 in e8.R1_LEVELS))
        cap = ("実験 8 地図 A ($\\alpha=0.1$): ERP 最小値の相対差 [\\%] (対 Base)" if which == "A" else
               "実験 8 地図 B ($\\rho_B=0.8$): ERP 最小値の相対差 [\\%] (対 Base)")
        t = _table(cap, f"tab:exp8-map{which.lower()}", "l" + "r" * len(e8.R1_LEVELS),
                   " & " + " & ".join(f"$r_1={r1:.3g}$" for r1 in e8.R1_LEVELS), body,
                   "負が改善. 出典: results/experiment\\_8/experiment\\_8\\_<行>.csv.")
        files.append(_write(d, f"exp8_map{which}.tex", t))

    # 実験 8: 電力の内訳 (mechanism)
    mp = os.path.join("results", "experiment_8", "mechanism_points.csv")
    rows = list(csv.DictReader(open(mp, encoding="utf-8")))
    cb = 0.04419 * 5 + 0.15503
    ci = 0.6 * cb
    body = []
    for r in rows:
        case = r["case"].replace("_", "\\_")
        ser = "Base" if r["series"] == "Base" else "P1 only (prot.)"
        body.append(f"{case} & {ser} & {float(r['beta']):.3g} & {cb * float(r['E_B']):.3f} & "
                    f"{cb * float(r['E_S']):.3f} & {ci * float(r['E_I']):.3f} & {float(r['Cost']):.3f} & "
                    f"{float(r['Cost_F0']):.3f} & {float(r['Cost_F1']):.3f}")
    t = _table("電力の内訳 (ERP 最小点): Base と P1 only (protect)", "tab:exp8-mechanism", "llrrrrrrr",
               "対象 & 系列 & $\\beta^*$ & $C_bE[B]$ & $C_sE[S]$ & $C_iE[I]$ & Cost & Cost$_{F=0}$ & Cost$_{F=1}$",
               body, "出典: results/experiment\\_8/mechanism\\_points.csv (定常分布から位相ごとに分けた値).",
               wide=True)
    files.append(_write(d, "exp8_mechanism.tex", t))

    # 保護の比較: 保護なし / 両方 / NoCancel (α=0.1)
    pp = os.path.join("results", "p1_protect", "p1_protect.csv")
    prow = list(csv.DictReader(open(pp, encoding="utf-8")))

    def pget(burst, rho, beta, gamma, mode):
        return next(r for r in prow if r["burst_name"] == burst and float(r["rho"]) == rho
                    and float(r["alpha"]) == 0.1 and float(r["beta"]) == beta
                    and float(r["gamma"]) == gamma and r["mode"] == mode)
    body = []
    for burst, rho in [("medium", 0.7), ("strong", 0.7), ("medium", 0.3)]:
        for beta in (0.005, 0.5):
            n = pget(burst, rho, beta, 1.0, "none")
            b = pget(burst, rho, beta, 1.0, "both")
            c = pget(burst, rho, beta, 1.0, "nocancel")
            e = float(n["ERP"])
            body.append(f"{burst}, $\\rho={rho}$ & {beta:g} & {e:.4f} & {float(b['ERP']):.4f} & "
                        f"${100 * (float(b['ERP']) / e - 1):+.2f}$ & {float(c['ERP']):.4f} & "
                        f"${100 * (float(c['ERP']) / e - 1):+.2f}$ & {float(n['setup_cancel_rate']):.3f} & "
                        f"{float(b['setup_cancel_rate']):.3f}")
    t = _table("保護と NoCancel の比較 ($\\alpha=0.1$, $\\gamma=1$, $n_{\\mathrm{target}}=10$)",
               "tab:protect-control", "lrrrrrrrr",
               "条件 & $\\beta$ & ERP (none) & ERP (both) & 差 [\\%] & ERP (NoCancel) & 差 [\\%] & "
               "取消率 (none) & 取消率 (both)", body,
               "none = 保護なし, both = protect\\_presetup と protect\\_delayoff, NoCancel = never\\_cancel\\_setup "
               "($n_{\\mathrm{target}}=0$). 差は none に対する ERP の相対差. "
               "出典: results/p1\\_protect/p1\\_protect.csv.", wide=True)
    files.append(_write(d, "protect_control.tex", t))

    # 実験 9 パート B: 5 区分のマス数
    body = []
    for name, short in B_NAMES.items():
        body.append(name.replace("γ", "$\\gamma$").replace("n_target", "$n_{\\mathrm{target}}$") + " & " +
                    " & ".join(f"\\ExpNine{short}{cn}Count" for cn in CAT_NAMES.values()))
    t = _table("実験 9 パート B: Base の $\\beta^*$ のまま規則を ON にした場合の区分 (35 マス)",
               "tab:exp9-partb", "lrrrrr", "設定 & " + " & ".join(CAT_NAMES), body,
               "$|\\Delta\\mathrm{ERP}| \\le 0.1\\%$, $P$ の比 0.95〜1.05 を同等とする. "
               "出典: results/experiment\\_9/experiment\\_9\\_B.csv.")
    files.append(_write(d, "exp9_partB.tex", t))

    # 実験 9 パート A: β×2 での P の比
    pa = os.path.join("results", "experiment_9", "experiment_9_A.csv")
    ra = list(csv.DictReader(open(pa, encoding="utf-8")))
    body = []
    for row, r1 in e9.A_CELLS:
        for s in e9.A_SERIES:
            pts = {float(r["value"]): r for r in ra if r["row"] == row and r["series"] == s
                   and r["sweep"] == "beta_mult" and abs(float(r["r1"]) / r1 - 1) < 1e-9}
            if 1.0 not in pts:
                continue
            body.append(f"{cell_tex(row, r1)} & {SER_TEX[s]} & {float(pts[1.0]['p_ratio']):.3f} & "
                        f"{float(pts[2.0]['p_ratio']):.3f} & ${100 * float(pts[2.0]['erp_rel']):+.2f}$")
    t = _table("実験 9 パート A: 基準点の $\\beta$ を 2 倍にしたときの $P/P_{\\mathrm{ref}}$",
               "tab:exp9-beta2", "llrrr",
               "マス & 系列 & $P/P_{\\mathrm{ref}}$ ($\\beta\\times1$) & $P/P_{\\mathrm{ref}}$ ($\\beta\\times2$) & "
               "ERP 差 [\\%] ($\\beta\\times2$)", body,
               "基準点は (b) の最良点. 出典: results/experiment\\_9/experiment\\_9\\_A.csv.")
    files.append(_write(d, "exp9_beta2.tex", t))
    return files


def _write(d, name, text) -> str:
    path = os.path.join(d, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# ============================================================
# 図と README
# ============================================================

# 論文の節 (論理の流れ): (節, 問い, 使う実験, 結論, 図, 表, 主なマクロ)
SECTIONS = [
    (1, "解析は正しいか", "実験 0 / 0-P (と保護ありの DES 照合)", "理論と DES が一致",
     ["experiment_0_validation", "experiment_0P_validation"], [],
     "ExpZero*, Bench*"),
    (2, "ベースモデルの特性", "実験 1〜4",
     "バースト性と有限バッファの影響. ρ_B > 1 ではどの方策でも棄却を防げない",
     ["experiment_1_traffic", "experiment_2_delayoff", "experiment_2_delayoff_strong",
      "experiment_3_burstiness_delta", "experiment_3_burstiness_sigma",
      "experiment_4_K_sensitivity_weak", "experiment_4_K_sensitivity_medium",
      "experiment_4_K_sensitivity_strong"], [],
     "ExpSevenCTwoBasePBlock, ExpSevenCTwoEpsTwoFeasibleCount, ExpEightBaseInfeasibleEpsTwoCount"),
    (3, "評価の基準", "―",
     "有限バッファでは ERP が棄却を含まない. 同じ信頼性での ERP を基準にする", [],
     ["exp7_samerel.tex"], "ExpSeven*PRatio, ExpEight*PRatioMax, ExpEightCategoryChangedCount"),
    (4, "各規則の働き", "保護の比較, NoCancel",
     "保護が不可欠. 価値は位相で時期を選ぶことから来る", [], ["protect_control.tex"],
     "ProtectExample*, ProtectNoCancel*, ExpSeven*NoCancel*, ExpSevenNoCancelWorse*"),
    (5, "公平な比較", "実験 7",
     "典型的な条件では差は小さい. β を固定した比較は過大評価だった",
     [f"experiment_7_{c}_EW_cost" for c in e7.CONDITIONS] +
     [f"experiment_7_{c}_cost_pblock" for c in e7.CONDITIONS] +
     ["compare_experiment_1_vs_1P", "compare_experiment_3_vs_3P_delta"],
     ["exp7_erp.tex", "exp7_samerel.tex"],
     "ExpSeven*ERPMin, ExpSeven*ERPRel, ExpSevenMax*, ExpOldBetaFixed*"),
    (6, "どこで優位か", "実験 8 + 再判定",
     "中心の領域で 2〜4%. 途中の領域では棄却との引き換え",
     ["experiment_8_A_erp", "experiment_8_A_cost", "experiment_8_B_erp", "experiment_8_B_cost",
      "experiment_8_B_p1_alpha", "experiment_8_A_blk_b", "experiment_8_B_blk_b",
      "experiment_8_A_blk_c1e-3", "experiment_8_B_blk_c1e-3"],
     ["exp8_mapA.tex", "exp8_mapB.tex"],
     "ExpEightCenter*, ExpEightCore*, ExpEightMaxImprovement*, ExpEight*SameRel*, ExpEightAlphaSpearman*"),
    (7, "なぜ優位か", "仕組みの集計", "通常期にサーバーを早く切れる",
     ["experiment_8_p1_fire"], ["exp8_mechanism.tex"],
     "ExpEightR*, ExpEightFire*, Mech*"),
    (8, "どこでベースモデルが優位か", "実験 9",
     "価値のない領域, P2 を β の調整なしで使う場合, n_target が大きすぎる場合. "
     "P1 は追加しても棄却を増やさない",
     [f"experiment_9_A_{row}_r1_{r1:.3g}" for row, r1 in e9.A_CELLS] +
     [f"experiment_9_B_{w}_{m}" for w in ("A", "B") for m in ("ERP", "EW", "Cost", "P")],
     ["exp9_partB.tex", "exp9_beta2.tex"], "ExpNine*"),
    (9, "結論と限界", "―",
     "位相の観測の仮定, 反応的な起動の理想化, 指数分布のセットアップ時間", [], [], "—"),
]


def figures_and_readme(macros: Macros, table_files: List[str]) -> List[str]:
    d = os.path.join(OUT, "figures")
    os.makedirs(d, exist_ok=True)
    copied, missing = [], []
    for _, _, _, _, figs, _, _ in SECTIONS:
        for stem in figs:
            for ext in ("pdf", "png"):
                src = os.path.join("figures", f"{stem}.{ext}")
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(d, f"{stem}.{ext}"))
                    copied.append(os.path.join(d, f"{stem}.{ext}"))
                else:
                    missing.append(src)
    if missing:
        raise FileNotFoundError(f"図が見つからない: {missing}")
    lines = ["# 論文用の数値・表・図 (thesis_assets)", "",
             "`python scripts/make_thesis_numbers.py` で results/ の CSV とレポートの集計から生成する "
             "(手で編集しない). 新しい計算はしない.", "",
             "- `numbers.tex`: 数値マクロ (`\\input{thesis_assets/numbers}` で読み込む). "
             f"{len(macros.items)} 個. 各マクロの直前に出典の CSV と計算方法をコメントで書いてある.",
             "- `tables/*.tex`: table 環境の表 (booktabs と graphicx (幅の広い表の \\resizebox) を使う. "
             "数値マクロを参照する表は numbers.tex の後に読み込む).",
             "- `figures/`: figures/ からのコピー (PDF と PNG).", "",
             "## 論文の節と図・表の対応", "",
             "論文の論理の流れ (節 1〜9) に合わせた. 図は figures/ の PDF (同名の PNG もある).", "",
             "| 節 | 問い | 使う実験 | 結論 | 図 | 表 | 主なマクロ |", "|---|---|---|---|---|---|---|"]
    for no, q, exp, concl, figs, tabs, mac in SECTIONS:
        lines.append(f"| {no} | {q} | {exp} | {concl} | " +
                     ("<br>".join(f"figures/{f}.pdf" for f in figs) or "—") + " | " +
                     ("<br>".join(f"tables/{t}" for t in tabs) or "—") + f" | {mac} |")
    lines += ["", "## 図の出典と注意", "",
              "- experiment_0*: scripts/experiment_0_validation.py, experiment_0P_validation.py. "
              "0-P は保護を入れる前のモデル. 保護ありのモデルの DES 照合は results/protect_des/ "
              "(図はなく, マクロ ExpZeroProtect* で引用する).",
              "- experiment_1〜4_*: ベースモデルだけの実験 (保護の有無に関係しない).",
              "- compare_experiment_*: β を固定した旧比較 (保護なしの旧モデル). 節 5 で "
              "「β を固定した比較は過大評価だった」ことを示すために使う.",
              "- experiment_7_*: scripts/experiment_7_frontier.py --report",
              "- experiment_8_{A,B}_{erp,cost}, B_p1_alpha, p1_fire: scripts/experiment_8_map.py --report",
              "- experiment_8_*_blk_*: scripts/blocking_reassessment.py",
              "- experiment_9_*: scripts/experiment_9_sensitivity.py --report", ""]
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return copied


def main():
    os.makedirs(OUT, exist_ok=True)
    m = Macros()
    exp0(m)
    exp7(m)
    old_comparison(m)
    protect_comparison(m)
    sm = exp8(m)
    out9 = exp9(m)
    bench(m)
    m.write(os.path.join(OUT, "numbers.tex"))
    tfiles = tables(sm, out9)
    figs = figures_and_readme(m, tfiles)
    print(f"マクロ {len(m.items)} 個, 表 {len(tfiles)} 個, 図 {len(figs)} ファイル")


if __name__ == "__main__":
    main()
