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
except ImportError:
    import scripts.experiment_7_frontier as e7
    import scripts.experiment_8_map as e8
    import scripts.experiment_9_sensitivity as e9
    import scripts.blocking_reassessment as br

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
            m.add(f"ExpSeven{cn}{sn}SameRelImprovement",
                  "該当なし" if v["b_rel"] is None else spct(-v["b_rel"]),
                  f"{src}\n条件 {cond}, 系列 {s} の同じ信頼性での改善: P ≤ P_ref (Base の ERP 最小点の P) を"
                  "\n満たす点の中の ERP 最小値の, Base の ERP 最小値に対する −(相対差)")
    return res


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
    return sm


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

# (仮の節番号, 節の内容, 図のファイル名 (拡張子なし) の一覧, 表, 主なマクロの接頭辞)
SECTIONS = [
    (1, "研究の動機と問題設定", [], [], ""),
    (2, "モデル (Base と Predictive, 保護)", [], [], ""),
    (3, "数値解法と検証 (GTH, DES との照合, 計算時間)",
     ["experiment_0_validation", "experiment_0P_validation"], [], "ExpZero, Bench"),
    (4, "公平な比較: 各方策の最良の設定どうし (実験 7)",
     [f"experiment_7_{c}_EW_cost" for c in e7.CONDITIONS] +
     [f"experiment_7_{c}_cost_pblock" for c in e7.CONDITIONS],
     ["exp7_erp.tex", "exp7_samerel.tex"], "ExpSeven"),
    (5, "位相の情報に価値がある領域の地図 (実験 8)",
     ["experiment_8_A_erp", "experiment_8_A_cost", "experiment_8_B_erp", "experiment_8_B_cost",
      "experiment_8_B_p1_alpha", "experiment_8_p1_fire"],
     ["exp8_mapA.tex", "exp8_mapB.tex"], "ExpEight"),
    (6, "Predictive が優位になる仕組み (電力と位相ごとの内訳)", [], ["exp8_mechanism.tex"],
     "ExpEightR"),
    (7, "ブロッキング確率を含めた判定 (同じ信頼性での比較)",
     ["experiment_8_A_blk_b", "experiment_8_B_blk_b", "experiment_8_A_blk_c1e-3",
      "experiment_8_B_blk_c1e-3"], [], "ExpSeven...SameRel, ExpEight...SameRel"),
    (8, "パラメータが最適でないときの感度 (実験 9 パート A)",
     [f"experiment_9_A_{row}_r1_{r1:.3g}" for row, r1 in e9.A_CELLS], ["exp9_beta2.tex"],
     "ExpNineBetaTwo"),
    (9, "Base の β のまま規則を ON にした場合 (実験 9 パート B) とまとめ",
     [f"experiment_9_B_{w}_{m}" for w in ("A", "B") for m in ("ERP", "EW", "Cost", "P")],
     ["exp9_partB.tex"], "ExpNine"),
]


def figures_and_readme(macros: Macros, table_files: List[str]) -> List[str]:
    d = os.path.join(OUT, "figures")
    os.makedirs(d, exist_ok=True)
    copied, missing = [], []
    for _, _, figs, _, _ in SECTIONS:
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
             "**注意: 節の番号と内容は仮である.** 依頼文の「論理の流れ」の節 1〜9 がリポジトリに見つからなかったため, "
             "実験の順に仮に割り当てた. 節の対応が決まったら scripts/make_thesis_numbers.py の SECTIONS を直して "
             "再生成する.", "",
             "| 節 (仮) | 内容 | 図 | 表 | 主なマクロ |", "|---|---|---|---|---|"]
    for no, title, figs, tabs, mac in SECTIONS:
        lines.append(f"| {no} | {title} | " + ("<br>".join(f"figures/{f}.pdf" for f in figs) or "—") +
                     " | " + ("<br>".join(f"tables/{t}" for t in tabs) or "—") + f" | {mac or '—'} |")
    lines += ["", "## 図の出典", "",
              "- experiment_0*: scripts/experiment_0_validation.py, experiment_0P_validation.py "
              "(保護を入れる前のモデル).",
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
    sm = exp8(m)
    out9 = exp9(m)
    bench(m)
    m.write(os.path.join(OUT, "numbers.tex"))
    tfiles = tables(sm, out9)
    figs = figures_and_readme(m, tfiles)
    print(f"マクロ {len(m.items)} 個, 表 {len(tfiles)} 個, 図 {len(figs)} ファイル")


if __name__ == "__main__":
    main()
