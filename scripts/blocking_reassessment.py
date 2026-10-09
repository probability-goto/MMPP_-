#!/usr/bin/env python
"""実験 7・8 を, ブロッキング確率を含めた基準で判定し直す (新しい計算はしない).

ERP = E[W] x Cost は棄却されたジョブを評価に含めないので, 既存の CSV から次を求める.
    (a) 各系列の ERP 最小点の P_block_arrival_stable と, Base の ERP 最小点に対する比
    (b) P_ref = Base の ERP 最小点の P_block_arrival_stable とし, P <= P_ref を満たす点の中の
        ERP 最小値と, Base の ERP 最小値に対する相対差 (満たす点がなければ「該当なし」)
    (c) eps in {1e-2, 1e-3, 1e-4} で P <= eps を満たす点の中の ERP 最小値 (Base と各系列)
    (d) (E[W], Cost, P) の 3 指標で, Base の 3 次元フロンティアの点のうち各系列に支配される点の数
    (e) 1 件あたりの電力 x 応答時間 = (Cost / lambda_eff) x E[W] の最小値
対象: 実験 7 (5 条件, NoCancel を含む全系列, 範囲を広げた点を含む; Base は β を細かくした点を含む)
と実験 8 (35 マス, Base と 3 系列; 細かい探索・範囲の拡張の点を含む).

出力: results/experiment_7/blocking_reassessment.md, results/experiment_8/blocking_reassessment.md,
figures/experiment_8_{A,B}_blk_{b,c1e-3}.{png,pdf}.

使用例:
    python scripts/blocking_reassessment.py
"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import experiment_7_frontier as e7
    import experiment_8_map as e8
except ImportError:
    import scripts.experiment_7_frontier as e7
    import scripts.experiment_8_map as e8

EPS = [1e-2, 1e-3, 1e-4]
RTOL = 1e-9
# ERP だけの判定と比べるときの区分 (改善 > 0.1%, |差| <= 0.1%, 悪化 < -0.1%)
TIE = 0.001
FIG_DIR = "figures"
P = "P_block_arrival_stable"


# ============================================================
# データの読み込み (系列名 -> 点のリスト)
# ============================================================

def _num(r):
    out = dict(r)
    for k in ("E_W", "Cost", P, "ERP", "lambda_eff", "beta", "gamma"):
        out[k] = float(r[k])
    out["n_target"] = int(r["n_target"])
    return out


def _dedupe(rows: List[dict]) -> List[dict]:
    """同じ (beta, n_target, gamma) の点を 1 つにする (β は相対 1e-9 で同じとみなす)."""
    seen = {}
    for r in rows:
        k = (round(np.log10(r["beta"]), 8), r["n_target"], r["gamma"])
        seen.setdefault(k, r)
    return list(seen.values())


def load_exp7() -> Dict[str, Dict[str, List[dict]]]:
    data = {}
    for cond in e7.CONDITIONS:
        grid = {e7.point_key(p) for p in e7.condition_points(cond)}
        groups: Dict[str, List[dict]] = {}
        for r in e7.read_rows(e7.csv_path(cond)):
            if r["series"] == "Check":
                continue
            x = _num(r)
            x["where"] = "13 点の格子" if e7.row_key(r) in grid else "範囲を広げた点"
            groups.setdefault(r["series"], []).append(x)
        for r in e7.read_rows(e7.refine_csv_path(cond)):
            x = _num(r)
            s = {"Base (dense)": "Base", "NoCancel (refine)": "NoCancel"}[r["series"]]
            x["where"] = ("β を細かくした点 (121 点)" if s == "Base"
                          else "ERP 最小点の近くを細かくした点")
            groups.setdefault(s, []).append(x)
        data[cond] = {s: _dedupe(v) for s, v in groups.items()}
    return data


STAGE_JA = {"coarse": "粗い格子", "fine": "細かい探索", "extend": "範囲を広げた点",
            "fine2": "広げた範囲の細かい探索"}


def load_exp8() -> Dict[Tuple[str, float], Dict[str, List[dict]]]:
    data = {}
    for row in dict.fromkeys(e8.MAP_A + e8.MAP_B):
        for r in e8.read_rows(e8.csv_path(row)):
            if r["series"] == "Check":
                continue
            x = _num(r)
            x["where"] = STAGE_JA.get(r["stage"], r["stage"])
            x["sigma"] = float(r["sigma"])
            data.setdefault((row, float(r["r1"])), {}).setdefault(r["series"], []).append(x)
    return {k: {s: _dedupe(v) for s, v in g.items()} for k, g in data.items()}


# ============================================================
# 判定
# ============================================================

def _edge_note(r: dict, rows: List[dict]) -> str:
    """点 r の β が, 同じ (n_target, gamma) の点の β の範囲の端かどうか."""
    bs = sorted({x["beta"] for x in rows
                 if x["n_target"] == r["n_target"] and x["gamma"] == r["gamma"]})
    if len(bs) < 2:
        return ""
    if r["beta"] <= bs[0] * (1 + RTOL):
        return "β が下端"
    if r["beta"] >= bs[-1] * (1 - RTOL):
        return "β が上端"
    return ""


def _min_erp(rows: List[dict], cap: Optional[float]) -> Optional[dict]:
    ok = rows if cap is None else [r for r in rows if r[P] <= cap * (1 + RTOL)]
    return min(ok, key=lambda r: r["ERP"]) if ok else None


def _dom3(a, b) -> bool:
    le = all(x <= y * (1 + RTOL) for x, y in zip(a, b))
    lt = any(x < y * (1 - RTOL) for x, y in zip(a, b))
    return le and lt


def front3(rows: List[dict]) -> List[dict]:
    pts = [(r["E_W"], r["Cost"], r[P]) for r in rows]
    return [r for r, p in zip(rows, pts) if not any(_dom3(q, p) for q in pts)]


def assess(groups: Dict[str, List[dict]], series: List[str]) -> dict:
    base = groups["Base"]
    b_erp = _min_erp(base, None)
    p_ref = b_erp[P]
    fb3 = front3(base)
    b_epj = min(r["Cost"] / r["lambda_eff"] * r["E_W"] for r in base)
    out = {"base_best": b_erp, "p_ref": p_ref, "n_front3": len(fb3), "series": {},
           "base_eps": {eps: _min_erp(base, eps) for eps in EPS}}
    for s in ["Base"] + series:
        rows = groups[s]
        best = _min_erp(rows, None)
        bb = _min_erp(rows, p_ref)
        res = {
            "best": best, "erp_rel": best["ERP"] / b_erp["ERP"] - 1,
            "p_ratio": best[P] / p_ref,
            "b": bb, "b_rel": None if bb is None else bb["ERP"] / b_erp["ERP"] - 1,
            "b_note": "" if bb is None else ", ".join(
                x for x in (bb["where"], _edge_note(bb, rows)) if x),
            "c": {}, "dom3": None, "epj": None,
        }
        for eps in EPS:
            cb = out["base_eps"][eps]
            cs = _min_erp(rows, eps)
            rel = None
            if cs is not None and cb is not None:
                rel = cs["ERP"] / cb["ERP"] - 1
            res["c"][eps] = {"pt": cs, "rel": rel,
                             "note": "" if cs is None else ", ".join(
                                 x for x in (cs["where"], _edge_note(cs, rows)) if x)}
        if s != "Base":
            pts = [(r["E_W"], r["Cost"], r[P]) for r in rows]
            k = sum(1 for f in fb3 if any(_dom3(q, (f["E_W"], f["Cost"], f[P])) for q in pts))
            res["dom3"] = (k, len(fb3))
        epj = min(r["Cost"] / r["lambda_eff"] * r["E_W"] for r in rows)
        res["epj"] = epj
        res["epj_rel"] = epj / b_epj - 1
        out["series"][s] = res
    return out


def _cat(v: Optional[float]) -> str:
    if v is None:
        return "該当なし"
    imp = -v
    if imp > TIE:
        return "改善"
    if imp < -TIE:
        return "悪化"
    return "同等"


def _pct(v: Optional[float]) -> str:
    return "該当なし" if v is None else f"{100 * v:+.2f}%"


def _fmt_pt(r: Optional[dict]) -> str:
    if r is None:
        return "—"
    s = f"β={r['beta']:.3g}"
    if r["series"] != "Base" and r["series"] != "Base (dense)":
        s += f", n_target={r['n_target']}, γ={r['gamma']:g}"
    return s


# ============================================================
# 実験 7 のレポート
# ============================================================

E7_SERIES = ["NoCancel", "P2 only", "P1 only", "P1 only (protect)", "P1+P2",
             "P1+P2 (protect)"]


def report_exp7() -> Tuple[str, list]:
    data = load_exp7()
    lines = ["# 実験 7 をブロッキング確率を含めた基準で判定し直す", "",
             "results/experiment_7/ の CSV から scripts/blocking_reassessment.py で生成 (新しい計算はしない). "
             "Base は 13 点と β を細かくした 121 点 (C4 は下限まで広げた点を含む) を合わせ, "
             "NoCancel は ERP 最小点の近くを細かくした点を含む. 同じ (β, n_target, γ) の点は 1 つにまとめた. "
             "P は P_block_arrival_stable. 相対差は Base の ERP 最小値 (細かくした Base) に対するもの "
             "((c) は同じ ε を満たす Base の ERP 最小値に対するもの). 改善 = −(相対差).", "",
             "Predictive の系列は β を 13 点 (と範囲を広げた点) でしか計算していないので, (b)(c) の最良点が "
             "13 点の格子の点になるのは当然で, 格子の間に良い点がありうる (Base は 121 点).", ""]
    changes = []
    for cond, groups in data.items():
        res = assess(groups, E7_SERIES)
        rho, delta, sigma, alpha, desc = e7.CONDITIONS[cond]
        bb = res["base_best"]
        lines += [f"## {cond} {desc} (ρ={rho}, σ={sigma}, α={alpha})", "",
                  f"Base の ERP 最小点: {_fmt_pt(bb)}, ERP={bb['ERP']:.4f}, P_ref={res['p_ref']:.4e}. "
                  f"Base の 3 次元フロンティアの点数 {res['n_front3']}.", "",
                  "| 系列 | ERP だけ (相対差) | (a) P | (a) P の比 | (b) ERP | (b) 相対差 | (b) の点 | "
                  "(c) 1e-2 | (c) 1e-3 | (c) 1e-4 | (d) 支配 | (e) 相対差 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in ["Base"] + E7_SERIES:
            v = res["series"][s]
            b = v["b"]
            cells = [f"{_pct(v['c'][eps]['rel'])}" if res["base_eps"][eps] is not None
                     else ("Base 不可" if v["c"][eps]["pt"] is None else
                           f"Base 不可 (系列は {v['c'][eps]['pt']['ERP']:.4f})")
                     for eps in EPS]
            dom = "—" if v["dom3"] is None else f"{v['dom3'][0]}/{v['dom3'][1]}"
            b_erp = "—" if b is None else f"{b['ERP']:.4f}"
            lines.append(
                f"| {s} | {_pct(v['erp_rel'])} | {v['best'][P]:.3e} | {v['p_ratio']:.2f} | "
                f"{b_erp} | {_pct(v['b_rel'])} | "
                f"{_fmt_pt(b)}{' (' + v['b_note'] + ')' if b is not None else ''} | "
                + " | ".join(cells) + f" | {dom} | {_pct(v['epj_rel'])} |")
            if s != "Base" and _cat(v["erp_rel"]) != _cat(v["b_rel"]):
                changes.append((cond, s, v["erp_rel"], v["b_rel"]))
        lines += ["", "(c) の ERP 最小点 (Base と各系列):", "",
                  "| 系列 | " + " | ".join(f"ε={eps:g}" for eps in EPS) + " |",
                  "|---|" + "---|" * len(EPS)]
        for s in ["Base"] + E7_SERIES:
            v = res["series"][s]
            cells = []
            for eps in EPS:
                pt = v["c"][eps]["pt"]
                cells.append("満たす点なし" if pt is None else
                             f"ERP={pt['ERP']:.4f} ({_fmt_pt(pt)}; {v['c'][eps]['note']})")
            lines.append(f"| {s} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## ERP だけで判定した場合と区分が変わった系列", "",
              f"区分: 改善 (改善 > {100 * TIE:g}%), 同等 (|差| ≤ {100 * TIE:g}%), 悪化, 該当なし. "
              "ERP だけ = 各系列の ERP 最小値の相対差, (b) = P ≤ P_ref の中の ERP 最小値の相対差.", "",
              "| 条件 | 系列 | ERP だけ | (b) | 区分 |", "|---|---|---|---|---|"]
    for cond, s, a, b in changes:
        lines.append(f"| {cond} | {s} | {_pct(a)} | {_pct(b)} | {_cat(a)} → {_cat(b)} |")
    if not changes:
        lines.append("| — | — | — | — | なし |")
    lines.append("")
    return "\n".join(lines), changes


# ============================================================
# 実験 8 のレポートと図
# ============================================================

E8_SERIES = list(e8.SERIES)


def _matrix_md(title, res, rows_order, which, fn) -> List[str]:
    lines = [title, "", "| | " + " | ".join(f"r1={r1:.3g}" for r1 in e8.R1_LEVELS) + " |",
             "|---|" + "---|" * len(e8.R1_LEVELS)]
    for row in rows_order:
        cells = [fn(res[(row, r1)]) for r1 in e8.R1_LEVELS]
        lines.append(f"| {e8._row_label(row, which)} | " + " | ".join(cells) + " |")
    return lines + [""]


def plot_blk(res, rows_order, which, stem, fn, label) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mats = {}
    for s in E8_SERIES:
        M = np.full((len(rows_order), len(e8.R1_LEVELS)), np.nan)
        for i, row in enumerate(rows_order):
            for j, r1 in enumerate(e8.R1_LEVELS):
                v = fn(res[(row, r1)], s)
                M[i, j] = np.nan if v is None else 100 * v
        mats[s] = M
    finite = [np.nanmax(np.abs(M)) for M in mats.values() if np.isfinite(M).any()]
    vmax = max(finite) if finite and max(finite) > 0 else 1.0
    ylabels = [e8._row_label(r, which).replace("ρ_B", r"$\rho_B$").replace("α", r"$\alpha$")
               for r in rows_order]
    fig, axes = plt.subplots(1, len(E8_SERIES), figsize=(5.2 * len(E8_SERIES),
                             0.9 * len(rows_order) + 2.2), sharey=True)
    for ax, s in zip(axes, E8_SERIES):
        M = mats[s]
        im = ax.imshow(M, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto", origin="lower")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                txt = "n/a" if not np.isfinite(M[i, j]) else f"{M[i, j]:+.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                        color="white" if np.isfinite(M[i, j]) and abs(M[i, j]) > 0.6 * vmax
                        else "black")
        ax.set_xticks(range(len(e8.R1_LEVELS)))
        ax.set_xticklabels([f"{v:.2f}" for v in np.log10(e8.R1_LEVELS)])
        ax.set_xlabel(r"$\log_{10} r_1$", fontsize=14)
        ax.set_title(s, fontsize=13)
    axes[0].set_yticks(range(len(rows_order)))
    axes[0].set_yticklabels(ylabels, fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    cax = fig.add_axes([0.93, 0.2, 0.012, 0.6])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=11)
    png = os.path.join(FIG_DIR, f"experiment_8_{which}_blk_{stem}.png")
    fig.savefig(png, dpi=120, bbox_inches="tight")
    fig.savefig(os.path.splitext(png)[0] + ".pdf", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [png, os.path.splitext(png)[0] + ".pdf"]


def report_exp8() -> Tuple[str, list, list]:
    data = load_exp8()
    res = {k: assess(g, E8_SERIES) for k, g in data.items()}
    lines = ["# 実験 8 をブロッキング確率を含めた基準で判定し直す", "",
             "results/experiment_8/ の CSV から scripts/blocking_reassessment.py で生成 (新しい計算はしない). "
             "Base と各系列の全点 (粗い格子, 細かい探索, 範囲を広げた点) を使う. P は P_block_arrival_stable. "
             "相対差は Base の ERP 最小値に対するもの ((c) は同じ ε を満たす Base の ERP 最小値に対するもの). "
             "改善 = −(相対差). β の格子は ERP 最小点の周りだけ細かいので, (b)(c) の最良点が粗い格子の点に"
             "なる場合は, その位置を下の「(b)(c) の最良点の位置」に示す.", ""]
    figs = []
    for which, rows_order in [("A", e8.MAP_A), ("B", e8.MAP_B)]:
        title = "地図 A (α=0.1, 行は ρ_B)" if which == "A" else "地図 B (ρ_B=0.8, 行は α)"
        lines += [f"## {title}", ""]
        lines += _matrix_md("### Base の ERP 最小点の P (= P_ref)", res, rows_order, which,
                            lambda c: f"{c['p_ref']:.2e}")
        for s in E8_SERIES:
            lines += [f"### {s}", ""]
            lines += _matrix_md(f"ERP だけ (ERP 最小値の相対差)", res, rows_order, which,
                                lambda c, s=s: _pct(c["series"][s]["erp_rel"]))
            lines += _matrix_md(f"(a) ERP 最小点の P の比 (対 Base)", res, rows_order, which,
                                lambda c, s=s: f"{c['series'][s]['p_ratio']:.2f}")
            lines += _matrix_md(f"(b) P ≤ P_ref の中の ERP 最小値の相対差", res, rows_order, which,
                                lambda c, s=s: _pct(c["series"][s]["b_rel"]))
            for eps in EPS:
                lines += _matrix_md(
                    f"(c) P ≤ {eps:g} の中の ERP 最小値の相対差", res, rows_order, which,
                    lambda c, s=s, eps=eps: ("Base 不可" if c["base_eps"][eps] is None else
                                             _pct(c["series"][s]["c"][eps]["rel"])))
            lines += _matrix_md(f"(d) Base の 3 次元フロンティアのうち支配される点", res, rows_order,
                                which, lambda c, s=s: "{}/{}".format(*c["series"][s]["dom3"]))
            lines += _matrix_md(f"(e) (Cost/λ_eff)×E[W] の最小値の相対差", res, rows_order, which,
                                lambda c, s=s: _pct(c["series"][s]["epj_rel"]))
        figs += plot_blk(res, rows_order, which, "b", lambda c, s: (
            None if c["series"][s]["b_rel"] is None else -c["series"][s]["b_rel"]),
            r"improvement at $P \leq P_{\mathrm{ref}}$ [%]")
        figs += plot_blk(res, rows_order, which, "c1e-3", lambda c, s: (
            None if c["base_eps"][1e-3] is None or c["series"][s]["c"][1e-3]["rel"] is None
            else -c["series"][s]["c"][1e-3]["rel"]),
            r"improvement at $P \leq 10^{-3}$ [%]")

    # (b)(c) の最良点の位置
    lines += ["## (b)(c) の最良点の位置", "",
              "(b) と ε=1e-3 の (c) の最良点が, 粗い格子の点 (ERP 最小点の周りの細かい探索の外) または "
              "β の範囲の端にある組の一覧. この組は, β を細かくすると ERP がさらに下がりうる "
              "(Base 側も同様).", "",
              "| マス | 系列 | 基準 | 点 | 位置 |", "|---|---|---|---|---|"]
    n_coarse = 0
    for (row, r1), c in sorted(res.items()):
        for s in ["Base"] + E8_SERIES:
            v = c["series"][s]
            for name, pt, note in [("(b)", v["b"], v["b_note"]),
                                   ("(c) 1e-3", v["c"][1e-3]["pt"], v["c"][1e-3]["note"])]:
                if pt is None:
                    continue
                if "粗い格子" in note or "端" in note:
                    n_coarse += 1
                    lines.append(f"| {row} r1={r1:.3g} | {s} | {name} | {_fmt_pt(pt)} | {note} |")
    lines += ["", f"該当 {n_coarse} 組.", ""]

    # 区分の変化
    changes, better_b = [], []
    for (row, r1), c in sorted(res.items()):
        for s in E8_SERIES:
            v = c["series"][s]
            if _cat(v["erp_rel"]) != _cat(v["b_rel"]):
                changes.append((row, r1, s, v["erp_rel"], v["b_rel"]))
            if v["b_rel"] is not None and -v["b_rel"] > TIE:
                better_b.append((row, r1, s, -v["b_rel"], v["b_note"]))
    lines += ["## (b) で Base より良い (改善 > 0.1%) マス", "",
              "| マス | 系列 | (b) の改善 | (b) の点の位置 |", "|---|---|---|---|"]
    for row, r1, s, imp, note in sorted(better_b, key=lambda t: -t[3]):
        lines.append(f"| {row} r1={r1:.3g} | {s} | {100 * imp:+.2f}% | {note} |")
    lines += ["", "## ERP だけで判定した場合と区分が変わったマス", "",
              f"区分: 改善 (改善 > {100 * TIE:g}%), 同等 (|差| ≤ {100 * TIE:g}%), 悪化, 該当なし.", "",
              "| マス | 系列 | ERP だけ | (b) | 区分 |", "|---|---|---|---|---|"]
    for row, r1, s, a, b in changes:
        lines.append(f"| {row} r1={r1:.3g} | {s} | {_pct(a)} | {_pct(b)} | {_cat(a)} → {_cat(b)} |")
    lines += ["", "## 図", ""] + [f"- {f}" for f in figs] + [""]
    return "\n".join(lines), changes, better_b, res, n_coarse


def main():
    t7, ch7 = report_exp7()
    concl7 = os.path.join(e7.OUT_DIR, "blocking_conclusion.md")
    if os.path.exists(concl7):
        t7 += "\n" + open(concl7, encoding="utf-8").read().rstrip() + "\n"
    with open(os.path.join(e7.OUT_DIR, "blocking_reassessment.md"), "w", encoding="utf-8") as f:
        f.write(t7)
    t8, ch8, better, _, n_coarse = report_exp8()
    concl8 = os.path.join(e8.OUT_DIR, "blocking_conclusion.md")
    if os.path.exists(concl8):
        t8 += "\n" + open(concl8, encoding="utf-8").read().rstrip() + "\n"
    with open(os.path.join(e8.OUT_DIR, "blocking_reassessment.md"), "w", encoding="utf-8") as f:
        f.write(t8)
    print(f"実験 7: 区分が変わった系列 {len(ch7)}. 実験 8: 区分が変わったマス×系列 {len(ch8)}, "
          f"(b) で改善 {len(better)}, (b)(c) の最良点が粗い格子/端 {n_coarse} 組.")


if __name__ == "__main__":
    main()
