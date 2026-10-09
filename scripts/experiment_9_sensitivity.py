#!/usr/bin/env python
"""実験 9: Predictive のパラメータが最適でないとき, ベースモデルより悪くなる範囲を調べる.

両方を最適に調整すると Predictive はベースモデルを特別な場合として含むので悪くならない.
ベースモデルが優位になるのは Predictive のパラメータが最適でないときなので, その範囲を調べる.
比較の基準は「同じ信頼性での ERP」: P_ref = Base の ERP 最小点の P_block_arrival_stable とし,
ERP の相対差 (対 Base の ERP 最小値) と P の比 (対 P_ref) を見る.

パート A (最良点の周りでの感度):
    実験 8 の 6 マス (改善が大きい 4 マスと, 価値がない対照 2 マス) で, P1 only (protect) と
    P1+P2 (protect) の (b) の最良点 (scripts/blocking_reassessment.py の結果) を基準点とし,
    パラメータを 1 つずつずらす: n_target in {0, 2, ..., 20}, gamma in {1, 2, 3, 5, 10, 30, 100,
    300, 1000}, beta = 基準点の beta x {0.1, 0.3, 0.5, 1, 2, 3, 10}.
    「Base より悪い」= ERP の相対差 > 0 または P の比 > 1.
パート B (Base の最良の β のまま Predictive の規則を ON にした場合):
    実験 8 の 35 マスで, Base の ERP 最小点の β* を固定し, P1 only (protect) (n_target in {10, 20},
    gamma=1) と P2 only (n_target=0, gamma in {3, 10}) を計算して, 同じ β* の Base と比べる.
    改善 (ERP が下がり P が上がらない), 引き換え (片方だけ改善), 悪化 (両方悪化) に分ける.

計算結果は 1 点ごとに results/experiment_9/experiment_9_{A,B}.csv に追記し, 再実行時は計算済みの
点を飛ばして再開する. 進捗は results/experiment_9/progress.log に追記する.

使用例:
    python scripts/experiment_9_sensitivity.py --part A
    python scripts/experiment_9_sensitivity.py --part B
    python scripts/experiment_9_sensitivity.py --count
    python scripts/experiment_9_sensitivity.py --report
"""
import argparse
import csv
import os
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import experiment_8_map as e8
    import blocking_reassessment as br
except ImportError:
    import scripts.experiment_8_map as e8
    import scripts.blocking_reassessment as br

warnings.filterwarnings("ignore", message="Predictive モデルの縮約後の帯幅が大きい")

OUT_DIR = os.path.join("results", "experiment_9")
FIG_DIR = "figures"
LOG_NAME = "progress.log"
P = "P_block_arrival_stable"
RTOL = 1e-9

A_CELLS = [("B_alpha1", 3.1622776601683795), ("A_rhoB0.8", 17.78279410038923),
           ("B_alpha0.005", 17.78279410038923), ("A_rhoB0.6", 3.1622776601683795),
           ("A_rhoB0.8", 0.1), ("B_alpha10", 17.78279410038923)]
A_CONTROL = {("A_rhoB0.8", 0.1), ("B_alpha10", 17.78279410038923)}
A_SERIES = ["P1 only (protect)", "P1+P2 (protect)"]
NT_SWEEP = list(range(0, 21, 2))
GAMMA_SWEEP = [1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
BETA_MULT = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 10.0]

# パート B の設定: (名前, 系列, n_target, gamma, 保護)
B_CONFIGS = [("P1 only (protect) n_target=10", "P1 only (protect)", 10, 1.0, True),
             ("P1 only (protect) n_target=20", "P1 only (protect)", 20, 1.0, True),
             ("P2 only γ=3", "P2 only", 0, 3.0, False),
             ("P2 only γ=10", "P2 only", 0, 10.0, False)]

METRICS = ["P_block", P, "E_N", "E_B", "E_I", "E_S", "E_off", "lambda_eff", "E_W",
           "rho_server", "Cost", "ERP", "p1_fire_rate", "setup_cancel_rate"]
HEAD = ["part", "row", "r1", "rho_B", "rho", "alpha", "sigma", "series", "sweep", "value",
        "beta", "n_target", "gamma", "protect"]
REF = ["base_erp_min", "p_ref", "base_beta_star", "erp_rel", "p_ratio"]
B_EXTRA = ["base_E_W", "base_Cost", "base_P", "base_ERP", "rel_E_W", "rel_Cost", "rel_P",
           "rel_ERP", "category"]
FIELDS_A = HEAD + METRICS + REF + ["min_pi", "elapsed_s", "check"]
FIELDS_B = HEAD + METRICS + REF + B_EXTRA + ["min_pi", "elapsed_s", "check"]


def csv_path(part: str, out_dir: str = OUT_DIR) -> str:
    return os.path.join(out_dir, f"experiment_9_{part}.csv")


def read_rows(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_row(path: str, fields, row: dict) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def log_progress(out_dir, done, total, t0, avg, last):
    line = (f"{datetime.now().isoformat(timespec='seconds')} done={done}/{total} "
            f"elapsed={time.time() - t0:.1f}s eta={max(total - done, 0) * avg:.1f}s last={last}")
    with open(os.path.join(out_dir, LOG_NAME), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


# ============================================================
# 基準 (実験 8 の結果から読む)
# ============================================================

_ASSESS = None


def assessments() -> Dict[Tuple[str, float], dict]:
    """実験 8 の各マスの判定し直しの結果 (blocking_reassessment.assess)."""
    global _ASSESS
    if _ASSESS is None:
        data = br.load_exp8()
        _ASSESS = {k: br.assess(g, br.E8_SERIES) for k, g in data.items()}
    return _ASSESS


def _cell_key(row, r1):
    for (rw, x) in assessments():
        if rw == row and abs(x / r1 - 1) < 1e-9:
            return (rw, x)
    raise KeyError((row, r1))


# ============================================================
# 点の一覧
# ============================================================

def points_A() -> List[dict]:
    pts = []
    for row, r1 in A_CELLS:
        key = _cell_key(row, r1)
        res = assessments()[key]
        for s in A_SERIES:
            b = res["series"][s]["b"]
            if b is None:
                continue
            beta0, nt0, g0 = b["beta"], b["n_target"], b["gamma"]
            for v in NT_SWEEP:
                pts.append(dict(row=row, r1=key[1], series=s, sweep="n_target", value=v,
                                beta=beta0, n_target=v, gamma=g0))
            for v in GAMMA_SWEEP:
                pts.append(dict(row=row, r1=key[1], series=s, sweep="gamma", value=v,
                                beta=beta0, n_target=nt0, gamma=v))
            for v in BETA_MULT:
                pts.append(dict(row=row, r1=key[1], series=s, sweep="beta_mult", value=v,
                                beta=beta0 * v, n_target=nt0, gamma=g0))
    return pts


def points_B() -> List[dict]:
    pts = []
    for row in dict.fromkeys(e8.MAP_A + e8.MAP_B):
        for r1 in e8.R1_LEVELS:
            key = _cell_key(row, r1)
            beta = assessments()[key]["base_best"]["beta"]
            for name, s, nt, g, prot in B_CONFIGS:
                pts.append(dict(row=row, r1=key[1], series=s, sweep="config", value=name,
                                beta=beta, n_target=nt, gamma=g, protect=prot))
    return pts


def pkey(p) -> Tuple:
    return (p["row"], round(float(p["r1"]), 9), p["series"], p["sweep"], str(p["value"]))


# ============================================================
# 計算
# ============================================================

def compute(part: str, p: dict) -> dict:
    key = _cell_key(p["row"], p["r1"])
    res = assessments()[key]
    cp = e8.cell_params(p["row"], key[1])
    protect = p.get("protect", True)
    out, _ = e8.compute_predictive(p["row"], cp, p["series"], f"exp9{part}", p["beta"],
                                   int(p["n_target"]), float(p["gamma"]), protect)
    out.update({"part": part, "sweep": p["sweep"], "value": p["value"], "protect": protect})
    base_best = res["base_best"]
    out["base_erp_min"] = base_best["ERP"]
    out["p_ref"] = res["p_ref"]
    out["base_beta_star"] = base_best["beta"]
    out["erp_rel"] = out["ERP"] / base_best["ERP"] - 1
    out["p_ratio"] = out[P] / res["p_ref"]
    if part == "B":
        # 同じ β* の Base は実験 8 の Base の ERP 最小点そのもの
        for k, bk in [("E_W", "E_W"), ("Cost", "Cost"), ("P", P), ("ERP", "ERP")]:
            out[f"base_{k}"] = base_best[bk]
            out[f"rel_{k}"] = out[bk] / base_best[bk] - 1
        out["category"] = category(out["rel_ERP"], out["rel_P"])
    return out


def category(rel_erp: float, rel_p: float) -> str:
    erp_better = rel_erp < 0
    p_ok = rel_p <= RTOL
    if erp_better and p_ok:
        return "改善"
    if not erp_better and not p_ok:
        return "悪化"
    return "引き換え"


def run(part: str, out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    pts = points_A() if part == "A" else points_B()
    path = csv_path(part, out_dir)
    done_keys = {pkey(r) for r in read_rows(path)}
    todo = [p for p in pts if pkey(p) not in done_keys]
    total, done = len(pts), len(pts) - len(todo)
    print(f"[{part}] 総点数 {total} (計算済み {done}, 今回計算 {len(todo)})", flush=True)
    fields = FIELDS_A if part == "A" else FIELDS_B
    t0 = time.time()
    last_log = t0
    for k, p in enumerate(todo):
        row = compute(part, p)
        append_row(path, fields, row)
        done += 1
        last = (f"[{part}] {p['row']},r1={p['r1']:.3g},{p['series']},{p['sweep']}={p['value']}")
        if row["check"] != "ok":
            print(f"  検査違反: {last}: {row['check']}", flush=True)
        if done % 10 == 0 or time.time() - last_log >= 60 or k + 1 == len(todo):
            log_progress(out_dir, done, total, t0, (time.time() - t0) / (k + 1), last)
            last_log = time.time()
    print(f"[{part}] 完了", flush=True)


# ============================================================
# 集計
# ============================================================

def _f(r, k):
    return float(r[k])


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


def _cell_label(row, r1):
    return f"{row} r1={float(r1):.3g}"


def report_A(rows: List[dict], fig_dir: str = FIG_DIR) -> Tuple[List[str], List[str]]:
    lines = ["## パート A: 最良点の周りでの感度", "",
             "基準点は各マスの (b) の最良点 (P ≤ P_ref の中の ERP 最小点; results/experiment_8/"
             "blocking_reassessment.md). ERP の相対差は Base の ERP 最小値に対するもの, P の比は "
             "P_ref (Base の ERP 最小点の P) に対するもの. 「Base より悪い」= ERP の相対差 > 0 または "
             "P の比 > 1. 各パラメータの範囲は, その格子の上で悪くなる値の区間 (β は基準点の β に"
             "対する倍率).", ""]
    fmts = {"n_target": lambda v: f"{int(v)}", "gamma": lambda v: f"{v:g}",
            "beta_mult": lambda v: f"×{v:g}"}
    names = {"n_target": "n_target", "gamma": "γ", "beta_mult": "β"}
    figs = []
    for row, r1 in A_CELLS:
        cr = [r for r in rows if r["row"] == row and abs(_f(r, "r1") / r1 - 1) < 1e-9]
        if not cr:
            continue
        tag = "対照 (価値がないマス)" if (row, r1) in A_CONTROL else "改善が大きいマス"
        lines += [f"### {_cell_label(row, r1)} ({tag})", "",
                  f"Base の ERP 最小値 {_f(cr[0], 'base_erp_min'):.4f}, P_ref {_f(cr[0], 'p_ref'):.3e}.",
                  "", "| 系列 | 基準点 | パラメータ | ERP のみで悪い | P のみで悪い | どちらかで悪い |",
                  "|---|---|---|---|---|---|"]
        for s in A_SERIES:
            sr = [r for r in cr if r["series"] == s]
            if not sr:
                lines.append(f"| {s} | (b) の最良点なし | | | | |")
                continue
            ref = next(r for r in sr if r["sweep"] == "beta_mult" and _f(r, "value") == 1.0)
            ref_txt = (f"β={_f(ref, 'beta'):.3g}, n_target={int(ref['n_target'])}, "
                       f"γ={_f(ref, 'gamma'):g} (ERP {100 * _f(ref, 'erp_rel'):+.2f}%, "
                       f"P 比 {_f(ref, 'p_ratio'):.2f})")
            for sw in ("n_target", "gamma", "beta_mult"):
                pts = sorted([r for r in sr if r["sweep"] == sw], key=lambda r: _f(r, "value"))
                vals = [_f(r, "value") for r in pts]
                fe = [_f(r, "erp_rel") > 0 for r in pts]
                fp = [_f(r, "p_ratio") > 1 + RTOL for r in pts]
                fa = [a or b for a, b in zip(fe, fp)]
                lines.append(f"| {s} | {ref_txt if sw == 'n_target' else ''} | {names[sw]} | "
                             f"{intervals(vals, fe, fmts[sw])} | {intervals(vals, fp, fmts[sw])} | "
                             f"{intervals(vals, fa, fmts[sw])} |")
        lines.append("")
        figs += plot_A(row, r1, cr, fig_dir)
    return lines, figs


def plot_A(row, r1, cr, fig_dir) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex="col")
    xlabels = {"n_target": r"$n_{\mathrm{target}}$", "gamma": r"$\gamma$",
               "beta_mult": r"$\beta / \beta_{\mathrm{ref}}$"}
    colors = {"P1 only (protect)": "tab:orange", "P1+P2 (protect)": "tab:purple"}
    for j, sw in enumerate(("n_target", "gamma", "beta_mult")):
        for s in A_SERIES:
            pts = sorted([r for r in cr if r["series"] == s and r["sweep"] == sw],
                         key=lambda r: _f(r, "value"))
            if not pts:
                continue
            x = [_f(r, "value") for r in pts]
            axes[0, j].plot(x, [100 * _f(r, "erp_rel") for r in pts], marker="o", markersize=4,
                            color=colors[s], label=s)
            axes[1, j].plot(x, [_f(r, "p_ratio") for r in pts], marker="o", markersize=4,
                            color=colors[s])
            ref = next(r for r in cr if r["series"] == s and r["sweep"] == "beta_mult"
                       and _f(r, "value") == 1.0)
            xr = _f(ref, "n_target") if sw == "n_target" else (
                _f(ref, "gamma") if sw == "gamma" else 1.0)
            axes[0, j].axvline(xr, color=colors[s], linestyle=":", linewidth=1)
            axes[1, j].axvline(xr, color=colors[s], linestyle=":", linewidth=1)
        axes[0, j].axhline(0, color="black", linewidth=1)
        axes[1, j].axhline(1, color="black", linewidth=1)
        axes[1, j].set_yscale("log")
        if sw != "n_target":
            axes[1, j].set_xscale("log")
        axes[1, j].set_xlabel(xlabels[sw], fontsize=16)
        for i in range(2):
            axes[i, j].grid(True, alpha=0.3)
    axes[0, 0].set_ylabel(r"$\Delta$ERP vs Base [%]", fontsize=14)
    axes[1, 0].set_ylabel(r"$P / P_{\mathrm{ref}}$", fontsize=14)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
               fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(fig_dir, exist_ok=True)
    png = os.path.join(fig_dir, f"experiment_9_A_{row}_r1_{float(r1):.3g}.png")
    fig.savefig(png, dpi=120, bbox_inches="tight")
    fig.savefig(os.path.splitext(png)[0] + ".pdf", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [png, os.path.splitext(png)[0] + ".pdf"]


def report_B(rows: List[dict], fig_dir: str = FIG_DIR) -> Tuple[List[str], List[str]]:
    lines = ["## パート B: Base の最良の β のまま Predictive の規則を ON にした場合", "",
             "各マスで Base の ERP 最小点の β* を固定し, 同じ β* の Base と比べた相対差. "
             "改善 = ERP が下がり P が上がらない, 引き換え = 片方だけ改善, 悪化 = ERP が下がらず P が上がる.", ""]
    figs = []
    for name, *_ in B_CONFIGS:
        cr = [r for r in rows if r["value"] == name]
        cats = {c: [r for r in cr if r["category"] == c] for c in ("改善", "引き換え", "悪化")}
        lines += [f"### {name}", "",
                  f"改善 {len(cats['改善'])} マス, 引き換え {len(cats['引き換え'])} マス, "
                  f"悪化 {len(cats['悪化'])} マス (計 {len(cr)}).", ""]
        if cats["改善"]:
            b = min(cats["改善"], key=lambda r: _f(r, "rel_ERP"))
            lines.append(f"- 最大の改善: {_cell_label(b['row'], b['r1'])} で ERP {100 * _f(b, 'rel_ERP'):+.2f}% "
                         f"(P {100 * _f(b, 'rel_P'):+.2f}%).")
        worst = max(cr, key=lambda r: _f(r, "rel_ERP"))
        lines.append(f"- ERP の最大の悪化: {_cell_label(worst['row'], worst['r1'])} で ERP "
                     f"{100 * _f(worst, 'rel_ERP'):+.2f}% (P {100 * _f(worst, 'rel_P'):+.2f}%, "
                     f"区分 {worst['category']}).")
        wp = max(cr, key=lambda r: _f(r, "rel_P"))
        lines.append(f"- P の最大の悪化: {_cell_label(wp['row'], wp['r1'])} で P "
                     f"{100 * _f(wp, 'rel_P'):+.2f}% (ERP {100 * _f(wp, 'rel_ERP'):+.2f}%, "
                     f"区分 {wp['category']}).")
        for c in ("改善", "引き換え", "悪化"):
            if cats[c]:
                lines.append(f"- {c}: " + ", ".join(_cell_label(r["row"], r["r1"]) for r in
                                                     sorted(cats[c], key=lambda r: (r["row"],
                                                                                    _f(r, "r1")))))
        lines.append("")
        for which, rows_order in [("A", e8.MAP_A), ("B", e8.MAP_B)]:
            lines += [f"地図 {which} (相対差: ERP / E[W] / Cost / P, 区分):", "",
                      "| | " + " | ".join(f"r1={r1:.3g}" for r1 in e8.R1_LEVELS) + " |",
                      "|---|" + "---|" * len(e8.R1_LEVELS)]
            for row in rows_order:
                cells = []
                for r1 in e8.R1_LEVELS:
                    m = [r for r in cr if r["row"] == row and abs(_f(r, "r1") / r1 - 1) < 1e-9]
                    if not m:
                        cells.append("—")
                        continue
                    r = m[0]
                    cells.append(f"{100 * _f(r, 'rel_ERP'):+.2f} / {100 * _f(r, 'rel_E_W'):+.2f} / "
                                 f"{100 * _f(r, 'rel_Cost'):+.2f} / {100 * _f(r, 'rel_P'):+.1f} "
                                 f"({r['category']})")
                lines.append(f"| {e8._row_label(row, which)} | " + " | ".join(cells) + " |")
            lines.append("")
    for which, rows_order in [("A", e8.MAP_A), ("B", e8.MAP_B)]:
        for metric, label, logp in [("rel_ERP", r"$\Delta$ERP [%]", False),
                                    ("rel_E_W", r"$\Delta E[W]$ [%]", False),
                                    ("rel_Cost", r"$\Delta$Cost [%]", False),
                                    ("rel_P", r"$\log_{10}(P / P_{\mathrm{Base}})$", True)]:
            figs += plot_B(rows, rows_order, which, metric, label, logp, fig_dir)
    return lines, figs


def plot_B(rows, rows_order, which, metric, label, logp, fig_dir) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mats = {}
    for name, *_ in B_CONFIGS:
        M = np.full((len(rows_order), len(e8.R1_LEVELS)), np.nan)
        for i, row in enumerate(rows_order):
            for j, r1 in enumerate(e8.R1_LEVELS):
                m = [r for r in rows if r["value"] == name and r["row"] == row
                     and abs(_f(r, "r1") / r1 - 1) < 1e-9]
                if m:
                    v = _f(m[0], metric)
                    M[i, j] = np.log10(1 + v) if logp else 100 * v
        mats[name] = M
    vmax = max(np.nanmax(np.abs(M)) for M in mats.values())
    vmax = vmax if vmax > 0 else 1.0
    ylabels = [e8._row_label(r, which).replace("ρ_B", r"$\rho_B$").replace("α", r"$\alpha$")
               for r in rows_order]
    fig, axes = plt.subplots(1, len(B_CONFIGS), figsize=(4.6 * len(B_CONFIGS),
                             0.9 * len(rows_order) + 2.2), sharey=True)
    for ax, (name, *_) in zip(axes, B_CONFIGS):
        M = mats[name]
        # 改善 (値が負) を青にそろえる
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", origin="lower")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=9,
                            color="white" if abs(M[i, j]) > 0.6 * vmax else "black")
        ax.set_xticks(range(len(e8.R1_LEVELS)))
        ax.set_xticklabels([f"{v:.2f}" for v in np.log10(e8.R1_LEVELS)])
        ax.set_xlabel(r"$\log_{10} r_1$", fontsize=14)
        ax.set_title(name.replace("γ", r"$\gamma$"), fontsize=12)
    axes[0].set_yticks(range(len(rows_order)))
    axes[0].set_yticklabels(ylabels, fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    cax = fig.add_axes([0.93, 0.2, 0.012, 0.6])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=11)
    stem = metric.replace("rel_", "").replace("E_W", "EW")
    png = os.path.join(fig_dir, f"experiment_9_B_{which}_{stem}.png")
    fig.savefig(png, dpi=120, bbox_inches="tight")
    fig.savefig(os.path.splitext(png)[0] + ".pdf", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [png, os.path.splitext(png)[0] + ".pdf"]


def report(out_dir: str = OUT_DIR, fig_dir: str = FIG_DIR) -> str:
    rows_a, rows_b = read_rows(csv_path("A", out_dir)), read_rows(csv_path("B", out_dir))
    lines = ["# 実験 9: Predictive のパラメータが最適でないとき, ベースモデルより悪くなる範囲", "",
             "results/experiment_9/experiment_9_{A,B}.csv から scripts/experiment_9_sensitivity.py で生成. "
             "固定: c=20, K=200, b=5, μ=1, δ=0.6 (実験 8 と同じ). 比較の基準は同じ信頼性での ERP "
             "(P_ref = Base の ERP 最小点の P_block_arrival_stable).", "",
             f"検査: パート A {len(rows_a)} 点, パート B {len(rows_b)} 点で, check 列の違反は "
             f"{sum(1 for r in rows_a + rows_b if r['check'] != 'ok')} 件.", "",
             "**旧実験 5・6 について.** results/experiment_5_*.csv, results/experiment_6_*.csv と "
             "figures/experiment_5_*, figures/experiment_6_* (と results/experiment_6_extended/) は, "
             "事前セットアップの保護 (protect_presetup, protect_delayoff) を入れる前の, "
             "保護なしの旧モデルによる結果である. 消さずに残してあるが, 保護ありのモデルの感度は "
             "この実験 9 で調べ直している.", ""]
    figs = []
    if rows_a:
        la, fa = report_A(rows_a, fig_dir)
        lines += la
        figs += fa
    if rows_b:
        lb, fb = report_B(rows_b, fig_dir)
        lines += lb
        figs += fb
    concl = os.path.join(out_dir, "conclusion.md")
    if os.path.exists(concl):
        lines += [open(concl, encoding="utf-8").read().rstrip(), ""]
    lines += ["## 図", ""] + [f"- {f}" for f in figs] + [""]
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=["A", "B"])
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--count", action="store_true")
    args = parser.parse_args()
    if args.count:
        print("A", len(points_A()), "B", len(points_B()))
        return
    if args.report:
        print(report())
        return
    run(args.part)


if __name__ == "__main__":
    main()
