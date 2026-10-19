#!/usr/bin/env python
"""頑健性: Predictive のパラメータが最適でないとき, Base より悪くなる範囲を alpha=0.005 で調べる.

旧実験 9 (scripts/experiment_9_sensitivity.py) を, 比較の地図 (scripts/compare_map.py) の結果を
基準にしてやり直す. 旧実験 9 の結果 (results/experiment_9/) はそのまま残す.

固定: c=20, K=200, b=5, mu=1, delta=0.6, alpha=0.005 (compare_map.py と同じ). 保護は常にあり
(protect_presetup=protect_delayoff=True). 比較の基準: P_ref と Base の ERP 最小値は
results/compare_map/summary.csv の値を使う.

パート A (最良点の周りでの感度; Base と Predictive の 2 者の比較):
    対象のマスは summary.csv から選ぶ: 改善率が大きい順に 2 マス, 対照 1 (rho_B=0.8, r1=0.1),
    対照 2 (rho_B=1.12 の行で改善率が最も小さいマス). 重複したら次の候補を使う.
    基準点 = 各マスの Predictive の主な基準での最良点 (beta*, n_target*, gamma*). 1 つのパラメータ
    だけを動かし, 他は基準点に固定する: beta = beta* x {0.1, 0.3, 0.5, 1, 2, 3, 10},
    n_target in {0, 2, ..., 20}, gamma in {1, 2, 3, 5, 10, 30, 100, 300, 1000}.
    「Base より悪い」= ERP の相対差 (対 Base の ERP 最小値) > +0.1% または P / P_ref > 1.05
    (旧実験 9 のしきい値付きの判定と同じ).
パート B (Base の最良の beta のまま規則を加える; 規則ごとに分ける):
    15 マスすべてで Base の ERP 最小点の beta* を固定し, 事前セットアップ (n_target, gamma) =
    (10, 1), (20, 1) と Delayoff の加速 (0, 3), (0, 10) を計算して, 同じ beta* の Base と比べる.
    判定は旧実験 9 のしきい値付きの 5 区分 (改善 / 同等 / 棄却のみ改善 / 引き換え / 悪化).

計算結果は 1 点ごとに results/robustness/robustness_{A,B}.csv に追記し, 再実行時は計算済みの点を
飛ばして再開する. 進捗は results/robustness/progress.log に追記する.

使用例:
    python scripts/robustness.py --part A
    python scripts/robustness.py --part B
    python scripts/robustness.py --count
    python scripts/robustness.py --report
"""
import argparse
import csv
import os
import time
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

try:
    import compare_map as cm
    import experiment_8_map as e8
    import experiment_9_sensitivity as e9
except ImportError:
    import scripts.compare_map as cm
    import scripts.experiment_8_map as e8
    import scripts.experiment_9_sensitivity as e9

warnings.filterwarnings("ignore", message="Predictive モデルの縮約後の帯幅が大きい")

OUT_DIR = os.path.join("results", "robustness")
FIG_DIR = "figures"
LOG_NAME = "progress.log"
SUMMARY_CSV = os.path.join(cm.OUT_DIR, "summary.csv")
P = "P_block_arrival_stable"

RHO_B_LEVELS = cm.RHO_B_LEVELS
R1_LEVELS = cm.R1_LEVELS
CONTROL_1 = (0.8, 0.1)
CONTROL_2_ROW = 1.12
N_TOP = 2

SWEEPS = ["beta_mult", "n_target", "gamma"]
BETA_MULT = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 10.0]
NT_SWEEP = list(range(0, 21, 2))
GAMMA_SWEEP = [1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, 300.0, 1000.0]

# パート B の設定: (名前, 規則, n_target, gamma)
B_CONFIGS = [("presetup nt=10", "事前セットアップ", 10, 1.0),
             ("presetup nt=20", "事前セットアップ", 20, 1.0),
             ("delayoff g=3", "Delayoff の加速", 0, 3.0),
             ("delayoff g=10", "Delayoff の加速", 0, 10.0)]

# しきい値 (旧実験 9 のしきい値付きの判定と同じ)
ERP_TIE = e9.ERP_TIE
P_LO, P_HI = e9.P_LO, e9.P_HI
CATEGORIES5 = e9.CATEGORIES5
category5 = e9.category5
erp_state = e9.erp_state
p_state = e9.p_state
intervals = e9.intervals

REF = ["base_ERP_min", "P_ref", "base_beta_star", "erp_rel", "p_ratio"]
FIELDS_A = (["part", "cell_role", "sweep", "value"] + e8.FIELDNAMES + REF
            + ["worse_erp", "worse_p", "worse"])
FIELDS_B = (["part", "config", "rule"] + e8.FIELDNAMES + REF
            + ["base_E_W", "base_Cost", "rel_E_W", "rel_Cost", "rel_ERP", "category"])


def csv_path(part: str, out_dir: str = OUT_DIR) -> str:
    return os.path.join(out_dir, f"robustness_{part}.csv")


def read_rows(path: str) -> List[dict]:
    return e9.read_rows(path)


def append_row(path: str, fields, row: dict) -> None:
    e9.append_row(path, fields, row)


def log_progress(out_dir, msg):
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    with open(os.path.join(out_dir, LOG_NAME), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


# ============================================================
# 基準 (results/compare_map/summary.csv)
# ============================================================

def load_summary(path: str = SUMMARY_CSV) -> Dict[Tuple[float, float], dict]:
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = dict(r)
            for k in ("rho_B", "r1", "sigma", "base_beta", "base_E_W", "base_Cost", "P_ref",
                      "base_ERP", "pred_beta", "pred_gamma", "pred_E_W", "pred_Cost", "pred_P",
                      "pred_P_over_Pref", "pred_ERP", "improvement"):
                d[k] = float(r[k])
            d["pred_n_target"] = int(r["pred_n_target"])
            d["pred_is_base"] = r["pred_is_base"] == "True"
            out[(d["rho_B"], d["r1"])] = d
    return out


def select_A_cells(sm: Dict[Tuple[float, float], dict]) -> List[Tuple[Tuple[float, float], str]]:
    """パート A のマスと役割: 改善率の上位 2 マス, 対照 1, 対照 2 (重複したら次の候補)."""
    chosen: List[Tuple[Tuple[float, float], str]] = []

    def take(cands, role):
        for k in cands:
            if k not in [c for c, _ in chosen]:
                chosen.append((k, role))
                return

    by_imp = sorted(sm, key=lambda k: -sm[k]["improvement"])
    for i in range(N_TOP):
        take(by_imp, f"改善率の上位 {i + 1}")
    c1 = [k for k in sm if k == CONTROL_1] + [k for k in by_imp[::-1] if k != CONTROL_1]
    take(c1, "対照 1")
    row2 = sorted([k for k in sm if k[0] == CONTROL_2_ROW], key=lambda k: sm[k]["improvement"])
    take(row2, "対照 2")
    return chosen


def cell_tag(key: Tuple[float, float]) -> str:
    return f"rhoB{key[0]:g}_r1_{key[1]:.3g}"


def cell_label(key: Tuple[float, float]) -> str:
    return f"ρ_B={key[0]:g}, r1={key[1]:.3g}"


# ============================================================
# 点の一覧
# ============================================================

def points_A(sm=None) -> List[dict]:
    sm = load_summary() if sm is None else sm
    pts = []
    for key, role in select_A_cells(sm):
        s = sm[key]
        beta0, nt0, g0 = s["pred_beta"], s["pred_n_target"], s["pred_gamma"]
        base = dict(rho_B=key[0], r1=key[1], cell_role=role)
        for v in BETA_MULT:
            pts.append(dict(base, sweep="beta_mult", value=v, beta=beta0 * v, n_target=nt0,
                            gamma=g0))
        for v in NT_SWEEP:
            pts.append(dict(base, sweep="n_target", value=v, beta=beta0, n_target=v, gamma=g0))
        for v in GAMMA_SWEEP:
            pts.append(dict(base, sweep="gamma", value=v, beta=beta0, n_target=nt0, gamma=v))
    return pts


def points_B(sm=None) -> List[dict]:
    sm = load_summary() if sm is None else sm
    pts = []
    for rho_B in RHO_B_LEVELS:
        for r1 in R1_LEVELS:
            s = sm[(rho_B, r1)]
            for name, rule, nt, g in B_CONFIGS:
                pts.append(dict(rho_B=rho_B, r1=r1, config=name, rule=rule,
                                beta=s["base_beta"], n_target=nt, gamma=g))
    return pts


def pkey(p) -> Tuple:
    tail = (p["sweep"], float(p["value"])) if "sweep" in p and p.get("sweep") else (p["config"],)
    return (float(p["rho_B"]), float(p["r1"])) + tail


# ============================================================
# 計算
# ============================================================

def compute(part: str, p: dict, sm=None) -> dict:
    sm = load_summary() if sm is None else sm
    rho_B, r1 = float(p["rho_B"]), float(p["r1"])
    s = sm[(rho_B, r1)]
    cp = cm.cell_params(rho_B, r1)
    out, _ = e8.compute_predictive(cm.row_name(rho_B), cp, "Predictive", f"robust{part}",
                                   float(p["beta"]), int(p["n_target"]), float(p["gamma"]), True)
    out["part"] = part
    out["base_ERP_min"] = s["base_ERP"]
    out["P_ref"] = s["P_ref"]
    out["base_beta_star"] = s["base_beta"]
    out["erp_rel"] = out["ERP"] / s["base_ERP"] - 1
    out["p_ratio"] = out[P] / s["P_ref"]
    if part == "A":
        out.update({"cell_role": p["cell_role"], "sweep": p["sweep"], "value": p["value"]})
        out["worse_erp"] = erp_state(out["erp_rel"]) == "worse"
        out["worse_p"] = p_state(out["p_ratio"]) == "worse"
        out["worse"] = out["worse_erp"] or out["worse_p"]
    else:
        # 同じ beta* の Base は compare_map の Base の ERP 最小点そのもの
        out.update({"config": p["config"], "rule": p["rule"],
                    "base_E_W": s["base_E_W"], "base_Cost": s["base_Cost"],
                    "rel_E_W": out["E_W"] / s["base_E_W"] - 1,
                    "rel_Cost": out["Cost"] / s["base_Cost"] - 1,
                    "rel_ERP": out["erp_rel"]})
        out["category"] = category5(out["rel_ERP"], out["p_ratio"])
    return out


def run(part: str, out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    sm = load_summary()
    pts = points_A(sm) if part == "A" else points_B(sm)
    path = csv_path(part, out_dir)
    done_keys = {pkey(r) for r in read_rows(path)}
    todo = [p for p in pts if pkey(p) not in done_keys]
    total, done = len(pts), len(pts) - len(todo)
    log_progress(out_dir, f"[{part}] 開始: 総点数 {total} (計算済み {done}, 今回計算 {len(todo)})")
    fields = FIELDS_A if part == "A" else FIELDS_B
    t0 = time.time()
    last_log = t0
    n_bad = 0
    for k, p in enumerate(todo):
        row = compute(part, p, sm)
        append_row(path, fields, row)
        done += 1
        tail = f"{p['sweep']}={p['value']:g}" if part == "A" else p["config"]
        last = f"[{part}] {cell_label((p['rho_B'], p['r1']))},{tail}"
        if row["check"] != "ok":
            n_bad += 1
            print(f"  検査違反: {last}: {row['check']}", flush=True)
        if done % 10 == 0 or time.time() - last_log >= 60 or k + 1 == len(todo):
            avg = (time.time() - t0) / (k + 1)
            log_progress(out_dir, f"done={done}/{total} elapsed={time.time() - t0:.1f}s "
                                  f"eta={(total - done) * avg:.1f}s last={last}")
            last_log = time.time()
    log_progress(out_dir, f"[{part}] 完了: 今回計算 {len(todo)} 点, 検査違反 {n_bad} 件, "
                          f"経過 {time.time() - t0:.1f}s")


# ============================================================
# 集計: パート A
# ============================================================

def _f(r, k):
    return float(r[k])


FMTS = {"n_target": lambda v: f"{int(v)}", "gamma": lambda v: f"{v:g}",
        "beta_mult": lambda v: f"×{v:g}"}
NAMES = {"n_target": "n_target", "gamma": "γ", "beta_mult": "β"}


def _cell_rows(rows, key):
    return [r for r in rows if _f(r, "rho_B") == key[0] and _f(r, "r1") == key[1]]


def _sweep_rows(cr, sw):
    return sorted([r for r in cr if r["sweep"] == sw], key=lambda r: _f(r, "value"))


def a_intervals(cr) -> Dict[str, Tuple[str, str, str]]:
    """パラメータごとの「Base より悪い」区間 (ERP のみ, P のみ, どちらか)."""
    out = {}
    for sw in SWEEPS:
        pts = _sweep_rows(cr, sw)
        vals = [_f(r, "value") for r in pts]
        fe = [erp_state(_f(r, "erp_rel")) == "worse" for r in pts]
        fp = [p_state(_f(r, "p_ratio")) == "worse" for r in pts]
        fa = [a or b for a, b in zip(fe, fp)]
        out[sw] = tuple(intervals(vals, f, FMTS[sw]) for f in (fe, fp, fa))
    return out


def report_A(rows: List[dict], sm, fig_dir: str = FIG_DIR) -> Tuple[List[str], List[str]]:
    lines = ["## パート A: 最良点の周りでの感度", "",
             "基準点は各マスの Predictive の主な基準での最良点 (summary.csv の pred_*). "
             "1 つのパラメータだけを動かし, 他は基準点に固定する (β は基準点の β に対する倍率). "
             "ERP の相対差は Base の ERP 最小値に対するもの, P の比は P_ref に対するもの. "
             f"「Base より悪い」= ERP の相対差 > +{100 * ERP_TIE:g}% または P の比 > {P_HI:g}. "
             "「ERP のみ」「P のみ」はそれぞれの条件だけで判定した区間 (重なりうる), "
             "「どちらか」はその和集合.", "",
             "### 選んだマス", "",
             "| 役割 | マス | 改善率 | 基準点 (β*, n_target*, γ*) | P*/P_ref | Base の ERP 最小値 | P_ref |",
             "|---|---|---|---|---|---|---|"]
    for key, role in select_A_cells(sm):
        s = sm[key]
        ref = (f"({s['pred_beta']:.4g}, {s['pred_n_target']}, {s['pred_gamma']:g})"
               + (" = Base の点" if s["pred_is_base"] else ""))
        lines.append(f"| {role} | {cell_label(key)} | {100 * s['improvement']:.3f}% | {ref} | "
                     f"{s['pred_P_over_Pref']:.3f} | {s['base_ERP']:.6g} | {s['P_ref']:.4e} |")
    lines += ["", "### 「Base より悪い」区間", "",
              "| マス | パラメータ | ERP のみで悪い | P のみで悪い | どちらかで悪い |",
              "|---|---|---|---|---|"]
    detail, figs = [], []
    for key, role in select_A_cells(sm):
        cr = _cell_rows(rows, key)
        if not cr:
            continue
        iv = a_intervals(cr)
        for sw in SWEEPS:
            lines.append(f"| {cell_label(key)} ({role}) | {NAMES[sw]} | " + " | ".join(iv[sw]) + " |")
        detail += [f"### {cell_label(key)} ({role})", ""]
        for sw in SWEEPS:
            pts = _sweep_rows(cr, sw)
            fe = [erp_state(_f(r, "erp_rel")) == "worse" for r in pts]
            fp = [p_state(_f(r, "p_ratio")) == "worse" for r in pts]
            detail += [f"{NAMES[sw]}:", "",
                       "| | " + " | ".join(FMTS[sw](_f(r, "value")) for r in pts) + " |",
                       "|---|" + "---|" * len(pts),
                       "| ERP の相対差 | " + " | ".join(
                           f"{100 * _f(r, 'erp_rel'):+.3f}%{'*' if a else ''}"
                           for r, a in zip(pts, fe)) + " |",
                       "| P の比 | " + " | ".join(
                           f"{_f(r, 'p_ratio'):.3f}{'*' if b else ''}"
                           for r, b in zip(pts, fp)) + " |", ""]
        figs += plot_A(key, cr, sm[key], fig_dir)
    lines += [""] + _a_notes(rows, sm)
    lines += ["### 各値の ERP の相対差と P の比 (悪い値に *)", ""] + detail
    return lines, figs


def _a_notes(rows, sm) -> List[str]:
    """表を読むときの注意: 格子に入らない基準点と, beta と gamma が積でしか効かないマス."""
    notes = []
    for key, role in select_A_cells(sm):
        s = sm[key]
        if s["pred_n_target"] not in NT_SWEEP:
            notes.append(f"- {cell_label(key)} ({role}): n_target* = {s['pred_n_target']} は "
                         "n_target の格子に入らない (表の n_target の行に基準点はない).")
        cr = [r for r in _cell_rows(rows, key) if r["sweep"] in ("beta_mult", "gamma")]
        same = {}
        for r in cr:
            same.setdefault(float(f"{_f(r, 'beta') * _f(r, 'gamma'):.12e}"), set()).add(
                (r["sweep"], _f(r, "value"), _f(r, "ERP"), _f(r, P)))
        pairs = [v for v in same.values() if len({(a, b) for a, b, _, _ in v}) > 1]
        if pairs and s["pred_n_target"] == e8.BASELINE["c"]:
            err = max(max(abs(x[2] / y[2] - 1), abs(x[3] / y[3] - 1))
                      for v in pairs for x in v for y in v)
            notes.append(
                f"- {cell_label(key)} ({role}): n_target* = c = {e8.BASELINE['c']} で保護ありなので, "
                "バースト中は台を切らず, β と γ は積 βγ (通常期の切る速さ) でしか効かない. "
                f"βγ が同じ点 ({len(pairs)} 組) で ERP と P の相対差は最大 {err:.1e}. "
                "このマスの β の行と γ の行は同じ 1 次元の走査を別の格子で見たもの.")
    return (["注意:", ""] + notes + [""]) if notes else []


def _style(ax):
    ax.grid(True, color="#e0e0e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#888888")
    ax.tick_params(colors="#333333", labelsize=10)


def plot_A(key, cr, s, fig_dir) -> List[str]:
    """パラメータごとに, ERP の相対差 (上段) と P/P_ref (下段). 「Base より悪い」点は赤."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    line_c, worse_c, thr_c = "#2a78d6", "#e34948", "#52514e"
    xlabels = {"beta_mult": r"$\beta / \beta^*$", "n_target": r"$n_{\mathrm{target}}$",
               "gamma": r"$\gamma$"}
    ref_x = {"beta_mult": 1.0, "n_target": s["pred_n_target"], "gamma": s["pred_gamma"]}
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharex="col")
    for j, sw in enumerate(SWEEPS):
        pts = _sweep_rows(cr, sw)
        x = np.array([_f(r, "value") for r in pts])
        ye = np.array([100 * _f(r, "erp_rel") for r in pts])
        yp = np.array([_f(r, "p_ratio") for r in pts])
        we = np.array([erp_state(_f(r, "erp_rel")) == "worse" for r in pts], dtype=bool)
        wp = np.array([p_state(_f(r, "p_ratio")) == "worse" for r in pts], dtype=bool)
        for ax, y, w in ((axes[0, j], ye, we), (axes[1, j], yp, wp)):
            _style(ax)
            ax.axvline(ref_x[sw], color="#9a9a9a", linestyle=":", linewidth=1.2)
            ax.plot(x, y, color=line_c, linewidth=2, zorder=2)
            ax.scatter(x[~w], y[~w], s=40, color=line_c, edgecolor="white", linewidth=1.5,
                       zorder=3, label="not worse than Base")
            ax.scatter(x[w], y[w], s=48, color=worse_c, edgecolor="white", linewidth=1.5,
                       marker="s", zorder=4, label="worse than Base")
        axes[0, j].axhline(100 * ERP_TIE, color=thr_c, linestyle="--", linewidth=1)
        axes[0, j].axhline(0, color="#888888", linewidth=0.8)
        axes[1, j].axhline(P_HI, color=thr_c, linestyle="--", linewidth=1)
        axes[1, j].axhline(1, color="#888888", linewidth=0.8)
        axes[1, j].set_yscale("log")
        # 範囲が 1 桁未満だと副目盛りに 1.04×10^0 のような表記が出るので, 小数で書く
        lo, hi = min(list(yp) + [1.0, P_HI]), max(list(yp) + [1.0, P_HI])
        axes[1, j].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        if hi / lo < 2:
            minor = FuncFormatter(lambda v, _: f"{v:.3g}")
        elif hi / lo < 10:
            minor = FuncFormatter(lambda v, _: f"{v:.3g}" if round(
                v / 10 ** np.floor(np.log10(v))) in (2, 5) else "")
        else:
            minor = NullFormatter()
        axes[1, j].yaxis.set_minor_formatter(minor)
        if sw != "n_target":
            axes[1, j].set_xscale("log")
        axes[1, j].set_xlabel(xlabels[sw], fontsize=14)
    axes[0, 0].set_ylabel(r"$\Delta$ERP vs Base min [%]", fontsize=12)
    axes[1, 0].set_ylabel(r"$P / P_{\mathrm{ref}}$", fontsize=12)
    fig.text(0.62, 0.985, f"dashed: +{100 * ERP_TIE:g}% / {P_HI:g}, dotted: reference point",
             ha="center", va="top", fontsize=10, color="#52514e")
    handles, labels = [], []
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, 1.0), ncol=2,
               fontsize=11, frameon=False)
    fig.suptitle(rf"$\rho_B$={key[0]:g}, $r_1$={key[1]:.3g}", x=0.98, y=0.995, ha="right",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(fig_dir, exist_ok=True)
    png = os.path.join(fig_dir, f"robustness_A_{cell_tag(key)}.png")
    pdf = os.path.splitext(png)[0] + ".pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


# ============================================================
# 集計: パート B
# ============================================================

def b_categories(rows: List[dict]) -> Dict[str, Dict[str, List[dict]]]:
    per = {}
    for name, *_ in B_CONFIGS:
        cats = {c: [] for c in CATEGORIES5}
        for r in rows:
            if r["config"] == name:
                cats[category5(_f(r, "rel_ERP"), _f(r, "p_ratio"))].append(r)
        per[name] = cats
    return per


def report_B(rows: List[dict], fig_dir: str = FIG_DIR) -> Tuple[List[str], List[str]]:
    lines = ["## パート B: Base の最良の β のまま規則を加える", "",
             "各マスで Base の ERP 最小点の β* を固定し, 規則を加えた Predictive (保護あり) を "
             "同じ β* の Base (= Base の ERP 最小点) と比べる. "
             f"改善 = ERP が {100 * ERP_TIE:g}% 以上下がり, P の比が {P_HI:g} を超えない. "
             "同等 = ERP も P も同等. 棄却のみ改善 = ERP は同等で, P の比が "
             f"{P_LO:g} 未満. 引き換え = 片方が改善し, もう片方が悪化する. "
             "悪化 = どちらかが悪化し, どちらも改善しない.", "",
             "| 設定 | 規則 | (n_target, γ) | " + " | ".join(CATEGORIES5) + " |",
             "|---|---|---|" + "---|" * len(CATEGORIES5)]
    per = b_categories(rows)
    for name, rule, nt, g in B_CONFIGS:
        lines.append(f"| {name} | {rule} | ({nt}, {g:g}) | "
                     + " | ".join(str(len(per[name][c])) for c in CATEGORIES5) + " |")
    lines.append("")
    for name, rule, nt, g in B_CONFIGS:
        cats = per[name]
        lines += [f"### {name} ({rule}, (n_target, γ) = ({nt}, {g:g}))", ""]
        for c in CATEGORIES5:
            if cats[c]:
                lines.append(f"- {c} ({len(cats[c])}): " + "; ".join(
                    cell_label((_f(r, "rho_B"), _f(r, "r1"))) for r in
                    sorted(cats[c], key=lambda r: (_f(r, "rho_B"), _f(r, "r1")))))
        lines += ["", "ERP の相対差 / P の比 (区分):", "",
                  "| | " + " | ".join(f"r1={r1:.3g}" for r1 in R1_LEVELS) + " |",
                  "|---|" + "---|" * len(R1_LEVELS)]
        for rho_B in RHO_B_LEVELS:
            cells = []
            for r1 in R1_LEVELS:
                m = [r for r in rows if r["config"] == name and _f(r, "rho_B") == rho_B
                     and _f(r, "r1") == r1]
                if not m:
                    cells.append("—")
                    continue
                r = m[0]
                cells.append(f"{100 * _f(r, 'rel_ERP'):+.3f}% / {_f(r, 'p_ratio'):.3f} "
                             f"({category5(_f(r, 'rel_ERP'), _f(r, 'p_ratio'))})")
            lines.append(f"| ρ_B={rho_B:g} | " + " | ".join(cells) + " |")
        lines.append("")
    figs = plot_B(rows, fig_dir)
    return lines, figs


# 5 区分の色と図の中の英語名 (日本語のフォントがないため)
CAT_STYLE = {"改善": ("#2a78d6", "Improved", "white"),
             "同等": ("#e3e3e0", "Equal", "#0b0b0b"),
             "棄却のみ改善": ("#1baf7a", "P only better", "#0b0b0b"),
             "引き換え": ("#eda100", "Trade-off", "#0b0b0b"),
             "悪化": ("#e34948", "Worse", "white")}


def plot_B(rows, fig_dir) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, len(B_CONFIGS), figsize=(4.4 * len(B_CONFIGS), 4.2),
                             sharey=True)
    for ax, (name, rule, nt, g) in zip(axes, B_CONFIGS):
        rgba = np.ones((len(RHO_B_LEVELS), len(R1_LEVELS), 4))
        for i, rho_B in enumerate(RHO_B_LEVELS):
            for j, r1 in enumerate(R1_LEVELS):
                m = [r for r in rows if r["config"] == name and _f(r, "rho_B") == rho_B
                     and _f(r, "r1") == r1]
                if not m:
                    continue
                r = m[0]
                cat = category5(_f(r, "rel_ERP"), _f(r, "p_ratio"))
                color, label, ink = CAT_STYLE[cat]
                rgba[i, j] = matplotlib.colors.to_rgba(color)
                ax.text(j, i + 0.17, label, ha="center", va="center", fontsize=9.5, color=ink,
                        fontweight="bold")
                ax.text(j, i - 0.17, f"{100 * _f(r, 'rel_ERP'):+.2f}%\n{_f(r, 'p_ratio'):.2f}",
                        ha="center", va="center", fontsize=7.5, color=ink, linespacing=1.1)
        ax.imshow(rgba, aspect="auto", origin="lower")
        ax.set_xticks(np.arange(-0.5, len(R1_LEVELS)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(RHO_B_LEVELS)), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(range(len(R1_LEVELS)))
        ax.set_xticklabels([f"{np.log10(v):.2f}" for v in R1_LEVELS], fontsize=10)
        ax.set_xlabel(r"$\log_{10} r_1$", fontsize=13)
        kind = "pre-setup" if nt > 0 else "delayoff accel."
        ax.set_title(rf"{kind}: $(n_{{\mathrm{{target}}}}, \gamma)$=({nt}, {g:g})", fontsize=11.5)
    axes[0].set_yticks(range(len(RHO_B_LEVELS)))
    axes[0].set_yticklabels([rf"$\rho_B$={v:g}" for v in RHO_B_LEVELS], fontsize=12)
    handles = [Patch(facecolor=CAT_STYLE[c][0], edgecolor="#bbbbbb", label=CAT_STYLE[c][1])
               for c in CATEGORIES5]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=len(CATEGORIES5), fontsize=11, frameon=False)
    fig.text(0.5, -0.02, r"cell text: category, $\Delta$ERP vs Base at $\beta^*$, "
             r"$P / P_{\mathrm{Base}}(\beta^*)$", ha="center", fontsize=10, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(fig_dir, exist_ok=True)
    png = os.path.join(fig_dir, "robustness_B.png")
    pdf = os.path.splitext(png)[0] + ".pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


# ============================================================
# レポート
# ============================================================

def report(out_dir: str = OUT_DIR, fig_dir: str = FIG_DIR) -> str:
    sm = load_summary()
    rows_a, rows_b = read_rows(csv_path("A", out_dir)), read_rows(csv_path("B", out_dir))
    bad = [r for r in rows_a + rows_b if r["check"] != "ok"]
    t_a = sum(_f(r, "elapsed_s") for r in rows_a)
    t_b = sum(_f(r, "elapsed_s") for r in rows_b)
    lines = ["# 頑健性: Predictive のパラメータが最適でないとき (α=0.005)", "",
             "results/robustness/robustness_{A,B}.csv から scripts/robustness.py --report で生成. "
             "固定: c=20, K=200, b=5, μ=1, δ=0.6, α=0.005 (scripts/compare_map.py と同じ). "
             "保護は常にあり. 比較の基準 (Base の ERP 最小値, その点の β*, P_ref) は "
             "results/compare_map/summary.csv の値. 旧実験 9 (results/experiment_9/) は別に残してある.", "",
             "## 点数と検査", "",
             f"- パート A: {len(rows_a)} / {len(points_A(sm))} 点 (求解時間の合計 {t_a:.0f} s)",
             f"- パート B: {len(rows_b)} / {len(points_B(sm))} 点 (求解時間の合計 {t_b:.0f} s)",
             f"- check 列の違反 (台数の保存, 流量バランス, min π ≥ 0, P_block > 0): {len(bad)} 件", ""]
    figs = []
    if rows_a:
        la, fa = report_A(rows_a, sm, fig_dir)
        lines += la
        figs += fa
    if rows_b:
        lb, fb = report_B(rows_b, fig_dir)
        lines += lb
        figs += fb
    lines += ["## 図", ""] + [f"- {f}" for f in figs] + [""]
    text = "\n".join(lines)
    os.makedirs(out_dir, exist_ok=True)
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
        sm = load_summary()
        for key, role in select_A_cells(sm):
            print(role, cell_label(key), f"improvement={sm[key]['improvement']:.4g}")
        print("A", len(points_A(sm)), "B", len(points_B(sm)))
        return
    if args.report:
        print(report())
        return
    if args.part is None:
        parser.error("--part, --report, --count のどれかを指定する")
    run(args.part)


if __name__ == "__main__":
    main()
