"""旧結果 (results/, splu) と新結果 (results_gth/, GTH) の比較画像を生成する.

各 CSV ペアについて, 主要指標を旧 (実線) vs 新 (破線) で重ね描きし,
さらに相対差を対数軸のサブプロットで示す。特に P_block が極小になる
低 rho 領域で splu 由来のノイズが GTH で解消されることを可視化する。

出力先: figures_gth/comparison/

使用例:
    python scripts/make_gth_vs_splu_figures.py
"""
import csv
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join("figures_gth", "comparison")

# ファイル名の接頭辞 -> (x 軸の列名, x 軸ラベル, 系列を分ける列名 or None, 対数 x 軸か)
# 先頭から順に照合し, 最初に前方一致したものを使う (長い接頭辞を先に置く)。
SWEEP_SPEC = [
    # ベースモデル
    ("experiment_1.csv", ("rho", r"$\rho$", "burst_name", False)),
    ("experiment_2_", ("beta", r"$\beta$", "alpha", True)),
    ("experiment_3_delta", ("delta", r"$\delta$", "alpha", False)),
    ("experiment_3_sigma", ("sigma", r"$\sigma$", "alpha", True)),
    ("experiment_4_", ("rho", r"$\rho$", "K", False)),
    # Predictive モデル
    ("experiment_1P_", ("rho", r"$\rho$", "burst_name", False)),
    ("experiment_2P_", ("beta", r"$\beta$", "alpha", True)),
    ("experiment_3P_delta", ("delta", r"$\delta$", "alpha", False)),
    ("experiment_3P_sigma", ("sigma", r"$\sigma$", "alpha", True)),
    ("experiment_4P_", ("rho", r"$\rho$", "K", False)),
    ("experiment_5_", ("n_target", r"$n_{\mathrm{target}}$", "alpha", False)),
    ("experiment_6_", ("gamma", r"$\gamma$", "alpha", True)),
]


def lookup_spec(name):
    """ファイル名に対応するスイープ仕様を返す (最長一致優先)."""
    best = None
    for prefix, spec in SWEEP_SPEC:
        if name.startswith(prefix):
            if best is None or len(prefix) > len(best[0]):
                best = (prefix, spec)
    return best[1] if best else None

# P_block_arrival (引き算版 1 - lambda_eff/lambda_bar) はライブラリから削除済みで,
# 新しく生成する CSV には出力されない。ここでは既存 CSV の旧列を読み,
# 安定版 P_block_arrival_stable と重ねて桁落ちを示す比較のためだけに残している
# (旧列がない CSV ではそのパネルは空になる)。
METRICS = ["P_block_arrival", "E_W", "Cost", "ERP"]


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_figure(name, old_rows, new_rows, x_col, x_label, series_col, log_x):
    series_vals = []
    if series_col:
        for r in old_rows:
            if r[series_col] not in series_vals:
                series_vals.append(r[series_col])
    else:
        series_vals = [None]

    fig, axes = plt.subplots(2, len(METRICS), figsize=(5 * len(METRICS), 8))
    if len(METRICS) == 1:
        axes = axes.reshape(2, 1)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(series_vals), 2)))

    for m_idx, metric in enumerate(METRICS):
        ax_val = axes[0, m_idx]
        ax_diff = axes[1, m_idx]

        for s_idx, sval in enumerate(series_vals):
            def sel(rows):
                if series_col is None:
                    sub = rows
                else:
                    sub = [r for r in rows if r[series_col] == sval]
                sub = sorted(sub, key=lambda r: float(r[x_col]))
                return sub

            old_sub, new_sub = sel(old_rows), sel(new_rows)
            if not old_sub or not new_sub or metric not in old_sub[0]:
                continue

            x = np.array([float(r[x_col]) for r in old_sub])
            y_old = np.array([float(r[metric]) for r in old_sub])
            y_new = np.array([float(r[metric]) for r in new_sub])

            c = colors[s_idx]
            lbl = f"{series_col}={sval}" if series_col else ""
            ax_val.plot(x, np.abs(y_old), "-", color=c, alpha=0.6,
                        label=f"splu {lbl}".strip())
            ax_val.plot(x, np.abs(y_new), "--", color=c,
                        label=f"GTH {lbl}".strip())

            # P_block_arrival (旧列) については, 引き算を含まない安定版も重ねる
            # (引き算版は 1 - lambda_eff/lambda_bar の桁落ちで極小領域が
            #  ノイズ・負値になるため, その差が一目で分かる)
            if metric == "P_block_arrival" and "P_block_arrival_stable" in new_sub[0]:
                y_stable = np.array(
                    [float(r["P_block_arrival_stable"]) for r in new_sub])
                ax_val.plot(x, np.abs(y_stable), ":", color=c, lw=2,
                            label=f"GTH stable {lbl}".strip())

            denom = np.maximum(np.maximum(np.abs(y_old), np.abs(y_new)), 1e-300)
            rel = np.abs(y_old - y_new) / denom
            ax_diff.plot(x, np.maximum(rel, 1e-18), "o-", color=c, ms=3, label=lbl)

        ax_val.set_yscale("log")
        ax_val.set_xlabel(x_label)
        ax_val.set_title(metric)
        if log_x:
            ax_val.set_xscale("log")
            ax_diff.set_xscale("log")
        ax_val.grid(alpha=0.3)
        if m_idx == 0:
            ax_val.legend(fontsize=7)

        ax_diff.set_yscale("log")
        ax_diff.set_xlabel(x_label)
        ax_diff.set_ylabel("relative difference")
        ax_diff.axhline(1e-4, color="red", ls=":", lw=1, label="1e-4 threshold")
        ax_diff.grid(alpha=0.3)
        if m_idx == 0:
            ax_diff.legend(fontsize=7)

    stem = name.replace(".csv", "")
    fig.suptitle(
        f"{stem}: splu (old, solid) vs GTH (new, dashed); bottom row = relative difference",
        fontsize=13,
    )
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, f"compare_{stem}.png")
    pdf = os.path.join(OUT_DIR, f"compare_{stem}.pdf")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def main():
    matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
    made = []
    for gth_path in sorted(glob.glob(os.path.join("results_gth", "*.csv"))):
        name = os.path.basename(gth_path)
        old_path = os.path.join("results", name)
        if not os.path.exists(old_path):
            continue
        spec = lookup_spec(name)
        if spec is None:
            continue
        x_col, x_label, series_col, log_x = spec
        old_rows, new_rows = load_csv(old_path), load_csv(gth_path)
        if not old_rows or not new_rows:
            continue
        png = make_figure(name, old_rows, new_rows, x_col, x_label, series_col, log_x)
        made.append(png)
        print(f"生成: {png}")

    print(f"\n合計 {len(made)} 枚の比較図を {OUT_DIR}/ に生成しました。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
