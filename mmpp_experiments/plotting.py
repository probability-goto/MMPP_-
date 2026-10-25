"""論文の図で共通に使う軸ラベルの書き方と, 4 指標の図."""
import os
from typing import List, Optional, Sequence, Tuple

# (CSV の列名, 軸ラベル, 対数目盛か)
METRIC_PANELS: List[Tuple[str, str, bool]] = [
    ("P_block_arrival_stable", r"$P_{\mathrm{block}}$", True),
    ("E_W", r"$E[W]$", False),
    ("Cost", "Cost", False),
    ("ERP", "ERP", False),
]


def log_scale_label(label: str) -> str:
    """縦軸を対数目盛にした指標のラベル. 元のラベルの後ろに " (log scale)" を付ける."""
    return f"{label} (log scale)"


def save_figure(fig, out_dir: str, stem: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, f"{stem}.pdf")
    png = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(pdf, dpi=100)
    fig.savefig(png, dpi=150)
    print(f"図を保存しました: {pdf}, {png}")
    return [pdf, png]


def plot_four_metrics(series: Sequence[Tuple[str, Sequence[dict], dict]], xkey: str,
                      xlabel: str, out_dir: str, stem: str, xlog: bool = False,
                      vline: Optional[float] = None, legend_ncol: int = 2) -> List[str]:
    """4 指標 (P_block, E[W], Cost, ERP) を 2x2 に描く.

    series は (凡例, 行のリスト, plot に渡す書式) の並び. 行は xkey の昇順に並べ替えて描く.
    vline を与えると, その x に点線を引く (rho_B = 1 の位置など).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (key, ylabel, ylog) in zip(axes.flat, METRIC_PANELS):
        for label, rows, style in series:
            rows = sorted(rows, key=lambda r: float(r[xkey]))
            ax.plot([float(r[xkey]) for r in rows], [float(r[key]) for r in rows],
                    marker="o", markersize=3, label=label if ax is axes.flat[0] else None,
                    **style)
        if vline is not None:
            ax.axvline(vline, color="0.5", linestyle=":", linewidth=1)
        if xlog:
            ax.set_xscale("log")
        if ylog:
            ax.set_yscale("log")
            ax.set_ylim(top=1)
            ylabel = log_scale_label(ylabel)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=legend_ncol, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    paths = save_figure(fig, out_dir, stem)
    plt.close(fig)
    return paths
