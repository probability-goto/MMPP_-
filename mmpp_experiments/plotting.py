"""論文の図で共通に使う軸の書き方と, 4 指標の図.

縦軸はすべて線形目盛にする. 対数目盛では縦の距離が倍率を表し, 小さい値どうしの差が大きく,
大きい値どうしの差が小さく見えるので, 差 (量) として読み誤りやすい. 横軸に走査パラメータ
(beta, r1, gamma など) を対数で等間隔にとる図だけ, 横軸を対数目盛にする.
"""
import os
from typing import Dict, List, Optional, Sequence, Tuple

# (CSV の列名, 軸ラベル)
METRIC_PANELS: List[Tuple[str, str]] = [
    ("P_block_arrival_stable", r"$P_{\mathrm{block}}$"),
    ("E_W", r"$E[W]$"),
    ("Cost", "Cost"),
    ("ERP", "ERP"),
]


def save_figure(fig, out_dir: str, stem: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, f"{stem}.pdf")
    png = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(pdf, dpi=100)
    fig.savefig(png, dpi=150)
    print(f"図を保存しました: {pdf}, {png}")
    return [pdf, png]


def ylim_from_zero(top: float, margin: float = 0.04) -> Tuple[float, float]:
    """0 から top までを描く縦軸の範囲. 0 付近の線が軸の枠に重ならないよう, 下限を 0 より少し下げる.

    下の余白 (0 より下の部分) が縦軸の全体の margin の割合になるようにする.
    """
    return (-margin * top / (1 - margin), top)


def set_log_xaxis(ax) -> None:
    """横軸を対数目盛にし, 目盛のラベルを 1 桁ごとに付ける (既定では桁が多いと 2 桁ごとになる)."""
    from matplotlib.ticker import LogLocator, NullFormatter

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=30))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=[k / 10 for k in range(2, 10)],
                                          numticks=30))
    ax.xaxis.set_minor_formatter(NullFormatter())


def plot_four_metrics(series: Sequence[Tuple[str, Sequence[dict], dict]], xkey: str,
                      xlabel: str, out_dir: str, stem: str, xlog: bool = False,
                      vline: Optional[float] = None, legend_ncol: int = 2,
                      ylims: Optional[Dict[str, Tuple[float, float]]] = None) -> List[str]:
    """4 指標 (P_block, E[W], Cost, ERP) を 2x2 に描く.

    series は (凡例, 行のリスト, plot に渡す書式) の並び. 行は xkey の昇順に並べ替えて描く.
    vline を与えると, その x に点線を引く (rho_B = 1 の位置など).
    ylims は {CSV の列名: (下限, 上限)}. 与えた指標は縦軸の範囲をその値に固定する
    (並べて見比べる図どうしで範囲をそろえるため).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ylims = ylims or {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (key, ylabel) in zip(axes.flat, METRIC_PANELS):
        for label, rows, style in series:
            rows = sorted(rows, key=lambda r: float(r[xkey]))
            ax.plot([float(r[xkey]) for r in rows], [float(r[key]) for r in rows],
                    marker="o", markersize=3, label=label if ax is axes.flat[0] else None,
                    **style)
        if vline is not None:
            ax.axvline(vline, color="0.5", linestyle=":", linewidth=1)
        if xlog:
            set_log_xaxis(ax)
        if key in ylims:
            ax.set_ylim(*ylims[key])
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
