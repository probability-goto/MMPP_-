# figures_gth/

**論文・スライドには `figures/` の図を使うこと。このディレクトリの図は使わない。**

ここにあるのは、到着平均ブロッキング確率を安定版の式に一本化した
コミット `b6afb70` より前に、引き算版の式

    P_block^arrival = 1 - λ_eff / λ̄

で描いた図である。引き算版は P_block^arrival が 1e-13 を下回る領域で桁落ちする
(負値や 1e-15 付近での頭打ちが出る) ため、低 ρ・大 K の領域の値は正しくない。
現在の図 (`figures/`) は、引き算を含まない式

    P_block^arrival = (1/λ̄) Σ π(·, K, F) λ_F

(CSV 列 `P_block_arrival_stable`) で描いている。

- `experiment_*.{png,pdf}`: GTH 法を既定ソルバーにした直後 (2026-09-25) の再計算時点の図。
  記録として残しているだけで、描き直していない。
- `comparison/`: splu と GTH の精度比較図 (`scripts/make_gth_vs_splu_figures.py`)。
  引き算版の桁落ちを示すために、既存 CSV の旧列 `P_block_arrival` を意図的に描いている。
