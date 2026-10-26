# archive/

**論文・スライドには使わない。** 記録として残しているファイル置き場。
現行の図は `figures/`, 現行の結果は `results/` にある (どちらも GTH 法 + 安定版の P_block で計算).

| ディレクトリ | 中身 | 旧パス |
|---|---|---|
| `gth_migration_figures_subtraction_pblock/` | GTH 法を既定ソルバーにした直後 (2026-09-25) に再計算した実験 0〜6 の図. P_block は引き算版の式 | `figures_gth/experiment_*` |
| `gth_migration_results_subtraction_pblock/` | 上の図の元になった CSV (実験 1〜6, 1P〜4P) | `results_gth/*.csv` |
| `splu_vs_gth_comparison_figures/` | splu と GTH の精度比較図 (`scripts/make_gth_vs_splu_figures.py`) | `figures_gth/comparison/` |
| `old_comparison_beta_fixed_no_protect/` | 旧比較 (β 固定, 保護なしの旧モデル) の図. 論文では使わないが再現できるように残す. CSV は `results/` のまま. 詳細と再生成の手順は中の README | `figures/compare_experiment_*`, `figures/experiment_{1P..4P,5,6}_*`, `figures/*_a0.1.*` |
| `pre_gth_splu_figures/` | GTH 導入より前 (splu 時代, 2026-07〜08) の図. `theory_vs_sim_*` は生成スクリプトが残っていない. `experiment_6_extended/experiment_6_*_nt10` は `figures/experiment_6_*_nt10` の古い版 | `figures/theory_vs_sim_*`, `figures/experiment_6_extended/experiment_6_*_nt10.*` |
| `figures_not_in_thesis/` | 論文で使わない図 (2026-10-25 に `figures/` から移した). 実験 7〜9 (比較の地図と頑健性の前の版. CSV とレポートは `results/experiment_{7,8,9}/` のまま. `scripts/experiment_{7_frontier,8_map,9_sensitivity}.py --report`, `scripts/blocking_reassessment.py` で `figures/` に作り直せる), 保護を入れる前のモデルの実験 0-P (`experiment_0P_validation`. P_block は時間平均の P(j=K)), 頑健性のパート A の改善率の上位 1 以外のマス (`robustness_A_*`. 数値は `results/robustness/report.md`), 実験 2 のフェーズ別の図 (`experiment_2_phasewise`. 論文では表で示す) | `figures/` |
| `old_design_experiments_1_4/` | 2026-10-25 に設計し直す前の実験 1〜4 の図と CSV (弱・中・強の 3 バースト水準, β=0.005, 実験 2・3 は α ∈ {0.1, 1, 10}). コミット `9abff86` の時点のもの. 今のスクリプトでは作り直せない (コミット `9abff86` のスクリプトで作れる). 見直しの理由は `paper/review_logic_flow.md` | `figures/experiment_{1..4}_*`, `results/experiment_{1..4}*.csv` |

GTH 移行の検証レポート・ベンチマーク・計測時間は論文の数値の出典なので, アーカイブではなく
`results/gth_validation/` に置いている (旧パス `results_gth/`).

## 「引き算版の P_block」について

`*_subtraction_pblock` の図・CSV は, 到着平均ブロッキング確率を安定版の式に一本化した
コミット `b6afb70` より前に, 引き算版の式

    P_block^arrival = 1 - λ_eff / λ̄

で描いた. 引き算版は P_block^arrival が 1e-13 を下回る領域で桁落ちする
(負値や 1e-15 付近での頭打ちが出る) ため, 低 ρ・大 K の領域の値は正しくない.
現行の図 (`figures/`) は引き算を含まない式

    P_block^arrival = (1/λ̄) Σ π(·, K, F) λ_F

(CSV 列 `P_block_arrival_stable`) で描いている.

`splu_vs_gth_comparison_figures/` は引き算版の桁落ちを示すために, CSV の旧列
`P_block_arrival` を意図的に描いている.

## 再生成について

GTH 移行用のスクリプトは `gth_migration_scripts/` に移した (旧パス `scripts/`):
`run_*gth_regen*.sh`, `regen_resume.sh`, `finalize_gth_results.sh`, `show_progress.sh`,
`compare_gth_vs_splu_results.py`, `make_gth_vs_splu_figures.py`, `validate_gth_v6_base.py`
(V6: 実験 0〜4 の全点で GTH と splu を比較), `test_predictive_basic.py` (Predictive の動作確認の使い捨て).
旧パス `results_gth/`, `figures_gth/`, `scripts/` を前提にしたままで, 当時の `results/` (splu) も
もう残っていないため, そのままでは再現できない.

今の solver でも実行できる検証は `scripts/` に残している: `validate_gth_mpmath.py`,
`validate_gth_mpmath_predictive.py` (mpmath 50 桁との照合), `benchmark_gth_vs_splu.py`
(論文の数値の出典 `results/gth_validation/benchmark.md` を作る).
