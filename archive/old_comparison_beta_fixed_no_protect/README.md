# 旧比較 (β 固定, 保護なし) の図

**論文では使わない.** 再現できるように残している.

Predictive モデルに保護 (`protect_presetup` / `protect_delayoff`, 2026-10-07 追加) を
入れる前の旧モデルで, ベースモデルの β を走査の値に固定して比べた実験の図.
どれも GTH 法と安定版の P_block (`P_block_arrival_stable`) で計算している.

| ディレクトリ | 図 | 作るスクリプト | 入力 / 出力の CSV (`results/`) |
|---|---|---|---|
| `compare/` | `compare_experiment_{1..4}_vs_{1..4}P_*` | `scripts/compare_experiment_{1..4}_vs_{1..4}P.py` | ベース: `experiment_1_a0.1.csv`, `experiment_2_*.csv`, `experiment_3_*.csv`, `experiment_4_*_a0.1.csv` / Predictive: `experiment_{1P..4P}_*_nt10_g5.0.csv` |
| `predictive_1P_4P/` | `experiment_{1P..4P}_*` | `scripts/experiment_{1P_traffic,2P_delayoff,3P_burstiness,4P_K_sensitivity}.py` | `experiment_{1P..4P}_*.csv` |
| `predictive_5_6/` | `experiment_5_*`, `experiment_6_*_nt10` | `scripts/experiment_5_n_target.py`, `scripts/experiment_6_gamma.py` | `experiment_5_*.csv`, `experiment_6_*_nt10.csv` |
| `predictive_5_6/experiment_6_extended/` | `experiment_6_extended_*` | `scripts/compare_experiment_6_extended.py` | `experiment_6_*_nt10.csv`, `experiment_6_extended/*.csv` |
| `reports/` | `evaluation_report*.md` (旧比較の勝敗と改善率の集計. γ=5 と γ を最適化した 2 通り. `evaluation_report.md` と `evaluation_report_gamma5.md` は同じ設定) | `scripts/evaluate_predictive_comparison.py` | 実験 1〜4 と 1P〜4P の CSV (旧パス: リポジトリのルート) |
| `base_alpha0.1/` | `experiment_1_traffic_a0.1`, `experiment_4_K_sensitivity_*_a0.1` | `scripts/experiment_1_traffic.py --alpha 0.1`, `scripts/experiment_4_K_sensitivity.py --alpha 0.1` | `experiment_1_a0.1.csv`, `experiment_4_*_a0.1.csv` |

ベースの実験 1・4 の既定は α=0.005 に変わったが, 旧比較は Predictive に合わせて
α=0.1 の結果 (`*_a0.1`) を使う. 実験 2・3 のベースの CSV は現行の図と共通.

## 再生成

CSV は `results/` に置いたまま. スクリプトの図の既定の出力先は `figures/` なので,
描き直すときは出力先をこのディレクトリに指定する (実験スクリプトは `--out-dir`,
`compare_experiment_{1..4}_vs_*` は `--out`). 例:

    python scripts/compare_experiment_1_vs_1P.py \
        --base-csv results/experiment_1_a0.1.csv \
        --pred-csv results/experiment_1P_nt10_g5.0.csv \
        --meta "n_target=10, gamma=5.0" \
        --out archive/old_comparison_beta_fixed_no_protect/compare/compare_experiment_1_vs_1P.png

    python scripts/experiment_1P_traffic.py \
        --out-dir archive/old_comparison_beta_fixed_no_protect/predictive_1P_4P

保護の既定は `False` なので, 今の `mmpp_predictive` でも旧モデルとして計算される.
