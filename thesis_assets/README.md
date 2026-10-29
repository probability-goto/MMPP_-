# 論文用の数値・表・図 (thesis_assets)

`python scripts/make_thesis_numbers.py` で results/ の CSV とレポートの集計から生成する (手で編集しない). 新しい計算はしない.

- `numbers.tex`: 数値マクロ (`\input{thesis_assets/numbers}` で読み込む). 264 個. 各マクロの直前に出典の CSV と計算方法をコメントで書いてある.
- `tables/*.tex`: table 環境の表 (booktabs と graphicx (幅の広い表の \resizebox) を使う. 数値マクロを参照する表は numbers.tex の後に読み込む).
- `figures/`: figures/ からのコピー (PDF と PNG).

## 表

- tables/exp7_erp.tex
- tables/exp7_samerel.tex
- tables/exp8_mapA.tex
- tables/exp8_mapB.tex
- tables/exp8_mechanism.tex
- tables/exp9_beta2.tex
- tables/exp9_partB.tex
- tables/protect_control.tex

## 図と出典

- 実験 0 / 0-P (理論と DES の照合): scripts/experiment_0_validation.py, experiment_0P_validation.py --protect (保護ありのモデル). 照合の記録は results/experiment_0/. 保護ありのモデルの 5 点の DES 照合 (NoCancel を含む) は results/protect_des/ (マクロ ExpZeroProtect* で引用する)
  - figures/experiment_0_validation.pdf
  - figures/experiment_0P_validation_protect.pdf
- 実験 1〜4 (ベースモデル): scripts/experiment_{1_traffic,2_delayoff,3_burstiness,4_K_sensitivity}.py. ベースモデルだけの実験 (保護の有無に関係しない)
  - figures/experiment_1_traffic.pdf
  - figures/experiment_2_delayoff_erp.pdf
  - figures/experiment_2_delayoff_pblock.pdf
  - figures/experiment_3_burstiness_delta.pdf
  - figures/experiment_3_burstiness_r1.pdf
  - figures/experiment_4_K_sensitivity_short.pdf
  - figures/experiment_4_K_sensitivity_long.pdf
- 比較の地図: scripts/compare_map.py --report, scripts/compare_map_mechanism.py
  - figures/compare_map_improvement.pdf
  - figures/compare_map_mechanism.pdf
- 頑健性: scripts/robustness.py --report
  - figures/robustness_A_rhoB0.8_r1_17.8.pdf
  - figures/robustness_B.pdf
