# 論文用の数値・表・図 (thesis_assets)

`python scripts/make_thesis_numbers.py` で results/ の CSV とレポートの集計から生成する (手で編集しない). 新しい計算はしない.

- `numbers.tex`: 数値マクロ (`\input{thesis_assets/numbers}` で読み込む). 263 個. 各マクロの直前に出典の CSV と計算方法をコメントで書いてある.
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

- 実験 0 / 0-P (理論と DES の照合): scripts/experiment_0_validation.py, experiment_0P_validation.py. 0-P は保護を入れる前のモデル. 保護ありのモデルの DES 照合は results/protect_des/ (図はなく, マクロ ExpZeroProtect* で引用する)
  - figures/experiment_0_validation.pdf
  - figures/experiment_0P_validation.pdf
- 実験 1〜4 (ベースモデル): scripts/experiment_{1_traffic,2_delayoff,3_burstiness,4_K_sensitivity}.py. ベースモデルだけの実験 (保護の有無に関係しない)
  - figures/experiment_1_traffic.pdf
  - figures/experiment_2_delayoff.pdf
  - figures/experiment_2_delayoff_strong.pdf
  - figures/experiment_3_burstiness_delta.pdf
  - figures/experiment_3_burstiness_sigma.pdf
  - figures/experiment_4_K_sensitivity_weak.pdf
  - figures/experiment_4_K_sensitivity_medium.pdf
  - figures/experiment_4_K_sensitivity_strong.pdf
- 実験 7: scripts/experiment_7_frontier.py --report
  - figures/experiment_7_C1_EW_cost.pdf
  - figures/experiment_7_C2_EW_cost.pdf
  - figures/experiment_7_C3_EW_cost.pdf
  - figures/experiment_7_C4_EW_cost.pdf
  - figures/experiment_7_C5_EW_cost.pdf
  - figures/experiment_7_C1_cost_pblock.pdf
  - figures/experiment_7_C2_cost_pblock.pdf
  - figures/experiment_7_C3_cost_pblock.pdf
  - figures/experiment_7_C4_cost_pblock.pdf
  - figures/experiment_7_C5_cost_pblock.pdf
- 実験 8: scripts/experiment_8_map.py --report
  - figures/experiment_8_A_erp.pdf
  - figures/experiment_8_A_cost.pdf
  - figures/experiment_8_B_erp.pdf
  - figures/experiment_8_B_cost.pdf
  - figures/experiment_8_B_p1_alpha.pdf
  - figures/experiment_8_p1_fire.pdf
- 実験 8 (棄却の再判定): scripts/blocking_reassessment.py
  - figures/experiment_8_A_blk_b.pdf
  - figures/experiment_8_B_blk_b.pdf
  - figures/experiment_8_A_blk_c1e-3.pdf
  - figures/experiment_8_B_blk_c1e-3.pdf
- 実験 9: scripts/experiment_9_sensitivity.py --report
  - figures/experiment_9_A_B_alpha1_r1_3.16.pdf
  - figures/experiment_9_A_A_rhoB0.8_r1_17.8.pdf
  - figures/experiment_9_A_B_alpha0.005_r1_17.8.pdf
  - figures/experiment_9_A_A_rhoB0.6_r1_3.16.pdf
  - figures/experiment_9_A_A_rhoB0.8_r1_0.1.pdf
  - figures/experiment_9_A_B_alpha10_r1_17.8.pdf
  - figures/experiment_9_B_A_ERP.pdf
  - figures/experiment_9_B_A_EW.pdf
  - figures/experiment_9_B_A_Cost.pdf
  - figures/experiment_9_B_A_P.pdf
  - figures/experiment_9_B_B_ERP.pdf
  - figures/experiment_9_B_B_EW.pdf
  - figures/experiment_9_B_B_Cost.pdf
  - figures/experiment_9_B_B_P.pdf
