# 論文用の数値・表・図 (thesis_assets)

`python scripts/make_thesis_numbers.py` で results/ の CSV とレポートの集計から生成する (手で編集しない). 新しい計算はしない.

- `numbers.tex`: 数値マクロ (`\input{thesis_assets/numbers}` で読み込む). 269 個. 各マクロの直前に出典の CSV と計算方法をコメントで書いてある.
- `tables/*.tex`: table 環境の表 (booktabs と graphicx (幅の広い表の \resizebox) を使う. 数値マクロを参照する表は numbers.tex の後に読み込む).
- `figures/`: figures/ からのコピー (PDF と PNG).

## 論文の節と図・表の対応

論文の論理の流れ (節 1〜9) に合わせた. 図は figures/ の PDF (同名の PNG もある).

| 節 | 問い | 使う実験 | 結論 | 図 | 表 | 主なマクロ |
|---|---|---|---|---|---|---|
| 1 | 解析は正しいか | 実験 0 / 0-P (と保護ありの DES 照合) | 理論と DES が一致 | figures/experiment_0_validation.pdf<br>figures/experiment_0P_validation.pdf | — | ExpZero*, Bench* |
| 2 | ベースモデルの特性 | 実験 1〜4 | バースト性と有限バッファの影響. ρ_B > 1 ではどの方策でも棄却を防げない | figures/experiment_1_traffic.pdf<br>figures/experiment_2_delayoff.pdf<br>figures/experiment_2_delayoff_strong.pdf<br>figures/experiment_3_burstiness_delta.pdf<br>figures/experiment_3_burstiness_sigma.pdf<br>figures/experiment_4_K_sensitivity_weak.pdf<br>figures/experiment_4_K_sensitivity_medium.pdf<br>figures/experiment_4_K_sensitivity_strong.pdf | — | ExpSevenCTwoBasePBlock, ExpSevenCTwoEpsTwoFeasibleCount, ExpEightBaseInfeasibleEpsTwoCount |
| 3 | 評価の基準 | ― | 有限バッファでは ERP が棄却を含まない. 同じ信頼性での ERP を基準にする | — | tables/exp7_samerel.tex | ExpSeven*PRatio, ExpEight*PRatioMax, ExpEightCategoryChangedCount |
| 4 | 各規則の働き | 保護の比較, NoCancel | 保護が不可欠. 価値は位相で時期を選ぶことから来る | — | tables/protect_control.tex | ProtectExample*, ProtectNoCancel*, ExpSeven*NoCancel*, ExpSevenNoCancelWorse* |
| 5 | 公平な比較 | 実験 7 | 典型的な条件では差は小さい. β を固定した比較は過大評価だった | figures/experiment_7_C1_EW_cost.pdf<br>figures/experiment_7_C2_EW_cost.pdf<br>figures/experiment_7_C3_EW_cost.pdf<br>figures/experiment_7_C4_EW_cost.pdf<br>figures/experiment_7_C5_EW_cost.pdf<br>figures/experiment_7_C1_cost_pblock.pdf<br>figures/experiment_7_C2_cost_pblock.pdf<br>figures/experiment_7_C3_cost_pblock.pdf<br>figures/experiment_7_C4_cost_pblock.pdf<br>figures/experiment_7_C5_cost_pblock.pdf<br>figures/compare_experiment_1_vs_1P.pdf<br>figures/compare_experiment_3_vs_3P_delta.pdf | tables/exp7_erp.tex<br>tables/exp7_samerel.tex | ExpSeven*ERPMin, ExpSeven*ERPRel, ExpSevenMax*, ExpOldBetaFixed* |
| 6 | どこで優位か | 実験 8 + 再判定 | 中心の領域で 2〜4%. 途中の領域では棄却との引き換え | figures/experiment_8_A_erp.pdf<br>figures/experiment_8_A_cost.pdf<br>figures/experiment_8_B_erp.pdf<br>figures/experiment_8_B_cost.pdf<br>figures/experiment_8_B_p1_alpha.pdf<br>figures/experiment_8_A_blk_b.pdf<br>figures/experiment_8_B_blk_b.pdf<br>figures/experiment_8_A_blk_c1e-3.pdf<br>figures/experiment_8_B_blk_c1e-3.pdf | tables/exp8_mapA.tex<br>tables/exp8_mapB.tex | ExpEightCenter*, ExpEightCore*, ExpEightMaxImprovement*, ExpEight*SameRel*, ExpEightAlphaSpearman* |
| 7 | なぜ優位か | 仕組みの集計 | 通常期にサーバーを早く切れる | figures/experiment_8_p1_fire.pdf | tables/exp8_mechanism.tex | ExpEightR*, ExpEightFire*, Mech* |
| 8 | どこでベースモデルが優位か | 実験 9 | 価値のない領域, P2 を β の調整なしで使う場合, n_target が大きすぎる場合. P1 は追加しても棄却を増やさない | figures/experiment_9_A_B_alpha1_r1_3.16.pdf<br>figures/experiment_9_A_A_rhoB0.8_r1_17.8.pdf<br>figures/experiment_9_A_B_alpha0.005_r1_17.8.pdf<br>figures/experiment_9_A_A_rhoB0.6_r1_3.16.pdf<br>figures/experiment_9_A_A_rhoB0.8_r1_0.1.pdf<br>figures/experiment_9_A_B_alpha10_r1_17.8.pdf<br>figures/experiment_9_B_A_ERP.pdf<br>figures/experiment_9_B_A_EW.pdf<br>figures/experiment_9_B_A_Cost.pdf<br>figures/experiment_9_B_A_P.pdf<br>figures/experiment_9_B_B_ERP.pdf<br>figures/experiment_9_B_B_EW.pdf<br>figures/experiment_9_B_B_Cost.pdf<br>figures/experiment_9_B_B_P.pdf | tables/exp9_partB.tex<br>tables/exp9_beta2.tex | ExpNine* |
| 9 | 結論と限界 | ― | 位相の観測の仮定, 反応的な起動の理想化, 指数分布のセットアップ時間 | — | — | — |

## 図の出典と注意

- experiment_0*: scripts/experiment_0_validation.py, experiment_0P_validation.py. 0-P は保護を入れる前のモデル. 保護ありのモデルの DES 照合は results/protect_des/ (図はなく, マクロ ExpZeroProtect* で引用する).
- experiment_1〜4_*: ベースモデルだけの実験 (保護の有無に関係しない).
- compare_experiment_*: β を固定した旧比較 (保護なしの旧モデル). 節 5 で 「β を固定した比較は過大評価だった」ことを示すために使う.
- experiment_7_*: scripts/experiment_7_frontier.py --report
- experiment_8_{A,B}_{erp,cost}, B_p1_alpha, p1_fire: scripts/experiment_8_map.py --report
- experiment_8_*_blk_*: scripts/blocking_reassessment.py
- experiment_9_*: scripts/experiment_9_sensitivity.py --report
