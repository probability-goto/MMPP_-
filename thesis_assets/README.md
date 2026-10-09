# 論文用の数値・表・図 (thesis_assets)

`python scripts/make_thesis_numbers.py` で results/ の CSV とレポートの集計から生成する (手で編集しない). 新しい計算はしない.

- `numbers.tex`: 数値マクロ (`\input{thesis_assets/numbers}` で読み込む). 194 個. 各マクロの直前に出典の CSV と計算方法をコメントで書いてある.
- `tables/*.tex`: table 環境の表 (booktabs と graphicx (幅の広い表の \resizebox) を使う. 数値マクロを参照する表は numbers.tex の後に読み込む).
- `figures/`: figures/ からのコピー (PDF と PNG).

## 論文の節と図・表の対応

**注意: 節の番号と内容は仮である.** 依頼文の「論理の流れ」の節 1〜9 がリポジトリに見つからなかったため, 実験の順に仮に割り当てた. 節の対応が決まったら scripts/make_thesis_numbers.py の SECTIONS を直して 再生成する.

| 節 (仮) | 内容 | 図 | 表 | 主なマクロ |
|---|---|---|---|---|
| 1 | 研究の動機と問題設定 | — | — | — |
| 2 | モデル (Base と Predictive, 保護) | — | — | — |
| 3 | 数値解法と検証 (GTH, DES との照合, 計算時間) | figures/experiment_0_validation.pdf<br>figures/experiment_0P_validation.pdf | — | ExpZero, Bench |
| 4 | 公平な比較: 各方策の最良の設定どうし (実験 7) | figures/experiment_7_C1_EW_cost.pdf<br>figures/experiment_7_C2_EW_cost.pdf<br>figures/experiment_7_C3_EW_cost.pdf<br>figures/experiment_7_C4_EW_cost.pdf<br>figures/experiment_7_C5_EW_cost.pdf<br>figures/experiment_7_C1_cost_pblock.pdf<br>figures/experiment_7_C2_cost_pblock.pdf<br>figures/experiment_7_C3_cost_pblock.pdf<br>figures/experiment_7_C4_cost_pblock.pdf<br>figures/experiment_7_C5_cost_pblock.pdf | tables/exp7_erp.tex<br>tables/exp7_samerel.tex | ExpSeven |
| 5 | 位相の情報に価値がある領域の地図 (実験 8) | figures/experiment_8_A_erp.pdf<br>figures/experiment_8_A_cost.pdf<br>figures/experiment_8_B_erp.pdf<br>figures/experiment_8_B_cost.pdf<br>figures/experiment_8_B_p1_alpha.pdf<br>figures/experiment_8_p1_fire.pdf | tables/exp8_mapA.tex<br>tables/exp8_mapB.tex | ExpEight |
| 6 | Predictive が優位になる仕組み (電力と位相ごとの内訳) | — | tables/exp8_mechanism.tex | ExpEightR |
| 7 | ブロッキング確率を含めた判定 (同じ信頼性での比較) | figures/experiment_8_A_blk_b.pdf<br>figures/experiment_8_B_blk_b.pdf<br>figures/experiment_8_A_blk_c1e-3.pdf<br>figures/experiment_8_B_blk_c1e-3.pdf | — | ExpSeven...SameRel, ExpEight...SameRel |
| 8 | パラメータが最適でないときの感度 (実験 9 パート A) | figures/experiment_9_A_B_alpha1_r1_3.16.pdf<br>figures/experiment_9_A_A_rhoB0.8_r1_17.8.pdf<br>figures/experiment_9_A_B_alpha0.005_r1_17.8.pdf<br>figures/experiment_9_A_A_rhoB0.6_r1_3.16.pdf<br>figures/experiment_9_A_A_rhoB0.8_r1_0.1.pdf<br>figures/experiment_9_A_B_alpha10_r1_17.8.pdf | tables/exp9_beta2.tex | ExpNineBetaTwo |
| 9 | Base の β のまま規則を ON にした場合 (実験 9 パート B) とまとめ | figures/experiment_9_B_A_ERP.pdf<br>figures/experiment_9_B_A_EW.pdf<br>figures/experiment_9_B_A_Cost.pdf<br>figures/experiment_9_B_A_P.pdf<br>figures/experiment_9_B_B_ERP.pdf<br>figures/experiment_9_B_B_EW.pdf<br>figures/experiment_9_B_B_Cost.pdf<br>figures/experiment_9_B_B_P.pdf | tables/exp9_partB.tex | ExpNine |

## 図の出典

- experiment_0*: scripts/experiment_0_validation.py, experiment_0P_validation.py (保護を入れる前のモデル).
- experiment_7_*: scripts/experiment_7_frontier.py --report
- experiment_8_{A,B}_{erp,cost}, B_p1_alpha, p1_fire: scripts/experiment_8_map.py --report
- experiment_8_*_blk_*: scripts/blocking_reassessment.py
- experiment_9_*: scripts/experiment_9_sensitivity.py --report
