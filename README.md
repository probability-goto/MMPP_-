# MMPP/M/c/SET-BATCH/delayoff 待ち行列モデル

MMPP (Markov Modulated Poisson Process) 到着を持ち、c 個のサーバー、
バッチサイズ b でのバッチサービス、セットアップ遅延、Delayoff タイムアウトを含む
待ち行列モデルの数値解析ライブラリと、その数値実験。

## モデル概要

### ベースモデル (`mmpp`)

- **状態**: (i, j, F)
  - i: アクティブサーバー数 (Busy または Delayoff/Idle)
  - j: 系内ジョブ数 (0 ≤ j ≤ K)
  - F: MMPP 環境フェーズ (0 ≤ F < D_M)
- **バッチサービス**: b 個揃うと処理開始 (フルバッチ方式), 完了率 μ
- **セットアップ**: 完了率 α, 需要 (floor(j/b)) に合わせて起動
- **Delayoff タイムアウト**: 率 β, アイドルサーバーの自動オフ
- **バッファ容量**: K (満杯時は到着ブロック)

### Predictive モデル (`mmpp_predictive`)

ベースモデルに MMPP の位相の情報を使う 2 つの規則を足したもの。状態は (i, s, j, F)
(s: セットアップ中のサーバー数)。n_target=0, γ=1 でベースモデルと一致する。

- **P1 (事前セットアップ)**: 位相が通常 (F=0) からバースト (F=1) に変わったとき,
  i+s が `n_target` になるまでセットアップを始める
- **P2 (Delayoff の加速)**: 通常期 (F=0) の Delayoff 率を β → γ·β にする (`gamma` ≥ 1)
- **保護** (`protect_presetup`, `protect_delayoff`): バースト中 (F=1) に i+s ≤ n_target なら,
  バッチ完了によるセットアップの取り消し (`protect_presetup`) と Delayoff (`protect_delayoff`) を
  起こさない. 両方 True で, P1 の発動後のバースト中は i+s ≥ n_target が保たれる
- **NoCancel** (`never_cancel_setup`): 位相を使わない対照. バッチ完了でのセットアップの取り消しを,
  位相によらず一切行わない

## ディレクトリ構成

```
MMPP_-/
├── mmpp/                  # ベースモデルの解析 (生成行列, GTH/splu ソルバー, 性能指標)
├── mmpp_predictive/       # Predictive モデルの解析
├── mmpp_sim/              # ベースモデルの DES (離散事象シミュレーション)
├── mmpp_predictive_sim/   # Predictive モデルの DES
├── mmpp_experiments/      # 数値実験のスクリプトで共通に使う部品
├── scripts/               # 数値実験のスクリプト
├── tests/                 # pytest
├── examples/              # 使用例 (basic_example.py)
├── results/               # 実験の CSV とレポート (results/gth_validation/ は GTH 移行の検証記録)
├── figures/               # 現行の図 (GTH 法 + 安定版の P_block)
├── thesis_assets/         # 論文用の数値マクロ・表・図 (scripts/make_thesis_numbers.py が生成)
├── docs/                  # モデルと解法の説明 (model_and_methods.tex)
└── archive/               # 使わないが記録として残すもの (旧比較, GTH 移行時の図・CSV・スクリプト)
```

### 解析と DES は独立に実装している

DES (`mmpp_sim`, `mmpp_predictive_sim`) は解析 (`mmpp`, `mmpp_predictive`) の正しさを検証するための
**独立な実装**で, 生成行列や状態空間の関数を import しない (サーバー数の関数なども別に書いている)。
また, Predictive (n_target=0, γ=1) がベースモデルと一致することを, 解析と DES の両方でテストしている
(`tests/test_predictive_baseline_consistency.py`, `tests/test_predictive_sim_baseline_consistency.py`)。
そのため, **解析と DES の間**で同じ計算 (サーバー数の関数など) が重複しているのは意図したもので, 共通化しない。

一方, 同じ側どうしは共通化している:

- DES: Predictive 版は, ベース版の `SimStats`, サーバー数の関数, `EventType`, `run_replications`
  (シミュレータのクラスを渡す) を使い, `SimMetrics` はベース版を継承して E[S] と電力の係数だけを上書きする。
  シミュレータ本体 (イベントの規則) は別々に書いている (両者の一致をテストで確かめるため)
- 解析: Predictive 版は, 生成行列の成分を積む `mmpp.generator.TransitionBuilder` と,
  ベースモデルのソルバー (`mmpp.solver`) を使う。遷移の規則は別々に書いている

### `mmpp_experiments/` (スクリプトの共通部品)

| モジュール | 中身 |
|---|---|
| `params` | 共通のシステム規模 `SYSTEM` (c=20, K=200, b=5, μ=1), バースト水準 `BURST_LEVELS`, 対称 2 位相 MMPP の構築 `build_mmpp` |
| `evaluate` | 1 点を GTH で解く (`solve_base`, `solve_predictive`), 指標を CSV の列名にそろえる |
| `checks` | 1 点の検査 (保存則, 流量バランス, min π ≥ 0, P_block > 0), DES のウォームアップの診断 |
| `csvio` | CSV の読み書き (1 点ごとに追記して中断から再開する), 進捗ログ |
| `frontier` | 2 指標をともに最小化する意味でのフロンティアと支配 |
| `classify` | Base と比べた判定の区分 (改善 / 同等 / 引き換え / 悪化 など) |
| `plotting` | 図の軸ラベル |

## インストール

```bash
pip install -e ".[dev,sim]"
```

新しいパッケージを足したときは `pip install -e .` をやり直す。

## 使い方

```python
from mmpp import ModelParameters, Metrics, build_generator, solve_stationary
from mmpp_experiments.params import SYSTEM, build_mmpp

# rho=0.5, 中バースト (delta=0.6, sigma=0.1) の対称 2 位相 MMPP
C0, C1 = build_mmpp(0.5, 0.6, 0.1, SYSTEM["c"], SYSTEM["b"], SYSTEM["mu"])
params = ModelParameters(C0=C0, C1=C1, alpha=0.005, beta=0.005, **SYSTEM)

Q = build_generator(params)   # 生成行列
pi = solve_stationary(Q)      # 定常分布 (既定: GTH 法)
print(Metrics(params, pi).all_metrics())
```

`examples/basic_example.py` も参照。

長い実験を Codespaces で動かし続けるときは, タイムアウトを防ぐために別の端末で次を実行しておく:

```bash
while true; do echo "[$(date)] Codespacesのタイムアウトを防止中..."; sleep 60; done
```

## テスト

```bash
pytest
pytest -m legacy    # 旧比較 (β 固定, 保護なし) のスクリプトのテスト (通常は実行しない)
```

コードを変えたあとは, 計算結果がコミット済みの `results/`, `figures/` と一致するかも確かめる:

```bash
python scripts/check_reproduction.py
```

実験 1〜4 はコマンドを実行し直して CSV と図をバイト単位で, 実験 7 以降は CSV から抜き取った行を
計算し直して, レポートは作り直して git の差分で照合する (PDF は作成日時を除く)。

## 数値実験

### バーストパラメータ δ, σ の定義

到着過程は対称 2 位相 MMPP (`mmpp_experiments/params.py` の `build_mmpp`):

- 平均到着率: λ̄ = ρ · c · b · μ
- 位相 0 (通常) の到着率: λ₀ = λ̄ (1 − δ)
- 位相 1 (バースト) の到着率: λ₁ = λ̄ (1 + δ)
- 位相遷移率: 0→1, 1→0 ともに σ (各位相の平均滞在時間 1/σ)

δ ∈ [0, 1) はバーストの**振幅** (δ=0 でポアソン到着), σ はバーストの**時定数**
(小さいほど 1 回のバーストが長く続く) を表す。定常位相確率は (1/2, 1/2) なので
δ, σ を変えても平均到着率 λ̄ は変わらない。実験 7 以降では, バースト中の負荷
ρ_B = ρ(1+δ) と, 位相の長さとセットアップ時間の比 r1 = α/σ でマスを決める。

バースト水準 (`BURST_LEVELS`): weak (δ=0.3, σ=1), medium (δ=0.6, σ=0.1), strong (δ=0.9, σ=0.01)。

### スクリプト

固定の設定はどれも c=20, K=200, b=5, μ=1 (実験 4 は K を走査)。出力先は既定のもの。

| スクリプト | 内容 | 出力 |
|---|---|---|
| `experiment_0_validation.py` / `experiment_0P_validation.py` | 解析と DES の照合 (ベース / Predictive. 0-P は保護を入れる前のモデル) | `figures/experiment_0*_validation.*` |
| `validate_protect_des.py` | 保護ありの Predictive と NoCancel の解析と DES の照合 | `results/protect_des/` |
| `experiment_1_traffic.py` | ベースモデル: ρ の走査 (α=β=0.005) | `results/experiment_1.csv`, `figures/experiment_1_traffic.*` |
| `experiment_2_delayoff.py` | ベースモデル: β の走査 (ρ=0.7, α ∈ {0.1, 1, 10}. `--strong-burst` で strong) | `results/experiment_2_{medium,strong}.csv` |
| `experiment_3_burstiness.py` | ベースモデル: δ または σ の走査 (`--sweep delta/sigma`) | `results/experiment_3_{delta,sigma}.csv` |
| `experiment_4_K_sensitivity.py` | ベースモデル: K ∈ {100, 200, 500, 1000} の感度 | `results/experiment_4_{weak,medium,strong}.csv` |
| `experiment_p1_protect.py` | 保護の有無による P1 の比較 | `results/p1_protect/` |
| `experiment_7_frontier.py` | 実験 7: 5 条件で各方策が β を最適化した公平な比較 (`--report` で図とレポート) | `results/experiment_7/`, `figures/experiment_7_*` |
| `experiment_8_map.py` | 実験 8: 位相の情報に価値がある領域の地図 (α=0.1 中心, Base と 3 系列) | `results/experiment_8/`, `figures/experiment_8_*` |
| `blocking_reassessment.py` | 実験 7・8 を同じ信頼性 (棄却を含めた基準) で判定し直す | `results/experiment_{7,8}/blocking_reassessment.md` |
| `analyze_mechanism.py` | 実験 7・8 で Predictive が優位になる仕組みの集計 | `results/experiment_8/mechanism*` |
| `experiment_9_sensitivity.py` | 実験 9: Predictive のパラメータが最適でないときの感度 (実験 8 が基準) | `results/experiment_9/`, `figures/experiment_9_*` |
| `compare_map.py` | 比較の地図: α=0.005 で Base と Predictive の 2 者を比べる | `results/compare_map/`, `figures/compare_map_improvement.*` |
| `robustness.py` | 頑健性: 比較の地図を基準にした感度 (α=0.005) | `results/robustness/`, `figures/robustness_*` |
| `make_thesis_numbers.py` | 論文用の数値マクロ・表・図を CSV から生成 (新しい計算はしない) | `thesis_assets/` |
| `benchmark_gth_vs_splu.py`, `validate_gth_mpmath*.py` | GTH 法の計算時間と精度 (mpmath 50 桁との照合) | `results/gth_validation/` |
| `replot_from_csv.py` | 実験 1〜4 (と旧比較の実験) の図を CSV から描き直す (計算はしない) | `figures/` |
| `check_reproduction.py`, `check_results_integrity.py` | コミット済みの結果との照合, CSV の健全性の検査 | — |

長い実験は 1 点ごとに CSV に追記し, 再実行すると計算済みの点を飛ばして再開する
(進捗は各出力先の `progress.log`)。

旧比較 (β 固定, 保護なしの旧モデル) のスクリプト (`experiment_{1P,2P,3P,4P}_*.py`, `experiment_5_n_target.py`,
`experiment_6_gamma.py`, `compare_experiment_*.py`, `evaluate_predictive_comparison.py`) は
論文では使わないが, 再現できるように残している。図と再生成の手順は
`archive/old_comparison_beta_fixed_no_protect/README.md` を参照。

## 数値解法の方針

**GTH 法** (Grassmann-Taksar-Heyman; O'Cinneide 1993, 1996) を既定のソルバーとして採用。

```python
pi = solve_stationary(Q)                  # 既定: GTH 法
pi = solve_stationary(Q, solver="splu")   # 従来の疎行列 LU 分解
```

- **GTH 法** (`solver="gth"`, 既定): 生成行列の非対角成分 (非負) のみを使い
  引き算を一切行わない消去法。各成分が**相対精度**で求まるため、
  P_block が 1e-15 を下回る領域でも精度が破綻しない
  (mpmath 50 桁の真値に対し相対誤差 ~1e-15 を維持)。
  帯格納 + numba により標準設定 (c=20, K=200, b=5) で 0.17 秒/点。
- **疎行列 LU 分解** (`solver="splu"`, scipy.sparse.linalg.spsolve): 従来手法。
  x_N = 1 固定で階数落ちを解消し、M 行列理論により非負性・一意性が保証される。
  ただし絶対誤差が ~1e-17 で頭打ちになるため、P_block が極小の領域では
  相対誤差が破綻する (負の確率も出うる)。
- 反復法 (GMRES 等) は本モデルの剛性 (κ ~ 10^3) と動的レンジ
  (κ_π ~ 10^10 - 10^20) では不適

検証の詳細は `results/gth_validation/VALIDATION_REPORT.md` を参照。
