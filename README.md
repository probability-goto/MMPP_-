# MMPP/M/c/SET-BATCH/delayoff 待ち行列モデル

MMPP (Markov Modulated Poisson Process) 到着を持ち、c 個のサーバー、
バッチサイズ b でのバッチサービス、セットアップ遅延、Delayoff タイムアウトを含む
待ち行列モデルの数値解析ライブラリ。

## モデル概要

- **状態**: (i, j, F)
  - i: アクティブサーバー数 (Busy または Delayoff/Idle)
  - j: 系内ジョブ数 (0 ≤ j ≤ K)
  - F: MMPP 環境フェーズ (0 ≤ F < D_M)
- **バッチサービス**: b 個揃うと処理開始 (フルバッチ方式), 完了率 μ
- **セットアップ**: 完了率 α, 需要 (floor(j/b)) に合わせて起動
- **Delayoff タイムアウト**: 率 β, アイドルサーバーの自動オフ
- **バッファ容量**: K (満杯時は到着ブロック)

## ディレクトリ構成

```
MMPP_-/
├── README.md
├── requirements.txt
├── pyproject.toml
├── mmpp/
│   ├── __init__.py
│   ├── model.py         # モデルパラメータ (ModelParameters)
│   ├── state_space.py   # 状態空間のインデックス変換 (StateSpace)
│   ├── generator.py     # 生成行列 Q の構築 (build_generator)
│   ├── solver.py        # 定常分布ソルバー (solve_stationary)
│   └── metrics.py       # 性能指標 (Metrics)
├── tests/
│   ├── conftest.py
│   ├── test_model.py
│   ├── test_state_space.py
│   ├── test_generator.py
│   ├── test_solver.py
│   ├── test_metrics.py
│   └── test_integration.py
```

## インストール

```bash
pip install -e .
```

## 使い方

```python
import numpy as np
from mmpp import ModelParameters, build_generator, solve_stationary, Metrics


# 生成行列 Q を構築
Q = build_generator(params)

# 定常分布を求める
pi = solve_stationary(Q)

# 性能指標を計算
metrics = Metrics(params, pi)
print(metrics.all_metrics())
```

## テスト実行

```bash
pytest tests/
```

## 数値実験の実行

### バーストパラメータ δ, σ の定義

全実験の到着過程は対称 2 位相 MMPP (`scripts/_mmpp_burst.py` の `build_mmpp`):

- 平均到着率: λ̄ = ρ · c · b · μ
- 位相 0 (通常) の到着率: λ₀ = λ̄ (1 − δ)
- 位相 1 (バースト) の到着率: λ₁ = λ̄ (1 + δ)
- 位相遷移率: 0→1, 1→0 ともに σ (各位相の平均滞在時間 1/σ)

δ ∈ [0, 1) はバーストの**振幅** (δ=0 でポアソン到着), σ はバーストの**時定数**
(小さいほど 1 回のバーストが長く続く) を表す。定常位相確率は (1/2, 1/2) なので
δ, σ を変えても平均到着率 λ̄ は変わらない。

### 共通パラメータ

```python
BASELINE = dict(c=20, K=200, b=5, mu=1.0, alpha=0.1, beta=0.005)
BURST_LEVELS = {            # (delta, sigma)
    "weak":   (0.3, 1.0),
    "medium": (0.6, 0.1),
    "strong": (0.9, 0.01),
}
ALPHA_LEVELS = [0.1, 1.0, 10.0]                          # 実験 2, 3, 5, 6 で走査
ALPHA_GAMMA_MAP = {0.1: 1.0, 1.0: 100.0, 10.0: 1000.0}   # --alpha-gamma-map 指定時の alpha 別 gamma
```

Predictive 拡張 (P 付き) の追加パラメータ:
`n_target` = バースト予測時 (位相 0→1) の目標稼働サーバー数 (0 ≤ n_target ≤ c),
`gamma` = 通常位相 (F=0) での Delayoff 加速係数 (beta → gamma·beta, gamma ≥ 1)。
既定値は `n_target=10, gamma=5.0`。

### 実験 1 / 1-P: トラフィック強度 ρ の走査

```python
params = BASELINE                         # c=20, K=200, b=5, mu=1.0, alpha=0.1, beta=0.005
bursts = ["weak", "medium", "strong"]
rho    = np.linspace(0.1, 0.95, 15)
# 1-P: n_target=10, gamma=5.0 (既定) / gamma=1.0
```

```bash
python scripts/experiment_1_traffic.py                 # -> results/experiment_1.csv
python scripts/experiment_1P_traffic.py                # -> results/experiment_1P_nt10_g5.0.csv
python scripts/experiment_1P_traffic.py --gamma 1.0    # -> results/experiment_1P_nt10_g1.0.csv
```

### 実験 2 / 2-P: Delayoff 率 β の走査

```python
params = dict(c=20, K=200, b=5, mu=1.0)
rho    = 0.7
alpha  = [0.1, 1.0, 10.0]
beta   = np.logspace(-2, 2, 15)           # 1e-2 .. 1e2
burst  = "medium"  # (delta=0.6, sigma=0.1), --strong-burst で "strong" (delta=0.9, sigma=0.01)
# 2-P: n_target=10, gamma=5.0 (既定) / --alpha-gamma-map で alpha 別 gamma
```

```bash
python scripts/experiment_2_delayoff.py                    # -> results/experiment_2_medium.csv
python scripts/experiment_2_delayoff.py --strong-burst     # -> results/experiment_2_strong.csv
python scripts/experiment_2P_delayoff.py                   # -> results/experiment_2P_medium_nt10_g5.0.csv
python scripts/experiment_2P_delayoff.py --alpha-gamma-map "0.1:1.0,1.0:100.0,10.0:1000.0"   # -> ..._medium_nt10_gmap.csv
python scripts/experiment_2P_delayoff.py --strong-burst    # -> results/experiment_2P_strong_nt10_g5.0.csv
python scripts/experiment_2P_delayoff.py --strong-burst --alpha-gamma-map "0.1:1.0,1.0:100.0,10.0:1000.0"   # -> ..._strong_nt10_gmap.csv
```

### 実験 3 / 3-P: バースト性 (δ, σ) の走査

```python
params = BASELINE                         # beta=0.005 固定
rho    = 0.7
alpha  = [0.1, 1.0, 10.0]
# --sweep delta (3-A):
delta  = np.linspace(0.0, 0.95, 15);  sigma = 0.1
# --sweep sigma (3-B):
sigma  = np.logspace(-3, 1, 15);      delta = 0.6     # 1e-3 .. 1e1
# 3-P: n_target=10, gamma=5.0 (既定) / --alpha-gamma-map で alpha 別 gamma
```

```bash
python scripts/experiment_3_burstiness.py --sweep delta    # -> results/experiment_3_delta.csv
python scripts/experiment_3_burstiness.py --sweep sigma    # -> results/experiment_3_sigma.csv
python scripts/experiment_3P_burstiness.py --sweep delta   # -> results/experiment_3P_delta_nt10_g5.0.csv
python scripts/experiment_3P_burstiness.py --sweep delta --alpha-gamma-map "0.1:1.0,1.0:100.0,10.0:1000.0"   # -> ..._delta_nt10_gmap.csv
python scripts/experiment_3P_burstiness.py --sweep sigma   # -> results/experiment_3P_sigma_nt10_g5.0.csv
python scripts/experiment_3P_burstiness.py --sweep sigma --alpha-gamma-map "0.1:1.0,1.0:100.0,10.0:1000.0"   # -> ..._sigma_nt10_gmap.csv
```

### 実験 4 / 4-P: バッファ容量 K の感度

```python
params = dict(c=20, b=5, mu=1.0, alpha=0.1, beta=0.005)
K      = [100, 200, 500, 1000]
bursts = ["weak", "medium", "strong"]
rho    = np.linspace(0.1, 0.95, 15)
# 4-P: n_target=10, gamma=5.0 (既定) / gamma=1.0
```

```bash
python scripts/experiment_4_K_sensitivity.py --burst all                 # -> results/experiment_4_{weak,medium,strong}.csv
python scripts/experiment_4P_K_sensitivity.py --burst all                # -> results/experiment_4P_{burst}_nt10_g5.0.csv
python scripts/experiment_4P_K_sensitivity.py --burst all --gamma 1.0    # -> results/experiment_4P_{burst}_nt10_g1.0.csv
```

### 実験 5: n_target の感度 (Predictive のみ)

```python
params   = BASELINE
rho      = 0.7
alpha    = [0.1, 1.0, 10.0]
bursts   = ["weak", "medium", "strong"]
gamma    = 5.0
n_target = [0, 1, 3, 4, 6, 7, 9, 10, 11, 13, 14, 16, 17, 19, 20]
```

```bash
python scripts/experiment_5_n_target.py --burst all    # -> results/experiment_5_{burst}_g5.0.csv
```

### 実験 6: gamma の感度 (Predictive のみ)

```python
params   = BASELINE
rho      = 0.7
alpha    = [0.1, 1.0, 10.0]
bursts   = ["weak", "medium", "strong"]
n_target = 10
gamma    = np.logspace(0, np.log10(20.0), 15)   # 1 .. 20 の対数 15 点のうち 5.54 を 5.0 に置換
# = [1.0, 1.239, 1.534, 1.900, 2.354, 2.915, 3.611, 4.472, 5.0, 6.861, 8.498, 10.525, 13.037, 16.147, 20.0]
```

```bash
python scripts/experiment_6_gamma.py --burst all       # -> results/experiment_6_{burst}_nt10.csv
```

### ベース vs Predictive の比較図

```bash
python scripts/compare_experiment_1_vs_1P.py \
  --base-csv results/experiment_1.csv \
  --pred-csv results/experiment_1P_nt10_g1.0.csv
python scripts/compare_experiment_2_vs_2P.py --burst-name medium \
  --base-csv results/experiment_2_medium.csv \
  --pred-csv results/experiment_2P_medium_nt10_gmap.csv
python scripts/compare_experiment_2_vs_2P.py --burst-name strong \
  --base-csv results/experiment_2_strong.csv \
  --pred-csv results/experiment_2P_strong_nt10_gmap.csv
python scripts/compare_experiment_3_vs_3P.py --sweep delta \
  --base-csv results/experiment_3_delta.csv \
  --pred-csv results/experiment_3P_delta_nt10_gmap.csv
python scripts/compare_experiment_3_vs_3P.py --sweep sigma \
  --base-csv results/experiment_3_sigma.csv \
  --pred-csv results/experiment_3P_sigma_nt10_gmap.csv
python scripts/compare_experiment_4_vs_4P.py --burst weak \
  --base-csv results/experiment_4_weak.csv \
  --pred-csv results/experiment_4P_weak_nt10_g1.0.csv
python scripts/compare_experiment_4_vs_4P.py --burst medium \
  --base-csv results/experiment_4_medium.csv \
  --pred-csv results/experiment_4P_medium_nt10_g1.0.csv
python scripts/compare_experiment_4_vs_4P.py --burst strong \
  --base-csv results/experiment_4_strong.csv \
  --pred-csv results/experiment_4P_strong_nt10_g1.0.csv
```

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

検証の詳細は `results_gth/VALIDATION_REPORT.md` を参照。
