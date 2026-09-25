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

実験 0-4 のスクリプトは `scripts/` にある. 各スクリプトは `--quick` で
走査点数・イベント数を絞った軽量実行ができる (開発中の動作確認用).
図は既定で `figures/` ディレクトリに保存される.

```bash
# 実験 0: 理論解析と DES シミュレーションの整合性検証 (中バースト水準, rho スイープ)
python scripts/experiment_0_validation.py
python scripts/experiment_0_validation.py --quick

# 実験 1: トラフィック強度 rho に対する応答 (弱/中/強バースト水準を並列比較)
python scripts/experiment_1_traffic.py
python scripts/experiment_1_traffic.py --quick

# 実験 2: delayoff 率 beta に対する応答 (alpha 3 水準)
python scripts/experiment_2_delayoff.py                  # 中バースト (メイン)
python scripts/experiment_2_delayoff.py --strong-burst   # 強バースト (補助)
python scripts/experiment_2_delayoff.py --quick

# 実験 3: バースト性パラメータ (delta, sigma) 自体の走査
python scripts/experiment_3_burstiness.py --sweep delta  # 実験 3-A: 振幅走査
python scripts/experiment_3_burstiness.py --sweep sigma  # 実験 3-B: 時定数走査
python scripts/experiment_3_burstiness.py --sweep delta --quick

# 実験 4: バッファ容量 K に対する感度分析 (K=100,200,500,1000 x 弱/中/強バースト)
python scripts/experiment_4_K_sensitivity.py --burst all     # 3 水準まとめて実行
python scripts/experiment_4_K_sensitivity.py --burst medium  # 水準を 1 つだけ指定
python scripts/experiment_4_K_sensitivity.py --burst all --quick
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
