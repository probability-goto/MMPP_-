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

# パラメータ設定
C0 = np.array([[-1.05, 0.05], [0.05, -1.05]])
C1 = np.array([[1.0, 0.0], [0.0, 1.0]])
params = ModelParameters(
    c=10, K=50, b=2,
    mu=1.0, alpha=0.5, beta=0.05,
    C0=C0, C1=C1
)

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

## 数値解法の方針

- **疎行列 LU 分解** (scipy.sparse.linalg.spsolve) を主要ソルバーとして採用
- 生成行列の階数落ちは x_N = 1 固定により解消
- M 行列理論により解の非負性・一意性が保証される
- 反復法 (GMRES 等) は本モデルの剛性 (κ ~ 10^3) と動的レンジ
  (κ_π ~ 10^10 - 10^20) では不適
