# 保護ありの方策と対照の DES 照合 (results/protect_des/protect_des.csv から生成)

固定条件: c=20, K=200, b=5, mu=1.0, alpha=0.1, beta=0.005, delta=0.6, sigma=0.1, 20 レプリケーション, ウォームアップ 100000 事象, 計測 1000000 事象. both: n_target=10, gamma=5, protect_presetup=protect_delayoff=True. nocancel: n_target=0, gamma=1, never_cancel_setup=True.

| 方策 | ρ | 6 指標の CI 内 | CI 外の指標 | P_block^arr | cancel | check |
|---|---|---|---|---|---|---|
| both | 0.3000 | 6/6 | - | 内 | 内 | ok |
| both | 0.4625 | 6/6 | - | 内 | 内 | ok |
| both | 0.6250 | 6/6 | - | 内 | 内 | ok |
| both | 0.7875 | 5/6 | E_W | 内 | 内 | ok |
| both | 0.9500 | 6/6 | - | 内 | 内 | ok |
| nocancel | 0.3000 | 4/6 | P_block, Cost | 外 | 内 | ok |
| nocancel | 0.4625 | 6/6 | - | 内 | 内 | ok |
| nocancel | 0.6250 | 6/6 | - | 内 | 内 | ok |
| nocancel | 0.7875 | 6/6 | - | 内 | 内 | ok |
| nocancel | 0.9500 | 6/6 | - | 内 | 内 | ok |

## CI 外の指標の詳細

| 方策 | ρ | 指標 | 理論 | DES 平均 | CI 下限 | CI 上限 | 相対差 |
|---|---|---|---|---|---|---|---|
| both | 0.7875 | E_W | 1.69894 | 1.69343 | 1.68898 | 1.69789 | -3.24e-03 |
| nocancel | 0.3000 | P_block | 4.94282e-08 | 0 | 0 | 0 | -1.00e+00 |
| nocancel | 0.3000 | Cost | 4.41282 | 4.39595 | 4.3839 | 4.40801 | -3.82e-03 |
| nocancel | 0.3000 | P_block_arrival | 7.82912e-08 | 0 | 0 | 0 | -1.00e+00 |