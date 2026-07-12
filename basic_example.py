"""基本的な使用例.

パラメータ設定から性能指標の表示まで一連の流れを示す.
"""
import numpy as np

from mmpp import (
    ModelParameters,
    build_generator,
    solve_stationary,
    Metrics,
)


def main():
    # --- 1. パラメータ設定 ---
    # MMPP 行列 (2 位相).
    # 位相 0: 低到着率 0.5, 位相 1: 高到着率 1.5.
    # 位相遷移: 各位相から他方への遷移率 0.05.
    C0 = np.array([
        [-(0.5 + 0.05), 0.05],
        [0.05, -(1.5 + 0.05)],
    ])
    C1 = np.array([
        [0.5, 0.0],
        [0.0, 1.5],
    ])

    params = ModelParameters(
        c=10,      # サーバー数
        K=50,      # バッファ容量
        b=2,       # バッチサイズ
        mu=1.0,    # サービス率
        alpha=0.5, # セットアップ完了率
        beta=0.05, # Delayoff タイムアウト率
        C0=C0, C1=C1,
    )

    print("=" * 60)
    print("モデル設定")
    print("=" * 60)
    print(params.summary())
    print(f"位相定常分布: {params.phase_stationary}")
    print(f"平均到着率  : {params.lambda_bar:.4f}")
    print()

    # --- 2. 生成行列を構築 ---
    print("生成行列 Q を構築中...")
    Q = build_generator(params)
    print(f"  Q.shape = {Q.shape}")
    print(f"  非ゼロ要素数 = {Q.nnz}")
    print(f"  疎密度 = {Q.nnz / (params.N ** 2):.2%}")
    print()

    # --- 3. 定常分布を計算 ---
    print("定常分布を計算中...")
    pi = solve_stationary(Q, method="sparse")
    print(f"  pi.shape = {pi.shape}")
    print(f"  sum(pi) = {pi.sum():.10f}")
    print(f"  ||pi Q||_inf = {np.abs(pi @ Q).max():.2e}")
    print()

    # --- 4. 性能指標 ---
    metrics = Metrics(params, pi)
    print("=" * 60)
    print("性能指標")
    print("=" * 60)
    for key, value in metrics.all_metrics().items():
        print(f"  {key:12s} = {value:.6f}")

    # --- 5. 保存則の検証 ---
    print()
    print("=" * 60)
    print("保存則検証")
    print("=" * 60)
    server_total = (
        metrics.mean_busy()
        + metrics.mean_idle()
        + metrics.mean_setup()
        + metrics.mean_off()
    )
    print(f"サーバー数保存:")
    print(f"  E[B] + E[I] + E[S] + E[Off] = {server_total:.10f}")
    print(f"  c                            = {params.c}")
    print(f"  誤差                         = {abs(server_total - params.c):.2e}")

    flow_lhs = metrics.effective_arrival_rate()
    flow_rhs = params.b * params.mu * metrics.mean_busy()
    print(f"流量バランス (λ_eff = b*μ*E[B]):")
    print(f"  λ_eff       = {flow_lhs:.10f}")
    print(f"  b * μ * E[B] = {flow_rhs:.10f}")
    print(f"  誤差         = {abs(flow_lhs - flow_rhs):.2e}")


if __name__ == "__main__":
    main()
