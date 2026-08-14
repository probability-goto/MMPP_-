"""Predictive モデルの基本動作を確認するスクリプト.

n_target と gamma の 2 パラメータを変えて出力を比較する.
"""
import numpy as np

from mmpp_predictive.model import PredictiveModelParameters
from mmpp_predictive.generator import build_generator
from mmpp_predictive.solver import solve_stationary
from mmpp_predictive.metrics import Metrics

try:
    from scripts._mmpp_burst import build_mmpp
except ImportError:
    from _mmpp_burst import build_mmpp


def run_case(n_target, gamma, rho=0.7, delta=0.6, sigma=0.1):
    c, K, b, mu = 20, 200, 5, 1.0
    C0, C1 = build_mmpp(rho=rho, delta=delta, sigma=sigma, c=c, b=b, mu=mu)

    p = PredictiveModelParameters(
        c=c, K=K, b=b, mu=mu, alpha=0.1, beta=0.005,
        C0=C0, C1=C1,
        n_target=n_target, gamma=gamma,
    )
    Q = build_generator(p)
    pi = solve_stationary(Q)
    m = Metrics(p, pi)
    return {
        "P_block_arrival": m.arrival_blocking_probability(),
        "E[W]": m.mean_waiting_time(),
        "Cost": m.energy_cost_paper(),
        "ERP": m.erp_paper(),
    }


def main():
    print("=== Predictive モデル動作確認 ===")
    print("設定: c=20, K=200, b=5, rho=0.7, 中バースト (delta=0.6, sigma=0.1)\n")

    cases = [
        ("ベース相当 (n_target=0, gamma=1)", 0, 1.0),
        ("弱 Predictive (n_target=5, gamma=2)", 5, 2.0),
        ("中 Predictive (n_target=10, gamma=5)", 10, 5.0),
        ("強 Predictive (n_target=20, gamma=20)", 20, 20.0),
    ]

    for name, nt, ga in cases:
        result = run_case(nt, ga)
        print(f"[{name}]")
        for k, v in result.items():
            print(f"  {k:<18}: {v:.5g}")
        print()


if __name__ == "__main__":
    main()
