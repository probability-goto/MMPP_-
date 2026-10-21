"""1 点の検査 (保存則, 流量バランス, 非負性) と DES のウォームアップの診断."""
import warnings
from typing import Dict, List, Tuple

# 検査のしきい値 (保存則は絶対誤差, 流量バランスは相対誤差)
CHECK_TOL = 1e-9


def conservation_and_flow_balance(
    c: int, b: int, mu: float,
    mean_busy: float, mean_idle: float, mean_setup: float, mean_off: float,
    lambda_eff: float,
) -> Tuple[float, float]:
    """保存則 E[B]+E[I]+E[S]+E[Off]=c と 流量バランス lambda_eff=b*mu*E[B] の誤差 (どちらも絶対誤差)."""
    conservation_error = abs(mean_busy + mean_idle + mean_setup + mean_off - c)
    flow_balance_error = abs(lambda_eff - b * mu * mean_busy)
    return conservation_error, flow_balance_error


def point_checks(params, theory) -> Dict[str, object]:
    """ベースモデルの 1 点の検査 (保存則, 流量バランス, min pi >= 0, P_block > 0) の結果を返す.

    Returns:
        min_pi, cons_err (絶対誤差), flow_rel_err (相対誤差) と,
        違反した検査名を ";" でつないだ check ("ok" なら違反なし).
    """
    lambda_eff = theory.effective_arrival_rate()
    cons_err = abs(
        theory.mean_busy() + theory.mean_idle() + theory.mean_setup()
        + theory.mean_off() - params.c
    )
    flow_rel_err = abs(lambda_eff - params.b * params.mu * theory.mean_busy()) / abs(lambda_eff)
    min_pi = float(theory.pi.min())
    violations = []
    if not cons_err <= CHECK_TOL:
        violations.append("conservation")
    if not flow_rel_err <= CHECK_TOL:
        violations.append("flow_balance")
    if not min_pi >= 0.0:
        violations.append("min_pi")
    if not theory.arrival_blocking_probability_stable() > 0.0:
        violations.append("P_block")
    return dict(
        min_pi=min_pi, cons_err=cons_err, flow_rel_err=flow_rel_err,
        check=";".join(violations) if violations else "ok",
    )


def row_check(c: int, b: int, mu: float, E_B, E_I, E_S, E_off, lambda_eff, P_arr,
              min_pi) -> str:
    """1 点の検査. 違反を ";" でつないだ文字列 (違反がなければ "ok") を返す.

    保存則 (絶対誤差), 流量バランス (相対誤差), min pi >= 0, P_block^arrival > 0.
    """
    out = []
    total = E_B + E_I + E_S + E_off
    if abs(total - c) > CHECK_TOL:
        out.append(f"conservation:{total - c:.3e}")
    flow = b * mu * E_B
    if abs(lambda_eff - flow) > CHECK_TOL * abs(lambda_eff):
        out.append(f"flow:{(lambda_eff - flow) / lambda_eff:.3e}")
    if min_pi < 0:
        out.append(f"min_pi:{min_pi:.3e}")
    if not P_arr > 0:
        out.append("P_block_nonpositive")
    return ";".join(out) if out else "ok"


def check_warmup_duration(params, replications: List) -> None:
    """ウォームアップ実時間が遅い時定数 1/beta に対して十分かを診断する."""
    warmup_duration = replications[0].warmup_duration
    warmup_ratio = warmup_duration * params.beta
    if warmup_ratio < 5.0:
        warnings.warn(
            f"ウォームアップが遅い時定数 1/beta = {1 / params.beta:.2f} に対して "
            f"{warmup_ratio:.2f} 倍しかありません. warmup_events を増やすことを検討してください.",
            RuntimeWarning,
            stacklevel=2,
        )
