"""ベースモデル / Predictive の 1 点を解き, 指標を標準の列名にそろえる.

ベースモデルの Metrics は all_metrics() の辞書 ("E[j]" など) で, Predictive の Metrics は
属性 (E_N など) とメソッドで指標を返す. CSV に書くときの列名 (METRIC_FIELDS) にそろえる.
定常分布は GTH 法で解く (戻り値の Metrics の .pi).
"""
from typing import Dict

from mmpp import Metrics as BaseMetrics
from mmpp import ModelParameters
from mmpp import build_generator as build_base_generator
from mmpp import solve_stationary as solve_base_stationary
from mmpp_predictive import Metrics, PredictiveModelParameters, build_generator, solve_stationary

METRIC_FIELDS = ["P_block", "P_block_arrival_stable", "E_N", "E_B", "E_I", "E_S",
                 "E_off", "lambda_eff", "E_W", "rho_server", "Cost", "ERP"]

# Predictive だけにある診断指標 (P1 の発動・起動, セットアップの完了・取り消し, Delayoff の率)
DIAG_FIELDS = ["p1_fire_rate", "p1_launch_rate", "setup_completion_rate",
               "setup_cancel_rate", "delayoff_rate"]


def solve_base(params: ModelParameters) -> BaseMetrics:
    """ベースモデルの 1 点を解く."""
    pi = solve_base_stationary(build_base_generator(params), solver="gth")
    return BaseMetrics(params, pi)


def solve_predictive(params: PredictiveModelParameters) -> Metrics:
    """Predictive の 1 点を解く (到達できない状態は確率 0)."""
    pi = solve_stationary(build_generator(params), solver="gth")
    return Metrics(params, pi)


def base_metrics(m: BaseMetrics) -> Dict[str, float]:
    """ベースモデルの指標を METRIC_FIELDS の列名で返す."""
    d = m.all_metrics()
    return {
        "P_block": d["P_block"], "P_block_arrival_stable": d["P_block_arrival_stable"],
        "E_N": d["E[j]"], "E_B": d["E[B]"], "E_I": d["E[I]"], "E_S": d["E[S]"],
        "E_off": d["E[Off]"], "lambda_eff": d["lambda_eff"], "E_W": d["E[W]"],
        "rho_server": d["rho"], "Cost": d["cost_paper"], "ERP": d["ERP_paper"],
    }


def predictive_metrics(m: Metrics) -> Dict[str, float]:
    """Predictive の指標を METRIC_FIELDS の列名で返す."""
    return {
        "P_block": m.blocking_probability(),
        "P_block_arrival_stable": m.arrival_blocking_probability_stable(),
        "E_N": m.E_N, "E_B": m.E_B, "E_I": m.E_I, "E_S": m.E_S, "E_off": m.E_off,
        "lambda_eff": m.lambda_eff, "E_W": m.mean_waiting_time(),
        "rho_server": m.utilization(), "Cost": m.energy_cost_paper(), "ERP": m.erp_paper(),
    }


def predictive_diagnostics(m: Metrics) -> Dict[str, float]:
    """Predictive の診断指標を DIAG_FIELDS の列名で返す."""
    return {
        "p1_fire_rate": m.p1_fire_rate, "p1_launch_rate": m.p1_launch_rate,
        "setup_completion_rate": m.setup_completion_rate,
        "setup_cancel_rate": m.setup_cancel_rate, "delayoff_rate": m.delayoff_rate,
    }
