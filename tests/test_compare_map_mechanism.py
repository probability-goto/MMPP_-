"""scripts/compare_map_mechanism.py (比較の地図の改善の仕組み) のテスト."""
import csv
import math
from pathlib import Path

import pytest

from mmpp_experiments import design as d
from scripts import compare_map_mechanism as mech

REPO = Path(__file__).resolve().parent.parent


def _rows(name):
    with open(REPO / "results" / "compare_map" / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _cell(rho_B, r1):
    return [x for x in _rows("mechanism.csv") if float(x["rho_B"]) == rho_B
            and abs(float(x["r1"]) / r1 - 1) < 1e-6][0]


def test_rows_cover_summary():
    mrows = {(r["rho_B"], r["r1"]): r for r in _rows("mechanism.csv")}
    for s in _rows("summary.csv"):
        m = mrows[(s["rho_B"], s["r1"])]
        assert float(m["improvement"]) == pytest.approx(float(s["improvement"]), rel=1e-12)


def test_quasi_static_recomputes():
    r = _cell(0.8, d.R1_LONG)
    beta, g = float(r["pred_beta"]), float(r["pred_gamma"])
    assert int(r["pred_n_target"]) == 20
    qb = mech.quasi_static(0.8, float(r["base_beta"]), float(r["base_beta"]))
    qp = mech.quasi_static(0.8, g * beta, 1e-12)
    assert qb["ERP"] == pytest.approx(float(r["qs_base_ERP"]), rel=1e-10)
    assert qp["ERP"] == pytest.approx(float(r["qs_pred_ERP"]), rel=1e-10)


def test_small_beta_point_recomputes():
    r = _cell(0.8, d.R1_LONG)
    row = d.solve_row("t", d.rho_of(0.8), 0.6, float(r["sigma"]), beta=float(r["beta_qs"]))
    assert float(row["ERP"]) == pytest.approx(float(r["small_ERP"]), rel=1e-10)


def test_quasi_static_only_where_representable():
    """0 < n_target < c の最良点では準静的な改善率を計算しない."""
    for r in _rows("mechanism.csv"):
        nt = int(r["pred_n_target"])
        assert math.isnan(float(r["qs_improvement"])) == (0 < nt < 20)
