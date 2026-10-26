"""scripts/compare_map_mechanism.py (比較の地図の改善の仕組み) のテスト."""
import csv
from pathlib import Path

import pytest

from scripts import compare_map_mechanism as mech

REPO = Path(__file__).resolve().parent.parent


def _rows(name):
    with open(REPO / "results" / "compare_map" / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_rows_cover_summary():
    mrows = {(r["rho_B"], r["r1"]): r for r in _rows("mechanism.csv")}
    for s in _rows("summary.csv"):
        m = mrows[(s["rho_B"], s["r1"])]
        assert float(m["improvement"]) == pytest.approx(float(s["improvement"]), rel=1e-12)


def test_flow_balance_per_cycle():
    """定常状態ではセットアップを終える回数と Delayoff の回数が等しい (Predictive の最良点)."""
    for r in _rows("mechanism.csv"):
        if float(r["pred_delayoff_per_cycle"]) > 0:
            assert float(r["pred_setups_per_cycle"]) == pytest.approx(
                float(r["pred_delayoff_per_cycle"]), rel=1e-6)


def test_quasi_static_recomputes():
    r = [x for x in _rows("mechanism.csv") if float(x["rho_B"]) == 0.8
         and abs(float(x["r1"]) - 17.7827941) < 1e-6][0]
    qs = mech.quasi_static(0.8, float(r["base_beta"]))
    assert qs["qs_ERP"] == pytest.approx(float(r["qs_ERP"]), rel=1e-10)
    assert 0 < float(r["trans_share_ERP"]) < 1
