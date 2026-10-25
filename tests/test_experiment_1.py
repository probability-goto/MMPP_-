"""実験 1 (rho の走査) と共通の設計 (mmpp_experiments/design.py) のテスト."""
import csv
import subprocess
import sys

import numpy as np
import pytest

from mmpp import ModelParameters
from mmpp_experiments import design as d
from mmpp_experiments.evaluate import solve_base
from mmpp_experiments.params import SYSTEM


def test_experiment_1_quick_runs(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/experiment_1_traffic.py", "--quick",
         "--csv-dir", str(tmp_path), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "experiment_1_traffic.png").exists()
    with open(tmp_path / "experiment_1.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 * 3
    assert all(r["check"] == "ok" for r in rows)


def test_design_levels():
    assert d.RHO_GRID[0] == 0.1 and d.RHO_GRID[-1] == 0.95 and 0.625 in d.RHO_GRID
    assert d.sigma_of(d.R1_SHORT) == pytest.approx(0.05)
    assert d.R1_LONG == pytest.approx(17.7827941)
    assert [d.rho_of(x) for x in d.RHO_B_LEVELS] == pytest.approx([0.5, 0.7])
    assert d.break_even_beta(0.005) == pytest.approx(0.003)


def test_break_even_beta_equalizes_idle_and_setup_energy():
    """損益分岐の beta では, 定常状態で Idle の電力とセットアップの電力が等しい (beta E[I] = alpha E[S])."""
    row = d.solve_row("t", 0.5, d.DELTA, d.sigma_of(d.R1_LONG))
    c_b = 0.04419 * SYSTEM["b"] + 0.15503
    idle_power = float(row["Cost"]) - c_b * (float(row["E_B"]) + float(row["E_S"]))
    setup_power = c_b * float(row["E_S"])
    assert idle_power == pytest.approx(setup_power, rel=1e-9)


def test_delta_zero_is_poisson():
    """delta = 0 の 2 位相 MMPP は, 同じ平均到着率の 1 位相ポアソン到着と一致する."""
    row = d.solve_row("t", 0.5, 0.0, 1.0, r1="")
    lam = 0.5 * SYSTEM["c"] * SYSTEM["b"] * SYSTEM["mu"]
    p = ModelParameters(C0=np.array([[-lam]]), C1=np.array([[lam]]), alpha=d.ALPHA,
                        beta=d.break_even_beta(d.ALPHA), **SYSTEM)
    m = solve_base(p)
    assert float(row["ERP"]) == pytest.approx(m.erp_paper(), rel=1e-9)
    assert float(row["P_block_arrival_stable"]) == pytest.approx(
        m.arrival_blocking_probability_stable(), rel=1e-9)
