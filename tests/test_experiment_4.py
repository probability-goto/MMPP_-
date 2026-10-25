"""実験 4 (バッファ容量 K の感度) のテスト."""
import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def test_experiment_4_quick_runs(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/experiment_4_K_sensitivity.py", "--quick",
         "--csv-dir", str(tmp_path), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=1200,
    )
    assert result.returncode == 0, result.stderr
    for phase in ("short", "long"):
        assert (tmp_path / f"experiment_4_K_sensitivity_{phase}.png").exists()
        with open(tmp_path / f"experiment_4_{phase}.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4 * 3
        assert all(r["check"] == "ok" for r in rows)


@pytest.mark.parametrize("phase,condition", [("short", "r1=0.1"), ("long", "r1=17.8")])
def test_K200_matches_experiment_1(phase, condition):
    """results/ の実験 4 の K=200 の行が, 実験 1 の同じ条件の行と一致する."""
    def load(name):
        with open(REPO / "results" / name, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    e1 = {float(r["rho"]): r for r in load("experiment_1.csv") if r["condition"] == condition}
    e4 = [r for r in load(f"experiment_4_{phase}.csv") if int(r["K"]) == 200]
    assert len(e4) == len(e1)
    for r in e4:
        ref = e1[float(r["rho"])]
        for k in ("P_block_arrival_stable", "E_W", "Cost", "ERP"):
            assert float(r[k]) == pytest.approx(float(ref[k]), rel=1e-12)
