"""実験 2 (beta の走査とフェーズ別の解析) のテスト."""
import csv
import subprocess
import sys

from scripts import experiment_2_delayoff as e2


def test_experiment_2_quick_runs(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/experiment_2_delayoff.py", "--quick",
         "--csv-dir", str(tmp_path), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, result.stderr
    for stem in ("experiment_2_delayoff_erp", "experiment_2_delayoff_pblock",
                 "experiment_2_phasewise"):
        assert (tmp_path / f"{stem}.png").exists()
    with open(tmp_path / "experiment_2.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # rho_B 2 x フェーズの長さ 2 x alpha 3 x (beta 5 点 + 損益分岐 1 点)
    assert len(rows) == 2 * 2 * 3 * 6
    assert all(r["check"] == "ok" for r in rows)


def test_levels():
    assert e2.ALPHA_LEVELS == [0.005, 0.1, 1.0]
    assert len(e2.BETA_GRID) == 41
    assert e2.betas_for(0.005, [1e-3, 1e-2]) == [1e-3, 0.003, 1e-2]
