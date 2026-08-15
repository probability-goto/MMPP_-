"""実験 1-P スクリプトの基本動作テスト."""
import csv
import subprocess
import sys
from pathlib import Path


def test_experiment_1P_quick_runs():
    """--quick で正常終了する."""
    result = subprocess.run(
        [
            sys.executable, "scripts/experiment_1P_traffic.py", "--quick",
            "--csv-dir", "/tmp/test_csv", "--out-dir", "/tmp/test_figs",
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"

    fig_files = list(Path("/tmp/test_figs").glob("experiment_1P_*.png"))
    csv_files = list(Path("/tmp/test_csv").glob("experiment_1P_*.csv"))
    assert len(fig_files) == 1
    assert len(csv_files) == 1


def test_experiment_1P_baseline_equivalence():
    """n_target=0, gamma=1 でも保存則 E[B]+E[S]+E[I]+E[Off]=c が成立する (quick)."""
    result = subprocess.run(
        [
            sys.executable, "scripts/experiment_1P_traffic.py",
            "--n-target", "0", "--gamma", "1.0", "--quick",
            "--csv-dir", "/tmp/test_csv_baseline",
            "--out-dir", "/tmp/test_figs_baseline",
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"

    csv_path = list(Path("/tmp/test_csv_baseline").glob("experiment_1P_*.csv"))[0]
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total = (
                float(row["E_B"]) + float(row["E_S"])
                + float(row["E_I"]) + float(row["E_off"])
            )
            assert abs(total - 20.0) < 1e-6, (
                f"保存則違反 at rho={row['rho']}, burst={row['burst_name']}: {total}"
            )
