"""実験 2-P スクリプトの基本動作テスト."""
import csv
import subprocess
import sys
from pathlib import Path


def test_experiment_2P_quick_runs_medium():
    """中バースト水準 (デフォルト) の --quick 実行."""
    result = subprocess.run(
        [
            sys.executable, "scripts/experiment_2P_delayoff.py", "--quick",
            "--csv-dir", "/tmp/test_2P_csv_medium",
            "--out-dir", "/tmp/test_2P_figs_medium",
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"

    fig_files = list(Path("/tmp/test_2P_figs_medium").glob("experiment_2P_medium_*.png"))
    csv_files = list(Path("/tmp/test_2P_csv_medium").glob("experiment_2P_medium_*.csv"))
    assert len(fig_files) == 1, f"期待: 1 figure, 実際: {len(fig_files)}"
    assert len(csv_files) == 1, f"期待: 1 CSV, 実際: {len(csv_files)}"


def test_experiment_2P_quick_runs_strong():
    """強バースト水準 (--strong-burst) の --quick 実行."""
    result = subprocess.run(
        [
            sys.executable, "scripts/experiment_2P_delayoff.py",
            "--strong-burst", "--quick",
            "--csv-dir", "/tmp/test_2P_csv_strong",
            "--out-dir", "/tmp/test_2P_figs_strong",
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"

    fig_files = list(Path("/tmp/test_2P_figs_strong").glob("experiment_2P_strong_*.png"))
    csv_files = list(Path("/tmp/test_2P_csv_strong").glob("experiment_2P_strong_*.csv"))
    assert len(fig_files) == 1
    assert len(csv_files) == 1


def test_experiment_2P_conservation_law():
    """保存則 E[B]+E[S]+E[I]+E[Off]=c が全点で成立する (quick, medium)."""
    result = subprocess.run(
        [
            sys.executable, "scripts/experiment_2P_delayoff.py", "--quick",
            "--csv-dir", "/tmp/test_2P_csv_cons",
            "--out-dir", "/tmp/test_2P_figs_cons",
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0

    csv_path = list(Path("/tmp/test_2P_csv_cons").glob("experiment_2P_medium_*.csv"))[0]
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total = (
                float(row["E_B"]) + float(row["E_S"])
                + float(row["E_I"]) + float(row["E_off"])
            )
            assert abs(total - 20.0) < 1e-6, (
                f"保存則違反 at alpha={row['alpha']}, beta={row['beta']}: {total}"
            )
