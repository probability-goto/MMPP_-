"""実験 3-P スクリプトの基本動作テスト."""
import csv
import subprocess
import sys
from pathlib import Path


def test_experiment_3P_quick_runs_delta():
    """delta 走査 (実験 3-P-A) の --quick 実行."""
    result = subprocess.run(
        [
            sys.executable, "scripts/experiment_3P_burstiness.py",
            "--sweep", "delta", "--quick",
            "--csv-dir", "/tmp/test_3P_csv_delta",
            "--out-dir", "/tmp/test_3P_figs_delta",
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"

    fig_files = list(Path("/tmp/test_3P_figs_delta").glob("experiment_3P_delta_*.png"))
    csv_files = list(Path("/tmp/test_3P_csv_delta").glob("experiment_3P_delta_*.csv"))
    assert len(fig_files) == 1, f"期待: 1 figure, 実際: {len(fig_files)}"
    assert len(csv_files) == 1, f"期待: 1 CSV, 実際: {len(csv_files)}"


def test_experiment_3P_quick_runs_sigma():
    """sigma 走査 (実験 3-P-B) の --quick 実行."""
    result = subprocess.run(
        [
            sys.executable, "scripts/experiment_3P_burstiness.py",
            "--sweep", "sigma", "--quick",
            "--csv-dir", "/tmp/test_3P_csv_sigma",
            "--out-dir", "/tmp/test_3P_figs_sigma",
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"

    fig_files = list(Path("/tmp/test_3P_figs_sigma").glob("experiment_3P_sigma_*.png"))
    csv_files = list(Path("/tmp/test_3P_csv_sigma").glob("experiment_3P_sigma_*.csv"))
    assert len(fig_files) == 1
    assert len(csv_files) == 1


def test_experiment_3P_conservation_law_delta():
    """保存則 E[B]+E[S]+E[I]+E[Off]=c が全点で成立 (delta 走査, quick)."""
    result = subprocess.run(
        [
            sys.executable, "scripts/experiment_3P_burstiness.py",
            "--sweep", "delta", "--quick",
            "--csv-dir", "/tmp/test_3P_csv_cons_delta",
            "--out-dir", "/tmp/test_3P_figs_cons_delta",
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0

    csv_path = list(Path("/tmp/test_3P_csv_cons_delta").glob("experiment_3P_delta_*.csv"))[0]
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total = (
                float(row["E_B"]) + float(row["E_S"])
                + float(row["E_I"]) + float(row["E_off"])
            )
            assert abs(total - 20.0) < 1e-6, (
                f"保存則違反 at alpha={row['alpha']}, delta={row['delta']}: {total}"
            )
