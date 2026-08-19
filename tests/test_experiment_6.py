"""実験 6 (gamma 感度分析) の基本動作テスト."""
import csv
import subprocess
import sys
from pathlib import Path


def test_experiment_6_quick_runs_medium():
    """medium バースト --quick 実行."""
    result = subprocess.run(
        [
            sys.executable, "scripts/experiment_6_gamma.py",
            "--burst", "medium", "--quick",
            "--csv-dir", "/tmp/test_6_csv_medium",
            "--out-dir", "/tmp/test_6_figs_medium",
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"

    fig_files = list(Path("/tmp/test_6_figs_medium").glob("experiment_6_medium*.png"))
    csv_files = list(Path("/tmp/test_6_csv_medium").glob("experiment_6_medium*.csv"))
    assert len(fig_files) == 1
    assert len(csv_files) == 1


def test_experiment_6_csv_all_gammas():
    """CSV が 3 alpha × 3 gamma = 9 行になっていることを確認 (quick, medium)."""
    result = subprocess.run(
        [
            sys.executable, "scripts/experiment_6_gamma.py",
            "--burst", "medium", "--quick",
            "--csv-dir", "/tmp/test_6_csv_gamma",
            "--out-dir", "/tmp/test_6_figs_gamma",
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0

    csv_path = list(Path("/tmp/test_6_csv_gamma").glob("experiment_6_medium*.csv"))[0]
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * 3, f"期待: 9 行, 実際: {len(rows)}"

    gamma_present = set(float(row["gamma"]) for row in rows)
    assert gamma_present == {1.0, 5.0, 20.0}, f"gamma 不整合: {gamma_present}"


def test_experiment_6_conservation_law():
    """保存則 E[B]+E[S]+E[I]+E[Off]=c が全点で成立 (quick, medium)."""
    result = subprocess.run(
        [
            sys.executable, "scripts/experiment_6_gamma.py",
            "--burst", "medium", "--quick",
            "--csv-dir", "/tmp/test_6_csv_cons",
            "--out-dir", "/tmp/test_6_figs_cons",
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0

    csv_path = list(Path("/tmp/test_6_csv_cons").glob("experiment_6_medium*.csv"))[0]
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total = (
                float(row["E_B"]) + float(row["E_S"])
                + float(row["E_I"]) + float(row["E_off"])
            )
            assert abs(total - 20.0) < 1e-6, (
                f"保存則違反 at alpha={row['alpha']}, gamma={row['gamma']}: {total}"
            )
