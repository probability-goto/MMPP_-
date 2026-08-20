"""alpha 別 gamma マッピング機能 (--alpha-gamma-map) のテスト."""
import csv
import subprocess
import sys
from pathlib import Path


def test_experiment_2P_alpha_gamma_map_quick():
    """--alpha-gamma-map で実験 2-P が正常終了する (quick, medium)."""
    result = subprocess.run(
        [
            sys.executable, "scripts/experiment_2P_delayoff.py", "--quick",
            "--alpha-gamma-map", "0.1:1.0,1.0:100.0,10.0:1000.0",
            "--csv-dir", "/tmp/test_2p_gmap_csv",
            "--out-dir", "/tmp/test_2p_gmap_figs",
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"

    csv_files = list(Path("/tmp/test_2p_gmap_csv").glob("experiment_2P_*gmap.csv"))
    assert len(csv_files) == 1, f"期待: 1 CSV, 実際: {len(csv_files)}"

    with open(csv_files[0], encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    alpha_gamma_pairs = set(
        (float(row["alpha"]), float(row["gamma"])) for row in rows
    )
    expected = {(0.1, 1.0), (1.0, 100.0), (10.0, 1000.0)}
    assert alpha_gamma_pairs == expected, (
        f"期待: {expected}, 実際: {alpha_gamma_pairs}"
    )


def test_experiment_3P_alpha_gamma_map_quick():
    """--alpha-gamma-map で実験 3-P が正常終了する (quick, delta)."""
    result = subprocess.run(
        [
            sys.executable, "scripts/experiment_3P_burstiness.py",
            "--sweep", "delta", "--quick",
            "--alpha-gamma-map", "0.1:1.0,1.0:100.0,10.0:1000.0",
            "--csv-dir", "/tmp/test_3p_gmap_csv",
            "--out-dir", "/tmp/test_3p_gmap_figs",
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"

    csv_files = list(Path("/tmp/test_3p_gmap_csv").glob("experiment_3P_delta_*gmap.csv"))
    assert len(csv_files) == 1, f"期待: 1 CSV, 実際: {len(csv_files)}"

    with open(csv_files[0], encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    alpha_gamma_pairs = set(
        (float(row["alpha"]), float(row["gamma"])) for row in rows
    )
    expected = {(0.1, 1.0), (1.0, 100.0), (10.0, 1000.0)}
    assert alpha_gamma_pairs == expected, (
        f"期待: {expected}, 実際: {alpha_gamma_pairs}"
    )


def test_backward_compat_single_gamma():
    """--gamma のみで従来通り動作することを確認 (実験 2-P, quick)."""
    result = subprocess.run(
        [
            sys.executable, "scripts/experiment_2P_delayoff.py", "--quick",
            "--gamma", "5.0",
            "--csv-dir", "/tmp/test_2p_g5_csv",
            "--out-dir", "/tmp/test_2p_g5_figs",
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"

    csv_files = list(Path("/tmp/test_2p_g5_csv").glob("experiment_2P_*g5.0.csv"))
    assert len(csv_files) == 1, "後方互換性: --gamma のみで動作すること"
