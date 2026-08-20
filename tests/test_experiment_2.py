"""実験 2 スクリプトの基本動作テスト."""
import subprocess
import sys
from pathlib import Path


def test_experiment_2_quick_runs():
    """--quick モードで最後まで実行できる (図が生成される)."""
    result = subprocess.run(
        [
            sys.executable, "scripts/experiment_2_delayoff.py", "--quick",
            "--csv-dir", "/tmp/test_2_csv",
            "--out-dir", "/tmp/test_2_figs",
        ],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert Path("/tmp/test_2_figs/experiment_2_delayoff.png").exists()


def test_experiment_2_strong_burst_option():
    """--strong-burst フラグで別ファイルが生成される."""
    result = subprocess.run(
        [
            sys.executable, "scripts/experiment_2_delayoff.py", "--quick", "--strong-burst",
            "--csv-dir", "/tmp/test_2_csv_strong",
            "--out-dir", "/tmp/test_2_figs_strong",
        ],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0
    assert Path("/tmp/test_2_figs_strong/experiment_2_delayoff_strong.png").exists()


def test_alpha_levels_valid():
    """alpha 3 水準が期待通り."""
    from scripts.experiment_2_delayoff import ALPHA_LEVELS

    assert len(ALPHA_LEVELS) == 3
    alphas = [a for _, a in ALPHA_LEVELS]
    assert 0.1 in alphas and 1.0 in alphas and 10.0 in alphas
