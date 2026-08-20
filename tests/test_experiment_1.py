"""実験 1 スクリプトの基本動作テスト."""
import subprocess
import sys
from pathlib import Path


def test_experiment_1_quick_runs():
    """--quick モードで最後まで実行できる (図が生成される)."""
    result = subprocess.run(
        [
            sys.executable, "scripts/experiment_1_traffic.py", "--quick",
            "--csv-dir", "/tmp/test_1_csv",
            "--out-dir", "/tmp/test_1_figs",
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"

    # 図が生成されている
    assert Path("/tmp/test_1_figs/experiment_1_traffic.png").exists()


def test_burst_levels_valid():
    """3 バースト水準の delta と sigma が妥当な範囲."""
    from scripts.experiment_1_traffic import BURST_LEVELS

    assert len(BURST_LEVELS) == 3
    for name, delta, sigma in BURST_LEVELS:
        assert 0 <= delta < 1
        assert sigma > 0
        assert name in ("weak", "medium", "strong")
