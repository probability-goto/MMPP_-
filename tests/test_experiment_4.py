"""実験 4 スクリプトの基本動作テスト."""
import subprocess
import sys
from pathlib import Path


def test_experiment_4_quick_runs():
    """--quick で実行できる."""
    result = subprocess.run(
        [sys.executable, "scripts/experiment_4_K_sensitivity.py",
         "--burst", "medium", "--quick"],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert Path("figures/experiment_4_K_sensitivity_medium.png").exists()


def test_experiment_4_all_bursts():
    """全バースト水準で動作する (quick)."""
    result = subprocess.run(
        [sys.executable, "scripts/experiment_4_K_sensitivity.py",
         "--burst", "all", "--quick"],
        capture_output=True, text=True, timeout=1200,
    )
    assert result.returncode == 0
    for level in ["weak", "medium", "strong"]:
        assert Path(f"figures/experiment_4_K_sensitivity_{level}.png").exists()
