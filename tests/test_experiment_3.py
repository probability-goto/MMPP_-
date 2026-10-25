"""実験 3 (delta と r1 の走査) のテスト."""
import subprocess
import sys

import pytest


@pytest.mark.parametrize("sweep", ["delta", "r1"])
def test_experiment_3_quick_runs(tmp_path, sweep):
    result = subprocess.run(
        [sys.executable, "scripts/experiment_3_burstiness.py", "--sweep", sweep, "--quick",
         "--csv-dir", str(tmp_path), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / f"experiment_3_burstiness_{sweep}.png").exists()
    assert (tmp_path / f"experiment_3_{sweep}.csv").exists()


def test_experiment_3_requires_sweep():
    result = subprocess.run([sys.executable, "scripts/experiment_3_burstiness.py"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "--sweep" in result.stderr
