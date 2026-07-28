"""実験 3 スクリプトの基本動作テスト."""
import subprocess
import sys
from pathlib import Path


def test_experiment_3_delta_quick_runs():
    """--sweep delta --quick で最後まで実行できる."""
    result = subprocess.run(
        [sys.executable, "scripts/experiment_3_burstiness.py", "--sweep", "delta", "--quick"],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert Path("figures/experiment_3_burstiness_delta.png").exists()


def test_experiment_3_sigma_quick_runs():
    """--sweep sigma --quick で最後まで実行できる."""
    result = subprocess.run(
        [sys.executable, "scripts/experiment_3_burstiness.py", "--sweep", "sigma", "--quick"],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0
    assert Path("figures/experiment_3_burstiness_sigma.png").exists()


def test_experiment_3_requires_sweep():
    """--sweep が必須 (指定しないとエラー)."""
    result = subprocess.run(
        [sys.executable, "scripts/experiment_3_burstiness.py"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    # argparse は "required" エラーを stderr に出力
    assert "--sweep" in result.stderr or "required" in result.stderr.lower()


def test_experiment_3_poisson_limit():
    """delta=0 の Poisson 一致確認が動作する (警告なしで完走することを確認)."""
    result = subprocess.run(
        [sys.executable, "scripts/experiment_3_burstiness.py",
         "--sweep", "delta", "--n-points", "3"],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0
    # Poisson limit check が実行されていることを stdout で確認
    assert "Poisson limit check" in result.stdout
