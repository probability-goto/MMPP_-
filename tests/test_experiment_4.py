"""実験 4 スクリプトの基本動作テスト."""
import csv
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


def test_experiment_4_csv_output():
    """CSV が K × rho の全組み合わせで生成されることを確認 (quick, medium)."""
    result = subprocess.run(
        [
            sys.executable, "scripts/experiment_4_K_sensitivity.py",
            "--burst", "medium", "--quick",
            "--csv-dir", "/tmp/test_4_csv_medium",
            "--out-dir", "/tmp/test_4_figs_medium",
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"

    csv_path = Path("/tmp/test_4_csv_medium/experiment_4_medium.csv")
    assert csv_path.exists(), f"CSV が生成されていない: {csv_path}"

    # CSV の行数チェック (K_LEVELS 4 種 × rho 5 点 = 20 行 + header)
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 * 5, f"期待: 20 行, 実際: {len(rows)}"

    # 4 指標が全て含まれているか
    for row in rows:
        for metric in ["P_block_arrival", "E_W", "Cost", "ERP"]:
            assert metric in row, f"metric {metric} が CSV に存在しない"
