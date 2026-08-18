"""実験 4-P スクリプトの基本動作テスト."""
import csv
import subprocess
import sys
from pathlib import Path


def test_experiment_4P_quick_runs_medium():
    """中バースト水準 (single burst) の --quick 実行."""
    result = subprocess.run(
        [
            sys.executable, "scripts/experiment_4P_K_sensitivity.py",
            "--burst", "medium", "--quick",
            "--csv-dir", "/tmp/test_4P_csv_medium",
            "--out-dir", "/tmp/test_4P_figs_medium",
        ],
        capture_output=True, text=True, timeout=1800,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"

    fig_files = list(Path("/tmp/test_4P_figs_medium").glob("experiment_4P_*medium*.png"))
    csv_files = list(Path("/tmp/test_4P_csv_medium").glob("experiment_4P_medium*.csv"))
    assert len(fig_files) == 1, f"期待: 1 figure, 実際: {len(fig_files)}"
    assert len(csv_files) == 1, f"期待: 1 CSV, 実際: {len(csv_files)}"


def test_experiment_4P_csv_all_K_levels():
    """CSV が K_LEVELS 4 種 × rho 5 点 = 20 行になっていることを確認 (quick, medium)."""
    result = subprocess.run(
        [
            sys.executable, "scripts/experiment_4P_K_sensitivity.py",
            "--burst", "medium", "--quick",
            "--csv-dir", "/tmp/test_4P_csv_all_K",
            "--out-dir", "/tmp/test_4P_figs_all_K",
        ],
        capture_output=True, text=True, timeout=1800,
    )
    assert result.returncode == 0

    csv_path = list(Path("/tmp/test_4P_csv_all_K").glob("experiment_4P_medium*.csv"))[0]
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 * 5, f"期待: 20 行, 実際: {len(rows)}"

    # 全 K_LEVELS が存在するか
    K_present = set(int(row["K"]) for row in rows)
    assert K_present == {100, 200, 500, 1000}, f"K_LEVELS 不整合: {K_present}"


def test_experiment_4P_conservation_law():
    """保存則 E[B]+E[S]+E[I]+E[Off]=c が全点で成立 (quick, medium).

    実験 4-P の CSV には E_B/E_S/E_I/E_off が含まれていないため, 標準出力で
    保存則エラーが 0 に近いことを確認する.
    """
    result = subprocess.run(
        [
            sys.executable, "scripts/experiment_4P_K_sensitivity.py",
            "--burst", "medium", "--quick",
            "--csv-dir", "/tmp/test_4P_csv_cons",
            "--out-dir", "/tmp/test_4P_figs_cons",
        ],
        capture_output=True, text=True, timeout=1800,
    )
    assert result.returncode == 0

    # 標準出力から「保存則の最大誤差」を抽出して確認
    for line in result.stdout.split("\n"):
        if "保存則の最大誤差" in line:
            # "保存則の最大誤差: X.XXe-YY" の形式
            err_str = line.split(":")[-1].strip()
            err = float(err_str)
            assert err < 1e-6, f"保存則違反: {err}"
            break
    else:
        raise AssertionError("保存則の最大誤差が標準出力に見つからない")
