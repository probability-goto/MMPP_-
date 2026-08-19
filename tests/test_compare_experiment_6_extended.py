"""実験 6 拡張走査統合スクリプトの基本動作テスト."""
import csv
import subprocess
import sys
import tempfile
from pathlib import Path


def _make_dummy_csv(path: Path, gammas: list, alpha: float = 1.0) -> None:
    """テスト用のダミー CSV を作成する.

    5 gammas × 1 alpha の 5 行を含む.
    ERP は gamma に対して線形減少 (単純パターン, テスト用).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "burst_name", "delta", "sigma", "alpha", "beta", "rho",
        "n_target", "gamma", "P_block_arrival", "E_W", "Cost", "ERP",
        "E_N", "lambda_eff", "E_B", "E_S", "E_I", "E_off", "N_states",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for gamma in gammas:
            writer.writerow({
                "burst_name": "medium", "delta": 0.6, "sigma": 0.1,
                "alpha": alpha, "beta": 0.005, "rho": 0.7,
                "n_target": 10, "gamma": gamma,
                "P_block_arrival": 0.05, "E_W": 1.5,
                "Cost": 6.0, "ERP": 10.0 - 0.1 * gamma,  # γ 増で ERP 減
                "E_N": 60, "lambda_eff": 40, "E_B": 15, "E_S": 2,
                "E_I": 1, "E_off": 2, "N_states": 90000,
            })


def test_extended_merge_and_plot():
    """初期走査と拡張走査の CSV をマージして図を生成する."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # 初期走査ダミー: γ ∈ {1, 2, 5, 10, 20}
        initial_dir = tmp / "results"
        _make_dummy_csv(initial_dir / "experiment_6_medium_nt10.csv",
                        gammas=[1, 2, 5, 10, 20])
        # 拡張走査ダミー: γ ∈ {20, 50, 100, 500, 1000}
        extended_dir = tmp / "results" / "experiment_6_extended"
        _make_dummy_csv(extended_dir / "experiment_6_medium_nt10.csv",
                        gammas=[20, 50, 100, 500, 1000])

        out_dir = tmp / "figures" / "experiment_6_extended"

        result = subprocess.run(
            [
                sys.executable, "scripts/compare_experiment_6_extended.py",
                "--burst", "medium",
                "--initial-csv-dir", str(initial_dir),
                "--extended-csv-dir", str(extended_dir),
                "--out-dir", str(out_dir),
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"stderr:\n{result.stderr}"

        fig_files = list(out_dir.glob("experiment_6_extended_medium.png"))
        assert len(fig_files) == 1

        # 最適 γ サマリの出力を確認
        assert "拡張走査後の最適 γ" in result.stdout
        assert "γ=20 の再現性チェック" in result.stdout


def test_extended_missing_csv_gracefully():
    """CSV ファイルが存在しない場合、エラーではなく警告を出して継続."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable, "scripts/compare_experiment_6_extended.py",
                "--burst", "medium",
                "--initial-csv-dir", tmpdir,  # 空ディレクトリ
                "--extended-csv-dir", tmpdir,
                "--out-dir", tmpdir,
            ],
            capture_output=True, text=True, timeout=60,
        )
        # 警告を出しても正常終了する
        assert result.returncode == 0
        assert "警告" in result.stdout or "警告" in result.stderr
