"""実験 1・4 のセットアップ率 alpha (既定値 0.005, --alpha で変更) のテスト.

1. 既定値 (alpha=0.005) で実行した結果が results/ の無印の CSV と全列で一致する.
2. --alpha 0.1 で実行した結果が旧 CSV (results/*_a0.1.csv) と全列で一致する (回帰テスト).
3. alpha=0.005 の点が 4 つの検査 (保存則, 流量バランス, min pi >= 0, P_block > 0)
   を満たす.
"""
import csv
import math
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import experiment_1_traffic as e1
from scripts import experiment_4_K_sensitivity as e4

REPO = Path(__file__).resolve().parent.parent
REL_TOL = 1e-10

# 比べない列: 旧 CSV にだけある引き算版の P_block (b6afb70 で削除) と,
# 丸め誤差の大きさそのものである検査の誤差列
SKIP_COLUMNS = {"P_block_arrival", "cons_err", "flow_rel_err"}


def _read_csv(path: Path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _assert_rows_match(new_rows, ref_rows, key_cols):
    """new_rows の各行が, キー列の一致する ref_rows の行と共通の全列で一致することを確かめる."""
    def key(row):
        return tuple(
            round(float(row[c]), 12) if c == "rho" else row[c] for c in key_cols
        )

    ref_by_key = {key(r): r for r in ref_rows}
    assert new_rows
    for new in new_rows:
        ref = ref_by_key.get(key(new))
        assert ref is not None, f"参照 CSV に対応する行がない: {key(new)}"
        common = (set(new) & set(ref)) - SKIP_COLUMNS
        assert {"P_block_arrival_stable", "E_W", "Cost", "ERP"} <= common
        for col in common:
            if col in ("burst_name", "check"):
                assert new[col] == ref[col], f"{key(new)} の {col}"
                continue
            a, b = float(new[col]), float(ref[col])
            assert math.isclose(a, b, rel_tol=REL_TOL, abs_tol=0.0), (
                f"{key(new)} の {col}: new={a!r}, ref={b!r}"
            )


def _run(args, timeout=1200):
    result = subprocess.run(
        [sys.executable, *args], capture_output=True, text=True,
        timeout=timeout, cwd=REPO,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    return result


def test_alpha_suffix():
    """既定値 (0.005) では接尾辞なし, それ以外では _a{alpha:g}."""
    for mod in (e1, e4):
        assert mod.DEFAULT_ALPHA == 0.005
        assert mod.BASELINE["alpha"] == 0.005
        assert mod.alpha_suffix(0.005) == ""
        assert mod.alpha_suffix(0.1) == "_a0.1"


@pytest.mark.parametrize("alpha_args,suffix", [([], ""), (["--alpha", "0.1"], "_a0.1")])
def test_experiment_1_matches_csv(tmp_path, alpha_args, suffix):
    """quick (3 点: 0.1, 0.525, 0.95. 15 点の格子に含まれる) の結果が results/ の CSV と一致する."""
    _run([
        "scripts/experiment_1_traffic.py", "--quick", *alpha_args,
        "--csv-dir", str(tmp_path / "csv"), "--out-dir", str(tmp_path / "fig"),
    ])
    new_rows = _read_csv(tmp_path / "csv" / f"experiment_1{suffix}.csv")
    assert len(new_rows) == 3 * 3
    assert all(r["check"] == "ok" for r in new_rows)
    _assert_rows_match(
        new_rows, _read_csv(REPO / "results" / f"experiment_1{suffix}.csv"),
        key_cols=("burst_name", "rho"),
    )
    for ext in ("png", "pdf"):
        assert (tmp_path / "fig" / f"experiment_1_traffic{suffix}.{ext}").exists()


@pytest.mark.parametrize("alpha_args,suffix", [([], ""), (["--alpha", "0.1"], "_a0.1")])
def test_experiment_4_matches_csv(tmp_path, alpha_args, suffix):
    """結果が results/ の CSV と一致する.

    計算時間を抑えるため K は {100, 200}, rho は 8 点 (15 点の格子の 1 つおき) にする.
    """
    _run([
        "scripts/experiment_4_K_sensitivity.py", "--burst", "all", *alpha_args,
        "--n-points", "8", "--K-levels", "100", "200",
        "--csv-dir", str(tmp_path / "csv"), "--out-dir", str(tmp_path / "fig"),
    ])
    for level in ("weak", "medium", "strong"):
        new_rows = _read_csv(tmp_path / "csv" / f"experiment_4_{level}{suffix}.csv")
        assert len(new_rows) == 2 * 8
        assert all(r["check"] == "ok" for r in new_rows)
        _assert_rows_match(
            new_rows, _read_csv(REPO / "results" / f"experiment_4_{level}{suffix}.csv"),
            key_cols=("K", "rho"),
        )
        for ext in ("png", "pdf"):
            assert (tmp_path / "fig" / f"experiment_4_K_sensitivity_{level}{suffix}.{ext}").exists()


@pytest.mark.parametrize("level_name,delta,sigma", e1.BURST_LEVELS)
def test_alpha_0005_point_passes_checks(level_name, delta, sigma):
    """既定値 alpha=0.005, rho=0.5 の 1 点が 4 つの検査を満たす (実験 1 の K=200, 実験 4 の K=100)."""
    p1 = e1.build_params(0.5, delta, sigma)
    assert p1.alpha == 0.005 and p1.K == 200 and p1.beta == 0.005
    checks = e1.point_checks(p1, e1.run_theory(p1))
    assert checks["check"] == "ok", checks
    assert checks["cons_err"] <= 1e-9
    assert checks["flow_rel_err"] <= 1e-9
    assert checks["min_pi"] >= 0.0

    p4 = e4.build_params(100, 0.5, delta, sigma)
    assert p4.alpha == 0.005 and p4.K == 100
    extra = e4.extended_columns(p4, e4.run_theory(p4))
    assert extra["check"] == "ok", extra
