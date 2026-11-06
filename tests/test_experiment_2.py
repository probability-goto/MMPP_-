"""実験 2 (beta の走査とフェーズ別の解析) のテスト."""
import csv
import subprocess
import sys

from scripts import experiment_2_delayoff as e2


def test_experiment_2_quick_runs(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/experiment_2_delayoff.py", "--quick",
         "--csv-dir", str(tmp_path), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, result.stderr
    for stem in ("experiment_2_delayoff_erp", "experiment_2_delayoff_pblock"):
        assert (tmp_path / f"{stem}.png").exists()
    assert (tmp_path / "experiment_2_phasewise.csv").exists()
    with open(tmp_path / "experiment_2.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # rho_B 2 x フェーズの長さ 2 x alpha 3 x (beta 5 点 + 損益分岐 1 点)
    assert len(rows) == 2 * 2 * 3 * 6
    assert all(r["check"] == "ok" for r in rows)


def test_levels():
    assert e2.ALPHA_LEVELS == [0.005, 0.1, 1.0]
    assert len(e2.BETA_GRID) == 41
    assert e2.betas_for(0.005, [1e-3, 1e-2]) == [1e-3, 0.003, 1e-2]


def _fake_solve(center):
    """ERP が log10(beta) の 2 次関数で, center で最小になる偽の solve."""
    import math

    def solve(beta, stage):
        return dict(beta=beta, stage=stage, ERP=(math.log10(beta) - math.log10(center)) ** 2)
    return solve


def test_search_beta_star_interior():
    sigma = 1e-3
    rows = e2.search_beta_star(_fake_solve(2e-3), sigma)
    assert {r["stage"] for r in rows} == {"coarse", "fine"}
    best = min(rows, key=lambda r: r["ERP"])
    assert abs(best["beta"] / 2e-3 - 1) < 0.08    # 細かい刻みは 1 桁の 1/16


def test_search_beta_star_extends_below_range():
    sigma = 1e-3
    rows = e2.search_beta_star(_fake_solve(3e-7), sigma)    # sigma / 1000 より下
    assert {r["stage"] for r in rows} == {"coarse", "fine", "extend", "fine2"}
    best = min(rows, key=lambda r: r["ERP"])
    assert best["beta"] < sigma / 1000


def test_beta_star_matches_compare_map():
    """同じ系 (alpha = 0.005, 長フェーズ) では, 実験 2 の beta* が実験 5 の Base の beta* と一致する."""
    with open("results/experiment_2.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open("results/compare_map/summary.csv", encoding="utf-8") as f:
        summary = list(csv.DictReader(f))
    for rho_B in (0.8, 1.12):
        b = e2.beta_star_row(rows, rho_B=float(e2.d.rho_of(rho_B) * (1 + e2.d.DELTA)),
                             phase="long", alpha=0.005)
        s = [x for x in summary if float(x["rho_B"]) == rho_B
             and abs(float(x["r1"]) / e2.d.R1_LONG - 1) < 1e-6][0]
        assert float(b["beta"]) == float(s["base_beta"])
