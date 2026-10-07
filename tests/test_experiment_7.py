"""実験 7 (scripts/experiment_7_frontier.py) のテスト.

- 走査の格子 (点数, beta の範囲)
- パレートフロンティア, 線形補間, 支配の判定
- 小規模設定での通し実行 (CSV の列, 再開, 整合性の検査, レポートと図)
"""
import csv
import os

import numpy as np
import pytest

import scripts.experiment_7_frontier as e7


def test_grid_counts():
    # Check 13 + Base 13 + NoCancel 13 + P2 78 + P1 104 + P1+P2 624
    for cond in e7.CONDITIONS:
        assert len(e7.condition_points(cond)) == 845
    assert e7.total_points(list(e7.CONDITIONS)) == 4225


@pytest.mark.parametrize("cond", list(e7.CONDITIONS))
def test_beta_levels(cond):
    sigma = e7.CONDITIONS[cond][2]
    b = e7.beta_levels(sigma)
    assert len(b) == 13
    assert b[0] == pytest.approx(sigma / 1000, rel=1e-12)
    assert b[-1] == pytest.approx(sigma * 100, rel=1e-12)
    ratios = np.diff(np.log(b))
    assert np.allclose(ratios, ratios[0])


def test_pareto_front():
    pts = [(1.0, 5.0, 0), (2.0, 3.0, 1), (2.5, 4.0, 2), (3.0, 1.0, 3), (1.0, 6.0, 4)]
    front = e7.pareto_front(pts)
    assert [p[2] for p in front] == [0, 1, 3]


def test_interp_cost():
    front = [(1.0, 5.0, 0), (2.0, 3.0, 1), (3.0, 1.0, 2)]
    assert e7.interp_cost(front, 0.5) is None
    assert e7.interp_cost(front, 1.5) == pytest.approx(4.0)
    assert e7.interp_cost(front, 3.0) == pytest.approx(1.0)
    assert e7.interp_cost(front, 10.0) == pytest.approx(1.0)


def test_dominance():
    ref = [(1.0, 5.0, 0), (2.0, 3.0, 1)]
    assert e7.dominance([(0.9, 2.0, 0)], ref)[0] == "全域"
    assert e7.dominance([(1.5, 2.0, 0)], ref) == ("一部", 1, 2)
    assert e7.dominance([(3.0, 6.0, 0)], ref)[0] == "しない"
    # 同じ点は支配しない
    assert e7.dominance(ref, ref)[0] == "しない"


@pytest.fixture
def small_setup(monkeypatch):
    """c=4, K=10 の小規模設定に差し替える."""
    monkeypatch.setattr(e7, "BASELINE", dict(c=4, K=10, b=2, mu=1.0))
    monkeypatch.setattr(e7, "CONDITIONS", {"C1": (0.5, 0.6, 0.02, 0.1, "有効候補")})
    monkeypatch.setattr(e7, "N_BETA", 3)
    monkeypatch.setattr(e7, "GAMMAS", [3.0])
    monkeypatch.setattr(e7, "N_TARGETS", [2])


def test_small_run_resume_and_report(small_setup, tmp_path):
    out = str(tmp_path / "res")
    figs = str(tmp_path / "fig")
    n = len(e7.condition_points("C1"))
    # Check 3 + Base 3 + NoCancel 3 + P2 3 + P1 3 + P1(protect) 3 + P1+P2 3 + 同(protect) 3
    assert n == 24
    e7.run(["C1"], out_dir=out)
    with open(e7.csv_path("C1", out), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == n
    assert set(e7.FIELDNAMES) <= set(rows[0])
    assert all(r["check"] == "ok" for r in rows)
    for r in rows:
        if r["series"] == "Check":
            assert float(r["consistency_max_relerr"]) <= 1e-10
        if r["series"] == "Base":
            assert r["diag_source"] == "Check"
        if r["series"] == "NoCancel":
            assert float(r["setup_cancel_rate"]) == 0.0

    # 再実行しても追記されない
    e7.run(["C1"], out_dir=out)
    with open(e7.csv_path("C1", out), newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == n

    text, summaries = e7.report(["C1"], out_dir=out, fig_dir=figs)
    assert "(a) 各系列の ERP 最小点" in text and "(d) フロンティアの支配関係" in text
    assert "NoCancel" in summaries["C1"]["dom"]
    assert os.path.exists(os.path.join(out, "report.md"))
    for stem in ("EW_cost", "cost_pblock"):
        for ext in ("png", "pdf"):
            assert os.path.exists(os.path.join(figs, f"experiment_7_C1_{stem}.{ext}"))


def test_consistency_failure_stops(small_setup, tmp_path, monkeypatch):
    monkeypatch.setattr(e7, "CONSISTENCY_TOL", -1.0)
    with pytest.raises(e7.ConsistencyError):
        e7.run(["C1"], out_dir=str(tmp_path))
    with open(e7.csv_path("C1", str(tmp_path)), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # 最初の beta の Check と Base だけが書かれて止まる
    assert [r["series"] for r in rows] == ["Check", "Base"]
    assert "consistency" in rows[0]["check"]
