"""実験 8 (scripts/experiment_8_map.py) のテスト.

- 地図の格子 (r1, sigma, rho)
- 2 段階の β の探索と範囲の拡張 (人工的な ERP で手順を追う)
- 小規模設定での通し実行 (再開, 整合性の検査, レポートと図)
"""
import csv
import os

import numpy as np
import pytest

import scripts.experiment_8_map as e8


def test_grid():
    assert e8.R1_LEVELS[0] == pytest.approx(0.1) and e8.R1_LEVELS[-1] == pytest.approx(100)
    assert len(e8.R1_LEVELS) == 5
    cp = e8.cell_params("B_alpha0.005", e8.R1_LEVELS[0])
    assert cp["sigma"] == pytest.approx(0.05)
    assert cp["rho"] == pytest.approx(0.5)
    # 地図 B の alpha=0.1 の行は地図 A の rho_B=0.8 の行と共有する
    assert "A_rhoB0.8" in e8.MAP_A and "A_rhoB0.8" in e8.MAP_B
    assert len(dict.fromkeys(e8.MAP_A + e8.MAP_B)) * len(e8.R1_LEVELS) == 35
    b = e8.coarse_betas(1.0)
    assert len(b) == 9 and b[0] == pytest.approx(1e-3) and b[-1] == pytest.approx(100)


def test_fine_between():
    grid = [1.0, 10.0, 100.0]
    assert e8._fine_between(grid, 10.0, 5) == pytest.approx(list(np.logspace(0, 2, 5)))
    assert e8._fine_between(grid, 1.0, 3) == pytest.approx([1.0, 10 ** 0.5, 10.0])


def _fake_rows(pts, r1, erp):
    return [{"r1": r1, "series": s, "stage": st, "beta": b, "n_target": nt, "gamma": g,
             "ERP": erp(s, b, nt, g)} for s, st, b, nt, g in pts]


def test_search_stages_with_edge():
    """ERP が β について増加 (最良点が下端) のとき, 細かい探索 -> 拡張 -> その周りの細かい探索."""
    row, r1 = "A_rhoB0.8", e8.R1_LEVELS[2]
    erp = lambda s, b, nt, g: b * (1 + 0.01 * nt + 0.001 * g)
    rows = []
    pts = e8.next_points(row, r1, rows)
    assert {p[1] for p in pts} == {"coarse", "check"}
    assert len(pts) == 9 * (1 + 5 + 4 + 12) + 1
    rows += _fake_rows(pts, r1, erp)

    pts = e8.next_points(row, r1, rows)
    assert {p[1] for p in pts} == {"fine"}
    # 最良点が下端なので, 両隣は下端とその次. 両端は計算済み
    assert len(pts) == (21 - 2) + 3 * (7 - 2)
    rows += _fake_rows(pts, r1, erp)

    pts = e8.next_points(row, r1, rows)
    assert {p[1] for p in pts} == {"extend"}
    lo = e8.coarse_betas(e8.cell_params(row, r1)["sigma"])[0]
    assert all(p[2] < lo for p in pts) and len(pts) == 4 * 3
    # 拡張は粗い探索で最良の (n_target, gamma) で行う
    assert {(p[0], p[3], p[4]) for p in pts} == {
        ("Base", 0, 1.0), ("P2 only", 0, 3.0), ("P1 only (protect)", 5, 1.0),
        ("P1+P2 (protect)", 5, 3.0)}
    rows += _fake_rows(pts, r1, erp)

    pts = e8.next_points(row, r1, rows)
    assert {p[1] for p in pts} == {"fine2"}
    rows += _fake_rows(pts, r1, erp)
    assert e8.next_points(row, r1, rows) == []


def test_search_stages_interior():
    """最良点が内側なら拡張しない."""
    row, r1 = "A_rhoB0.8", e8.R1_LEVELS[2]
    mid = e8.coarse_betas(e8.cell_params(row, r1)["sigma"])[4]
    erp = lambda s, b, nt, g: abs(np.log(b / mid)) + 1
    rows = []
    for _ in range(5):
        pts = e8.next_points(row, r1, rows)
        if not pts:
            break
        assert "extend" not in {p[1] for p in pts}
        rows += _fake_rows(pts, r1, erp)
    assert e8.next_points(row, r1, rows) == []


@pytest.fixture
def small_setup(monkeypatch):
    monkeypatch.setattr(e8, "BASELINE", dict(c=4, K=10, b=2, mu=1.0))
    monkeypatch.setattr(e8, "ROWS", {"A_rhoB0.8": (0.8, 0.1), "B_alpha1": (0.8, 1.0)})
    monkeypatch.setattr(e8, "MAP_A", ["A_rhoB0.8"])
    monkeypatch.setattr(e8, "MAP_B", ["A_rhoB0.8", "B_alpha1"])
    monkeypatch.setattr(e8, "R1_LEVELS", [0.1, 10.0])
    monkeypatch.setattr(e8, "SERIES", {
        "P2 only": ([(0, 3.0)], False),
        "P1 only (protect)": ([(2, 1.0), (4, 1.0)], True),
        "P1+P2 (protect)": ([(2, 3.0)], True),
    })
    monkeypatch.setattr(e8, "N_BETA", 5)
    monkeypatch.setattr(e8, "N_FINE", 3)
    monkeypatch.setattr(e8, "N_FINE_BASE", 5)


def test_small_run_and_report(small_setup, tmp_path):
    out = str(tmp_path / "res")
    e8.run(["A_rhoB0.8", "B_alpha1"], out_dir=out)
    for row in ("A_rhoB0.8", "B_alpha1"):
        with open(e8.csv_path(row, out), newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert all(r["check"] == "ok" for r in rows)
        checks = [r for r in rows if r["series"] == "Check"]
        assert len(checks) == 2
        assert all(float(r["consistency_max_relerr"]) <= 1e-10 for r in checks)
        for r1 in e8.R1_LEVELS:
            assert e8.next_points(row, r1, [r for r in rows if float(r["r1"]) == r1]) == []
        n = len(rows)
        e8.run([row], out_dir=out)  # 再実行しても追記されない
        with open(e8.csv_path(row, out), newline="", encoding="utf-8") as f:
            assert len(list(csv.DictReader(f))) == n

    figs = str(tmp_path / "fig")
    text = e8.report(out_dir=out, fig_dir=figs)
    for key in ("整合性の検査", "改善が 0.5% を超えるマス", "r1=0.1 の列",
                "地図 B: α と P1 only (protect) の改善", "p1_fire_rate と 1/(1+1/r1) の相関"):
        assert key in text
    for stem in ("A_erp", "A_cost", "B_erp", "B_cost", "B_p1_alpha", "p1_fire"):
        assert os.path.exists(os.path.join(figs, f"experiment_8_{stem}.png"))
        assert os.path.exists(os.path.join(figs, f"experiment_8_{stem}.pdf"))


def test_consistency_failure_stops(small_setup, tmp_path, monkeypatch):
    monkeypatch.setattr(e8, "CONSISTENCY_TOL", -1.0)
    with pytest.raises(e8.ConsistencyError):
        e8.run(["A_rhoB0.8"], out_dir=str(tmp_path))
