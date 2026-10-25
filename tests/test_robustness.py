"""scripts/robustness.py (頑健性, alpha=0.005) のテスト."""
import pytest

from scripts import robustness as rb

TOL = 1e-10
SM = rb.load_summary()
A_CELLS = [k for k, _ in rb.select_A_cells(SM)]


def _rel(a, b):
    return 0.0 if max(abs(a), abs(b)) == 0 else abs(a - b) / max(abs(a), abs(b))


def test_select_A_cells():
    """改善率の上位 2 マス, 対照 1 (rho_B=0.8, r1=0.1), 対照 2 (rho_B=1.12 の行の最小) を重複なく選ぶ."""
    sel = rb.select_A_cells(SM)
    keys = [k for k, _ in sel]
    assert len(set(keys)) == 4
    by_imp = sorted(SM, key=lambda k: -SM[k]["improvement"])
    assert keys[:2] == by_imp[:2]
    assert keys[2] == (0.8, 0.1)
    row = [k for k in SM if k[0] == 1.12]
    assert SM[keys[3]]["improvement"] == min(SM[k]["improvement"] for k in row)


def test_point_counts():
    assert len(rb.points_A(SM)) == 4 * (7 + 11 + 9)
    assert len(rb.points_B(SM)) == 15 * 4


@pytest.mark.parametrize("cell", A_CELLS, ids=[rb.cell_tag(k) for k in A_CELLS])
def test_reference_point_matches_summary(cell):
    """パート A の基準点 (beta x1, n_target=n_target*, gamma=gamma*) が summary.csv の最良点と一致する."""
    s = SM[cell]
    pts = [p for p in rb.points_A(SM) if (p["rho_B"], p["r1"]) == cell]
    ref = [p for p in pts if (p["sweep"], p["value"]) in
           [("beta_mult", 1.0), ("n_target", s["pred_n_target"]), ("gamma", s["pred_gamma"])]]
    # n_target* が奇数 (15 など) のとき, n_target の格子 {0, 2, ..., 20} には基準点が入らない
    assert len(ref) == 2 + (s["pred_n_target"] in rb.NT_SWEEP)
    for p in ref:
        assert (p["beta"], p["n_target"], p["gamma"]) == \
            (s["pred_beta"], s["pred_n_target"], s["pred_gamma"])
    out = rb.compute("A", ref[0], SM)
    assert out["check"] == "ok"
    for k, sk in [("ERP", "pred_ERP"), ("E_W", "pred_E_W"), ("Cost", "pred_Cost"),
                  ("P_block_arrival_stable", "pred_P")]:
        assert _rel(out[k], s[sk]) <= TOL, (k, out[k], s[sk])
    assert _rel(out["p_ratio"], s["pred_P_over_Pref"]) <= TOL
    assert abs(out["erp_rel"] + s["improvement"]) <= TOL


@pytest.mark.parametrize("cell", [(0.6, rb.R1_LEVELS[3]), (0.8, rb.R1_LEVELS[0]),
                                  (1.12, rb.R1_LEVELS[2])])
def test_part_B_neutral_config_matches_base(cell):
    """パート B で (n_target, gamma) = (0, 1) を加えると, 同じ beta* の Base と一致し「同等」になる."""
    s = SM[cell]
    p = dict(rho_B=cell[0], r1=cell[1], config="neutral", rule="なし", beta=s["base_beta"],
             n_target=0, gamma=1.0)
    out = rb.compute("B", p, SM)
    assert out["check"] == "ok"
    for k, sk in [("ERP", "base_ERP"), ("E_W", "base_E_W"), ("Cost", "base_Cost"),
                  ("P_block_arrival_stable", "P_ref")]:
        assert _rel(out[k], s[sk]) <= TOL, (k, out[k], s[sk])
    assert out["category"] == "同等"


def test_betagamma_points():
    """βγ の検査の点: パート A にない n_target* = c のマスで, 最良点と beta*gamma が同じ点を 2 点ずつ."""
    c = 20
    pts = rb.points_BG(SM)
    bg = [p for p in pts if p["kind"] == "βγ"]
    want = {k for k in SM if SM[k]["pred_n_target"] == c} - set(A_CELLS)
    assert {(p["rho_B"], p["r1"]) for p in bg} == want
    for k in want:
        assert len([p for p in bg if (p["rho_B"], p["r1"]) == k]) == 2
    for p in bg:
        assert p["n_target"] == c and p["gamma"] != p["ref_gamma"]
        assert _rel(p["beta"] * p["gamma"], p["ref_beta"] * p["ref_gamma"]) <= TOL
    ctrl = [p for p in pts if p["kind"] != "βγ"]
    assert len(ctrl) <= 1
    for p in ctrl:
        assert p["n_target"] == c - 2 and p["gamma"] != p["ref_gamma"]
        assert _rel(p["beta"] * p["gamma"], p["ref_beta"] * p["ref_gamma"]) <= TOL
