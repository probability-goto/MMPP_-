"""scripts/compare_map.py (比較の地図, alpha=0.005, Base と Predictive) のテスト."""
import pytest

from mmpp_experiments.csvio import read_rows
from scripts import compare_map as cm
from scripts import experiment_8_map as e8

# quick 設定: Predictive の組を 3 つに減らす (保護あり)
QUICK_COMBOS = [(0, 10.0), (5, 1.0), (5, 10.0)]
QUICK_CELL = (0.6, cm.R1_LEVELS[3])


@pytest.fixture(scope="module")
def quick_cell_rows(tmp_path_factory):
    out_dir = str(tmp_path_factory.mktemp("compare_map"))
    rho_B, r1 = QUICK_CELL
    stats = cm.run_cell(rho_B, r1, out_dir, combos=QUICK_COMBOS, reuse=False)
    rows = read_rows(cm.csv_path(rho_B, out_dir))
    assert stats["computed"] == len(rows)
    assert not cm.next_points(rho_B, r1, rows, QUICK_COMBOS)
    return rows


def test_quick_cell_consistency_and_checks(quick_cell_rows):
    """整合性の検査 (Predictive (0, 1) とベースモデル) を通り, 全点の check が ok."""
    check = [r for r in quick_cell_rows if r["series"] == "Check"]
    assert len(check) == 1
    assert float(check[0]["consistency_max_relerr"]) <= cm.CONSISTENCY_TOL
    assert all(r["check"] == "ok" for r in quick_cell_rows)
    assert all(r["source"] == "computed" for r in quick_cell_rows)
    pred = [r for r in quick_cell_rows if r["series"] == "Predictive"]
    assert all(r["protect_presetup"] == "True" and r["protect_delayoff"] == "True" for r in pred)


def test_quick_cell_predictive_not_worse_than_base(quick_cell_rows):
    """Predictive の最良点の ERP <= Base の ERP 最小値 (Base の点を候補に含めるので必ず成り立つ)."""
    c = cm.cell_summary(cm._convert([dict(r) for r in quick_cell_rows]))
    assert c["pred"]["ERP"] <= c["base"]["ERP"]
    assert c["pred"]["P_block_arrival_stable"] <= c["p_ref"]
    assert c["improvement"] >= 0.0
    assert c["unc"]["ERP"] <= c["pred"]["ERP"]


@pytest.mark.parametrize("kind", ["protect", "p2_only_unprotected"])
def test_reused_point_matches_recomputation(kind):
    """再利用する既存の点と, 同じパラメータで保護ありで計算し直した値が相対誤差 1e-10 以下で一致する.

    p2_only_unprotected は, 保護なしで計算された n_target=0 の点 (保護ありとして再利用する).
    """
    index = cm.load_reuse_index(0.8)
    if not index:
        pytest.skip("再利用元の CSV がない")
    r1 = cm.R1_LEVELS[2]
    beta = cm.coarse_betas(cm.ALPHA / r1)[3]
    nt, g = (10, 30.0) if kind == "protect" else (0, 100.0)
    src = index[cm.pkey(r1, "Predictive", beta, nt, g)]
    assert (src["protect_presetup"] == "True") == (kind == "protect")
    cp = cm.cell_params(0.8, r1)
    new, _ = e8.compute_predictive("t", cp, "Predictive", "coarse", beta, nt, g, True)
    for k in e8.METRIC_FIELDS + e8.DIAG_FIELDS:
        a, b = float(src[k]), float(new[k])
        rel = 0.0 if max(abs(a), abs(b)) == 0 else abs(a - b) / max(abs(a), abs(b))
        assert rel <= 1e-10, (k, a, b)
