"""実験 9 (scripts/experiment_9_sensitivity.py) の集計用関数のテスト."""
import scripts.experiment_9_sensitivity as e9


def test_intervals():
    vals = [0, 2, 4, 6, 8]
    f = lambda v: str(v)
    assert e9.intervals(vals, [False] * 5, f) == "なし"
    assert e9.intervals(vals, [True] * 5, f) == "全域"
    assert e9.intervals(vals, [True, True, False, True, False], f) == "[0, 2] ∪ 6"


def test_category():
    assert e9.category(-0.01, 0.0) == "改善"
    assert e9.category(-0.01, 0.5) == "引き換え"
    assert e9.category(0.01, -0.5) == "引き換え"
    assert e9.category(0.01, 0.5) == "悪化"


def test_point_counts():
    assert len(e9.points_B()) == 35 * 4
    assert len(e9.points_A()) == 12 * (11 + 9 + 7)


def test_category5():
    assert e9.category5(-0.002, 1.04) == "改善"
    assert e9.category5(-0.002, 0.5) == "改善"
    assert e9.category5(0.0005, 1.0) == "同等"
    assert e9.category5(0.0005, 0.9) == "棄却のみ改善"
    assert e9.category5(-0.002, 1.2) == "引き換え"
    assert e9.category5(0.002, 0.9) == "引き換え"
    assert e9.category5(0.002, 1.0) == "悪化"
    assert e9.category5(0.0, 1.2) == "悪化"
