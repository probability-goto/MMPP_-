"""scripts/make_thesis_numbers.py (論文用の数値・表・図の生成) のテスト."""
import os
import re

import pytest

import scripts.make_thesis_numbers as mt


def test_macro_names():
    m = mt.Macros()
    m.add("ExpEightMaxImprovement", "3.79\\%", "出典")
    with pytest.raises(ValueError):
        m.add("ExpEightMaxImprovement", "1", "重複")
    with pytest.raises(ValueError):
        m.add("Exp8Max", "1", "数字を含む")
    assert mt.word_num("C1") == "COne"


def test_generate(tmp_path, monkeypatch):
    if not os.path.exists(os.path.join("results", "experiment_9", "experiment_9_B.csv")):
        pytest.skip("実験の結果がない")
    monkeypatch.setattr(mt, "OUT", str(tmp_path))
    mt.main()
    text = (tmp_path / "numbers.tex").read_text(encoding="utf-8")
    lines = text.splitlines()
    names = []
    for k, line in enumerate(lines):
        if line.startswith("\\newcommand"):
            names.append(re.match(r"\\newcommand\{\\([A-Za-z]+)\}", line).group(1))
            # 直前の行は出典と計算方法のコメント
            assert lines[k - 1].startswith("% ")
    assert len(names) == len(set(names)) > 100
    for key in ("ExpZeroBaseInCI", "ExpSevenCOneBaseERPMin", "ExpEightMaxImprovement",
                "ExpEightRPearsonAll", "ExpNinePOneNTenImprovedCount", "BenchPredGTHs"):
        assert key in names
    tables = sorted(os.listdir(tmp_path / "tables"))
    assert tables == sorted(["exp7_erp.tex", "exp7_samerel.tex", "exp8_mapA.tex", "exp8_mapB.tex",
                             "exp8_mechanism.tex", "exp9_partB.tex", "exp9_beta2.tex",
                             "protect_control.tex"])
    for t in tables:
        body = (tmp_path / "tables" / t).read_text(encoding="utf-8")
        assert "\\begin{table}" in body and "\\end{table}" in body
        # 表が参照するマクロは numbers.tex で定義されている
        for used in re.findall(r"\\(Exp[A-Za-z]+)", body):
            assert used in names, used
    figs = os.listdir(tmp_path / "figures")
    assert any(f.startswith("experiment_8_A_erp") for f in figs)
    assert (tmp_path / "README.md").exists()
