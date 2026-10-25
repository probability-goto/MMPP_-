#!/usr/bin/env python
"""コードを変えても計算結果が変わらないことを, コミット済みの results/ と figures/ と照合して確かめる.

照合の方法:
    cli      実験 1〜4 を既定の設定で一時ディレクトリに実行し直し, CSV と図が results/, figures/ と
             バイト単位で一致するか (PDF は作成日時を除く).
    rows     実験 7・8・9, compare_map, robustness, p1_protect の CSV を一時ディレクトリに写して
             各ファイルから行を抜き取り, 各スクリプトの run() に計算し直させて, 計算し直した行が
             元の行と全列 (elapsed_s を除く) で一致するか. 抜き取った行以外が計算されたら失敗とする.
    reports  レポートを作るコマンド (--report など) をその場で実行し直し, results/, figures/,
             thesis_assets/ に差分が出ないか (PDF は作成日時を除く. 作成日時だけ違う PDF は元に戻す).

DES (実験 0 / 0-P, validate_protect_des の照合) は時間がかかるので対象にしない.

使用例:
    python scripts/check_reproduction.py                 # すべて
    python scripts/check_reproduction.py --only rows --n-rows 1
"""
import argparse
import csv
import importlib
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Dict, List, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # scripts.<名前> として import するため

IGNORE_COLUMNS = {"elapsed_s"}
WATCHED_DIRS = ["results", "figures", "thesis_assets"]


# ============================================================
# 共通
# ============================================================

def _strip_pdf_date(data: bytes) -> bytes:
    return re.sub(rb"/CreationDate \(D:[^)]*\)", b"", data)


def same_file(new: str, ref: str) -> bool:
    with open(new, "rb") as f:
        a = f.read()
    with open(ref, "rb") as f:
        b = f.read()
    if new.endswith(".pdf"):
        return _strip_pdf_date(a) == _strip_pdf_date(b)
    return a == b


def read_csv(path: str) -> Tuple[List[str], List[dict]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: str, fieldnames: List[str], rows: List[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def comparable(row: dict) -> Tuple:
    return tuple(sorted((k, v) for k, v in row.items() if k not in IGNORE_COLUMNS))


# ============================================================
# cli: 実験 1〜4
# ============================================================

CLI_RUNS = [
    ("experiment_1_traffic.py", []),
    ("experiment_2_delayoff.py", []),
    ("experiment_3_burstiness.py", ["--sweep", "delta"]),
    ("experiment_3_burstiness.py", ["--sweep", "r1"]),
    ("experiment_4_K_sensitivity.py", []),
]


def check_cli(tmp: str) -> List[str]:
    failures = []
    csv_dir, fig_dir = os.path.join(tmp, "csv"), os.path.join(tmp, "fig")
    for script, extra in CLI_RUNS:
        cmd = [sys.executable, os.path.join("scripts", script), *extra,
               "--csv-dir", csv_dir, "--out-dir", fig_dir]
        print("実行:", " ".join(cmd[1:]), flush=True)
        res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if res.returncode != 0:
            failures.append(f"{script} {' '.join(extra)}: 終了コード {res.returncode}\n{res.stderr[-2000:]}")
    for out_dir, ref_dir in [(csv_dir, "results"), (fig_dir, "figures")]:
        for name in sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []:
            ref = os.path.join(REPO, ref_dir, name)
            if not os.path.exists(ref):
                failures.append(f"{ref_dir}/{name}: コミット済みのファイルがない")
            elif not same_file(os.path.join(out_dir, name), ref):
                failures.append(f"{ref_dir}/{name}: 一致しない")
            else:
                print(f"  一致: {ref_dir}/{name}")
    return failures


# ============================================================
# rows: 行を抜き取って計算し直す
# ============================================================

def _rerun_e7(m, d):
    conds = list(m.CONDITIONS)
    m.run(conds, d)
    m.run_extension(conds, d)
    m.run_refine(conds, d)


def _rerun_p1(m, d):
    # p1_protect は出力先をモジュールの定数で持つので, 一時ディレクトリに差し替える
    m.OUT_DIR = d
    m.CSV_PATH = os.path.join(d, "p1_protect.csv")
    m.V1_CSV_PATH = os.path.join(d, "p1_protect_v1.csv")
    m.LOG_PATH = os.path.join(d, "progress.log")
    m.run(m.BLOCKS)


# (名前, モジュール, 元のディレクトリ, 抜き取る CSV, 計算し直す関数)
ROW_TARGETS: List[Tuple[str, str, str, List[str], Callable]] = [
    ("experiment_7", "scripts.experiment_7_frontier", "results/experiment_7",
     [f"experiment_7_{p}C{i}.csv" for p in ("", "refine_") for i in range(1, 6)], _rerun_e7),
    ("experiment_8", "scripts.experiment_8_map", "results/experiment_8",
     [f"experiment_8_A_rhoB{x}.csv" for x in ("0.6", "0.8", "1.12")] +
     [f"experiment_8_B_alpha{x}.csv" for x in ("0.005", "0.02", "1", "10")],
     lambda m, d: m.run(list(dict.fromkeys(m.MAP_A + m.MAP_B)), d)),
    ("experiment_9", "scripts.experiment_9_sensitivity", "results/experiment_9",
     ["experiment_9_A.csv", "experiment_9_B.csv"],
     lambda m, d: [m.run(p, d) for p in ("A", "B")]),
    ("compare_map", "scripts.compare_map", "results/compare_map",
     [f"compare_map_rhoB{x}.csv" for x in ("0.6", "0.8", "1.12")],
     lambda m, d: m.run(m.RHO_B_LEVELS, d)),
    ("robustness", "scripts.robustness", "results/robustness",
     ["robustness_A.csv", "robustness_B.csv", "robustness_BG.csv"],
     lambda m, d: [m.run(p, d) for p in ("A", "B", "BG")]),
    ("p1_protect", "scripts.experiment_p1_protect", "results/p1_protect",
     ["p1_protect.csv"], _rerun_p1),
]


def check_rows(tmp: str, n_rows: int, seed: int, names: List[str]) -> List[str]:
    failures = []
    rng = random.Random(seed)
    for name, modname, src, files, rerun in ROW_TARGETS:
        if names and name not in names:
            continue
        d = os.path.join(tmp, name)
        shutil.copytree(os.path.join(REPO, src), d)
        removed: Dict[str, List[dict]] = {}
        kept_n: Dict[str, int] = {}
        for fn in files:
            path = os.path.join(d, fn)
            fields, rows = read_csv(path)
            idx = sorted(rng.sample(range(len(rows)), min(n_rows, len(rows))))
            removed[fn] = [rows[i] for i in idx]
            kept = [r for i, r in enumerate(rows) if i not in set(idx)]
            kept_n[fn] = len(kept)
            write_csv(path, fields, kept)
        print(f"[{name}] {sum(len(v) for v in removed.values())} 行を抜き取って計算し直す", flush=True)
        rerun(importlib.import_module(modname), d)
        for fn in files:
            _, rows = read_csv(os.path.join(d, fn))
            new = rows[kept_n[fn]:]
            pool = [comparable(r) for r in new]
            for r in removed[fn]:
                key = comparable(r)
                if key in pool:
                    pool.remove(key)
                    continue
                # いちばん多くの列が一致する行と比べて, 違う列を示す
                cols = [k for k in r if k not in IGNORE_COLUMNS]
                near = max(new, key=lambda n: sum(n.get(k) == r[k] for k in cols), default={})
                diff = "; ".join(f"{k}: {r[k]} -> {near.get(k)}" for k in cols if near.get(k) != r[k])
                failures.append(f"{src}/{fn}: 計算し直した行が元の行と一致しない: "
                                + ", ".join(f"{k}={r[k]}" for k in list(r)[:12])
                                + f"\n    違う列: {diff}")
            if pool:
                failures.append(f"{src}/{fn}: 抜き取っていない行が {len(pool)} 行計算された")
        n_ok = sum(len(v) for v in removed.values())
        print(f"[{name}] 照合した行 {n_ok}", flush=True)
    return failures


# ============================================================
# reports: レポートをその場で作り直す
# ============================================================

REPORT_RUNS = [
    ["experiment_7_frontier.py", "--report"],
    ["experiment_8_map.py", "--report"],
    ["blocking_reassessment.py"],
    ["analyze_mechanism.py"],
    ["experiment_9_sensitivity.py", "--report"],
    ["compare_map.py", "--report"],
    ["robustness.py", "--report"],
    ["experiment_p1_protect.py", "--summary"],
    ["validate_protect_des.py", "--summary"],
    ["make_thesis_numbers.py"],
]


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout


def check_reports() -> List[str]:
    if _git("status", "--porcelain", "--", *WATCHED_DIRS).strip():
        return [f"{', '.join(WATCHED_DIRS)} に未コミットの変更がある (先にコミットするか戻す)"]
    failures = []
    for run in REPORT_RUNS:
        print("実行:", " ".join(run), flush=True)
        res = subprocess.run([sys.executable, os.path.join("scripts", run[0]), *run[1:]],
                             cwd=REPO, capture_output=True, text=True)
        if res.returncode != 0:
            failures.append(f"{' '.join(run)}: 終了コード {res.returncode}\n{res.stderr[-2000:]}")
    for line in _git("status", "--porcelain", "--", *WATCHED_DIRS).splitlines():
        status, path = line[:2], line[3:]
        if status.strip() == "M" and path.endswith(".pdf"):
            with open(os.path.join(REPO, path), "rb") as f:
                new = f.read()
            old = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=REPO,
                                 capture_output=True, check=True).stdout
            if _strip_pdf_date(new) == _strip_pdf_date(old):
                _git("checkout", "--", path)
                continue
        failures.append(f"{path}: 差分がある ({status.strip()})")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["cli", "rows", "reports"], action="append",
                        help="照合の方法を選ぶ (複数指定可. 既定はすべて)")
    parser.add_argument("--n-rows", type=int, default=2, help="rows で 1 ファイルから抜き取る行数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", action="append", default=[],
                        help="rows の対象を名前で絞る (" + ", ".join(t[0] for t in ROW_TARGETS) + ")")
    args = parser.parse_args()
    modes = args.only or ["cli", "rows", "reports"]

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        if "cli" in modes:
            failures += check_cli(os.path.join(tmp, "cli"))
        if "rows" in modes:
            failures += check_rows(os.path.join(tmp, "rows"), args.n_rows, args.seed, args.target)
    if "reports" in modes:
        failures += check_reports()

    print()
    if failures:
        print(f"不一致 {len(failures)} 件:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("すべて一致した:", ", ".join(modes))


if __name__ == "__main__":
    main()
