"""results_gth/ の CSV の健全性を検査する.

検査項目:
    1. P_block_arrival_stable 列が存在し, 空セルがないこと
       (CSV 書き出しの dict にキーを足し忘れると DictWriter が
        空文字を書くため, 目視では気づきにくい)
    2. 安定版の値が厳密に正であること (引き算を含まないので負にならない)
    3. 参考として現行式 P_block_arrival の負値・ゼロの件数を報告する

使用例:
    python scripts/check_results_integrity.py
    python scripts/check_results_integrity.py results_gth
"""
import csv
import glob
import os
import sys

STABLE = "P_block_arrival_stable"
CURRENT = "P_block_arrival"


def check_file(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    name = os.path.basename(path)
    if not rows:
        return name, "空ファイル", True

    if STABLE not in rows[0]:
        return name, f"{STABLE} 列が存在しない", False

    empties = sum(1 for r in rows if not str(r[STABLE]).strip())
    if empties:
        return name, f"{STABLE} に空セル {empties}/{len(rows)} 件", False

    stable_vals = [float(r[STABLE]) for r in rows]
    neg_stable = [v for v in stable_vals if v < 0]
    if neg_stable:
        return name, f"{STABLE} に負値 {len(neg_stable)} 件 (最小 {min(neg_stable):.3e})", False

    msg = f"{len(rows):>4} 行, {STABLE} 最小={min(stable_vals):.2e}"
    if CURRENT in rows[0]:
        cur = [float(r[CURRENT]) for r in rows if str(r[CURRENT]).strip()]
        n_bad = sum(1 for v in cur if v <= 0)
        if n_bad:
            msg += f"  (現行式は非正値 {n_bad} 件 → 桁落ち)"
    return name, msg, True


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "results_gth"
    files = sorted(glob.glob(os.path.join(target, "*.csv")))
    if not files:
        print(f"{target}/ に CSV がありません。", file=sys.stderr)
        return 1

    bad = []
    for path in files:
        name, msg, ok = check_file(path)
        mark = " " if ok else "!"
        print(f"{mark} {name:<45} {msg}")
        if not ok:
            bad.append(name)

    print()
    if bad:
        print(f"問題のあるファイル {len(bad)} 件: {bad}")
        return 1
    print(f"全 {len(files)} ファイルで安定版列が正常です。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
