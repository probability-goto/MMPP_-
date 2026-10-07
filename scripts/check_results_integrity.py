"""results_gth/ の CSV の健全性を検査する.

検査項目:
    1. P_block_arrival_stable 列が存在し, 空セルがないこと
       (DES 照合の CSV では, 理論値を写した列 theory_P_block_arrival_stable
        を代わりに検査する)
       (CSV 書き出しの dict にキーを足し忘れると DictWriter が
        空文字を書くため, 目視では気づきにくい)
    2. 安定版の値が厳密に正であること (引き算を含まないので負にならない)
    3. 参考として, 旧列 P_block_arrival (引き算版 1 - lambda_eff/lambda_bar) が
       残っている既存 CSV では, その負値・ゼロの件数を報告する
       (引き算版はライブラリから削除済みで, 新しい CSV には列自体がない)

使用例:
    python scripts/check_results_integrity.py
    python scripts/check_results_integrity.py results_gth
"""
import csv
import glob
import os
import sys

STABLE = "P_block_arrival_stable"
# DES 照合の CSV で理論値を写した列 (DES の推定値と区別するため接頭辞を付ける)
THEORY_STABLE = "theory_P_block_arrival_stable"
# 引き算版の旧列. 既存 CSV にのみ存在し, 桁落ちの報告のためだけに参照する
CURRENT = "P_block_arrival"


def check_file(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    name = os.path.basename(path)
    if not rows:
        return name, "空ファイル", True

    if STABLE in rows[0]:
        col = STABLE
    elif THEORY_STABLE in rows[0]:
        col = THEORY_STABLE
    else:
        return name, f"{STABLE} 列 ({THEORY_STABLE} 列も) が存在しない", False

    empties = sum(1 for r in rows if not str(r[col]).strip())
    if empties:
        return name, f"{col} に空セル {empties}/{len(rows)} 件", False

    stable_vals = [float(r[col]) for r in rows]
    neg_stable = [v for v in stable_vals if v < 0]
    if neg_stable:
        return name, f"{col} に負値 {len(neg_stable)} 件 (最小 {min(neg_stable):.3e})", False

    msg = f"{len(rows):>4} 行, {col} 最小={min(stable_vals):.2e}"
    if CURRENT in rows[0]:
        cur = [float(r[CURRENT]) for r in rows if str(r[CURRENT]).strip()]
        n_bad = sum(1 for v in cur if v <= 0)
        if n_bad:
            msg += f"  (旧列の引き算版は非正値 {n_bad} 件 → 桁落ち)"
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
