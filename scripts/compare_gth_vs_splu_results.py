"""V6: results/ (旧, splu) と results_gth/ (新, GTH) の CSV を比較する.

results_gth/ 配下の各 CSV について, 同名ファイルが results/ にあれば
パラメータ列で突き合わせ, 指標列の相対差を計算する。

合格基準: P_block (または P_block_arrival) >= 1e-12 の行で相対差 > 1e-4
のものがあれば報告する (閾値は緩めない)。

使用例:
    python scripts/compare_gth_vs_splu_results.py
"""
import csv
import glob
import os
import sys

METRIC_COLS = [
    "P_block", "P_block_arrival", "P_block_arrival_stable",
    "E_W", "Cost", "ERP",
    "E_N", "lambda_eff", "E_B", "E_S", "E_I", "E_off",
]
# 合格基準の対象となるブロック確率系の列。
# P_block_arrival_stable は新設列で旧 results/ には存在しないため,
# 比較対象になるのは P_block と P_block_arrival のみ。
BLOCK_COLS = ["P_block", "P_block_arrival", "P_block_arrival_stable"]


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rel_diff(a, b):
    a, b = float(a), float(b)
    denom = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / denom


def compare_file(old_path, new_path):
    old_rows = load_csv(old_path)
    new_rows = load_csv(new_path)
    if not old_rows or not new_rows:
        return None

    all_cols = set(old_rows[0]) & set(new_rows[0])
    metric_cols = [c for c in METRIC_COLS if c in all_cols]
    join_cols = [c for c in all_cols if c not in METRIC_COLS and c != "N_states"]

    def key(row):
        return tuple(row[c] for c in join_cols)

    new_by_key = {key(r): r for r in new_rows}

    max_diff = {c: 0.0 for c in metric_cols}
    violations = []
    n_matched = 0

    for old_row in old_rows:
        k = key(old_row)
        new_row = new_by_key.get(k)
        if new_row is None:
            continue
        n_matched += 1
        for c in metric_cols:
            try:
                d = rel_diff(old_row[c], new_row[c])
            except (ValueError, KeyError):
                continue
            max_diff[c] = max(max_diff[c], d)
            if c in BLOCK_COLS:
                val_new = abs(float(new_row[c]))
                if val_new >= 1e-12 and d > 1e-4:
                    violations.append((k, c, val_new, d))

    return dict(
        n_old=len(old_rows), n_new=len(new_rows), n_matched=n_matched,
        max_diff=max_diff, violations=violations,
    )


def main():
    results_dir = "results"
    gth_dir = "results_gth"

    gth_files = sorted(glob.glob(os.path.join(gth_dir, "*.csv")))
    if not gth_files:
        print(f"{gth_dir}/ に CSV が見つかりません。", file=sys.stderr)
        return 1

    overall_max = {}
    total_violations = 0
    unmatched_files = []

    for gth_path in gth_files:
        name = os.path.basename(gth_path)
        old_path = os.path.join(results_dir, name)
        if not os.path.exists(old_path):
            unmatched_files.append(name)
            continue

        result = compare_file(old_path, gth_path)
        if result is None:
            continue

        print(f"\n=== {name} ===")
        print(f"  旧(splu) 行数={result['n_old']}  新(gth) 行数={result['n_new']}  一致行数={result['n_matched']}")
        for c, d in result["max_diff"].items():
            print(f"  {c:<18} max_rel_diff = {d:.3e}")
            overall_max[c] = max(overall_max.get(c, 0.0), d)

        if result["violations"]:
            print(f"  !! 違反 {len(result['violations'])} 件 (P_block>=1e-12 かつ相対差>1e-4)")
            for k, c, val, d in result["violations"][:10]:
                print(f"     key={k} {c}={val:.3e} rel_diff={d:.3e}")
            total_violations += len(result["violations"])

    print("\n" + "=" * 60)
    print("=== 全体サマリ ===")
    for c, d in sorted(overall_max.items()):
        print(f"  {c:<18} max_rel_diff = {d:.3e}")
    print(f"\n比較対象外 (results/ に旧ファイルなし): {unmatched_files}")
    print(f"\n合計違反件数: {total_violations}")
    print("V6 (results/ vs results_gth/) result:", "PASS" if total_violations == 0 else "FAIL (要確認)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
