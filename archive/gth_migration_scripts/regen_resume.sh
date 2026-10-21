#!/bin/bash
# 未完了の実験だけを実行する冪等な再開スクリプト。
#
# 出力 CSV が既に results_gth/ にあればスキップするため、何度実行しても
# 続きから進む。セッションが切れてバックグラウンドジョブが止まっても、
# もう一度これを実行すれば残りだけを処理する。
#
# 進捗確認: bash scripts/show_progress.sh
#
# 使用例:
#   bash scripts/regen_resume.sh          # 残り全部
#   bash scripts/regen_resume.sh exp5     # 実験 5 だけ
set -uo pipefail
cd "$(dirname "$0")/.."

CSV=results_gth
FIG=figures_gth
TIMING=$CSV/timing_report_resume.txt
mkdir -p "$CSV" "$FIG"
PY="python3 -u"
ONLY="${1:-all}"

# 出力 CSV が揃っていれば skip する
need() {
    for f in "$@"; do
        [ -f "$CSV/$f" ] || return 0        # 1 つでも無ければ実行が必要
        head -2 "$CSV/$f" | tail -1 | grep -q ",," && return 0   # 空列があれば要再実行
    done
    return 1
}

run() {
    local label="$1"; shift
    echo "=== $label 開始 $(date +%H:%M:%S) ==="
    echo "=== $label ===" >> "$TIMING"
    local s=$(date +%s)
    "$@"
    local ec=$?
    local e=$(date +%s)
    echo "=== $label 完了 exit=$ec $((e-s))s ==="
    echo "  exit=$ec elapsed=$((e-s))s" >> "$TIMING"
}

want() { [ "$ONLY" = "all" ] || [ "$ONLY" = "$1" ]; }

# ---- 実験 5 (burst 水準ごとに分割して中断に強くする) ----
for b in weak medium strong; do
    want exp5 || break
    if need "experiment_5_${b}_g5.0.csv"; then
        run "exp5_$b" $PY scripts/experiment_5_n_target.py --burst "$b" --csv-dir "$CSV" --out-dir "$FIG"
    else
        echo "--- exp5_$b: 完了済みのためスキップ"
    fi
done

# ---- 実験 6 ----
for b in weak medium strong; do
    want exp6 || break
    if need "experiment_6_${b}_nt10.csv"; then
        run "exp6_$b" $PY scripts/experiment_6_gamma.py --burst "$b" --csv-dir "$CSV" --out-dir "$FIG"
    else
        echo "--- exp6_$b: 完了済みのためスキップ"
    fi
done

# ---- 実験 4P (最も重い。burst 水準 x gamma ごとに分割) ----
for g in 5.0 1.0; do
    for b in weak medium strong; do
        want exp4P || break 2
        if need "experiment_4P_${b}_nt10_g${g}.csv"; then
            run "exp4P_${b}_g${g}" $PY scripts/experiment_4P_K_sensitivity.py \
                --burst "$b" --gamma "$g" --csv-dir "$CSV" --out-dir "$FIG"
        else
            echo "--- exp4P_${b}_g${g}: 完了済みのためスキップ"
        fi
    done
done

echo ">>> RESUME BATCH DONE <<<"
bash scripts/show_progress.sh
