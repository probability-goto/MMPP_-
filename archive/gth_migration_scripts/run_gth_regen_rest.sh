#!/bin/bash
# run_gth_regen_v2.sh の残り (実験 5, 6, 4P) + 実験 1-P の再実行。
#
# 実験 1-P は初回実行時に CSV 書き出しの dict へ P_block_arrival_stable キーを
# 足し忘れていたため当該列が空になっていた (スクリプトは修正済み)。
#
# 実験 4P は K=1000 の Predictive モデルが約 66 秒/点と最も重いため最後に置く。
#
# セッションから切り離して実行する:
#   setsid nohup bash scripts/run_gth_regen_rest.sh > <log> 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

CSV=results_gth
FIG=figures_gth
TIMING=$CSV/timing_report_rest.txt
mkdir -p "$CSV" "$FIG"
: > "$TIMING"

run() {
    local label="$1"; shift
    echo "=== $label ==="
    echo "=== $label ===" >> "$TIMING"
    local start=$(date +%s)
    "$@"
    local ec=$?
    local end=$(date +%s)
    echo "  [$label] exit=$ec elapsed=$((end-start))s"
    echo "  exit=$ec elapsed=$((end-start))s" >> "$TIMING"
}

PY="python3 -u"

# 実験 1-P の再実行 (安定版列の修正を反映)
run "exp1P_g5_redo" $PY scripts/experiment_1P_traffic.py --csv-dir "$CSV" --out-dir "$FIG"
run "exp1P_g1_redo" $PY scripts/experiment_1P_traffic.py --gamma 1.0 --csv-dir "$CSV" --out-dir "$FIG"

# 中断した実験 5 をやり直し (weak のみ完了していたため 3 水準まとめて再実行)
run "exp5_all" $PY scripts/experiment_5_n_target.py --burst all --csv-dir "$CSV" --out-dir "$FIG"
run "exp6_all" $PY scripts/experiment_6_gamma.py --burst all --csv-dir "$CSV" --out-dir "$FIG"
echo ">>> 5,6 DONE <<<"

# 最も重い実験 4P
run "exp4P_all_g5" $PY scripts/experiment_4P_K_sensitivity.py --burst all --csv-dir "$CSV" --out-dir "$FIG"
run "exp4P_all_g1" $PY scripts/experiment_4P_K_sensitivity.py --burst all --gamma 1.0 --csv-dir "$CSV" --out-dir "$FIG"

echo ">>> ALL DONE <<<"
