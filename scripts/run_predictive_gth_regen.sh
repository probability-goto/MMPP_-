#!/bin/bash
# Predictive 実験 (0P〜6) を solver="gth" (新既定値) で再実行し,
# results_gth/ と figures_gth/ に保存する.
# ベースモデル (実験 0〜4) は run_all_gth_regen.sh で完了済みのため含まない.
#
# セッションから切り離して実行する想定:
#   setsid nohup bash scripts/run_predictive_gth_regen.sh > <log> 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

CSV=results_gth
FIG=figures_gth
TIMING=$CSV/timing_report_predictive.txt
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
GMAP="0.1:1.0,1.0:100.0,10.0:1000.0"

run "exp1P_g5" $PY scripts/experiment_1P_traffic.py --csv-dir "$CSV" --out-dir "$FIG"
run "exp1P_g1" $PY scripts/experiment_1P_traffic.py --gamma 1.0 --csv-dir "$CSV" --out-dir "$FIG"

run "exp2P_medium_g5" $PY scripts/experiment_2P_delayoff.py --csv-dir "$CSV" --out-dir "$FIG"
run "exp2P_medium_gmap" $PY scripts/experiment_2P_delayoff.py --alpha-gamma-map "$GMAP" --csv-dir "$CSV" --out-dir "$FIG"
run "exp2P_strong_g5" $PY scripts/experiment_2P_delayoff.py --strong-burst --csv-dir "$CSV" --out-dir "$FIG"
run "exp2P_strong_gmap" $PY scripts/experiment_2P_delayoff.py --strong-burst --alpha-gamma-map "$GMAP" --csv-dir "$CSV" --out-dir "$FIG"

run "exp3P_delta_g5" $PY scripts/experiment_3P_burstiness.py --sweep delta --csv-dir "$CSV" --out-dir "$FIG"
run "exp3P_delta_gmap" $PY scripts/experiment_3P_burstiness.py --sweep delta --alpha-gamma-map "$GMAP" --csv-dir "$CSV" --out-dir "$FIG"
run "exp3P_sigma_g5" $PY scripts/experiment_3P_burstiness.py --sweep sigma --csv-dir "$CSV" --out-dir "$FIG"
run "exp3P_sigma_gmap" $PY scripts/experiment_3P_burstiness.py --sweep sigma --alpha-gamma-map "$GMAP" --csv-dir "$CSV" --out-dir "$FIG"

run "exp4P_all_g5" $PY scripts/experiment_4P_K_sensitivity.py --burst all --csv-dir "$CSV" --out-dir "$FIG"
run "exp4P_all_g1" $PY scripts/experiment_4P_K_sensitivity.py --burst all --gamma 1.0 --csv-dir "$CSV" --out-dir "$FIG"

run "exp5_all" $PY scripts/experiment_5_n_target.py --burst all --csv-dir "$CSV" --out-dir "$FIG"
run "exp6_all" $PY scripts/experiment_6_gamma.py --burst all --csv-dir "$CSV" --out-dir "$FIG"

# 実験 0P (理論 vs DES シミュレーション) は計算が重く (実験 0 で 32 分),
# CSV を出力せず図のみのため最後に回す.
run "exp0P" $PY scripts/experiment_0P_validation.py --out-dir "$FIG"

echo ">>> PREDICTIVE ALL DONE <<<"
