#!/bin/bash
# 既存の全実験 (0〜6, Predictive含む) を solver="gth" (新既定値) で再実行し,
# results_gth/ と figures_gth/ に新しい CSV・図を保存する.
# 既存の results/, figures/ は一切変更しない (--csv-dir / --out-dir で分離).
#
# 各コマンドの実行時間を計測して timing_report.txt に記録する.
set -uo pipefail
cd "$(dirname "$0")/.."

CSV=results_gth
FIG=figures_gth
TIMING=$CSV/timing_report.txt
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

PY=python3

# ---------------- ベースモデル ----------------
run "exp0" $PY scripts/experiment_0_validation.py --out-dir "$FIG"
run "exp1" $PY scripts/experiment_1_traffic.py --csv-dir "$CSV" --out-dir "$FIG"
run "exp2_medium" $PY scripts/experiment_2_delayoff.py --csv-dir "$CSV" --out-dir "$FIG"
run "exp2_strong" $PY scripts/experiment_2_delayoff.py --strong-burst --csv-dir "$CSV" --out-dir "$FIG"
run "exp3_delta" $PY scripts/experiment_3_burstiness.py --sweep delta --csv-dir "$CSV" --out-dir "$FIG"
run "exp3_sigma" $PY scripts/experiment_3_burstiness.py --sweep sigma --csv-dir "$CSV" --out-dir "$FIG"
run "exp4_all" $PY scripts/experiment_4_K_sensitivity.py --burst all --csv-dir "$CSV" --out-dir "$FIG"

echo ">>> BASE MODELS DONE <<<"

# ---------------- Predictive モデル ----------------
run "exp0P" $PY scripts/experiment_0P_validation.py --out-dir "$FIG"
run "exp1P_g5" $PY scripts/experiment_1P_traffic.py --csv-dir "$CSV" --out-dir "$FIG"
run "exp1P_g1" $PY scripts/experiment_1P_traffic.py --gamma 1.0 --csv-dir "$CSV" --out-dir "$FIG"

run "exp2P_medium_g5" $PY scripts/experiment_2P_delayoff.py --csv-dir "$CSV" --out-dir "$FIG"
run "exp2P_medium_gmap" $PY scripts/experiment_2P_delayoff.py --alpha-gamma-map "0.1:1.0,1.0:100.0,10.0:1000.0" --csv-dir "$CSV" --out-dir "$FIG"
run "exp2P_strong_g5" $PY scripts/experiment_2P_delayoff.py --strong-burst --csv-dir "$CSV" --out-dir "$FIG"
run "exp2P_strong_gmap" $PY scripts/experiment_2P_delayoff.py --strong-burst --alpha-gamma-map "0.1:1.0,1.0:100.0,10.0:1000.0" --csv-dir "$CSV" --out-dir "$FIG"

run "exp3P_delta_g5" $PY scripts/experiment_3P_burstiness.py --sweep delta --csv-dir "$CSV" --out-dir "$FIG"
run "exp3P_delta_gmap" $PY scripts/experiment_3P_burstiness.py --sweep delta --alpha-gamma-map "0.1:1.0,1.0:100.0,10.0:1000.0" --csv-dir "$CSV" --out-dir "$FIG"
run "exp3P_sigma_g5" $PY scripts/experiment_3P_burstiness.py --sweep sigma --csv-dir "$CSV" --out-dir "$FIG"
run "exp3P_sigma_gmap" $PY scripts/experiment_3P_burstiness.py --sweep sigma --alpha-gamma-map "0.1:1.0,1.0:100.0,10.0:1000.0" --csv-dir "$CSV" --out-dir "$FIG"

run "exp4P_all_g5" $PY scripts/experiment_4P_K_sensitivity.py --burst all --csv-dir "$CSV" --out-dir "$FIG"
run "exp4P_all_g1" $PY scripts/experiment_4P_K_sensitivity.py --burst all --gamma 1.0 --csv-dir "$CSV" --out-dir "$FIG"

run "exp5_all" $PY scripts/experiment_5_n_target.py --burst all --csv-dir "$CSV" --out-dir "$FIG"
run "exp6_all" $PY scripts/experiment_6_gamma.py --burst all --csv-dir "$CSV" --out-dir "$FIG"

echo ">>> ALL DONE <<<"
