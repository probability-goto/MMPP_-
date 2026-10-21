#!/bin/bash
# 再生成の仕上げ:
#   1. 実験 1-P を再実行する
#      (初回実行時, CSV 書き出しの dict に P_block_arrival_stable キーを
#       足し忘れていたため当該列が空になっていた。スクリプトは修正済み。)
#   2. 全 CSV の健全性を検査する
#   3. results/ (旧 splu) と results_gth/ (新 GTH) を比較する (V6)
#   4. 旧新比較図を生成する
set -uo pipefail
cd "$(dirname "$0")/.."

CSV=results_gth
FIG=figures_gth
PY="python3 -u"

echo "=== 1. 実験 1-P の再実行 ==="
$PY scripts/experiment_1P_traffic.py --csv-dir "$CSV" --out-dir "$FIG"
$PY scripts/experiment_1P_traffic.py --gamma 1.0 --csv-dir "$CSV" --out-dir "$FIG"

echo
echo "=== 2. CSV 健全性検査 ==="
python3 scripts/check_results_integrity.py "$CSV"

echo
echo "=== 3. V6 比較 (results/ vs results_gth/) ==="
python3 scripts/compare_gth_vs_splu_results.py

echo
echo "=== 4. 旧新比較図の生成 ==="
python3 scripts/make_gth_vs_splu_figures.py 2>&1 | grep -v Warning

echo
echo ">>> FINALIZE DONE <<<"
