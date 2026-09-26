#!/bin/bash
# GTH 再計算の進捗を一覧表示する。
#
# results/ にある既存 CSV (旧 splu 結果) を「あるべき成果物」の基準とし、
# results_gth/ に同名ファイルが揃っているかで進捗を判定する。
#
# 使用例: bash scripts/show_progress.sh
cd "$(dirname "$0")/.."

OLD=results
NEW=results_gth
LOGDIR=.gth_logs

echo "=================================================="
echo " GTH 再計算 進捗   $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="

# --- 実行中プロセス ---
echo
running=$(pgrep -af "scripts/experiment_|python3 -m pytest" \
    | grep -vE "pgrep|show_progress|claude-[0-9a-f]+-cwd" \
    | sed 's/.*scripts\///;s/.*python3 -m /python3 -m /')
if [ -n "$running" ]; then
    echo "▶ 稼働中:"
    echo "$running" | while read -r line; do echo "    $line"; done
else
    echo "■ 稼働中のプロセスなし (停止中)"
fi

# --- テストスイートの進捗 ---
if [ -f "$LOGDIR/pytest.log" ]; then
    echo
    echo "--- テストスイート ---"
    tlog="$LOGDIR/pytest.log"
    # pytest -q の進捗文字 (. F s x E) を数えて完了数とする
    n_done=$(tr -d '\n ' < "$tlog" | tr -cd '.FsxE' | wc -c)
    n_fail=$(tr -d '\n ' < "$tlog" | tr -cd 'F' | wc -c)
    mtime=$(stat -c %y "$tlog" 2>/dev/null | cut -d. -f1)
    if grep -qE "passed|failed|error" "$tlog"; then
        echo "    完了: $(grep -E 'passed|failed' "$tlog" | tail -1)"
    else
        echo "    進行中: ${n_done}/226 件 (失敗 ${n_fail} 件)  最終更新 ${mtime}"
    fi
fi

# --- 個別ファイルの進捗 ---
echo
total=0; done_n=0
declare -a missing=()
while IFS= read -r path; do
    name=$(basename "$path")
    total=$((total + 1))
    if [ -f "$NEW/$name" ]; then
        # 安定版列が空でないかも確認する
        if head -2 "$NEW/$name" | tail -1 | grep -q ",," ; then
            mark="△"; note=" (列に空あり)"
        else
            mark="✓"; note=""
        fi
        done_n=$((done_n + 1))
    else
        mark="—"; note=""
        missing+=("$name")
    fi
    printf "  %s %-45s%s\n" "$mark" "$name" "$note"
done < <(ls "$OLD"/*.csv 2>/dev/null | sort)

# --- 進捗バー ---
echo
if [ "$total" -gt 0 ]; then
    pct=$((done_n * 100 / total))
    filled=$((pct / 5))
    bar=$(printf '█%.0s' $(seq 1 $filled 2>/dev/null))
    empty=$(printf '░%.0s' $(seq 1 $((20 - filled)) 2>/dev/null))
    echo "  進捗: [${bar}${empty}] ${done_n}/${total} (${pct}%)"
fi

# --- 残り作業と所要時間の見積もり ---
if [ ${#missing[@]} -gt 0 ]; then
    echo
    echo "  残り ${#missing[@]} ファイル:"
    est=0
    for m in "${missing[@]}"; do
        # 実測に基づく所要時間 (分)。Predictive は n_target/gamma が大きいほど
        # 到達可能状態が増えて帯幅も広がるため、点あたりのコストが変わる。
        case "$m" in
            experiment_5_*)  t=16 ;;  # 実測 944 s (n_target 15 水準 x alpha 3 水準)
            experiment_6_*)  t=16 ;;  # gamma 15 水準 x alpha 3 水準
            experiment_4P_*) t=23 ;;  # K=100..1000 x 15 rho (K=1000 が約 66 s/点)
            experiment_1P_*) t=3 ;;   # 実測 約 200 s
            experiment_2P_*|experiment_3P_*) t=3 ;;
            *) t=1 ;;
        esac
        est=$((est + t))
        printf "      %-45s 推定 %2d 分\n" "$m" "$t"
    done
    echo
    echo "  推定残り時間: 約 ${est} 分"
fi

# --- 直近の完了ログ ---
echo
echo "--- 直近の完了ステップ ---"
for f in "$NEW"/timing_report*.txt; do
    [ -f "$f" ] || continue
    echo "  [$(basename "$f")]"
    paste -d' ' <(grep "^=== " "$f" | sed 's/=== //;s/ ===//') \
                <(grep "exit=" "$f" | sed 's/^ *//') 2>/dev/null | tail -5 | sed 's/^/    /'
done
echo
