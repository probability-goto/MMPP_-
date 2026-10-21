"""CSV の読み書きと進捗ログ.

長い実験は 1 点ごとに CSV へ追記し, 中断しても計算済みの点を飛ばして再開できるようにする.
"""
import csv
import os
import time
from datetime import datetime
from typing import List, Sequence


def read_rows(path: str) -> List[dict]:
    """CSV を辞書のリストで読む (ファイルがなければ空)."""
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_row(path: str, fieldnames: Sequence[str], row: dict,
               extrasaction: str = "raise") -> None:
    """CSV に 1 行追記する (ファイルがなければヘッダも書く)."""
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction=extrasaction)
        if new:
            w.writeheader()
        w.writerow(row)


def log_line(path: str, msg: str) -> None:
    """時刻を付けた 1 行をログファイルに追記し, 標準出力にも出す."""
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def progress_message(done: int, total: int, t0: float, avg: float, last: str) -> str:
    """進捗の 1 行 (完了数, 経過時間, 残り時間の見積もり, 最後に計算した点)."""
    return (f"done={done}/{total} elapsed={time.time() - t0:.1f}s "
            f"eta={max(total - done, 0) * avg:.1f}s last={last}")
