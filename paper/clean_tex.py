#!/usr/bin/env python
"""日本語の LaTeX 論文から, 日本語と英数字・インライン数式の間などにある不要な半角スペースを削除する.

削除する空白 (同じ行の中にある半角スペース U+0020 の並びだけ):
    日本語 ␣ 日本語           評価 する          -> 評価する
    日本語 ␣ 英数字           実験 4 を          -> 実験4を
    英数字 ␣ 日本語           Delayoff 率        -> Delayoff率
    日本語 ␣ インライン数式    サーバー台数 $c=20$ -> サーバー台数$c=20$   ($...$ と \\(...\\))
    インライン数式 ␣ 日本語    約 $200$ で        -> 約$200$で

削除しない空白:
    - 英単語どうしの空白 (Fig. 1, machine learning). 日本語との境目は削除する (Fig. 1 に -> Fig. 1に).
    - 数式 ($...$, \\(...\\), \\[...\\], $$...$$, equation・align などの環境) の内部.
    - % から始まるコメント, プリアンブル, \\end{document} より後.
    - コマンド名の直後 (\\item 日本語 など), \\ref{...}・\\cite{...} などのコマンドと引数,
      \\begin{...}・\\end{...}, \\verb, verbatim 系の環境, \\newcommand などの定義.
    - PROTECTED_ENVS の環境 (表紙 titlepage, tabbing, 参考文献 thebibliography など).
    - 全角スペース, タブ, 行頭のインデント, 行末の空白, 改行.
    - 解析できない構文 (段落の中で閉じていない $ など) を含む段落.

\\input・\\include などで読み込まれる .tex も再帰的にたどる. 書き換える前に, 処理前と処理後を
一時フォルダでコンパイルしてエラーが増えないことを確かめ, 元のファイルをバックアップしてから書き込む.
書き込んだ後にもう一度コンパイルし, エラーが増えていれば自動でバックアップから戻す.

使用例 (論文のフォルダで実行する):
    python clean_tex.py --dry-run                 # 変更予定箇所を表示するだけ (何も書き換えない)
    python clean_tex.py --dry-run --show-skipped  # 削除しなかった候補も表示する
    python clean_tex.py --dry-run --diff-out changes.diff   # 差分をファイルに保存する
    python clean_tex.py                           # 確認のうえ, バックアップを取って書き換える
    python clean_tex.py --restore                 # 最新のバックアップから元に戻す
    python clean_tex.py --list-backups            # バックアップの一覧
"""
from __future__ import annotations

import argparse
import atexit
import bisect
import datetime
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# ----------------------------------------------------------------------
# 設定 (必要に応じて書き換える)
# ----------------------------------------------------------------------

# 中身を丸ごと保護する数式環境
MATH_ENVS = {
    "equation", "equation*", "align", "align*", "alignat", "alignat*",
    "flalign", "flalign*", "gather", "gather*", "multline", "multline*",
    "eqnarray", "eqnarray*", "displaymath", "math", "dmath", "dmath*",
    "array", "split", "cases", "matrix", "pmatrix", "bmatrix", "Bmatrix",
    "vmatrix", "Vmatrix", "smallmatrix",
}
# 中身を解析せずに丸ごと保護する環境 (\end{...} を文字列として探す)
VERBATIM_ENVS = {
    "verbatim", "verbatim*", "Verbatim", "Verbatim*", "BVerbatim", "LVerbatim",
    "lstlisting", "minted", "comment", "alltt", "filecontents", "filecontents*",
    "luacode", "luacode*",
}
# 本文だが空白に意図がありうるので丸ごと保護する環境
PROTECTED_ENVS = {
    "titlepage", "tabbing", "thebibliography", "picture", "tikzpicture", "algorithmic",
}
# \begin{env} の直後の引数のうち, 文章でないもの (s: *, o: [...], m: {...})
ENV_ARGS = {
    "tabular": "om", "tabular*": "mom", "tabularx": "mom", "tabulary": "mom",
    "longtable": "om", "figure": "o", "figure*": "o", "table": "o", "table*": "o",
    "minipage": "ooom", "subfigure": "om", "subtable": "om", "wrapfigure": "omom",
    "multicols": "m", "enumerate": "o", "itemize": "o", "description": "o",
}
# コマンドと引数を丸ごと保護するコマンド (f: {...} またはスペースで区切られたファイル名)
PROTECTED_CMDS = {
    # 相互参照・引用
    "label": "m", "ref": "sm", "eqref": "m", "pageref": "sm", "autoref": "sm",
    "nameref": "sm", "cref": "sm", "Cref": "sm", "cpageref": "sm", "Cpageref": "sm",
    "vref": "sm", "cite": "soom", "citep": "soom", "citet": "soom", "citealp": "soom",
    "citealt": "soom", "citeauthor": "soom", "citeyear": "soom", "parencite": "soom",
    "textcite": "soom", "autocite": "soom", "footcite": "soom", "nocite": "m",
    "bibitem": "om",
    # ファイル
    "input": "f", "include": "m", "includeonly": "m", "subfile": "m",
    "import": "smm", "subimport": "smm", "InputIfFileExists": "m",
    "includegraphics": "soom", "graphicspath": "m", "bibliography": "m",
    "bibliographystyle": "m", "addbibresource": "om",
    # パッケージ・定義・カウンタ
    "documentclass": "om", "usepackage": "om", "RequirePackage": "om",
    "newcommand": "smoom", "renewcommand": "smoom", "providecommand": "smoom",
    "DeclareRobustCommand": "smoom", "NewDocumentCommand": "mmm",
    "RenewDocumentCommand": "mmm", "newenvironment": "smoomm",
    "renewenvironment": "smoomm", "newtheorem": "smomo", "DeclareMathOperator": "smm",
    "newcounter": "mo", "setcounter": "mm", "addtocounter": "mm", "newlength": "m",
    "setlength": "mm", "addtolength": "mm", "settowidth": "mm", "definecolor": "mmm",
    "hypersetup": "m", "geometry": "m", "pagestyle": "m", "thispagestyle": "m",
    "pagenumbering": "m",
    # 寸法・色・表
    "vspace": "sm", "hspace": "sm", "rule": "omm", "raisebox": "moo", "color": "om",
    "textcolor": "om", "colorbox": "om", "fcolorbox": "omm", "multicolumn": "mm",
    "multirow": "omomo", "cline": "m",
}
# 引数を中括弧の対応だけで読み, 中身を解析しないコマンド (URL には % や # が入りうる)
RAW_ARG_CMDS = {"url", "path", "nolinkurl", "href"}
# 本体を保護する定義コマンド (\def\foo#1{...})
DEF_CMDS = {"def", "gdef", "edef", "xdef"}
# 現れたら解析が不確実になるコマンド
UNCERTAIN_CMDS = {"catcode", "MakeShortVerb", "DeclareShortVerb", "obeyspaces", "obeylines"}

# \input などで読み込むファイルのコマンドと引数の正規表現
INCLUDE_RE = re.compile(
    r"\\(input|include|subfile|InputIfFileExists)(?![A-Za-z])\s*(?:\{([^{}]+)\}|([^\s{}%\\]+))"
    r"|\\(import|subimport)(?![A-Za-z])\*?\s*\{([^{}]*)\}\s*\{([^{}]+)\}"
)

BACKUP_DIRNAME = ".clean_tex_backup"
CHECK_JOBNAME = "clean_tex_check"
CONTEXT = 18  # 変更箇所の表示で前後に出す文字数

# ----------------------------------------------------------------------
# 文字の判定
# ----------------------------------------------------------------------


def is_ja(c: str) -> bool:
    """日本語の文字 (かな・漢字・和文の記号・全角英数・半角カナ). 全角スペース U+3000 は含めない."""
    o = ord(c)
    return (
        0x3001 <= o <= 0x303F         # 和文の句読点・括弧・々 など
        or 0x3041 <= o <= 0x30FF      # ひらがな・カタカナ (・ ー を含む)
        or 0x31F0 <= o <= 0x31FF      # カタカナ拡張
        or 0x3400 <= o <= 0x4DBF      # CJK 統合漢字拡張 A
        or 0x4E00 <= o <= 0x9FFF      # CJK 統合漢字
        or 0xF900 <= o <= 0xFAFF      # CJK 互換漢字
        or 0xFF01 <= o <= 0xFF9F      # 全角英数・全角記号・半角カナ
        or 0xFFE0 <= o <= 0xFFEF      # 全角の円記号など
        or 0x20000 <= o <= 0x3FFFF    # CJK 統合漢字拡張 B 以降
    )


def is_alnum(c: str) -> bool:
    return c.isascii() and c.isalnum()


def is_cs_letter(c: str) -> bool:
    """コマンド名に使われうる文字. pTeX は和文文字もコマンド名に含めるので, 安全側に含める."""
    return ("a" <= c <= "z") or ("A" <= c <= "Z") or is_ja(c)


PARA_BREAK = re.compile(r"\n[ \t\r]*\n")

# 削除の対象とする空白の両側の組 (J: 日本語, A: 英数字, M: インライン数式)
TARGET_PAIRS = {("J", "J"), ("J", "A"), ("A", "J"), ("J", "M"), ("M", "J")}
PAIR_LABELS = {
    ("J", "J"): "日本語␣日本語",
    ("J", "A"): "日本語␣英数字",
    ("A", "J"): "英数字␣日本語",
    ("J", "M"): "日本語␣数式",
    ("M", "J"): "数式␣日本語",
}

# ----------------------------------------------------------------------
# LaTeX の解析: 保護する範囲とインライン数式の位置を求める
# ----------------------------------------------------------------------


class Scanner:
    """1 ファイルの本文を走査し, 書き換えてはいけない文字に印を付ける.

    prot[i] が 1 の文字は保護する (数式, コメント, コマンド名, 保護するコマンドの引数など).
    math_open / math_close はインライン数式の最初の文字 ($ か \\() と最後の文字 ($ か ) ) の位置.
    """

    def __init__(self, text: str):
        self.t = text
        self.n = len(text)
        self.prot = bytearray(self.n)
        self.kind_at = [0] * self.n
        self.kinds = [""]
        self.kind_ids: dict[str, int] = {"": 0}
        self.spans: list[tuple[int, int, str]] = []
        self.math_open: set[int] = set()
        self.math_close: set[int] = set()
        self.uncertain: list[tuple[int, int, str]] = []
        self.warnings: list[tuple[int, str]] = []
        self.begin_doc_end: int | None = None
        self.end_doc: int | None = None
        self.has_documentclass = False

    # -- 補助 ---------------------------------------------------------

    def protect(self, a: int, b: int, kind: str) -> None:
        if b <= a:
            return
        kid = self.kind_ids.setdefault(kind, len(self.kinds))
        if kid == len(self.kinds):
            self.kinds.append(kind)
        self.prot[a:b] = b"\x01" * (b - a)
        self.kind_at[a:b] = [kid] * (b - a)
        self.spans.append((a, b, kind))

    def kind(self, i: int) -> str:
        return self.kinds[self.kind_at[i]]

    def eol(self, i: int) -> int:
        j = self.t.find("\n", i)
        return self.n if j < 0 else j

    def para_bounds(self, i: int) -> tuple[int, int]:
        """位置 i を含む段落 (空行で区切られた範囲) の [始まり, 終わり)."""
        start = 0
        for m in PARA_BREAK.finditer(self.t, 0, i):
            start = m.end()
        m = PARA_BREAK.search(self.t, i)
        return start, (m.start() + 1 if m else self.n)

    def mark_uncertain(self, i: int, msg: str, to_eof: bool = False) -> int:
        ps, pe = self.para_bounds(i)
        if to_eof:
            pe = self.n
        self.uncertain.append((ps, pe, msg))
        self.warnings.append((i, msg))
        return pe

    def skip_ws(self, k: int) -> int:
        """空白と改行 1 つまでを飛ばす (空行は越えない)."""
        t, n = self.t, self.n
        while k < n and t[k] in " \t":
            k += 1
        if k < n and t[k] == "\r":
            k += 1
        if k < n and t[k] == "\n":
            k += 1
            while k < n and t[k] in " \t":
                k += 1
        return k

    def match_brace(self, k: int) -> int | None:
        """t[k] == '{' に対応する '}' の次の位置. 閉じていなければ None."""
        t, n = self.t, self.n
        depth = 0
        while k < n:
            c = t[k]
            if c == "%":
                k = self.eol(k)
                continue
            if c == "\\":
                k += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return k + 1
            k += 1
        return None

    def match_bracket(self, k: int) -> int | None:
        """t[k] == '[' に対応する ']' の次の位置 (中括弧の中の ] は数えない)."""
        t, n = self.t, self.n
        depth = 0
        k += 1
        while k < n:
            c = t[k]
            if c == "%":
                k = self.eol(k)
                continue
            if c == "\\":
                k += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            elif c == "]" and depth <= 0:
                return k + 1
            elif c == "\n" and PARA_BREAK.match(t, k):
                return None
            k += 1
        return None

    def parse_args(self, k: int, spec: str) -> int | None:
        """引数の並び spec を読み, 読み終えた位置を返す. 解析できなければ None."""
        t, n = self.t, self.n
        for s in spec:
            if s == "s":
                if k < n and t[k] == "*":
                    k += 1
            elif s == "o":
                k2 = self.skip_ws(k)
                if k2 < n and t[k2] == "[":
                    e = self.match_bracket(k2)
                    if e is None:
                        return None
                    k = e
            else:  # m, f
                k2 = self.skip_ws(k)
                if k2 >= n:
                    return None
                c = t[k2]
                if c == "{":
                    e = self.match_brace(k2)
                    if e is None:
                        return None
                    k = e
                elif c in "}%":
                    return None
                elif c == "\\":
                    e = k2 + 1
                    if e < n and is_cs_letter(t[e]):
                        while e < n and is_cs_letter(t[e]):
                            e += 1
                    else:
                        e += 1
                    k = e
                elif s == "f":
                    e = k2
                    while e < n and not t[e].isspace() and t[e] not in "{}%\\":
                        e += 1
                    k = e
                else:
                    k = k2 + 1
        return k

    # -- 走査 ---------------------------------------------------------

    def scan(self) -> "Scanner":
        t, n = self.t, self.n
        i = 0
        while i < n:
            c = t[i]
            if c == "%":
                j = self.eol(i)
                self.protect(i, j, "comment")
                i = j
            elif c == "\\":
                i = self._backslash(i)
            elif c == "$":
                i = self._dollar(i)
            else:
                i += 1
        self._finish()
        return self

    def _backslash(self, i: int) -> int:
        t, n = self.t, self.n
        if i + 1 >= n:
            self.protect(i, n, "command")
            return n
        d = t[i + 1]
        if d == "(":
            return self._inline_math(i, 2, "\\)")
        if d == "[":
            return self._display_bracket(i)
        if not is_cs_letter(d):
            self.protect(i, i + 2, "command")  # \\ \% \$ \, \ (制御空白) など
            return i + 2
        j = i + 1
        while j < n and is_cs_letter(t[j]):
            j += 1
        name = t[i + 1:j]
        if name == "begin":
            return self._begin(i, j)
        if name == "end":
            return self._end(i, j)
        if name in ("verb", "lstinline"):
            return self._verb(i, j, name)
        if name == "makeatletter":
            e = t.find("\\makeatother", j)
            if e < 0:
                return self.mark_uncertain(i, "\\makeatletter に対応する \\makeatother がない", to_eof=True)
            e += len("\\makeatother")
            self.protect(i, e, "makeatletter")
            return e
        if name in DEF_CMDS:
            return self._def(i, j)
        if name in RAW_ARG_CMDS:
            return self._raw_arg(i, j, name)
        if name in PROTECTED_CMDS:
            if name == "documentclass":
                self.has_documentclass = True
            e = self.parse_args(j, PROTECTED_CMDS[name])
            if e is None:
                return self.mark_uncertain(i, f"\\{name} の引数を解析できない", to_eof=True)
            self.protect(i, e, "command:" + name)
            return e
        if name in UNCERTAIN_CMDS:
            m = re.compile(r"\\catcode\s*`\\?@\s*=\s*1[12]").match(t, i)
            if m:  # \catcode`\@=11 は \makeatletter と同じなので問題ない
                self.protect(i, m.end(), "command")
                return m.end()
            return self.mark_uncertain(i, f"\\{name} があるため, 以降は解析が不確実", to_eof=True)
        self.protect(i, j, "command")
        return j

    def _begin(self, i: int, j: int) -> int:
        t = self.t
        m = re.compile(r"[ \t]*\{([^{}\s%]+)\}").match(t, j)
        if not m:
            self.protect(i, j, "command")
            return j
        env, k = m.group(1), m.end()
        if env == "document":
            self.begin_doc_end = k
            self.protect(i, k, "begin")
            return k
        if env in VERBATIM_ENVS:
            e = t.find("\\end{" + env + "}", k)
            if e < 0:
                return self.mark_uncertain(i, f"\\begin{{{env}}} に対応する \\end がない", to_eof=True)
            e += len("\\end{" + env + "}")
            self.protect(i, e, "verbatim:" + env)
            return e
        if env in MATH_ENVS or env in PROTECTED_ENVS:
            e = self._find_env_end(k, env)
            if e is None:
                return self.mark_uncertain(i, f"\\begin{{{env}}} に対応する \\end がない", to_eof=True)
            self.protect(i, e, ("math:" if env in MATH_ENVS else "env:") + env)
            return e
        e = self.parse_args(k, ENV_ARGS.get(env, ""))
        if e is None:
            return self.mark_uncertain(i, f"\\begin{{{env}}} の引数を解析できない")
        self.protect(i, e, "begin")
        return e

    def _end(self, i: int, j: int) -> int:
        m = re.compile(r"[ \t]*\{([^{}\s%]+)\}").match(self.t, j)
        if not m:
            self.protect(i, j, "command")
            return j
        if m.group(1) == "document":
            self.end_doc = i
            return self.n  # \end{document} より後は TeX が読まない
        self.protect(i, m.end(), "end")
        return m.end()

    def _find_env_end(self, k: int, env: str) -> int | None:
        """\\begin{env} の中身の始まり k から, 対応する \\end{env} の次の位置を探す."""
        t, n = self.t, self.n
        pat = re.compile(r"\\(begin|end)[ \t]*\{" + re.escape(env) + r"\}")
        depth = 1
        while k < n:
            c = t[k]
            if c == "%":
                k = self.eol(k)
                continue
            if c == "\\":
                m = pat.match(t, k)
                if m:
                    depth += 1 if m.group(1) == "begin" else -1
                    k = m.end()
                    if depth == 0:
                        return k
                    continue
                k += 2
                continue
            k += 1
        return None

    def _dollar(self, i: int) -> int:
        t, n = self.t, self.n
        if not t.startswith("$$", i):
            return self._inline_math(i, 1, "$")
        k = i + 2
        while k < n:
            c = t[k]
            if c == "%":
                k = self.eol(k)
                continue
            if c == "\\":
                k += 2
                continue
            if c == "$" and t.startswith("$$", k):
                self.protect(i, k + 2, "math:$$")
                return k + 2
            if c == "\n" and PARA_BREAK.match(t, k):
                break
            k += 1
        return self.mark_uncertain(i, "$$ が段落の中で閉じていない")

    def _inline_math(self, i: int, open_len: int, closer: str) -> int:
        t, n = self.t, self.n
        k = i + open_len
        depth = 0  # \text{... $x$ ...} の中の $ で閉じないよう, 中括弧の深さを数える
        while k < n:
            c = t[k]
            if c == "%":
                k = self.eol(k)
                continue
            if c == "\\":
                if closer == "\\)" and depth <= 0 and t.startswith("\\)", k):
                    end = k + 2
                    break
                k += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            elif c == "$" and closer == "$" and depth <= 0:
                end = k + 1
                break
            elif c == "\n" and PARA_BREAK.match(t, k):
                return self.mark_uncertain(i, "インライン数式が段落の中で閉じていない")
            k += 1
        else:
            return self.mark_uncertain(i, "インライン数式が閉じていない")
        self.protect(i, end, "math:inline")
        self.math_open.add(i)
        self.math_close.add(end - 1)
        return end

    def _display_bracket(self, i: int) -> int:
        t, n = self.t, self.n
        k = i + 2
        while k < n:
            c = t[k]
            if c == "%":
                k = self.eol(k)
                continue
            if c == "\\":
                if t.startswith("\\]", k):
                    self.protect(i, k + 2, "math:\\[\\]")
                    return k + 2
                k += 2
                continue
            if c == "\n" and PARA_BREAK.match(t, k):
                break
            k += 1
        return self.mark_uncertain(i, "\\[ が段落の中で閉じていない")

    def _verb(self, i: int, j: int, name: str) -> int:
        t, n = self.t, self.n
        k = j
        if k < n and t[k] == "*":
            k += 1
        if name == "lstinline" and k < n and t[k] == "[":
            e = self.match_bracket(k)
            if e is None:
                return self.mark_uncertain(i, "\\lstinline の引数を解析できない")
            k = e
        if k >= n or t[k].isspace() or t[k].isalpha():
            return self.mark_uncertain(i, f"\\{name} の区切り文字を解析できない")
        close = "}" if t[k] == "{" else t[k]
        e = t.find(close, k + 1)
        nl = t.find("\n", k + 1)
        if e < 0 or (0 <= nl < e):
            return self.mark_uncertain(i, f"\\{name} が行の中で閉じていない")
        self.protect(i, e + 1, "verb")
        return e + 1

    def _def(self, i: int, j: int) -> int:
        t, n = self.t, self.n
        k = self.skip_ws(j)
        if k < n and t[k] == "\\":  # 定義するコマンド名
            k += 1
            if k < n and is_cs_letter(t[k]):
                while k < n and is_cs_letter(t[k]):
                    k += 1
            else:
                k += 1
        elif k < n:
            k += 1
        b = k  # 引数テキスト (#1#2 など) から { まで
        while b < n and t[b] not in "{\n%":
            b += 1
        e = self.match_brace(b) if b < n and t[b] == "{" else None
        if e is None:
            return self.mark_uncertain(i, "\\def の本体を解析できない", to_eof=True)
        self.protect(i, e, "definition")
        return e

    def _raw_arg(self, i: int, j: int, name: str) -> int:
        t, n = self.t, self.n
        k = self.skip_ws(j)
        if k < n and t[k] == "{":
            depth, e = 0, k
            while e < n:
                if t[e] == "{":
                    depth += 1
                elif t[e] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                e += 1
            else:
                return self.mark_uncertain(i, f"\\{name} の引数が閉じていない", to_eof=True)
            self.protect(i, e + 1, "command:" + name)
            return e + 1
        if name != "href" and k < n and not t[k].isspace() and not t[k].isalpha():
            e = t.find(t[k], k + 1)  # \url|...| の形
            nl = t.find("\n", k + 1)
            if e >= 0 and not (0 <= nl < e):
                self.protect(i, e + 1, "command:" + name)
                return e + 1
        return self.mark_uncertain(i, f"\\{name} の引数を解析できない")

    def _finish(self) -> None:
        for a, b, _msg in self.uncertain:
            self.protect(a, b, "uncertain")
        if self.begin_doc_end is not None:
            self.protect(0, self.begin_doc_end, "preamble")
        if self.end_doc is not None:
            self.protect(self.end_doc, self.n, "after-document")
        # プリアンブル・解析できない段落の中のインライン数式は, 本文の数式として扱わない
        wide = {"preamble", "after-document", "uncertain"}
        self.math_open = {x for x in self.math_open if self.kind(x) not in wide}
        self.math_close = {x for x in self.math_close if self.kind(x) not in wide}


# ----------------------------------------------------------------------
# 空白の判定
# ----------------------------------------------------------------------


@dataclass
class Gap:
    start: int          # 半角スペースの並び [start, end)
    end: int
    action: str         # remove / skip / other
    label: str          # 種類 (表示用)


class GapFinder:
    def __init__(self, sc: Scanner):
        self.sc = sc
        self.t = sc.t
        self.n = sc.n

    def side_type(self, k: int, left: bool) -> str:
        sc = self.sc
        if left and k in sc.math_close:
            return "M"
        if not left and k in sc.math_open:
            return "M"
        if sc.prot[k]:
            return "P"
        c = self.t[k]
        if is_ja(c):
            return "J"
        if is_alnum(c):
            return "A"
        return "O"

    # -- 対象外の境目の分類 (参考表示用) -------------------------------

    def other_label(self, lt: str, rt: str, s: int, e: int) -> str | None:
        """日本語に接しているが削除の対象にしない境目の種類. 日本語に接していなければ None."""
        t, sc = self.t, self.sc
        if lt == "J" and rt == "J":
            return None
        if lt == "J":
            kind = sc.kind(e) if rt == "P" else ""
            if kind == "command" or kind.startswith("command:"):
                return "日本語␣\\コマンド (\\cite, \\ref など)"
            if rt == "P":
                return f"日本語␣保護領域 ({kind})"
            if rt == "O":
                return f"日本語␣記号 「{t[e]}」"
            return None
        if rt == "J":
            kind = sc.kind(s - 1) if lt == "P" else ""
            if kind == "command":
                return "コマンド名の直後 (\\item 日本語 など)"
            if kind.startswith("command:"):
                return f"\\{kind[8:]}{{…}}␣日本語"
            if lt == "P":
                return f"保護領域 ({kind})␣日本語"
            if lt == "O":
                c = t[s - 1]
                if c in ",.":
                    return "ASCII の句読点 , . の後"
                return f"記号 「{c}」␣日本語"
        return None

    def find(self) -> list[Gap]:
        t, n, sc = self.t, self.n, self.sc
        gaps: list[Gap] = []
        for m in re.finditer(r" +", t):
            s, e = m.span()
            if s == 0 or e >= n or t[s - 1] in "\r\n\t" or t[e] in "\r\n\t":
                continue  # 行頭のインデント, 行末の空白, タブ
            if any(sc.prot[s:e]):
                kind = sc.kind(s)
                p = s - 1
                while p >= 0 and "a" <= t[p].lower() <= "z":
                    p -= 1
                after_cs = p >= 0 and t[p] == "\\" and p < s - 1  # \\huge 令和 などはもともと対象外
                if (kind.startswith("env:") or kind == "uncertain") and not after_cs:
                    raw = tuple("J" if is_ja(c) else "A" if is_alnum(c) else "M" if c == "$" else "O"
                                for c in (t[s - 1], t[e]))
                    if raw in TARGET_PAIRS:
                        label = "解析できない段落" if kind == "uncertain" else f"保護環境 {kind[4:]}"
                        gaps.append(Gap(s, e, "skip", label))
                continue
            lt = self.side_type(s - 1, left=True)
            rt = self.side_type(e, left=False)
            if (lt, rt) in TARGET_PAIRS:
                gaps.append(Gap(s, e, "remove", PAIR_LABELS[(lt, rt)]))
                continue
            label = self.other_label(lt, rt, s, e)
            if label:
                gaps.append(Gap(s, e, "other", label))
        return gaps


def apply_gaps(text: str, gaps: list[Gap]) -> str:
    parts, pos = [], 0
    for g in gaps:
        if g.action == "remove":
            parts.append(text[pos:g.start])
            pos = g.end
    parts.append(text[pos:])
    return "".join(parts)


# ----------------------------------------------------------------------
# ファイルの収集
# ----------------------------------------------------------------------


@dataclass
class TexFile:
    path: Path
    rel: str
    role: str                 # ルート / \input など
    target: bool
    reason: str = ""
    include_name: str = ""    # 読み込み元に書かれた名前


def read_text(path: Path) -> tuple[str, bytes]:
    data = path.read_bytes()
    return data.decode("utf-8"), data


def rel_to(path: Path, base: Path) -> str:
    return os.path.relpath(path, base)


def resolve_include(name: str, base: Path) -> Path | None:
    p = Path(name)
    cands = [p.with_name(p.name + ".tex"), p] if p.suffix != ".tex" else [p]
    for c in cands:
        q = c if c.is_absolute() else base / c
        if q.is_file():
            return q.resolve()
    return None


def collect_files(main: Path) -> list[TexFile]:
    root_dir = main.parent
    files: list[TexFile] = []
    seen: set[Path] = set()

    def visit(path: Path, role: str, include_name: str) -> None:
        seen.add(path)
        tf = TexFile(path, rel_to(path, root_dir), role, True, include_name=include_name)
        files.append(tf)
        try:
            text, _ = read_text(path)
        except UnicodeDecodeError:
            tf.target, tf.reason = False, "UTF-8 として読めない"
            return
        sc = Scanner(text).scan()
        if path != main and sc.has_documentclass and sc.begin_doc_end is None:
            tf.target, tf.reason = False, "\\documentclass があるのに \\begin{document} がない"
            return
        for m in INCLUDE_RE.finditer(text):
            kind = sc.kind(m.start())
            if kind in ("comment", "after-document") or kind.startswith("verbatim") or kind == "verb":
                continue
            if m.group(1):
                cmd, name, base = m.group(1), (m.group(2) or m.group(3)).strip(), root_dir
            else:
                cmd, name = m.group(4), m.group(6).strip()
                d = m.group(5).strip()
                base = (path.parent if cmd == "subimport" else root_dir) / d
            child = resolve_include(name, base)
            role_c = f"\\{cmd} ({tf.rel} {text.count(chr(10), 0, m.start()) + 1} 行目)"
            if child is None:
                files.append(TexFile(base / name, name, role_c, False,
                                     "見つからない (TeX の検索パス上のファイルの可能性)", name))
                continue
            if child in seen:
                continue
            if child.suffix != ".tex":
                seen.add(child)
                files.append(TexFile(child, rel_to(child, root_dir), role_c, False, ".tex 以外は変更しない", name))
            elif kind == "preamble":
                seen.add(child)
                files.append(TexFile(child, rel_to(child, root_dir), role_c, False,
                                     "プリアンブルから読み込まれる (定義用とみなす)", name))
            elif not (kind.startswith("command:") or kind == ""):
                seen.add(child)
                files.append(TexFile(child, rel_to(child, root_dir), role_c, False,
                                     f"保護領域 ({kind}) の中から読み込まれる", name))
            else:
                visit(child, role_c, name)

    visit(main.resolve(), "ルート", main.name)
    return files


def find_main(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        if not p.is_file():
            sys.exit(f"エラー: {arg} が見つかりません.")
        return p.resolve()
    pat = re.compile(r"^[^%\n]*\\documentclass", re.M)
    for d in (Path.cwd(), Path(__file__).resolve().parent):
        cands = []
        for p in sorted(d.glob("*.tex")):
            try:
                text = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if pat.search(text) and "\\begin{document}" in text:
                cands.append(p)
        if len(cands) == 1:
            return cands[0].resolve()
        if len(cands) > 1:
            names = "\n".join(f"  python clean_tex.py \"{c.name}\" --dry-run" for c in cands)
            sys.exit(f"エラー: {d} に \\documentclass を含む .tex が複数あります. 次のように指定してください:\n{names}")
    sys.exit("エラー: \\documentclass と \\begin{document} を含む .tex が見つかりません. ファイル名を指定してください.")


# ----------------------------------------------------------------------
# 1 ファイルの処理と検証
# ----------------------------------------------------------------------


@dataclass
class FileResult:
    tf: TexFile
    text: str
    data: bytes
    new_text: str
    gaps: list[Gap]
    sc: Scanner
    checks: list[tuple[bool, str]] = field(default_factory=list)

    @property
    def removed(self) -> list[Gap]:
        return [g for g in self.gaps if g.action == "remove"]

    @property
    def changed(self) -> bool:
        return self.new_text != self.text

    @property
    def sha_before(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    @property
    def new_data(self) -> bytes:
        return self.new_text.encode("utf-8")


def process_file(tf: TexFile, is_root: bool) -> FileResult:
    text, data = read_text(tf.path)
    sc = Scanner(text).scan()
    if is_root and sc.begin_doc_end is None:
        sys.exit(f"エラー: {tf.rel} に \\begin{{document}} が見つかりません (ルートのファイルではない?).")
    gaps = GapFinder(sc).find()
    new_text = apply_gaps(text, gaps)
    r = FileResult(tf, text, data, new_text, gaps, sc)
    r.checks = verify(r)
    return r


def verify(r: FileResult) -> list[tuple[bool, str]]:
    """書き換えの結果を自分で検査する. どれか 1 つでも失敗したら書き込まない."""
    old, new = r.text, r.new_text
    checks = []
    n_removed = sum(g.end - g.start for g in r.removed)
    only_spaces = len(old) - len(new) == n_removed and all(old[g.start:g.end].strip(" ") == "" for g in r.removed)
    checks.append((only_spaces, f"削除したのは半角スペース {n_removed} 個だけ"))
    checks.append((old.count("\n") == new.count("\n"), f"改行は不変 ({old.count(chr(10)) + 1} 行)"))
    sc_new = Scanner(new).scan()
    a = [(k, old[s:e]) for s, e, k in r.sc.spans]
    b = [(k, new[s:e]) for s, e, k in sc_new.spans]
    checks.append((a == b, f"保護領域 {len(a)} 個 (数式・コメント・コマンドなど) は処理前と完全に一致"))
    again = [g for g in GapFinder(sc_new).find() if g.action == "remove"]
    checks.append((not again, "もう一度実行しても追加の変更はない"))
    return checks


# ----------------------------------------------------------------------
# 表示
# ----------------------------------------------------------------------


class Style:
    def __init__(self, enabled: bool):
        e = enabled
        self.red = "\x1b[1;31m" if e else ""
        self.green = "\x1b[32m" if e else ""
        self.yellow = "\x1b[33m" if e else ""
        self.cyan = "\x1b[36m" if e else ""
        self.dim = "\x1b[2m" if e else ""
        self.bold = "\x1b[1m" if e else ""
        self.reset = "\x1b[0m" if e else ""


S = Style(False)


def ok_mark(ok: bool) -> str:
    return f"{S.green}✔{S.reset}" if ok else f"{S.red}✘{S.reset}"


def pad(s: str, width: int) -> str:
    """全角文字を 2 桁と数えて右を空白で埋める."""
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    return s + " " * max(1, width - w)


def line_index(text: str) -> list[int]:
    starts = [0]
    for m in re.finditer("\n", text):
        starts.append(m.end())
    return starts


def render_line(text: str, ls: int, le: int, gaps: list[Gap], marker: str) -> str:
    """1 行のうち gaps の位置を記号で示し, 離れた部分は … で省く."""
    line = text[ls:le].rstrip("\r")
    pieces, pos = [], 0
    for g in gaps:
        a, b = g.start - ls, g.end - ls
        seg = line[pos:a]
        if pos == 0 and len(seg) > CONTEXT:
            seg = "…" + seg[-CONTEXT:]
        elif pos > 0 and len(seg) > 2 * CONTEXT + 3:
            seg = seg[:CONTEXT] + " … " + seg[-CONTEXT:]
        pieces.append(seg)
        pieces.append(marker * (b - a))
        pos = b
    tail = line[pos:]
    pieces.append(tail if len(tail) <= CONTEXT else tail[:CONTEXT] + "…")
    return "".join(pieces).lstrip()


def show_gaps(r: FileResult, gaps: list[Gap], limit: int | None, with_label: bool) -> int:
    """gaps を行ごとにまとめて表示し, 表示した行数を返す."""
    starts = line_index(r.text)
    by_line: dict[int, list[Gap]] = {}
    for g in gaps:
        by_line.setdefault(bisect.bisect_right(starts, g.start) - 1, []).append(g)
    shown = 0
    for ln, gs in sorted(by_line.items()):
        if limit is not None and shown >= limit:
            print(f"    {S.dim}… ほか {len(by_line) - shown} 行 (--max-show で表示数を変更){S.reset}")
            break
        ls = starts[ln]
        le = starts[ln + 1] - 1 if ln + 1 < len(starts) else len(r.text)
        marker = f"{S.red}␣{S.reset}" if not with_label else f"{S.yellow}␣{S.reset}"
        body = render_line(r.text, ls, le, gs, marker)
        suffix = f"  {S.dim}[{', '.join(sorted({g.label for g in gs}))}]{S.reset}" if with_label else ""
        print(f"  {S.cyan}{ln + 1:>5}{S.reset}  {body}{suffix}")
        shown += 1
    return len(by_line)


def unified_diff(results: list[FileResult]) -> str:
    out = []
    for r in results:
        if r.changed:
            out.extend(difflib.unified_diff(
                r.text.splitlines(keepends=True), r.new_text.splitlines(keepends=True),
                fromfile="a/" + r.tf.rel, tofile="b/" + r.tf.rel, n=0))
    return "".join(x if x.endswith("\n") else x + "\n" for x in out)


SPAN_GROUPS = [
    ("インライン数式 $…$ \\(…\\)", lambda k: k == "math:inline"),
    ("別行立て数式 \\[…\\] $$…$$", lambda k: k in ("math:\\[\\]", "math:$$")),
    ("数式環境 (equation, align など)", lambda k: k.startswith("math:") and k not in ("math:inline", "math:\\[\\]", "math:$$")),
    ("コメント %", lambda k: k == "comment"),
    ("\\label・\\ref・\\cite などと引数", lambda k: k.startswith("command:")),
    ("保護環境 (titlepage など)", lambda k: k.startswith("env:")),
    ("verbatim・\\verb", lambda k: k.startswith("verbatim") or k == "verb"),
    ("定義 (\\def, \\makeatletter)", lambda k: k in ("definition", "makeatletter")),
    ("プリアンブル", lambda k: k == "preamble"),
    ("\\end{document} より後", lambda k: k == "after-document"),
    ("解析できない段落", lambda k: k == "uncertain"),
]


def print_summary(results: list[FileResult], show_skipped: bool) -> None:
    removed = [g for r in results for g in r.removed]
    print(f"\n{S.bold}== 結果 =={S.reset}")
    n_changed = sum(r.changed for r in results)
    print(f"変更されるファイル: {n_changed} / {len(results)}")
    print(f"削除する半角スペース: {sum(g.end - g.start for g in removed)} 個 ({len(removed)} か所)")
    by = Counter(g.label for g in removed)
    for label in PAIR_LABELS.values():
        print(f"    {pad(label, 16)}{by.get(label, 0):>6} か所")

    skipped = Counter(g.label for r in results for g in r.gaps if g.action == "skip")
    print("\n削除しなかった候補 (日本語と英数字・数式の間だが, 安全のため残す):")
    if not skipped:
        print("    なし")
    for label, c in skipped.most_common():
        print(f"    {pad(label, 44)}{c:>5} か所")
    if skipped and not show_skipped:
        print(f"    {S.dim}→ --show-skipped で一覧を表示{S.reset}")

    others = Counter(g.label for r in results for g in r.gaps if g.action == "other")
    print("\n対象外の境目 (参考: 日本語に接する空白のうち, 今回の規則では触らないもの):")
    for label, c in others.most_common(12):
        print(f"    {pad(label, 44)}{c:>5} か所")
    if len(others) > 12:
        print(f"    {S.dim}ほか {len(others) - 12} 種類{S.reset}")

    print("\n保護した領域 (中身は一切変更しない):")
    kinds = Counter(k for r in results for _s, _e, k in r.sc.spans)
    for name, pred in SPAN_GROUPS:
        c = sum(v for k, v in kinds.items() if pred(k))
        if c:
            print(f"    {pad(name, 36)}{c:>6}")
    spaces_in_math = sum(r.text.count(" ", s, e) for r in results for s, e, k in r.sc.spans if k.startswith("math:"))
    print(f"    {S.dim}(数式の中の半角スペース {spaces_in_math} 個はすべてそのまま){S.reset}")

    warns = [(r, i, msg) for r in results for i, msg in r.sc.warnings]
    if warns:
        print(f"\n{S.yellow}解析できずにスキップした箇所:{S.reset}")
        for r, i, msg in warns:
            print(f"    {r.tf.rel}:{r.text.count(chr(10), 0, i) + 1}: {msg}")

    print("\n検証:")
    for r in results:
        for ok, msg in r.checks:
            print(f"    {ok_mark(ok)} {r.tf.rel}: {msg}")


# ----------------------------------------------------------------------
# コンパイル検査
# ----------------------------------------------------------------------


@dataclass
class CompileResult:
    ran: bool
    fatal: bool = False
    errors: list[str] = field(default_factory=list)
    pages: int | None = None
    warnings: Counter = field(default_factory=Counter)
    message: str = ""
    missing_sty: str = ""


def detect_engine(main: Path, text: str) -> tuple[str, str]:
    head = "\n".join(text.splitlines()[:20])
    m = re.search(r"^%\s*!TeX\s+(?:TS-)?program\s*=\s*(\S+)", head, re.I | re.M) or re.search(r"^%#!\s*(\S+)", head, re.M)
    if m:
        return m.group(1), "マジックコメント"
    rc = main.parent / ".latexmkrc"
    if rc.is_file():
        m = re.search(r"^\s*\$latex\s*=\s*['\"](\S+)", rc.read_text(encoding="utf-8", errors="replace"), re.M)
        if m:
            return m.group(1), ".latexmkrc"
    m = re.search(r"^[^%\n]*\\documentclass\s*(?:\[([^\]]*)\])?\s*\{([^}]+)\}", text, re.M)
    opts, cls = (m.group(1) or "", m.group(2).strip()) if m else ("", "")
    why = f"\\documentclass{{{cls}}}"
    body = re.sub(r"%.*", "", text)
    if "luatexja" in body or cls.startswith("ltj"):
        return "lualatex", why
    if re.search(r"\\usepackage(\[[^\]]*\])?\{[^}]*(xeCJK|zxjatype)", body):
        return "xelatex", why
    if cls.startswith("u") and cls[1:] in ("jarticle", "jreport", "jbook", "tarticle", "treport", "tbook"):
        return "uplatex", why
    if cls in ("jsarticle", "jsreport", "jsbook", "jarticle", "jreport", "jbook", "tarticle", "treport", "tbook"):
        return ("uplatex" if "uplatex" in opts else "platex"), why
    if cls.startswith("bxjs"):
        for eng in ("lualatex", "xelatex", "uplatex", "platex", "pdflatex"):
            if eng in opts:
                return eng, why
        return "uplatex", why
    return "pdflatex", why


def run_engine(engine: str, main_copy: Path, cwd: Path, outdir: Path, texinputs: list[Path],
               stubs: Path, replace: list[tuple[str, str]]) -> CompileResult:
    outdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(max_print_line="100000", error_line="254", half_error_line="238")
    env["TEXINPUTS"] = os.pathsep.join(str(p) for p in [*texinputs, stubs]) + os.pathsep + env.get("TEXINPUTS", "")
    cmd = [engine, "-interaction=nonstopmode", "-file-line-error", f"-output-directory={outdir}",
           f"-jobname={CHECK_JOBNAME}", str(main_copy)]
    try:
        subprocess.run(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=600)
    except FileNotFoundError:
        return CompileResult(False, message=f"{engine} が見つからない")
    except subprocess.TimeoutExpired:
        return CompileResult(True, fatal=True, message="時間切れ (600 秒)")
    log_path = outdir / (CHECK_JOBNAME + ".log")
    if not log_path.is_file():
        return CompileResult(True, fatal=True, message="ログが出力されなかった")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    for a, b in replace:
        log = log.replace(a, b)
    res = CompileResult(True)
    for line in log.splitlines():
        m = re.match(r"^(.+?):(\d+): (.*)$", line)
        if m and not line.startswith(("Package", "LaTeX", "Class")):
            res.errors.append(f"{m.group(1).removeprefix('./')}:{m.group(2)}: {m.group(3)}")
        elif line.startswith("! "):
            res.errors.append(line)
    m = re.search(r"Output written on .*?\((\d+) pages?", log)
    res.pages = int(m.group(1)) if m else (0 if "No pages of output" in log else None)
    res.fatal = any(s in log for s in ("Emergency stop", "Fatal error occurred", "TeX capacity exceeded")) or res.pages is None
    res.warnings = Counter({
        "LaTeX Warning": len(re.findall(r"LaTeX Warning", log)),
        "Overfull \\hbox": len(re.findall(r"Overfull \\hbox", log)),
        "Underfull \\hbox": len(re.findall(r"Underfull \\hbox", log)),
    })
    for name in re.findall(r"File `([^']+)\.sty' not found", log):
        res.missing_sty = name  # 呼び出し側で空のスタブを作ってやり直す
    return res


class CompileChecker:
    """処理前と処理後を一時フォルダでコンパイルし, エラーが増えないか比べる.

    出力 (.aux, .log, .dvi) はすべて一時フォルダに書き, 論文のフォルダには何も書かない.
    相対パスの図などが見つかるよう, 作業フォルダはルートのファイルのフォルダにする.
    書き換えた子ファイルは TEXINPUTS の先頭に置いた一時フォルダから読ませる.
    """

    def __init__(self, engine: str, main: Path):
        self.engine = engine
        self.main = main
        self.root_dir = main.parent
        self.tmp = Path(tempfile.mkdtemp(prefix="clean_tex_"))
        self.stubs = self.tmp / "stubs"
        self.stubs.mkdir()
        self.stubbed: list[str] = []
        self.count = 0
        atexit.register(self.cleanup)  # 途中で中止しても一時フォルダを残さない

    def compile(self, files: dict[str, str]) -> CompileResult:
        """files: ルートのフォルダからの相対パス -> 内容 (ルート自身は main.name). 他のファイルは実物を使う."""
        self.count += 1
        src = self.tmp / f"src{self.count}"
        out = self.tmp / f"out{self.count}"
        src.mkdir()
        main_copy = src / "_clean_tex_main.tex"
        for rel, content in files.items():
            dest = main_copy if rel == self.main.name else src / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            (out / rel).parent.mkdir(parents=True, exist_ok=True)  # \include の .aux 用
        replace = [(str(main_copy), self.main.name), (str(src) + os.sep, ""), (str(self.root_dir) + os.sep, "")]
        for _ in range(30):
            res = run_engine(self.engine, main_copy, self.root_dir, out, [src], self.stubs, replace)
            missing = res.missing_sty if res.ran and res.fatal else ""
            if not missing or missing in self.stubbed or not re.fullmatch(r"[\w\-.]+", missing):
                return res
            (self.stubs / (missing + ".sty")).write_text("% clean_tex.py の検査用スタブ\n", encoding="utf-8")
            self.stubbed.append(missing)
        return res

    def cleanup(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


def compare(base: CompileResult, after: CompileResult) -> tuple[bool, list[str]]:
    lines = []
    lines.append(f"処理前: エラー {len(base.errors)} 件, {base.pages if base.pages is not None else '?'} ページ"
                 f"{' (途中で停止)' if base.fatal else ''}")
    lines.append(f"処理後: エラー {len(after.errors)} 件, {after.pages if after.pages is not None else '?'} ページ"
                 f"{' (途中で停止)' if after.fatal else ''}")
    new_errors = list((Counter(after.errors) - Counter(base.errors)).elements())
    ok = not new_errors and not (after.fatal and not base.fatal)
    for e in new_errors[:10]:
        lines.append(f"  {S.red}新しいエラー: {e}{S.reset}")
    if after.fatal and not base.fatal:
        lines.append(f"  {S.red}処理後だけ途中で停止した{S.reset}")
    if base.errors and ok:
        kinds = Counter(re.sub(r"^.*?:\d+: ", "", e)[:60] for e in base.errors)
        lines.append(f"  {S.dim}(処理前からあるエラー: " + "; ".join(f"{k}… ×{v}" for k, v in kinds.most_common(3)) + f"){S.reset}")
    if base.pages != after.pages:
        lines.append(f"  {S.yellow}ページ数が変わった: {base.pages} → {after.pages}{S.reset}")
    for k in base.warnings.keys() | after.warnings.keys():
        if base.warnings[k] != after.warnings[k]:
            lines.append(f"  {S.dim}{k}: {base.warnings[k]} → {after.warnings[k]}{S.reset}")
    return ok, lines


# ----------------------------------------------------------------------
# バックアップ・書き込み・復元
# ----------------------------------------------------------------------


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def backup_rel(path: Path, root_dir: Path) -> Path:
    rel = Path(os.path.relpath(path, root_dir))
    if rel.parts and rel.parts[0] == "..":
        rel = Path("_outside") / str(path).lstrip(os.sep).replace(os.sep, "__")
    return rel


def atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name("." + path.name + ".clean_tex_tmp")
    tmp.write_bytes(data)
    shutil.copymode(path, tmp)
    os.replace(tmp, path)


def write_with_backup(results: list[FileResult], main: Path, diff_text: str) -> Path:
    root_dir = main.parent
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    bdir = root_dir / BACKUP_DIRNAME / stamp
    i = 1
    while bdir.exists():
        i += 1
        bdir = root_dir / BACKUP_DIRNAME / f"{stamp}-{i}"
    changed = [r for r in results if r.changed]
    # 解析した後にファイルが書き換えられていないか (エディタで保存したなど)
    for r in changed:
        if sha(r.tf.path.read_bytes()) != r.sha_before:
            sys.exit(f"中止: {r.tf.rel} が解析の後に変更されました. もう一度実行してください (何も書き換えていません).")
    manifest = {"created": stamp, "main": str(main), "files": []}
    for r in changed:
        dest = bdir / backup_rel(r.tf.path, root_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r.tf.path, dest)
        if sha(dest.read_bytes()) != r.sha_before:
            sys.exit(f"中止: {r.tf.rel} のバックアップを検証できません (何も書き換えていません).")
        manifest["files"].append({
            "path": str(r.tf.path), "backup": str(dest.relative_to(bdir)),
            "sha_before": r.sha_before, "sha_after": sha(r.new_data),
            "removed_spaces": sum(g.end - g.start for g in r.removed),
        })
    (bdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (bdir / "changes.diff").write_text(diff_text, encoding="utf-8")
    for r in changed:
        atomic_write(r.tf.path, r.new_data)
    return bdir


def backups(root_dir: Path) -> list[Path]:
    d = root_dir / BACKUP_DIRNAME
    return sorted(p for p in d.iterdir() if (p / "manifest.json").is_file()) if d.is_dir() else []


def restore(bdir: Path, force: bool) -> bool:
    manifest = json.loads((bdir / "manifest.json").read_text(encoding="utf-8"))
    plan = []
    for f in manifest["files"]:
        path, src = Path(f["path"]), bdir / f["backup"]
        cur = sha(path.read_bytes()) if path.is_file() else None
        if cur == f["sha_before"]:
            print(f"  {path.name}: すでにバックアップと同じ内容")
            continue
        if cur != f["sha_after"] and not force:
            print(f"  {S.red}{path}: 書き換えた後に編集されています. 戻すと編集が失われるため中止します "
                  f"(上書きしてよければ --force).{S.reset}")
            return False
        plan.append((path, src))
    for path, src in plan:
        atomic_write(path, src.read_bytes())
        print(f"  {ok_mark(True)} {path.name} をバックアップ {bdir.name} から戻しました")
    return True


# ----------------------------------------------------------------------
# メイン
# ----------------------------------------------------------------------


def parse_cli(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="日本語 LaTeX 論文の, 日本語と英数字・インライン数式の間の不要な半角スペースを削除する.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="例: python clean_tex.py --dry-run  →  確認後に  python clean_tex.py")
    p.add_argument("main", nargs="?", help="ルートの .tex (省略時は \\documentclass を含む .tex を探す)")
    p.add_argument("-n", "--dry-run", action="store_true", help="変更予定箇所を表示するだけで, 何も書き換えない")
    p.add_argument("-y", "--yes", action="store_true", help="書き込み前の確認を省く")
    p.add_argument("--no-compile", action="store_true", help="コンパイル検査を省く")
    p.add_argument("--engine", help="コンパイルに使うコマンド (省略時は自動判定. 例: platex, uplatex, lualatex)")
    p.add_argument("--show-skipped", action="store_true", help="削除しなかった候補の一覧も表示する")
    p.add_argument("--show-other", action="store_true", help="対象外の境目 (日本語␣\\cite など) の一覧も表示する")
    p.add_argument("--max-show", type=int, default=None,
                   help="変更箇所を表示する行数の上限 (既定: ドライランは全部, 実行時は 20, 0 で表示しない)")
    p.add_argument("--diff-out", metavar="FILE", help="unified diff を FILE に書き出す")
    p.add_argument("--restore", nargs="?", const="latest", metavar="STAMP",
                   help="バックアップから元に戻す (省略時は最新)")
    p.add_argument("--list-backups", action="store_true", help="バックアップの一覧を表示する")
    p.add_argument("--force", action="store_true", help="--restore で, 書き換え後に編集されたファイルも上書きする")
    p.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    return p.parse_args(argv)


def confirm(question: str) -> bool:
    if not sys.stdin.isatty():
        print("対話入力ができないため中止しました. 確認を省くには --yes を付けてください.")
        return False
    try:
        return input(f"{question} [y/N]: ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def main(argv: list[str] | None = None) -> int:
    global S
    args = parse_cli(argv)
    use_color = args.color == "always" or (args.color == "auto" and sys.stdout.isatty() and not os.environ.get("NO_COLOR"))
    S = Style(use_color)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    main_path = find_main(args.main)
    root_dir = main_path.parent

    if args.list_backups or args.restore:
        bl = backups(root_dir)
        if not bl:
            print(f"バックアップはありません ({root_dir / BACKUP_DIRNAME}).")
            return 0 if args.list_backups else 1
        if args.list_backups:
            for b in bl:
                m = json.loads((b / "manifest.json").read_text(encoding="utf-8"))
                files = ", ".join(f"{Path(f['path']).name} (−{f['removed_spaces']})" for f in m["files"])
                print(f"  {b.name}  {files}")
            return 0
        bdir = bl[-1] if args.restore == "latest" else root_dir / BACKUP_DIRNAME / args.restore
        if not (bdir / "manifest.json").is_file():
            print(f"バックアップ {args.restore} が見つかりません. --list-backups で一覧を確認してください.")
            return 1
        print(f"バックアップ {bdir.name} から元に戻します.")
        return 0 if restore(bdir, args.force) else 1

    # 1. ファイルの一覧
    files = collect_files(main_path)
    targets = [f for f in files if f.target]
    print(f"{S.bold}== 処理対象のファイル ({len(targets)}) =={S.reset}  {S.dim}(フォルダ: {root_dir}){S.reset}")
    for f in targets:
        print(f"  {f.rel}  {S.dim}[{f.role}]{S.reset}")
    excluded = [f for f in files if not f.target]
    if excluded:
        print(f"対象外 ({len(excluded)}):")
        for f in excluded:
            print(f"  {f.rel}  {S.dim}[{f.role}] {f.reason}{S.reset}")

    # 2. 解析
    results = [process_file(f, f.path == main_path) for f in targets]
    limit = args.max_show if args.max_show is not None else (None if args.dry_run else 20)
    for r in results:
        if r.removed and limit != 0:
            print(f"\n{S.bold}== {r.tf.rel}: 削除する空白 ({S.red}␣{S.reset}{S.bold}) =={S.reset}")
            show_gaps(r, r.removed, limit, with_label=False)
        skipped = [g for g in r.gaps if g.action == "skip"]
        if args.show_skipped and skipped:
            print(f"\n{S.bold}== {r.tf.rel}: 削除しなかった候補 ({S.yellow}␣{S.reset}{S.bold}) =={S.reset}")
            show_gaps(r, skipped, None, with_label=True)
        others = [g for g in r.gaps if g.action == "other"]
        if args.show_other and others:
            print(f"\n{S.bold}== {r.tf.rel}: 対象外の境目 ({S.yellow}␣{S.reset}{S.bold}) =={S.reset}")
            show_gaps(r, others, None, with_label=True)
    print_summary(results, args.show_skipped)

    diff_text = unified_diff(results)
    if args.diff_out:
        Path(args.diff_out).write_text(diff_text, encoding="utf-8")
        print(f"\n差分を {args.diff_out} に書き出しました.")

    if not all(ok for r in results for ok, _ in r.checks):
        print(f"\n{S.red}検証に失敗したため中止します (何も書き換えていません).{S.reset}")
        return 2
    changed = [r for r in results if r.changed]
    if not changed:
        print("\n変更はありません.")
        return 0

    # 3. コンパイル検査 (一時フォルダで処理前と処理後を比べる)
    compile_ok: bool | None = None
    checker = None
    base = None
    if not args.no_compile:
        engine, why = (args.engine, "--engine") if args.engine else detect_engine(main_path, results[0].text)
        print(f"\n{S.bold}== コンパイル検査 ({engine}, {why} から判定) =={S.reset}")
        checker = CompileChecker(engine, main_path)
        base = checker.compile({main_path.name: results[0].text})
        if not base.ran:
            print(f"  {S.yellow}検査できません: {base.message}.{S.reset}")
        else:
            mod_files = {main_path.name: results[0].new_text}
            unchecked = []
            for r in (r for r in results[1:] if r.changed):
                if r.tf.include_name.startswith(("./", "../", "/")) or r.tf.rel.startswith(".."):
                    unchecked.append(r.tf.rel)
                else:
                    mod_files[os.path.relpath(r.tf.path, root_dir)] = r.new_text
            after = checker.compile(mod_files)
            if checker.stubbed:
                print(f"  {S.yellow}次のパッケージがこの環境に無いため, 検査では空のスタブで代用: "
                      f"{', '.join(checker.stubbed)}{S.reset}")
            compile_ok, lines = compare(base, after)
            for line in lines:
                print("  " + line)
            if unchecked:
                print(f"  {S.yellow}次のファイルは書き込み後の検査でだけ確かめます: {', '.join(unchecked)}{S.reset}")
            print(f"  {ok_mark(compile_ok)} " + ("処理によって増えたエラーはありません" if compile_ok
                                                  else "処理後にエラーが増えます"))

    if args.dry_run:
        if checker:
            checker.cleanup()
        print(f"\n{S.bold}ドライランのため, 何も書き換えていません.{S.reset} 実行するには: python clean_tex.py")
        return 0
    if compile_ok is False:
        print(f"\n{S.red}処理後にコンパイルエラーが増えるため中止します (何も書き換えていません).{S.reset}")
        checker.cleanup()
        return 3
    if compile_ok is None and not args.no_compile:
        print(f"{S.yellow}コンパイル検査ができませんでした. 検査なしで進めるには --no-compile を付けてください.{S.reset}")
        if checker:
            checker.cleanup()
        return 3

    # 4. 確認, バックアップ, 書き込み
    n_sp = sum(g.end - g.start for r in changed for g in r.removed)
    if not args.yes and not confirm(f"\n{len(changed)} ファイルから半角スペース {n_sp} 個を削除します. 書き込みますか?"):
        print("中止しました (何も書き換えていません).")
        if checker:
            checker.cleanup()
        return 1
    bdir = write_with_backup(results, main_path, diff_text)
    print(f"\n{ok_mark(True)} {len(changed)} ファイルを書き換えました. バックアップ: {bdir}")
    print(f"  差分: {bdir / 'changes.diff'}")

    # 5. 書き込んだ実物でもう一度コンパイルし, エラーが増えていれば戻す
    if checker and base and base.ran:
        post = checker.compile({main_path.name: main_path.read_text(encoding="utf-8")})
        ok, lines = compare(base, post)
        print(f"\n{S.bold}== 書き込み後のコンパイル検査 =={S.reset}")
        for line in lines:
            print("  " + line)
        checker.cleanup()
        if not ok:
            print(f"  {S.red}エラーが増えたため, バックアップから元に戻します.{S.reset}")
            restore(bdir, force=True)
            return 3
        print(f"  {ok_mark(True)} 書き込み後もエラーは増えていません")
    print(f"\n元に戻すには: python clean_tex.py --restore {bdir.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
