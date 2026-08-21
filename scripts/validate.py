#!/usr/bin/env python3
"""E資格 学び直しノートの機械検証。

★このスクリプトは ERROR が1件でもあれば exit 1 で終わる（fail-closed）。
  「--strict を付け忘れて ERROR があるのに exit 0」という事故を構造的に起こさないため、
  緩める側にフラグを置いた（--warn-only）。CI では絶対に付けないこと。

ネットワークは既定で使わない。--check-links / --check-code は明示的に on にする。

  python3 scripts/validate.py                     # 全ノート（ネットワーク無し）
  python3 scripts/validate.py content/foo.md      # 1本だけ
  python3 scripts/validate.py --check-code        # コードブロックを実行して確かめる
  python3 scripts/validate.py --check-links       # 出典URLの生死を確かめる
  python3 scripts/validate.py --json              # 機械可読
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content"
LEDGER = ROOT / "ledger" / "concepts.json"

REQUIRED_FM = ("exam", "concept", "slug", "tier", "area", "summary", "updated")
VALID_TIERS = {"A", "B", "C"}
VALID_AREAS = {"応用数学", "機械学習", "深層学習", "開発・運用環境"}

# 本文の H2 はこの順で固定する。読者が別のノートへ移っても同じ場所に同じものがある状態を作る。
H2_ORDER = (
    "ひとことで言うと",
    "なぜ必要か",
    "仕組み",
    "試験でどう問われるか",
    "実装で確かめる",
    "取り違えやすいもの",
    "想起チェック",
)
H2_REQUIRED = {"ひとことで言うと", "仕組み", "試験でどう問われるか", "想起チェック"}

# §5-0 相当。全 H2 に視覚要素を1つ以上置く。
VISUAL_PATTERNS = (
    r"^\s*\|.*\|",                    # 表
    r'<div class="analogy"',
    r'<div class="caution"',
    r'<details class="recall"',
    r"^\s*\$\$",                      # ディスプレイ数式
    r"^```",                          # コードブロック
    r"^\s*[-*]\s+\[[ x]\]",           # チェックリスト
)

# 出典に使ってよいドメイン。**一次情報だけ**。
# 他社の教材・問題集・個人ブログは、正しくても入れない（著作物であり、
# 「他人のものを抜かず自分で鋳造する」方針のため）。増やすときは Issue で相談。
SOURCE_ALLOWLIST = {
    # 論文・プレプリント
    "arxiv.org", "doi.org", "www.jmlr.org", "jmlr.org",
    "proceedings.neurips.cc", "proceedings.mlr.press", "openreview.net",
    "dl.acm.org", "ieeexplore.ieee.org", "aclanthology.org",
    # 公式ドキュメント
    "pytorch.org", "docs.pytorch.org", "numpy.org", "scipy.org",
    "scikit-learn.org", "www.tensorflow.org", "keras.io",
    "docs.python.org", "pandas.pydata.org", "matplotlib.org",
    "huggingface.co", "developer.nvidia.com", "docs.nvidia.com",
    # 教科書・講義（著者が公開しているもの）
    "www.deeplearningbook.org",
    # 資格運営
    "www.jdla.org", "jdla.org",
}


@dataclass
class Finding:
    level: str      # ERROR / WARN
    rule: str
    message: str
    line: int | None = None


# ------------------------------------------------------------------ helpers

def split_frontmatter(text: str) -> tuple[str | None, str]:
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    return (m.group(1), m.group(2)) if m else (None, text)


def strip_code_blocks(body: str) -> str:
    return re.sub(r"^```.*?^```", "", body, flags=re.S | re.M)


def load_ledger() -> tuple[dict[str, dict], list[Finding]]:
    if not LEDGER.exists():
        return {}, [Finding("ERROR", "LEDGER", f"概念台帳がありません: {LEDGER.relative_to(ROOT)}")]
    try:
        data = json.loads(LEDGER.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {}, [Finding("ERROR", "LEDGER", f"概念台帳が JSON として壊れています: {e}")]
    return {c["slug"]: c for c in data.get("concepts", [])}, []


def note_paths() -> list[Path]:
    return sorted(CONTENT.glob("*.md")) if CONTENT.exists() else []


# ------------------------------------------------------------------ 各検査

def check_frontmatter(stem: str, fm: str, ledger: dict[str, dict]) -> list[Finding]:
    out: list[Finding] = []
    for key in REQUIRED_FM:
        if not re.search(rf"^{key}:\s*\S", fm, re.M):
            out.append(Finding("ERROR", "FRONTMATTER", f"{stem}: {key} が未設定"))

    m = re.search(r"^slug:\s*(\S+)", fm, re.M)
    if m and m.group(1).strip("\"'") != stem:
        out.append(Finding("ERROR", "SLUG_MISMATCH",
                           f"{stem}: frontmatter の slug がファイル名と不一致"))

    m = re.search(r"^tier:\s*(\S+)", fm, re.M)
    if m and m.group(1).strip("\"'") not in VALID_TIERS:
        out.append(Finding("ERROR", "TIER", f"{stem}: tier は A/B/C のいずれか"))

    m = re.search(r"^area:\s*(.+)$", fm, re.M)
    if m and m.group(1).strip().strip("\"'") not in VALID_AREAS:
        out.append(Finding("ERROR", "AREA",
                           f"{stem}: area が台帳の区分にない: {m.group(1).strip()}"))

    if ledger and stem not in ledger:
        out.append(Finding("ERROR", "LEDGER",
                           f"{stem}: 概念台帳に無い slug（一覧に出ないので迷子になります）"))
    return out


def check_sources(stem: str, fm: str) -> list[Finding]:
    """出典は一次情報のみ。空も不可。"""
    out: list[Finding] = []
    urls = re.findall(r"^\s*url:\s*(\S+)", fm, re.M)
    if not urls:
        out.append(Finding("ERROR", "SOURCES", f"{stem}: sources が空（出典なしの解説は公開しない）"))
    for u in urls:
        host = urllib.parse.urlparse(u).netloc.lower()
        if host not in SOURCE_ALLOWLIST:
            out.append(Finding("ERROR", "SOURCES",
                               f"{stem}: 一次情報の許可リストに無い出典: {u}"))
    return out


def check_h2(stem: str, body: str) -> list[Finding]:
    """H2 の見出し名と順序を固定する。"""
    out: list[Finding] = []
    found = [m.group(1).strip() for m in re.finditer(r"^##\s+(.+)$", strip_code_blocks(body), re.M)]
    unknown = [h for h in found if h not in H2_ORDER]
    for h in unknown:
        out.append(Finding("ERROR", "H2_UNKNOWN", f"{stem}: 規定にない H2 見出し: 「{h}」"))
    for req in H2_REQUIRED:
        if req not in found:
            out.append(Finding("ERROR", "H2_MISSING", f"{stem}: 必須の H2 がありません: 「{req}」"))
    known = [h for h in found if h in H2_ORDER]
    if known != sorted(known, key=H2_ORDER.index):
        out.append(Finding("ERROR", "H2_ORDER", f"{stem}: H2 の順序が規定と違います"))
    return out


def check_visual_gap(stem: str, body: str) -> list[Finding]:
    """全 H2 章に視覚要素を1つ以上。文字段落だけの章を作らない。"""
    out: list[Finding] = []
    parts = re.split(r"^(##\s+.+)$", body, flags=re.M)
    for i in range(1, len(parts), 2):
        head = parts[i].lstrip("# ").strip()
        section = parts[i + 1] if i + 1 < len(parts) else ""
        if not any(re.search(p, section, re.M) for p in VISUAL_PATTERNS):
            out.append(Finding("ERROR", "VISUAL_GAP",
                               f"{stem}: 「{head}」に視覚要素がありません（表・囲み・数式・コード・想起チェック）"))
    return out


# 表の列数が揃っていない形は、機械検証を通ったまま崩れて表示される。
# 実害（2026-08-22）: 見出し3列に対し区切り行が `|---|---:|---|---|` の4列という
# ノートが ERROR 0 で通り、人間が読んで初めて見つかった。
TABLE_SEP = re.compile(r"^\s*\|(?:\s*:?-{1,}:?\s*\|)+\s*$")
# セル区切りに見えるが区切りではないもの: 数式の中の \| （ノルム記法）と、
# エスケープされた \|。数えるより先に伏せる。
INLINE_MATH = re.compile(r"\$[^$\n]+\$")


def _cells(row: str) -> int:
    """行頭・行末のパイプを落として列数を数える。数式とエスケープは伏せてから数える。"""
    s = INLINE_MATH.sub(lambda m: "x" * len(m.group(0)), row.strip())
    s = s.replace(r"\|", "x")
    s = s[1:] if s.startswith("|") else s
    s = s[:-1] if s.endswith("|") else s
    return len(s.split("|"))


def check_tables(stem: str, body: str) -> list[Finding]:
    """表の見出し・区切り・本体で列数が揃っているか。"""
    out: list[Finding] = []
    lines = body.splitlines()
    fenced = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced or not TABLE_SEP.match(line) or i == 0:
            continue
        head = lines[i - 1]
        if not head.strip().startswith("|"):
            continue
        n = _cells(head)
        if _cells(line) != n:
            out.append(Finding("ERROR", "TABLE_COLUMNS",
                               f"{stem}: 表の区切り行が {_cells(line)} 列、見出しが {n} 列で揃っていません",
                               i + 1))
        for j in range(i + 1, len(lines)):
            row = lines[j]
            if not row.strip().startswith("|"):
                break
            if _cells(row) != n:
                out.append(Finding("ERROR", "TABLE_COLUMNS",
                                   f"{stem}: 表の行が {_cells(row)} 列、見出しが {n} 列で揃っていません",
                                   j + 1))
    return out


# raw HTML ブロック（囲み・想起チェック）の中は、開きタグの直後に空行が無いと
# CommonMark が中身をまるごと生の HTML として扱う＝Markdown も数式も解釈されない。
# 実害（2026-08-18）: caution / details に書いた $...$ が、KaTeX を通らず
# 「$\varepsilon$」の生文字のままページに出た。ERROR 0 のまま壊れていたので機械で塞ぐ。
RAW_HTML_OPEN = re.compile(r'^\s*<(div class="(?:analogy|caution)"|details class="recall")>\s*$')
RAW_HTML_CLOSE = re.compile(r'^\s*</(div|details)>\s*$')


def check_raw_html_blocks(stem: str, body: str) -> list[Finding]:
    """囲み・想起チェックの中身が Markdown として解釈される形になっているか。"""
    out: list[Finding] = []
    lines = strip_code_blocks(body).splitlines()
    for i, line in enumerate(lines):
        if RAW_HTML_OPEN.match(line):
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if nxt.strip() and not nxt.lstrip().startswith("<summary"):
                out.append(Finding("ERROR", "RAW_HTML_TIGHT",
                                   f"{stem}: {i+1}行目 の囲みは開きタグの直後に空行が必要です"
                                   "（無いと中身の Markdown と数式が解釈されません）"))
        if RAW_HTML_CLOSE.match(line):
            prv = lines[i - 1] if i > 0 else ""
            if prv.strip():
                out.append(Finding("ERROR", "RAW_HTML_TIGHT",
                                   f"{stem}: {i+1}行目 の閉じタグの直前に空行が必要です"))
        if "</summary>" in line:
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if nxt.strip():
                out.append(Finding("ERROR", "RAW_HTML_TIGHT",
                                   f"{stem}: {i+1}行目 </summary> の直後に空行が必要です"
                                   "（無いと答えの Markdown と数式が解釈されません）"))
        if line.lstrip().startswith("<summary") and "$" in line:
            out.append(Finding("ERROR", "MATH_IN_SUMMARY",
                               f"{stem}: {i+1}行目 <summary> の中の数式は描画されません"
                               "（言葉で書くか、本文側に出す）"))
    return out


def check_internal_links(stem: str, body: str, existing: set[str]) -> list[Finding]:
    out: list[Finding] = []
    for m in re.finditer(r"/learn/e-shikaku/([a-z0-9-]+)/", body):
        if m.group(1) not in existing:
            out.append(Finding("ERROR", "LINK_BROKEN",
                               f"{stem}: 未作成のページへの内部リンク: /learn/e-shikaku/{m.group(1)}/"))
    return out


def check_math(stem: str, body: str) -> list[Finding]:
    """数式の構文。KaTeX が入っていればパースまで、無ければ区切りの対応だけ見る。"""
    out: list[Finding] = []
    stripped = strip_code_blocks(body)

    if re.search(r"\\\(|\\\[", stripped):
        out.append(Finding("ERROR", "MATH_DELIM",
                           f"{stem}: \\( \\[ 記法は使いません（$ / $$ に統一）"))

    if stripped.count("$$") % 2 != 0:
        out.append(Finding("ERROR", "MATH_DELIM", f"{stem}: $$ の数が奇数（閉じ忘れ）"))

    inline = re.sub(r"\$\$.*?\$\$", "", stripped, flags=re.S)
    if len(re.findall(r"(?<!\\)\$", inline)) % 2 != 0:
        out.append(Finding("ERROR", "MATH_DELIM", f"{stem}: インライン $ の数が奇数（閉じ忘れ）"))

    exprs = [m.group(1) for m in re.finditer(r"\$\$(.+?)\$\$", stripped, re.S)]
    exprs += [m.group(1) for m in re.finditer(r"(?<!\$)\$([^$\n]+)\$(?!\$)", inline)]
    if exprs:
        out += _katex_parse(stem, exprs)
    out += check_math_escape(stem, body)
    return out


# バックスラッシュが落ちた LaTeX コマンド（\mathbf → mathbf）。KaTeX は
# ただの英字列として通してしまうため、パーサでは捕まらない。読者には
# 「mathbf{x}」がそのまま表示される。
BARE_CMDS = (
    "mathbf mathbb mathrm mathsf mathcal operatorname frac sqrt sum prod "
    "ldots cdots cdot odot oplus otimes times partial nabla infty "
    "lVert rVert langle rangle boldsymbol propto approx neq leq geq notin subset mapsto "
    "alpha beta gamma delta epsilon varepsilon zeta eta theta kappa lambda mu nu xi "
    "rho sigma tau phi varphi chi psi omega "
    "Gamma Delta Theta Lambda Sigma Phi Psi Omega"
).split()
BARE_RE = re.compile(r"(?<![\\A-Za-z])(" + "|".join(sorted(BARE_CMDS, key=len, reverse=True)) + r")(?![A-Za-z])")
MATH_SPAN = re.compile(r"\$\$.*?\$\$|\$[^$\n]+\$", re.S)


def check_math_escape(stem: str, body: str) -> list[Finding]:
    out: list[Finding] = []
    stripped = strip_code_blocks(body)
    for span in MATH_SPAN.finditer(stripped):
        expr = span.group(0)
        for m in BARE_RE.finditer(expr):
            # x_{max} のような添字の英単語は誤検出なので除く
            before = expr[:m.start()]
            if before.endswith("{") and before[:-1].endswith(("_", "^")):
                continue
            out.append(Finding("ERROR", "MATH_ESCAPE",
                               f"{stem}: 数式内の {m.group(1)} にバックスラッシュがありません: {expr[:60]}"))
    if "\t" in stripped:
        out.append(Finding("ERROR", "MATH_ESCAPE", f"{stem}: 本文にタブ文字があります（\\theta 等の書き損じの疑い）"))
    return out


def _katex_parse(stem: str, exprs: list[str]) -> list[Finding]:
    """node + katex があれば本物のパーサに掛ける。無ければ WARN で素通し。"""
    probe = subprocess.run(["node", "-e", "require.resolve('katex')"],
                           cwd=ROOT, capture_output=True, text=True)
    if probe.returncode != 0:
        return [Finding("WARN", "MATH_SKIP", f"{stem}: katex 未導入のため数式パースは未実施")]

    script = (
        "const katex=require('katex');"
        "const list=JSON.parse(process.argv[1]);const bad=[];"
        "for(const e of list){try{katex.renderToString(e,{throwOnError:true});}"
        "catch(err){bad.push([e.slice(0,60),err.message]);}}"
        "console.log(JSON.stringify(bad));"
    )
    r = subprocess.run(["node", "-e", script, json.dumps(exprs)],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        return [Finding("WARN", "MATH_SKIP", f"{stem}: 数式パースを実行できませんでした")]
    return [Finding("ERROR", "MATH_PARSE", f"{stem}: 数式がパースできません: {e} — {msg}")
            for e, msg in json.loads(r.stdout or "[]")]


def check_code(stem: str, body: str) -> list[Finding]:
    """掲載した Python を実際に走らせる。動かないコードを教材に置かない。

    ```python no-run  を付けたブロックは対象外（外部データ・GPU が要るもの）。
    """
    out: list[Finding] = []
    for i, m in enumerate(re.finditer(r"^```python(?P<flags>[^\n]*)\n(?P<code>.*?)^```", body, re.S | re.M), 1):
        if "no-run" in m.group("flags"):
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
            fh.write(m.group("code"))
            tmp = fh.name
        try:
            r = subprocess.run([sys.executable, tmp], capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                tail = (r.stderr or "").strip().splitlines()[-1:] or ["(no stderr)"]
                out.append(Finding("ERROR", "CODE_FAILED",
                                   f"{stem}: コードブロック #{i} が実行に失敗: {tail[0]}"))
        except subprocess.TimeoutExpired:
            out.append(Finding("ERROR", "CODE_TIMEOUT", f"{stem}: コードブロック #{i} が60秒で終わりません"))
        finally:
            Path(tmp).unlink(missing_ok=True)
    return out


def http_status(url: str) -> int:
    """HEAD の 404 を信用しない。HEAD を正しく実装していないサイトがあるため GET で確かめる。"""
    for method in ("HEAD", "GET"):
        req = urllib.request.Request(url, method=method,
                                     headers={"User-Agent": "e-shikaku-notes-validator"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status
        except urllib.error.HTTPError as e:
            if method == "GET" or e.code not in (403, 404, 405):
                return e.code
        except Exception:
            return 0
    return 0


def check_links(stem: str, fm: str) -> list[Finding]:
    out: list[Finding] = []
    for u in re.findall(r"^\s*url:\s*(\S+)", fm, re.M):
        code = http_status(u)
        if code in (404, 410):
            out.append(Finding("ERROR", "SOURCE_DEAD", f"{stem}: 出典が {code}: {u}"))
        elif code == 0 or code >= 400:
            # 回線都合や bot 対策で止めない。壊れていないのに止まるのは安全ではない。
            out.append(Finding("WARN", "SOURCE_UNCHECKED", f"{stem}: 出典を確認できません({code}): {u}"))
    return out


# ------------------------------------------------------------------ 実行

def validate(paths: list[Path], *, check_links_: bool, check_code_: bool) -> list[Finding]:
    ledger, findings = load_ledger()
    existing = {p.stem for p in note_paths()}

    for p in paths:
        stem = p.stem
        text = p.read_text(encoding="utf-8")
        fm, body = split_frontmatter(text)
        if fm is None:
            findings.append(Finding("ERROR", "FRONTMATTER", f"{stem}: YAML frontmatter がありません"))
            continue
        findings += check_frontmatter(stem, fm, ledger)
        findings += check_sources(stem, fm)
        findings += check_h2(stem, body)
        findings += check_visual_gap(stem, body)
        findings += check_tables(stem, body)
        findings += check_raw_html_blocks(stem, body)
        findings += check_internal_links(stem, body, existing)
        findings += check_math(stem, body)
        if check_code_:
            findings += check_code(stem, body)
        if check_links_:
            findings += check_links(stem, fm)
    return findings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E資格 学び直しノートの機械検証")
    ap.add_argument("paths", nargs="*", help="検査する .md（省略時は content/ 全部）")
    ap.add_argument("--check-links", action="store_true", help="出典URLの生死を確かめる（ネットワーク使用）")
    ap.add_argument("--check-code", action="store_true", help="コードブロックを実行して確かめる")
    ap.add_argument("--json", action="store_true", help="機械可読で出す")
    ap.add_argument("--warn-only", action="store_true",
                    help="ERROR でも exit 0 にする。★CI では絶対に付けない")
    args = ap.parse_args(argv)

    paths = [Path(p) for p in args.paths] if args.paths else note_paths()
    if not paths:
        print("検査対象がありません（content/*.md が0件）", file=sys.stderr)
        return 0

    findings = validate(paths, check_links_=args.check_links, check_code_=args.check_code)
    errors = [f for f in findings if f.level == "ERROR"]
    warns = [f for f in findings if f.level == "WARN"]

    if args.json:
        print(json.dumps({"errors": [asdict(f) for f in errors],
                          "warns": [asdict(f) for f in warns]}, ensure_ascii=False, indent=2))
    else:
        for f in errors + warns:
            print(f"[{f.level}] {f.rule}: {f.message}")
        print(f"\n{len(paths)} 本を検査 — ERROR {len(errors)} / WARN {len(warns)}")

    return 1 if (errors and not args.warn_only) else 0


if __name__ == "__main__":
    sys.exit(main())
