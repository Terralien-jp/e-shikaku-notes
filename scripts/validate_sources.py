#!/usr/bin/env python3
"""出典カード（ledger/sources/<slug>.json）の機械検証。

★fail-closed。ERROR が1件でもあれば exit 1。

出典カードは「執筆前に一次情報を1回だけ集めておく」ための中間成果物。
112本の執筆エージェントが各自で探し回ると、同じ論文・同じ公式ドキュメントを
何度も取得することになる（AIP-C01 の実測で WebFetch だけで約107万トークン）。

★本文の抜粋は保存しない。URL とタイトルと「何を裏付ける出典か」の1行だけ。
  他人の著作物をリポジトリに溜め込まないため（LICENSE.md 3 項）。

  python3 scripts/validate_sources.py                # 全カード
  python3 scripts/validate_sources.py --check-links   # URLの生死も見る
  python3 scripts/validate_sources.py --check-titles  # arXiv ID とタイトルの対応を照合

★--check-titles が要る理由（2026-08-22）。URL が生きていることは「その論文である」ことの
  証拠にならない。arXiv ID を取り違えたカードが3件見つかっており、いずれも 200 を返すため
  --check-links では捕まらなかった（別の論文を指したまま執筆に渡ると、記事の主張の帰属が壊れる）。
  arXiv API はレート制限が厳しいので CI には入れず、カードを作った直後に手で回す。
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "ledger" / "sources"
LEDGER = ROOT / "ledger" / "concepts.json"

sys.path.insert(0, str(ROOT / "scripts"))
from validate import SOURCE_ALLOWLIST, http_status  # noqa: E402

MAX_WHY = 120


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def check_arxiv_titles(cards) -> list[str]:
    """arXiv の abs ページを引いて、カードの title がその ID の論文かを確かめる。"""
    out: list[str] = []
    targets: dict[str, list[tuple[str, str]]] = {}
    for p in cards:
        for s in json.loads(p.read_text(encoding="utf-8")).get("sources", []):
            m = re.search(r"arxiv\.org/abs/([\d.]+)", str(s.get("url", "")))
            if m:
                targets.setdefault(m.group(1), []).append((p.stem, str(s.get("title", ""))))
    for n, (aid, uses) in enumerate(sorted(targets.items()), 1):
        req = urllib.request.Request(f"https://arxiv.org/abs/{aid}",
                                     headers={"User-Agent": "Mozilla/5.0 e-shikaku-notes"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                page = r.read().decode("utf-8", "replace")
        except Exception as e:
            out.append(f"SOURCES_TITLE_UNCHECKED: arXiv {aid} を取得できません（{e}）")
            continue
        m = re.search(r"<title>\[[\d.v]+\]\s*(.*?)</title>", page, re.S)
        if not m:
            out.append(f"SOURCES_TITLE_UNCHECKED: arXiv {aid} のタイトルを読めません")
            continue
        real = html.unescape(re.sub(r"\s+", " ", m.group(1)).strip())
        for stem, claimed in uses:
            if not (_norm(claimed)[:35] in _norm(real) or _norm(real)[:35] in _norm(claimed)):
                out.append(f"SOURCES_TITLE_MISMATCH: {stem}: {aid} は「{real}」であって"
                           f"「{claimed}」ではありません")
        time.sleep(1)
    return out


def check_doi_titles(cards) -> list[str]:
    """Crossref に DOI を引いて、カードの title がその DOI の論文かを確かめる。

    DOI は解決先が出版社サイトなので、生死チェック（HTTP 200）では「別の論文の DOI」を
    一切捕まえられない。実際に 10.1007/BF02591564 を Lin 1991 として登録したカードがあり、
    その DOI は 1980 年の応用地質学の総括記事のものだった（2026-08-22）。
    """
    out: list[str] = []
    targets: dict[str, list[tuple[str, str]]] = {}
    for p in cards:
        for s in json.loads(p.read_text(encoding="utf-8")).get("sources", []):
            m = re.search(r"doi\.org/(10\.[^\s\"']+)", str(s.get("url", "")))
            if m:
                targets.setdefault(m.group(1).rstrip("/"), []).append((p.stem, str(s.get("title", ""))))
    for doi, uses in sorted(targets.items()):
        req = urllib.request.Request(
            "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe=""),
            headers={"User-Agent": "e-shikaku-notes (mailto:terralienjp@gmail.com)"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                msg = json.loads(r.read().decode("utf-8", "replace"))["message"]
        except Exception as e:
            out.append(f"SOURCES_TITLE_UNCHECKED: DOI {doi} を Crossref で引けません（{e}）")
            continue
        titles = msg.get("title") or []
        if not titles:
            out.append(f"SOURCES_TITLE_UNCHECKED: DOI {doi} に title がありません")
            continue
        real = html.unescape(re.sub(r"\s+", " ", titles[0]).strip())
        for stem, claimed in uses:
            if not (_norm(claimed)[:35] in _norm(real) or _norm(real)[:35] in _norm(claimed)):
                out.append(f"SOURCES_TITLE_MISMATCH: {stem}: {doi} は「{real}」であって"
                           f"「{claimed}」ではありません")
        time.sleep(1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="出典カードの機械検証")
    ap.add_argument("--check-links", action="store_true")
    ap.add_argument("--check-titles", action="store_true",
                    help="arXiv 出典の ID と title の対応を照合する（ネットワーク使用・低速）")
    ap.add_argument("--warn-only", action="store_true", help="★CI では絶対に付けない")
    args = ap.parse_args(argv)

    errors: list[str] = []
    warns: list[str] = []

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    known = {c["slug"] for c in ledger["concepts"]}
    cards = sorted(SRC_DIR.glob("*.json")) if SRC_DIR.exists() else []

    seen_urls: dict[str, str] = {}
    for path in cards:
        stem = path.stem
        if stem not in known:
            errors.append(f"SOURCES_ORPHAN: {stem}: 台帳に無い slug の出典カード")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            errors.append(f"SOURCES_JSON: {stem}: JSON が壊れています: {e}")
            continue

        items = data.get("sources")
        if not isinstance(items, list) or not items:
            errors.append(f"SOURCES_EMPTY: {stem}: sources が空です")
            continue
        if len(items) > 6:
            warns.append(f"SOURCES_MANY: {stem}: 出典が {len(items)} 件（多すぎると執筆が散ります）")

        for i, s in enumerate(items, 1):
            for key in ("title", "url", "why"):
                if not str(s.get(key, "")).strip():
                    errors.append(f"SOURCES_SHAPE: {stem}#{i}: {key} が空です")
            url = str(s.get("url", ""))
            host = urllib.parse.urlparse(url).netloc.lower()
            if host and host not in SOURCE_ALLOWLIST:
                errors.append(f"SOURCES_DOMAIN: {stem}#{i}: 許可リスト外の出典: {url}")
            if len(str(s.get("why", ""))) > MAX_WHY:
                errors.append(f"SOURCES_WHY_LONG: {stem}#{i}: why が長すぎます"
                              f"（{MAX_WHY}字以内。出典の要約ではなく『何を裏付けるか』の1行）")
            if "excerpt" in s or "content" in s or "body" in s:
                errors.append(f"SOURCES_EXCERPT: {stem}#{i}: 本文の抜粋は保存しません（LICENSE.md 3項）")
            if url:
                seen_urls.setdefault(url, stem)

        if args.check_links:
            for i, s in enumerate(items, 1):
                code = http_status(str(s.get("url", "")))
                if code in (404, 410):
                    errors.append(f"SOURCES_DEAD: {stem}#{i}: 出典が {code}: {s.get('url')}")
                elif code == 0 or code >= 400:
                    warns.append(f"SOURCES_UNCHECKED: {stem}#{i}: 確認できません({code}): {s.get('url')}")

    if args.check_titles:
        errors += check_arxiv_titles(cards)
        errors += check_doi_titles(cards)

    todo = sorted(known - {p.stem for p in cards})
    for f in errors:
        print(f"[ERROR] {f}")
    for f in warns:
        print(f"[WARN] {f}")
    print(f"\n出典カード {len(cards)}/{len(known)} 件 — ERROR {len(errors)} / WARN {len(warns)}")
    if todo:
        print(f"未作成 {len(todo)} 件（先頭）: {', '.join(todo[:8])}")
    return 1 if (errors and not args.warn_only) else 0


if __name__ == "__main__":
    sys.exit(main())
