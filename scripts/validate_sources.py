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
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "ledger" / "sources"
LEDGER = ROOT / "ledger" / "concepts.json"

sys.path.insert(0, str(ROOT / "scripts"))
from validate import SOURCE_ALLOWLIST, http_status  # noqa: E402

MAX_WHY = 120


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="出典カードの機械検証")
    ap.add_argument("--check-links", action="store_true")
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
