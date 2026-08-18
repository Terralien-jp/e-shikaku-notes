#!/usr/bin/env python3
"""概念台帳（ledger/concepts.json）の機械検証。

★fail-closed。ERROR が1件でもあれば exit 1（緩める側に --warn-only を置く）。

台帳は「何を書くべきか」の唯一の真なので、ここが壊れると全ノートが道連れになる。
特に **JDLA シラバス本文の転記** を機械で弾くことを重視している。著作物であり、
転記すると公開できなくなるため（docs/writer-brief.md §0-5）。

  python3 scripts/validate_ledger.py
  python3 scripts/validate_ledger.py --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "ledger" / "concepts.json"
CONTENT = ROOT / "content"

VALID_AREAS = ("応用数学", "機械学習", "深層学習", "開発・運用環境")
VALID_TIERS = ("A", "B", "C")
VALID_STATUS = ("todo", "drafting", "done")
SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

# 転記の検出。シラバスは箇条書きの語句で構成されるので、**文になっていたら疑う**。
# 自分の言葉で付けた概念名は名詞句であって、句点や「〜すること」を含まない。
TRANSCRIPT_MARKERS = ("。", "を理解する", "を説明できる", "について", "に関する", "を用いた")
CONCEPT_MAX_LEN = 30


@dataclass
class Finding:
    level: str
    rule: str
    message: str


def check(data: dict) -> list[Finding]:
    out: list[Finding] = []

    # 台帳が出来上がる前（bootstrap）は「網羅性」だけ WARN に落とす。
    # 形式・転記禁止・整合は bootstrap でも ERROR のまま＝緩めるのは網羅性だけ。
    bootstrap = data.get("ledger_status") == "bootstrap"
    cover = "WARN" if bootstrap else "ERROR"

    for key in ("exam", "syllabus_version", "syllabus_source", "areas", "concepts", "ledger_status"):
        if key not in data:
            out.append(Finding("ERROR", "LEDGER_SHAPE", f"トップレベルに {key} がありません"))
    if out:
        return out

    if data["ledger_status"] not in ("bootstrap", "complete"):
        out.append(Finding("ERROR", "LEDGER_STATUS",
                           "ledger_status は bootstrap / complete のいずれか"))

    if str(data["syllabus_version"]).strip().upper() in ("", "TBD"):
        out.append(Finding(cover, "SYLLABUS_VERSION",
                           "syllabus_version が TBD のままです（どの版に対する台帳か特定できません）"))

    src = str(data.get("syllabus_source", ""))
    if not src.startswith("https://www.jdla.org/"):
        out.append(Finding("ERROR", "SYLLABUS_SOURCE",
                           f"syllabus_source が JDLA 公式のURLではありません: {src}"))

    concepts = data["concepts"]
    if not concepts:
        out.append(Finding(cover, "LEDGER_EMPTY", "concepts が空です"))
        return out

    seen_slugs: set[str] = set()
    seen_names: set[str] = set()
    per_area: dict[str, int] = {a: 0 for a in VALID_AREAS}

    for i, c in enumerate(concepts):
        tag = c.get("slug") or f"#{i}"

        for key in ("slug", "concept", "area", "tier", "syllabus_refs", "status", "optional"):
            if key not in c:
                out.append(Finding("ERROR", "CONCEPT_SHAPE", f"{tag}: {key} がありません"))
                continue

        slug = str(c.get("slug", ""))
        if not SLUG_RE.match(slug):
            out.append(Finding("ERROR", "SLUG_FORMAT", f"{tag}: slug の形式が不正（英小文字・数字・ハイフン）"))
        if slug in seen_slugs:
            out.append(Finding("ERROR", "SLUG_DUP", f"{tag}: slug が重複しています"))
        seen_slugs.add(slug)

        name = str(c.get("concept", "")).strip()
        if not name:
            out.append(Finding("ERROR", "CONCEPT_NAME", f"{tag}: concept が空です"))
        if name in seen_names:
            out.append(Finding("WARN", "CONCEPT_DUP", f"{tag}: 同じ concept 名が複数あります: {name}"))
        seen_names.add(name)
        if len(name) > CONCEPT_MAX_LEN:
            out.append(Finding("ERROR", "CONCEPT_TRANSCRIPT",
                               f"{tag}: concept が長すぎます（{len(name)}字）。"
                               "シラバス本文の転記が疑われます。自分の言葉の名詞句にしてください"))
        for mark in TRANSCRIPT_MARKERS:
            if mark in name:
                out.append(Finding("ERROR", "CONCEPT_TRANSCRIPT",
                                   f"{tag}: concept が文になっています（「{mark}」を含む）。"
                                   "シラバス本文の転記は禁止です"))
                break

        # ★シラバスでグレー網掛（出題対象外）の節に紐づく概念は optional: true でなければならない。
        #   ここを人手の注意に任せると、対象外の節が黙って本編に混ざる（実際に2件混ざった）。
        opt_sections = set(data.get("optional_sections", []))
        refs_set = set(c.get("syllabus_refs") or [])
        if opt_sections and refs_set:
            all_opt = refs_set <= opt_sections
            if all_opt and c.get("optional") is not True:
                out.append(Finding("ERROR", "OPTIONAL_DRIFT",
                                   f"{tag}: 出題対象外の節のみを参照しているのに optional が true ではありません"))
            if not (refs_set & opt_sections) and c.get("optional") is True:
                out.append(Finding("ERROR", "OPTIONAL_DRIFT",
                                   f"{tag}: 出題対象内の節を参照しているのに optional: true になっています"))
        if not isinstance(c.get("optional"), bool):
            out.append(Finding("ERROR", "OPTIONAL_TYPE", f"{tag}: optional は true / false"))

        area = c.get("area")
        if area not in VALID_AREAS:
            out.append(Finding("ERROR", "AREA", f"{tag}: area が不正: {area}"))
        else:
            per_area[area] += 1

        if c.get("tier") not in VALID_TIERS:
            out.append(Finding("ERROR", "TIER", f"{tag}: tier は A/B/C のいずれか"))
        if c.get("status") not in VALID_STATUS:
            out.append(Finding("ERROR", "STATUS", f"{tag}: status は todo/drafting/done のいずれか"))

        refs = c.get("syllabus_refs")
        if not isinstance(refs, list) or not refs:
            out.append(Finding("ERROR", "SYLLABUS_REFS",
                               f"{tag}: syllabus_refs が空です（どの項目に対応するか辿れません）"))
        else:
            for r in refs:
                if any(m in str(r) for m in TRANSCRIPT_MARKERS) or len(str(r)) > 40:
                    out.append(Finding("ERROR", "REFS_TRANSCRIPT",
                                       f"{tag}: syllabus_refs に本文らしき文字列があります: {str(r)[:30]}…"
                                       "（項目の識別子だけを入れてください）"))
                    break

        # status: done なのに本文が無い / 本文があるのに todo のまま
        exists = (CONTENT / f"{slug}.md").exists()
        if c.get("status") == "done" and not exists:
            out.append(Finding("ERROR", "STATUS_DRIFT", f"{tag}: status=done だが content/{slug}.md がありません"))
        if exists and c.get("status") == "todo":
            out.append(Finding("WARN", "STATUS_DRIFT", f"{tag}: 本文があるのに status=todo のままです"))

    for area, n in per_area.items():
        if n == 0:
            out.append(Finding(cover, "AREA_COVERAGE", f"区分「{area}」の概念が0件です（取りこぼしの疑い）"))

    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="概念台帳の機械検証")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--warn-only", action="store_true", help="★CI では絶対に付けない")
    args = ap.parse_args(argv)

    if not LEDGER.exists():
        print(f"[ERROR] LEDGER: {LEDGER.relative_to(ROOT)} がありません")
        return 1
    try:
        data = json.loads(LEDGER.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[ERROR] LEDGER: JSON が壊れています: {e}")
        return 1

    findings = check(data)
    errors = [f for f in findings if f.level == "ERROR"]
    warns = [f for f in findings if f.level == "WARN"]

    if args.json:
        print(json.dumps({"errors": [asdict(f) for f in errors],
                          "warns": [asdict(f) for f in warns]}, ensure_ascii=False, indent=2))
    else:
        for f in errors + warns:
            print(f"[{f.level}] {f.rule}: {f.message}")
        print(f"\n概念 {len(data.get('concepts', []))} 件 — ERROR {len(errors)} / WARN {len(warns)}")

    return 1 if (errors and not args.warn_only) else 0


if __name__ == "__main__":
    sys.exit(main())
