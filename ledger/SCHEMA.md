# 概念台帳のスキーマ

`concepts.json` が**書くべきものの唯一の真**です。ここに無い slug のノートは検証で落ちます（一覧に出ず迷子になるため）。

| フィールド | 必須 | 内容 |
|---|---|---|
| `slug` | ✅ | ファイル名と完全一致（`content/<slug>.md`）。英小文字・数字・ハイフンのみ |
| `concept` | ✅ | 表示名（日本語）。例: 誤差逆伝播法 |
| `area` | ✅ | `応用数学` / `機械学習` / `深層学習` / `開発・運用環境` のいずれか |
| `tier` | ✅ | `A`（厚く書く） / `B`（標準） / `C`（短くてよい） |
| `syllabus_refs` | ✅ | 対応するシラバス項目の識別子。**公式の文言は入れない**（項目を指すIDのみ） |
| `status` | ✅ | `todo` / `drafting` / `done` |

## トップレベル

| フィールド | 内容 |
|---|---|
| `exam` | `E資格` 固定 |
| `ledger_status` | `bootstrap`（作成途中・網羅性の検査を WARN に落とす） / `complete` |
| `syllabus_version` | 対象シラバスの版。例 `E2026#2` |
| `syllabus_source` | 参照した JDLA 公式ページのURL |
| `areas` | 4区分固定。増やさない |

検証: `python3 scripts/validate_ledger.py`（fail-closed）。作り方は [`docs/phase0-ledger-brief.md`](../docs/phase0-ledger-brief.md)。

## Tier の決め方

出題比重ではなく**「間違えたときに他へ波及するか」**で決めます。誤差逆伝播を取り違えると
最適化も正則化も崩れるので A。特定ライブラリの引数仕様は単独で閉じるので C。

## シラバス項目の扱い

**JDLA の公式文言は台帳にも本文にも転記しません。** 参照するのは項目の識別子と、
自分の言葉で書いた `concept` 名だけです（`docs/writer-brief.md` の出典ルールと同じ理由）。
