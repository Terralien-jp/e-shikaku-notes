#!/usr/bin/env python3
"""執筆エージェント（Luna）へ渡すブリーフを作る。

台帳から slug / concept / area / tier を引き、規約と機械ゲートの注意を1枚にまとめる。
**「狙い」と「担当範囲」は編集判断なので、呼び出し側が渡す。** ここで埋められるのは
機械が知っていること（登録値・分量の下限・ゲートの名前）だけ。

  python3 scripts/make_brief.py <slug> --aim "..." --scope "..." > brief.txt

なぜ台帳から引くか: frontmatter の値が台帳とずれると validate.py の FRONTMATTER が
ERROR になる。手で書き写すとそこで落ちる。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FLOOR = {"A": "1,800〜2,800字", "B": "900〜1,500字", "C": "400〜700字"}

TEMPLATE = """あなたは E資格 学び直しノートの執筆者です。成果物は content/{slug}.md 1本だけです。

作業ディレクトリ（カレント）がリポジトリです。git コマンドは打たないでください。

## 最初に必ず読むもの（読まずに書き始めない）
1. docs/writer-brief.md — ライター規約の全文。H2の名前と順序・全H2に視覚要素・KaTeX記法・
   囲み（analogy/caution/recall）の中は開きタグ直後と閉じタグ直前に空行・summary の中に数式を書かない
2. content/backpropagation.md — 見本
3. docs/agent-roles.md — この仕事の分担

## 分量（機械で検査されます）
Tier {tier} は {length}。**表・コード・数式ブロックを除いた地の文で下限を割ると PROSE_SHORT が ERROR になります。**
表を足して字数を稼ぐことはできません。段落の説明を厚くしてください。**上限に収めるために説明を削らないこと。**

## 数式（実害が出た点）
★**LaTeX コマンドのバックスラッシュを落とさないこと。** \\mathbf{{x}} を mathbf{{x}} と書くと KaTeX は
素通しし、読者の画面に mathbf{{x}} と表示されます。**シェルのヒアドキュメント経由で書くと脱落・
タブ化（\\theta → タブ文字）しやすい**ので、書き終えたらファイルを読み直して確認してください。
validate.py の MATH_ESCAPE が検査します。

## 今回の対象（台帳の登録値。frontmatter にそのまま書く）
- slug: {slug}
- concept: {concept}
- area: {area}
- tier: {tier}
- updated: {updated}

## 出典
ledger/sources/{slug}.json に決まっています。**必ず中身を読み、URL を実際に取得して読んでから書く。**
記憶で書かない。原論文はその論文が実際に主張していることだけを書く（数値・提案年・結論を盛らない）。
ブラウザ系ツールは使えないので curl で取得すること。**取得できなかった URL は出典に書かない。**
カードに無い事実を本文に書かない。同じ論文の abs と html を2件並べるのは出典を足したことになりません。

## 内容の狙い
読者は「一度 E資格を取ったが手が動かなくなった中堅エンジニア」。用語紹介ではなく、試験と実装で迷う場所を潰す。
{aim}

## 担当範囲（越境しない）
{scope}
**隣接ノートと同じ説明を繰り返さないこと。** 他ノートへの内部リンクは書かない（未作成ページへのリンクは ERROR）。

## 禁止
- 過去問・想定問題の再現。「## 試験でどう問われるか」は問われ方の型を書く場所
- JDLA シラバス本文の転記
- ledger/ の編集、git 操作
- 説教で締めること

## 書き終えたら
/Users/master/.pyenv/shims/python scripts/validate.py content/{slug}.md --check-code
ERROR が0になるまで直す（PROSE_SHORT / MATH_ESCAPE / TABLE_COLUMNS を含む）。
表は見出し・区切り・本体の列数を揃えること。コードは実際に実行される（依存は numpy まで。
PyTorch を使うなら python no-run のフラグを付ける）。出力値を書くなら実際に実行した値を貼ること。
WARN の MATH_SKIP は無視してよい。

## 最後に報告すること
slug / 地の文字数 / validate の ERROR・WARN 件数 / 実際に読んだ出典URL / 判断に迷って選んだ点。
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="執筆ブリーフを台帳から組み立てる")
    ap.add_argument("slug")
    ap.add_argument("--aim", required=True, help="この記事で潰す『迷う場所』（編集判断）")
    ap.add_argument("--scope", required=True, help="担当範囲と、隣接ノートとの切り分け（編集判断）")
    ap.add_argument("--updated", default="", help="frontmatter の updated（既定は台帳の syllabus 日付ではなく当日を呼び出し側が渡す）")
    args = ap.parse_args()

    ledger = json.loads((ROOT / "ledger" / "concepts.json").read_text(encoding="utf-8"))
    hit = next((c for c in ledger["concepts"] if c["slug"] == args.slug), None)
    if hit is None:
        print(f"台帳に {args.slug} がありません", file=sys.stderr)
        return 1
    card = ROOT / "ledger" / "sources" / f"{args.slug}.json"
    if not card.exists():
        print(f"出典カードがありません: {card}（Phase 2 を先に回す）", file=sys.stderr)
        return 1

    print(TEMPLATE.format(slug=hit["slug"], concept=hit["concept"], area=hit["area"],
                          tier=hit["tier"], length=FLOOR.get(hit["tier"], "台帳の tier を確認"),
                          updated=args.updated or "（当日の日付）",
                          aim=args.aim.strip(), scope=args.scope.strip()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
