# 誰がこのノートを書くか（役割分担）

**このリポジトリの執筆は Luna（`gpt-5.6-luna` / Codex CLI）が主担当です。Claude は編集長側であって、既定の書き手ではありません。**

この1枚が無かったせいで、2026-08-22 に Claude のサブエージェント8体で Tier A を8本書き、
Claude 枠から約 886k トークンを使う取り違えが起きました。ブリーフ（[writer-brief.md](writer-brief.md)）は
「どう書くか」しか書いておらず、「誰が書くか」がリポジトリのどこにも無かったのが原因です。

## 分担

| 役 | 誰 | 仕事 |
|---|---|---|
| **執筆（初稿）** | **Luna**（`gpt-5.6-luna`・Codex CLI） | `content/<slug>.md` を書く。Phase0 の台帳・Phase2 の出典カードも Luna が作った |
| **編集長** | Claude Code | ブリーフを書く／担当範囲を切る／`validate.py` を通す／抜き取り監査／台帳更新／PR とマージ |
| **外部監査** | agy（Antigravity CLI・`~/.local/bin/agy`） | 公開前に読者目線で読む。リポジトリの外から見る役 |
| **人間** | 本人 | 不可逆な判断（公開・方針変更）と、機械では見られない事実の当否 |

**理由はコストです。** Luna は $0.20/$1.20 per 1M、Claude Sonnet 5 は $3.00/$15.00。
このノートは112概念ぶん書く前提なので、初稿を Claude で回すと Claude 枠が先に尽きます。
品質は「Luna が書いて機械検証＋Claude の抜き取り＋agy の外部監査」で担保する設計です。

## Luna の呼び方（この機での実際）

`codex` は PATH に無いことがあります。ChatGPT.app に同梱されているものが実体です。

```bash
CODEX=$(command -v codex || echo /Applications/ChatGPT.app/Contents/Resources/codex)
"$CODEX" exec --model gpt-5.6-luna --skip-git-repo-check \
  --sandbox workspace-write -c sandbox_workspace_write.network_access=true \
  "<執筆ブリーフ本文>"
```

- **Claude へのフェイルオーバーを付けない。** 外れたときに両方の枠を食い、配線ミスが見えなくなる（Issue #1396 の教訓）
- ブラウザ系ツールは承認が要るため自動拒否される。**一次情報の取得は `curl`**（[phase2-sources-brief.md](phase2-sources-brief.md) と同じ）
- 執筆ブリーフには必ず「対象 slug・台帳の登録値・出典カードの中身・担当範囲（隣接ノートと越境しない）・`validate.py` を ERROR 0 にすること」を入れる

## Claude を書き手に使ってよい場合

- **見本を作るとき**（`content/backpropagation.md` はこれ）
- Luna が2回続けて機械検証を通せなかった概念の引き取り
- ユーザーが明示的に指示したとき

いずれも例外です。**既定は Luna。**
