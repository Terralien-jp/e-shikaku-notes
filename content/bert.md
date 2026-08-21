---
exam: E資格
concept: BERT
slug: bert
tier: A
area: 深層学習
summary: Transformerエンコーダで左右の文脈を同時に扱う言語表現モデル。MLMとNSPで事前学習した重みを下流タスクへ移し、出力層を足してファインチューニングします。
updated: 2026-08-22
sources:
  - title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    url: https://arxiv.org/abs/1810.04805
  - title: "BERT model documentation"
    url: https://huggingface.co/docs/transformers/model_doc/bert
---

## ひとことで言うと

BERT（Bidirectional Encoder Representations from Transformers）は、Transformerの**エンコーダだけ**を多層に重ね、入力トークンの左側と右側の文脈を同時に使って表現を作るモデルです。ラベルのないテキストで事前学習し、その重みを下流タスクのモデル初期値にして、タスク用の出力層と一緒にファインチューニングします。

<div class="analogy">

文章を読むとき、左から右へ一度だけ読むのではなく、文の途中の空欄を前後から見比べて意味を決める読解係です。読解係そのものを毎回作り直さず、先に大量の文章で鍛え、分類や質問応答では最後の判定器だけを目的に合わせます。

</div>

## なぜ必要か

従来の左から右へ進む言語モデルでは、あるトークンが右側の情報を見られません。文の一部を分類するだけなら後段で補える場合もありますが、質問応答のように答えの位置を決める仕事では、前後の文脈を同時に使える表現が欲しくなります。

BERTはこの制約を、入力の一部を隠して元の語を当てるMasked Language Model（MLM）で回避します。隠された位置の正解を、左の文脈だけでなく右の文脈も含めて予測するため、深いエンコーダ全体が双方向の表現を学習できます。さらに、2文を入力したときに後続文かどうかを判定するNext Sentence Prediction（NSP）も使い、文ペアの表現を事前学習します。事前学習の手順や損失の細部ではなく、ここでは「2つの目的でエンコーダの重みを作る」という役割を押さえます。

<div class="caution">

「双方向」は、推論時に未来の文章を生成するという意味ではありません。入力として与えた列の中で左右を参照するという意味です。したがって、次の語を左から順に出力する生成器と、入力全体を読んでラベルを返すBERTを同じものとして扱わないでください。

</div>

## 仕組み

BERTの入力は、1文なら `[CLS]` で始まり `[SEP]` で終わります。2文なら `[CLS]` 文A `[SEP]` 文B `[SEP]` の1列にします。各位置の入力ベクトルは、次の3種類の埋め込みを足したものです。

| 埋め込み | 表すもの | 実装で対応する情報 |
|---|---|---|
| token embedding | そのトークンの語彙ID | `input_ids` |
| segment embedding | 文Aか文Bか | `token_type_ids` |
| position embedding | 列の何番目か | 位置ID |

位置 $i$ の入力を $\mathbf{x}_i$、トークン・セグメント・位置の埋め込みをそれぞれ $\mathbf{e}^{\text{tok}}_i$、$\mathbf{e}^{\text{seg}}_i$、$\mathbf{e}^{\text{pos}}_i$ とすると、

$$
\mathbf{x}_i = \mathbf{e}^{\text{tok}}_i + \mathbf{e}^{\text{seg}}_i + \mathbf{e}^{\text{pos}}_i
$$

です。$\mathbf{x}_i$ は位置 $i$ の入力ベクトル、3つの $\mathbf{e}$ はそれぞれ対応する埋め込みベクトルを意味します。`[CLS]` の最終隠れ状態は文全体の分類などに、各トークンの最終隠れ状態は系列ラベリングや質問応答の位置予測に使えます。`[SEP]` は文の境界を表し、`token_type_ids` は2つの文の区別をモデルへ渡します。パディングを入れる場合は `attention_mask` で無効な位置を示します。

構成は、入力表現をTransformerエンコーダへ通す流れです。エンコーダは自己注意によって各トークンから同じ列の他のトークンを参照します。BERTではGPTのような左側だけの制約を置かないため、同じ層で左右の文脈を結合できます。事前学習後は、共通のエンコーダにタスク固有の出力層を接続し、下流データで**全パラメータ**を調整します。これが「特徴量を固定して使う」だけでなく、タスクに合わせて表現自体も変えるファインチューニングです。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| BERTの構成を説明する | Transformerエンコーダのみを使う多層の双方向自己注意 | デコーダを含む生成モデルとする |
| 双方向性の理由を問う | MLMで隠した語を左右の文脈から予測する | 左から右へ次語を予測するだけとする |
| 入力表現を対応付ける | token・segment・positionの3埋め込みを加算する | `attention_mask` を埋め込みの1種とする |
| 特殊トークンの役割を問う | `[CLS]` は先頭、`[SEP]` は文境界。文全体の判定には `[CLS]` 表現を使う | `[CLS]` が文を区切る、`[SEP]` が分類結果を表す |
| 利用手順を問う | MLMとNSPで事前学習→タスク出力層を追加→全体を微調整 | 下流タスクごとにゼロから学習する |
| GPTとの違いを問う | BERTは双方向エンコーダ、GPTは左側だけを見る一方向型 | BERTも次トークン生成を主目的とする |

## 実装で確かめる

`input_ids`、`token_type_ids`、`position_ids` に対応する3つの表を足すだけなら、NumPyで入力表現の形を確認できます。ここでは学習済み重みやトークナイザを仮定せず、埋め込みの合成だけを切り出します。

```python
import numpy as np

vocab, segments, length, hidden = 8, 2, 5, 4
token_table = np.arange(vocab * hidden, dtype=float).reshape(vocab, hidden)
segment_table = np.arange(segments * hidden, dtype=float).reshape(segments, hidden)
position_table = np.arange(length * hidden, dtype=float).reshape(length, hidden)
input_ids = np.array([0, 3, 5, 1, 2])
token_type_ids = np.array([0, 0, 0, 1, 1])
position_ids = np.arange(length)
x = token_table[input_ids] + segment_table[token_type_ids] + position_table[position_ids]
assert x.shape == (length, hidden)
print(x.shape)
```

`token_type_ids` を全て0にすれば単一文、途中から1にすれば文Aと文Bを同じ列に置く入力になります。実際のライブラリではトークナイザが特殊トークンやマスクを整え、モデル側がこの入力表現をエンコーダへ渡します。

## 取り違えやすいもの

| 用語 | BERTとの切り分け |
|---|---|
| [Transformerエンコーダ](/learn/e-shikaku/transformer/) | BERTが採用する本体。入力列全体を双方向に自己注意へかける |
| [Transformerデコーダ](/learn/e-shikaku/transformer/) | 左側の文脈だけを見る制約を置けば、次トークン生成に使える構成 |
| [GPT](/learn/e-shikaku/gpt-and-rag/) | 原論文で比較される一方向型のTransformer。各トークンが左側だけを参照する |
| MLM | BERTの事前学習タスク。モデル名や推論時の分類ヘッドではない |
| NSP | 2文の関係を学習するもう一つの事前学習タスク。`[SEP]` そのものと同じではない |
| ファインチューニング | 事前学習済み重みを初期値に、下流タスクの教師データで全体を調整する段階 |

## 想起チェック

<details class="recall">
<summary>BERTはTransformerのどの部分を使い、GPTとどこが異なるか</summary>

BERTはエンコーダのみで、自己注意が左右の文脈を参照します。GPTは左側だけを参照する一方向型です。

</details>

<details class="recall">
<summary>BERTの入力表現を作る3種類の埋め込みは何か</summary>

トークン、セグメント、位置の埋め込みを加算します。`attention_mask` はパディングなどを注意の対象外にする情報で、3種の埋め込みには含めません。

</details>

<details class="recall">
<summary>MLMとNSPはそれぞれ何を学習する目的か</summary>

MLMは隠したトークンの復元、NSPは2つの文が後続関係にあるかの判定です。前者が双方向の文脈、後者が文ペアの表現に関係します。

</details>

<details class="recall">
<summary>事前学習済みBERTを下流タスクで使う手順は何か</summary>

事前学習済みパラメータで初期化し、タスク用の出力層を接続して、ラベル付きデータで全パラメータをファインチューニングします。

</details>
