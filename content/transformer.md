---
exam: E資格
concept: Transformer
slug: transformer
tier: A
area: 深層学習
summary: 再帰も畳み込みも使わず、注意機構で系列内の位置同士を直接結び付けるエンコーダ・デコーダ型のモデルです。
updated: 2026-08-22
sources:
  - title: "Attention Is All You Need"
    url: https://arxiv.org/abs/1706.03762
  - title: "Deep Learning: Sequence Modeling"
    url: https://www.deeplearningbook.org/contents/rnn.html
---

## ひとことで言うと

Transformerは、系列を1ステップずつ処理する再帰や局所窓を動かす畳み込みを使わず、注意機構で系列中の位置同士を直接参照するモデルです。入力系列を表現するエンコーダと、過去の出力から次の記号を生成するデコーダを積み重ねます。位置情報は注意だけでは生まれないため、埋め込みに位置エンコーディングを加えます。

<div class="analogy">

会議の議事録を作るとき、発言を順番に一人ずつ聞いて要約するのが再帰モデルです。Transformerは全員の発言を一覧に並べ、ある発言を理解するたびに「どの発言をどれだけ参照するか」を同時に決めます。ただし、一覧だけでは発言順が消えるので、各発言に座席番号を添えます。

</div>

## なぜ必要か

系列モデルでは、離れた位置の依存関係をどう伝えるかが難所です。RNNは前の状態を次へ渡すので、位置をまたぐ計算が直列になり、遠い位置ほど情報の経路が長くなります。畳み込みは並列化できますが、狭いカーネルで系列全体を結ぶには層を重ねます。

Transformerの自己注意なら、1つの層で全位置の組を比較できます。原論文は、再帰・畳み込みを完全に外した構成により、より多くの計算を並列化でき、機械翻訳で高い品質を得たと報告しています。これは「どんな系列でも常に高速」という意味ではありません。注意の比較対象が全位置なので、系列長が増えると別のコストが現れます。

| 系列の扱い方 | 遠い位置への経路 | 並列化とコスト |
|---|---|---|
| RNN | 時刻を順に渡す | 直列計算が必要 |
| 畳み込み | 局所窓を層で広げる | 位置は並列だが層を要する |
| 自己注意 | 1層で全位置を直接接続 | 並列化しやすいが $O(n^2d)$ |

## 仕組み

入力を埋め込み、位置エンコーディングを加えた行列をエンコーダへ入れます。原論文の基本構成では、エンコーダは同じ層を6個積み、各層を **Multi-Head Self-Attention → 残差接続とLayerNorm → 位置ごとのFFN → 残差接続とLayerNorm** とします。デコーダも6層で、これに「エンコーダ出力を参照するMulti-Head Attention」が加わります。各サブ層の出力は概念的に $\mathrm{LayerNorm}(x+\mathrm{Sublayer}(x))$ です。

Scaled Dot-Product Attentionは、クエリ行列 $Q$、キー行列 $K$、値行列 $V$ から次を計算します。

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\mathsf{T}}{\sqrt{d_k}}\right)V
$$

$d_k$ はキーとクエリのベクトル次元です。$QK^\mathsf{T}$ は各クエリと各キーの相性を表し、softmaxの重みで $V$ を加重平均します。$d_k$ が大きいと、成分の分散が1程度のとき内積の分散はおおむね $d_k$ まで大きくなります。そのままsoftmaxへ入れると値が極端になり、飽和して勾配が小さくなるため、$\sqrt{d_k}$ で割って尺度を戻します。

Multi-Headは、$Q,K,V$ を異なる学習済み行列で複数の低次元空間へ射影し、各ヘッドで注意を計算して結合し、最後に再射影する仕組みです。1つの平均的な注意だけでは混ざる関係を、異なる位置や表現部分空間から同時に拾えます。原論文の基本モデルは8ヘッドで、各ヘッドの次元を小さくして全体の計算量を単一ヘッドと同程度にしています。

FFNは位置ごとに独立して同じ関数を適用します。

$$
\mathrm{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2
$$

$W_1,W_2$ は線形変換の重み、$b_1,b_2$ はバイアス、$\max(0,\cdot)$ はReLUです。注意が位置間を混ぜ、FFNが各位置を変換する、という役割分担です。

位置 $pos$、次元番号 $i$、モデル次元 $d_{model}$ に対して、原論文の固定方式は次です。

$$
PE(pos,2i)=\sin\left(pos/10000^{2i/d_{model}}\right),\quad
PE(pos,2i+1)=\cos\left(pos/10000^{2i/d_{model}}\right)
$$

再帰も畳み込みもないため、同じトークン集合を並べ替えても注意だけでは順序を区別できません。そこで埋め込みと同じ次元の位置表現を足します。デコーダの自己注意では、現在位置より後ろを見られないようsoftmax前のスコアをマスクします。これがないと、学習時に正解の未来トークンを見てしまい、自己回帰生成との条件がずれます。

自己注意は系列長を $n$、表現次元を $d$ とすると、全位置対のスコア行列を作るため1層あたり $O(n^2d)$ です。一方、全位置を結ぶ経路長は一定で、RNNのような $O(n)$ の直列ステップは要りません。長い系列では、並列性の利点と二乗のメモリ・計算コストを同時に見ます。

## 試験でどう問われるか

| 問われ方 | 正解に寄る条件 | 引っかけ |
|---|---|---|
| Transformerの構成 | 再帰・畳み込みを使わず、注意と位置ごとのFFNを積む | 「位置情報は注意が自動で持つ」 |
| Attentionの式 | $QK^\mathsf{T}$ を $\sqrt{d_k}$ で割ってsoftmax後に $V$ を掛ける | $V$ にsoftmaxを掛ける／スケールを掛けない |
| $1/\sqrt{d_k}$ の理由 | 内積の分散増大を抑え、softmaxの飽和を避ける | 単に出力を正規化するため、とする |
| Multi-Headの意味 | 異なる射影空間・位置関係を並列に捉える | ヘッドごとに別モデルを学習して出力を足す |
| デコーダのマスク | 位置 $i$ は既知の過去と現在までを参照し、未来を参照しない | エンコーダ側の全位置参照まで禁止する |
| 計算量 | 全位置対のため系列長に対して二乗になる | 並列化できるから系列長に比例する |

## 実装で確かめる

下の最小実装は、デコーダの未来位置をマスクしたScaled Dot-Product Attentionです。行列の最後の次元を特徴次元、中央の次元を系列位置として扱います。

```python
import numpy as np

rng = np.random.default_rng(0)
q = rng.normal(size=(1, 3, 4))
k = rng.normal(size=(1, 3, 4))
v = rng.normal(size=(1, 3, 2))
scores = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1])
future = np.triu(np.ones((3, 3), dtype=bool), 1)
scores = np.where(future[None, :, :], -np.inf, scores)
weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
weights /= weights.sum(axis=-1, keepdims=True)
out = weights @ v
assert out.shape == (1, 3, 2)
assert np.allclose(weights[0, 0, 1:], 0.0)
```

`future` の上三角を $-\infty$ にしてからsoftmaxするので、1番目の位置の重みは自分より後ろへ流れません。スケールをsoftmaxの後に置くと、相性スコアの温度を調整できず、式の意味が変わります。

## 取り違えやすいもの

| 用語 | Transformerとの切り分け |
|---|---|
| Self-Attention | 同じ系列から $Q,K,V$ を作る注意。TransformerはこれをMulti-Head化し、ブロックとして積む |
| Encoder-Decoder Attention | デコーダのクエリとエンコーダ出力のキー・値を結ぶ注意。Self-Attentionとは入力元が違う |
| RNN | 隠れ状態を時系列に渡すため直列計算になる。Transformerは注意で全位置を直接結ぶ |
| 畳み込み | 局所的な受容野を持つ。TransformerのFFNは位置ごとの全結合で、系列位置間の混合は注意が担う |
| 位置エンコーディング | トークンの順序を補う入力情報。Attentionそのものやマスクとは別の機構 |

## 想起チェック

<details class="recall">
<summary>Transformerが再帰なしで系列の順序を扱えるようにするものは何か</summary>

埋め込みに加える位置エンコーディングです。注意だけではトークンの並び順を区別できません。

</details>

<details class="recall">
<summary>Scaled Dot-Product Attentionで内積を平方根で割る理由は何か</summary>

キー次元 $d_k$ が大きいと内積の分散が増え、softmaxが飽和して勾配が小さくなりやすいためです。$\sqrt{d_k}$ でスコアの尺度を抑えます。

</details>

<details class="recall">
<summary>デコーダの未来位置マスクは何を防ぐか</summary>

位置 $i$ の予測が、まだ生成していない位置 $i+1$ 以降の正解を参照することを防ぎます。自己回帰生成と同じ情報条件にそろえるためです。

</details>

<details class="recall">
<summary>系列長を2倍にしたとき、自己注意の主なコストはどうなるか</summary>

全位置対のスコア行列を作るため、系列長に関する項はおおむね4倍になります。並列化できることと、二乗コストであることは両立します。

</details>
