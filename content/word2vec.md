---
exam: E資格
concept: Word2vec
slug: word2vec
tier: B
area: 深層学習
summary: 文脈から語を予測する学習を通じて単語ベクトルを得る手法群です。CBOWとSkip-gram、負例サンプリングの役割と、語順を捨てる制約を整理します。
updated: 2026-08-22
sources:
  - title: "Efficient Estimation of Word Representations in Vector Space"
    url: https://arxiv.org/abs/1301.3781
  - title: "Distributed Representations of Words and Phrases and their Compositionality"
    url: https://arxiv.org/abs/1310.4546
---

## ひとことで言うと

Word2vecは、単語そのものを説明するのではなく、ある語から同じ文脈に現れる語を予測する学習からベクトルを得るモデル群です。予測を当てる圧力で、似た文脈を持つ語の表現が似ます。原論文は、この表現が統語的・意味的な単語関係の評価で有効だったと報告しています。

<div class="analogy">

単語の意味を辞書から転記するのではなく、「この語の近くには何が来るか」を大量に採点して、似た出題傾向の語を同じ棚へ並べる仕組みです。

</div>

## なぜ必要か

語彙全体に対するsoftmaxは、1つの正解語の確率を出すだけでも全語彙を比較します。語彙数を $V$ とすると、出力側の計算は $V$ に比例し、大規模コーパスでは学習のボトルネックになります。Word2vecはモデルの予測構造を単純化し、さらに出力の学習方法を工夫して、精度と計算量の両方を扱います。

|工夫|軽くするもの|実装で見る点|
|---|---|---|
|階層的softmax|二分木の経路だけを計算|全語彙ではなく経路上の二値分類|
|負例サンプリング|正例と少数の負例の分類だけを計算|負例数 $k$ を固定して更新|
|頻出語のsubsampling|冗長な訓練例と頻出語の偏り|入力語を確率的に捨てる|

## 仕組み

中心語を $w_t$、窓内の周辺語を $w_{t+j}$ とします。CBOWは周辺語から中心語を予測し、Skip-gramは中心語から周辺語を予測します。CBOWでは周辺語の位置を平均するため、窓内の語順は入力に残りません。Skip-gramも中心語と周辺語の組を作るだけなので、文全体の順序をモデル化するものではありません。

$$
\mathcal{L}_{\mathrm{SG}}=\frac{1}{T}\sum_{t=1}^{T}\sum_{\substack{-c\le j\le c\\j\ne0}}\log p(w_{t+j}\mid w_t)
$$

$T$ は訓練語数、$c$ は文脈窓の大きさ、$w_t$ は中心語です。Skip-gramは1つの中心語から窓内の複数語を予測するので、窓を広げるほど訓練対が増え、計算量も増えます。CBOWは複数の周辺語をまとめて中心語1語を予測するため、同じデータでも更新の向きと例のまとめ方が異なります。

負例サンプリングでは、正しい組 $(w_t,w_{t+j})$ を正例、ノイズ分布 $P_n$ から引いた語との組を負例にします。入力ベクトルを $\mathbf{v}_{w_t}$、出力ベクトルを $\mathbf{u}_w$、負例数を $k$ とすると、1組の目的は次です。

$$
\log\sigma(\mathbf{u}_{w_{t+j}}^{\mathsf T}\mathbf{v}_{w_t})+\sum_{i=1}^{k}\mathbb{E}_{w_i\sim P_n}\left[\log\sigma(-\mathbf{u}_{w_i}^{\mathsf T}\mathbf{v}_{w_t})\right]
$$

$\sigma$ はsigmoid関数、$P_n$ は負例を引く分布です。全語彙の正規化をせず、正例1つと負例 $k$ 個だけを分類するので、出力更新の規模を $V$ からおおむね $k$ に置き換えられます。ここでの高速化は、語順を理解する機能を追加するものではありません。

## 試験でどう問われるか

|問われ方|正解に寄る条件|引っかけ|
|---|---|---|
|2方式の対応|CBOWは周辺語→中心語、Skip-gramは中心語→周辺語|名前を入力ベクトルの種類で区別する|
|負例サンプリングの目的|全語彙softmaxを避け、少数の負例と正例を分類する|負例を増やすほど必ず高速になる|
|窓幅の影響|Skip-gramでは訓練対と計算量が増える|窓幅は精度だけを変え、計算量は不変|
|表現の限界|局所的な共起予測であり、語順や慣用句をそのまま表さない|得られたベクトルが文脈依存表現になる|

## 実装で確かめる

次の最小例は、Skip-gramの正例1つと負例2つをsigmoidで更新します。出力語彙全体ではなく、指定した3語だけを使う点が負例サンプリングです。

```python
import numpy as np
rng = np.random.default_rng(0)
v = rng.normal(0, .1, 2)                 # 中心語の入力ベクトル
u = rng.normal(0, .1, (3, 2))            # 出力語ベクトル
positive, negatives, lr = 1, [0, 2], .1
for _ in range(20):
    ids, labels = [positive] + negatives, [1, 0, 0]
    old = u[ids].copy()
    scores = old @ v
    p = 1 / (1 + np.exp(-scores))
    g = p - labels
    u[ids] -= lr * g[:, None] * v
    v -= lr * (g[:, None] * old).sum(axis=0)
print(np.round(u @ v, 3))
```

実行すると `[-0.069  0.06  -0.102]` となります。正例の内積が上がり、負例の内積が下がる方向へ更新されます。実際の実装では、負例の選択、窓の切り出し、入力側と出力側のどの行列を保存するかを分けて確認します。

## 取り違えやすいもの

|用語|Word2vecとの切り分け|
|---|---|
|CBOW|周辺語をまとめて中心語を当てるアーキテクチャ|
|Skip-gram|中心語から窓内の周辺語を当てるアーキテクチャ|
|階層的softmax|二分木の経路をたどって確率を計算する高速化|
|負例サンプリング|正例と少数のノイズ語を二値分類する高速化|
|言語モデル|語順に沿う次語確率を扱う枠組み。Word2vecの目的は窓内予測で、同じものではない|

## 想起チェック

<details class="recall">
<summary>CBOWとSkip-gramの予測方向は</summary>

CBOWは周辺語から中心語、Skip-gramは中心語から周辺語です。

</details>

<details class="recall">
<summary>負例サンプリングで全語彙softmaxを避けられる理由は</summary>

正例1つと負例 $k$ 個の二値分類だけを計算するためです。$V$ 個すべての出力を正規化しません。

</details>

<details class="recall">
<summary>Word2vecが苦手とする情報は</summary>

CBOWの平均やSkip-gramの窓内組によって、文全体の語順を直接表しません。原論文は、語順への無関心と慣用句の表現にくさを限界として挙げています。

</details>
