---
exam: E資格
concept: 単語埋め込み
slug: word-embedding
tier: A
area: 深層学習
summary: 単語を共起する文脈から低次元の連続ベクトルへ写像し、Skip-gram や CBOW で単語間の近さを計算可能にする方法。
updated: 2026-08-22
sources:
  - title: "Efficient Estimation of Word Representations in Vector Space"
    url: https://arxiv.org/abs/1301.3781
  - title: "Deep Learning: Representation Learning"
    url: https://www.deeplearningbook.org/contents/representation.html
---

## ひとことで言うと

単語埋め込みは、語彙の各単語を、学習可能な低次元の実数ベクトルへ写像する方法です。単語IDをそのまま特徴量にするのではなく、似た文脈に現れる単語がベクトル空間でも近くなるように、予測タスクを通じて表現を学習します。

<div class="analogy">

単語を社員証の番号だけで管理するのが one-hot 表現です。番号同士には「部署が近い」「担当が似ている」という関係がありません。埋め込みは、業務上の関係を反映した座標に社員を置き直す作業です。

</div>

## なぜ必要か

語彙数を $V$ とすると、one-hot ベクトルは $V$ 次元で、1箇所だけが1、それ以外が0です。単語を区別するには十分ですが、異なる単語の内積は常に0なので、語彙中の2語がどれだけ似ているかを表せません。語彙が増えるほど入力も疎で高次元になります。

埋め込み行列を $E in \mathbb{R}^{V\times d}$、埋め込み次元を $d$ とすると、単語 $w$ のベクトルは one-hot ベクトル $\mathbf{x}_w$ に対して $\mathbf{e}_w=E^{\mathsf T}\mathbf{x}_w$ と取り出せます。通常は $d\ll V$ です。学習後は cosine 類似度

$$
\operatorname{cos}(\mathbf{e}_i,\mathbf{e}_j)=\frac{\mathbf{e}_i^{\mathsf T}\mathbf{e}_j}{\lVert\mathbf{e}_i\rVert\lVert\mathbf{e}_j\rVert}
$$

（$\mathbf{e}_i,\mathbf{e}_j$ は2単語の埋め込み）で近さを比較できます。これは辞書の定義を直接教えた結果ではなく、共起する文脈を予測する圧力から生じる近さです。

## 仕組み

文中の中心語を $w_t$、その周辺の語を $w_{t+j}$ とします。Skip-gram は中心語から周辺語を予測し、CBOW（Continuous Bag-of-Words）は周辺語から中心語を予測します。

|方式|入力|予測するもの|文脈の順序|
|---|---|---|---|
|Skip-gram|中心語 $w_t$|周辺語 $w_{t+j}$|周辺語ごとに扱う|
|CBOW|周辺語のベクトル|中心語 $w_t$|平均するため使わない|

出力側のベクトルを $\mathbf{u}_w$ とし、Skip-gram の基本的な確率を

$$
p(w_o\mid w_i)=\frac{\exp(\mathbf{u}_{w_o}^{\mathsf T}\mathbf{e}_{w_i})}{\sum_{w=1}^{V}\exp(\mathbf{u}_{w}^{\mathsf T}\mathbf{e}_{w_i})}
$$

とします。$w_i$ は入力語、$w_o$ は正解の出力語、$V$ は語彙数です。正解の周辺語の対数確率を最大化すると、同じ文脈で予測される語の内積が上がり、共起のパターンが埋め込みに分散して入ります。CBOW では入力側の周辺語ベクトルを平均して $\mathbf{e}_{\mathrm{ctx}}$ とし、同じ形式で中心語を予測します。

この式の分母は全語彙の計算を要求するため、1例ごとのコストが $V$ に依存します。階層的 softmax は語彙を二分木に置き、対象語までの経路上の二値分類だけを計算します。経路長はおおむね $\log_2 V$ で、原論文では頻度の高い語に短い符号を割り当てる Huffman 木を使っています。負例サンプリングは、正例と少数の負例を二値分類することで全語彙の正規化を避ける別の工夫です。したがって、どちらも「埋め込みの意味を定義する別手法」ではなく、出力確率の学習コストを下げる方法です。

<div class="caution">

Skip-gram と CBOW の違いを「入力層が one-hot か dense か」で覚えると混乱します。両方とも単語を埋め込みへ写像します。違うのは予測の向きで、Skip-gram は中心語 $→$ 周辺語、CBOW は周辺語 $→$ 中心語です。

</div>

## 試験でどう問われるか

|問われ方|正解に寄る条件|引っかけ|
|---|---|---|
|one-hot と分散表現の比較|one-hot は語の同一性、分散表現は低次元の連続値で関係を表す|one-hot の距離だけで意味の近さを測れる|
|Skip-gram と CBOW の対応|中心語から周辺語が Skip-gram、周辺語から中心語が CBOW|名称を入力側の表現形式で区別する|
|学習後の類似度|ベクトルの cosine 類似度などで比較する|単語IDの差や one-hot の内積を使う|
|高速化の選択|階層的 softmax は木の経路、負例サンプリングは少数の負例を使う|どちらも語彙全体の softmax を毎回計算する|
|論文の主張の範囲|共起から得たベクトルが統語・意味の類似度評価で有効だった|埋め込みが単語の意味を完全に定義すると言い切る|

## 実装で確かめる

小さな Skip-gram を NumPy だけで学習します。`pairs` は同じ文の中心語と周辺語の組で、更新後に中心語 `0` と `1` の cosine 類似度を表示します。

```python
import numpy as np
rng = np.random.default_rng(0)
pairs, V, d = [(0, 2), (1, 2), (2, 0), (2, 1)] * 20, 3, 2
E, U = rng.normal(0, .2, (V, d)), rng.normal(0, .2, (V, d))
for _ in range(100):
    for i, o in pairs:
        s = U @ E[i]; p = np.exp(s - s.max()); p /= p.sum()
        g = p.copy(); g[o] -= 1
        old = U.copy()
        U -= .02 * g[:, None] * E[i]
        E[i] -= .02 * (g @ old)
sim = E[0] @ E[1] / (np.linalg.norm(E[0]) * np.linalg.norm(E[1]))
print(f"cosine(0, 1)={sim:.6f}")
```

実行結果は `cosine(0, 1)=0.999074` です。語 `0` と `1` が同じ文脈語 `2` を予測する対を繰り返したため、入力側のベクトルが近づきました。実データでは、窓幅、頻度、負例の選び方で得られる空間が変わります。

## 取り違えやすいもの

|用語|単語埋め込みとの違い|
|---|---|
|[one-hot 表現](/learn/e-shikaku/softmax-and-onehot/)|語彙の各語に固有の座標を割り当てる疎表現。類似度を学習していない|
|分散表現|複数の次元に情報を分散して持たせる考え方。word2vec はその学習方法の一つ|
|[CBOW](/learn/e-shikaku/word2vec/)|周辺語から中心語を予測するモデル。埋め込みそのものの別名ではない|
|[Skip-gram](/learn/e-shikaku/word2vec/)|中心語から周辺語を予測するモデル。入力と出力のどちらの行列を使うかも実装で確認する|
|[文脈依存表現](/learn/e-shikaku/bert/)|同じ単語でも文脈でベクトルが変わる。word2vec 系の静的な埋め込みとは扱う情報が違う|

## 想起チェック

<details class="recall">
<summary>one-hot では単語間の意味的な近さを表しにくい理由は</summary>

異なる単語の one-hot ベクトルの内積が0になるためです。語彙数 $V$ に比例する疎な座標であり、共起から得た連続的な近さを持ちません。

</details>

<details class="recall">
<summary>Skip-gram と CBOW は何から何を予測するか</summary>

Skip-gram は中心語から周辺語、CBOW は周辺語から中心語を予測します。

</details>

<details class="recall">
<summary>階層的 softmax と負例サンプリングは何を軽くするか</summary>

どちらも出力側の全語彙 softmax の計算を避ける工夫です。前者は二分木の経路、後者は正例と少数の負例を使います。

</details>

<details class="recall">
<summary>学習されたベクトルの近さは何に由来するか</summary>

単語そのものの定義を直接符号化したものではなく、文脈中の共起を予測する学習から生じます。

</details>
